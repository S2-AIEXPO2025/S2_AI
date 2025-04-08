model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"

from transformers import (
  AutoConfig, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling,
  PreTrainedTokenizer,
  Trainer, TrainingArguments, AutoModelForCausalLM
)
from trl import SFTConfig, SFTTrainer, setup_chat_format
import pandas as pd
import torch

from typing import Dict, List, Optional
from torch.utils.data import Dataset

from edit_dataset import emotion_list, make_emotional_data

print(emotion_list)

if torch.backends.mps.is_available():
  print("M3 MAX GPU is available.")
else:
  print("CPU is using now. GPU is not available now.")
print(f"MPS is available on this project?  {torch.backends.mps.is_built()}")

device = torch.device("mps")

# df = pd.read_csv("Korean Smile Style Dataset.tsv", sep="\t")
# print(df.shape)

# row_notna_count = df.notna().sum(axis=1) # df에서 NaN이 아닌 것들을 True로 변경 후, 가로 방향으로 더한 것을 dataframe으로 나타낸 것
# row_notna_count.plot.hist(bins=row_notna_count.max()) # 히스토그램의 가로축 범위를 지정하는 bins를 사용하여 가로축 범위를 row_notna_count의 최댓값인 17로 설정되게 함.

# df = df[row_notna_count >= 2] # NaN이 아닌 값이 2개 이상의 열에 들어있는 것만 df에 입력
# print(len(df))

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# df_dropped = df.drop(["emoticon"], axis=1) # emoticon 말투를 제외한 데이터셋
# lengths = []

# for column in df_dropped.columns:
#   out = tokenizer(df[column][df[column].notna()].tolist()) # df[column].notna()를 해서 True로 나오는 NaN이 아닌 값들을 모아 list 형태로 변환하여 tokenizer에 전송
#   out = [len(x) for x in out['input_ids']] # 토큰화된 문장들의 길이를 리스트로 저장
#   lengths.extend(out) # 리스트로 추가하는 것이 아닌 각각의 숫자로 lengths 리스트에 추가 (append, extend 차이 검색)

# lengths = pd.Series(lengths) # lengths를 데이터프레임 형식으로 변환
# lengths.plot.hist(bins=80) # 히스토그램에서 80개의 막대로 lengths의 데이터 분포를 그래프로 출력

class TextStyleTransferDataset(Dataset):
  def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer, tokenizer_params: Dict = {}):
    self.df = df
    self.tokenizer = tokenizer
    self.tokenizer_params = tokenizer_params

  def __len__(self):
    return len(self.df)  # 데이터프레임 길이 반환

  def __getitem__(self, index):
    row = self.df.iloc[index, :].dropna().sample(2)  # 샘플 2개 선택
    text1 = row.iloc[0]
    text2 = row.iloc[1]

    # 스타일 정보가 열 이름에 존재한다고 가정
    style1 = row.index[0]  # text1의 스타일 정보
    style2 = row.index[1]  # text2의 스타일 정보

    # 대화 형식에 맞게 변환
    # 사용자 메시지와 모델 응답 형식으로 변환
    chat_data = [
        (f"User: {text1}", f"Model: {text2}")
    ]
        
    # setup_chat_format을 사용해 대화 형식 설정
    formatted_data = setup_chat_format(chat_data) # default format is 'chatml'

    # 토큰화 수행
    out = self.tokenizer(formatted_data[0], **self.tokenizer_params, return_tensors='pt', padding=True, truncation=True)
        
    # SFT 모델에 적합한 형식으로 데이터 반환
    return {
        'input_ids': out['input_ids'].squeeze(0),  # 배치 차원 제거
        'attention_mask': out['attention_mask'].squeeze(0),  # 배치 차원 제거
        'labels': out['input_ids'].squeeze(0)  # 'input_ids'가 labels로 사용될 수 있음
    }

tokenizer_params = dict(
  return_tensors='pt', # 파라미터를 pytorch tensor로 반환
  truncation=True, # token화된 문장을 max_length 길이로 자르기 (긴 경우)
  padding="max_length", # max_length 길이로 패딩 (짧은 경우)
  add_special_tokens=True, # 함수 내부에서 자동적으로 [CLS], [SEP] 토큰을 추가해줌
  max_length=50 # max_length 지정
)

dataset = TextStyleTransferDataset(df_dropped, tokenizer, tokenizer_params) # 이모티콘을 제외한 데이터프레임을 전달하여 토큰 임베딩
out = dataset[0]
print(out['input_ids'][0])
print(tokenizer.decode(out['input_ids'][0]))

out = dataset[1]
print(out['input_ids'][0])
print(tokenizer.decode(out['input_ids'][0]))

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df_dropped, test_size=0.1, random_state=42) # 학습데이터, 테스트데이터 분리
print(len(df_train), len(df_test))

tokenizer_params = dict(
  return_tensors='pt',
  truncation=True,
  padding="max_length",
  add_special_tokens=True,
  max_length=100
)
train_dataset = TextStyleTransferDataset(
  df_train, # 학습데이터를 토큰화
  tokenizer,
  tokenizer_params
)
test_dataset = TextStyleTransferDataset(
  df_test, # 테스트데이터 토큰화
  tokenizer,
  tokenizer_params
)

data_collator = DataCollatorForLanguageModeling(
  tokenizer=tokenizer, mlm=False, # 토큰 자동 패딩을 허용하지 않음
)

model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = model.config.eos_token_id

# training_args = TrainingArguments(
#   output_dir=model_path, #The output directory / 모델 학습 이후 weight 저장 경로 지정
#   overwrite_output_dir=True, #overwrite the content of the output directory / weight 파일이 존재하면 덮어쓰기
#   num_train_epochs=40, # number of training epochs / 전체 데이터셋이 신경망을 40번 통과하게 함
#   per_device_train_batch_size=16, # batch size for training / 학습시킬 때 사용할 배치 사이즈 (한번에 학습시킬 데이터의 크기)
#   per_device_eval_batch_size=16,  # batch size for evaluation / 결과 낼 때 사용할 배치 사이즈
#   eval_steps=500, # Number of update steps between two evaluations. / 500번의 배치마다 평가를 함
#   save_steps=500, # after # steps model is saved / 500번의 배치마다 체크포인트를 저장함
#   warmup_steps=300,# number of warmup steps for learning rate scheduler / 300번의 배치까지 점진적으로 학습률을 올리다가 그 후로는 steps의 규칙에 따라 학습하게 함
#   prediction_loss_only=True, # prediction을 실행할 때마다 loss값만 반환하게 함
#   evaluation_strategy="steps" # 평가를 어떠한 형식으로 할 것인지 설정. (no, steps, epoch로 설정할 수 있으며, no로 정해둘 경우 eval_steps와 save_steps가 의미 없어짐.)
# )

# trainer = Trainer(
#   model=model, # 지정한 모델로 학습할 것이라고 설정
#   args=training_args, # 위에서 설정한 arguments(전달 인자)를 Trainer로 입력
#   data_collator=data_collator, # 패딩, 마스킹 등 전처리 작업을 설정한 값을 전달
#   train_dataset=train_dataset, # 학습 데이터셋 전달
#   eval_dataset=test_dataset, # 평가 데이터셋 전달
# )

for i in len(emotion_list):
  training_args = SFTConfig(
    #   max_seq_length=100, # 설정하지 않으면 .으로 기본 설정
    output_dir=f"./data/{emotion_list[i]}_data",
    use_mps_device=True,
    )

  trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
  )

trainer.train() # 학습 시작