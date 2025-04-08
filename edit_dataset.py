import pandas as pd
import matplotlib.pyplot as plt

emotion_list = ['formal', 'informal', 'android', 'azae', 'chat','choding', 'enfp', 'gentle', 'halbae', 'halmae', 'joongding', 'king', 'naruto', 'seonbi', 'sosim', 'translator']

def make_emotional_data():
  df = pd.read_csv("Korean Smile Style Dataset.tsv", sep="\t") # tsv파일은 ','가 아닌 tab으로 구분되어 있기 때문에, sep="\t"를 이용하여 pandas로 불러옴.
  print(df.shape) # df의 크기

  df_formal = df['formal']
  df_informal = df['informal']
  df_android = df['android']
  df_azae = df['azae']
  df_chat = df['chat']
  df_choding = df['choding']
  df_enfp = df['enfp']
  df_gentle = df['gentle']
  df_halbae = df['halbae']
  df_halmae = df['halmae']
  df_joongding = df['joongding']
  df_king = df['king']
  df_naruto = df['naruto']
  df_seonbi = df['seonbi']
  df_sosim = df['sosim']
  df_translator = df['translator']

  df_formal = pd.DataFrame(df_formal)
  df_informal = pd.DataFrame(df_informal)
  df_android = pd.DataFrame(df_android)
  df_azae = pd.DataFrame(df_azae)
  df_chat = pd.DataFrame(df_chat)
  df_choding = pd.DataFrame(df_choding)
  df_enfp = pd.DataFrame(df_enfp)
  df_gentle = pd.DataFrame(df_gentle)
  df_halbae = pd.DataFrame(df_halbae)
  df_halmae = pd.DataFrame(df_halmae)
  df_joongding = pd.DataFrame(df_joongding)
  df_king = pd.DataFrame(df_king)
  df_naruto = pd.DataFrame(df_naruto)
  df_seonbi = pd.DataFrame(df_seonbi)
  df_sosim = pd.DataFrame(df_sosim)
  df_translator = pd.DataFrame(df_translator)

  df_formal['Group'] = df_formal['formal'].isna().cumsum()
  groups = df_formal.groupby('Group')
  formal_df = pd.DataFrame(columns=df_formal.columns)

  for group_name, group_data in groups:
    cleaned_group = group_data.dropna(subset=['formal'], axis=0)  # 'formal' 열에서 NaN인 행 제거
    formal_df = pd.concat([formal_df, cleaned_group], ignore_index=True)

  df_informal['Group'] = df_informal['informal'].isna().cumsum()
  groups = df_informal.groupby('Group')
  informal_df = pd.DataFrame(columns=df_informal.columns)

  for group_name, group_data in groups:
    cleaned_group = group_data.dropna(subset=['informal'], axis=0)  # 'formal' 열에서 NaN인 행 제거
    informal_df = pd.concat([informal_df, cleaned_group], ignore_index=True)

  df_android['Group'] = df_android['android'].isna().cumsum()
  groups = df_android.groupby('Group')
  android_df = pd.DataFrame(columns=df_android.columns)

  for group_name, group_data in groups:
    cleaned_group = group_data.dropna(subset=['android'], axis=0)  # 'formal' 열에서 NaN인 행 제거
    android_df = pd.concat([android_df, cleaned_group], ignore_index=True)

  df_azae['Group'] = df_azae['azae'].isna().cumsum()
  groups = df_azae.groupby('Group')
  azae_df = pd.DataFrame(columns=df_azae.columns)

  for group_name, group_data in groups:
    cleaned_group = group_data.dropna(subset=['azae'], axis=0)  # 'formal' 열에서 NaN인 행 제거
    azae_df = pd.concat([azae_df, cleaned_group], ignore_index=True)

  df_chat['Group'] = df_chat['chat'].isna().cumsum()
  groups = df_chat.groupby('Group')
  chat_df = pd.DataFrame(columns=df_chat.columns)

  for group_name, group_data in groups:
    cleaned_group = group_data.dropna(subset=['chat'], axis=0)  # 'formal' 열에서 NaN인 행 제거
    chat_df = pd.concat([chat_df, cleaned_group], ignore_index=True)

  df_choding['Group'] = df_choding['choding'].isna().cumsum()
  groups = df_choding.groupby('Group')
  choding_df = pd.DataFrame(columns=df_choding.columns)

  for group_name, group_data in groups:
    cleaned_group = group_data.dropna(subset=['choding'], axis=0)  # 'formal' 열에서 NaN인 행 제거
    choding_df = pd.concat([choding_df, cleaned_group], ignore_index=True)

  df_enfp['Group'] = df_enfp['enfp'].isna().cumsum()
  groups = df_enfp.groupby('Group')
  enfp_df = pd.DataFrame(columns=df_enfp.columns)

  for group_name, group_data in groups:
    cleaned_group = group_data.dropna(subset=['enfp'], axis=0)  # 'formal' 열에서 NaN인 행 제거
    enfp_df = pd.concat([enfp_df, cleaned_group], ignore_index=True)

  df_gentle['Group'] = df_gentle['gentle'].isna().cumsum()
  groups = df_gentle.groupby('Group')
  gentle_df = pd.DataFrame(columns=df_gentle.columns)

  for group_name, group_data in groups:
    cleaned_group = group_data.dropna(subset=['gentle'], axis=0)  # 'formal' 열에서 NaN인 행 제거
    gentle_df = pd.concat([gentle_df, cleaned_group], ignore_index=True)

  df_halbae['Group'] = df_halbae['halbae'].isna().cumsum()
  groups = df_halbae.groupby('Group')
  halbae_df = pd.DataFrame(columns=df_halbae.columns)

  for group_name, group_data in groups:
    cleaned_group = group_data.dropna(subset=['halbae'], axis=0)  # 'formal' 열에서 NaN인 행 제거
    halbae_df = pd.concat([halbae_df, cleaned_group], ignore_index=True)

  df_halmae['Group'] = df_halmae['halmae'].isna().cumsum()
  groups = df_halmae.groupby('Group')
  halmae_df = pd.DataFrame(columns=df_halmae.columns)

  for group_name, group_data in groups:
    cleaned_group = group_data.dropna(subset=['halmae'], axis=0)  # 'formal' 열에서 NaN인 행 제거
    halmae_df = pd.concat([halmae_df, cleaned_group], ignore_index=True)

  df_joongding['Group'] = df_joongding['joongding'].isna().cumsum()
  groups = df_joongding.groupby('Group')
  joongding_df = pd.DataFrame(columns=df_joongding.columns)

  for group_name, group_data in groups:
    cleaned_group = group_data.dropna(subset=['joongding'], axis=0)  # 'formal' 열에서 NaN인 행 제거
    joongding_df = pd.concat([joongding_df, cleaned_group], ignore_index=True)

  df_king['Group'] = df_king['king'].isna().cumsum()
  groups = df_king.groupby('Group')
  king_df = pd.DataFrame(columns=df_king.columns)

  for group_name, group_data in groups:
    cleaned_group = group_data.dropna(subset=['king'], axis=0)  # 'formal' 열에서 NaN인 행 제거
    king_df = pd.concat([king_df, cleaned_group], ignore_index=True)

  df_naruto['Group'] = df_naruto['naruto'].isna().cumsum()
  groups = df_naruto.groupby('Group')
  naruto_df = pd.DataFrame(columns=df_naruto.columns)

  for group_name, group_data in groups:
    cleaned_group = group_data.dropna(subset=['naruto'], axis=0)  # 'formal' 열에서 NaN인 행 제거
    naruto_df = pd.concat([naruto_df, cleaned_group], ignore_index=True)

  df_seonbi['Group'] = df_seonbi['seonbi'].isna().cumsum()
  groups = df_seonbi.groupby('Group')
  seonbi_df = pd.DataFrame(columns=df_seonbi.columns)

  for group_name, group_data in groups:
    cleaned_group = group_data.dropna(subset=['seonbi'], axis=0)  # 'formal' 열에서 NaN인 행 제거
    seonbi_df = pd.concat([seonbi_df, cleaned_group], ignore_index=True)

  df_sosim['Group'] = df_sosim['sosim'].isna().cumsum()
  groups = df_sosim.groupby('Group')
  sosim_df = pd.DataFrame(columns=df_sosim.columns)

  for group_name, group_data in groups:
    cleaned_group = group_data.dropna(subset=['sosim'], axis=0)  # 'formal' 열에서 NaN인 행 제거
    sosim_df = pd.concat([sosim_df, cleaned_group], ignore_index=True)

  df_translator['Group'] = df_translator['translator'].isna().cumsum()
  groups = df_translator.groupby('Group')
  translator_df = pd.DataFrame(columns=df_translator.columns)

  for group_name, group_data in groups:
    cleaned_group = group_data.dropna(subset=['translator'], axis=0)  # 'formal' 열에서 NaN인 행 제거
    translator_df = pd.concat([translator_df, cleaned_group], ignore_index=True)

  training_data = []
  training_data.append(formal_df)
  training_data.append(informal_df)
  training_data.append(android_df)
  training_data.append(azae_df)
  training_data.append(chat_df)
  training_data.append(choding_df)
  training_data.append(enfp_df)
  training_data.append(gentle_df)
  training_data.append(halbae_df)
  training_data.append(halmae_df)
  training_data.append(joongding_df)
  training_data.append(king_df)
  training_data.append(naruto_df)
  training_data.append(seonbi_df)
  training_data.append(sosim_df)
  training_data.append(translator_df)

  return training_data