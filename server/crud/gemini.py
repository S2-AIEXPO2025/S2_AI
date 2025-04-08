from google import genai
from google.genai import types

def ai_response(api_key: str, text: str, gender: str, option: str, name: str):
  client = genai.Client(api_key=api_key)

  response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
      system_instruction=f"너는 {option} 애인이고, 그에 맞는 말투로 이모지 없이 대답해줘. 너는 {gender}이며, 너의 애인의 이름은 {name}이야."
    ),
    contents= [text]
  )
  print(response.text)
  return response.text