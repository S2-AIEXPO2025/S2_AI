from fastapi import FastAPI, HTTPException, status, WebSocket

from core.config import API_KEY
from crud import gemini

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, gender: str, lover_option: str):
  gender = websocket.query_params.get('gender')
  lover_option = websocket.query_params.get('lover_option')
  name = websocket.query_params.get('name')
  
  if not gender or not lover_option:
    await websocket.close()
    raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Missing query parameters.")

  await websocket.accept()
  while True:
    text = await websocket.receive_text()
    try:
      res = gemini.ai_response(api_key=API_KEY, text=text, gender=gender, option=lover_option, name=name).strip()
      await websocket.send_text(f"{res}")
    except Exception as e:
      await websocket.send_text(f"Error processing request: {str(e)}")
      break

if __name__ == "__main__":
  import uvicorn
  uvicorn.run("main:app", host="0.0.0.0", port=3060, reload=True)