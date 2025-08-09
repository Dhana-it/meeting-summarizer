from fastapi import FastAPI, Request
from pydantic import BaseModel
from transcriptobot_ml import summarize_transcript, extract_action_items

app = FastAPI()

class TranscriptInput(BaseModel):
    transcript: str

@app.post("/process")
async def process_transcript(data: TranscriptInput):
    summary = summarize_transcript(data.transcript)
    actions = extract_action_items(data.transcript)
    return {"summary": summary, "action_items": actions}