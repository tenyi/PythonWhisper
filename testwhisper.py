import whisper
model = whisper.load_model("turbo")
model = whisper.load_model("large-v3", device="cuda")
result = model.transcribe("test.mp3")
print(result["text"])

