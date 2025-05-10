import requests

res = requests.post("http://localhost:11434/api/generate", json={
    "model": "mistral",
    "prompt": "What is Nifty50",
    "stream": False
})

print(res.json())
