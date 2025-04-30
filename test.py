import requests

url = "http://127.0.0.1:8000/ask"
data = {"query": "mother sign"}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Response received:", response.json())
else:
    print("Request failed with status code:", response.status_code)
