import requests

url = "http://localhost:8000/ask"
data = {"query": "how to home sign"}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Response received:", response.json())
else:
    print("Request failed with status code:", response.status_code)
