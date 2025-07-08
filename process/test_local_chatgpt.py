import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-fake-key"
)

response = client.chat.completions.create(
    model="llama3",
    messages=[
        {"role": "user", "content": "你好，介绍一下你自己"}
    ]
)

print(response.choices[0].message.content)