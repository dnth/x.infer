from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="NOT A REAL KEY")

chat_completion = client.chat.completions.create(
    model="image-classifier",  # This is ignored by our deployment
    messages=[
        # {"role": "system", "content": "You are an image classification assistant."},
        {
            "role": "user",
            "content": "https://raw.githubusercontent.com/vikhyat/moondream/main/assets/demo-1.jpg",
        },
    ],
)

print(chat_completion.choices[0].message.content)
