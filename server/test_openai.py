import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def test_openai():
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": "You are Cooper Bot."},
                      {"role": "user", "content": "What’s your name?"}]
        )
        print("✅ Response:", response.choices[0].message.content.strip())
    except Exception as e:
        print("❌ OpenAI Error:", e)

if __name__ == "__main__":
    test_openai()
