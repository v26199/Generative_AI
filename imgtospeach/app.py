import requests
import os
from dotenv import load_dotenv

from langchain import PromptTemplate, LLMChain, OpenAI
from transformers import pipeline

# Load environment variables
load_dotenv()
h_token = os.getenv("API")
# Access the OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

def img2txt(url):
    image_to_txt = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_txt(url)[0]['generated_text']
    return text

def generate_story(scenario):
    story_generate = pipeline("text-generation", model="nvidia/Llama3-ChatQA-1.5-8B")
    print(story_generate(scenario))


def text2speech(message):
    # API_URL = "https://api-inference.huggingface.co/models/myshell-ai/MeloTTS-English"
    API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"
    headers = {"Authorization": f"Bearer {h_token}"}

    payload = {"inputs": message}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        with open('audio.flac', 'wb') as file:
            file.write(response.content)
    else:
        print("Failed to generate audio. Error:", response.text)

txt = img2txt("1.jpg")
story = generate_story(txt)
print(story)
text2speech(story)
