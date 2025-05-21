# This code snippet is utilizing the Google Generative AI API to interact with a specific generative
# model named "gemini-1.5-flash". Here's a breakdown of what each part of the code is doing:
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
# API key
API_KEY = os.getenv("Gemini_API_KEY")

# Configure the API key
genai.configure(api_key=API_KEY)

# Initialize the model
model = genai.GenerativeModel("gemini-1.5-flash")

# Generate content
response = model.generate_content("How are you doing today?")

# Print the response
print(response.text)
