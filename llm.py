import os
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GEMINI_API_KEY"] = "AIzaSyBcFWDGPOSAjO7VMqJe40nIjXAAdwdfdzM"

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")