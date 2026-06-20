import os
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GEMINI_API_KEY"] = "ohew almost leaked it lmao"

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")