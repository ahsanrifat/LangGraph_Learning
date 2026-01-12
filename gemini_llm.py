from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  # Or "gemini-1.5-pro", etc.
    temperature=0,  # For more deterministic responses
    max_tokens=None,
    timeout=None,
    max_retries=2,
)