from dotenv import load_dotenv
import os

load_dotenv()

def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("❌ GEMINI_API_KEY not set in .env file")
    return api_key
