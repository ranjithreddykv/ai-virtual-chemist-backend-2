from dotenv import load_dotenv
from pathlib import Path
import os
from groq import Groq

# Path to .env inside app/ folder
ENV_PATH = Path(__file__).resolve().parent / ".env"

print("🔍 Looking for .env at:", ENV_PATH)

load_dotenv(ENV_PATH)

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")

    def validate(self):
        if not self.GROQ_API_KEY:
            raise RuntimeError("❌ GROQ_API_KEY missing! Check app/.env file.")

settings = Settings()
settings.validate()

groq_client = Groq(api_key=settings.GROQ_API_KEY)

print("🤖 Groq client initialized successfully.")
