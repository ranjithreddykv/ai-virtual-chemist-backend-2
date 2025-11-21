from dotenv import load_dotenv
from pathlib import Path
import os
from groq import Groq

# Absolute path to the folder containing this file (app/)
BASE_DIR = Path(__file__).resolve().parent

# Absolute path to .env inside app/
ENV_PATH = BASE_DIR / ".env"

print("🔍 Looking for .env at:", ENV_PATH)

# Load .env explicitly
load_dotenv(dotenv_path=ENV_PATH, override=True)

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")

    def validate(self):
        if not self.GROQ_API_KEY:
            raise RuntimeError(
                "❌ GROQ_API_KEY not loaded! Make sure .env exists inside app/ and contains:\n"
                "GROQ_API_KEY=your_key_here\n"
                f"Checked path: {ENV_PATH}"
            )

settings = Settings()
settings.validate()

# Initialize Groq client
groq_client = Groq(api_key=settings.GROQ_API_KEY)

print("🤖 Groq client initialized successfully.")
