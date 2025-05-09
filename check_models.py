import os
from dotenv import load_dotenv

# Load environment variables from .env file
print("Loading environment variables...")
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

print(f"OPENAI_API_KEY: {'Set' if OPENAI_API_KEY else 'Not Set'}")
print(f"EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")

# Import and check chatbot utils
try:
    from chatbot.utils import EMBEDDING_MODEL as UTILS_EMBEDDING_MODEL
    print(f"chatbot.utils EMBEDDING_MODEL: {UTILS_EMBEDDING_MODEL}")
except Exception as e:
    print(f"Error importing from chatbot.utils: {e}")

# Import and check ingest_data
try:
    from ingest_data import EMBEDDING_MODEL as INGEST_EMBEDDING_MODEL
    print(f"ingest_data EMBEDDING_MODEL: {INGEST_EMBEDDING_MODEL}")
except Exception as e:
    print(f"Error importing from ingest_data: {e}")

print("\nSUMMARY:")
print(f".env EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}")
print(f".env EMBEDDING_MODEL: {EMBEDDING_MODEL}")
print(f"chatbot.utils EMBEDDING_MODEL: {UTILS_EMBEDDING_MODEL if 'UTILS_EMBEDDING_MODEL' in locals() else 'Error'}")
print(f"ingest_data EMBEDDING_MODEL: {INGEST_EMBEDDING_MODEL if 'INGEST_EMBEDDING_MODEL' in locals() else 'Error'}") 