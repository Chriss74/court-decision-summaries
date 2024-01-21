from dotenv import load_dotenv
from os import getenv

load_dotenv()

OPENAI_KEY=getenv("OPENAI_KEY")
print(OPENAI_KEY)