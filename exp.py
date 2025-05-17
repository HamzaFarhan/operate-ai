from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

load_dotenv()


agent = Agent(model='google-gla:gemini-2.0-flash')