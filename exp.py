from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

load_dotenv()


agent = Agent(model='google-gla:gemini-2.0-flash')

print(Path("src/operate_ai/prompts/cfo.md").read_text())