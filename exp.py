from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

load_dotenv()


def get_weather()

agent = Agent(model='google-gla:gemini-2.5-flash')
