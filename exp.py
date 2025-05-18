from sqlmodel import create_engine

DB_URL = "sqlite:///operate_ai.db"

engine = create_engine(DB_URL, echo=False)
