# app/db.py
from sqlmodel import SQLModel, create_engine, Session

DATABASE_URL = "sqlite:///./testing_unit_tracker.db"

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},  # needed for sqlite+fastapi
)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
