"""Database configuration and Session setup using SQLAlchemy."""

import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ==============================================================================
# DATABASE URL CONFIGURATION
# SQLite is used for local development.
# To swap to PostgreSQL in production, set:
# SQLALCHEMY_DATABASE_URL = "postgresql://username:password@localhost:5432/rammadb"
# or set DATABASE_URL in your .env file: os.getenv("DATABASE_URL")
# ==============================================================================
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ramma_local.db")

connect_args = {"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args=connect_args,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Dependency yielding a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Creates database tables if they do not exist."""
    Base.metadata.create_all(bind=engine)
