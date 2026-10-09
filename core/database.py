from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool

from core.config import settings

# NullPool: no idle connections held open between requests. Railway's Serverless
# sleep feature keys off outbound traffic, and a persistent pool's idle connection
# to Postgres counts as continuous outbound traffic, which keeps both this service
# and the database awake indefinitely.
engine = create_engine(settings.database_url, poolclass=NullPool)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all_tables() -> None:
    import models.orm  # noqa: F401 - registers all models on Base.metadata

    Base.metadata.create_all(bind=engine)
