"""Database connection management for Content Review Pipeline."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine,
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine


class DatabaseManager:
    """Manages async and sync database connections."""

    def __init__(self):
        self._async_engine: AsyncEngine | None = None
        self._sync_engine = None
        self._async_session_factory = None
        self._sync_session_factory = None

    def initialize(self, database_url: str | None = None, sync_url: str | None = None):
        """Initialize database connections."""
        database_url = database_url or os.getenv("DATABASE_URL")
        sync_url = sync_url or os.getenv("DATABASE_SYNC_URL")

        if not database_url:
            raise ValueError("DATABASE_URL environment variable is required")

        # Async engine for normal operations
        self._async_engine = create_async_engine(
            database_url,
            echo=os.getenv("ENV") == "development",
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        self._async_session_factory = async_sessionmaker(
            self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

        # Sync engine for operations requiring psycopg2
        if sync_url:
            self._sync_engine = create_engine(
                sync_url,
                echo=os.getenv("ENV") == "development",
                pool_pre_ping=True,
            )
            self._sync_session_factory = sessionmaker(
                self._sync_engine,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )

    @property
    def async_engine(self) -> AsyncEngine:
        if self._async_engine is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._async_engine

    @property
    def sync_engine(self):
        if self._sync_engine is None:
            raise RuntimeError(
                "Sync database not initialized. Call initialize() with sync_url."
            )
        return self._sync_engine

    @asynccontextmanager
    async def async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session."""
        session = self._async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    @asynccontextmanager
    def sync_session(self):
        """Get a sync database session."""
        session = self._sync_session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    async def create_tables(self):
        """Create all tables in the database."""
        from .models import Base

        async with self._async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self):
        """Drop all tables in the database."""
        from .models import Base

        async with self._async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)


# Global database manager instance
db = DatabaseManager()


def get_db() -> DatabaseManager:
    """Get the global database manager instance."""
    return db
