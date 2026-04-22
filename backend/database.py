from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.settings import DATABASE_URL


engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def ensure_auth_columns():
    """Keep older local databases compatible with the current auth model."""
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_verified BOOLEAN"))
        connection.execute(text("UPDATE users SET is_verified = TRUE WHERE is_verified IS NULL"))
        connection.execute(text("ALTER TABLE users ALTER COLUMN is_verified SET DEFAULT FALSE"))
        connection.execute(text("ALTER TABLE users ALTER COLUMN is_verified SET NOT NULL"))
        connection.execute(
            text(
                "ALTER TABLE users "
                "ADD COLUMN IF NOT EXISTS last_verification_sent TIMESTAMP WITH TIME ZONE"
            )
        )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
