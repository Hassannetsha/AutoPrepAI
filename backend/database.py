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


def ensure_conversation_columns():
    """Ensure conversations table has all required columns."""
    commands = [
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS title VARCHAR DEFAULT 'New Chat'",
        "UPDATE conversations SET title = 'New Chat' WHERE title IS NULL",
        "ALTER TABLE conversations ALTER COLUMN title SET NOT NULL",
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS user_id UUID",
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP",
        "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE updated_at IS NULL",
        "ALTER TABLE conversations ALTER COLUMN updated_at SET NOT NULL",
    ]
    
    # Execute each command in its own transaction to avoid aborting on errors
    for cmd in commands:
        try:
            with engine.begin() as connection:
                connection.execute(text(cmd))
        except Exception as e:
            # Log but continue - column might already exist or constraint already in place
            print(f"Note: {cmd[:50]}... - {type(e).__name__}")
    
    # Try to add foreign key constraint in its own transaction
    try:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE conversations ADD CONSTRAINT fk_conversations_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE"))
    except Exception:
        # Constraint already exists, that's fine
        pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
