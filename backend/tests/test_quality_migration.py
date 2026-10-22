"""An old history table gains metadata without clearing any row."""
from sqlalchemy import create_engine, text
from app.models.db import init_db


def test_existing_history_metadata_migration_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    with engine.begin() as c:
        c.execute(text("CREATE TABLE qa_history (id TEXT PRIMARY KEY, created_at DATETIME NOT NULL, question TEXT NOT NULL, resolved_question TEXT, answer TEXT NOT NULL, sources TEXT NOT NULL, documents TEXT NOT NULL, question_vec BLOB NOT NULL, store_version INTEGER NOT NULL, from_cache INTEGER NOT NULL DEFAULT 0)"))
        c.execute(text("INSERT INTO qa_history VALUES ('kept','2026-10-18','问题',NULL,'历史答案','[]','[]',X'00000000',0,0)"))
    # Production init receives a fresh engine; don't reuse a pre-listener DBAPI connection.
    engine.dispose()
    init_db(engine, dim=4)
    init_db(engine, dim=4)
    with engine.connect() as c:
        row = c.execute(text("SELECT answer,execution FROM qa_history WHERE id='kept'")).one()
        assert row == ("历史答案", "{}")
