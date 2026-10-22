"""本机单用户配置和数据库会话；每个 DATA_DIR 使用独立连接池。"""
import threading
from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import Settings
from app.core import llm, model_preferences
from app.models.db import init_db

_cache: dict = {}
_lock = threading.Lock()

def get_settings() -> Settings:
    return model_preferences.apply(Settings())

def _session_factory(settings: Settings):
    key = str(settings.data_dir)
    if key not in _cache:
        with _lock:
            if key not in _cache:
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                engine = create_engine(f"sqlite:///{settings.data_dir / 'mem.db'}",
                                       connect_args={"check_same_thread": False, "timeout": 30})
                init_db(engine, llm.embedding_dim(settings.embedding_model))
                _cache[key] = sessionmaker(bind=engine)
    return _cache[key]

def get_session(settings: Settings = Depends(get_settings)):
    db = _session_factory(settings)()
    try:
        yield db
    finally:
        db.close()
