# checkpointer_manager.py
import os
from langgraph.checkpoint.postgres import PostgresSaver

class CheckpointerManager:
    _instance: PostgresSaver | None = None

    @classmethod
    def get_checkpointer(cls) -> PostgresSaver:
        if cls._instance is None:
            DB_URI = os.getenv("DOC_AGENT_DB_URI")
            saver = PostgresSaver.from_conn_string(DB_URI)
            saver = saver.__enter__()   # init connection pool
            cls._instance = saver
        return cls._instance

    @classmethod
    def close_checkpointer(cls):
        if cls._instance:
            cls._instance.__exit__(None, None, None)
            cls._instance = None