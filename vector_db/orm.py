import os
import chromadb
from chromadb.config import Settings

class VectorORM:
    def __init__(self):
        CHROMA_HOST = os.getenv("CHROMA_HOST")              # e.g. chromatushar.railway.internal
        CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000")) # Railway internal port

        # Use ChromaDB v2 REST API client
        self.client = chromadb.Client(
            Settings(
                chroma_api_impl="rest",
                chroma_server_host=CHROMA_HOST,
                chroma_server_http_port=CHROMA_PORT,
                ssl_enabled=False      # Railway internal network --> NO SSL
            )
        )

        self.predefined = "predefined_context"
        self.user_history = "user_history"

        self._ensure_collection(self.predefined)
        self._ensure_collection(self.user_history)

    def _ensure_collection(self, name):
        try:
            self.client.get_collection(name)
        except Exception:
            self.client.create_collection(name=name)

    def insert(self, collection, text, embedding, metadata):
        col = self.client.get_collection(collection)
        col.add(
            ids=[metadata["id"]],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )

    def search(self, collection, embedding, limit=4):
        col = self.client.get_collection(collection)
        results = col.query(
            query_embeddings=[embedding],
            n_results=limit
        )
        return results
