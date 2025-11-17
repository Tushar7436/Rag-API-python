import os
from chromadb import HttpClient

class VectorORM:
    def __init__(self):
        CHROMA_HOST = os.getenv("CHROMA_HOST")
        CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))

        self.client = HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            ssl=False
        )

        self.predefined = "predefined_context"
        self.user_history = "user_history"

        self._ensure_collection(self.predefined)
        self._ensure_collection(self.user_history)

    def _ensure_collection(self, name):
        try:
            self.client.get_collection(name)
        except:
            self.client.create_collection(name=name)

    def insert(self, collection, text, embedding, metadata):
        col = self.client.get_collection(collection)

        col.add(
            ids=[metadata["id"]],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )

    def search(self, collection, embedding, limit=4):
        col = self.client.get_collection(collection)
        results = col.query(
            query_embeddings=[embedding],
            n_results=limit
        )
        return results
