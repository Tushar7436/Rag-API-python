from voyageai import Client
import os

class EmbeddingGenerator:
    def __init__(self):
        self.client = Client(api_key=os.getenv("VOYAGE_API_KEY"))
        self.model = "voyage-lite-01"

    def create_embedding(self, text: str):
        result = self.client.embed(
            model=self.model,
            texts=[text]     # <-- Correct argument
        )
        return result.embeddings[0]   # List of floats (shape 1536)
