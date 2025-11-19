from embeddings.generator import EmbeddingGenerator
from vector_db.orm import VectorORM

class VectorSearchEngine:
    def __init__(self):
        self.db = VectorORM()
        self.embedder = EmbeddingGenerator()

    def search_user_history(self, user_id: str, embedding, top_k=4):
        return self.db.search(
            collection=self.db.user_history,
            embedding=embedding,
            limit=top_k,
            user_id=user_id
        )

    def search_relevant_chunks(self, query: str, user_id: str, top_k=4):
        embedding = self.embedder.create_embedding(query)

        predefined = self.db.search(
            collection=self.db.predefined,
            embedding=embedding,
            limit=top_k
        )

        user_history = self.search_user_history(user_id, embedding, top_k)

        merged = []

        # Predefined
        if predefined["documents"]:
            for idx in range(len(predefined["documents"][0])):
                merged.append({
                    "text": predefined["documents"][0][idx],
                    "score": predefined["distances"][0][idx]
                })

        # User history
        if user_history["documents"]:
            for idx in range(len(user_history["documents"][0])):
                merged.append({
                    "text": user_history["documents"][0][idx],
                    "score": user_history["distances"][0][idx]
                })

        merged_sorted = sorted(merged, key=lambda x: x["score"])

        return merged_sorted[:top_k]
