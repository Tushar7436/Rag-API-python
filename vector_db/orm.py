import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

load_dotenv()

class VectorORM:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=False
        )

        self.predefined = "predefined_context"
        self.user_history = "user_history"

        self._ensure_collection(self.predefined)
        self._ensure_collection(self.user_history)

    def _ensure_collection(self, name):
        try:
            self.client.get_collection(name)
        except:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=int(os.getenv("EMBEDDING_DIM", "1024")),
                    distance=Distance.COSINE
                ),
                on_disk_payload=True
            )

            # Create index for filtering by user_id
            self.client.create_payload_index(
                collection_name=name,
                field_name="user_id",
                field_schema="keyword"
            )

    def insert(self, collection, text, embedding, metadata):
        point = PointStruct(
            id=metadata["id"],
            vector=embedding,
            payload={
                "text": text,
                "user_id": metadata.get("user_id"),
                **metadata
            }
        )

        self.client.upsert(
            collection_name=collection,
            points=[point]
        )

    def search(self, collection, embedding, limit=4, user_id=None):
        q_filter = None
        if user_id:
            q_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            )

        results = self.client.search(
            collection_name=collection,
            query_vector=embedding,
            limit=limit,
            query_filter=q_filter
        )

        docs = [[r.payload.get("text") for r in results]]
        dists = [[1 - r.score for r in results]]

        return {
            "documents": docs,
            "distances": dists
        }
