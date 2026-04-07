import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import AsyncQdrantClient, models

from core.config import get_settings
from core.models.chunk import DocumentChunk
from core.vector_store.base_vector_store import BaseVectorStore
from core.vector_store.utils import build_store_metrics

logger = logging.getLogger(__name__)


class QdrantStore(BaseVectorStore):
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize Qdrant connection for vector storage."""
        settings = get_settings()
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.collection_name = "vector_embeddings"

        if not self.url:
            raise ValueError("Qdrant URL is required for QdrantStore. Please set QDRANT_URL in your configuration.")

        self.client = AsyncQdrantClient(url=self.url, api_key=self.api_key)
        self._last_store_metrics: Dict[str, Any] = {}

    async def initialize(self):
        """Initialize Qdrant collection if it doesn't exist."""
        settings = get_settings()
        vector_size = settings.VECTOR_DIMENSIONS

        # Map similarity metric to Qdrant distance
        distance_map = {
            "cosine": models.Distance.COSINE,
            "dotProduct": models.Distance.DOT,
        }
        similarity_metric = getattr(settings, "EMBEDDING_SIMILARITY_METRIC", "cosine")
        distance = distance_map.get(similarity_metric, models.Distance.COSINE)

        try:
            collections = await self.client.get_collections()
            exists = any(c.name == self.collection_name for c in collections.collections)

            if not exists:
                logger.info(
                    f"Creating Qdrant collection: {self.collection_name} with size {vector_size} and distance {distance}"
                )
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=vector_size, distance=distance),
                )
                # Create keyword index for document_id to speed up filtering and deletions
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                # Create integer index for chunk_number
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="chunk_number",
                    field_schema=models.PayloadSchemaType.INTEGER,
                )
                logger.info(f"Successfully initialized Qdrant collection: {self.collection_name}")
            else:
                logger.debug(f"Qdrant collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {str(e)}")
            raise

    async def store_embeddings(
        self, chunks: List[DocumentChunk], app_id: Optional[str] = None
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Store document chunks and their embeddings in Qdrant."""
        if not chunks:
            self._last_store_metrics = build_store_metrics(
                chunk_payload_backend="none",
                multivector_backend="none",
                vector_store_backend="qdrant",
            )
            return True, [], self._last_store_metrics

        points = []
        stored_ids = []
        chunk_payload_bytes = 0

        for chunk in chunks:
            if chunk.embedding is None:
                continue

            # Use a deterministic UUID based on document_id and chunk_number
            # to avoid duplicates and allow easy retrieval/updates
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.document_id}-{chunk.chunk_number}"))

            # Prepare payload - store metadata as is
            payload = {
                "document_id": chunk.document_id,
                "chunk_number": chunk.chunk_number,
                "content": chunk.content,
                "metadata": chunk.metadata or {},
            }

            # Handle numpy arrays if present
            vector = chunk.embedding
            if hasattr(vector, "tolist"):
                vector = vector.tolist()

            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

            stored_ids.append(f"{chunk.document_id}-{chunk.chunk_number}")
            if chunk.content:
                chunk_payload_bytes += len(chunk.content.encode("utf-8"))

        if not points:
            logger.warning("No valid embeddings to store in Qdrant")
            self._last_store_metrics = build_store_metrics(
                chunk_payload_backend="none",
                multivector_backend="none",
                vector_store_backend="qdrant",
            )
            return True, [], self._last_store_metrics

        write_start = time.perf_counter()
        try:
            await self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
            write_duration = time.perf_counter() - write_start

            self._last_store_metrics = build_store_metrics(
                chunk_payload_backend="none",
                multivector_backend="none",
                vector_store_backend="qdrant",
                chunk_payload_bytes=chunk_payload_bytes,
                vector_store_write_s=write_duration,
                vector_store_rows=len(points),
            )
            return True, stored_ids, self._last_store_metrics
        except Exception as e:
            logger.error(f"Error storing embeddings in Qdrant: {str(e)}")
            return False, [], {}

    async def query_similar(
        self,
        query_embedding: List[float],
        k: int,
        doc_ids: Optional[List[str]] = None,
        app_id: Optional[str] = None,
        skip_image_content: bool = False,
    ) -> List[DocumentChunk]:
        """Find similar chunks in Qdrant."""
        try:
            query_filter = None
            if doc_ids:
                query_filter = models.Filter(
                    must=[models.FieldCondition(key="document_id", match=models.MatchAny(any=doc_ids))]
                )

            search_result = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=k,
                query_filter=query_filter,
                with_payload=True,
            )

            chunks = []
            settings = get_settings()
            similarity_metric = getattr(settings, "EMBEDDING_SIMILARITY_METRIC", "cosine")

            for hit in search_result.points:
                payload = hit.payload or {}

                # Normalize score if using cosine similarity to match PGVector's [0, 1] range
                score = float(hit.score)
                if similarity_metric == "cosine":
                    score = (score + 1.0) / 2.0

                chunk = DocumentChunk(
                    document_id=payload.get("document_id", ""),
                    chunk_number=payload.get("chunk_number", 0),
                    content=payload.get("content", ""),
                    embedding=[],  # Don't return embedding in search results
                    metadata=payload.get("metadata", {}),
                    score=score,
                )
                chunks.append(chunk)

            return chunks
        except Exception as e:
            logger.error(f"Error querying Qdrant: {str(e)}")
            return []

    async def get_chunks_by_id(
        self,
        chunk_identifiers: List[Tuple[str, int]],
        app_id: Optional[str] = None,
        skip_image_content: bool = False,
    ) -> List[DocumentChunk]:
        """Retrieve specific chunks by document ID and chunk number from Qdrant."""
        if not chunk_identifiers:
            return []

        # Generate deterministic UUIDs for the requested chunks
        point_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}-{chunk_num}")) for doc_id, chunk_num in chunk_identifiers
        ]

        try:
            points = await self.client.retrieve(
                collection_name=self.collection_name, ids=point_ids, with_payload=True, with_vectors=False
            )

            # Map results to DocumentChunk objects
            chunks = []
            for point in points:
                payload = point.payload or {}
                chunks.append(
                    DocumentChunk(
                        document_id=payload.get("document_id", ""),
                        chunk_number=payload.get("chunk_number", 0),
                        content=payload.get("content", ""),
                        embedding=[],
                        metadata=payload.get("metadata", {}),
                    )
                )

            return chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks from Qdrant: {str(e)}")
            return []

    async def delete_chunks_by_document_id(self, document_id: str, app_id: Optional[str] = None) -> bool:
        """Delete all chunks associated with a document from Qdrant."""
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.Filter(
                    must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
                ),
                wait=True,
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting chunks from Qdrant for document {document_id}: {str(e)}")
            return False
