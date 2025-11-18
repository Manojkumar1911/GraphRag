"""
Traditional RAG Pipeline using FAISS
====================================
Simple vector-based RAG with mechanical chunking and FAISS indexing.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import logging

import faiss
import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class TraditionalChunk:
    """Represents a mechanically chunked text span."""
    chunk_id: str
    text: str
    start_token: int
    end_token: int


class TraditionalRAGPipeline:
    """Pipeline implementing the traditional RAG steps using FAISS."""

    INDEX_FILENAME = "traditional_index.faiss"
    METADATA_FILENAME = "traditional_metadata.json"

    def __init__(self, config) -> None:
        self.config = config
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize Gemini
        try:
            genai.configure(api_key=config.gemini_api_key)
            self.gemini_model = genai.GenerativeModel(
                getattr(config, 'gemini_model', 'gemini-2.0-flash')
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
        
        # Setup paths
        self.faiss_dir = Path(getattr(config, 'faiss_dir', './faiss_store'))
        self.faiss_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.faiss_dir / self.INDEX_FILENAME
        self.metadata_path = self.faiss_dir / self.METADATA_FILENAME
        
        # Initialize index and metadata
        self.index: faiss.Index | None = None
        self.metadata: list[TraditionalChunk] = []
        
        # Load existing index if available
        self._load_existing_index()

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------
    def build(self, reset_index: bool = True) -> list[TraditionalChunk]:
        """Build traditional RAG index from knowledge base."""
        kb_path = Path(getattr(self.config, 'kb_path', './kb.txt'))
        logger.info(f"Building traditional RAG index from {kb_path}")
        
        if not kb_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {kb_path}")
        
        text = kb_path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("Knowledge base is empty")
        
        # Chunk text
        chunks = self._chunk_text(text)
        if not chunks:
            raise ValueError("Chunking resulted in no chunks")
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedder.encode(
            [chunk.text for chunk in chunks],
            show_progress_bar=True
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        embeddings = self._normalize_embeddings(embeddings)
        
        # Reset or extend index
        if reset_index:
            self._reset_index()
        
        # Create or extend FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            logger.info(f"Created new FAISS index with dimension {embeddings.shape[1]}")
        
        self.index.add(embeddings)
        self.metadata.extend(chunks)
        
        # Persist to disk
        self._persist()
        logger.info(f"Traditional RAG index now has {len(self.metadata)} chunks")
        
        return chunks

    # ------------------------------------------------------------------
    # Query flow
    # ------------------------------------------------------------------
    def query(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        """Query the traditional RAG system."""
        if self.index is None or not self.metadata:
            raise RuntimeError("Traditional RAG index is empty. Run 'build trad' first.")
        
        k = top_k or getattr(self.config, 'traditional_top_k', 5)
        logger.info(f"Querying with top_k={k}")
        
        # Encode query
        query_vec = self.embedder.encode([question]).astype(np.float32)
        query_vec = self._normalize_embeddings(query_vec)
        
        # Search FAISS index
        scores, indices = self.index.search(query_vec, k)
        
        # Collect retrieved chunks
        retrieved_chunks: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            chunk = self.metadata[idx]
            retrieved_chunks.append({
                "chunk_id": chunk.chunk_id,
                "score": float(score),
                "text": chunk.text,
                "start_token": chunk.start_token,
                "end_token": chunk.end_token,
            })
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        
        # Build context
        context = "\n\n".join(
            f"[Chunk {item['chunk_id']}]:\n{item['text']}" 
            for item in retrieved_chunks
        )
        
        # Generate answer
        prompt = self._answer_prompt(question, context)
        
        try:
            response = self.gemini_model.generate_content(prompt)
            answer = response.text.strip() if response.text else "No answer generated."
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            answer = f"Error generating answer: {str(e)}"
        
        return {
            "answer": answer,
            "chunks": retrieved_chunks,
        }

    def has_index(self) -> bool:
        """Check if index exists and has data."""
        return self.index is not None and bool(self.metadata)

    def chunk_count(self) -> int:
        """Return number of chunks in index."""
        return len(self.metadata)

    def close(self) -> None:
        """Close resources (no persistent connections needed)."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _chunk_text(self, text: str) -> list[TraditionalChunk]:
        """Split text into overlapping chunks based on token count."""
        # Simple whitespace tokenization
        tokens = text.split()
        
        chunk_size = getattr(self.config, 'traditional_chunk_size_tokens', 200)
        overlap = getattr(self.config, 'traditional_chunk_overlap_tokens', 50)
        
        if chunk_size <= overlap:
            raise ValueError(f"Chunk size ({chunk_size}) must be greater than overlap ({overlap})")
        
        chunks: list[TraditionalChunk] = []
        step = chunk_size - overlap
        start = 0
        chunk_id = 0
        
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            
            if not chunk_tokens:
                break
            
            chunk_text = " ".join(chunk_tokens)
            
            chunks.append(TraditionalChunk(
                chunk_id=f"trad_chunk_{chunk_id}",
                text=chunk_text,
                start_token=start,
                end_token=end,
            ))
            
            chunk_id += 1
            start += step
            
            # Break if we've reached the end
            if end >= len(tokens):
                break
        
        return chunks

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings for cosine similarity via inner product."""
        faiss.normalize_L2(embeddings)
        return embeddings

    def _reset_index(self) -> None:
        """Clear index and metadata."""
        logger.info("Resetting traditional RAG index")
        self.index = None
        self.metadata = []
        
        if self.index_path.exists():
            self.index_path.unlink()
            logger.info(f"Deleted {self.index_path}")
        
        if self.metadata_path.exists():
            self.metadata_path.unlink()
            logger.info(f"Deleted {self.metadata_path}")

    def _persist(self) -> None:
        """Save index and metadata to disk."""
        if self.index is None:
            logger.warning("Cannot persist: index is None")
            return
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            logger.info(f"Saved FAISS index to {self.index_path}")
            
            # Save metadata
            with self.metadata_path.open("w", encoding="utf-8") as fh:
                json.dump(
                    [asdict(chunk) for chunk in self.metadata],
                    fh,
                    ensure_ascii=False,
                    indent=2
                )
            logger.info(f"Saved metadata to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Failed to persist index: {e}")
            raise

    def _load_existing_index(self) -> None:
        """Load existing FAISS index and metadata from disk."""
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                self.index = faiss.read_index(str(self.index_path))
                
                with self.metadata_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                
                self.metadata = [TraditionalChunk(**item) for item in data]
                logger.info(f"Loaded {len(self.metadata)} chunks from disk")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
                self.index = None
                self.metadata = []
        else:
            logger.info("No existing index found")
            self.index = None
            self.metadata = []

    def _answer_prompt(self, question: str, context: str) -> str:
        """Generate prompt for answer generation."""
        return (
            "You are a helpful assistant answering questions using provided document chunks.\n"
            "Base your answer solely on the supplied context. If the context is insufficient, say so.\n"
            "Be concise and accurate.\n"
            "\n"
            "Question:\n"
            f"{question}\n"
            "\n"
            "Context:\n"
            f"{context}\n"
            "\n"
            "Answer:"
        )