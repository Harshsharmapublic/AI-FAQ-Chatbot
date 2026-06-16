"""
chatbot.py - Semantic FAQ Search Engine
========================================
Author: AI-Powered FAQ Chatbot System
Description:
    Core chatbot intelligence using TF-IDF Vectorization and Cosine Similarity
    for semantic FAQ matching. Also integrates PDF-indexed content.
    
    Architecture:
        - FAQEngine: TF-IDF + Cosine Similarity over FAQ knowledge base
        - Hybrid search: FAQ knowledge base + PDF chunks (if uploaded)
        - Confidence thresholding for fallback responses
        - Session-based conversation history

Usage:
    from chatbot import FAQEngine
    engine = FAQEngine()
    result = engine.query("How much is the college fee?")
    print(result.answer, result.confidence)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import TextPreprocessor
from faq_manager import FAQManager, FAQItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
MAX_HISTORY_LENGTH = 100


# ---------------------------------------------------------------------------
# Response Data Model
# ---------------------------------------------------------------------------

@dataclass
class ChatResponse:
    """Encapsulates a single chatbot response with metadata."""
    query: str
    answer: str
    confidence: float                     # 0.0 – 1.0
    matched_question: str = ""
    source: str = "FAQ"                   # 'FAQ' | 'PDF' | 'Fallback'
    category: str = "General"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    response_time_ms: float = 0.0
    is_successful: bool = True

    @property
    def confidence_pct(self) -> str:
        return f"{self.confidence * 100:.1f}%"

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 0.70:
            return "High"
        elif self.confidence >= 0.40:
            return "Moderate"
        elif self.confidence > 0.0:
            return "Low"
        return "No Match"


@dataclass
class ConversationTurn:
    """Represents one turn in the conversation (user + bot)."""
    role: str          # 'user' | 'assistant'
    content: str
    confidence: Optional[float] = None
    source: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# TF-IDF Index
# ---------------------------------------------------------------------------

class TFIDFIndex:
    """
    Builds and queries a TF-IDF vector space model over a corpus of texts.

    Supports incremental updates and batch queries.
    """

    def __init__(self, preprocessor: TextPreprocessor) -> None:
        self.preprocessor = preprocessor
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),      # Unigrams + bigrams for richer context
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,       # Apply log normalization to TF
        )
        self._corpus: List[str] = []       # Original texts
        self._processed: List[str] = []    # Preprocessed texts
        self._tfidf_matrix = None
        self._is_fitted: bool = False

    def fit(self, texts: List[str]) -> None:
        """
        Fit the TF-IDF model on a corpus.

        Args:
            texts:  List of raw text strings to index.
        """
        if not texts:
            logger.warning("Empty corpus provided to TFIDFIndex.fit()")
            self._is_fitted = False
            return

        self._corpus = texts
        self._processed = self.preprocessor.preprocess_batch(texts)
        self._tfidf_matrix = self.vectorizer.fit_transform(self._processed)
        self._is_fitted = True
        logger.debug("TFIDFIndex fitted | corpus_size=%d", len(texts))

    def query(self, text: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Find the most similar texts in the index.

        Args:
            text:   Raw query string.
            top_k:  Number of top results to return.

        Returns:
            List of (index, similarity_score) tuples, sorted by score descending.
        """
        if not self._is_fitted:
            return []

        processed_query = self.preprocessor.preprocess(text)
        if not processed_query.strip():
            return []

        query_vec = self.vectorizer.transform([processed_query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)


# ---------------------------------------------------------------------------
# Main FAQ Engine
# ---------------------------------------------------------------------------

class FAQEngine:
    """
    The core chatbot intelligence engine.

    - Maintains a TF-IDF index over the FAQ knowledge base.
    - Supports a secondary TF-IDF index over PDF chunks.
    - Provides fallback responses when confidence is below threshold.
    - Maintains per-session conversation history.
    """

    def __init__(
        self,
        faq_manager: Optional[FAQManager] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        """
        Initialize the FAQ Engine.

        Args:
            faq_manager:           FAQManager instance (creates a new one if None).
            confidence_threshold:  Minimum similarity score to consider a match valid.
        """
        self.faq_manager = faq_manager or FAQManager()
        self.threshold = confidence_threshold
        self.preprocessor = TextPreprocessor()

        # Indices
        self._faq_index = TFIDFIndex(self.preprocessor)
        self._pdf_index: Optional[TFIDFIndex] = None
        self._pdf_chunks: List[str] = []

        # Conversation history
        self._history: List[ConversationTurn] = []

        # Build the initial FAQ index
        self.rebuild_faq_index()

    # ------------------------------------------------------------------
    # Index Management
    # ------------------------------------------------------------------

    def rebuild_faq_index(self) -> None:
        """Rebuild the TF-IDF index from the current FAQ knowledge base."""
        faqs = self.faq_manager.get_all()
        if not faqs:
            logger.warning("No FAQs found — index will be empty.")
            return

        # Index over questions only (answers are used for retrieval)
        questions = [faq.question for faq in faqs]
        self._faq_index.fit(questions)
        logger.info("FAQ TF-IDF index rebuilt | %d questions indexed", len(questions))

    def index_pdf_chunks(self, chunks: List[str]) -> None:
        """
        Index text chunks extracted from a PDF document.

        Args:
            chunks:  List of text paragraphs/chunks from the PDF.
        """
        if not chunks:
            logger.warning("No PDF chunks provided for indexing.")
            return

        self._pdf_chunks = chunks
        self._pdf_index = TFIDFIndex(self.preprocessor)
        self._pdf_index.fit(chunks)
        logger.info("PDF TF-IDF index built | %d chunks indexed", len(chunks))

    def clear_pdf_index(self) -> None:
        """Remove the PDF index."""
        self._pdf_index = None
        self._pdf_chunks = []
        logger.info("PDF index cleared.")

    # ------------------------------------------------------------------
    # Query Interface
    # ------------------------------------------------------------------

    def query(self, user_query: str) -> ChatResponse:
        """
        Answer a user question by searching FAQ and PDF indices.

        Args:
            user_query:  Raw user question string.

        Returns:
            A ChatResponse object with the answer and metadata.
        """
        start_time = time.perf_counter()

        # ---- 1. Sanitize query ----
        query = user_query.strip()
        if not query:
            return self._build_fallback(query, start_time, "Empty query received.")

        # ---- 2. Search FAQ index ----
        faq_result = self._search_faq(query)
        faq_confidence = faq_result[1] if faq_result else 0.0

        # ---- 3. Search PDF index (if available) ----
        pdf_result = None
        pdf_confidence = 0.0
        if self._pdf_index and self._pdf_index.is_fitted:
            pdf_result = self._search_pdf(query)
            pdf_confidence = pdf_result[1] if pdf_result else 0.0

        # ---- 4. Determine best source ----
        if faq_confidence >= self.threshold and faq_confidence >= pdf_confidence:
            response = self._build_faq_response(query, faq_result, start_time)
        elif pdf_confidence >= self.threshold and pdf_confidence > faq_confidence:
            response = self._build_pdf_response(query, pdf_result, start_time)
        else:
            response = self._build_fallback(query, start_time)

        # ---- 5. Update history ----
        self._add_to_history("user", query)
        self._add_to_history(
            "assistant",
            response.answer,
            confidence=response.confidence,
            source=response.source,
        )

        return response

    # ------------------------------------------------------------------
    # Private search helpers
    # ------------------------------------------------------------------

    def _search_faq(self, query: str) -> Optional[Tuple[int, float]]:
        """Run FAQ index search and return the best (index, score) or None."""
        results = self._faq_index.query(query, top_k=1)
        return results[0] if results else None

    def _search_pdf(self, query: str) -> Optional[Tuple[int, float]]:
        """Run PDF index search and return the best (index, score) or None."""
        if self._pdf_index is None:
            return None
        results = self._pdf_index.query(query, top_k=1)
        return results[0] if results else None

    def _build_faq_response(
        self, query: str, result: Tuple[int, float], start_time: float
    ) -> ChatResponse:
        faqs = self.faq_manager.get_all()
        idx, score = result
        matched_faq: FAQItem = faqs[idx]
        elapsed = (time.perf_counter() - start_time) * 1000
        return ChatResponse(
            query=query,
            answer=matched_faq.answer,
            confidence=score,
            matched_question=matched_faq.question,
            source="FAQ",
            category=matched_faq.category,
            response_time_ms=elapsed,
            is_successful=True,
        )

    def _build_pdf_response(
        self, query: str, result: Tuple[int, float], start_time: float
    ) -> ChatResponse:
        idx, score = result
        chunk = self._pdf_chunks[idx]
        elapsed = (time.perf_counter() - start_time) * 1000
        # Return the chunk as the answer with a note about source
        return ChatResponse(
            query=query,
            answer=chunk,
            confidence=score,
            matched_question="[From uploaded PDF]",
            source="PDF",
            category="Document",
            response_time_ms=elapsed,
            is_successful=True,
        )

    def _build_fallback(
        self, query: str, start_time: float, reason: str = ""
    ) -> ChatResponse:
        elapsed = (time.perf_counter() - start_time) * 1000
        fallback_message = (
            "I'm sorry, I couldn't find a reliable answer to your question in my knowledge base. "
            "Please try rephrasing your question, or contact support for assistance."
        )
        return ChatResponse(
            query=query,
            answer=fallback_message,
            confidence=0.0,
            matched_question="",
            source="Fallback",
            category="Unknown",
            response_time_ms=elapsed,
            is_successful=False,
        )

    # ------------------------------------------------------------------
    # Conversation History
    # ------------------------------------------------------------------

    def _add_to_history(
        self,
        role: str,
        content: str,
        confidence: Optional[float] = None,
        source: Optional[str] = None,
    ) -> None:
        turn = ConversationTurn(role=role, content=content, confidence=confidence, source=source)
        self._history.append(turn)
        # Trim history if too long
        if len(self._history) > MAX_HISTORY_LENGTH * 2:
            self._history = self._history[-(MAX_HISTORY_LENGTH * 2):]

    def get_history(self) -> List[ConversationTurn]:
        """Return the full conversation history."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear all conversation history."""
        self._history = []
        logger.info("Conversation history cleared.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_pdf(self) -> bool:
        return self._pdf_index is not None and self._pdf_index.is_fitted

    @property
    def pdf_chunk_count(self) -> int:
        return len(self._pdf_chunks)

    @property
    def faq_count(self) -> int:
        return self.faq_manager.count
