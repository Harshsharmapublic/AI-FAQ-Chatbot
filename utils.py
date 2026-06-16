"""
utils.py - NLP Preprocessing Pipeline
=====================================
Author: AI-Powered FAQ Chatbot System
Description:
    Provides a reusable, industry-grade text preprocessing pipeline using NLTK.
    Handles lowercasing, tokenization, stopword removal, and lemmatization.
    Auto-downloads required NLTK datasets on first run.

Usage:
    from utils import TextPreprocessor
    preprocessor = TextPreprocessor()
    cleaned = preprocessor.preprocess("What is the Admission Fee?")
"""

import re
import os
import logging
import unicodedata
from typing import List, Optional
import nltk

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK Data Bootstrap
# ---------------------------------------------------------------------------

# Use a local directory for NLTK data so the project is self-contained.
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.insert(0, NLTK_DATA_DIR)

# Map each NLTK package to its correct lookup path
_NLTK_PACKAGES = {
    "punkt":     "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet":   "corpora/wordnet",
    "omw-1.4":   "corpora/omw-1.4",
}


def _bootstrap_nltk() -> None:
    """Download required NLTK packages silently if not already present."""
    for package, lookup_path in _NLTK_PACKAGES.items():
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            logger.info("Downloading NLTK package: %s", package)
            nltk.download(package, download_dir=NLTK_DATA_DIR, quiet=True)


_bootstrap_nltk()

from nltk.tokenize import word_tokenize          # noqa: E402
from nltk.corpus import stopwords as nltk_stopwords  # noqa: E402
from nltk.stem import WordNetLemmatizer         # noqa: E402


# ---------------------------------------------------------------------------
# Text Preprocessor
# ---------------------------------------------------------------------------

class TextPreprocessor:
    """
    Industry-grade NLP text preprocessing pipeline.

    Pipeline steps (in order):
        1. Unicode normalization
        2. Lowercase conversion
        3. URL / email / special-character removal
        4. Tokenization
        5. Stopword removal (with configurable extra stopwords)
        6. Lemmatization

    Example:
        >>> tp = TextPreprocessor()
        >>> tp.preprocess("What is the Admission Fee for first year?")
        'admission fee first year'
    """

    def __init__(
        self,
        language: str = "english",
        extra_stopwords: Optional[List[str]] = None,
        min_token_length: int = 2,
    ) -> None:
        """
        Initialize the preprocessor.

        Args:
            language:          NLTK stopword language corpus (default: 'english').
            extra_stopwords:   Additional domain-specific stopwords to filter out.
            min_token_length:  Minimum character length of tokens to keep (default: 2).
        """
        self.language = language
        self.min_token_length = min_token_length
        self.lemmatizer = WordNetLemmatizer()

        # Build combined stopword set
        base_stops = set(nltk_stopwords.words(language))
        domain_stops = set(extra_stopwords or [])
        self.stop_words = base_stops | domain_stops

        logger.debug(
            "TextPreprocessor initialized | language=%s | stopwords=%d",
            language,
            len(self.stop_words),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> str:
        """
        Run the full preprocessing pipeline on a single string.

        Args:
            text:  Raw input text.

        Returns:
            Cleaned, lemmatized text as a single whitespace-joined string.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        tokens = self._pipeline(text)
        return " ".join(tokens)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Apply preprocessing to a list of texts.

        Args:
            texts: List of raw text strings.

        Returns:
            List of preprocessed strings in the same order.
        """
        return [self.preprocess(t) for t in texts]

    def tokenize(self, text: str) -> List[str]:
        """Return only the tokens (list) without joining them."""
        return self._pipeline(text)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pipeline(self, text: str) -> List[str]:
        """Execute the full preprocessing pipeline, returning a token list."""
        text = self._normalize_unicode(text)
        text = self._to_lowercase(text)
        text = self._remove_noise(text)
        tokens = self._tokenize(text)
        tokens = self._remove_stopwords(tokens)
        tokens = self._lemmatize(tokens)
        tokens = self._filter_by_length(tokens)
        return tokens

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Normalize unicode characters to ASCII-safe equivalents."""
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

    @staticmethod
    def _to_lowercase(text: str) -> str:
        return text.lower()

    @staticmethod
    def _remove_noise(text: str) -> str:
        """Remove URLs, email addresses, punctuation, and extra whitespace."""
        text = re.sub(r"https?://\S+|www\.\S+", "", text)          # URLs
        text = re.sub(r"\S+@\S+\.\S+", "", text)                   # Emails
        text = re.sub(r"[^\w\s]", " ", text)                       # Punctuation
        text = re.sub(r"\d+", " ", text)                           # Numbers (optional)
        text = re.sub(r"\s+", " ", text).strip()                   # Extra whitespace
        return text

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return word_tokenize(text)

    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in self.stop_words]

    def _lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def _filter_by_length(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if len(t) >= self.min_token_length]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def truncate_text(text: str, max_length: int = 300, suffix: str = "...") -> str:
    """Truncate text to max_length characters, appending suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_confidence(score: float) -> str:
    """Convert a 0-1 similarity score to a human-readable percentage string."""
    return f"{score * 100:.1f}%"


def get_confidence_label(score: float, threshold: float = 0.30) -> str:
    """Return a human-readable confidence label based on score."""
    if score >= 0.70:
        return "High Confidence"
    elif score >= threshold:
        return "Moderate Confidence"
    else:
        return "Low Confidence"


def sanitize_filename(name: str) -> str:
    """Remove unsafe characters from a filename."""
    return re.sub(r"[^\w\-_\.]", "_", name)
