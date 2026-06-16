"""
faq_manager.py - FAQ Knowledge Base Manager
============================================
Author: AI-Powered FAQ Chatbot System
Description:
    Manages CRUD operations on the FAQ knowledge base stored in JSON.
    Supports importing from CSV and JSON files.
    Thread-safe file operations using a lock mechanism.

Usage:
    from faq_manager import FAQManager
    manager = FAQManager()
    manager.add_faq("What is the fee?", "The fee is ₹50,000.", "Admissions")
"""

import json
import csv
import uuid
import logging
import threading
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_FAQ_PATH = Path(__file__).parent / "data" / "faq_data.json"


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

class FAQItem:
    """Represents a single FAQ entry."""

    def __init__(
        self,
        question: str,
        answer: str,
        category: str = "General",
        tags: Optional[List[str]] = None,
        faq_id: Optional[str] = None,
    ) -> None:
        self.id: str = faq_id or f"faq_{uuid.uuid4().hex[:8]}"
        self.question: str = question.strip()
        self.answer: str = answer.strip()
        self.category: str = category.strip()
        self.tags: List[str] = tags or []
        self.created_at: str = datetime.now().isoformat()
        self.updated_at: str = self.created_at

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "category": self.category,
            "question": self.question,
            "answer": self.answer,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FAQItem":
        item = cls(
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            category=data.get("category", "General"),
            tags=data.get("tags", []),
            faq_id=data.get("id"),
        )
        item.created_at = data.get("created_at", item.created_at)
        item.updated_at = data.get("updated_at", item.updated_at)
        return item


# ---------------------------------------------------------------------------
# FAQ Manager
# ---------------------------------------------------------------------------

class FAQManager:
    """
    Thread-safe CRUD manager for the FAQ knowledge base.

    Supports:
        - Loading / saving FAQs to JSON
        - Adding, editing, and deleting FAQ items
        - Importing from CSV or JSON files
        - Listing categories and searching by keyword
    """

    def __init__(self, faq_path: Path = DEFAULT_FAQ_PATH) -> None:
        """
        Initialize the FAQ Manager.

        Args:
            faq_path:  Path to the JSON file used as the knowledge base.
        """
        self.faq_path = Path(faq_path)
        self._lock = threading.Lock()
        self._faqs: List[FAQItem] = []

        # Ensure the directory exists
        self.faq_path.parent.mkdir(parents=True, exist_ok=True)

        self.load()
        logger.info("FAQManager initialized | %d FAQs loaded", len(self._faqs))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load FAQs from disk into memory."""
        with self._lock:
            if not self.faq_path.exists():
                logger.warning("FAQ file not found at %s — starting empty.", self.faq_path)
                self._faqs = []
                return
            try:
                with open(self.faq_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._faqs = [FAQItem.from_dict(item) for item in raw]
                logger.info("Loaded %d FAQs from %s", len(self._faqs), self.faq_path)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.error("Failed to parse FAQ file: %s", exc)
                self._faqs = []

    def save(self) -> None:
        """Persist the in-memory FAQ list to disk atomically."""
        with self._lock:
            tmp_path = self.faq_path.with_suffix(".tmp")
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(
                        [faq.to_dict() for faq in self._faqs],
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                tmp_path.replace(self.faq_path)
                logger.debug("FAQ data saved to %s", self.faq_path)
            except OSError as exc:
                logger.error("Failed to save FAQ data: %s", exc)
                if tmp_path.exists():
                    tmp_path.unlink()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get_all(self) -> List[FAQItem]:
        """Return all FAQ items (shallow copy of list)."""
        return list(self._faqs)

    def get_by_id(self, faq_id: str) -> Optional[FAQItem]:
        """Find and return a FAQ item by its ID, or None if not found."""
        return next((f for f in self._faqs if f.id == faq_id), None)

    def add_faq(
        self,
        question: str,
        answer: str,
        category: str = "General",
        tags: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """
        Add a new FAQ entry.

        Returns:
            (True, faq_id) on success or (False, error_message) on failure.
        """
        if not question.strip():
            return False, "Question cannot be empty."
        if not answer.strip():
            return False, "Answer cannot be empty."

        # Check for duplicate question
        if any(f.question.lower() == question.strip().lower() for f in self._faqs):
            return False, "A FAQ with this exact question already exists."

        item = FAQItem(question, answer, category, tags)
        self._faqs.append(item)
        self.save()
        logger.info("Added FAQ [%s]: %s", item.id, item.question[:60])
        return True, item.id

    def edit_faq(
        self,
        faq_id: str,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """
        Update fields of an existing FAQ.

        Returns:
            (True, 'Updated successfully') or (False, error_message).
        """
        item = self.get_by_id(faq_id)
        if item is None:
            return False, f"FAQ with id '{faq_id}' not found."

        if question is not None:
            item.question = question.strip()
        if answer is not None:
            item.answer = answer.strip()
        if category is not None:
            item.category = category.strip()
        if tags is not None:
            item.tags = tags
        item.updated_at = datetime.now().isoformat()
        self.save()
        logger.info("Edited FAQ [%s]", faq_id)
        return True, "Updated successfully."

    def delete_faq(self, faq_id: str) -> Tuple[bool, str]:
        """
        Delete a FAQ by ID.

        Returns:
            (True, 'Deleted successfully') or (False, error_message).
        """
        original_count = len(self._faqs)
        self._faqs = [f for f in self._faqs if f.id != faq_id]
        if len(self._faqs) == original_count:
            return False, f"FAQ with id '{faq_id}' not found."
        self.save()
        logger.info("Deleted FAQ [%s]", faq_id)
        return True, "Deleted successfully."

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def import_from_json(self, file_path: str) -> Tuple[int, int]:
        """
        Import FAQs from an external JSON file.

        Returns:
            (added_count, skipped_count)
        """
        added, skipped = 0, 0
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                q = item.get("question", "")
                a = item.get("answer", "")
                cat = item.get("category", "General")
                tags = item.get("tags", [])
                success, _ = self.add_faq(q, a, cat, tags)
                if success:
                    added += 1
                else:
                    skipped += 1
        except Exception as exc:
            logger.error("JSON import failed: %s", exc)
        return added, skipped

    def import_from_csv(self, file_path: str) -> Tuple[int, int]:
        """
        Import FAQs from a CSV file.
        Expected columns: question, answer, category (optional), tags (optional, comma-separated).

        Returns:
            (added_count, skipped_count)
        """
        added, skipped = 0, 0
        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    q = row.get("question", "").strip()
                    a = row.get("answer", "").strip()
                    cat = row.get("category", "General").strip()
                    raw_tags = row.get("tags", "")
                    tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
                    success, _ = self.add_faq(q, a, cat, tags)
                    if success:
                        added += 1
                    else:
                        skipped += 1
        except Exception as exc:
            logger.error("CSV import failed: %s", exc)
        return added, skipped

    def export_to_json(self, file_path: str) -> bool:
        """Export all FAQs to a JSON file."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([faq.to_dict() for faq in self._faqs], f, indent=2, ensure_ascii=False)
            return True
        except OSError as exc:
            logger.error("Export failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Querying Helpers
    # ------------------------------------------------------------------

    def get_categories(self) -> List[str]:
        """Return a sorted, deduplicated list of all FAQ categories."""
        return sorted(set(f.category for f in self._faqs))

    def get_by_category(self, category: str) -> List[FAQItem]:
        """Return all FAQs belonging to a specific category."""
        return [f for f in self._faqs if f.category.lower() == category.lower()]

    def search(self, keyword: str) -> List[FAQItem]:
        """Return FAQs whose question or answer contains the keyword (case-insensitive)."""
        kw = keyword.lower()
        return [
            f for f in self._faqs
            if kw in f.question.lower() or kw in f.answer.lower()
        ]

    @property
    def count(self) -> int:
        return len(self._faqs)
