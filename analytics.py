"""
analytics.py - Query Analytics Logger and Dashboard
=====================================================
Author: AI-Powered FAQ Chatbot System
Description:
    Logs all chatbot queries and responses to a CSV file.
    Provides methods to generate Plotly charts for the analytics dashboard:
      - Query volume over time
      - Success vs. failure rate
      - Confidence score distribution
      - Most frequently asked questions
      - Category breakdown

Usage:
    from analytics import AnalyticsManager
    mgr = AnalyticsManager()
    mgr.log_query(response)
    fig = mgr.plot_query_volume()
"""

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOGS_DIR = Path(__file__).parent / "logs"
LOG_FILE = LOGS_DIR / "query_log.csv"

CSV_COLUMNS = [
    "timestamp",
    "query",
    "matched_question",
    "answer",
    "confidence",
    "source",
    "category",
    "is_successful",
    "response_time_ms",
]

# Color palette (dark-mode friendly)
COLOR_SUCCESS = "#00D4AA"
COLOR_FAILURE = "#FF4B4B"
COLOR_PRIMARY = "#7C3AED"
COLOR_SECONDARY = "#4F46E5"
PLOTLY_TEMPLATE = "plotly_dark"


# ---------------------------------------------------------------------------
# Analytics Manager
# ---------------------------------------------------------------------------

class AnalyticsManager:
    """
    Logs chatbot interactions and provides rich Plotly visualizations.

    All data is stored in a CSV log file at logs/query_log.csv.
    Analytics are computed from the full log on-demand.
    """

    def __init__(self, log_file: Path = LOG_FILE) -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_log_header()
        logger.info("AnalyticsManager initialized | log=%s", self.log_file)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _ensure_log_header(self) -> None:
        """Create log file with header if it doesn't exist."""
        if not self.log_file.exists() or self.log_file.stat().st_size == 0:
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    def log_query(self, response_data: Dict[str, Any]) -> None:
        """
        Append a query-response record to the log.

        Args:
            response_data: Dictionary with keys matching CSV_COLUMNS.
                           Typically built from a ChatResponse object.
        """
        try:
            row = {col: response_data.get(col, "") for col in CSV_COLUMNS}
            if not row["timestamp"]:
                row["timestamp"] = datetime.now().isoformat()
            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(row)
        except OSError as exc:
            logger.error("Failed to log query: %s", exc)

    def log_chat_response(self, chat_response) -> None:
        """
        Convenience method: log a ChatResponse dataclass instance.

        Args:
            chat_response:  ChatResponse object from chatbot.py.
        """
        self.log_query({
            "timestamp": chat_response.timestamp,
            "query": chat_response.query,
            "matched_question": chat_response.matched_question,
            "answer": chat_response.answer[:500],   # Truncate long answers
            "confidence": round(chat_response.confidence, 4),
            "source": chat_response.source,
            "category": chat_response.category,
            "is_successful": chat_response.is_successful,
            "response_time_ms": round(chat_response.response_time_ms, 2),
        })

    # ------------------------------------------------------------------
    # Data Access
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """Load the full analytics log as a DataFrame."""
        try:
            if not self.log_file.exists() or self.log_file.stat().st_size == 0:
                return pd.DataFrame(columns=CSV_COLUMNS)
            df = pd.read_csv(self.log_file, parse_dates=["timestamp"])
            df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0)
            df["is_successful"] = df["is_successful"].map(
                {True: True, "True": True, False: False, "False": False}
            ).fillna(False)
            df["response_time_ms"] = pd.to_numeric(df["response_time_ms"], errors="coerce").fillna(0)
            return df
        except Exception as exc:
            logger.error("Failed to load analytics data: %s", exc)
            return pd.DataFrame(columns=CSV_COLUMNS)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Return key performance metrics as a dictionary."""
        df = self.load_data()
        if df.empty:
            return {
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "success_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_response_time_ms": 0.0,
                "total_sources_faq": 0,
                "total_sources_pdf": 0,
            }

        total = len(df)
        successful = df["is_successful"].sum()
        failed = total - successful

        return {
            "total_queries": total,
            "successful_queries": int(successful),
            "failed_queries": int(failed),
            "success_rate": round(successful / total * 100, 1) if total > 0 else 0.0,
            "avg_confidence": round(df.loc[df["is_successful"], "confidence"].mean() * 100, 1),
            "avg_response_time_ms": round(df["response_time_ms"].mean(), 1),
            "total_sources_faq": int((df["source"] == "FAQ").sum()),
            "total_sources_pdf": int((df["source"] == "PDF").sum()),
        }

    def get_top_queries(self, n: int = 10) -> pd.DataFrame:
        """Return the top-N most frequently asked queries."""
        df = self.load_data()
        if df.empty:
            return pd.DataFrame(columns=["query", "count"])
        return (
            df["query"]
            .value_counts()
            .head(n)
            .reset_index()
            .rename(columns={"query": "query", "count": "count"})
        )

    def clear_logs(self) -> None:
        """Delete and reinitialize the log file."""
        try:
            if self.log_file.exists():
                self.log_file.unlink()
            self._ensure_log_header()
            logger.info("Analytics log cleared.")
        except OSError as exc:
            logger.error("Failed to clear logs: %s", exc)

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def plot_query_volume(self) -> Optional[go.Figure]:
        """Line chart: number of queries over time (by day)."""
        df = self.load_data()
        if df.empty:
            return None

        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        daily = df.groupby("date").size().reset_index(name="count")

        fig = px.area(
            daily,
            x="date",
            y="count",
            title="📊 Query Volume Over Time",
            labels={"date": "Date", "count": "Queries"},
            color_discrete_sequence=[COLOR_PRIMARY],
            template=PLOTLY_TEMPLATE,
        )
        fig.update_traces(fill="tozeroy", line_color=COLOR_PRIMARY, fillcolor="rgba(124,58,237,0.15)")
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def plot_success_rate(self) -> Optional[go.Figure]:
        """Donut chart: successful vs. failed queries."""
        df = self.load_data()
        if df.empty:
            return None

        success = int(df["is_successful"].sum())
        fail = len(df) - success

        fig = go.Figure(go.Pie(
            labels=["Successful", "Failed"],
            values=[success, fail],
            hole=0.60,
            marker_colors=[COLOR_SUCCESS, COLOR_FAILURE],
            textinfo="label+percent",
            hoverinfo="label+value",
        ))
        fig.update_layout(
            title_text="✅ Response Success Rate",
            template=PLOTLY_TEMPLATE,
            showlegend=True,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text=f"{success}/{len(df)}", x=0.5, y=0.5, font_size=18, showarrow=False)]
        )
        return fig

    def plot_confidence_distribution(self) -> Optional[go.Figure]:
        """Histogram: distribution of confidence scores."""
        df = self.load_data()
        if df.empty or "confidence" not in df.columns:
            return None

        successful = df[df["is_successful"] == True]["confidence"] * 100

        fig = px.histogram(
            successful,
            nbins=20,
            title="🎯 Confidence Score Distribution",
            labels={"value": "Confidence (%)", "count": "Frequency"},
            color_discrete_sequence=[COLOR_SECONDARY],
            template=PLOTLY_TEMPLATE,
        )
        fig.add_vline(
            x=successful.mean(),
            line_dash="dash",
            line_color=COLOR_SUCCESS,
            annotation_text=f"Avg: {successful.mean():.1f}%",
            annotation_position="top right",
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        return fig

    def plot_top_questions(self, n: int = 10) -> Optional[go.Figure]:
        """Horizontal bar chart: most frequently asked questions."""
        df = self.load_data()
        if df.empty:
            return None

        top = df["query"].value_counts().head(n).reset_index()
        top.columns = ["query", "count"]
        top["query_short"] = top["query"].str[:55] + "..."

        fig = px.bar(
            top.sort_values("count"),
            x="count",
            y="query_short",
            orientation="h",
            title=f"🔥 Top {n} Most Asked Questions",
            labels={"count": "Times Asked", "query_short": "Question"},
            color="count",
            color_continuous_scale=["#4F46E5", "#7C3AED", "#A855F7"],
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            yaxis_title=None,
        )
        return fig

    def plot_category_breakdown(self) -> Optional[go.Figure]:
        """Bar chart: queries grouped by FAQ category."""
        df = self.load_data()
        if df.empty or "category" not in df.columns:
            return None

        cat_counts = df["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]

        fig = px.bar(
            cat_counts,
            x="category",
            y="count",
            title="📂 Queries by Category",
            labels={"category": "Category", "count": "Queries"},
            color="count",
            color_continuous_scale=["#4F46E5", "#7C3AED", "#EC4899"],
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
        )
        return fig

    def plot_source_breakdown(self) -> Optional[go.Figure]:
        """Pie chart: responses from FAQ vs. PDF vs. Fallback."""
        df = self.load_data()
        if df.empty:
            return None

        source_counts = df["source"].value_counts().reset_index()
        source_counts.columns = ["source", "count"]

        color_map = {"FAQ": COLOR_PRIMARY, "PDF": COLOR_SECONDARY, "Fallback": COLOR_FAILURE}
        colors = [color_map.get(s, "#888") for s in source_counts["source"]]

        fig = go.Figure(go.Bar(
            x=source_counts["source"],
            y=source_counts["count"],
            marker_color=colors,
            text=source_counts["count"],
            textposition="outside",
        ))
        fig.update_layout(
            title="📚 Response Source Breakdown",
            template=PLOTLY_TEMPLATE,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Source",
            yaxis_title="Count",
        )
        return fig
