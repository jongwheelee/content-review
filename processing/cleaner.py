"""Data cleaning utilities for Content Review Pipeline."""

import re
from typing import Any

import pandas as pd
import numpy as np


class DataCleaner:
    """Utilities for cleaning ingested data."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s.,!?;:()\-\'\"]", "", text)

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("''", '"').replace("''", '"')

        # Remove URLs
        text = re.sub(r"http[s]?://\S+", "", text)

        # Strip leading/trailing whitespace
        return text.strip()

    @staticmethod
    def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric columns in a DataFrame."""
        df_clean = df.copy()

        for col in df_clean.select_dtypes(include=[np.number]).columns:
            # Replace inf with NaN
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)

            # Fill NaN with column median
            median = df_clean[col].median()
            if not np.isnan(median):
                df_clean[col] = df_clean[col].fillna(median)

            # Z-score outlier capping (clip at 3 standard deviations)
            if len(df_clean) > 3:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                if std > 0:
                    lower = mean - 3 * std
                    upper = mean + 3 * std
                    df_clean[col] = df_clean[col].clip(lower, upper)

        return df_clean

    @staticmethod
    def deduplicate_records(
        records: list[dict[str, Any]], key_fields: list[str]
    ) -> list[dict[str, Any]]:
        """Remove duplicate records based on key fields."""
        seen = set()
        unique = []

        for record in records:
            key = tuple(record.get(field, "") for field in key_fields)
            if key not in seen:
                seen.add(key)
                unique.append(record)

        return unique

    @staticmethod
    def validate_json_structure(
        data: dict[str, Any], required_fields: list[str]
    ) -> tuple[bool, list[str]]:
        """Validate that JSON has required fields."""
        missing = []

        for field in required_fields:
            if field not in data:
                missing.append(field)

        return len(missing) == 0, missing

    @staticmethod
    def sanitize_for_json(text: str) -> str:
        """Sanitize text for safe JSON storage."""
        # Escape backslashes
        text = text.replace("\\", "\\\\")
        # Escape quotes
        text = text.replace('"', '\\"')
        # Remove control characters
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
        return text

    @staticmethod
    def parse_date(date_str: str) -> str | None:
        """Parse various date formats to ISO format."""
        if not date_str:
            return None

        # Common date formats
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%B %d, %Y",
            "%b %d, %Y",
        ]

        from datetime import datetime

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        return None

    @staticmethod
    def chunk_text(text: str, max_length: int = 4000, overlap: int = 200) -> list[str]:
        """Split long text into overlapping chunks."""
        if len(text) <= max_length:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + max_length

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending in the last 200 chars
                search_text = text[max(start, end - 200) : end]
                for punct in [". ", ".\n", "! ", "? "]:
                    idx = search_text.rfind(punct)
                    if idx != -1:
                        end = max(start, end - 200) + idx + len(punct)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks
