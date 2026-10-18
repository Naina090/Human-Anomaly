"""Backward-compatible import location for the MongoDB report query."""
from app.services.repository import report_data

__all__ = ["report_data"]
