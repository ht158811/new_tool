"""Compatibility re-export for parser consumers.

The DTO lives in the application layer so application ports never depend on an
infrastructure implementation.
"""

from dataset_manager.application.dto.parse_result import ParseIssue, ParseResult, Severity

__all__ = ["ParseIssue", "ParseResult", "Severity"]

