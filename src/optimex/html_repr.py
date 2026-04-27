"""Helpers for rich HTML representations in notebooks."""

from html import escape
from typing import Any, Dict


def _format_value(value: Any) -> str:
    """Format common container values for compact HTML display."""
    if value is None:
        return "None"
    if isinstance(value, (list, tuple, set, dict)):
        return f"{type(value).__name__}(len={len(value)})"
    return escape(str(value))


def simple_repr_html(title: str, rows: Dict[str, Any]) -> str:
    """Return a compact two-column HTML summary table."""
    body = "".join(
        "<tr>"
        f"<th style='text-align:left;padding:4px 8px;background:#f6f8fa;'>{escape(str(k))}</th>"
        f"<td style='padding:4px 8px;'>{_format_value(v)}</td>"
        "</tr>"
        for k, v in rows.items()
    )
    return (
        "<div style='font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;'>"
        f"<div style='font-weight:600;margin:0 0 6px 0;'>{escape(title)}</div>"
        "<table style='border-collapse:collapse;border:1px solid #d0d7de;'>"
        f"{body}</table></div>"
    )
