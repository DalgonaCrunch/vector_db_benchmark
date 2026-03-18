"""
On-demand package availability check and pip installation helper.

Used by the Streamlit UI to detect missing optional dependencies
and offer one-click installation.
"""
from __future__ import annotations

import importlib
import subprocess
import sys


def is_available(import_name: str) -> bool:
    """Return True if *import_name* can be imported."""
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def check_strategy_deps(requires: list[str], import_check: str | None) -> tuple[bool, list[str]]:
    """
    Return (all_available, missing_packages).

    Parameters
    ----------
    requires:
        pip package names required by the strategy.
    import_check:
        Python module name to import-test for the primary package.
    """
    missing: list[str] = []
    if import_check and not is_available(import_check):
        # Map import name back to pip package name
        pkg = _IMPORT_TO_PKG.get(import_check, import_check)
        if pkg not in missing:
            missing.append(pkg)
    return len(missing) == 0, missing


def install_package(pkg_name: str) -> tuple[bool, str]:
    """
    Install *pkg_name* via pip in the current interpreter.

    Returns
    -------
    (success, message)
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg_name],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return True, f"`{pkg_name}` 설치 완료."
        return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, f"`{pkg_name}` 설치 시간 초과."
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Import name → pip package name mapping
# ---------------------------------------------------------------------------

_IMPORT_TO_PKG: dict[str, str] = {
    "pypdf": "pypdf",
    "pdfplumber": "pdfplumber",
    "fitz": "pymupdf",
    "pytesseract": "pytesseract",
    "pdf2image": "pdf2image",
    "docx": "python-docx",
    "bs4": "beautifulsoup4",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
}
