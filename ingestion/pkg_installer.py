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


def _pip_available() -> bool:
    """Return True if pip is usable in the current interpreter."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _uv_available() -> str | None:
    """Return the path to uv if it is on PATH, else None."""
    import shutil
    return shutil.which("uv")


def install_package(pkg_name: str) -> tuple[bool, str]:
    """
    Install *pkg_name* in the current interpreter.

    Tries, in order:
    1. ``python -m pip install``  (fastest if pip is already present)
    2. ``python -m ensurepip --upgrade`` then pip  (pip bootstrap)
    3. ``uv pip install --python <executable>``  (uv-managed venvs)

    Returns
    -------
    (success, message)
    """
    try:
        # ── Strategy 1: pip already available ────────────────────────────────
        if _pip_available():
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg_name],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                return True, f"`{pkg_name}` 설치 완료."
            return False, result.stderr.strip()

        # ── Strategy 2: bootstrap pip via ensurepip ───────────────────────────
        bootstrap = subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            capture_output=True, text=True,
        )
        if bootstrap.returncode == 0:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg_name],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                return True, f"`{pkg_name}` 설치 완료."
            return False, result.stderr.strip()

        # ── Strategy 3: uv pip install (uv-managed venvs without pip) ─────────
        uv = _uv_available()
        if uv:
            result = subprocess.run(
                [uv, "pip", "install", "--python", sys.executable, pkg_name],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                return True, f"`{pkg_name}` 설치 완료."
            return False, result.stderr.strip()

        return False, "pip를 사용할 수 없습니다. `uv pip install " + pkg_name + "` 를 직접 실행해주세요."

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
