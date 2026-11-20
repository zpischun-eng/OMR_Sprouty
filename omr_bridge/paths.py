# omr_bridge/paths.py
# ---------------------------------------------------------------------------
# [v3.30] 운영체제별 사용자 폴더(맥 지원).
#   Windows: %LOCALAPPDATA%\Sprouty\...      (예전과 같음)
#   macOS:   ~/Library/Caches/Sprouty/...    (맥 관례)
#   그 밖:   $XDG_CACHE_HOME 또는 ~/.cache/Sprouty/...
# ---------------------------------------------------------------------------
import os
import sys
from pathlib import Path


def user_cache_root() -> Path:
    if os.environ.get("LOCALAPPDATA"):
        base = os.environ["LOCALAPPDATA"]
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "Sprouty"
