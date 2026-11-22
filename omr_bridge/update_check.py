# omr_bridge/update_check.py
# ---------------------------------------------------------------------------
# [v3.50] 새 버전 알림 — GitHub Releases의 '최신 릴리스'를 읽어 지금 버전보다 새것이 있는지 본다.
#   · 프로그램을 켤 때 하루 한 번(설정 파일에 마지막 확인 날짜), '정보' 창에서는 언제든 직접 확인.
#   · '이 버전은 알리지 않기'를 고르면 그 버전은 다시 묻지 않는다(더 새 버전이 나오면 다시 알림).
#   · 인터넷이 안 되거나 GitHub가 응답하지 않으면 조용히 넘어간다(자동 확인일 때).
#   · 끄기: 환경 변수 SPROUTY_NO_UPDATE_CHECK=1 (자동 확인만 끈다).
# ---------------------------------------------------------------------------
import json
import os
import re
import sys
import time

REPO = "zpischun-eng/OMR_Sprouty"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
PAGE_LATEST = f"https://github.com/{REPO}/releases/latest"


def parse_version(text):
    """'v3.50' · '3.5' · 'OMRSprouty_v3_50' → (3, 50). 소수 둘째 자리까지 같은 자릿수로 맞춘다(3.5 = 3.50)."""
    mo = re.search(r"(\d+)[._](\d+)", text or "")
    if not mo:
        mo2 = re.search(r"(\d+)", text or "")
        return (int(mo2.group(1)), 0) if mo2 else (0, 0)
    major, minor = mo.group(1), mo.group(2)
    if len(minor) == 1:
        minor += "0"
    return int(major), int(minor)


def is_newer(latest, current):
    return parse_version(latest) > parse_version(current)


def _state_path():
    from .paths import user_cache_root
    return user_cache_root() / "update_check.json"


def load_state():
    try:
        with open(_state_path(), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(st):
    try:
        p = _state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False)
    except OSError:
        pass


def due_today(st=None):
    """자동 확인을 할 때인가(오늘 아직 안 했고, 꺼 두지 않았으면)."""
    if os.environ.get("SPROUTY_NO_UPDATE_CHECK") == "1":
        return False
    st = load_state() if st is None else st
    return st.get("last_check") != time.strftime("%Y-%m-%d")


def mark_checked():
    st = load_state()
    st["last_check"] = time.strftime("%Y-%m-%d")
    save_state(st)


def skip_version(ver):
    st = load_state()
    st["skip"] = ver
    save_state(st)


def is_skipped(ver):
    return load_state().get("skip") == ver


def pick_asset(assets, platform=None, machine=None):
    """이 컴퓨터에 맞는 내려받기 파일 {"name","url"}(없으면 None).
    Windows: 설치 프로그램(_Setup.exe) → 폴더형 zip, 맥: Apple 칩(arm64)·인텔(x86_64) zip."""
    import platform as _pf
    platform = platform or sys.platform
    machine = (machine or _pf.machine() or "").lower()
    names = {a.get("name", ""): a for a in assets or []}

    def first(pred):
        for n, a in names.items():
            if pred(n):
                return {"name": n, "url": a.get("browser_download_url")}
        return None
    if platform.startswith("win"):
        return first(lambda n: n.lower().endswith("_setup.exe")) or first(lambda n: n.endswith("_Windows.zip"))
    if platform == "darwin":
        arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
        return first(lambda n: n.endswith(f"_Mac_{arch}.zip"))
    return None


def fetch_latest(timeout=8):
    """최신 릴리스 정보 {"version","page","notes","asset"}. 실패하면 예외."""
    import urllib.request
    req = urllib.request.Request(API_LATEST, headers={"Accept": "application/vnd.github+json",
                                                      "User-Agent": "OMR_Sprouty-update-check"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    tag = data.get("tag_name") or data.get("name") or ""
    major, minor = parse_version(tag)
    return {"version": f"{major}.{minor:02d}", "page": data.get("html_url") or PAGE_LATEST,
            "notes": (data.get("body") or "").strip(), "asset": pick_asset(data.get("assets"))}
