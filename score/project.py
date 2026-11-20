# score/project.py
# ---------------------------------------------------------------------------
# [v2.50 편집 저장·이어서 편집] 편집한 악보를 원본 옆에 같은 이름 + ".sprouty"로 저장하고, 다음에 원본을
# 열 때 이어서 편집할 수 있게 다시 읽는다.
#
# 담는 것: 편집된 악보(Score 전체 — 음표·이음줄·아티큘레이션·넣은 음·정렬·기둥 뒤집기 등), 곡 정보(header_items),
# 원본 파일 지문(이름·크기·수정 시각), 저장 시각, 프로그램 버전, 파일 형식 버전.
# 담지 않는 것: 원본 그림·오선 위치·페이지 배치 — 원본을 다시 열면 인식(캐시라 빠름)이 그대로 다시 만든다.
#
# 형식: gzip으로 압축한 pickle(파이썬 객체 그대로 — 음표↔이음줄 같은 서로 가리키는 관계가 유지된다).
# pickle은 믿을 수 있는 파일만 열어야 하므로, 이 프로그램이 만든 자기 편집 파일만 연다(원본 옆 같은 이름).
# ---------------------------------------------------------------------------
import gzip
import os
import pickle
import time

EXT = ".sprouty"
FORMAT_VERSION = 1
_MAGIC = "OMR_Sprouty-edit"


def project_path(source_path: str) -> str:
    """원본과 같은 경로·같은 이름, 확장자만 .sprouty."""
    return os.path.splitext(source_path)[0] + EXT


def _fingerprint(source_path: str) -> dict:
    try:
        st = os.stat(source_path)
        return {"name": os.path.basename(source_path), "size": st.st_size, "mtime": int(st.st_mtime)}
    except OSError:
        return {"name": os.path.basename(source_path), "size": None, "mtime": None}


def save(source_path: str, score, header_items, app_version: str) -> str:
    """편집 파일을 쓴다(먼저 임시 파일에 쓰고 바꿔 넣어, 쓰다가 끊겨도 이전 파일이 깨지지 않게). 경로를 돌려준다."""
    path = project_path(source_path)
    data = {"magic": _MAGIC, "format": FORMAT_VERSION, "app_version": app_version,
            "saved_at": time.strftime("%Y-%m-%d %H:%M"), "source": _fingerprint(source_path),
            "score": score, "header_items": list(header_items or [])}
    tmp = path + ".tmp"
    with gzip.open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


def peek(source_path: str):
    """원본 옆에 편집 파일이 있으면 요약 {"path","saved_at","app_version","changed"}를, 없으면 None.
    changed: 저장한 뒤 원본 파일이 바뀌었는가(크기·수정 시각)."""
    path = project_path(source_path)
    if not os.path.exists(path):
        return None
    try:
        data = load(path)
    except Exception:
        return {"path": path, "saved_at": "?", "app_version": "?", "changed": False, "broken": True}
    fp_now, fp_old = _fingerprint(source_path), data.get("source") or {}
    changed = bool(fp_old.get("size")) and (fp_old.get("size") != fp_now.get("size")
                                             or fp_old.get("mtime") != fp_now.get("mtime"))
    return {"path": path, "saved_at": data.get("saved_at", "?"), "app_version": data.get("app_version", "?"),
            "changed": changed, "broken": False}


def load(path: str) -> dict:
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or data.get("magic") != _MAGIC:
        raise ValueError("콩나물지기 편집 파일이 아닙니다")
    if data.get("format", 0) > FORMAT_VERSION:
        raise ValueError("더 새 버전의 콩나물지기에서 저장한 편집 파일입니다 — 프로그램을 업데이트해 주세요")
    return data
