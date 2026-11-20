# omr_bridge/ocr_data.py
# ---------------------------------------------------------------------------
# [v3.20] Audiveris 글자 인식(코드 이름·빠르기말 등)용 언어 데이터(Tesseract tessdata).
# Audiveris는 글자 인식 엔진을 내장하지만 언어 데이터가 없으면 글자 인식을 통째로 끈다
# (로그: "No installed OCR languages"). 그래서 그림 악보의 기타 코드를 하나도 못 읽었다.
#   · 찾는 곳: ① 프로그램 폴더 resource_dist/tessdata ② %LOCALAPPDATA%/Sprouty/tessdata
#   · 받기: download() — 정식 tessdata(옛 엔진 포함판. 'fast'판은 Audiveris가 쓰는 옛 엔진이 없어 실패한다)
# ---------------------------------------------------------------------------
import os
from pathlib import Path

LANG = "eng"                       # 코드 이름은 라틴 문자 — 한국어까지 넣으면 느리고 이득이 없다
URL = "https://raw.githubusercontent.com/tesseract-ocr/tessdata/main/{lang}.traineddata"
MIN_SIZE = 5_000_000               # 정식판 eng ≈ 23MB (fast판 ≈ 4MB — 쓰면 안 됨)


def _user_dir() -> Path:
    from omr_bridge.paths import user_cache_root          # [v3.30] 운영체제별
    return user_cache_root() / "tessdata"


def _bundled_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "resource_dist" / "tessdata"


def tessdata_dir():
    """쓸 수 있는 언어 데이터 폴더(없으면 None)."""
    for d in (os.environ.get("SPROUTY_TESSDATA"), _bundled_dir(), _user_dir()):
        if not d:
            continue
        f = Path(d) / f"{LANG}.traineddata"
        try:
            if f.exists() and f.stat().st_size >= MIN_SIZE:
                return Path(d)
        except OSError:
            continue
    return None


def available() -> bool:
    return tessdata_dir() is not None


def apply(env: dict, cmd_head: list) -> list:
    """Audiveris 실행 환경·옵션에 글자 인식을 켠다(데이터가 있을 때만). 반환: 덧붙일 옵션 목록."""
    d = tessdata_dir()
    if d is None or not audiveris_ocr_on():
        return []
    env["TESSDATA_PREFIX"] = str(d)
    return ["-option", f"org.audiveris.omr.text.Language.defaultSpecification={LANG}"]


def cache_tag() -> str:
    """캐시 열쇠에 붙일 꼬리 — 글자 인식을 켜면 예전(글자 인식 없이) 결과를 다시 쓰지 않게."""
    return "+ocr" if (available() and audiveris_ocr_on()) else ""


def audiveris_ocr_on() -> bool:
    """Audiveris 자체 글자 인식은 시험 그림에서 코드 12개 중 1개만 읽어(나머지는 쪼개지거나 빠짐) 기본으로 끈다.
    코드는 chord_ocr(Tesseract 직접)가 읽는다. 시험용: 환경변수 SPROUTY_AUDIVERIS_OCR=1."""
    return os.environ.get("SPROUTY_AUDIVERIS_OCR") == "1"


def download(progress=None) -> Path:
    """언어 데이터를 사용자 폴더로 받는다. progress(받은 바이트, 전체 바이트). 실패하면 예외."""
    import urllib.request
    d = _user_dir()
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"{LANG}.traineddata"
    tmp = d / f"{LANG}.traineddata.part"
    with urllib.request.urlopen(URL.format(lang=LANG), timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if progress:
                progress(got, total)
    if tmp.stat().st_size < MIN_SIZE:
        tmp.unlink()
        raise RuntimeError("받은 파일이 너무 작습니다(정식 언어 데이터가 아님)")
    os.replace(tmp, dst)
    return dst


def find_tesseract():
    """[v3.20] 이 PC의 Tesseract 실행 파일과 쓸 언어 데이터 폴더((exe, tessdata) 또는 None).
    언어 데이터는 우리 폴더(정식 eng)를 먼저, 없으면 Tesseract 설치 폴더의 것."""
    import shutil
    exe = shutil.which("tesseract") or next(
        (p for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe", "/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract",
                     r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe") if os.path.exists(p)), None)
    if not exe:
        return None
    d = tessdata_dir()
    if d is None:
        for c in (os.path.join(os.path.dirname(exe), "tessdata"), "/usr/share/tesseract-ocr/5/tessdata",
                  "/usr/share/tesseract-ocr/4.00/tessdata", "/usr/local/share/tessdata"):
            if os.path.exists(os.path.join(c, "eng.traineddata")):
                d = Path(c)
                break
    return (exe, str(d)) if d else None
