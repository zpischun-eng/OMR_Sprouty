# metadata.py
# ---------------------------------------------------------------------------
# [업그레이드 4단계] 곡 제목·작사/작곡 크레딧(한글) 인식.
#
# 설계:
#   * 첫 시스템 '위쪽 여백'(제목·크레딧이 있을 만한 영역)을 원본 해상도로
#     크롭해 OCR 한다 — 좌표 재활용 방식은 omr/lyrics.py 와 동일하다.
#   * 제목(큰 글씨)과 크레딧("작사 곡 아무개")은 보통 글자 크기·위치가
#     확연히 다른데, 이걸 한 번에 OCR 하면 Tesseract 가 세그멘테이션에
#     실패해 둘 다 깨진다(실측: 함께 넣으면 제목까지 깨짐, 따로 넣으면
#     둘 다 정상). 그래서 먼저 '행 잉크 밀도'로 텍스트 밴드를 나누고,
#     밴드별로 따로 OCR 한다.
#   * 밴드 중 일부는 코드 기호(F, Dm7 등) 줄일 수 있다 — 이건 한글이
#     아니므로 OCR 결과에 한글 비율로 걸러낸다.
#   * 가장 높이가 큰(=가장 큰 글씨) 밴드를 제목으로, 나머지 한글 밴드를
#     크레딧으로 묶는다. 완전히 휴리스틱이라 모든 악보에서 되진 않는다 —
#     실패하면 조용히 None 을 반환한다(가사와 같은 원칙: 실패해도 나머지
#     인식에 영향 없어야 함).
# ---------------------------------------------------------------------------
import os
import re

import cv2
import numpy as np

try:
    import pytesseract
    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False
    pytesseract = None

_HANGUL_RE = re.compile(r"[가-힣]")

MIN_BAND_H = 10          # 이보다 얇은 밴드는 잡음으로 버림
MERGE_GAP = 12           # 이 간격(px) 이내의 밴드는 한 문단으로 합침
MIN_HANGUL_RATIO = 0.4   # OCR 결과 중 한글 비율이 이 미만이면 버림(코드 기호 등)

# --- 아래는 원래 omr/lyrics.py(가사 인식, 인식률 저조로 전체 삭제됨)에
#     있던 Tesseract 가용성 진단 유틸리티다. 제목/크레딧 인식(이 파일)도
#     같은 Tesseract 엔진을 쓰므로 그대로 재사용한다. ---
# Windows 설치 프로그램은 기본적으로 PATH 에 자동으로 추가하지 않는
# 경우가 흔하다(설치는 됐는데 pytesseract 가 못 찾는 실측 사례). PATH
# 탐색이 실패하면 이 흔한 기본 설치 경로들을 대신 찾아본다 — 있으면
# pytesseract 에 직접 경로를 알려준다(사용자가 PATH 를 직접 안 고쳐도
# 되게).
_WINDOWS_FALLBACK_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    os.path.expandvars(r"%USERPROFILE%\AppData\Local\Tesseract-OCR\tesseract.exe"),
]


def _try_fallback_paths():
    """PATH 에 없어도 흔한 Windows 기본 설치 경로에 있으면 자동으로 잡는다."""
    if not _HAS_TESSERACT:
        return False
    for p in _WINDOWS_FALLBACK_PATHS:
        if p and os.path.isfile(p):
            pytesseract.pytesseract.tesseract_cmd = p
            return True
    return False


def tesseract_available() -> bool:
    if not _HAS_TESSERACT:
        return False
    try:
        langs = pytesseract.get_languages(config="")
        return "kor" in langs
    except Exception:
        pass
    # PATH 에서 바로 안 잡히면 흔한 Windows 설치 경로를 대신 찾아본다.
    if _try_fallback_paths():
        try:
            langs = pytesseract.get_languages(config="")
            return "kor" in langs
        except Exception:
            return False
    return False


def diagnose() -> str | None:
    """가사 인식이 왜 비활성인지 사람이 읽을 문장으로 알려준다. 정상이면 None.

    지금까지는 Tesseract 가 없으면 조용히 건너뛰기만 해서, 사용자가
    "가사가 왜 안 나오는지" 전혀 알 수 없었다(실측 제보: 크래시도
    경고창도 없이 그냥 비어 있음). pytesseract 는 파이썬 바인딩일 뿐
    실제 OCR 엔진(Tesseract-OCR 실행 파일)은 별도 설치가 필요한데,
    이 둘을 혼동하기 쉽다 — 그래서 어느 단계에서 막혔는지 구분해 알려준다.

    Windows 에서는 '설치는 했는데 PATH 에 못 잡는' 경우가 실측으로
    확인됐다(설치 프로그램이 기본적으로 PATH 자동 추가를 안 하거나,
    설치 후 터미널/앱을 재시작 안 해 이전 PATH 값이 남아 있는 경우).
    이 경우 흔한 기본 설치 경로를 자동으로 찾아 시도한 뒤에도 안 되면,
    정확히 어디를 찾아봤는지까지 알려준다(사용자가 직접 경로를 눈으로
    확인하고 비교할 수 있게).
    """
    if not _HAS_TESSERACT:
        return ("pytesseract 파이썬 패키지가 설치되어 있지 않습니다.\n"
                "pip install pytesseract")
    try:
        ver = pytesseract.get_tesseract_version()
    except Exception:
        if _try_fallback_paths():
            try:
                ver = pytesseract.get_tesseract_version()
            except Exception:
                ver = None
        else:
            ver = None
        if ver is None:
            checked = "\n".join(f"  · {p}" for p in _WINDOWS_FALLBACK_PATHS)
            return (
                "Tesseract-OCR 실행 파일을 찾을 수 없습니다(pytesseract는 "
                "설치돼 있지만, 이건 파이썬 연결 라이브러리일 뿐 실제 OCR "
                "엔진은 별도 설치가 필요합니다).\n"
                "설치는 하셨다면 다음 중 하나일 가능성이 높습니다:\n"
                "  1) 설치 시 PATH 자동 추가를 안 했다 — 설치 경로를 "
                "환경변수 PATH에 직접 추가한 뒤 터미널/앱을 재시작하세요.\n"
                "  2) 설치 직후라 이전 PATH 값이 아직 적용 중이다 — 컴퓨터를 "
                "재시작해 보세요.\n"
                "  3) 기본 위치가 아닌 다른 폴더에 설치했다 — 코드에서 "
                "pytesseract.pytesseract.tesseract_cmd = r'설치경로\\tesseract.exe' "
                "로 직접 지정하세요.\n"
                "자동으로 다음 기본 경로들을 확인했지만 못 찾았습니다:\n"
                f"{checked}\n"
                "Windows 설치: https://github.com/UB-Mannheim/tesseract/wiki")
    try:
        langs = pytesseract.get_languages(config="")
    except Exception:
        return f"Tesseract-OCR({ver})은 실행되지만 언어팩 목록을 읽지 못했습니다."
    if "kor" not in langs:
        return (f"Tesseract-OCR({ver})은 설치돼 있지만 한국어 언어팩(kor)이 "
                "없습니다.\nWindows: Tesseract 설치 프로그램을 다시 실행해 "
                "'Korean' 언어를 추가로 체크하거나, kor.traineddata 파일을 "
                "Tesseract의 tessdata 폴더에 넣으세요.")
    return None



def _find_bands(gray_crop):
    """행 잉크 밀도로 텍스트 밴드(y0,y1) 목록을 찾는다(가까운 것은 병합)."""
    ink = (255 - gray_crop) > 40
    row_ink = ink.sum(axis=1)
    in_band = row_ink > 3
    raw = []
    start = None
    for y, v in enumerate(in_band):
        if v and start is None:
            start = y
        elif not v and start is not None:
            if y - start >= MIN_BAND_H:
                raw.append([start, y])
            start = None
    if start is not None and len(in_band) - start >= MIN_BAND_H:
        raw.append([start, len(in_band)])
    merged = []
    for b in raw:
        if merged and b[0] - merged[-1][1] <= MERGE_GAP:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    return [tuple(b) for b in merged]


def _ocr_band(bgr_crop):
    """텍스트 + 평균 신뢰도 + 실제 글자 높이(폰트 크기 추정용)를 돌려준다."""
    if bgr_crop.size == 0 or bgr_crop.shape[0] < 3:
        return "", 0.0, 0.0
    big = cv2.resize(bgr_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        # psm 6('균일한 문단 블록'): 제목은 한 줄이라 문제없고, 크레딧처럼
        # '작사 …' / '노래 …' 두 줄이 붙어 있는 밴드는 psm 7(한 줄 전용)로
        # 하면 실패한다(실측: 빈 결과) — 6은 둘 다 처리한다.
        txt = pytesseract.image_to_string(th, lang="kor", config="--psm 6").strip()
        data = pytesseract.image_to_data(th, lang="kor", config="--psm 6",
                                         output_type=pytesseract.Output.DICT)
        # conf는 글자를 못 찾은 자리에 -1(정수)이 섞여 나온다 — 이걸 평균에
        # 넣으면 진짜 인식된 글자의 신뢰도가 크게 저평가된다(실측: 문자열
        # "-1" 비교로 걸러내려 했으나 실제 값은 정수라 안 걸러지던 버그).
        confs = [float(c) for c in data.get("conf", []) if float(c) >= 0]
        conf = sum(confs) / len(confs) if confs else 0.0
        # 폰트 크기는 '밴드 전체 픽셀 높이'가 아니라 '실제 인식된 단어의
        # 높이 평균'으로 추정한다 — 크레딧처럼 두 줄이 한 밴드로 뭉치면
        # 밴드 전체 높이가 오히려 제목(한 줄)보다 커져 큰 글씨로 착각한다
        # (실측: 이 버그로 제목↔크레딧이 뒤바뀜).
        hs = [float(h) for h, c in zip(data.get("height", []), data.get("conf", []))
              if float(c) >= 0]
        font_h = (sum(hs) / len(hs) / 3.0) if hs else 0.0   # 3배 확대 되돌림
    except Exception:
        return "", 0.0, 0.0
    return txt, conf, font_h


def extract_title_credits(ctx):
    """ctx → {"title": str|None, "credits": str|None}. 실패해도 예외 없음."""
    out = {"title": None, "credits": None}
    if not tesseract_available():
        return out
    eng = ctx.get("engine_result")
    original = ctx.get("image_0")
    if eng is None or not eng.systems or original is None:
        return out
    try:
        ch, cw = ctx["image_clean"].shape[:2]
        oh, ow = original.shape[:2]
        sx, sy = ow / cw, oh / ch
        s0 = eng.systems[0]
        y1 = int(s0.top * sy)
        if y1 < 20:
            return out
        crop = original[0:y1, :]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bands = _find_bands(gray)
        results = []       # [(font_h, text)]
        for (by0, by1) in bands:
            txt, conf, font_h = _ocr_band(crop[by0:by1, :])
            hangul = _HANGUL_RE.findall(txt)
            if not txt or conf < 40:
                continue
            ratio = len(hangul) / max(1, len(txt.replace(" ", "").replace("\n", "")))
            if ratio < MIN_HANGUL_RATIO:
                continue       # 코드 기호 등 한글 아닌 잡음 밴드
            results.append((font_h, txt))
        if not results:
            return out
        results.sort(key=lambda t: -t[0])
        out["title"] = results[0][1]
        rest = [t for _, t in results[1:]]
        if rest:
            out["credits"] = " · ".join(t.replace("\n", " ") for t in rest)
    except Exception:
        pass
    return out
