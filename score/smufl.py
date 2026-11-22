# score/smufl.py
# ---------------------------------------------------------------------------
# SMuFL(Standard Music Font Layout) 글리프 지원.
#
# Bravura 등 SMuFL 표준 음악 폰트가 src/fonts/ 에 있으면 음표 머리·쉼표·
# 음자리표·조표·박자표 등을 표준 유니코드 글리프로 '텍스트처럼' 그린다.
# 폰트가 없으면 font_available()==False 를 돌려주고, render 는 기존 벡터
# 드로잉으로 자동 폴백한다(프로그램이 깨지지 않는다).
#
# 폰트 파일 준비 방법(둘 중 하나를 src/fonts/ 에 넣으면 자동 인식):
#   · Bravura.otf   — https://github.com/steinbergmedia/bravura (SIL OFL, 상업배포 자유)
#   · Leland.otf    — MuseScore 4 기본 폰트(역시 OFL)
# ---------------------------------------------------------------------------
from __future__ import annotations

import os
import sys
from functools import lru_cache
from PIL import ImageFont


def _base_dir():
    # PyInstaller 로 동결(onefile) 실행 시엔 임시 추출 폴더(_MEIPASS)에 자료가 있다.
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


_FONT_DIR = os.path.join(_base_dir(), "src", "fonts")
_CANDIDATES = ["Bravura.otf", "Bravura.ttf", "Leland.otf", "Petaluma.otf"]

# SMuFL 표준 코드포인트(https://w3c.github.io/smufl/latest/tables/)
GLYPHS = {
    # 음자리표
    "gClef": "\uE050",            # 높은음자리표
    "fClef": "\uE062",            # 낮은음자리표
    "cClef": "\uE05C",
    # 음표 머리
    "noteheadBlack": "\uE0A4",    # 채운 머리(4분 이하)
    "noteheadHalf": "\uE0A3",     # 빈 머리(2분)
    "noteheadWhole": "\uE0A2",    # 온음표 머리
    # 개별 꼬리(깃발) — 위/아래
    "flag8thUp": "\uE240", "flag8thDown": "\uE241",
    "flag16thUp": "\uE242", "flag16thDown": "\uE243",
    "flag32ndUp": "\uE244", "flag32ndDown": "\uE245",
    # 쉼표
    "restWhole": "\uE4E3", "restHalf": "\uE4E4", "restQuarter": "\uE4E5",
    "rest8th": "\uE4E6", "rest16th": "\uE4E7", "rest32nd": "\uE4E8",
    # 조표
    "accidentalFlat": "\uE260", "accidentalNatural": "\uE261",
    "accidentalSharp": "\uE262",
    # 박자표 숫자(timeSig0..9 = E080..E089)
    "timeSigCommon": "\uE08A",
    "timeSigCutCommon": "\uE08B",            # [v3.40] ₵(2/2)
    "timeSig": {str(i): chr(0xE080 + i) for i in range(10)},
    # 부점
    "augmentationDot": "\uE1E7",
    # 브레이스(중괄호) — 그랜드 스태프(피아노 등) 좌측에 여러 보표를 묶는 기호.
    # [실사용 검증 후 추가] 손으로 그린 근사 곡선이 부등호(<)처럼 보이는
    # 문제가 있어, Bravura 폰트에 이미 있는 정식 SMuFL brace 글리프
    # (U+E000)를 쓰도록 확인해 등록함.
    "brace": "\uE000",
}

# 폰트 em(=1 staff space 기준)에 대한 글리프 배치 보정.
# SMuFL 좌표계에서 em = 4 staff-space(한 보표 높이)이므로, 오선 간격(ss)
# 기준 폰트 크기 = ss * 4. 그리기 기준선(anchor)은 PIL 로 별도 정렬한다.
EM_IN_STAFF_SPACES = 4.0


@lru_cache(maxsize=1)
def _font_path():
    for name in _CANDIDATES:
        p = os.path.normpath(os.path.join(_FONT_DIR, name))
        if os.path.exists(p):
            return p
    return None


def font_available() -> bool:
    return _font_path() is not None


@lru_cache(maxsize=64)
def get_font(px: int) -> ImageFont.FreeTypeFont | None:
    path = _font_path()
    if not path:
        return None
    try:
        return ImageFont.truetype(path, max(6, int(px)))
    except Exception:
        return None


def size_for_staff_space(ss: float) -> int:
    """오선 간격 ss(렌더 좌표)에 맞는 SMuFL 폰트 픽셀 크기."""
    return int(round(ss * EM_IN_STAFF_SPACES))
