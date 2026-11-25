# editor_app_qt.py
# ---------------------------------------------------------------------------
# 악보 편집기 — PySide6 "1b(소프트 패널)" 디자인 + 기존 인식/편집 로직 통합판.
#
#   * UI 셸    : score_editor_ui.py 의 디자인 스펙(카드 패널 / 캡슐 툴바 /
#                칩 버튼 그룹 / 세그먼트 빔토글 / 옅은 틴트 + 경고 배지)
#   * 로직     : pipeline(인식) → score_model → score_layout → score_render(PIL)
#                편집 = score_editing, 내보내기 = musicxml_export
#
# 디자인 사양 반영:
#   - 배경 #f3f3f5 위 흰색 카드 패널, 포인트 컬러 #2F6FED
#   - 좌(원본) / 우(편집 악보)가 항상 동일 배율로 확대·축소·패닝 (스크롤바 없음,
#     드래그 팬 + 휠 줌, 커서 기준 줌, 양방향 동기화)
#   - 음자리표·조표·박자표를 원본 이미지에서 잘라 붙이지 않고 벡터로 통일
#     (score_render 의 자체 벡터 그리기 사용 → header_strips 미사용)
#   - 마디 경고: 옅은 틴트(score_render) + 작은 원형 "!" 배지(오버레이)
#   - 하단 편집 바: 좌 "선택 요소" 정보 / 우 버튼 그룹(음높이·박자·빔그룹·동작)
#   - 상단 툴바 캡슐은 중앙 정렬, 통계 칩은 우측 끝
#
# 실행:  pip install PySide6 opencv-python numpy pillow
#        python editor_app_qt.py
# ---------------------------------------------------------------------------
from __future__ import annotations

import os
import re
import sys
import threading
import time
from pathlib import Path
import copy
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QTimer, QStandardPaths, QSize, QObject, QEvent
from PySide6.QtGui import (
    QPainter, QPen, QColor, QFont, QPixmap, QImage,
    QWheelEvent, QMouseEvent, QKeySequence, QShortcut, QIcon, QAction,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QPushButton, QToolButton, QButtonGroup, QFileDialog, QMessageBox, QInputDialog,
    QDialog, QComboBox, QRadioButton, QDialogButtonBox,
    QLineEdit, QScrollArea, QSizePolicy, QTextEdit, QMenu, QAbstractButton,
)

from omr import modules  # 볼타/하프핀 등 보조 검출기는 당분간 유지(2단계 범위 밖)
from omr_bridge import document_adapter as document  # [2단계] 인식 엔진을 Audiveris로 교체
from score import model as score_model
from score import midi as score_midi
from score import editing as ed
from score import layout as sl
from score import render as sr
from score import musicxml as mx

# 프로그램 이름·버전(한 곳에서만 관리). 버전 규칙: 큰 기능 = 정수 자리, 작은 기능 = 소수 첫째 자리,
# 오류 수정 = 소수 둘째 자리 (예: 1.00 → 1.10 → 1.11 → 2.00)
APP_NAME = "콩나물지기"
APP_ID = "OMR_Sprouty"
APP_VERSION = "4.10"
_USER_MSG = "\x00USER\x00"     # [v1.21] 안내 문구만 보여줄 오류 표시(코드 추적 대신)

# [화면 정리 — 실사용 요청] 편집 악보의 음표별 박수·마디 합계 숫자는 가사와 겹치기도 하고,
# 인식이 안정된 지금은 마디 상태(색·느낌표 배지)만으로 충분해 기본으로 숨긴다.
SHOW_NOTE_BEATS = False
SHOW_MEASURE_BEATS = False

DEFAULT_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "resource_dist", "demo_song_hires.png")
# [5단계 중요 수정] 개발용 resource/ 폴더에는 546개 회귀 테스트 샘플과
# 선린교가(저작권 있는 저작물, 계획서 13장)가 함께 들어있어, 그 폴더를
# 통째로 배포 패키징하면 저작권 문제가 있는 자료가 실제 배포판에 섞여
# 들어가는 사고가 난다(PyInstaller 시험 빌드 중 실제로 발견함). 배포에는
# 데모곡 하나만 들어있는 resource_dist/ 폴더를 따로 두고 이것만 쓴다
# (OMREditor.spec의 datas도 resource_dist만 포함하도록 함께 수정함).
F = 4                                    # 악보 렌더 확대 배수(선명도) — 3→4로
                                          # 상향(실측: 전체 악보를 축소해 볼 때
                                          # 오선·마디선이 흐려 보인다는 지적)
PLAY_BPM = 100                           # [2단계] 재생 기본 템포
_ACC = {0: "", 1: "\u266f", -1: "\u266d"}
_DUR_LABEL = {1: "온", 2: "2", 4: "4", 8: "8", 16: "16", 32: "32"}
_DUR_NAME = {1: "온", 2: "2분", 4: "4분", 8: "8분", 16: "16분", 32: "32분"}


# [음표 모양 버튼 — 실사용 피드백] "박자" 칸의 숫자(온/2/4/8…) 대신 음표 모양을 보여 준다.
# 번들 악보 글꼴(Bravura, SMuFL)의 정식 음표 글리프를 그림으로 그려 아이콘으로 쓴다.
# 선택된 버튼(파란 바탕)에는 흰 음표가 보이도록 켜짐/꺼짐 두 벌을 만든다.
_NOTE_GLYPH = {1: "\uE1D2", 2: "\uE1D3", 4: "\uE1D5", 8: "\uE1D7", 16: "\uE1D9", 32: "\uE1DB"}
_AUG_DOT = "\uE1E7"


def _note_icon(base: int, dotted: bool = False) -> Optional["QIcon"]:
    from score import smufl
    # 기둥 없는 온음표는 같은 크기면 너무 작아 보여 조금 크게 그린다
    fnt = smufl.get_font(80 if base == 1 else 52) if smufl.font_available() else None
    if fnt is None:
        return None
    text = _NOTE_GLYPH.get(base, "")
    icon = QIcon()
    for color, state in (((40, 40, 46), QIcon.Off), ((255, 255, 255), QIcon.On)):
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        dr = ImageDraw.Draw(img)
        l, t, r, b = dr.textbbox((0, 0), text, font=fnt)
        shift = -6 if dotted else 0
        ox, oy = (64 - (r - l)) / 2 - l + shift, (64 - (b - t)) / 2 - t
        dr.text((ox, oy), text, font=fnt, fill=color + (255,))
        if dotted:                    # 점은 글리프 그대로면 너무 작아 잘 보이는 크기로 직접 찍는다
            cx, cy = ox + r + 6, oy + b - 9
            dr.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=color + (255,))
        qimg = QImage(img.tobytes("raw", "RGBA"), 64, 64, QImage.Format_RGBA8888).copy()
        icon.addPixmap(QPixmap.fromImage(qimg), QIcon.Normal, state)
    return icon

def _draw_icon(w: int, h: int, paint) -> "QIcon":
    """[v1.41 편집 막대 아이콘] paint(draw, color) 로 그린 꺼짐/켜짐(흰색) 두 상태 아이콘."""
    icon = QIcon()
    for color, state in (((40, 40, 46), QIcon.Off), ((255, 255, 255), QIcon.On)):
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        paint(ImageDraw.Draw(img), color + (255,))
        qimg = QImage(img.tobytes("raw", "RGBA"), w, h, QImage.Format_RGBA8888).copy()
        icon.addPixmap(QPixmap.fromImage(qimg), QIcon.Normal, state)
    return icon


def _dot_icon():
    """점 버튼: 점(.) 하나만."""
    return _draw_icon(64, 64, lambda d, c: d.ellipse([24, 24, 40, 40], fill=c))


def _hairpin_icon(kind: str):
    """크레셴도(<)·디크레셴도(>) 버튼: 가로로 긴 부등호."""
    def paint(d, c):
        if kind == "cresc":
            d.line([(6, 16), (90, 4)], fill=c, width=4)
            d.line([(6, 16), (90, 28)], fill=c, width=4)
        else:
            d.line([(6, 4), (90, 16)], fill=c, width=4)
            d.line([(6, 28), (90, 16)], fill=c, width=4)
    return _draw_icon(96, 32, paint)


def _smufl_icon(ch, size=56):
    """[v2.20] SMuFL(Bravura) 글리프 하나로 만든 아이콘."""
    from score import smufl
    fnt = smufl.get_font(size) if smufl.font_available() else None
    if fnt is None:
        return None

    def paint(d, c):
        l, t, r, b = d.textbbox((0, 0), ch, font=fnt)
        d.text(((64 - (r - l)) / 2 - l, (64 - (b - t)) / 2 - t), ch, font=fnt, fill=c)
    return _draw_icon(64, 64, paint)


def _glyph_icon(parts, w=96, h=64):
    """[v2.21 동작 칸 아이콘] parts = [(글리프 또는 그리기 함수, x, y, 크기)] 를 한 아이콘에."""
    from score import smufl
    if not smufl.font_available():
        return None

    def paint(d, c):
        for item, x, y, size in parts:
            if callable(item):
                item(d, c)
                continue
            fnt = smufl.get_font(size)
            d.text((x, y), item, font=fnt, fill=c, anchor="ms")
    return _draw_icon(w, h, paint)


def _note_rest_icon():
    """4분음표 ⇄ 4분쉼표: 쉼표를 음표와 같은 높이·같은 세로 위치로(v2.24). 가운데 화살표 — 위는 오른쪽,
    아래는 왼쪽(촉은 바깥쪽 반만)."""
    from score import smufl
    if not smufl.font_available():
        return None

    def paint(d, c):
        fn = smufl.get_font(50)
        nb = d.textbbox((0, 0), "\uE1D5", font=fn, anchor="ls")
        nh = nb[3] - nb[1]
        top, bot = 6, 6 + nh
        d.text((10 - nb[0], top - nb[1]), "\uE1D5", font=fn, fill=c, anchor="ls")
        # 쉼표: 음표 높이에 맞는 크기를 찾아 같은 위·아래 선에 맞춘다
        size = 50
        for _ in range(6):
            rb = d.textbbox((0, 0), "\uE4E5", font=smufl.get_font(size), anchor="ls")
            size = max(10, int(size * nh / max(1, rb[3] - rb[1])))
        fr = smufl.get_font(size)
        rb = d.textbbox((0, 0), "\uE4E5", font=fr, anchor="ls")
        d.text((84 - (rb[2] - rb[0]) / 2 - rb[0], top - rb[1]), "\uE4E5", font=fr, fill=c, anchor="ls")
        d.line([(38, 26), (58, 26)], fill=c, width=3)
        d.line([(58, 26), (51, 19)], fill=c, width=3)        # 위쪽(바깥) 반촉
        d.line([(38, 38), (58, 38)], fill=c, width=3)
        d.line([(38, 38), (45, 45)], fill=c, width=3)        # 아래쪽(바깥) 반촉
    return _draw_icon(96, 64, paint)


def _insert_icon(before: bool):
    """앞 삽입 = '+ ♩', 뒤 삽입 = '♩ +' — 새 음이 들어갈 쪽에 +."""
    def plus(d, c, cx):
        d.line([(cx - 9, 32), (cx + 9, 32)], fill=c, width=4)
        d.line([(cx, 23), (cx, 41)], fill=c, width=4)
    if before:
        return _glyph_icon([(lambda d, c: plus(d, c, 28), 0, 0, 0), ("\uE1D5", 62, 52, 46)])
    return _glyph_icon([("\uE1D5", 34, 52, 46), (lambda d, c: plus(d, c, 70), 0, 0, 0)])


def _grace_icon():
    """작은 8분음표 + 그 뒤 이음줄."""
    def slur(d, c):
        _crescent(d, c, 22, 52, 50, 50, 6, 5)          # 머리 오른쪽 아래에서 시작(머리에 닿게), 초승달
    return _glyph_icon([("\uE1D7", 24, 46, 32), (slur, 0, 0, 0)], w=72, h=64)


def _crescent(d, c, x0, x1, y_end_l, y_end_r, sag, thick):
    """초승달 모양 이음줄: x0→x1, 양 끝 높이(기울기), 가운데로 처지는 깊이(sag), 가운데 두께(thick)."""
    import math
    n = 30
    outer, inner = [], []
    for i in range(n + 1):
        t = i / n
        x = x0 + (x1 - x0) * t
        base = y_end_l + (y_end_r - y_end_l) * t
        bend = math.sin(math.pi * t)
        outer.append((x, base + sag * bend + thick * bend))
        inner.append((x, base + sag * bend))
    d.polygon(outer + inner[::-1], fill=c)


def _stems_beams_icon(n_beams=2, tuplet=False):
    """[v2.23] 빔 버튼: 기둥 2 + 빔 2(16분 두 개). 잇단 버튼: 기둥 3 + 빔 1 + 위에 작은 '3' — 머리 없이."""
    from score import smufl
    if not tuplet:
        def paint(d, c):
            # 실제 음표처럼: 빔은 오른쪽으로 살짝 올라가고, 두 기둥 길이가 달라(음높이가 다름) 상자처럼 안 보이게
            yl, yr = 14, 6
            d.line([(8, yl), (8, 62)], fill=c, width=6)
            d.line([(56, yr), (56, 50)], fill=c, width=6)
            for k in range(n_beams):
                o = k * 15
                d.polygon([(5, yl + o), (59, yr + o), (59, yr + o + 9), (5, yl + o + 9)], fill=c)
        return _draw_icon(64, 64, paint)

    def paint_t(d, c):
        # 살짝 기운 빔(왼쪽 낮고 오른쪽 높게) + 기둥 3, 숫자 3은 빔 위에 온전히 보이게
        xs, y_l, y_r = (8, 40, 72), 34, 26
        for x in xs:
            yb = y_l + (y_r - y_l) * (x - xs[0]) / (xs[-1] - xs[0])
            d.line([(x, yb), (x, 63)], fill=c, width=5)
        d.polygon([(6, y_l), (74, y_r), (74, y_r + 8), (6, y_l + 8)], fill=c)
        if smufl.font_available():
            fnt = smufl.get_font(56)
            d.text((40, 24), "\uE883", font=fnt, fill=c, anchor="ms")
        else:
            d.text((37, 6), "3", fill=c)
    return _draw_icon(80, 64, paint_t)


def _slur_icon():
    """[v2.23] 이음줄: 뒤집은 초승달, 낮은 음 → 높은 음으로 살짝 올라가게(폭은 좁게)."""
    def paint(d, c):
        _crescent(d, c, 6, 58, 30, 18, 14, 7)
    return _draw_icon(64, 64, paint)


def _dynamics_icon():
    """[v2.23] 셈여림: p ~ f (악보 글꼴의 셈여림 글자, 크게)."""
    import math
    from score import smufl
    if not smufl.font_available():
        return None

    def paint(d, c):
        fnt = smufl.get_font(70)
        d.text((18, 50), "\uE520", font=fnt, fill=c, anchor="ms")       # p
        d.text((106, 50), "\uE522", font=fnt, fill=c, anchor="ms")      # f
        pts = [(44 + i, 32 + 5 * math.sin(i / 3.6)) for i in range(0, 37)]
        d.line(pts, fill=c, width=4)
    return _draw_icon(128, 64, paint)


def _accidental_cycle_icon():
    """[v2.23] 임시표 순환: ♯ ♭ ♮ (악보 글꼴, 크게)."""
    return _glyph_icon([("\uE262", 17, 48, 66), ("\uE260", 48, 45, 66), ("\uE261", 79, 48, 66)], w=96, h=64)


def _fermata_icon():
    from score import smufl
    fnt = smufl.get_font(56) if smufl.font_available() else None
    if fnt is None:
        return None

    def paint(d, c):
        l, t, r, b = d.textbbox((0, 0), "\uE4C0", font=fnt)
        d.text(((64 - (r - l)) / 2 - l, (64 - (b - t)) / 2 - t), "\uE4C0", font=fnt, fill=c)
    return _draw_icon(64, 64, paint)


# fifths(오도권) → 장조 이름
_KEY_NAMES = {
    -7: "C♭장조", -6: "G♭장조", -5: "D♭장조", -4: "A♭장조", -3: "E♭장조",
    -2: "B♭장조", -1: "F장조", 0: "C장조", 1: "G장조", 2: "D장조",
    3: "A장조", 4: "E장조", 5: "B장조", 6: "F♯장조", 7: "C♯장조",
}


# ----------------------------------------------------------------------------
# 디자인 토큰
# ----------------------------------------------------------------------------

class Tokens:
    BG = "#f3f3f5"
    CARD = "#ffffff"
    BORDER = "#e6e6e9"
    TEXT = "#1c1c1f"
    TEXT_DIM = "#6b6b74"
    CHIP_BG = "#f6f6f8"
    ACCENT = "#2F6FED"
    ACCENT_SOFT = "rgba(47,111,237,0.08)"
    OK_GREEN = "#1a8a4b"
    WARN_RED = "#E5484D"
    RADIUS = 10


# ----------------------------------------------------------------------------
# 이미지 변환 헬퍼
# ----------------------------------------------------------------------------

def cv_bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
    rgb = pil_img.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    qimg = QImage(data, rgb.width, rgb.height, 3 * rgb.width, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ----------------------------------------------------------------------------
# 공용: 확대/축소/패닝 동기화 뷰포트
#   - 두 패널 모두 '원본 이미지 좌표계(world)'로 그린다.
#     screen = world * zoom + pan
#   - 우측(렌더 F배 이미지)은 pixmap 을 1/F 로 축소해 그려 좌표계를 맞춘다.
# ----------------------------------------------------------------------------

def _min_pool2(a):
    """2×2 블록마다 가장 진한(작은) 값을 남겨 절반 크기로 줄인다. 일반 축소(평균·샘플링)는
    가는 오선·기둥이 흐려지거나 빠지는데, 이렇게 하면 1픽셀 이상의 검은 선으로 남는다."""
    h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    return np.minimum(np.minimum(a[0:h:2, 0:w:2], a[1:h:2, 0:w:2]),
                      np.minimum(a[0:h:2, 1:w:2], a[1:h:2, 1:w:2]))


def _build_min_levels(arr, min_side=400):
    levels, x = [], arr
    while min(x.shape[:2]) // 2 >= min_side // 4 and max(x.shape[:2]) > min_side:
        x = np.ascontiguousarray(_min_pool2(x))
        levels.append(x)
    return levels


# ----------------------------------------------------------------------------
# 공용: 확대/축소/패닝 동기화 뷰포트
# ----------------------------------------------------------------------------

class SyncedViewport(QWidget):
    viewChanged = Signal(float, QPointF)  # zoom, pan(px)
    # [축소 화질 — 실사용 피드백] 화면 맞춤에서는 그림이 0.1배 안팎으로 줄어드는데, Qt의
    # 부드러운 축소는 픽셀 약 10개 중 1개만 골라 봐서 가는 오선이 흐려지거나 끊겼다(선을
    # 굵게 해도 근본 해결이 안 됨). 진한 픽셀을 보존하는 2·4·8·16배 축소본을 뒤에서 만들어,
    # 그릴 때 배율에 맞는 단계를 골라 0.5배 이상으로만 줄인다.
    _pyramidReady = Signal(int, object)   # (세대 번호, [(배수, QImage 데이터 배열)])

    def _pyr_init(self):
        if not hasattr(self, "_pyr_gen"):
            self._pyr_gen = 0
            self._pyr_levels = []
            self._pyr_fmt = QImage.Format_RGB888
            self._pyramidReady.connect(self._on_pyramid_ready)

    def _start_pyramid(self, arr, fmt=QImage.Format_RGB888):
        self._pyr_init()
        self._pyr_gen += 1
        gen = self._pyr_gen
        self._pyr_levels = []             # 이전 내용의 축소본은 즉시 버린다(내용 불일치 방지)
        self._pyr_fmt = fmt

        def work():
            try:
                lv = _build_min_levels(arr)
            except Exception:
                lv = []
            self._pyramidReady.emit(gen, lv)
        threading.Thread(target=work, daemon=True).start()

    def _on_pyramid_ready(self, gen, levels):
        if gen != getattr(self, "_pyr_gen", -1):
            return                        # 그 사이 다시 렌더됐으면 오래된 결과는 버린다
        out = []
        for i, a in enumerate(levels):
            qi = QImage(a.data, a.shape[1], a.shape[0], a.shape[1] * 3, self._pyr_fmt).copy()
            out.append((2 ** (i + 1), qi))
        self._pyr_levels = out
        self.update()

    def _pick_level(self, scale):
        """그림 1픽셀이 화면 몇 픽셀인지(scale)를 보고 쓸 축소 단계를 고른다. 없으면 None."""
        scale *= self.devicePixelRatioF()
        best = None
        for k, qi in getattr(self, "_pyr_levels", []):
            if scale * k <= 1.0:
                best = (k, qi)
        return best


    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.world_size: Optional[tuple[float, float]] = None  # (w, h) 원본 좌표
        self.page_bounds: Optional[tuple[float, float]] = None  # (dy, h) 페이지 잠금
        self._dragging = False
        self._drag_start = QPointF()
        self._pan_start = QPointF()
        self._suppress_emit = False
        self.setMouseTracking(True)
        self.setMinimumHeight(360)

    # ---- 동기화 ----
    def set_view(self, zoom: float, pan: QPointF):
        if self._suppress_emit:
            return
        self.zoom = zoom
        self.pan = QPointF(pan)
        self._clamp_to_page()
        self.update()

    def _emit_view(self):
        self._suppress_emit = True
        self.viewChanged.emit(self.zoom, QPointF(self.pan))
        self._suppress_emit = False

    # ---- fit ----
    def fit_to_view(self):
        self.page_bounds = None    # 전체 문서 보기 — 페이지 잠금 해제
        if not self.world_size:
            return
        ww, wh = self.world_size
        cw, ch = max(1, self.width()), max(1, self.height())
        margin = 16
        z = min((cw - margin) / ww, (ch - margin) / wh)
        z = max(0.02, z)
        self.zoom = z
        self.pan = QPointF((cw - ww * z) / 2, (ch - wh * z) / 2)
        self._emit_view()
        self.update()

    def fit_to_page(self, dy: float, page_h: float):
        """[페이지 네비게이션] 문서 좌표(dy~dy+page_h) 구간 하나만 패널에
        꽉 차게 맞추고, 그 구간 밖으로는 확대·이동이 안 되도록 '잠근다'
        (page_bounds). 사용자 요청: "페이지별로만 뷰가 제한되고 확대·
        패닝도 그 페이지 내에서만 가능하게" — 이후의 휠 확대·드래그
        패닝은 모두 _clamp_to_page() 를 거쳐 이 범위를 못 벗어난다.
        """
        self.page_bounds = (dy, max(1.0, page_h))
        if not self.world_size:
            return
        ww, _wh = self.world_size
        cw, ch = max(1, self.width()), max(1, self.height())
        margin = 16
        z = min((cw - margin) / ww, (ch - margin) / max(1.0, page_h))
        z = max(0.02, z)
        self.zoom = z
        self.pan = QPointF((cw - ww * z) / 2, (ch - page_h * z) / 2 - dy * z)
        self._emit_view()
        self.update()

    def _clamp_to_page(self):
        """page_bounds 가 설정돼 있으면 zoom/pan 이 그 페이지 범위를 벗어나지
        않도록 강제한다 — 축소는 '페이지 전체가 보이는 배율'까지만
        허용하고(그 아래로 축소하면 옆/다른 페이지가 보이게 되므로 막음),
        패닝은 페이지의 위/아래 경계에서 더 못 가게 막는다. 가로는 원래도
        전체 문서 폭이 페이지 폭과 같으므로(페이지마다 폭 동일) 일반적인
        가로 스크롤 제한과 동일하게 처리한다.
        """
        if not self.page_bounds or not self.world_size:
            return
        dy, ph = self.page_bounds
        ww, _wh = self.world_size
        cw, ch = max(1, self.width()), max(1, self.height())
        margin = 16
        min_zoom = max(0.02, min((cw - margin) / ww, (ch - margin) / ph))
        if self.zoom < min_zoom:
            self.zoom = min_zoom
        z = self.zoom
        if ww * z <= cw:
            self.pan.setX((cw - ww * z) / 2)
        else:
            self.pan.setX(min(0.0, max(cw - ww * z, self.pan.x())))
        if ph * z <= ch:
            self.pan.setY((ch - ph * z) / 2 - dy * z)
        else:
            y_max = -dy * z
            y_min = ch - (dy + ph) * z
            self.pan.setY(min(y_max, max(y_min, self.pan.y())))

    # ---- 입력 ----
    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.12 if delta > 0 else (1 / 1.12)
        new_zoom = max(0.02, min(16.0, self.zoom * factor))
        # 커서 아래 world 지점 고정
        pos = event.position()
        wx = (pos.x() - self.pan.x()) / self.zoom
        wy = (pos.y() - self.pan.y()) / self.zoom
        self.zoom = new_zoom
        self.pan = QPointF(pos.x() - wx * new_zoom, pos.y() - wy * new_zoom)
        self._clamp_to_page()
        self._emit_view()
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        # [좌측 패널 우클릭 패닝 추가 — 실사용 검증 후] ScorePanel(편집
        # 악보 패널)은 이 메서드를 완전히 오버라이드해서 우클릭 전용
        # 패닝 + 좌클릭 드래그 다중선택을 쓰므로 아래 로직은 적용되지
        # 않는다. OriginalImagePanel(원본 이미지 패널)은 다중선택이
        # 필요 없어 이 메서드를 오버라이드하지 않고 그대로 쓰는데,
        # 좌클릭 패닝만 있고 우클릭 패닝이 없다는 지적을 받아 양쪽
        # 버튼 다 패닝되게 한다.
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self._dragging = True
            self._drag_start = event.position()
            self._pan_start = QPointF(self.pan)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            delta = event.position() - self._drag_start
            self.pan = self._pan_start + delta
            self._clamp_to_page()
            self._emit_view()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._dragging = False

    # ---- 좌표 ----
    def widget_to_world(self, pos: QPointF) -> tuple[float, float]:
        return ((pos.x() - self.pan.x()) / self.zoom,
                (pos.y() - self.pan.y()) / self.zoom)

    def world_to_widget(self, wx: float, wy: float) -> tuple[float, float]:
        return (wx * self.zoom + self.pan.x(), wy * self.zoom + self.pan.y())

    # ---- 배경/플레이스홀더 ----
    def _paint_placeholder(self, painter: QPainter, text: str):
        rect = self.rect()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#f4f4f6"))
        painter.drawRect(rect)
        painter.setPen(QPen(QColor("#eceef0"), 10))
        step = 20
        for x in range(-rect.height(), rect.width(), step):
            painter.drawLine(x, rect.height(), x + rect.height(), 0)
        painter.setPen(QColor("#9a9aa2"))
        painter.setFont(QFont("Consolas" if sys.platform.startswith("win") else "Menlo", 11))
        painter.drawText(rect, Qt.AlignCenter, text)


# ----------------------------------------------------------------------------
# 좌측: 원본 이미지 패널 (+ 선택 요소 빨간 하이라이트)
# ----------------------------------------------------------------------------

class OriginalImagePanel(SyncedViewport):
    pickRequested = Signal(float, float)   # [v2.40] 원본에서 기호 넣기 — 두 번 클릭한 자리(곡 좌표)
    brushDone = Signal(list, float)        # [v2.60] 붓질을 마침 — (붓 자국 점 목록들, 반지름) 곡 좌표
    candidateChosen = Signal(str)          # [v2.60] 후보 고름(내부 키)
    markClicked = Signal(float, float, float, float)   # [v4.10] 주황 점선(붓 후보 자리)을 클릭 — 상자(곡 좌표)
    brushToggled = Signal(bool)

    # ---------------- [v2.60] 붓으로 기호 넣기 ----------------
    def _init_brush(self):
        self._brush_on = False
        self._brush_r = 9.0                  # 붓 반지름(곡 좌표 px) — Ctrl+휠로 조절
        self._strokes = []
        self._painting = False
        self._hover = None
        self.setMouseTracking(True)
        self.brush_btn = QToolButton(self)
        self.brush_btn.setCheckable(True)
        self.brush_btn.setText("🖌")
        self.brush_btn.setToolTip("붓으로 기호 넣기 (B) — 원본에서 빠진 기호를 칠하면 후보를 보여 줍니다\n"
                                  "주황 점선(읽히지 않은 잉크)을 클릭하면 그 자리를 바로 읽습니다\n"
                                  "Ctrl+휠: 붓 크기 · Esc: 칠한 것 지우고 붓 끄기 · 오른쪽 끌기: 화면 이동")
        self.brush_btn.setFixedSize(38, 38)
        self.brush_btn.setStyleSheet(
            "QToolButton { font-size:20px; background:rgba(255,255,255,230); border:1px solid #c9ced6;"
            " border-radius:8px; } QToolButton:checked { background:#ffd6e8; border:2px solid #e0559b; }")
        self.brush_btn.move(10, 34)
        self.brush_btn.toggled.connect(self.set_brush)
        self.cand_bar = QFrame(self)
        self.cand_bar.setStyleSheet("QFrame { background:rgba(255,255,255,240); border:1px solid #c9ced6;"
                                    " border-radius:8px; }")
        self._cand_lay = QHBoxLayout(self.cand_bar)
        self._cand_lay.setContentsMargins(6, 4, 6, 4)
        self._cand_lay.setSpacing(4)
        self.cand_bar.move(54, 34)
        self.cand_bar.hide()

    def set_brush(self, on: bool):
        self._brush_on = bool(on)
        if self.brush_btn.isChecked() != self._brush_on:
            self.brush_btn.setChecked(self._brush_on)
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)
        if not on:
            self.clear_brush()
        self.brushToggled.emit(self._brush_on)
        self.update()

    def clear_brush(self):
        self._strokes = []
        self._painting = False
        self.show_candidates([])
        self.update()

    def show_candidates(self, items):
        """items: [(key, label, QIcon or None, best: bool)]"""
        while self._cand_lay.count():
            w_ = self._cand_lay.takeAt(0).widget()
            if w_ is not None:
                w_.deleteLater()
        if not items:
            self.cand_bar.hide()
            return
        for key, label, icon, best in items:
            b = QToolButton(self.cand_bar)
            b.setText(label)
            if icon is not None:
                b.setIcon(icon)
                b.setIconSize(QSize(22, 22))
                b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            b.setStyleSheet("QToolButton { padding:4px 8px; border-radius:6px; border:1px solid %s; "
                            "background:%s; font-size:12px; }" % (("#3d7ee8", "#e8f0fe") if best else
                                                                    ("#d8dce3", "white")))
            b.clicked.connect(lambda _=False, k=key: self.candidateChosen.emit(k))
            self._cand_lay.addWidget(b)
        self.cand_bar.adjustSize()
        self.cand_bar.show()
        self.cand_bar.raise_()

    def _mark_at(self, pos):
        """[v4.10] 화면 좌표 pos가 들어 있는 주황 점선 상자(곡 좌표) — 없으면 None. 겹치면 가장 작은 것."""
        best = None
        for x, y, w, h in getattr(self, "_ink_marks", None) or []:
            x0, y0 = self.world_to_widget(x, y)
            x1, y1 = self.world_to_widget(x + w, y + h)
            if x0 - 6 <= pos.x() <= x1 + 6 and y0 - 6 <= pos.y() <= y1 + 6:
                if best is None or w * h < best[2] * best[3]:
                    best = (x, y, w, h)
        return best

    def fill_box_strokes(self, x, y, w, h):
        """[v4.10] 상자를 붓으로 덮어 칠한 것과 같은 붓 자국을 만든다(가로 줄 여러 개). 반환 (자국들, 반지름)."""
        r = max(3.0, min(self._brush_r, max(w, h) / 2.0 + 2.0))
        pad = 3.0
        ys_, yy = [], y - pad + r
        while yy < y + h + pad - r:
            ys_.append(yy)
            yy += r
        ys_.append(max(y - pad + r, y + h + pad - r))
        if h + 2 * pad <= 2 * r:
            ys_ = [y + h / 2.0]
        return [[(x - pad, v), (x + w + pad, v)] for v in ys_], r

    def mousePressEvent(self, event):
        self._press_pos = event.position() if event.button() == Qt.LeftButton else None
        if getattr(self, "_brush_on", False) and event.button() == Qt.LeftButton:
            self._painting = True
            self.show_candidates([])
            wx, wy = self.widget_to_world(event.position())
            self._strokes.append([(wx, wy)])
            self.update()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if getattr(self, "_brush_on", False):
            self._hover = event.position()
            if self._painting and self._strokes:
                wx, wy = self.widget_to_world(event.position())
                self._strokes[-1].append((wx, wy))
            self.update()
            if self._painting:
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # [v4.10] 끌지 않고 주황 점선 안을 클릭 → 그 상자를 붓으로 칠한 것으로(붓이 꺼져 있어도 켜진다)
        pp = getattr(self, "_press_pos", None)
        clicked = (event.button() == Qt.LeftButton and pp is not None
                   and (event.position() - pp).manhattanLength() < 5)
        if clicked:
            mk = self._mark_at(event.position())
            if mk is not None:
                self._painting = False
                self._dragging = False
                if getattr(self, "_brush_on", False) and self._strokes:
                    self._strokes.pop()                # 방금 찍힌 점 하나는 버린다
                self.markClicked.emit(*mk)
                return
        if getattr(self, "_brush_on", False) and event.button() == Qt.LeftButton and self._painting:
            self._painting = False
            self.brushDone.emit([list(s) for s in self._strokes], float(self._brush_r))
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if getattr(self, "_brush_on", False) and (event.modifiers() & Qt.ControlModifier):
            d = event.angleDelta().y()
            self._brush_r = max(2.0, min(80.0, self._brush_r * (1.15 if d > 0 else 1 / 1.15)))
            self.update()
            return
        super().wheelEvent(event)

    def _paint_brush(self, painter):
        if not getattr(self, "_brush_on", False):
            return
        col = QColor(255, 105, 180, 95)
        pen = QPen(col, max(1.0, 2 * self._brush_r * self.zoom), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        for s_ in self._strokes:
            pts = [QPointF(*self.world_to_widget(x, y)) for x, y in s_]
            if len(pts) == 1:
                painter.drawPoint(pts[0])
            else:
                painter.drawPolyline(pts)
        if self._hover is not None:
            painter.setPen(QPen(QColor(224, 85, 155), 1.5))
            r = self._brush_r * self.zoom
            painter.drawEllipse(self._hover, r, r)

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            wx, wy = self.widget_to_world(ev.position())
            self.pickRequested.emit(wx, wy)
            ev.accept()
            return
        super().mouseDoubleClickEvent(ev)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._highlight = None  # (x, y, w, h) 원본 좌표
        self._highlight_multi = []   # [3-B] 다중 선택된 나머지 요소들의 bbox
        self._init_brush()

    def set_image(self, pixmap: QPixmap, arr=None):
        self._pixmap = pixmap
        if arr is not None:                       # BGR 원본 배열 → 축소 단계 만들기
            self._start_pyramid(np.ascontiguousarray(arr), QImage.Format_BGR888)
        self.world_size = (pixmap.width(), pixmap.height())
        self.update()

    def set_highlight(self, bbox):
        self._highlight = bbox
        self.update()

    def set_highlight_multi(self, bboxes):
        self._highlight_multi = bboxes or []
        self.update()

    def set_ink_marks(self, bboxes):
        """[v1.40] 인식되지 않은 잉크 자리(주황 점선)."""
        self._ink_marks = bboxes or []
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, self.zoom < 1.5)
        if self._pixmap is None:
            self._paint_placeholder(painter, "scanned score image")
            return
        painter.save()
        painter.translate(self.pan)
        painter.scale(self.zoom, self.zoom)
        if self.page_bounds:
            # 페이지 잠금 중엔 그 세로 범위 밖을 그리지 않는다 — 클램프만
            # 해서는 배율이 낮아 페이지 전체가 뷰포트보다 작을 때 옆
            # 페이지 내용이 위/아래로 살짝 비쳐 보일 수 있다(실측). 렌더
            # 좌표계에서 그대로 클립하면 줌·팬 값과 무관하게 항상 정확히
            # 그 페이지만 보인다.
            dy, ph = self.page_bounds
            ww, _wh = self.world_size or (self._pixmap.width(), self._pixmap.height())
            painter.setClipRect(QRectF(0, dy, ww, ph))
        painter.translate(0, getattr(self, "img_dy", 0.0))     # [v2.00] 이 페이지 그림의 곡 좌표 위치
        lvl = self._pick_level(self.zoom)
        if lvl is not None:
            k, qi = lvl
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.scale(k, k)
            painter.drawImage(0, 0, qi)
        else:
            painter.drawPixmap(0, 0, self._pixmap)
        painter.restore()
        # 하이라이트(선폭이 줌과 무관하도록 화면 좌표에서 그림)
        if getattr(self, "_ink_marks", None):
            pen = QPen(QColor("#e67e22"), 2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            for x, y, w, h in self._ink_marks:
                x0, y0 = self.world_to_widget(x, y)
                x1, y1 = self.world_to_widget(x + w, y + h)
                painter.drawRect(QRectF(x0 - 5, y0 - 5, (x1 - x0) + 10, (y1 - y0) + 10))
        if self._highlight_multi:
            painter.setPen(QPen(QColor("#3d7ee8"), 2))
            painter.setBrush(Qt.NoBrush)
            for x, y, w, h in self._highlight_multi:
                x0, y0 = self.world_to_widget(x, y)
                x1, y1 = self.world_to_widget(x + w, y + h)
                painter.drawRect(QRectF(x0 - 3, y0 - 3, (x1 - x0) + 6, (y1 - y0) + 6))
        if self._highlight:
            x, y, w, h = self._highlight
            x0, y0 = self.world_to_widget(x, y)
            x1, y1 = self.world_to_widget(x + w, y + h)
            painter.setPen(QPen(QColor("#ff2d2d"), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(x0 - 3, y0 - 3, (x1 - x0) + 6, (y1 - y0) + 6))
        self._paint_brush(painter)                 # [v2.60] 붓 자국·붓 크기


# ----------------------------------------------------------------------------
# 우측: 편집 악보 패널
#   - score_render 가 만든 F배 PIL 렌더를 1/F 로 그려 좌표계를 원본과 통일
#   - 클릭 → layout.hit_test(원본 좌표) → 선택
#   - 오버레이: 선택 사각형, 마디 합계 라벨, 경고 배지, (확대 시) 요소별 길이
# ----------------------------------------------------------------------------

class ScorePanel(SyncedViewport):
    elementPicked = Signal(object, bool)  # (Glyph 또는 None, Ctrl 눌림 여부)
    contextRequested = Signal(object)      # [v2.41] 오른쪽 버튼 메뉴(화면 좌표 QPoint)

    def contextMenuEvent(self, ev):
        # [v2.51] 오른쪽 버튼은 끌기(화면 옮기기)에도 쓴다 — 5px 넘게 끌었으면 메뉴를 띄우지 않는다
        # (Windows는 버튼을 놓을 때 메뉴 이벤트가 와서, 옮긴 뒤마다 메뉴가 떴다)
        if getattr(self, "_rb_moved", False):
            self._rb_moved = False
            ev.accept()
            return
        # [v2.52] 고른 음표 위에서 누를 때만 메뉴(그 밖의 오른쪽 버튼은 화면 끌기 전용)
        lay = getattr(self, "layout_obj", None)
        if lay is not None:
            wx, wy = self.widget_to_world(QPointF(ev.pos()))
            g = lay.hit_test(wx, wy)
            if g is not None:
                self.contextRequested.emit((ev.globalPos(), g))
        ev.accept()
    marqueeSelected = Signal(list)         # [드래그 다중선택] 사각형 안의 Glyph 목록

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QImage] = None   # [v1.31] 렌더 그림(QImage, self._qimg_arr 메모리 공유)
        self._qimg_arr = None
        self.layout_obj = None            # score_layout.Layout
        self.sel_cell = None              # (x0, y0, x1, y1) 원본 좌표
        self.multi_cells = []             # [3-B] 다중 선택된 나머지 요소들의 셀
        self._press_pos = None
        # [드래그 다중선택 + 우클릭 패닝] 이전에는 좌클릭 드래그가 곧
        # 화면 패닝이라, Ctrl+클릭을 여러 번 반복해야만 다중 선택이
        # 가능해 tie/slur/triplet처럼 "두 음 이상을 한 번에 골라야" 하는
        # 편집에 불편하다는 지적을 받았다. 이제 패닝은 우클릭 드래그로
        # 옮기고, 좌클릭 드래그는 사각형 안의 음표를 한 번에 다중
        # 선택하는 용도로 쓴다.
        self._marquee_start = None
        self._marquee_rect = None
        self._panning_rb = False   # 우클릭 패닝 중 여부(부모의 _dragging과 별개)

    def release_render(self):
        """[v1.31 메모리] 다시 그리기 전에 이전 그림·축소본을 먼저 놓아준다(둘이 동시에 메모리에
        있지 않게)."""
        self._pixmap = None
        self._qimg_arr = None
        self._pyr_init()
        self._pyr_gen += 1                    # 진행 중인 축소본 작업 결과도 버리게
        self._pyr_levels = []

    def set_render(self, pil_img: Image.Image, world_w: float, world_h: float,
                   left_margin: float = 0.0, render_f: int = None, img_dy: float = 0.0):
        # [v1.31 메모리 — 실사용 MemoryError] 예전엔 RGB 변환 → tobytes(조립 중 2배) → QImage 복사 →
        # QPixmap으로 큰 그림(교회 1p: 6264×8426, 158MB)을 4~5번 복사해, 다시 그릴 때마다 +790MB가
        # 순간적으로 필요했다. 이제 numpy 배열 하나를 띠 단위로 채우고, 화면(QImage)과 축소본이 그
        # 배열을 그대로 함께 쓴다(복사 없음).
        rgb = pil_img if pil_img.mode == "RGB" else pil_img.convert("RGB")
        w_, h_ = rgb.size
        arr = np.empty((h_, w_, 3), np.uint8)
        step = 512
        for y in range(0, h_, step):
            arr[y:y + step] = np.asarray(rgb.crop((0, y, w_, min(h_, y + step))))
        del rgb
        self._qimg_arr = arr                   # QImage가 이 메모리를 직접 가리키므로 함께 보관
        self._pixmap = QImage(arr.data, w_, h_, 3 * w_, QImage.Format_RGB888)
        self._start_pyramid(arr)
        self._left_margin = left_margin      # 렌더 픽스맵 좌측 헤더 여백(px, F배 좌표)
        self._render_f = render_f or F        # 이 렌더에 실제 쓰인 슈퍼샘플링 배율
        self._img_dy = img_dy                  # [v2.00] 이 그림의 위쪽이 곡 좌표에서 어디인지(페이지 위치)
        self.world_size = (world_w, world_h)
        self.update()

    def set_selection(self, cell):
        self.sel_cell = cell
        self.update()

    def set_multi_selection(self, cells):
        """[3-B] 다중 선택 셀 목록(Ctrl+클릭으로 추가된 것들, 마지막 클릭
        요소 제외). 마지막 클릭 요소는 기존 sel_cell(초록)로 강조되고,
        나머지는 이 목록으로 파란 강조된다 — 둘을 합쳐야 전체 선택이다."""
        self.multi_cells = cells or []
        self.update()

    # 클릭(짧은 클릭)과 드래그(팬/마퀴)를 구분
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.RightButton:
            # [우클릭 패닝] 부모(SyncedViewport)의 좌클릭 패닝 로직을
            # 그대로 재현하되 버튼만 우클릭으로 바꾼다.
            self._panning_rb = True
            self._rb_moved = False                 # [v2.51] 끌어서 옮겼으면 메뉴를 띄우지 않는다
            self._drag_start = event.position()
            self._pan_start = QPointF(self.pan)
        elif event.button() == Qt.LeftButton:
            self._press_pos = event.position()
            self._marquee_start = event.position()
            self._marquee_rect = None

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning_rb:
            delta = event.position() - self._drag_start
            if delta.x() * delta.x() + delta.y() * delta.y() > 25:
                self._rb_moved = True
            self.pan = self._pan_start + delta
            self._clamp_to_page()
            self._emit_view()
            self.update()
        elif self._press_pos is not None:
            d = event.position() - self._press_pos
            if d.x() * d.x() + d.y() * d.y() > 25:
                self._marquee_rect = QRectF(self._marquee_start, event.position()).normalized()
                self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.RightButton:
            self._panning_rb = False
            return
        if event.button() != Qt.LeftButton:
            return
        if self._marquee_rect is not None and self.layout_obj is not None:
            glyphs = self._glyphs_in_rect(self._marquee_rect)
            self._marquee_rect = None
            self._press_pos = None
            self.update()
            self.marqueeSelected.emit(glyphs)
            return
        self._marquee_rect = None
        if self._press_pos is None:
            return
        d = event.position() - self._press_pos
        if d.x() * d.x() + d.y() * d.y() <= 25 and self.layout_obj is not None:
            wx, wy = self.widget_to_world(event.position())
            ctrl = bool(event.modifiers() & Qt.ControlModifier)
            self.elementPicked.emit(self.layout_obj.hit_test(wx, wy), ctrl)
        self._press_pos = None

    def _glyphs_in_rect(self, rect_widget: QRectF):
        """화면(위젯) 좌표 사각형 안에 노트헤드 중심이 들어오는 Glyph들을
        음악적 순서(x좌표 순)로 반환한다."""
        found = []
        for g in self.layout_obj.glyphs:
            wx, wy = self.world_to_widget(g.x, g.y)
            if rect_widget.contains(QPointF(wx, wy)):
                found.append(g)
        found.sort(key=lambda g: g.x)
        return found

    def paintEvent(self, event):
        painter = QPainter(self)
        if self._pixmap is None:
            self._paint_placeholder(painter, "recognized score")
            return
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.save()
        painter.translate(self.pan)
        render_f = getattr(self, "_render_f", F)
        painter.scale(self.zoom / render_f, self.zoom / render_f)   # F배 렌더 → 원본 좌표계
        # 렌더 픽스맵은 좌측에 헤더 여백(left_margin px)만큼 더 넓다. 내용이
        # 그만큼 오른쪽으로 밀려 있으므로 픽스맵을 좌측으로 되돌려 그려,
        # 글리프가 원본 좌표계(좌측 원본 이미지와 동일)에 정렬되게 한다.
        lm = getattr(self, "_left_margin", 0.0)
        if self.page_bounds:
            # 좌측 패널과 같은 이유로 클리핑한다. 다만 여기 좌표계는 F배
            # 렌더 픽셀 단위라(painter.scale 이 render_f 로 나눴으므로),
            # page_bounds(원본 이미지 단위)를 render_f 배 해서 맞춘다.
            dy, ph = self.page_bounds
            painter.setClipRect(QRectF(-lm, dy * render_f,
                                       self._pixmap.width(), ph * render_f))
        painter.translate(0, getattr(self, "_img_dy", 0.0) * render_f)   # [v2.00] 페이지 위치
        lvl = self._pick_level(self.zoom / render_f)
        if lvl is not None:
            k, qi = lvl
            painter.scale(k, k)
            painter.drawImage(int(round(-lm / k)), 0, qi)
        else:
            painter.drawImage(int(-lm), 0, self._pixmap)   # [v1.31] QImage 그대로(복사 없음)
        painter.restore()

        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        self._draw_overlays(painter)

        # 선택 사각형
        if self.multi_cells:
            painter.setPen(QPen(QColor("#3d7ee8"), 2))
            painter.setBrush(Qt.NoBrush)
            for cell in self.multi_cells:
                x0, y0, x1, y1 = cell
                X0, Y0 = self.world_to_widget(x0, y0)
                X1, Y1 = self.world_to_widget(x1, y1)
                painter.drawRect(QRectF(X0, Y0, X1 - X0, Y1 - Y0))
        if self.sel_cell:
            x0, y0, x1, y1 = self.sel_cell
            X0, Y0 = self.world_to_widget(x0, y0)
            X1, Y1 = self.world_to_widget(x1, y1)
            painter.setPen(QPen(QColor("#1d9e75"), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(X0, Y0, X1 - X0, Y1 - Y0))

        # [드래그 다중선택] 진행 중인 사각형(반투명 파란 배경 + 테두리)
        if self._marquee_rect is not None:
            painter.setPen(QPen(QColor("#3d7ee8"), 1.5, Qt.DashLine))
            painter.setBrush(QColor(61, 126, 232, 40))
            painter.drawRect(self._marquee_rect)

    def _draw_overlays(self, painter: QPainter):
        L = self.layout_obj
        if L is None:
            return
        cap = L.header["beats"] * 4.0 / L.header["beat_type"]

        def num(v):
            return f"{v:g}"

        mono = QFont("Consolas" if sys.platform.startswith("win") else "Menlo", 9)
        mono.setBold(True)

        # 요소별 길이(충분히 확대됐을 때만)
        if SHOW_NOTE_BEATS and L.staff_space * self.zoom >= 26:
            painter.setFont(QFont("", 8))
            painter.setPen(QColor("#8a8a8a"))
            # [실사용 검증 후 수정] 화음(chord_group) 구성음은 같은 x좌표에
            # 여러 Glyph가 겹쳐 있어, 모든 멤버에 각각 라벨을 그리면 같은
            # 숫자가 겹쳐 찍혀 "4 4"처럼 중복돼 보이는 문제가 있었다(실사용
            # 검증 중 세밀히 확대해서 발견됨). 화음은 대표(먼저 만난 멤버)
            # 하나만 그린다 — 어차피 화음 구성음은 항상 같은 길이(quarter_
            # length)를 공유하므로 대표 하나로 충분하다.
            seen_chords = set()
            for g in L.glyphs:
                cg = getattr(g.element, "chord_group", -1)
                if cg != -1:
                    if cg in seen_chords:
                        continue
                    seen_chords.add(cg)
                staff_no = getattr(g.element, "staff_no", 1)
                sid = g.measure.system
                sysL = next((sy for sy in L.systems
                            if sy.index == sid and sy.staff_no == staff_no),
                           next(sy for sy in L.systems if sy.index == sid))
                X, Y = self.world_to_widget(g.x, sysL.lines_y[4] + L.staff_space * 1.3)
                painter.drawText(QRectF(X - 20, Y - 8, 40, 16), Qt.AlignCenter,
                                 num(g.element.quarter_length()))

        # 마디별 합계 라벨 + 경고 배지
        # [실사용 검증 후 재설계] 예전에는 마디 하나에 라벨 하나만 그려서
        # "가장 어긋난 성부"의 숫자만 보여줬다 — 다성부(피아노 등)에서
        # 한쪽 보표(예: 오른손)를 고쳐도 다른 쪽 보표(왼손)가 여전히
        # 문제라면 라벨이 안 바뀐 것처럼 보여, "편집이 반영 안 됐다"고
        # 오해하기 쉬웠다(실사용 검증 중 실제로 이 문제로 혼란을 겪음 —
        # 사용자가 오른쪽 화음만 고치고 왼쪽 화음에 같은 문제가 있는 것을
        # 못 보고 지나침). 이제 보표가 여러 개인 마디는 **각 보표 바로
        # 아래에 그 보표 자신의 라벨을 따로** 그려서, 어느 보표가 문제인지
        # 한눈에 보이게 한다.
        painter.setFont(mono)
        systems_by_idx = {}
        for s in L.systems:
            systems_by_idx.setdefault(s.index, []).append(s)

        for mb in L.measure_boxes:
            sys_members = sorted(systems_by_idx[mb.measure.system],
                                 key=lambda s: s.staff_top)
            m_cap = mb.measure.target_capacity if mb.measure.target_capacity is not None \
                else cap
            by_voice = mb.measure.quarter_sum_by_voice(m_cap)
            # [실사용 검증 후 통일] 마디 전체 단위로 먼저 유령 성부 병합/
            # 제거 처리(score/editing.py._real_voice_pool 과 완전히 동일한
            # 기준)를 한 번 하고, 그 결과를 보표별로 나눠서 라벨에 쓴다 —
            # 예전에는 이 파일에 별도로 비슷한 로직을 다시 짜뒀었는데,
            # 그 사본이 "합치기" 개선을 놓쳐 배지(정확한 판정)와 라벨
            # 숫자(옛 무시-전용 로직)가 서로 다른 기준으로 계산되는 사고로
            # 이어졌었다. 이제 한 곳(ed._real_voice_pool)만 쓴다.
            full_pool = ed._real_voice_pool(mb.measure, by_voice, m_cap)

            overall_ok = (mb.status == "ok")

            for sysL in sys_members:
                sn = sysL.staff_no
                voices_here = {k: v for k, v in full_pool.items() if k[0] == sn}
                if not voices_here:
                    continue
                _, total = max(voices_here.items(), key=lambda kv: abs(kv[1] - m_cap))
                ok_here = abs(total - m_cap) < 0.01
                color = QColor(Tokens.OK_GREEN) if ok_here else \
                    QColor(Tokens.WARN_RED) if total > m_cap else QColor(Tokens.ACCENT)

                cx_w = (mb.bbox[0] + mb.bbox[2]) / 2
                # 라벨을 각 보표 바로 아래(보표 간 간격이 좁아도 안 겹치게
                # 1.3칸만 띄움 — 기존엔 항상 2.6칸이었는데 이는 시스템
                # 전체 기준 위치라 다성부에서 각 보표 라벨을 구분해 그리기
                # 어려웠다).
                cy_w = sysL.lines_y[4] + L.staff_space * 1.7
                X, Y = self.world_to_widget(cx_w, cy_w)
                if SHOW_MEASURE_BEATS:
                    painter.setPen(color)
                    painter.drawText(QRectF(X - 40, Y - 8, 80, 16), Qt.AlignCenter,
                                     f"{num(total)}/{num(m_cap)}")

            # 옅은 틴트는 score_render 가 처리 → 여기서는 작은 원형 "!" 배지만
            # (마디 전체를 대표하는 배지 하나 — 위치는 첫 보표 위, 기존과 동일)
            if not overall_ok:
                sysL = sys_members[0]
                bx_w = mb.bbox[2] - L.staff_space * 0.9
                by_w = sysL.lines_y[0] - L.staff_space * 1.6
                BX, BY = self.world_to_widget(bx_w, by_w)
                r = 9
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(Tokens.WARN_RED))
                painter.drawEllipse(QPointF(BX, BY), r, r)
                painter.setPen(QColor("#ffffff"))
                bold = QFont("sans-serif", 9)
                bold.setBold(True)
                painter.setFont(bold)
                painter.drawText(QRectF(BX - r, BY - r, 2 * r, 2 * r), Qt.AlignCenter, "!")
                painter.setFont(mono)


# ----------------------------------------------------------------------------
# 카드/칩 헬퍼 위젯 (디자인 파일과 동일)
# ----------------------------------------------------------------------------

def make_panel_card() -> QFrame:
    card = QFrame()
    card.setObjectName("panelCard")
    card.setStyleSheet(
        f"QFrame#panelCard {{ background:{Tokens.CARD}; border:1px solid {Tokens.BORDER}; "
        f"border-radius:{Tokens.RADIUS}px; }}"
    )
    return card


EDIT_BTN_H = 30            # [v1.41] 하단 편집 막대 버튼 공통 높이


def make_chip_group(title: str, buttons: list[QWidget]) -> QFrame:
    chip = QFrame()
    chip.setObjectName("chip")
    chip.setStyleSheet(f"QFrame#chip {{ background:{Tokens.CHIP_BG}; border-radius:8px; }}")
    v = QVBoxLayout(chip)
    v.setContentsMargins(12, 8, 12, 8)
    v.setSpacing(6)
    label = QLabel(title)
    label.setStyleSheet(f"color:{Tokens.TEXT_DIM}; font-weight:600; font-size:10px; letter-spacing:0.4px;")
    v.addWidget(label)
    row = QHBoxLayout()
    row.setSpacing(6)
    for b in buttons:
        # [v1.41 편집 막대 정리] 칸마다 버튼 높이가 제각각이던 것(음표 아이콘 버튼이 더 높았다)을
        # 모두 같은 높이로 맞춘다.
        if isinstance(b, QAbstractButton):
            b.setFixedHeight(EDIT_BTN_H)
            if not b.icon().isNull():
                b.setIconSize(QSize(min(b.iconSize().width(), 40), EDIT_BTN_H - 8))
        row.addWidget(b, 0, Qt.AlignVCenter)
    v.addLayout(row)
    return chip


def flat_button(text: str, *, danger=False, tip: str = "") -> QPushButton:
    b = QPushButton(text)
    if danger:
        bg, fg = "#fdf1f1", Tokens.WARN_RED
        checked_bg, checked_fg = Tokens.WARN_RED, "#ffffff"
    else:
        bg, fg = Tokens.CARD, Tokens.TEXT
        checked_bg, checked_fg = Tokens.ACCENT, "#ffffff"
    b.setStyleSheet(
        f"QPushButton {{ background:{bg}; color:{fg}; border:none; border-radius:6px; "
        f"padding:6px 10px; font-weight:600; font-size:11px; }}"
        f"QPushButton:hover {{ background:{Tokens.BORDER}; }}"
        f"QPushButton:checked {{ background:{checked_bg}; color:{checked_fg}; }}"
    )
    b.setCursor(Qt.PointingHandCursor)
    if tip:
        b.setToolTip(tip)
    return b


def make_segmented_toggle(title: str, options: list[str],
                          tips: list[str] | None = None) -> tuple[QFrame, QButtonGroup]:
    chip = QFrame()
    chip.setObjectName("chip")
    chip.setStyleSheet(f"QFrame#chip {{ background:{Tokens.CHIP_BG}; border-radius:8px; }}")
    v = QVBoxLayout(chip)
    v.setContentsMargins(12, 8, 12, 8)
    v.setSpacing(6)
    label = QLabel(title)
    label.setStyleSheet(f"color:{Tokens.TEXT_DIM}; font-weight:600; font-size:10px; letter-spacing:0.4px;")
    v.addWidget(label)

    track = QFrame()
    track.setObjectName("track")
    track.setStyleSheet(f"QFrame#track {{ background:{Tokens.CARD}; border-radius:6px; }}")
    track_layout = QHBoxLayout(track)
    track_layout.setContentsMargins(2, 2, 2, 2)
    track_layout.setSpacing(2)

    group = QButtonGroup(chip)
    group.setExclusive(False)          # 편집 도구: 상태 표시는 코드에서 제어
    for idx, opt in enumerate(options):
        btn = QToolButton()
        btn.setText(opt)
        btn.setCheckable(True)
        btn.setCursor(Qt.PointingHandCursor)
        if tips and idx < len(tips):
            btn.setToolTip(tips[idx])
        btn.setStyleSheet(
            "QToolButton { border:none; border-radius:5px; padding:4px 10px; "
            f"font-weight:600; font-size:11px; color:{Tokens.TEXT}; background:transparent; }}"
            f"QToolButton:checked {{ background:{Tokens.ACCENT}; color:white; }}"
        )
        group.addButton(btn, idx)
        track_layout.addWidget(btn)
    v.addWidget(track)
    return chip, group


# ----------------------------------------------------------------------------
# 상단 툴바 (캡슐 중앙 정렬 + 우측 통계 칩)
# ----------------------------------------------------------------------------

class LyricsFillDialog(QDialog):
    """[v3.10] 가사 흘려 넣기: 붙여 넣은 가사를 고른 음부터 차례로."""

    def __init__(self, parent, where: str, rows_used, ocr_fn=None):
        super().__init__(parent)
        self.setWindowTitle("가사 흘려 넣기")
        self.resize(560, 420)
        v = QVBoxLayout(self)
        info = QLabel(f"<b>{where}</b>부터 음표에 한 음절씩 차례로 넣습니다.<br>"
                      "한글은 한 글자 = 한 음절, 영어는 <code>Hal-le-lu-jah</code>처럼 '-'로 나눕니다. "
                      "<code>_</code> = 그 음은 건너뛰기, 혼자 쓴 <code>-</code> = 늘임(–).<br>"
                      "넣은 뒤 어긋난 곳은 그 음을 고르고 <b>Alt+→ / Alt+←</b>로 가사를 한 칸 밉니다.")
        info.setWordWrap(True)
        v.addWidget(info)
        from PySide6.QtWidgets import QPlainTextEdit, QComboBox, QCheckBox
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText("예) 기도할 때 상한 맘 만지시며 ...")
        v.addWidget(self.text, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("절:"))
        self.verse = QComboBox()
        for k in range(6):
            self.verse.addItem(f"{k + 1}절" + (" (가사 있음)" if k in rows_used else ""), k)
        row.addWidget(self.verse)
        row.addStretch(1)
        if ocr_fn is not None:
            b = QPushButton("원본에서 글자 읽기")
            b.setToolTip("원본 악보의 가사 줄을 글자 인식(Tesseract)으로 읽어 위 칸에 채웁니다 — 틀린 글자는 고쳐 주세요")
            b.clicked.connect(lambda: self.text.setPlainText(ocr_fn(self.verse.currentData()) or self.text.toPlainText()))
            row.addWidget(b)
        v.addLayout(row)
        self.skip_tied = QCheckBox("붙임줄로 이어진 음(마디를 넘는 당김음 포함)은 건너뛰기")
        self.skip_slur = QCheckBox("이음줄 안의 뒷음은 건너뛰기(한 음절을 여러 음으로 — 멜리스마)")
        self.voice_only = QCheckBox("고른 음의 성부에만(한 보표에 성부가 둘일 때)")
        self.replace = QCheckBox("그 절의 기존 가사를 바꾸기")
        for cb, on in ((self.skip_tied, True), (self.skip_slur, True), (self.voice_only, False), (self.replace, True)):
            cb.setChecked(on)
            v.addWidget(cb)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)


class AboutDialog(QDialog):
    """[5단계] 프로그램 정보/라이선스 안내.

    HYBRID_OMR_PLAN_v6.md 6.2절 근거: AGPLv3 소스 공개 의무는 GitHub
    저장소 링크를 여기 명시하는 방식으로 충족한다(완성 시점부터 공개 —
    개발 중인 지금은 링크가 준비되면 채워 넣을 자리로 둔다).
    """

    # [배포 시 채워 넣을 값] 계획서 6.2절: 최종 완성본 시점에 GitHub 공개.
    # [v3.21] 소스 코드 공개 주소(AGPLv3: 받은 사람이 소스를 얻을 수 있어야 한다). 저장소를 만들면 여기에 적는다.
    # 비어 있으면 '배포한 사람에게 요청' 안내를 보여 준다(예전엔 "[조직명]/[저장소명]"이 그대로 화면에 나왔다).
    GITHUB_URL = "https://github.com/zpischun-eng/OMR_Sprouty"

    def __init__(self, parent=None, version: str = APP_VERSION, load_info=None, on_rerecognize=None,
                 on_diagnose=None, on_check_update=None, on_batch_verify=None):
        super().__init__(parent)
        self.setWindowTitle("프로그램 정보")
        self.setModal(True)
        self.setMinimumWidth(440)
        self.setStyleSheet(
            f"QDialog {{ background:{Tokens.CARD}; }}"
            f"QLabel {{ color:{Tokens.TEXT}; font-size:12.5px; }}"
            f"QLabel#title {{ font-size:16px; font-weight:700; }}"
            f"QLabel#notice {{ color:{Tokens.TEXT_DIM}; font-size:11.5px; }}"
        )
        v = QVBoxLayout(self)
        v.setContentsMargins(24, 20, 24, 16)
        v.setSpacing(8)

        title = QLabel(f"♪ {APP_NAME}")
        title.setObjectName("title")
        v.addWidget(title)
        ver_row = QHBoxLayout()
        ver_row.addWidget(QLabel(f"버전 {version}"))
        if on_check_update:                            # [v3.50] 새 버전 확인
            bu = QPushButton("새 버전 확인")
            bu.setToolTip("GitHub에 더 새 버전이 나왔는지 확인합니다")
            bu.clicked.connect(on_check_update)
            ver_row.addWidget(bu)
        ver_row.addStretch(1)
        v.addLayout(ver_row)

        notice = QLabel(
            "이 프로그램은 <b>현직 교사를 위한 비상업적 배포용</b>입니다."
        )
        notice.setWordWrap(True)
        v.addWidget(notice)

        v.addSpacing(6)
        if load_info:
            v.addWidget(QLabel("<b>최근 불러온 악보의 인식 안내</b>"))
            info_lbl = QLabel("\n".join("· " + w for w in load_info))
            info_lbl.setWordWrap(True)
            info_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            v.addWidget(info_lbl)
        if on_rerecognize or on_diagnose:
            # [v2.53] 인식이 이상할 때: 저장된 인식 결과를 지우고 다시 인식 / 진단 자료 저장
            row = QHBoxLayout()
            if on_rerecognize:
                b1 = QPushButton("이 악보 다시 인식")
                b1.setToolTip("저장해 둔 인식 결과(캐시)를 지우고 처음부터 다시 인식합니다(편집 파일은 그대로)")
                b1.clicked.connect(lambda: (self.accept(), on_rerecognize()))
                row.addWidget(b1)
            if on_diagnose:
                b2 = QPushButton("진단 자료 저장")
                b2.setToolTip("인식 기록·결과·환경 정보를 zip 하나로 원본 옆에 저장합니다(문제를 알릴 때 보내 주세요)")
                b2.clicked.connect(on_diagnose)
                row.addWidget(b2)
            row.addStretch(1)
            v.addLayout(row)
        if on_batch_verify:
            # [v3.60] 여러 악보를 한꺼번에 인식해 원본과 비교(프로그램 품질 점검용)
            row2 = QHBoxLayout()
            b3 = QPushButton("폴더 일괄 검증…")
            b3.setToolTip("폴더의 악보를 모두 인식해 원본 그림과 편집 악보를 비교하고,\n"
                          "점수표·차이 그림·보내기용 zip을 만듭니다(시간이 오래 걸립니다)")
            b3.clicked.connect(lambda: (self.accept(), on_batch_verify()))
            row2.addWidget(b3)
            row2.addStretch(1)
            v.addLayout(row2)
        v.addWidget(QLabel("<b>사용된 오픈소스 구성 요소</b>"))

        components = QLabel(
            "· Audiveris (AGPLv3) — 악보 인식 엔진<br>"
            "&nbsp;&nbsp;https://github.com/Audiveris/audiveris<br>"
            "· 이 프로그램 자체 (AGPLv3)<br>"
            + (f"&nbsp;&nbsp;{self.GITHUB_URL}<br>" if self.GITHUB_URL else
               "&nbsp;&nbsp;(소스 코드는 이 프로그램을 배포한 사람에게 요청할 수 있습니다)<br>") +
            "· PySide6 (LGPLv3) — 화면 구성 프레임워크"
        )
        components.setWordWrap(True)
        components.setOpenExternalLinks(True)
        components.setTextInteractionFlags(Qt.TextBrowserInteraction)
        v.addWidget(components)

        note2 = QLabel(
            "AGPLv3에 따라 이 프로그램의 소스코드는 위 GitHub 저장소에서 "
            "확인하실 수 있습니다."
        )
        note2.setObjectName("notice")
        note2.setWordWrap(True)
        v.addWidget(note2)

        v.addSpacing(10)
        license_btn = QToolButton()
        license_btn.setText("라이선스 전문 보기")
        license_btn.setCursor(Qt.PointingHandCursor)
        license_btn.clicked.connect(self._show_full_license)
        v.addWidget(license_btn)

        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.button(QDialogButtonBox.Ok).setText("닫기")
        bb.accepted.connect(self.accept)
        v.addWidget(bb)

    def _show_full_license(self):
        """AGPLv3 전문 텍스트 파일을 찾아 별도 창으로 보여준다.

        배포 패키지에는 LICENSE_AGPLv3.txt(자체 코드용)와
        LICENSE_Audiveris_AGPLv3.txt(Audiveris용)를 함께 넣는다 —
        6.1절 "배포 파일에 라이선스 전문 포함" 요건.
        """
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "LICENSE_AGPLv3.txt")
        text = "라이선스 파일을 찾을 수 없습니다."
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                pass
        dlg = QDialog(self)
        dlg.setWindowTitle("AGPLv3 라이선스 전문")
        dlg.resize(600, 500)
        lay = QVBoxLayout(dlg)
        browser = QTextEdit()
        browser.setReadOnly(True)
        browser.setPlainText(text)
        lay.addWidget(browser)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.exec()


# [v2.25] 빠르기말 음표: (내부 키, 넣을 글자, 보이는 이름). 메트로놈 표시는 '한 박이 되는 음표'를 쓴다 —
# 4/4 등은 4분음표, 6/8·9/8·12/8은 점4분음표, 2/2(alla breve)는 2분음표, 빠른 3/4·6/4는 점2분음표,
# 3/8 등 느린 곡은 8분음표(Dorico·MuseScore 안내 등). 렌더러가 이 글자들을 악보 글꼴 음표로 그린다.
TEMPO_NOTES = (("q", "\u2669", "♩ 4분음표"), ("qd", "\u2669.", "♩. 점4분음표 (6/8·9/8·12/8)"),
               ("h", "\U0001D15E", "𝅗𝅥 2분음표 (2/2)"), ("hd", "\U0001D15E.", "𝅗𝅥. 점2분음표 (빠른 3/4·6/4)"),
               ("e", "\u266A", "♪ 8분음표 (3/8 등 느린 곡)"))


def _tempo_note_for_meter(beats, beat_type):
    """박자에 맞는 기본 음표 키."""
    if beat_type == 8 and beats % 3 == 0 and beats >= 6:
        return "qd"
    if beat_type == 2:
        return "h"
    if beat_type == 8:
        return "e"
    return "q"


class SongInfoDialog(QDialog):
    """곡 정보 팝업: 제목·작사/작곡 표기·빠르기말 입력. PDF는 텍스트 레이어에서 자동으로
    채워지고, 이미지(스캔) 입력은 여기서 직접 적는다. 비워 두면 그 항목은 표시하지 않는다.
    [v2.25] 작사/작곡은 '/'로 나누면 원본처럼 여러 줄로, 빠르기말에는 음표(♩ 등)를 넣는 목록."""

    def __init__(self, parent=None, title="", credit="", tempo="", meter=(4, 4)):
        super().__init__(parent)
        self.setWindowTitle("곡 정보")
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setStyleSheet(
            f"QDialog {{ background:{Tokens.CARD}; }}"
            f"QLabel {{ color:{Tokens.TEXT}; font-size:13px; }}"
            f"QLineEdit {{ padding:6px 10px; border:1px solid {Tokens.BORDER}; border-radius:6px; }}")
        v = QVBoxLayout(self)
        v.setSpacing(8)
        self.ed_title, self.ed_credit, self.ed_tempo = QLineEdit(title), QLineEdit(credit), QLineEdit(tempo)
        self.ed_credit.setPlaceholderText("예) 작사 염평안 / 작곡 염평안 / 편곡 강태숙  ('/'마다 줄바꿈)")
        self.ed_tempo.setPlaceholderText("예) 밝게, 리듬을 가지고 (♩=82-85)")
        v.addWidget(QLabel("제목"))
        v.addWidget(self.ed_title)
        v.addWidget(QLabel("작사/작곡  — '/'로 나누면 여러 줄"))
        v.addWidget(self.ed_credit)
        row = QHBoxLayout()
        row.addWidget(QLabel("빠르기말"))
        row.addStretch(1)
        self.note_combo = QComboBox()
        self.note_combo.setToolTip("고르면 빠르기말의 커서 자리에 음표와 '='를 넣습니다")
        self.note_combo.addItem("음표 넣기…", None)
        first = _tempo_note_for_meter(*meter)
        for key, ch, label in sorted(TEMPO_NOTES, key=lambda t: t[0] != first):
            self.note_combo.addItem(label + ("  ← 이 곡 박자" if key == first else ""), ch)
        self.note_combo.activated.connect(self._insert_note)
        row.addWidget(self.note_combo)
        v.addLayout(row)
        v.addWidget(self.ed_tempo)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("적용")
        bb.button(QDialogButtonBox.Cancel).setText("취소")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _insert_note(self, idx):
        ch = self.note_combo.itemData(idx)
        self.note_combo.setCurrentIndex(0)
        if not ch:
            return
        ed_ = self.ed_tempo
        pos = ed_.cursorPosition()
        txt = ed_.text()
        ins = ch if txt[pos:pos + 1] == "=" else ch + "="
        ed_.setText(txt[:pos] + ins + txt[pos:])
        ed_.setCursorPosition(pos + len(ins))
        ed_.setFocus()

    def values(self):
        return (self.ed_title.text().strip(), self.ed_credit.text().strip(),
                self.ed_tempo.text().strip())


class TransposeDialog(QDialog):
    """조바꿈 팝업: 목표 조(15개 장조) + 이동 방향(가까운 쪽/올림/내림) 선택."""

    def __init__(self, parent=None, current_fifths: int = 0):
        super().__init__(parent)
        self.setWindowTitle("조바꿈")
        self.setModal(True)
        self.setStyleSheet(
            f"QDialog {{ background:{Tokens.CARD}; }}"
            f"QLabel {{ color:{Tokens.TEXT}; font-size:13px; }}"
            f"QComboBox {{ padding:6px 10px; border:1px solid {Tokens.BORDER}; "
            f"border-radius:6px; background:white; font-size:13px; }}"
            f"QRadioButton {{ color:{Tokens.TEXT}; font-size:12.5px; padding:2px; }}"
        )
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 18, 20, 14)
        v.setSpacing(10)

        cur_name = dict((f, n) for f, n in ed.KEY_CHOICES).get(
            current_fifths, str(current_fifths))
        v.addWidget(QLabel(f"현재 조: <b>{cur_name}</b>"))
        v.addWidget(QLabel("바꿀 조를 선택하세요:"))

        self.combo = QComboBox()
        for fifths, name in ed.KEY_CHOICES:
            self.combo.addItem(name, fifths)
            if fifths == current_fifths:
                self.combo.setCurrentIndex(self.combo.count() - 1)
        v.addWidget(self.combo)

        v.addWidget(QLabel("이동 방향:"))
        self.rb_near = QRadioButton("가까운 쪽으로 (반음 6개 이내)")
        self.rb_up = QRadioButton("올림 (위쪽으로)")
        self.rb_down = QRadioButton("내림 (아래쪽으로)")
        self.rb_near.setChecked(True)
        for rb in (self.rb_near, self.rb_up, self.rb_down):
            v.addWidget(rb)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("조바꿈")
        bb.button(QDialogButtonBox.Cancel).setText("취소")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def result_values(self):
        direction = ("up" if self.rb_up.isChecked()
                     else "down" if self.rb_down.isChecked() else "nearest")
        return self.combo.currentData(), direction


class HScrollWrap(QScrollArea):
    """[v3.91] 도구 막대를 감싸는 가로 스크롤 틀 — 창이 막대의 최소 폭보다 좁아지면 가로 스크롤바가 생긴다.
    (아래 편집 막대의 최소 폭이 약 1900px이라 창을 그보다 좁힐 수 없었다 — FHD 화면에서 창 크기 조절이 안 됨)"""

    def __init__(self, inner, parent=None):
        super().__init__(parent)
        self._inner = inner
        self.setWidget(inner)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setStyleSheet("QScrollArea { background: transparent; } "
                           "QScrollArea > QWidget > QWidget { background: transparent; }")
        self.viewport().setAutoFillBackground(False)
        self._fit()

    def _fit(self):
        h = max(self._inner.minimumSizeHint().height(), self._inner.sizeHint().height(), self._inner.minimumHeight())
        if self.width() < self._inner.minimumSizeHint().width():
            h += self.horizontalScrollBar().sizeHint().height()
        if self.height() != h:
            self.setFixedHeight(h)

    def minimumSizeHint(self):
        return QSize(200, self.height())

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._fit()


class TopToolbar(QFrame):
    openImageRequested = Signal()
    saveRequested = Signal()           # [v2.50] 편집 저장
    exportMusicXmlRequested = Signal()
    nextErrorRequested = Signal()      # 버튼은 없지만 단축키(F3)용으로 유지
    undoRequested = Signal()           # 버튼은 없지만 단축키(Ctrl+Z)용으로 유지
    transposeRequested = Signal()      # 조바꿈 팝업 열기
    songInfoRequested = Signal()       # 곡 정보(제목·작사작곡·빠르기말) 팝업 열기
    playRequested = Signal()           # [2단계] 재생/정지 토글
    exportMidiRequested = Signal()     # [2단계] .mid 파일 내보내기
    fitToScreenRequested = Signal()
    aboutRequested = Signal()          # [5단계] 프로그램 정보/라이선스

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("toolbar")
        self.setStyleSheet(
            f"QFrame#toolbar {{ background:{Tokens.CARD}; border:1px solid {Tokens.BORDER}; "
            f"border-radius:{Tokens.RADIUS}px; }}"
        )
        self.setFixedHeight(64)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(20, 0, 20, 0)

        capsule = QFrame()
        capsule.setObjectName("capsule")
        capsule.setStyleSheet(
            f"QFrame#capsule {{ border:1px solid {Tokens.BORDER}; border-radius:999px; "
            f"background:{Tokens.CARD}; }}"
        )
        cap_layout = QHBoxLayout(capsule)
        cap_layout.setContentsMargins(4, 4, 4, 4)
        cap_layout.setSpacing(0)

        actions = [
            ("악보 열기", self.openImageRequested, "악보 이미지/PDF 파일 열기 (Ctrl+O)"),
            ("저장", self.saveRequested, "편집한 악보를 원본 옆에 같은 이름(.sprouty)으로 저장 — 다음에 열 때 이어서 편집 (Ctrl+S)"),
            ("MusicXML 내보내기", self.exportMusicXmlRequested, "편집 결과를 MusicXML 로 저장 (Ctrl+E)"),
            ("MIDI 내보내기", self.exportMidiRequested, "편집 결과를 MIDI(.mid) 로 저장"),
            ("▶ 재생", self.playRequested, "악보를 소리로 재생하며 검수 (Space)"),
            ("조바꿈", self.transposeRequested, "악보 전체를 다른 조로 옮기기 (Ctrl+T)"),
            ("곡 정보", self.songInfoRequested, "제목·작사/작곡·빠르기말 입력/수정"),
            ("화면 맞춤", self.fitToScreenRequested, "이미지를 화면에 맞춤 (F)"),
            ("정보", self.aboutRequested, "프로그램 정보 및 라이선스"),
        ]
        self.play_btn = None
        for idx, (label, signal, tip) in enumerate(actions):
            btn = QToolButton()
            btn.setText(label)
            if signal is self.playRequested:
                self.play_btn = btn
            btn.setToolTip(tip)
            btn.setCursor(Qt.PointingHandCursor)
            border = f"border-right:1px solid {Tokens.BORDER};" if idx < len(actions) - 1 else ""
            btn.setStyleSheet(
                f"QToolButton {{ border:none; {border} border-radius:0; padding:10px 16px; "
                f"font-weight:600; font-size:13px; color:{Tokens.TEXT}; background:transparent; }}"
                f"QToolButton:hover {{ background:{Tokens.CHIP_BG}; }}"
            )
            btn.clicked.connect(signal.emit)
            cap_layout.addWidget(btn)
            if signal is self.playRequested:
                # [v3.00] 코드 반주 켜기/끄기(재생·MIDI 내보내기)
                self.chord_btn = QToolButton()
                self.chord_btn.setText("코드 반주")
                self.chord_btn.setCheckable(True)
                self.chord_btn.setToolTip("재생·MIDI에 기타 코드 반주를 넣습니다(켜기/끄기)")
                self.chord_btn.setCursor(Qt.PointingHandCursor)
                self.chord_btn.setStyleSheet(
                    f"QToolButton {{ border:none; border-right:1px solid {Tokens.BORDER}; border-radius:0; "
                    f"padding:10px 12px; font-weight:600; font-size:12px; color:{Tokens.TEXT}; background:transparent; }}"
                    f"QToolButton:checked {{ background:#e8f0fe; color:#1a56c4; }}"
                    f"QToolButton:hover {{ background:{Tokens.CHIP_BG}; }}")
                cap_layout.addWidget(self.chord_btn)

        outer.addStretch(1)
        outer.addWidget(capsule)
        outer.addStretch(1)

        self.stats_label = QLabel("이미지를 열어 주세요")
        self.stats_label.setStyleSheet(
            f"background:{Tokens.CHIP_BG}; border-radius:999px; padding:7px 14px; "
            f"font-weight:600; font-size:11.5px; color:{Tokens.TEXT_DIM};"
        )
        outer.addWidget(self.stats_label, 0, Qt.AlignRight)

    def set_stats(self, total: int, ok: int, error: int):
        self.stats_label.setText(f"마디 {total}개 중 정상 {ok} · 오류 {error}")


# ----------------------------------------------------------------------------
# 하단 편집 바 (좌: 선택 정보 / 우: 버튼 그룹)
# ----------------------------------------------------------------------------

# [셈여림 메뉴] 내부 키 → 화면 문구. 키는 score.editing / 렌더러 / MusicXML이 쓰는 값이고,
# 문구는 표시 시점에만 쓰인다(번역·문구 변경이 동작에 영향을 주지 않도록 분리).
DYN_MENU_LABELS = {
    "pp": "pp   피아니시모 (매우 여리게)", "p": "p    피아노 (여리게)",
    "mp": "mp   메조피아노 (조금 여리게)", "mf": "mf   메조포르테 (조금 세게)",
    "f": "f    포르테 (세게)", "ff": "ff   포르티시모 (매우 세게)",
    "sfz": "sfz  스포르찬도 (그 음만 세게)", "fp": "fp   포르테피아노 (세게 곧 여리게)",
    "cresc": "크레셴도  <   (2개 이상 선택 · 쐐기 안 음표는 지우기)",
    "decresc": "디크레셴도  >   (2개 이상 선택 · 쐐기 안 음표는 지우기)",
    "clear": "선택한 셈여림 지우기", "clear_auto": "자동 인식 셈여림 모두 지우기",
}
TUPLET_MENU_LABELS = {
    "t3": "셋잇단  3   (Ctrl+3, Y)", "t5": "5잇단  5   (Ctrl+5)", "t6": "6잇단  6   (Ctrl+6)",
    "t7": "7잇단  7   (Ctrl+7)", "t2": "2잇단  2   (복합박자, Ctrl+2)",
    "t4": "4잇단  4   (복합박자, Ctrl+4)", "clear": "잇단음표 해제",
}
TUPLET_MENU_ORDER = ["t3", "t5", "t6", "t7", None, "t2", "t4", None, "clear"]
TUPLET_NAMES = {"t3": "셋잇단", "t5": "5잇단", "t6": "6잇단", "t7": "7잇단",
                "t2": "2잇단", "t4": "4잇단", "clear": "잇단음표"}
DYN_MENU_ORDER = ["pp", "p", "mp", "mf", "f", "ff", "sfz", "fp", None,
                  "cresc", "decresc", None, "clear", "clear_auto"]


class BottomEditBar(QFrame):
    pitchUpRequested = Signal()
    pitchDownRequested = Signal()
    accidentalToggleRequested = Signal()
    durationSelected = Signal(int)     # 1/2/4/8/16/32
    dotToggleRequested = Signal()
    fermataToggleRequested = Signal()  # [v1.41] 늘임표 버튼
    articulationRequested = Signal(str)  # [v2.20] 아티큘레이션 종류(내부 키)
    graceRequested = Signal()            # [v2.21] 앞꾸밈음 넣기
    lyricsFillRequested = Signal()       # [v3.10] 가사 흘려 넣기
    beamGroupCycleRequested = Signal()  # [UI 재구성] 8→16→32→8... 순환 토글
    bondToggleRequested = Signal()
    dynamicActionRequested = Signal(str)   # [셈여림] 내부 키(pp…/cresc/decresc/clear/clear_auto)
    tupletActionRequested = Signal(str)    # [잇단음표] 내부 키(t3/t5/t6/t7/t2/t4/clear)
    tripletToggleRequested = Signal()
    noteRestToggleRequested = Signal()
    deleteRequested = Signal()
    insertAfterRequested = Signal()
    insertBeforeRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("bottomBar")
        self.setStyleSheet(
            f"QFrame#bottomBar {{ background:{Tokens.CARD}; border:1px solid {Tokens.BORDER}; "
            f"border-radius:{Tokens.RADIUS}px; }}"
        )

        outer = QHBoxLayout(self)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(12)                 # [v2.20] 아티큘레이션 칸이 늘어 1920 화면에 맞게 줄임

        # --- 좌측: 조표 · 박자표 정보 ---
        sig_chip = QFrame()
        sig_chip.setObjectName("chip")
        sig_chip.setStyleSheet(f"QFrame#chip {{ background:{Tokens.CHIP_BG}; border-radius:8px; }}")
        sig_layout = QVBoxLayout(sig_chip)
        sig_layout.setContentsMargins(16, 10, 16, 10)
        sig_layout.setSpacing(3)
        sig_caption = QLabel("조표 · 박자표")
        sig_caption.setStyleSheet("color:#9a9aa2; font-size:11px;")
        self.sig_label = QLabel("—")
        self.sig_label.setStyleSheet(
            f"color:{Tokens.ACCENT}; font-weight:700; font-size:14.5px;")
        self.sig_label.setMinimumWidth(150)
        sig_layout.addWidget(sig_caption)
        sig_layout.addWidget(self.sig_label)
        outer.addWidget(sig_chip, 0)

        # --- 좌측: 선택 요소 정보 ---
        info_chip = QFrame()
        info_chip.setObjectName("chip")
        info_chip.setStyleSheet(f"QFrame#chip {{ background:{Tokens.CHIP_BG}; border-radius:8px; }}")
        info_layout = QVBoxLayout(info_chip)
        info_layout.setContentsMargins(16, 10, 16, 10)
        info_layout.setSpacing(3)
        caption = QLabel("선택 요소")
        caption.setStyleSheet("color:#9a9aa2; font-size:11px;")
        self.selected_label = QLabel("요소를 클릭하면 여기에 정보가 표시됩니다")
        self.selected_label.setStyleSheet(
            f"color:{Tokens.TEXT}; font-weight:600; font-size:14.5px;")
        self.selected_label.setMinimumWidth(180)    # [v2.20] 긴 글은 …으로 줄이고 툴팁에 전체
        self.selected_label.setMaximumWidth(200)
        info_layout.addWidget(caption)
        info_layout.addWidget(self.selected_label)
        outer.addWidget(info_chip, 0)

        # --- 우측: 버튼 그룹들 ---
        groups = QHBoxLayout()
        groups.setSpacing(8)

        up_btn = flat_button("↑", tip="음 올림  (↑)")
        down_btn = flat_button("↓", tip="음 내림  (↓)")
        up_btn.clicked.connect(self.pitchUpRequested.emit)
        down_btn.clicked.connect(self.pitchDownRequested.emit)
        acc_btn = flat_button("♯♭♮", tip="임시표 순환 ♯→♭→♮→없음  (A)")
        acc_btn.clicked.connect(self.accidentalToggleRequested.emit)
        self.pitch_btns = [up_btn, down_btn, acc_btn]  # [3-B]
        groups.addWidget(make_chip_group("음높이", self.pitch_btns))

        keyhint = {1: "1", 2: "2", 4: "3", 8: "4", 16: "5", 32: "6"}
        self._dur_buttons: dict[int, QPushButton] = {}
        dur_widgets = []
        for base in ed.DURATION_CHOICES:
            btn = flat_button(_DUR_LABEL[base],
                              tip=f"{_DUR_NAME[base]}음표/쉼표  ({keyhint[base]})")
            icon = _note_icon(base)
            if icon is not None:              # 악보 글꼴이 있으면 숫자 대신 음표 모양
                btn.setText("")
                btn.setIcon(icon)
                btn.setIconSize(QSize(26, 26))
            btn.setCheckable(True)
            btn.clicked.connect(lambda _=False, b=base: self.durationSelected.emit(b))
            self._dur_buttons[base] = btn
            dur_widgets.append(btn)
        self.dot_btn = flat_button("", tip="점 → 겹점 → 없음 순서로 바꾸기  (.)")
        self.dot_btn.setIcon(_dot_icon())             # [v1.41] 점(.) 하나만
        self.dot_btn.setIconSize(QSize(22, 22))
        self.dot_btn.setCheckable(True)
        self.dot_btn.clicked.connect(self.dotToggleRequested.emit)
        dur_widgets.append(self.dot_btn)
        groups.addWidget(make_chip_group("음표", dur_widgets))
        # [v2.11 버튼 크기 맞추기 — 실사용] 음표 버튼(50px)이 음높이 버튼(30px)보다 넓어 뚱뚱해 보였고, 아이콘
        # 크기도 달랐다. 두 칸의 버튼 폭을 같게(36px), 음표 아이콘은 20px, 음높이 화살표 글자는 그만큼 키운다.
        for b in self.pitch_btns + dur_widgets:
            b.setFixedWidth(36)
        for b in dur_widgets:
            if not b.icon().isNull():
                b.setIconSize(QSize(20, 20))
        for b in self.pitch_btns[:2]:
            b.setStyleSheet(b.styleSheet() + "QPushButton { font-size:16px; padding:0px; }")
        aci = _accidental_cycle_icon()                    # [v2.23] ♯♭♮ 아이콘(넓게·크게)
        if aci is not None:
            self.pitch_btns[2].setText("")
            self.pitch_btns[2].setIcon(aci)
            self.pitch_btns[2].setIconSize(QSize(42, 28))
        self.pitch_btns[2].setFixedWidth(54)

        # [UI 재구성] 이전에는 8분/16분/32분 버튼 3개(세그먼트 토글)였는데,
        # 버튼 하나로 줄이고 클릭할 때마다 8→16→32→8...로 순환하게 바꿔
        # 확보한 공간을 아래 이음줄/붙임줄/셋잇단음표 편집에 쓴다(실사용
        # 요청 — UI가 계속 늘어나면 "간단한 앱"이라는 미학이 깨진다는
        # 우려에 대한 절충).
        self._beam_cycle = [8, 16, 32]
        self._beam_cycle_idx = 0
        self.beam_btn = flat_button("빔↻", tip="음표 1개: 빔그룹 전체 박자 순환 8→16→32→8… · "
                                              "여러 개: 빔으로 묶기/풀기  (7)")
        self.beam_btn.clicked.connect(self.beamGroupCycleRequested.emit)
        # [크레셴도/디크레셴도 버튼 — 실사용 요청] 셈여림 메뉴 안에만 있어 찾기 어렵다는 지적.
        # 넓게 늘어나 있던 빔그룹 칸에 함께 둔다. 2개 이상 선택 → 만들기, 쐐기 안의 음표를 골라
        # 다시 누르기 → 지우기. 키보드 < / > 와 같은 동작.
        self.cresc_btn = flat_button("", tip="크레셴도: 2개 이상 선택해 만들기 · 쐐기 안 음표를 골라 다시 누르면 지우기  (<)")
        self.cresc_btn.setIcon(_hairpin_icon("cresc"))      # [v1.41] 글자 없이 긴 부등호
        self.cresc_btn.setIconSize(QSize(40, 14))
        self.cresc_btn.setMinimumWidth(48)
        self.cresc_btn.clicked.connect(lambda: self.dynamicActionRequested.emit("cresc"))
        self.decresc_btn = flat_button("", tip="디크레셴도: 2개 이상 선택해 만들기 · 쐐기 안 음표를 골라 다시 누르면 지우기  (>)")
        self.decresc_btn.setIcon(_hairpin_icon("decresc"))
        self.decresc_btn.setIconSize(QSize(40, 14))
        self.decresc_btn.setMinimumWidth(48)
        self.decresc_btn.clicked.connect(lambda: self.dynamicActionRequested.emit("decresc"))
        groups.addWidget(make_chip_group("빔그룹 · 크레셴도",
                                         [self.beam_btn, self.cresc_btn, self.decresc_btn]))
        self.beam_btn.setText("")                             # [v2.22] 기둥+빔 아이콘
        self.beam_btn.setIcon(_stems_beams_icon(2))
        self.beam_btn.setFixedWidth(40)
        self.beam_btn.setIconSize(QSize(26, 26))

        # [편집 기능 추가] Audiveris가 음정·기본 박자는 매우 정확해도
        # 이음줄/붙임줄/셋잇단음표는 이따금 오인식한다(실사용 검증:
        # "여정" 샘플에서 결정적으로 재현). 다중 선택했을 때 그 조합에
        # 실제로 적용 가능한 것만 활성화되고, 클릭/단축키로 있으면
        # 지우고 없으면 만드는 토글 방식이다.
        # [편집 기능 추가] Audiveris가 음정·기본 박자는 매우 정확해도
        # 이음줄/붙임줄/셋잇단음표는 이따금 오인식한다(실사용 검증:
        # "여정" 샘플에서 결정적으로 재현). 다중 선택했을 때 그 조합에
        # 실제로 적용 가능한 것만 활성화되고, 클릭/단축키로 있으면
        # 지우고 없으면 만드는 토글 방식이다.
        #
        # [버튼 통합 — 실사용 요청] "붙임줄"/"이음줄" 두 버튼이 영역
        # 이름(이음/셋잇단)과 중복돼 UI가 번잡하다는 지적을 받아 하나로
        # 합쳤다 — 선택한 두 음의 음정이 같으면 자동으로 붙임줄, 다르면
        # 이음줄로 판정해 적용한다.
        self.bond_btn = flat_button("이음", tip="선택한 두 음표 사이 이음/붙임줄 토글 — 음정이 같으면 "
                                               "붙임줄, 다르면 이음줄로 자동 판정  (T)")
        self.bond_btn.clicked.connect(self.bondToggleRequested.emit)
        # [잇단음표 일반화] 분할 버튼: 누르면 셋잇단, ▾를 누르면 5·6·7·2·4잇단/해제 메뉴.
        # 음표 1개 선택이면 그 음을 쪼개고(4분→셋잇단 8분 3개), 여러 개 선택이면 표시만 고친다.
        self.triplet_btn = QToolButton()
        self.triplet_btn.setText("잇단")
        self.triplet_btn.setToolTip("음표 1개: 잇단음표로 쪼개기 · 여러 개: 잇단음표 표시/해제  "
                                    "(Y=셋잇단, Ctrl+3/5/6/7/2/4)")
        self.triplet_btn.setPopupMode(QToolButton.MenuButtonPopup)
        self.triplet_btn.setCheckable(True)
        self.triplet_btn.setCursor(Qt.PointingHandCursor)
        self.triplet_btn.setStyleSheet(flat_button("x").styleSheet().replace("QPushButton", "QToolButton"))
        self.tup_menu = QMenu(self.triplet_btn)
        for key in TUPLET_MENU_ORDER:
            if key is None:
                self.tup_menu.addSeparator()
                continue
            act = QAction(TUPLET_MENU_LABELS[key], self.tup_menu)
            act.setData(key)
            self.tup_menu.addAction(act)
        self.tup_menu.triggered.connect(lambda a: self.tupletActionRequested.emit(a.data()))
        self.triplet_btn.setMenu(self.tup_menu)
        self.triplet_btn.clicked.connect(lambda: self.tupletActionRequested.emit("t3"))
        self.bond_btn.setEnabled(False)
        self.triplet_btn.setEnabled(False)
        # [셈여림 — 실사용 요청] 버튼 하나 + 메뉴. 세부 항목은 메뉴 안에 두어 화면이
        # 복잡해지지 않게 한다. 자동 인식이 가사·코드기호를 셈여림으로 오인하는 경우가
        # 많아 "자동 인식 셈여림 모두 지우기"도 함께 둔다.
        self.dyn_btn = flat_button("셈여림", tip="셈여림·크레셴도/디크레셴도 붙이기·지우기  (D, <, >)")
        self.dyn_menu = QMenu(self.dyn_btn)
        self.dyn_actions = {}
        for key in DYN_MENU_ORDER:
            if key is None:
                self.dyn_menu.addSeparator()
                continue
            act = QAction(DYN_MENU_LABELS[key], self.dyn_menu)
            act.setData(key)
            self.dyn_menu.addAction(act)
            self.dyn_actions[key] = act
        self.dyn_menu.triggered.connect(lambda a: self.dynamicActionRequested.emit(a.data()))
        self.dyn_btn.setMenu(self.dyn_menu)
        self.set_dyn_context(0, False)
        # [v1.41] 늘임표 켜기/끄기(U) — 단축키만 있어 잘못 인식된 늘임표를 지울 방법을 찾기 어려웠다
        self.fermata_btn = flat_button("", tip="늘임표 붙이기/지우기  (U)")
        fic = _fermata_icon()
        if fic is not None:
            self.fermata_btn.setIcon(fic)
            self.fermata_btn.setIconSize(QSize(24, 22))
        else:
            self.fermata_btn.setText("늘임표")
        self.fermata_btn.clicked.connect(self.fermataToggleRequested.emit)
        groups.addWidget(make_chip_group("이음·잇단·셈여림",
                                         [self.bond_btn, self.triplet_btn, self.dyn_btn]))
        # [v2.22] 모든 버튼을 악보 기호 아이콘으로: 이음줄(뒤집은 초승달) · 셋잇단(기둥+빔+3) · p~f
        for b, ic, wdt, isz in ((self.bond_btn, _slur_icon(), 36, QSize(24, 24)),
                                (self.triplet_btn, _stems_beams_icon(1, True), 56, QSize(32, 26)),
                                (self.dyn_btn, _dynamics_icon(), 72, QSize(48, 24))):
            if ic is not None:
                b.setText("")
                b.setIcon(ic)
                b.setIconSize(isz)
            b.setFixedWidth(wdt)
        # [v2.20 아티큘레이션 칸] 음표 하나하나에 붙는 표시는 따로 모은다(MuseScore 등과 같은 구성).
        # 고른 음표(여러 개 가능)에 붙이기/떼기 — 이미 있으면 지운다. 늘임표도 이 칸으로 옮겼다.
        self.art_btns = {}
        art_widgets = []
        for kind, ch, label, key in (("accent", "\uE4A0", "악센트", "Shift+V"),
                                     ("staccato", "\uE4A2", "스타카토", "Shift+S"),
                                     ("tenuto", "\uE4A4", "테누토", "Shift+N"),
                                     ("marcato", "\uE4AC", "마르카토", "Shift+O")):
            b = flat_button("", tip=f"{label} 붙이기/지우기  ({key})")
            ic = _smufl_icon(ch, 120)            # 아티큘레이션 글리프는 작아서 크게 그린다
            if ic is not None:
                b.setIcon(ic)
            else:
                b.setText({"accent": ">", "staccato": "·", "tenuto": "—", "marcato": "^"}[kind])
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=kind: self.articulationRequested.emit(k))
            self.art_btns[kind] = b
            art_widgets.append(b)
        art_widgets.append(self.fermata_btn)
        groups.addWidget(make_chip_group("아티큘레이션", art_widgets))
        for b in art_widgets:
            b.setFixedWidth(36)
            if not b.icon().isNull():
                b.setIconSize(QSize(20, 20))

        note_rest_btn = flat_button("음표/쉼표", tip="음표 ↔ 쉼표 전환  (`)")
        nri = _note_rest_icon()
        if nri is not None:                               # [v2.21] 악보 기호 아이콘으로 통일
            note_rest_btn.setText("")
            note_rest_btn.setIcon(nri)
        note_rest_btn.clicked.connect(self.noteRestToggleRequested.emit)
        delete_btn = flat_button("Del", danger=True, tip="선택 요소 삭제  (Del)")
        delete_btn.clicked.connect(self.deleteRequested.emit)
        self.insert_btn = flat_button("뒤 삽입", tip="선택 요소 뒤에 새 음표 삽입  (Ins)")
        self.insert_btn.clicked.connect(self.insertAfterRequested.emit)
        # [기능 추가 — 실사용 요청] Audiveris가 마디 첫 음표를 통째로
        # 놓치는 경우(실사용 검증: 원본 데이터에 그 음 자체가 없음을
        # 확인)가 있는데, 그동안 "뒤에 삽입"만 있어 그 자리를 채울
        # 방법이 없었다.
        self.insert_before_btn = flat_button("앞 삽입", tip="선택 요소 앞에 새 음표 삽입  (Shift+Ins)")
        self.insert_before_btn.clicked.connect(self.insertBeforeRequested.emit)
        # [v2.21] 앞꾸밈음 넣기 — 고른 음 앞에 작은 8분 꾸밈음(+이음줄). 음높이는 넣은 뒤 ↑↓
        self.grace_btn = flat_button("꾸밈음", tip="고른 음 앞에 앞꾸밈음 넣기 — 넣은 뒤 ↑↓로 음높이  (Shift+G)")
        self.grace_btn.clicked.connect(self.graceRequested.emit)
        for b, ic in ((self.insert_before_btn, _insert_icon(True)), (self.insert_btn, _insert_icon(False)),
                      (self.grace_btn, _grace_icon())):  # (꾸밈음 아이콘은 72×64)
            if ic is not None:
                b.setText("")
                b.setIcon(ic)
        self.lyrics_btn = flat_button("가사", tip="고른 음부터 가사 흘려 넣기 — 붙여 넣은 가사를 한 음절씩 (L) · 밀기 Alt+→/←")
        self.lyrics_btn.clicked.connect(self.lyricsFillRequested.emit)
        action_btns = [note_rest_btn, delete_btn, self.insert_before_btn, self.insert_btn, self.grace_btn,
                       self.lyrics_btn]
        groups.addWidget(make_chip_group("동작", action_btns))
        for b in action_btns:
            b.setFixedWidth(46)
            if not b.icon().isNull():
                b.setIconSize(QSize(36, 24))

        outer.addLayout(groups, 1)

    # 선택 상태를 버튼(체크 표시)에 반영
    def set_dyn_context(self, n_selected: int, can_hairpin: bool):
        """선택 상태에 따라 셈여림 메뉴 항목을 켜고 끈다."""
        for b in (getattr(self, "cresc_btn", None), getattr(self, "decresc_btn", None)):
            if b is not None:
                b.setEnabled(n_selected >= 1)
        for key, act in self.dyn_actions.items():
            if key == "clear_auto":
                act.setEnabled(True)
            elif key in ("cresc", "decresc"):
                act.setEnabled(n_selected >= 1)
            else:
                act.setEnabled(n_selected >= 1)

    def reflect_selection(self, element, group_base: Optional[int]):
        for base, btn in self._dur_buttons.items():
            btn.setChecked(element is not None and element.duration_base == base)
        self.dot_btn.setChecked(element is not None and bool(element.dotted))
        self.bond_btn.setEnabled(False)
        self.set_dyn_context(1 if element is not None else 0, False)
        self.triplet_btn.setEnabled(element is not None)
        self.triplet_btn.setChecked(element is not None and bool(element.in_triplet))

    def reflect_multi_selection(self, elements: list):
        """[3-B] 다중 선택: 전부 값이 같은 항목만 체크 표시(공통값 없으면
        비워 둠 — '지금 다 다르다'는 뜻이고, 버튼을 누르면 그 값으로
        통일된다)."""
        bases = {e.duration_base for e in elements}
        common_base = next(iter(bases)) if len(bases) == 1 else None
        for base, btn in self._dur_buttons.items():
            btn.setChecked(common_base == base)
        dots = {bool(e.dotted) for e in elements}
        self.dot_btn.setChecked(len(dots) == 1 and True in dots)
        self.bond_btn.setEnabled(ed.can_toggle_bond(elements))
        self.set_dyn_context(len(elements), ed.can_toggle_hairpin(elements))
        self.triplet_btn.setEnabled(ed.can_toggle_triplet(elements))
        tris = {bool(e.in_triplet) for e in elements}
        self.triplet_btn.setChecked(len(tris) == 1 and True in tris)

    def reflect_articulations(self, elements, measure_of=None):
        """[v2.20] 고른 음표들이 모두 그 기호를 가지면 버튼을 눌린 상태로."""
        notes = [e for e in (elements or []) if e is not None and e.kind == "note"]
        for kind, b in getattr(self, "art_btns", {}).items():
            b.setEnabled(bool(notes))
            ok = bool(notes)
            for e in notes:
                members = [e]
                m = measure_of(e) if measure_of else None
                if m is not None and e.chord_group != -1:
                    members = [x for x in m.elements if x.chord_group == e.chord_group]
                if not any(a["kind"] == kind for x in members for a in (x.articulations or [])):
                    ok = False
                    break
            b.setChecked(ok)

    def set_multi_mode(self, n: int, all_notes: bool):
        """[3-B] n>=2 면 다중 선택 모드. 여러 요소에 '공통으로' 적용
        가능한 편집만 켠다: 음높이·임시표는 전부 음표일 때만(쉼표가 섞이면
        의미가 없다), 박자·점·음표/쉼표전환·삭제는 항상 허용, 빔그룹
        일괄변경·뒤에 삽입은 '어느 하나'를 기준으로 해야 해 다중선택에서는
        끈다(단일 선택으로 좁혀서 쓰도록 유도).
        """
        multi = n >= 2
        for b in self.pitch_btns:
            # [v2.20] 음표가 하나라도 있으면 켠다 — 드래그로 고르면 쉼표가 섞이기 마련이라 예전엔 늘 꺼졌다.
            # 쉼표는 음높이 편집에서 건너뛴다(shift_step·cycle_accidental 모두 음표만).
            b.setEnabled((not multi) or all_notes)
        self.insert_btn.setEnabled(not multi)
        self.insert_before_btn.setEnabled(not multi)
        self.beam_btn.setEnabled(True)       # [v1.44] 여러 개 고르면 빔으로 묶기/풀기

    def set_selected_text(self, text: str):
        # 칸에 맞게 줄이고(…), 전체 글은 툴팁으로
        fm = self.selected_label.fontMetrics()
        self.selected_label.setText(fm.elidedText(text, Qt.ElideRight, self.selected_label.maximumWidth()))
        self.selected_label.setToolTip(text)

    def set_signatures(self, fifths: int, beats: int, beat_type: int, ts_known: bool = True,
                       ts_symbol: str | None = None):
        """조표(플랫/샵 개수 + 조 이름)와 박자표를 표시한다.

        ts_known=False면 원본에서 박자표를 찾지 못해(발췌 악보 등) 4/4로
        임시 대체했다는 뜻이므로, 사용자가 알아채고 직접 확인하도록 표시한다.
        """
        key_name = _KEY_NAMES.get(fifths, "?")
        if fifths == 0:
            acc = "♮ 없음"
        elif fifths < 0:
            acc = f"♭ {-fifths}개"
        else:
            acc = f"♯ {fifths}개"
        # [v3.63] 원본에 박자표가 없으면(그레고리오 성가 등 자유 리듬) 마디마다 실제 길이로 검사한다
        suffix = "" if ts_known else ""
        ts_txt = ("박자표 없음(자유 리듬)" if not ts_known else
                  f"C({beats}/{beat_type})" if ts_symbol == "common" else f"{beats}/{beat_type}")
        self.sig_label.setText(f"{key_name} ({acc})  ·  {ts_txt}{suffix}")
        self.sig_label.setStyleSheet(
            f"color:{'#e8a33d' if not ts_known else Tokens.ACCENT}; "
            f"font-weight:700; font-size:14.5px;")


# ----------------------------------------------------------------------------
# 가사(한글) 텍스트 패널 [3단계]
#   - OCR 인식 결과를 줄 단위로 보여주고, 사람이 눈으로 검수·수정한다.
#   - 개별 글자를 클릭해 옮기는 정교한 UI 대신 '줄 단위 텍스트 상자 +
#     적용' 방식을 택했다(PLAN.md "텍스트 패널 우선"): 한글 가사는 보통
#     음표 1개=음절 1개라서, 텍스트를 고치고 적용을 누르면 공백을 뺀
#     글자를 그 줄의 음표에 순서대로 다시 배정하는 것만으로 충분하고,
#     OCR 오탈자 교정이 타자로 훨씬 빠르다.
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 페이지 네비게이션(점) [18장 후속]
#   - 다중 페이지 문서를 하나의 세로 이미지로 이어붙인 지금 구조에서,
#     특정 페이지로 바로 점프해 그 페이지에 맞춰 확대하는 용도.
#   - 점을 누르면 fit_to_page() 로 좌우 패널이 함께(동기화돼 있으므로)
#     그 페이지만 꽉 차게 보여준다 — 페이지가 1개뿐이면 아예 숨긴다.
# ----------------------------------------------------------------------------

class PageNavStrip(QFrame):
    pageSelected = Signal(int)   # 0부터 시작하는 페이지 번호

    _DOT = 8          # 일반 점 지름(px)
    _PILL_W = 8       # 선택된 페이지: 폭은 점과 같고
    _PILL_H = 22      # 세로로 길게 늘어난 '알약' 모양

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(20)                 # [v2.11] 가운데 틈 20% 줄이기(12+24+12 → 9+20+9)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 10, 2, 10)
        lay.setSpacing(8)
        lay.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)   # [v2.11] 점은 세로 가운데
        self._buttons = []
        self._current = -1

    def set_pages(self, n: int):
        """n<=1 이면 숨긴다(단일 페이지 문서에선 페이지 이동 자체가 의미 없음)."""
        lay = self.layout()
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._buttons = []
        if n <= 1:
            self.setVisible(False)
            return
        self.setVisible(True)
        for i in range(n):
            b = QToolButton()
            b.setText("")
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(f"{i + 1}페이지로 이동")
            b.clicked.connect(lambda _=False, idx=i: self.pageSelected.emit(idx))
            self._style_button(b, False)
            self.layout().addWidget(b)
            self._buttons.append(b)
        self._current = -1

    def set_current(self, idx: int):
        if idx == self._current:
            return
        self._current = idx
        for i, b in enumerate(self._buttons):
            self._style_button(b, i == idx)

    def set_ready(self, n_ready: int):
        """[v2.00] 인식이 끝난 페이지 수 — 그 뒤 페이지 점은 흐리게, 툴팁에 '인식 중'."""
        self._n_ready = n_ready
        for i, b in enumerate(self._buttons):
            b.setToolTip(f"{i + 1}페이지로 이동" if i < n_ready else f"{i + 1}페이지 — 인식 중…")
            self._style_button(b, i == self._current)

    def _style_button(self, b, active):
        """활성 페이지만 세로로 긴 알약 모양, 나머지는 작은 점 — 번호나
        테두리 없이 최대한 밋밋하게(사용자 지적: 원 안에 숫자가 있는
        디자인이 밋밋하지 않고 무겁다)."""
        if active:
            b.setFixedSize(self._PILL_W, self._PILL_H)
            b.setStyleSheet(
                f"QToolButton {{ background:{Tokens.ACCENT}; border:none; "
                f"border-radius:{self._PILL_W // 2}px; }}")
        else:
            b.setFixedSize(self._DOT, self._DOT)
            idx = self._buttons.index(b) if b in self._buttons else 0
            pending = idx >= getattr(self, "_n_ready", 10 ** 6)
            b.setStyleSheet(
                f"QToolButton {{ background:{'transparent' if pending else Tokens.BORDER}; "
                f"border:1px {'dashed' if pending else 'none'} {Tokens.TEXT_DIM}; "
                f"border-radius:{self._DOT // 2}px; }}"
                f"QToolButton:hover {{ background:{Tokens.TEXT_DIM}; }}")


# ----------------------------------------------------------------------------
# 메인 윈도우
# ----------------------------------------------------------------------------

class BusyOverlay(QWidget):
    """[대기 화면 — 실사용 요청, FolderKit과 같은 방식] 시간이 걸리는 작업(악보 인식) 동안
    창 전체를 어둡게 덮어 클릭·휠 입력을 막고, 가운데 카드에 회전 애니메이션·경과 시간·
    안내를 보여 준다. 상태 표시줄 문구만으로는 '멈춘 것 같다'는 지적을 받아 만들었다."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.StrongFocus)          # 키 입력도 여기서 흡수
        self._angle = 0
        self._t0 = time.time()
        self._title = ""
        self._hint = ""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        parent.installEventFilter(self)
        self.hide()

    def start(self, title, hint=""):
        self._title, self._hint = title, hint
        self._t0 = time.time()
        self.setGeometry(self.parent().rect())
        self.raise_()
        self.show()
        self.setFocus()
        self._timer.start(33)                        # 약 30fps

    def stop(self):
        self._timer.stop()
        self.hide()

    def _tick(self):
        self._angle = (self._angle + 9) % 360
        self.update()

    def eventFilter(self, obj, ev):
        if obj is self.parent() and ev.type() == ev.Type.Resize and self.isVisible():
            self.setGeometry(self.parent().rect())
        return False

    # 아래 이벤트는 모두 삼켜서 뒤의 화면이 반응하지 않게 한다
    def mousePressEvent(self, e): e.accept()
    def mouseReleaseEvent(self, e): e.accept()
    def mouseDoubleClickEvent(self, e): e.accept()
    def wheelEvent(self, e): e.accept()
    def keyPressEvent(self, e): e.accept()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(20, 24, 33, 120))          # 화면 전체 dim
        cw, ch = 380, 200
        card = QRectF((self.width() - cw) / 2, (self.height() - ch) / 2, cw, ch)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(Tokens.CARD))
        p.drawRoundedRect(card, 16, 16)
        # 회전 고리: 옅은 원 + 진한 호
        cx, cy, r = card.center().x(), card.top() + 62, 22
        ring = QRectF(cx - r, cy - r, 2 * r, 2 * r)
        pen = QPen(QColor("#e3e9f8"), 5)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(ring)
        pen = QPen(QColor(Tokens.ACCENT), 5)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(ring, int(-self._angle * 16), int(100 * 16))
        # 글자
        p.setPen(QColor(Tokens.TEXT))
        f = QFont()
        f.setPointSize(12)
        f.setBold(True)
        p.setFont(f)
        p.drawText(QRectF(card.left(), cy + 34, cw, 26), Qt.AlignCenter,
                   f"{self._title}  {int(time.time() - self._t0)}초")
        f.setPointSize(9)
        f.setBold(False)
        p.setFont(f)
        p.setPen(QColor("#6b7280"))
        p.drawText(QRectF(card.left() + 18, cy + 64, cw - 36, 44),
                   Qt.AlignHCenter | Qt.TextWordWrap, self._hint)


class ScoreEditorWindow(QMainWindow):
    _updateResult = Signal(object, bool)       # [v3.50] 새 버전 확인 결과(정보 dict | 예외 | None, 직접 확인?)
    _verifyProgress = Signal(int, int, str, str)   # [v3.60] 일괄 검증 진행(끝난 수, 전체, 파일, 단계)
    _verifyDone = Signal(object)                   # 요약 dict 또는 예외

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)                 # [v3.20] 악보 파일을 창에 끌어다 놓아 열기
        self.setWindowTitle(f"♪ {APP_NAME} ({APP_ID} v{APP_VERSION}) — 이미지 → MusicXML")
        self.resize(1440, 900)
        self.setStyleSheet(f"QMainWindow {{ background:{Tokens.BG}; }}")

        # [앱 아이콘] 콩나물지기(Sprouty) 로고. PyInstaller onedir 빌드 후
        # 실행 위치가 바뀌므로, __file__ 기준 상대경로로 찾는다(개발/배포
        # 양쪽 모두 이 폴더 구조를 그대로 유지 — src/icon/app_icon.ico).
        _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "src", "icon", "app_icon.ico")
        if os.path.exists(_icon_path):
            self.setWindowIcon(QIcon(_icon_path))

        # ---- 상태 ----
        self.original_bgr: Optional[np.ndarray] = None
        self.score = None
        self.header_items = []   # 곡 정보 텍스트(제목·크레딧·빠르기말) — 원본 픽셀 좌표
        self.by_sys = None
        self.staff_groups = None   # [Tier2] {line_idx: [[y0..y4], ...]} 다중 보표 좌표
        self.staff_left = None     # [Tier2 버그 수정] {line_idx: 실측 오선 시작 x}
        self.barlines = {}
        self.sx = self.sy = 1.0
        self.layout_obj = None
        self.selected = None
        self.selected_multi: list = []    # [3-B] Ctrl+클릭 다중 선택(2개 이상일 때)
        self._undo: list = []
        self.doc_pages = []               # [1단계] 다중 페이지 정보
        self._title_credits = {}          # [4단계] 곡 제목·크레딧
        self._playing = False             # [2단계] 재생 상태
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._sound = None

        # ---- 레이아웃 ----
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        self.toolbar = TopToolbar()
        if getattr(self.toolbar, "chord_btn", None) is not None:          # [v3.00] 코드 반주
            self.toolbar.chord_btn.toggled.connect(lambda on: setattr(score_midi, "CHORD_ACCOMP", bool(on)))
        self._toolbar_wrap = HScrollWrap(self.toolbar)      # [v3.91] 좁은 창에서는 가로 스크롤
        root.addWidget(self._toolbar_wrap)

        panels = QHBoxLayout()
        panels.setSpacing(2)                 # [v2.21] 가운데 틈 2+20+2px

        left_card = make_panel_card()
        left_v = QVBoxLayout(left_card)
        left_v.setContentsMargins(1, 1, 1, 1)
        left_v.setSpacing(0)
        left_v.addWidget(self._make_panel_header(
            "원본 이미지 · 드래그 이동 / 휠 확대", QColor("#9a9aa2"), accent=False))
        self.image_panel = OriginalImagePanel()
        self.image_panel.pickRequested.connect(self._pick_from_original)
        self.image_panel.brushDone.connect(self._on_brush_done)
        self.image_panel.markClicked.connect(self._on_mark_clicked)      # [v4.10]
        self.image_panel.brushToggled.connect(self._on_brush_toggled_marks)
        self.image_panel.candidateChosen.connect(self._apply_brush)
        left_v.addWidget(self.image_panel, 1)
        panels.addWidget(left_card, 1)

        self.page_nav = PageNavStrip()
        self.page_nav.setVisible(False)
        panels.addWidget(self.page_nav, 0)

        right_card = make_panel_card()
        right_v = QVBoxLayout(right_card)
        right_v.setContentsMargins(1, 1, 1, 1)
        right_v.setSpacing(0)
        right_v.addWidget(self._make_panel_header(
            "편집 악보 · 좌측과 동일 배율 동기화 · 클릭해 선택", QColor(Tokens.ACCENT), accent=True))
        self.score_panel = ScorePanel()
        self.score_panel.contextRequested.connect(self._score_context_menu)   # [v2.41] 오른쪽 버튼 메뉴
        right_v.addWidget(self.score_panel, 1)
        panels.addWidget(right_card, 1)

        root.addLayout(panels, 1)

        self.bottom_bar = BottomEditBar()
        self._bottom_wrap = HScrollWrap(self.bottom_bar)
        root.addWidget(self._bottom_wrap)

        self.status = QLabel("")
        self.status.setStyleSheet(
            f"color:{Tokens.TEXT_DIM}; font-size:11.5px; padding:0 6px;")
        root.addWidget(self.status)

        # ---- 배선 ----
        self._wire_view_sync()
        self._wire_toolbar()
        self._wire_bottom_bar()
        self._bind_shortcuts()

        if os.path.exists(DEFAULT_IMAGE):
            self.load_image(DEFAULT_IMAGE)

    # ------------------------------------------------------------------ UI
    def _make_panel_header(self, text: str, dot_color: QColor, accent: bool) -> QWidget:
        header = QWidget()
        header.setObjectName("panelHeader")
        h = QHBoxLayout(header)
        h.setContentsMargins(18, 12, 18, 12)
        h.setSpacing(8)
        dot = QLabel()
        dot.setFixedSize(6, 6)
        dot.setStyleSheet(f"background:{dot_color.name()}; border-radius:3px;")
        label = QLabel(text)
        label.setStyleSheet(f"color:{dot_color.name()}; font-weight:600; font-size:12px; background:transparent;")
        h.addWidget(dot)
        h.addWidget(label)
        h.addStretch(1)
        bg = "rgba(47,111,237,0.05)" if accent else Tokens.CARD
        header.setStyleSheet(
            f"QWidget#panelHeader {{ background:{bg}; border-bottom:1px solid {Tokens.BORDER}; "
            f"border-top-left-radius:{Tokens.RADIUS}px; border-top-right-radius:{Tokens.RADIUS}px; }}")
        return header

    def _wire_view_sync(self):
        self.image_panel.viewChanged.connect(self.score_panel.set_view)
        self.score_panel.viewChanged.connect(self.image_panel.set_view)
        self.score_panel.elementPicked.connect(self.on_select)
        self.score_panel.marqueeSelected.connect(self.on_marquee_select)

    def _wire_toolbar(self):
        self.toolbar.openImageRequested.connect(self.on_open)
        self.toolbar.saveRequested.connect(self.on_save)
        self.toolbar.exportMidiRequested.connect(self.on_export_midi)
        self.toolbar.playRequested.connect(self.on_play)
        self.page_nav.pageSelected.connect(self.on_goto_page)
        self.toolbar.exportMusicXmlRequested.connect(self.on_export)
        self.toolbar.nextErrorRequested.connect(self.on_next_error)
        self.toolbar.undoRequested.connect(self.on_undo)
        self.toolbar.transposeRequested.connect(self.on_transpose)
        self.toolbar.songInfoRequested.connect(self.on_song_info)
        self.toolbar.fitToScreenRequested.connect(self.fit_both)
        self.toolbar.aboutRequested.connect(self.on_about)

    def _wire_bottom_bar(self):
        bar = self.bottom_bar
        bar.pitchUpRequested.connect(lambda: self._edit(ed.shift_step, 1))
        bar.pitchDownRequested.connect(lambda: self._edit(ed.shift_step, -1))
        bar.accidentalToggleRequested.connect(lambda: self._edit(self._cycle_acc))
        bar.durationSelected.connect(lambda b: self._edit(ed.set_duration, b))
        bar.dotToggleRequested.connect(lambda: self._edit(ed.toggle_dot))
        bar.fermataToggleRequested.connect(lambda: self._edit(ed.toggle_fermata))
        bar.articulationRequested.connect(self._toggle_articulation)
        bar.graceRequested.connect(self._add_grace)
        bar.lyricsFillRequested.connect(self._lyrics_fill)
        bar.beamGroupCycleRequested.connect(self._edit_group_cycle)
        bar.bondToggleRequested.connect(self._toggle_bond)
        bar.dynamicActionRequested.connect(self._dynamic_action)
        bar.tupletActionRequested.connect(self._tuplet_action)
        bar.tripletToggleRequested.connect(self._toggle_triplet)
        bar.noteRestToggleRequested.connect(lambda: self._edit(ed.toggle_note_rest))
        bar.deleteRequested.connect(self._delete)
        bar.insertAfterRequested.connect(self._insert)
        bar.insertBeforeRequested.connect(self._insert_before)

    def _bind_shortcuts(self):
        def sc(seq, fn):
            s = QShortcut(QKeySequence(seq), self)
            # 인식(대기 화면) 중에는 단축키도 막는다
            s.activated.connect(lambda: None if getattr(self, "_loading", False) else fn())

        for key, base in {"1": 1, "2": 2, "3": 4, "4": 8, "5": 16, "6": 32}.items():
            sc(key, lambda b=base: self._edit(ed.set_duration, b))
        sc("7", self._edit_group_cycle)  # [UI 재구성] 8→16→32→8… 순환(이전엔 7/8/9 세 키)
        sc(".", lambda: self._edit(ed.toggle_dot))
        sc("`", lambda: self._edit(ed.toggle_note_rest))
        sc("A", lambda: self._edit(self._cycle_acc))
        sc("Up", lambda: self._edit(ed.shift_step, 1))
        sc("Down", lambda: self._edit(ed.shift_step, -1))
        sc("Left", lambda: self._nudge_position(-1))
        sc("Right", lambda: self._nudge_position(1))
        sc("Ctrl+Left", lambda: self._nudge_position(-4))     # [v2.41] 4배 거리
        sc("Ctrl+Right", lambda: self._nudge_position(4))
        sc("Delete", self._delete)
        sc("PgUp", lambda: self._page_step(-1))      # [v2.00] 이전·다음 페이지
        sc("PgDown", lambda: self._page_step(1))
        sc("Return", self._accept_suggestion)        # [v1.40] 제안 음표 넣기
        sc("Enter", self._accept_suggestion)
        sc("Insert", self._insert)
        sc("Shift+Insert", self._insert_before)
        sc("Ctrl+Z", self.on_undo)
        sc("Ctrl+T", self.on_transpose)
        sc("Ctrl+O", self.on_open)
        sc("Ctrl+E", self.on_export)
        sc("F3", self.on_next_error)
        sc("F", self.fit_both)
        sc("Space", self.on_play)

        # [Tier2] 늘임표·꾸밈음 토글, 화음 추가/제거, 보표·성부 전환.
        # 기존 단축키(1~9, ., `, A, 화살표, Delete, Insert, Ctrl+Z/T/O/E,
        # F3, F, Space)와 겹치지 않는 키로 배정했다.
        sc("U", lambda: self._edit(ed.toggle_fermata))       # 늘임표(U자 모양 연상)
        # [v2.20] 아티큘레이션 — MuseScore와 같은 단축키
        sc("Ctrl+S", self.on_save)                    # [v2.50] 편집 저장
        sc("B", lambda: self.image_panel.set_brush(not self.image_panel._brush_on))   # [v2.60] 붓 켜기/끄기
        sc("K", self._edit_chord)                     # [v3.00] 고른 음에 기타 코드 넣기·고치기·지우기
        sc("L", self._lyrics_fill)                    # [v3.10] 가사 흘려 넣기
        sc("Alt+Right", lambda: self._lyrics_shift(+1))   # [v3.10] 가사 한 칸 밀기(오른쪽)
        sc("Alt+Left", lambda: self._lyrics_shift(-1))    # [v3.10] 가사 한 칸 당기기(왼쪽)
        sc("Escape", self._on_escape)                 # [v3.40] 칠한 것 지우기 + 붓 끄기
        sc("Shift+X", self._flip_stems)               # [v2.25] 기둥 방향 뒤집기(X는 '화음에서 음 빼기')
        sc("Shift+G", self._add_grace)                # [v2.21] 앞꾸밈음 넣기(G는 '고른 음을 꾸밈음으로')
        sc("Shift+V", lambda: self._toggle_articulation("accent"))
        sc("Shift+S", lambda: self._toggle_articulation("staccato"))
        sc("Shift+N", lambda: self._toggle_articulation("tenuto"))
        sc("Shift+O", lambda: self._toggle_articulation("marcato"))
        sc("G", lambda: self._edit(ed.toggle_grace))         # 꾸밈음(Grace)
        sc("C", self._add_chord_note)                        # 화음에 음 추가(Chord)
        sc("X", self._remove_chord_note)                     # 화음에서 음 제거
        sc("S", self._cycle_staff)                            # 보표 전환(Staff)
        sc("V", self._cycle_voice)                            # 성부 전환(Voice)

        # [편집 기능 추가] 붙임줄(Tie)/이음줄(sLur)/셋잇단음표(tripletY).
        # S는 이미 보표 전환에 쓰이고 있어 이음줄은 L로 배정했다.
        sc("T", self._toggle_bond)
        # [셈여림] D는 셈여림 메뉴 열기. < / > 는 keyPressEvent에서 '입력된 문자'로 처리한다
        # (자판에 따라 Shift+쉼표 등으로 들어와 단축키 조합으로는 안정적으로 안 잡힘).
        sc("D", lambda: self.bottom_bar.dyn_btn.showMenu())
        # [화면 캡처 — 실사용 요청] 노트북 자판에서 Alt+PrintScreen이 안 돼서 프로그램 안에서
        # 바로 캡처한다(화면 버튼은 두지 않고 단축키만).
        sc("Ctrl+Alt+P", self._capture_window)
        sc("Y", lambda: self._tuplet_action("t3"))
        for _k in ("3", "5", "6", "7", "2", "4"):
            sc("Ctrl+" + _k, lambda k=_k: self._tuplet_action("t" + k))

    # --------------------------------------------------------------- 동기화
    def fit_both(self):
        if getattr(self, "_book_state", None) is not None:
            self.on_goto_page(getattr(self, "_cur_page", 0))   # [v2.00] 여러 페이지는 지금 페이지에 맞춤
            return
        self.score_panel.page_bounds = None  # 전체 문서 보기로 — 페이지 잠금 해제
        self.image_panel.fit_to_view()       # 브로드캐스트 → 우측 자동 정렬

    def _page_step(self, d):
        if getattr(self, "_book_job", None) is None:
            return
        self.on_goto_page(max(0, min(self._book_job.n_pages - 1, getattr(self, "_cur_page", 0) + d)))

    def on_goto_page(self, idx: int):
        """[페이지 네비게이션] 점을 눌러 특정 페이지로 바로 이동.

        page_bounds(페이지 잠금 범위)는 패널 각자의 상태라 zoom/pan 만
        옮겨주는 viewChanged 동기화 신호로는 전파되지 않는다 — 그래서
        두 패널 모두에 명시적으로 걸어준다. 그래야 어느 쪽 패널을 직접
        드래그·휠 확대하더라도(둘 다 상호작용 가능) 그 페이지 범위를
        벗어나지 못한다.
        """
        pages = getattr(self, "doc_pages", []) or []
        book = getattr(self, "_book_state", None)
        if book is not None and self._book_job is not None and len(pages) <= idx < self._book_job.n_pages:
            self._set_status(f"{idx + 1}페이지는 아직 인식 중입니다 — 끝나면 바로 볼 수 있습니다")
            return
        if not (0 <= idx < len(pages)):
            return
        if book is not None and idx != getattr(self, "_cur_page", 0):
            # [v2.00] 페이지를 바꾸면 두 패널의 그림을 그 페이지 것으로 바꾼다
            self._cur_page = idx
            self.selected, self.selected_multi, self.selected_suggestion = None, [], None
            self._show_book_page_image(idx)
            self._rerender()
            self.on_select(None)
        p = pages[idx]
        self.score_panel.page_bounds = (p.dy_orig, max(1.0, p.h_orig))
        self.image_panel.fit_to_page(p.dy_orig, p.h_orig)
        self.page_nav.set_current(idx)
        self._set_status(f"{idx + 1}/{len(pages)}페이지로 이동")

    # ------------------------------------------------------------- 이미지
    # ---------------------------------------------------------- [v3.20] 파일 끌어다 놓기
    _DROP_EXT = (".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def _dropped_path(self, mime):
        if mime is None or not mime.hasUrls():
            return None
        for u in mime.urls():
            p = u.toLocalFile()
            if p and p.lower().endswith(self._DROP_EXT) and os.path.isfile(p):
                return p
        return None

    def dragEnterEvent(self, event):
        if self._dropped_path(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._dropped_path(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        p = self._dropped_path(event.mimeData())
        if not p:
            self._set_status("끌어다 놓을 수 있는 파일: PDF, JPG, PNG, BMP, TIF")
            event.ignore()
            return
        event.acceptProposedAction()
        self.load_image(p)

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "악보 파일 선택", "",
            "악보 파일 (*.pdf *.jpg *.jpeg *.png *.bmp *.tif *.tiff);;"
            "PDF (*.pdf);;이미지 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;"
            "모든 파일 (*.*)")
        if path:
            self.load_image(path)

    _loadFinished = Signal(object, object, object)   # (경로, 문서 결과, 오류 메시지)
    _bookFirst = Signal(object, object, object, object)  # [v2.00] (경로, 책 작업, 1쪽 결과, 오류)
    _bookPage = Signal(int, int, object, object)          # [v2.00] (세대, 페이지 번호, 결과, 오류)

    def load_image(self, path):
        """이미지 또는 PDF를 연다.

        [로딩 속도 — 실사용 피드백] 인식(Audiveris)은 처음 여는 악보에서 수십 초가 걸려,
        예전엔 그동안 창 전체가 멈췄다. 이제 인식은 뒤에서(별도 스레드) 돌리고 상태
        표시줄에 경과 시간을 보여 준다. 끝나면 화면 갱신은 원래대로 이 창에서 한다.
        한 번 인식한 악보는 결과가 저장돼(document_adapter 캐시) 다음부터 곧바로 열린다."""
        if getattr(self, "_loading", False):
            self._set_status("이전 악보를 아직 인식하는 중입니다 — 끝난 뒤 다시 열어 주세요")
            return
        if not self._confirm_discard():
            return
        self._source_path = path
        self._dirty = False
        self._pending_project = self._ask_resume(path) if getattr(self, "_ask_resume_enabled", True) else None
        if str(path).lower().endswith(".pdf"):
            try:
                from omr_bridge.book import n_pdf_pages
                if n_pdf_pages(path) > 1:
                    return self._load_book(path)       # [v2.00] 여러 페이지 PDF
            except Exception:
                pass
        self._loading = True
        self._load_t0 = time.time()
        self._set_status("악보 인식 중… (처음 여는 악보는 시간이 걸립니다)")
        QApplication.setOverrideCursor(Qt.BusyCursor)
        if not hasattr(self, "_load_timer"):
            self._load_timer = QTimer(self)
            self._load_timer.timeout.connect(lambda: self._set_status(
                f"악보 인식 중… {int(time.time() - self._load_t0)}초 "
                f"(처음 여는 악보는 시간이 걸립니다)"))
            self._loadFinished.connect(self._on_load_finished)
        self._load_timer.start(1000)
        if not hasattr(self, "_busy"):
            self._busy = BusyOverlay(self)
        # 캐시로 금방 끝나는 경우 깜빡임이 없도록 0.3초가 지나도 인식 중일 때만 덮는다
        QTimer.singleShot(300, lambda: self._busy.start(
            "악보를 인식하고 있습니다",
            "처음 여는 악보는 시간이 걸립니다.\n한 번 인식한 악보는 다음부터 바로 열립니다.")
            if self._loading else None)

        def work():
            try:
                doc_ = document.run_document(path)
                self._loadFinished.emit(path, doc_, None)
            except document.OmrBridgeError as ex:
                # [v1.21] 사용자가 고칠 수 있는 문제(엔진 없음 등)는 코드 추적 없이 안내 문구만
                self._loadFinished.emit(path, None, _USER_MSG + str(ex))
            except Exception:
                import traceback
                self._loadFinished.emit(path, None, traceback.format_exc())
        threading.Thread(target=work, daemon=True).start()

    def _on_load_finished(self, path, doc, tb):
        self._load_timer.stop()
        if hasattr(self, "_busy"):
            self._busy.stop()
        QApplication.restoreOverrideCursor()
        self._loading = False
        if tb is not None and tb.startswith(_USER_MSG):
            QMessageBox.warning(self, "악보를 인식할 수 없습니다", tb[len(_USER_MSG):])
            self._set_status("인식 실패")
            return
        if tb is not None:
            print(tb)
            QMessageBox.critical(self, "인식 오류",
                                 "악보 인식 중 오류가 발생했습니다.\n\n" + tb[-1500:])
            self._set_status("인식 실패")
            return
        self._apply_loaded(path, doc)

    # ------------------------------------------------------------- [v2.00] 여러 페이지
    def _load_book(self, path):
        """여러 페이지 PDF: 1쪽을 인식하는 동안만 대기 화면 → 1쪽을 바로 보여 주고, 나머지 쪽은 뒤에서
        차례로 인식해 곡 끝에 덧붙인다(편집은 1쪽부터 바로 할 수 있다)."""
        self._loading = True
        self._load_t0 = time.time()
        QApplication.setOverrideCursor(Qt.BusyCursor)
        if not hasattr(self, "_book_timer"):
            self._book_timer = QTimer(self)
            self._book_timer.timeout.connect(lambda: self._set_status(
                f"1페이지 인식 중… {int(time.time() - self._load_t0)}초"))
            self._bookFirst.connect(self._on_book_first)
            self._bookPage.connect(self._on_book_page)
        self._book_timer.start(1000)
        if not hasattr(self, "_busy"):
            self._busy = BusyOverlay(self)
        QTimer.singleShot(300, lambda: self._busy.start(
            "1페이지를 인식하고 있습니다",
            "여러 페이지 악보는 1페이지가 끝나면 바로 보여 드리고,\n나머지 페이지는 편집하는 동안 뒤에서 인식합니다.")
            if self._loading else None)

        def work():
            try:
                from omr_bridge.book import BookJob
                from omr_bridge.page_builder import build_page
                job = BookJob(path)
                job.prepare()
                bad = job.verify()                       # [v2.53] 저장된 인식 결과가 서로 맞는지
                if bad:
                    job.reset_recognition()
                    self._cache_reset_note = (f"저장된 인식 결과가 서로 맞지 않아({', '.join(map(str, bad))}페이지) "
                                              "처음부터 다시 인식했습니다")
                job.process_sheet(1)
                doc = job.page_doc(1)
                pm = build_page(doc, sheet=1)
                size = (doc.stacked_bgr.shape[1], doc.stacked_bgr.shape[0])
                self._bookFirst.emit(path, job, (pm, size, job.logical_parts(1)), None)
            except document.OmrBridgeError as ex:
                self._bookFirst.emit(path, None, None, _USER_MSG + str(ex))
            except Exception:
                import traceback
                self._bookFirst.emit(path, None, None, traceback.format_exc())
        threading.Thread(target=work, daemon=True).start()

    def _on_book_first(self, path, job, payload, tb):
        self._book_timer.stop()
        if hasattr(self, "_busy"):
            self._busy.stop()
        QApplication.restoreOverrideCursor()
        self._loading = False
        if tb is not None:
            if tb.startswith(_USER_MSG):
                QMessageBox.warning(self, "악보를 인식할 수 없습니다", tb[len(_USER_MSG):])
            else:
                print(tb)
                QMessageBox.critical(self, "인식 오류", "악보 인식 중 오류가 발생했습니다.\n\n" + tb[-1500:])
            self._set_status("인식 실패")
            return
        from omr_bridge.book import BookState, append_page
        pm, size, logical = payload
        st = BookState()
        append_page(st, pm, 1, size, logical)
        self._book_gen = getattr(self, "_book_gen", 0) + 1
        self._book_job, self._book_state, self._book_path = job, st, path
        self._apply_book_state(path)
        if job.n_pages > 1:
            self._start_book_rest(job, self._book_gen)

    def _start_book_rest(self, job, gen):
        def work():
            from omr_bridge.page_builder import build_page
            for n in range(2, job.n_pages + 1):
                if gen != getattr(self, "_book_gen", 0):
                    return                               # 다른 악보를 열었다
                try:
                    job.process_sheet(n)
                    doc = job.page_doc(n)
                    pm = build_page(doc, sheet=n)
                    size = (doc.stacked_bgr.shape[1], doc.stacked_bgr.shape[0])
                    self._bookPage.emit(gen, n, (pm, size, job.logical_parts(n)), None)
                except Exception:
                    import traceback
                    self._bookPage.emit(gen, n, None, traceback.format_exc())
                    return
        threading.Thread(target=work, daemon=True).start()

    def _on_book_page(self, gen, n, payload, tb):
        """뒤에서 인식이 끝난 페이지를 곡 끝에 잇는다(되돌리기 기록에도 같은 내용을 덧붙여, 되돌려도
        새 페이지가 사라지지 않게)."""
        if gen != getattr(self, "_book_gen", 0) or self._book_state is None:
            return
        st = self._book_state
        if tb is not None:
            print(tb)
            self._set_status(f"{n}페이지 인식에 실패했습니다 — 앞 페이지까지만 편집할 수 있습니다")
            self._last_load_info = list(getattr(self, "_last_load_info", []) or []) + [f"{n}페이지 인식 실패"]
            return
        from omr_bridge.book import append_page, renumber_endings
        pm, size, logical = payload
        st.score = self.score                          # 지금 편집 중인 악보(되돌리기 후일 수 있음)
        append_page(st, pm, n, size, logical)
        page = pm.score
        parts = (page.measures, page.arcs, list(getattr(page, "hairpins", [])),
                 list(getattr(page, "suggestions", None) or []), list(getattr(page, "ink_blobs", None) or []))
        for snap in self._undo:
            ms, arcs, hps, sugs, blobs = copy.deepcopy(parts)
            snap.measures += ms
            snap.arcs += arcs
            snap.hairpins = list(getattr(snap, "hairpins", [])) + hps
            snap.suggestions = list(getattr(snap, "suggestions", None) or []) + sugs
            snap.ink_blobs = list(getattr(snap, "ink_blobs", None) or []) + blobs
            snap.staff_parts, snap.staff_clefs, snap.part_names = st.staff_parts, st.staff_clefs, st.part_names
            renumber_endings(snap)
        self.doc_pages = st.pages
        self._last_load_info = list(getattr(self, "_last_load_info", []) or []) + \
            [x for x in st.notes if x.startswith(f"{n}페이지")]
        self.page_nav.set_ready(len(st.pages))
        self._refresh_ink_marks()
        self._rerender()                               # 마디 수·검사 표시 갱신(그림은 지금 페이지만)
        note_ = getattr(self, "_cache_reset_note", None)
        if note_ and note_ not in (self._last_load_info or []):
            self._last_load_info = [note_] + list(self._last_load_info or [])
        left = self._book_job.n_pages - len(st.pages)
        self._set_status(f"{n}페이지 인식 완료" + (f" — 나머지 {left}페이지 인식 중…" if left else
                                                  " — 모든 페이지를 불러왔습니다"))
        if not left:
            self._apply_pending_project()             # [v2.50] 모든 페이지가 들어온 뒤 저장한 편집으로 바꾼다

    def _apply_book_state(self, path):
        st, job = self._book_state, self._book_job
        self.score = st.score
        self.by_sys = st.by_sys
        self.staff_groups, self.staff_left = st.staff_groups, st.staff_left
        self.staff_clefs, self.staff_parts = st.staff_clefs, st.staff_parts
        self.barlines = st.barlines
        self.header_items = st.header_items
        self._title_credits = st.title_credits
        self.sx = self.sy = 1.0
        self.doc_pages = st.pages
        self._cur_page = 0
        self._sync_meta_from_items()
        self._undo.clear()
        self.selected, self.selected_multi, self.selected_suggestion = None, [], None
        self._show_book_page_image(0)
        self.image_panel.set_highlight(None)
        self._refresh_ink_marks()
        self._rerender()
        self.on_select(None)
        self.page_nav.set_pages(job.n_pages)
        self.page_nav.set_ready(len(st.pages))
        QTimer.singleShot(60, lambda: self.on_goto_page(0))
        title = (st.title_credits or {}).get("title")
        self.setWindowTitle(f"♪ {APP_NAME} v{APP_VERSION} — {title} ({path})" if title
                            else f"♪ {APP_NAME} v{APP_VERSION} — {path}")
        self._last_load_info = list(st.warnings) + list(st.notes)
        left = job.n_pages - len(st.pages)
        self._set_status(f"불러옴: {path} · {job.n_pages}페이지" +
                         (f" — 나머지 {left}페이지는 뒤에서 인식 중(편집은 바로 하셔도 됩니다)" if left else ""))

    def _show_book_page_image(self, idx):
        """원본 패널에는 지금 페이지 그림만 올린다(메모리 — 여러 페이지를 다 올리지 않는다)."""
        st, job = self._book_state, self._book_job
        img = job.page_image(idx + 1)
        self.original_bgr = img
        self.image_panel.set_image(cv_bgr_to_qpixmap(img), arr=img)
        self.image_panel.img_dy = st.pages[idx].dy_orig
        self.image_panel.world_size = (max(w for w, _h in st.page_sizes), st.total_h)
        self.image_panel.update()

    def load_image_sync(self, path, resume=False):
        """(시험·스크립트용) 인식부터 화면 갱신까지 이 자리에서 끝까지 실행한다.
        여러 페이지 PDF는 이미 인식된(캐시) 페이지까지 모두 잇는다. resume=True면 편집 파일로 이어서."""
        from score import project as _pj
        self._source_path = path
        self._dirty = False
        self._pending_project = _pj.project_path(path) if resume and _pj.peek(path) else None
        if str(path).lower().endswith(".pdf"):
            from omr_bridge.book import n_pdf_pages, BookJob, BookState, append_page
            if n_pdf_pages(path) > 1:
                from omr_bridge.page_builder import build_page
                job = BookJob(path)
                job.prepare()
                job.process_sheet(1)
                st = BookState()
                for n in range(1, job.n_pages + 1):
                    if not job.is_done(n):
                        break
                    doc = job.page_doc(n)
                    append_page(st, build_page(doc, sheet=n), n,
                                (doc.stacked_bgr.shape[1], doc.stacked_bgr.shape[0]), job.logical_parts(n))
                self._book_gen = getattr(self, "_book_gen", 0) + 1
                self._book_job, self._book_state, self._book_path = job, st, path
                self._apply_book_state(path)
                self._apply_pending_project()
                return
        self._apply_loaded(path, document.run_document(path))

    def _apply_loaded(self, path, doc):
        self._load_notes = []
        self._book_gen = getattr(self, "_book_gen", 0) + 1     # [v2.00] 이전 책의 뒤 작업 결과는 무시
        self._book_state = None
        self._book_job = None
        self.image_panel.img_dy = 0.0
        try:
            img = doc.stacked_bgr
            ctx = doc.ctx
            self.original_bgr = img
            self.doc_pages = doc.pages
            # [v2.01] 한 페이지도 여러 페이지와 같은 처리 단계(page_builder)를 거친다 —
            # 유령 쉼표·유령 머리·붙임줄 오인·미인식 잉크·이음줄 되살리기·자동 보완·꾸밈음·가사
            from omr_bridge.page_builder import build_page
            pm = build_page(doc, sheet=1)
            self.score = pm.score
            self._load_notes += pm.notes
            self.header_items = pm.header_items
            self._sync_meta_from_items()
            barlines = pm.barlines
            self.barlines = barlines
            self.sx, self.sy = pm.sx, pm.sy
            self.by_sys = pm.by_sys
            self.staff_groups = pm.staff_groups
            self.staff_left = pm.staff_left
            self.staff_clefs = pm.staff_clefs
            self.staff_parts = pm.staff_parts
            self._title_credits = pm.title_credits
        except Exception:
            import traceback
            tb = traceback.format_exc()
            print(tb)
            QMessageBox.critical(self, "인식 오류",
                                 "악보 인식 중 오류가 발생했습니다.\n\n" + tb[-1500:])
            self._set_status("인식 실패")
            return

        self._undo.clear()
        self.selected = None
        self.selected_suggestion = None
        self.image_panel.set_image(cv_bgr_to_qpixmap(img), arr=img)
        self._compute_diff_marks()             # [v4.10] 붓 후보 자리(원본에만 있는 잉크)
        self._refresh_ink_marks()
        self.image_panel.set_highlight(None)
        n_pages = len(getattr(self, "doc_pages", []) or []) or 1
        page_txt = f" · {n_pages}페이지" if n_pages > 1 else ""
        try:
            # 렌더링·가사 패널 채우기는 인식 자체와 별개 단계다 — 여기서
            # 예외가 나더라도(예: 그리기 관련 환경 차이) 상태 메시지가
            # "인식 중…"에 영원히 머무르지 않도록 finally 로 반드시
            # 최종 상태를 갱신한다(실측: 렌더는 됐는데 상태줄만 안 풀리는
            # 제보가 있어 방어적으로 강화).
            self._rerender()
            self.on_select(None)
        except Exception:
            import traceback
            tb = traceback.format_exc()
            print(tb)
            self._set_status(f"불러옴(일부 표시 오류): {path}{page_txt}")
            self.setWindowTitle(f"♪ {APP_NAME} v{APP_VERSION} — {path}")
            QMessageBox.warning(
                self, "표시 오류",
                "인식은 끝났지만 화면 표시 중 오류가 있었습니다.\n"
                "MusicXML 내보내기 등 다른 기능은 정상 작동할 수 있습니다.\n\n"
                + tb[-1000:])
            return
        # 위젯 크기가 확정된 뒤 화면 맞춤
        n_pages_now = len(getattr(self, "doc_pages", []) or [])
        self.page_nav.set_pages(n_pages_now)
        if n_pages_now > 1:
            # 다중 페이지 문서는 처음부터 1페이지에 '잠긴' 상태로 시작한다
            # — 사용자 요청: 페이지별로만 뷰가 제한되는 방식을 기본으로.
            QTimer.singleShot(60, lambda: self.on_goto_page(0))
        else:
            QTimer.singleShot(60, self.fit_both)
        title = (getattr(self, "_title_credits", {}) or {}).get("title")
        credits_ = (getattr(self, "_title_credits", {}) or {}).get("credits")
        credit_txt = f" · {credits_}" if credits_ else ""
        self._set_status(
            f"불러옴: {path} ({img.shape[1]}x{img.shape[0]}){page_txt}{credit_txt}")
        self.setWindowTitle(
            f"♪ {APP_NAME} v{APP_VERSION} — {title} ({path})" if title
            else f"♪ {APP_NAME} v{APP_VERSION} — {path}")
        # [안내 팝업 제거 — 실사용 요청] 불러올 때마다 뜨던 안내 창은 없애고, 내용은
        # "정보" 메뉴에서 볼 수 있게 보관한다(상태 표시줄에 건수만 알린다).
        info = list(getattr(doc, "warnings", None) or []) + list(getattr(self, "_load_notes", []))
        self._last_load_info = info
        if info:
            self._set_status(self.status.text()
                             + f"   · 인식 안내 {len(info)}건 — '정보'에서 확인")
        self._apply_pending_project()                 # [v2.50] 이어서 편집

    def _rerender(self):
        """모델 → 원본좌표 배치 → 벡터 렌더(원본 조각 없이) → 우측 갱신."""
        self.layout_obj = sl.layout_score_world(
            self.score, self.by_sys, self.sx, self.sy, self.barlines or None,
            staff_groups=self.staff_groups, staff_left_by_line=self.staff_left,
            staff_clefs=getattr(self, "staff_clefs", None),
            staff_parts=getattr(self, "staff_parts", None))
        self.layout_obj.header_items = self.header_items
        book = getattr(self, "_book_state", None)
        if book is not None and book.pages:
            # [v2.00] 지금 페이지만 그린다(곡 전체 좌표에서 그 페이지 구간) — 몇 페이지든 한 페이지만큼만
            pi = min(getattr(self, "_cur_page", 0), len(book.pages) - 1)
            ow, oh = book.page_sizes[pi]
            page_dy = book.pages[pi].dy_orig
            world_w, world_h = max(w for w, _h in book.page_sizes), book.total_h
        else:
            oh, ow = self.original_bgr.shape[:2]
            page_dy = 0.0
            world_w, world_h = ow, oh
        # 슈퍼샘플링 배율(F)을 그대로 곱하면, 페이지가 많은 모음곡 등에서
        # 캔버스 높이가 수만~수십만 px 로 치솟아 Windows GDI/QPixmap 의
        # 실질적 크기 한계(약 32767px)를 넘는다 — 이 경우 이미지 데이터가
        # 깨져 화면에 "지지직" 노이즈처럼 나온다(실측: 12페이지 모음곡).
        # 안전 한도(28000px) 아래로 F를 자동으로 낮춰 항상 유효한 크기를
        # 유지한다(작은 악보는 영향 없음 — F=4 그대로). 원본 이미지 자체가
        # 이미 28000px 를 넘는 극단적인 경우(예: 12페이지 이상 모음곡)는
        # F=1(무압축)로도 한계를 못 벗어날 수 있다 — 이건 이 프로젝트의
        # 대상 범위(2~6페이지 단일 곡) 밖이라 임시방편이며, 근본 해결은
        # 페이지별 독립 캔버스(페이징 UI)로의 재설계가 필요하다.
        SAFE_CANVAS_H = 28000
        eff_f = max(1, min(F, SAFE_CANVAS_H // max(1, oh)))
        # [실사용 검증 후 추가] 위 SAFE_CANVAS_H는 "세로 픽셀 수"만 보므로,
        # 세로는 안전 범위 안인데 가로*세로 전체 픽셀 수(=메모리 사용량)가
        # 큰 악보(예: 32마디 여러 시스템, 2987x4418)에서는 걸러내지 못했다
        # — F=4로 최종 캔버스가 1억 2천만~2억 픽셀(RGB만 600MB 이상)이
        # 되어 MemoryError로 이어지는 사고가 실사용 중 있었다(원본이
        # 높이(oh) 기준으로는 28000px에 한참 못 미쳤음에도 발생). 최종
        # 총 픽셀 수 자체에 별도 상한을 둬서, 가로가 넓은 악보도 함께
        # 방어한다(작은 악보는 영향 없음).
        MAX_TOTAL_PIXELS = 60_000_000  # RGB 3바이트 기준 최종 약 180MB
        max_f_by_area = max(1, int((MAX_TOTAL_PIXELS / max(1, ow * oh)) ** 0.5))
        eff_f = max(1, min(eff_f, max_f_by_area))
        self.score_panel.release_render()     # [v1.31] 새 그림을 만들기 전에 이전 그림을 놓아준다
        # [v1.31] 메모리가 모자라면(학교 PC 등) 해상도를 한 단계씩 낮춰 다시 그린다 — 오류 창 대신
        while True:
            try:
                draw_lay = sl.page_view(self.layout_obj, page_dy) if book is not None else self.layout_obj
                pil, left_margin = sr.render(draw_lay, ow, oh, F=eff_f, header_strips=None,
                                             return_margin=True, lyric_y_off=page_dy)
                self.score_panel.layout_obj = self.layout_obj
                self.score_panel.set_render(pil, world_w, world_h, left_margin=left_margin,
                                            render_f=eff_f, img_dy=page_dy)
                self._refresh_ink_marks()          # [v2.11] 편집으로 해결된 미인식 표시는 지운다
                del pil
                break
            except MemoryError:
                pil = None
                self.score_panel.release_render()
                import gc
                gc.collect()
                if eff_f <= 1:
                    raise
                eff_f -= 1
                self._low_mem_f = eff_f
        if getattr(self, "_low_mem_f", None):
            self._set_status(f"메모리가 부족해 편집 악보를 낮은 해상도({self._low_mem_f}배)로 그렸습니다 — "
                             "다른 프로그램을 닫으면 다음에 다시 그릴 때 원래대로 돌아옵니다")
            self._low_mem_f = None
        self._refresh_summary()

    # ------------------------------------------------------------- 선택
    def on_select(self, glyph, ctrl=False):
        """요소 클릭 처리. ctrl=True 면 [3-B] 다중 선택 토글:
        이미 선택된 요소를 다시 Ctrl+클릭하면 선택에서 빠지고, 새 요소를
        Ctrl+클릭하면 선택에 더해진다. 일반(Ctrl 없는) 클릭은 항상 그
        요소 하나만 남기고 다중 선택을 초기화한다 — '새로 고르기' 시작.
        """
        self.selected_suggestion = None
        if glyph is not None and getattr(glyph, "suggestion", None) is not None:
            # [v1.40] 제안 음표 선택 — Enter 넣기 / Delete 무시
            sug = glyph.suggestion
            self.selected_suggestion = sug
            if sug.get("kind") == "grace":
                # [v1.43] 꾸밈음 표시는 클릭만 하면 바로 넣는다(잘못이면 선택 후 삭제)
                self._accept_suggestion()
                return
            self.selected, self.selected_multi = None, []
            self.score_panel.set_selection(glyph.cell)
            self.score_panel.set_multi_selection([])
            x0, y0, x1, y1 = sug["blob"]["box"]
            self.image_panel.set_highlight((x0 * self.sx, y0 * self.sy, (x1 - x0) * self.sx, (y1 - y0) * self.sy))
            self.image_panel.set_highlight_multi([])
            self._set_status(f"제안 음표: {sug['text']}  —  Enter로 넣기, Delete로 무시")
            return
        if glyph is None:
            if ctrl:
                return   # 빈 곳 Ctrl+클릭은 무시(실수로 선택 잃지 않게)
            self.selected = None
            self.selected_multi = []
            self._refresh_selection_ui()
            return

        e = glyph.element
        if ctrl:
            group = list(self.selected_multi) if self.selected_multi else (
                [self.selected] if self.selected is not None else [])
            if e in group:
                group.remove(e)
            else:
                group.append(e)
            if len(group) <= 1:
                self.selected = group[0] if group else None
                self.selected_multi = []
            else:
                self.selected = e
                self.selected_multi = group
        else:
            self.selected = e
            self.selected_multi = []
        self._refresh_selection_ui()

    def on_marquee_select(self, glyphs: list):
        """[드래그 다중선택] 사각형 안의 요소들을 한 번에 선택한다.
        Ctrl+클릭을 여러 번 반복할 필요 없이, 드래그 한 번으로 tie/slur/
        triplet 편집에 필요한 2개 이상의 음표를 바로 고를 수 있다."""
        elements = []
        seen = set()
        for g in glyphs:
            e = g.element
            if id(e) not in seen:
                seen.add(id(e))
                elements.append(e)
        if not elements:
            return
        if len(elements) == 1:
            self.selected = elements[0]
            self.selected_multi = []
        else:
            self.selected = elements[-1]
            self.selected_multi = elements
        self._refresh_selection_ui()

    def _refresh_selection_ui(self):
        """selected/selected_multi 상태를 패널·하단 바에 그대로 반영."""
        if self.selected_multi:
            self._refresh_selection_ui_multi()
            return
        e = self.selected
        if e is None:
            self.score_panel.set_selection(None)
            self.score_panel.set_multi_selection([])
            self.image_panel.set_highlight(None)
            self.image_panel.set_highlight_multi([])
            self.bottom_bar.set_selected_text("요소를 클릭하면 여기에 정보가 표시됩니다")
            self.bottom_bar.reflect_selection(None, None)
            self.bottom_bar.reflect_articulations([])
            self.bottom_bar.set_multi_mode(0, False)
            return
        g = self._glyph_of(e)
        self.score_panel.set_selection(self._hl_rect(g) if g else None)
        self.score_panel.set_multi_selection([])
        self.image_panel.set_highlight_multi([])
        m = self._find_measure(e)
        mnum = m.number if m else "?"
        dot = ("겹점" if getattr(e, "double_dot", False) else "점") if e.dotted else ""
        if e.kind == "note":
            pitch = f"{e.step}{_ACC.get(e.alter, '')}{e.octave}"
            txt = f"마디 {mnum} · {dot}{_DUR_NAME.get(e.duration_base)}음표 · 음정 {pitch}"
        else:
            txt = f"마디 {mnum} · {dot}{_DUR_NAME.get(e.duration_base)}쉼표"
        group_base = None
        if m:
            grp = ed.group_of(m, e)
            if len(grp) >= 2:
                txt += f"  · 빔그룹 {len(grp)}개"
                group_base = e.duration_base if e.duration_base in (8, 16, 32) else None
        self.bottom_bar.set_selected_text(txt)
        self.bottom_bar.reflect_selection(e, group_base)
        self.bottom_bar.reflect_articulations([e], self._find_measure)
        self.bottom_bar.set_multi_mode(0, False)
        if e.src_bbox:
            x, y, w, h = e.src_bbox
            self.image_panel.set_highlight(
                (x * self.sx, y * self.sy, w * self.sx, h * self.sy))
        else:
            self.image_panel.set_highlight(None)

    def _refresh_selection_ui_multi(self):
        """[3-B] 2개 이상 선택됐을 때의 표시: 마지막 클릭 요소는 초록,
        나머지는 파랑으로 양쪽 패널에 동시 강조하고, 하단 바는 '공통으로
        적용 가능한' 편집만 켠다."""
        group = self.selected_multi
        primary = self.selected if self.selected in group else group[-1]
        others = [x for x in group if x is not primary]

        g_primary = self._glyph_of(primary)
        self.score_panel.set_selection(self._hl_rect(g_primary) if g_primary else None)
        self.score_panel.set_multi_selection(
            [self._hl_rect(g) for o in others if (g := self._glyph_of(o))])

        if primary.src_bbox:
            x, y, w, h = primary.src_bbox
            self.image_panel.set_highlight((x * self.sx, y * self.sy,
                                            w * self.sx, h * self.sy))
        else:
            self.image_panel.set_highlight(None)
        multi_bboxes = []
        for o in others:
            if o.src_bbox:
                x, y, w, h = o.src_bbox
                multi_bboxes.append((x * self.sx, y * self.sy, w * self.sx, h * self.sy))
        self.image_panel.set_highlight_multi(multi_bboxes)

        n_notes = sum(1 for x in group if x.kind == "note")
        kind_txt = "음표" if n_notes == len(group) else \
            ("쉼표" if n_notes == 0 else "음표/쉼표 혼합")
        self.bottom_bar.set_selected_text(
            f"{len(group)}개 선택됨 ({kind_txt}) · Ctrl+클릭으로 추가/제외")
        self.bottom_bar.reflect_multi_selection(group)
        self.bottom_bar.set_multi_mode(len(group), n_notes > 0)   # [v2.20] 음표가 하나라도 있으면 음높이 켬
        self.bottom_bar.reflect_articulations(group, self._find_measure)

    def _glyph_of(self, element):
        if not self.layout_obj:
            return None
        for g in self.layout_obj.glyphs:
            if g.element is element:
                return g
        return None

    # ------------------------------------------------------------- 편집
    def _find_measure(self, element):
        for m in self.score.measures:
            if element in m.elements:
                return m
        return None

    def _snapshot(self):
        self._dirty = True                            # [v2.50] 저장하지 않은 편집이 있다
        self._undo.append(copy.deepcopy(self.score))
        if len(self._undo) > 60:
            self._undo.pop(0)

    def _edit(self, fn, *args):
        """[3-B] 다중 선택 중이면 같은 편집을 선택된 요소 전부에 적용한다
        (스냅샷은 한 번만 찍어 실행취소도 한 번에 묶인다)."""
        targets = self.selected_multi or ([self.selected] if self.selected is not None else [])
        if not targets:
            return
        self._snapshot()
        for t in targets:
            fn(t, *args)
        self._after_edit_targets(targets)

    def _edit_group_cycle(self):
        if self.selected_multi:
            # [v1.44] 여러 음을 고르면 빔으로 묶기(이미 한 빔이면 풀기)
            m = self._find_measure(self.selected_multi[0])
            if m is None or any(self._find_measure(e) is not m for e in self.selected_multi):
                self._set_status("빔으로 묶기: 같은 마디의 음표만 묶을 수 있습니다")
                return
            self._snapshot()
            r = ed.toggle_beam(m, self.selected_multi)
            if r == "unsupported":
                self._undo.pop()
                self._set_status("빔으로 묶기: 같은 보표의 8분음표 이하 음표를 2개 이상 고르세요")
                return
            self._after_edit_targets(list(self.selected_multi))
            self._set_status("빔으로 묶었습니다" if r == "joined" else "빔을 풀었습니다")
            return
        if self.selected is None:
            return
        m = self._find_measure(self.selected)
        if m:
            cycle = [8, 16, 32]
            cur = self.selected.duration_base
            next_base = cycle[(cycle.index(cur) + 1) % 3] if cur in cycle else 8
            self._snapshot()
            ed.set_beam_group_duration(m, self.selected, next_base, dotted=False)
            self._after_edit(self.selected)

    def keyPressEvent(self, event):
        """[셈여림] MuseScore와 같은 관례: '<' 크레셴도, '>' 디크레셴도."""
        if getattr(self, "_loading", False):
            return
        txt = event.text()
        if txt == "<":
            self._dynamic_action("cresc")
            return
        if txt == ">":
            self._dynamic_action("decresc")
            return
        super().keyPressEvent(event)

    def _tuplet_action(self, key):
        """[잇단음표 일반화] 메뉴/단축키 처리. key는 내부 키(화면 문구 아님)."""
        if self.score is None:
            return
        targets = self.selected_multi or ([self.selected] if self.selected is not None else [])
        if not targets:
            self._set_status("먼저 음표를 선택해 주세요")
            return
        name = TUPLET_NAMES.get(key, "잇단음표")
        # [사용성 개선 — 실사용 검증] 빔으로 묶인 음표 '하나'만 골라 잇단을 누르면 예전엔 그 음을
        # 셋으로 쪼갰다(세 음을 하나씩 눌러 표시하려다 음표가 9개로 불어나는 혼란). 빔 묶음은
        # 그 자체가 잇단음표 그룹인 경우가 대부분이라, 쪼개지 않고 빔 묶음 전체에 표시한다.
        if len(targets) == 1:
            m0 = self._find_measure(targets[0])
            bm = ed.beam_members(m0, targets[0]) if m0 is not None else []
            if bm:
                targets = bm
        if len(targets) == 1 and key != "clear" and not targets[0].in_triplet:
            if targets[0].chord_group != -1:
                self._set_status("화음은 음표를 여러 개 선택해 잇단음표 표시만 할 수 있습니다")
                return
            if ed.tuplet_split_spec(targets[0], key) is None:
                self._set_status(f"이 음표 길이로는 {name}을(를) 만들 수 없습니다"
                                 f"(예: 2·4잇단은 점음표에서)")
                return
        m = self._find_measure(targets[0])
        if m is None:
            return
        self._snapshot()
        min_gap = self.layout_obj.staff_space * 1.4 if self.layout_obj else None
        result, focus = ed.apply_tuplet(self.score, m, targets, key, min_gap=min_gap)
        msg = {"split": f"{name}으로 쪼갰습니다 — 음높이는 ↑↓로 고쳐 주세요",
               "marked": f"{name} 표시를 했습니다", "cleared": f"{name} 표시를 해제했습니다"}
        self._set_status(msg.get(result, ""))
        if result == "split":
            self._after_edit(focus)
        else:
            self._after_edit_targets(targets)

    def _capture_window(self):
        """프로그램 창 전체(제목 표시줄 포함)를 캡처해 클립보드에 복사하고 사진 폴더에 저장."""
        pm = None
        try:
            scr = self.screen()
            fg = self.frameGeometry()
            if scr is not None:
                g = scr.geometry()
                pm = scr.grabWindow(0, fg.x() - g.x(), fg.y() - g.y(), fg.width(), fg.height())
        except Exception:
            pm = None
        if pm is None or pm.isNull() or pm.width() < 50:
            pm = self.grab()                       # 화면 캡처가 안 되는 환경: 창 내용만
        QApplication.clipboard().setPixmap(pm)
        base = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation) or str(Path.home())
        folder = Path(base) / "콩나물지기_캡처"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / time.strftime("캡처_%Y%m%d_%H%M%S.png")
            pm.save(str(path), "PNG")
            self._set_status(f"화면을 캡처했습니다 — 클립보드에 복사됨(Ctrl+V로 붙여넣기), 저장: {path}")
        except Exception as ex:
            self._set_status(f"화면을 캡처해 클립보드에 복사했습니다(파일 저장 실패: {ex})")

    def _dynamic_action(self, key):
        """[셈여림] 메뉴/단축키 처리. key는 내부 키(화면 문구 아님)."""
        if self.score is None:
            return
        targets = self.selected_multi or ([self.selected] if self.selected is not None else [])
        if key == "clear_auto":
            self._snapshot()
            n = ed.clear_auto_dynamics(self.score)
            if targets:
                self._after_edit_targets(targets)
            else:
                self._rerender()
            self._set_status(f"자동 인식 셈여림·쐐기 {n}개를 지웠습니다")
            return
        if not targets:
            self._set_status("먼저 음표를 선택해 주세요")
            return
        if key in ed.DYNAMIC_KEYS:
            self._snapshot()
            ed.set_dynamic(targets, key)
        elif key in ed.HAIRPIN_KEYS:
            name = "크레셴도" if key == "cresc" else "디크레셴도"
            if not ed.can_toggle_hairpin(targets) and not ed.hairpins_covering(self.score, targets, key):
                self._set_status(f"{name}를 만들려면 같은 보표에서 2개 이상 선택해 주세요"
                                 f"(지우려면 그 쐐기 안의 음표를 고르고 다시 누르기)")
                return
            self._snapshot()
            res = ed.toggle_hairpin(self.score, targets, key)
            self._set_status({"added": f"{name}를 넣었습니다", "removed": f"{name}를 지웠습니다"}.get(res, ""))
        elif key == "clear":
            self._snapshot()
            ed.clear_dynamics(self.score, targets)
        else:
            return
        self._after_edit_targets(targets)

    def _toggle_bond(self):
        """[v2.31] 짝을 너그럽게 고르고(같은 자리 두 성부를 함께 골라도), 새로 생긴 이음줄은 원본 그림에서
        그 곡선을 찾아 그대로 쓴다(원본이 정한다 — 못 찾으면 계산)."""
        pairs = ed.bond_pairs(self.selected_multi)
        if not pairs:
            self._set_status("이음: 같은 보표의 음표 두 개를 고르세요")
            return
        sel = [e for p in pairs for e in p]
        before = set(id(a) for a in self.score.arcs)
        self._snapshot()
        ed.toggle_bond(self.score, sel)
        new = [a for a in self.score.arcs if id(a) not in before]
        traced = 0
        for a in new:
            if self._trace_added_arc(a):
                traced += 1
        self._after_edit_targets(sel)
        if new:
            self._set_status("이음줄을 넣었습니다" + (" — 원본의 곡선 자리에 맞췄습니다" if traced else ""))
        else:
            self._set_status("이음줄을 지웠습니다")

    def _trace_added_arc(self, arc):
        """원본 그림에서 두 음 사이의 곡선 잉크를 찾아 arc.src_curve로(그 음들은 원본 그대로일 때만 쓰임)."""
        try:
            from omr_bridge.ink_check import trace_arc_between
        except Exception:
            return False
        m = self._find_measure(arc.start)
        if m is None or not arc.start.src_bbox or not arc.stop.src_bbox:
            return False
        dy = 0.0
        if getattr(self, "_book_state", None) is not None:
            pg = getattr(m, "page", 1)
            img = self._book_job.page_image(pg)
            dy = self._book_state.pages[pg - 1].dy_orig
        else:
            img = self.original_bgr
        grp = (self.staff_groups or {}).get(m.system) or []
        sn = arc.start.staff_no
        if sn - 1 >= len(grp):
            return False
        ys = [v - dy for v in grp[sn - 1]]
        # 찾을 쪽(조판 관례): 같은 보표에 다른 성부가 있으면 기둥 쪽, 한 성부면 머리 쪽(기둥 반대)
        g1 = self._glyph_of(arc.start)
        prefer_above = None
        if g1 is not None and g1.stem_up is not None:
            poly = any(e.staff_no == sn and e.voice_no != arc.start.voice_no and e.kind == "note"
                       for e in m.elements)
            prefer_above = bool(g1.stem_up) if poly else (not g1.stem_up)
        heads = [e.src_bbox for mm in self.score.measures if mm.system == m.system
                 for e in mm.elements if e.staff_no == sn and e.kind == "note" and e.src_bbox]
        # 이웃 보표와의 가운데까지만 찾는다(곡 좌표)
        lo_lim = (grp[sn - 2][4] + grp[sn - 1][0]) / 2.0 if sn >= 2 else None
        hi_lim = (grp[sn - 1][4] + grp[sn][0]) / 2.0 if sn < len(grp) else None
        cv = trace_arc_between(img, arc.start, arc.stop, ys, dy, prefer_above, heads, limits=(lo_lim, hi_lim))
        if cv is None:
            return False
        arc.src_poly = list(cv)                   # 원본 잉크를 따라간 점 목록(곡 좌표)
        return True

    def _toggle_triplet(self):
        targets = self.selected_multi or ([self.selected] if self.selected is not None else [])
        if not ed.can_toggle_triplet(targets):
            return
        self._snapshot()
        if len(targets) >= 2:
            # [동작 재설계 — 실사용 요청] 여러 개 선택 시 개별 반전이
            # 아니라 "전부 같은 상태로 맞추는" 정규화 토글을 쓴다 —
            # 뒤섞인 상태(일부만 셋잇단)면 우선 전부 끄고, 이미 다
            # 같으면 다같이 반전한다.
            ed.toggle_triplet_group(targets)
        else:
            ed.toggle_triplet(targets[0])
        self._after_edit_targets(targets)

    def _flip_stems(self):
        """[v2.25] 고른 음표(여러 개 가능)의 기둥 방향을 뒤집는다 — 화음·빔 묶음은 묶음 전체."""
        targets = [t for t in (self.selected_multi or ([self.selected] if self.selected is not None else []))
                   if t.kind == "note"]
        if not targets or self.layout_obj is None:
            self._set_status("기둥 뒤집기: 음표를 고른 뒤 Shift+X")
            return
        cur = {}
        for g in self.layout_obj.glyphs:
            if g.element.kind == "note":
                up = g.stem_up
                if up is None:
                    sysL = next((s_ for s_ in self.layout_obj.systems
                                 if s_.index == g.measure.system and s_.staff_no == g.element.staff_no), None)
                    up = (g.y > sysL.staff_top + 2 * sysL.staff_space) if sysL is not None else True
                cur[id(g.element)] = up
        self._snapshot()
        done = set()
        for t in targets:
            m = self._find_measure(t)
            if m is None or id(t) in done:
                continue
            ed.flip_stems(m, [t], cur)
            done.update(id(x) for x in m.elements if getattr(x, "stem_override", None))
        self._after_edit_targets(targets)
        self._set_status("기둥 방향을 바꿨습니다 (Shift+X를 다시 누르면 반대로)")

    def _add_grace(self):
        """[v2.21] 고른 음 앞에 앞꾸밈음을 넣고, 넣은 꾸밈음을 골라 둔다(바로 ↑↓로 음높이 맞춤)."""
        if self.selected_multi or self.selected is None or self.selected.kind != "note" \
                or self.selected.is_grace or self.layout_obj is None:
            self._set_status("앞꾸밈음: 음표 하나를 고른 뒤 누르세요")
            return
        m = self._find_measure(self.selected)
        if m is None:
            return
        self._snapshot()
        space = self.layout_obj.staff_space / max(1e-6, getattr(self, "sy", 1.0))
        g = ed.insert_grace(self.score, m, self.selected, space)
        self._after_edit(g)
        self._set_status("앞꾸밈음을 넣었습니다 — ↑↓로 음높이를 맞추세요")

    def _toggle_articulation(self, kind):
        """[v2.20] 고른 음표들(화음은 한 번만)에 아티큘레이션 붙이기/떼기. 붙이는 쪽은 기둥 반대쪽
        (기둥이 위면 아래, 아래면 위). 고른 것 중 하나라도 이미 있으면 모두 뗀다(켜기/끄기 버튼처럼)."""
        targets = [t for t in (self.selected_multi or ([self.selected] if self.selected is not None else []))
                   if t.kind == "note"]
        if not targets or self.layout_obj is None:
            return
        space = self.layout_obj.staff_space / max(1e-6, getattr(self, "sy", 1.0))
        self._snapshot()
        has_any = any(a["kind"] == kind for t in targets for a in (t.articulations or []))
        seen = set()
        for t in targets:
            m = self._find_measure(t)
            if m is None:
                continue
            key = (id(m), t.chord_group) if t.chord_group != -1 else (id(m), id(t))
            if key in seen:
                continue
            seen.add(key)
            members = [t] if t.chord_group == -1 else [e for e in m.elements if e.chord_group == t.chord_group]
            mine = any(a["kind"] == kind for e in members for a in (e.articulations or []))
            if has_any != mine:
                continue                      # 끄는 중이면 없는 음은 건너뛰고, 켜는 중이면 있는 음은 건너뛴다
            gs = [g for g in (self._glyph_of(e) for e in members) if g is not None]
            above = True
            if gs:
                g0 = gs[0]
                sysL = next((s for s in self.layout_obj.systems
                             if s.index == m.system and s.staff_no == t.staff_no), None)
                if g0.stem_up is not None:
                    above = not g0.stem_up
                elif sysL is not None:
                    mid = sysL.staff_top + 2 * sysL.staff_space
                    above = (sum(g.y for g in gs) / len(gs)) <= mid      # 높은 음 → 기둥 아래 → 기호 위
            ed.toggle_articulation(m, t, kind, space=space, above=above)
        self._after_edit_targets(targets)
        names = {"accent": "악센트", "staccato": "스타카토", "tenuto": "테누토", "marcato": "마르카토"}
        self._set_status(f"{names.get(kind, kind)} {'지웠습니다' if has_any else '붙였습니다'}")

    def _cycle_acc(self, el):
        """[v1.43] 임시표 ♯ → ♭ → ♮ → 없음(조표대로) 순환 — '없음'은 그 마디의 조표로 음높이를 돌린다."""
        m = self._find_measure(el)
        ed.cycle_accidental(el, self.score.key_at(m.number) if m is not None else self.score.fifths)

    def _accept_suggestion(self):
        """[v1.40] 선택한 제안 음표를 악보에 넣는다(실행 취소 가능)."""
        sug = getattr(self, "selected_suggestion", None)
        if sug is None:
            return
        from omr_bridge.ink_check import accept_suggestion
        self._snapshot()
        el = accept_suggestion(self.score, sug)
        self.selected_suggestion = None
        self._refresh_ink_marks()
        self._after_edit(el)
        self._set_status(f"넣음: {sug['text']}")

    def _dismiss_suggestion(self):
        sug = getattr(self, "selected_suggestion", None)
        if sug is None:
            return
        from omr_bridge.ink_check import dismiss_suggestion
        self._snapshot()
        dismiss_suggestion(self.score, sug)
        self.selected_suggestion = None
        self._refresh_ink_marks()
        self._after_edit(None)

    def _ink_resolved(self, b):
        """[v2.11] 편집으로 해결된 미인식 잉크인가 — 원본의 주황 점선을 지운다.
        · 이음줄·붙임줄 조각: 그 자리를 지나는 이음줄이 편집 악보에 있으면
        · 점: 바로 왼쪽 같은 높이의 음표에 점이 붙어 있으면"""
        lay = getattr(self, "layout_obj", None)
        if lay is None:
            return False
        x0, y0, x1, y1 = b["box"]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        il = b.get("il", 20.0)
        for a in getattr(lay, "arcs", []):
            lo, hi = min(a.x1, a.x2) - 0.5 * il, max(a.x1, a.x2) + 0.5 * il
            top = min(a.y1, a.y2) - 2.5 * il
            bot = max(a.y1, a.y2) + 2.5 * il
            if lo <= cx <= hi and top <= cy <= bot:
                return True
        # [v2.24] 노래 파트 보표 아래(가사 자리)의 잉크는 가사 글자·늘임줄('—')이다 — 표시하지 않는다
        sc_ = self.score
        parts_ln = (getattr(sc_, "staff_parts", None) or {}).get(b.get("line")) or []
        sn_ = b.get("staff_no", 1)
        pidx = parts_ln[sn_ - 1] if 0 < sn_ <= len(parts_ln) else None
        names_ = getattr(sc_, "part_names", None) or []
        grp_ = (getattr(self, "staff_groups", None) or {}).get(b.get("line")) or []
        if pidx is not None and pidx < len(names_) and 0 < sn_ <= len(grp_):
            from omr_bridge.merge import _is_vocal
            if _is_vocal(names_[pidx]) and cy > grp_[sn_ - 1][4] + 0.5 * il:
                return True
        # [v2.24] 그 자리에 머리가 그려져 있으면(화음 열 정렬 등으로 바로잡힌 머리) 해결
        for g in lay.glyphs:
            if g.element.kind == "note" and abs(g.x - cx) <= 0.8 * il and abs(g.y - cy) <= 0.6 * il:
                return True
        # [v2.24] 박자표를 추론해 넣은 마디의 첫머리(놓친 박자표 기호 자리)면 해결
        for mb in getattr(lay, "measure_boxes", []):
            if getattr(mb.measure, "time_sig_change", None):
                x0b, y0b, x1b, y1b = mb.bbox
                if x0b - 1.0 * il <= cx <= x0b + 3.0 * il and y0b - il <= cy <= y1b + il:
                    return True
        # [v3.00] 기타 코드 글자 자리(원본 위치가 있는 코드)의 잉크는 코드 이름의 일부
        if b["kind"] in ("other", "dot"):
            for m_ in (self.score.measures if self.score else []):
                for e_ in m_.elements:
                    h_ = getattr(e_, "harmony", None)
                    if h_ and h_.get("x") is not None and h_.get("y") is not None and \
                            abs(h_["x"] - cx) <= 3.0 * il and abs(h_["y"] - cy) <= 1.2 * il:
                        return True
        # [v2.70] 숨은 다른 성부 음 표시: 그 자리 가까이에 그 방향 기둥의 음이 생겼으면(원본에서 넣기 등) 해결
        if b["kind"] == "hidden":
            for g in lay.glyphs:
                e = g.element
                if e.kind == "note" and e.src_bbox and abs((e.src_bbox[0] + e.src_bbox[2] / 2.0) - b.get("x", cx)) \
                        <= 1.5 * e.src_bbox[2] and abs(g.y - cy) <= 1.5 * il and \
                        (getattr(e, "stem_dir_img", None) == b.get("dir") or
                         (g.stem_up is not None and g.stem_up == (b.get("dir") == "up"))) and \
                        getattr(e, "from_original", False):
                    return True
            return False
        # [v2.56] 빔 묶음의 음들 사이·빔 높이의 납작한 잉크 = 이제 그려지는 빔(빔을 바로잡은 뒤 남던 주황 점선)
        if b["kind"] == "other" and (x1 - x0) > 1.5 * (y1 - y0) and (x1 - x0) >= 1.5 * il:
            groups_ = {}
            for g in lay.glyphs:
                e = g.element
                if e.kind == "note" and e.beam_group not in (None, -1):
                    groups_.setdefault((id(g.measure), e.staff_no, e.beam_group), []).append(g)
            for mem in groups_.values():
                if len(mem) < 2:
                    continue
                gx0, gx1 = min(g.x for g in mem), max(g.x for g in mem)
                gy0, gy1 = min(g.y for g in mem), max(g.y for g in mem)
                if gx0 - il <= cx <= gx1 + il and gy0 - 5 * il <= cy <= gy1 + 5 * il:
                    return True
        if b["kind"] == "dot":
            for g in lay.glyphs:
                e = g.element
                if e.kind == "note" and e.dotted and e.src_bbox:
                    hx1 = e.src_bbox[0] + e.src_bbox[2]
                    if hx1 <= cx <= hx1 + 1.8 * il and abs(cy - (e.src_bbox[1] + e.src_bbox[3] / 2)) <= 0.8 * il:
                        return True
        return False

    def _refresh_ink_marks(self):
        """원본 패널의 주황 점선: 아직 남은 제안 음표 자리 + 그 밖의 미인식 기호."""
        if not hasattr(self.image_panel, "set_ink_marks") or self.score is None:
            return
        sx, sy = getattr(self, "sx", 1.0), getattr(self, "sy", 1.0)
        boxes = [q["blob"]["box"] for q in (getattr(self.score, "suggestions", None) or [])]
        boxes += [b["box"] for b in (getattr(self.score, "ink_blobs", None) or [])
                  if b["kind"] in ("dot", "other", "hidden") and b.get("measure") is not None      # "text"는 글자 — 제외
                  and not self._ink_resolved(b)]
        # [v4.10] 원본과 겹쳐 본 차이(원본에만 있는 잉크)도 붓 후보 자리로 — 이미 표시된 자리와 겹치면 하나만
        for d in getattr(self, "_diff_marks", None) or []:
            if not any(d[0] < b[2] and b[0] < d[2] and d[1] < b[3] and b[1] < d[3] for b in boxes):
                boxes.append(d[:4])
        self.image_panel.set_ink_marks([(x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy)
                                        for x0, y0, x1, y1 in boxes])

    def _delete(self):
        if getattr(self, "selected_suggestion", None) is not None:
            self._dismiss_suggestion()
            return
        targets = self.selected_multi or ([self.selected] if self.selected is not None else [])
        if not targets:
            return
        self._snapshot()
        touched = []
        for t in targets:
            m = self._find_measure(t)
            if m:
                ed.delete_element(m, t)
                touched.append(m)
        self.selected = None
        self.selected_multi = []
        self._auto_repair(touched)            # [v2.11] 지운 마디의 남은 제안 음표를 다시 넣어 본다
        self._after_edit(None)

    def _insert(self):
        if self.selected_multi or self.selected is None:
            return
        m = self._find_measure(self.selected)
        if m:
            self._snapshot()
            min_gap = self.layout_obj.staff_space * 1.4 if self.layout_obj else None
            new = ed.insert_after(m, self.selected, min_gap=min_gap)
            self._after_edit(new)

    def _insert_before(self):
        if self.selected_multi or self.selected is None:
            return
        m = self._find_measure(self.selected)
        if m:
            self._snapshot()
            min_gap = self.layout_obj.staff_space * 1.4 if self.layout_obj else None
            new = ed.insert_before(m, self.selected, min_gap=min_gap)
            self._after_edit(new)

    # --------------------------------------------------------- [Tier2] 편집
    def _add_chord_note(self):
        """선택한 음 위에 3도 음정을 추가해 화음으로 만든다."""
        if self.selected_multi or self.selected is None:
            return
        m = self._find_measure(self.selected)
        if m:
            self._snapshot()
            new = ed.add_chord_note(m, self.selected, step_delta=2)
            if new is not None:
                self._after_edit(new)
            else:
                self._set_status("화음 추가는 음표에서만 가능합니다(쉼표 선택됨)")

    def _remove_chord_note(self):
        """선택한 음을 화음에서 제거(단독음이면 그냥 삭제와 동일)."""
        if self.selected_multi or self.selected is None:
            return
        m = self._find_measure(self.selected)
        if m:
            self._snapshot()
            ed.remove_chord_note(m, self.selected)
            self.selected = None
            self.selected_multi = []
            self._after_edit(None)

    def _cycle_staff(self):
        """선택한 요소의 보표를 다음 번호로 전환(피아노 등 다중 보표 악보용)."""
        if self.selected_multi or self.selected is None or self.score is None:
            return
        n_staves = max((getattr(e, "staff_no", 1)
                        for meas in self.score.measures for e in meas.elements),
                       default=1)
        if n_staves < 2:
            self._set_status("이 악보는 보표가 1개뿐입니다")
            return
        self._snapshot()
        ed.cycle_staff(self.selected, n_staves=n_staves)
        self._after_edit(self.selected)

    def _cycle_voice(self):
        """선택한 요소의 성부를 그 마디에 실제로 등장하는 성부 사이에서 전환."""
        if self.selected_multi or self.selected is None:
            return
        m = self._find_measure(self.selected)
        if m is None:
            return
        available = sorted({getattr(e, "voice_no", 1) for e in m.elements})
        if len(available) < 2:
            self._set_status("이 마디는 성부가 1개뿐입니다")
            return
        self._snapshot()
        ed.cycle_voice(self.selected, available_voices=available)
        self._after_edit(self.selected)

    # ------------------------------------------------------------- 조바꿈
    # ------------------------------------------------------- [5단계] 정보
    def on_about(self):
        has = bool(getattr(self, "_source_path", None))
        dlg = AboutDialog(self, load_info=getattr(self, "_last_load_info", None),
                          on_rerecognize=self._rerecognize if has else None,
                          on_diagnose=self._save_diagnostics if has else None,
                          on_check_update=lambda: self.check_update(manual=True),
                          on_batch_verify=self.on_batch_verify)
        dlg.exec()

    # ------------------------------------------------------------- [v3.60] 폴더 일괄 검증
    def on_batch_verify(self):
        """폴더(하위 폴더 포함)의 악보를 모두 인식해 원본과 비교한다. 뒤에서 돌고, 끝난 파일은 다음에 건너뛴다."""
        from omr_bridge import verify as vf
        from PySide6.QtWidgets import QProgressDialog
        if getattr(self, "_verify_running", False):
            return
        folder = QFileDialog.getExistingDirectory(self, "검증할 악보 폴더 고르기")
        if not folder:
            return
        files = vf.find_scores(folder)
        if not files:
            QMessageBox.information(self, "폴더 일괄 검증", "이 폴더에는 악보 파일(PDF·JPG·PNG 등)이 없습니다.")
            return
        limit, ok = QInputDialog.getInt(
            self, "폴더 일괄 검증",
            f"악보 {len(files)}개를 찾았습니다. 몇 곡을 검증할까요?\n"
            f"(0 = 전부. 숫자를 적으면 전체에서 고르게 그만큼만 — 처음엔 20곡쯤으로 시험해 보세요)\n"
            f"한 쪽에 1~2분쯤 걸립니다. 이미 끝난 곡은 건너뜁니다.",
            min(20, len(files)), 0, len(files))
        if not ok:
            return
        dlg = QProgressDialog("준비 중…", "멈추기", 0, max(1, limit or len(files)), self)
        dlg.setWindowTitle("폴더 일괄 검증")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumWidth(460)
        self._verify_dlg, self._verify_running, self._verify_t0 = dlg, True, time.time()
        self._verify_cancel = False
        dlg.canceled.connect(lambda: (setattr(self, "_verify_cancel", True),
                                      self._set_status("일괄 검증: 지금 하던 곡까지만 마치고 멈춥니다…")))
        if not getattr(self, "_verify_wired", False):
            self._verifyProgress.connect(self._on_verify_progress)
            self._verifyDone.connect(self._on_verify_done)
            self._verify_wired = True
        import threading

        def work():
            try:
                res = vf.run_batch(folder, limit=limit, app_version=APP_VERSION,
                                   progress=lambda i, n, name, stage: self._verifyProgress.emit(i, n, name, stage),
                                   cancelled=lambda: self._verify_cancel)
            except Exception as ex:
                res = ex
            self._verifyDone.emit(res)
        threading.Thread(target=work, daemon=True).start()
        dlg.show()

    def _on_verify_progress(self, i, n, name, stage):
        dlg = getattr(self, "_verify_dlg", None)
        if dlg is None:
            return
        dlg.setMaximum(max(1, n))
        dlg.setValue(min(i, n))
        el = time.time() - self._verify_t0
        left = f" · 남은 시간 약 {int(el / i * (n - i) / 60) + 1}분" if i >= 2 and n > i else ""
        dlg.setLabelText(f"{i}/{n} 끝남{left}\n{name} {stage}" if name else stage)

    def _on_verify_done(self, res):
        self._verify_running = False
        dlg = getattr(self, "_verify_dlg", None)
        if dlg is not None:
            dlg.close()
            self._verify_dlg = None
        if isinstance(res, Exception):
            QMessageBox.warning(self, "폴더 일괄 검증", f"검증하지 못했습니다.\n{type(res).__name__}: {res}")
            return
        box = QMessageBox(self)
        box.setWindowTitle("폴더 일괄 검증")
        box.setIcon(QMessageBox.Information)
        part = "" if res["files_done"] >= res["files_total"] else f" (전체 {res['files_total']}개 중 — 멈춤)"
        box.setText(f"<b>{res['files_done']}곡 검증을 마쳤습니다</b>{part}")
        box.setInformativeText(
            f"정상 {res['files_ok']}곡 · 인식 실패 {res['files_fail']}곡\n"
            f"원본에만 있는 잉크 평균 {res['avg_miss_pct']}% · 편집 악보에만 평균 {res['avg_extra_pct']}%\n"
            f"마디 검산 통과 평균 {res['avg_measures_ok_pct']}%\n\n"
            f"결과 폴더: {res['out_dir']}\n"
            f"· index.html — 차이가 큰 순서 보고서\n· summary.csv — 점수표\n"
            f"· {os.path.basename(res['zip'])} — 보내기용(점수표 + 유형별 조각 그림)")
        b_open = box.addButton("결과 폴더 열기", QMessageBox.AcceptRole)
        b_rep = box.addButton("보고서 열기", QMessageBox.ActionRole)
        box.addButton("닫기", QMessageBox.RejectRole)
        box.exec()
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        if box.clickedButton() is b_open:
            QDesktopServices.openUrl(QUrl.fromLocalFile(res["out_dir"]))
        elif box.clickedButton() is b_rep:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.join(res["out_dir"], "index.html")))

    # ------------------------------------------------------------- [v3.50] 새 버전 알림
    def check_update(self, manual=False):
        """GitHub 최신 릴리스를 뒤에서(화면이 멈추지 않게) 확인한다. 자동이면 하루 한 번만."""
        from omr_bridge import update_check as uc
        if not manual and not uc.due_today():
            return
        if not getattr(self, "_update_wired", False):
            self._updateResult.connect(self._on_update_result)
            self._update_wired = True
        if manual:
            self._set_status("새 버전을 확인하는 중…")
        import threading

        def work():
            try:
                info = uc.fetch_latest()
            except Exception as ex:                 # 인터넷 없음·GitHub 응답 없음·아직 릴리스 없음(404)
                info = ex
            self._updateResult.emit(info, manual)
        threading.Thread(target=work, daemon=True).start()

    def _on_update_result(self, info, manual):
        from omr_bridge import update_check as uc
        if isinstance(info, Exception):
            if manual:
                code = getattr(info, "code", None)
                msg = ("아직 GitHub에 공개된 버전이 없습니다." if code == 404 else
                       "새 버전을 확인하지 못했습니다(인터넷 연결을 확인해 주세요).")
                QMessageBox.information(self, "새 버전 확인", msg)
            return
        uc.mark_checked()
        latest = info["version"]
        if not uc.is_newer(latest, APP_VERSION):
            if manual:
                QMessageBox.information(self, "새 버전 확인", f"지금 쓰시는 v{APP_VERSION}이 최신 버전입니다.")
            return
        if not manual and uc.is_skipped(latest):
            return
        notes = info.get("notes") or ""
        if len(notes) > 700:
            notes = notes[:700].rstrip() + " …"
        box = QMessageBox(self)
        box.setWindowTitle("새 버전 알림")
        box.setIcon(QMessageBox.Information)
        box.setText(f"<b>{APP_NAME} 새 버전 v{latest}</b>이 나왔습니다. (지금 v{APP_VERSION})")
        box.setInformativeText((notes + "\n\n" if notes else "") +
                               "내려받아 설치하면 지금 버전 위에 그대로 바뀝니다(편집 파일은 그대로).")
        asset = info.get("asset")
        b_dl = box.addButton("지금 내려받기" if asset else "내려받기 페이지 열기", QMessageBox.AcceptRole)
        b_page = box.addButton("바뀐 점 보기", QMessageBox.HelpRole) if asset else None
        box.addButton("나중에", QMessageBox.RejectRole)
        b_skip = box.addButton("이 버전은 알리지 않기", QMessageBox.DestructiveRole)
        box.exec()
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        clicked = box.clickedButton()
        if clicked is b_dl:
            QDesktopServices.openUrl(QUrl(asset["url"] if asset else info["page"]))
        elif b_page is not None and clicked is b_page:
            QDesktopServices.openUrl(QUrl(info["page"]))
        elif clicked is b_skip:
            uc.skip_version(latest)

    def _cache_dirs(self, path):
        """[v2.53] 이 악보의 인식 결과(캐시) 폴더들."""
        dirs = []
        try:
            if str(path).lower().endswith(".pdf"):
                from omr_bridge.book import n_pdf_pages, BookJob
                if n_pdf_pages(path) > 1:
                    dirs.append(BookJob(path).dir)
            from omr_bridge import document_adapter as da
            d = da._user_cache_root() / da._cache_key(path)
            if d.exists():
                dirs.append(d)
        except Exception:
            pass
        return dirs

    def _rerecognize(self):
        """[v2.53] 저장된 인식 결과를 지우고 처음부터 다시 인식한다(편집 파일 .sprouty는 지우지 않는다)."""
        path = getattr(self, "_source_path", None)
        if not path or getattr(self, "_loading", False):
            return
        if QMessageBox.question(self, "다시 인식",
                                "저장된 인식 결과를 지우고 이 악보를 처음부터 다시 인식합니다.\n"
                                "처음 여는 것처럼 시간이 걸립니다. 계속할까요?") != QMessageBox.Yes:
            return
        import shutil
        for d in self._cache_dirs(path):
            if d.name.startswith("book_"):
                from omr_bridge.book import BookJob
                BookJob(path).reset_recognition()
            else:
                shutil.rmtree(d, ignore_errors=True)
        self._dirty = False
        self.load_image(path)

    def _save_diagnostics(self):
        """[v2.53] 진단 자료 zip: 프로그램·Audiveris·환경 정보, 인식 기록(log), 쪽별 결과(.mxl), 인식 자료(.omr),
        자르기 정보(meta.json), 지금 화면 악보의 요약. 원본 옆 '<이름>_진단.zip'."""
        import json
        import platform
        import zipfile
        path = getattr(self, "_source_path", None)
        if not path:
            return
        out = os.path.splitext(path)[0] + "_진단.zip"
        info = {"app": f"{APP_NAME} v{APP_VERSION}", "python": platform.python_version(),
                "os": platform.platform(), "source": os.path.basename(path)}
        try:
            from omr_bridge.audiveris_runner import find_audiveris_bin
            info["audiveris"] = find_audiveris_bin()
        except Exception as ex:
            info["audiveris"] = f"(찾지 못함: {ex})"
        try:
            st = os.stat(path)
            info["source_size"] = st.st_size
        except OSError:
            pass
        if self.score is not None:
            from score import editing as ed_
            v_ = ed_.validate(self.score)
            info["measures"] = len(self.score.measures)
            info["errors"] = [k for k, x in v_.items() if x[0] not in ("ok", "anacrusis", "time_change")]
            info["pages_loaded"] = len(self._book_state.pages) if getattr(self, "_book_state", None) else 1
        info["load_notes"] = list(getattr(self, "_last_load_info", None) or [])
        try:
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("info.json", json.dumps(info, ensure_ascii=False, indent=2))
                for d in self._cache_dirs(path):
                    for f in sorted(d.iterdir()):
                        if f.suffix in (".log", ".mxl", ".omr", ".json"):
                            z.write(str(f), f"{d.name}/{f.name}")
        except Exception as ex:
            QMessageBox.warning(self, "진단 자료", f"진단 자료를 저장하지 못했습니다.\n\n{ex}")
            return
        QMessageBox.information(self, "진단 자료", f"진단 자료를 저장했습니다.\n\n{out}")

    def _sync_meta_from_items(self):
        """header_items(화면 표시용) → score.meta(MusicXML 내보내기용)로 옮긴다."""
        meta = {}
        for it in self.header_items:
            t = it.get("text", "").strip()
            if not t:
                continue
            if it["kind"] == "title":
                meta["title"] = t
            elif it["kind"] == "credit":
                meta["credit"] = (meta["credit"] + " / " + t) if meta.get("credit") else t
                for part in re.split(r"[/,·]", t):
                    if "작사" in part or "작시" in part:
                        meta["lyricist"] = re.sub(r"작사|작시", "", part).strip()
                    elif "작곡" in part:
                        meta["composer"] = re.sub(r"작곡", "", part).strip()
                    elif "역사" in part or "번역" in part:
                        meta["translator"] = re.sub(r"역사|번역", "", part).strip()
            elif it["kind"] == "tempo":
                meta["tempo"] = t
        self.score.meta = meta

    def on_song_info(self):
        """곡 정보 팝업. 인식된 항목이 있으면 그 자리·크기 그대로 글자만 바꾼다. 없으면(스캔 악보 등)
        [v2.25] 원본 1쪽 첫 줄 위의 글자 덩어리(제목·작사작곡·빠르기말)를 찾아 그 자리·크기에 넣는다
        (못 찾은 항목만 기본 자리)."""
        if self.score is None or not self.by_sys:
            self._set_status("먼저 악보 이미지를 열어 주세요")
            return
        titles = [it for it in self.header_items if it["kind"] == "title"]
        credits = sorted([it for it in self.header_items if it["kind"] == "credit"], key=lambda it: it["y"])
        tempos = [it for it in self.header_items if it["kind"] == "tempo"]
        meter = (self.score.beats, self.score.beat_type)
        dlg = SongInfoDialog(self, title=titles[0]["text"] if titles else "",
                             credit=" / ".join(it["text"] for it in credits),
                             tempo=tempos[0]["text"] if tempos else "", meter=meter)
        if dlg.exec() != QDialog.Accepted:
            return
        title, credit, tempo = dlg.values()
        top = self.by_sys[0][0]
        il = self.by_sys[0][1] - self.by_sys[0][0]
        img1 = self._book_job.page_image(1) if getattr(self, "_book_job", None) is not None else self.original_bgr
        x_left = (self.staff_left or {}).get(0, 0) or 0
        x_right = max((self.barlines or {}).get(0) or [img1.shape[1] * 0.95])
        try:
            from omr_bridge.header_layout import find_header_blocks
            blocks = find_header_blocks(img1, top, x_left, x_right, il)
        except Exception:
            blocks = {}

        def from_box(kind, b, h_ratio=1.0):
            return dict(kind=kind, x0=b[0], x1=b[2], y=(b[1] + b[3]) / 2.0, h=(b[3] - b[1]) * h_ratio)
        defaults = {"title": dict(kind="title", x0=x_left + (x_right - x_left) * 0.25,
                                  x1=x_left + (x_right - x_left) * 0.75, y=top - 6.5 * il, h=3.4 * il),
                    "credit": dict(kind="credit", x0=x_right - 8 * il, x1=x_right, y=top - 4.2 * il, h=1.8 * il),
                    "tempo": dict(kind="tempo", x0=x_left, x1=x_left + 8 * il, y=top - 3.2 * il, h=2.0 * il)}
        others = [it for it in self.header_items if it["kind"] not in ("title", "credit", "tempo")]
        new_items = []
        # 제목
        if title:
            it = titles[0] if titles else (from_box("title", blocks["title"]) if "title" in blocks
                                           else dict(defaults["title"]))
            it["text"] = title
            new_items.append(it)
        # 작사/작곡: '/'마다 한 줄
        lines = [t.strip() for t in credit.split("/") if t.strip()]
        if lines:
            if credits and len(credits) == len(lines):
                for it, t in zip(credits, lines):
                    it["text"] = t
                    new_items.append(it)
            else:
                boxes = [(it["x0"], it["y"] - it["h"] / 2, it["x1"], it["y"] + it["h"] / 2) for it in credits] \
                    or list(blocks.get("credit") or [])
                if boxes:
                    pitch = (boxes[1][1] - boxes[0][1]) if len(boxes) >= 2 else (boxes[0][3] - boxes[0][1]) * 1.25
                    for i, t in enumerate(lines):
                        bx = boxes[i] if i < len(boxes) else (boxes[-1][0], boxes[-1][1] + pitch * (i - len(boxes) + 1),
                                                               boxes[-1][2], boxes[-1][3] + pitch * (i - len(boxes) + 1))
                        it = from_box("credit", bx)
                        it["text"] = t
                        new_items.append(it)
                else:
                    d0 = defaults["credit"]
                    for i, t in enumerate(lines):
                        it = dict(d0, y=d0["y"] - (len(lines) - 1 - i) * d0["h"] * 1.3, text=t)
                        new_items.append(it)
        # 빠르기말(괄호·음표 때문에 덩어리가 글자보다 커서 글자 높이는 약 3/4)
        if tempo:
            it = tempos[0] if tempos else (from_box("tempo", blocks["tempo"], 0.75) if "tempo" in blocks
                                           else dict(defaults["tempo"]))
            it["text"] = tempo
            new_items.append(it)
        self.header_items = others + new_items
        self._sync_meta_from_items()
        self._rerender()
        self._set_status("곡 정보를 적용했습니다" + (" — 원본의 글자 자리에 맞췄습니다" if blocks else ""))

    def on_transpose(self):
        """조바꿈 팝업: 목표 조(15개 장조)와 이동 방향을 골라 전체 조옮김."""
        if self.score is None:
            self._set_status("먼저 악보 이미지를 열어 주세요")
            return
        dlg = TransposeDialog(self, current_fifths=self.score.fifths)
        if dlg.exec() != QDialog.Accepted:
            return
        target, direction = dlg.result_values()
        if target == self.score.fifths and direction == "nearest":
            self._set_status("이미 같은 조입니다")
            return
        self._snapshot()
        ed.transpose_score(self.score, target, direction)
        self.selected = None
        self.selected_multi = []
        self._rerender()
        self.on_select(None)
        name = dict((f, n) for f, n in ed.KEY_CHOICES)[target]
        self._set_status(f"조바꿈 완료: {name}")

    def on_undo(self):
        if not self._undo:
            self._set_status("실행 취소할 항목이 없습니다")
            return
        self.score = self._undo.pop()
        if getattr(self, "_book_state", None) is not None:
            self._book_state.score = self.score
        self.selected_suggestion = None          # [v1.40] 되돌린 악보의 제안 목록으로 다시 그린다
        self._refresh_ink_marks()
        self.selected = None
        self.selected_multi = []
        self._rerender()
        self.on_select(None)

    def _nudge_position(self, direction: int):
        """[v2.20] 여러 개 골랐으면 모두 같은 만큼 좌우로 옮긴다(쉼표 포함)."""
        if self.selected_multi and self.layout_obj is not None:
            step = self.layout_obj.staff_space * 0.15
            delta = int(round(step * direction))
            for e in self.selected_multi:
                e.src_x += delta
                if e.src_bbox:
                    bx, by, bw, bh = e.src_bbox
                    e.src_bbox = (bx + delta, by, bw, bh)
            keep = list(self.selected_multi)
            self._rerender()
            self.selected_multi = keep
            self._refresh_selection_ui_multi()
            return
        self._nudge_position_one(direction)

    def _nudge_position_one(self, direction: int):
        """[기능 추가 — 실사용 요청] 원본 이미지 위의 강조 사각형
        위치(src_x/src_bbox)를 좌우로 미세 조정한다. 자동으로 추정한
        위치(특히 "앞/뒤에 삽입"으로 만든 synthetic 요소)가 실제 음표
        머리와 어긋나 보일 때, 사람이 눈으로 보면서 직접 맞출 수 있게
        한다. 마디 배정·박자 계산과는 무관한 순수 표시 위치 보정이라
        _snapshot()/검증 갱신 없이 즉시 반영한다."""
        if self.selected_multi or self.selected is None or self.layout_obj is None:
            return
        step = self.layout_obj.staff_space * 0.15
        delta = int(round(step * direction))
        e = self.selected
        e.src_x += delta
        if e.src_bbox:
            bx, by, bw, bh = e.src_bbox
            e.src_bbox = (bx + delta, by, bw, bh)
        self._rerender()
        g = self._glyph_of(e)
        if g is not None:
            self.on_select(g)

    def _resync_synthetic_pos(self, element):
        """[편집 기능 추가 — 실사용 검증 후] "앞/뒤에 삽입"으로 사람이
        새로 만든 요소(synthetic=True)는 원본 스캔에 실제 잉크가 없으므로
        y좌표를 추측할 수밖에 없다 — 처음 삽입 시점엔 참조 요소의 y를
        그대로 복사해뒀는데, 그 뒤 음정을 바꿔도(위/아래 화살표 등)
        y가 갱신되지 않아 원본 이미지 패널의 강조 사각형이 실제로
        의도한 음높이와 동떨어진 곳(엉뚱하게 위/아래로 치우친 곳)에
        그려지는 문제가 있었다. 음정이 바뀔 때마다, 그 마디가 속한
        오선 좌표를 기준으로 diatonic 위치를 다시 계산해 y를 맞춘다."""
        if element is None or not getattr(element, "synthetic", False):
            return
        if element.kind != "note" or element.step is None or self.layout_obj is None:
            return
        m = self._find_measure(element)
        if m is None:
            return
        staff_no = getattr(element, "staff_no", 1)
        sysL = next((s for s in self.layout_obj.systems
                    if s.index == m.system and s.staff_no == staff_no), None)
        if sysL is None:
            return
        # [v1.10 음자리표 변경] 줄 머리 음자리표가 아니라 이 요소에 실제 적용되는 음자리표로 계산
        g_ = next((g for g in self.layout_obj.glyphs if g.element is element), None)
        clef_k = (g_.clef if g_ is not None and g_.clef else None) or \
            getattr(sysL, "clef", "F" if staff_no >= 2 else "G")
        dia_top = sl.CLEF_DIA.get(clef_k, sl.CLEF_DIA["G"])[0]
        dia = sl.diatonic(element.step, element.octave)
        ss = sysL.staff_space
        cy = sysL.staff_top + (dia_top - dia) * (ss / 2)
        # [크기 버그 수정 — 실사용 검증 후] 이전에는 참조 요소(대개
        # 쉼표)의 bbox 크기(bw,bh)를 그대로 이어받아 y만 바꿨는데,
        # 쉼표 글리프는 음표 머리보다 세로로 훨씬 길쭉한 모양이라, 새
        # 음표를 선택했을 때 강조 사각형이 "음표 머리 하나 크기"가
        # 아니라 "쉼표 크기 그대로" 표시되는 문제가 있었다(실사용
        # 검증 중 발견 — 다른 음표를 선택했을 때와 확연히 다른 크기로
        # 보임). 음표로 바뀐 뒤에는 실측 음표 머리 비율(폭≈1.25칸,
        # 높이≈1.05칸 — 다른 인식된 음표들의 실제 bbox 비율 참고)로
        # 강제 재계산한다.
        bx0, by0, bw0, bh0 = element.src_bbox if element.src_bbox else \
            (element.src_x, cy, 0, 0)
        cx = bx0 + bw0 / 2.0
        bw, bh = ss * 1.25, ss * 1.05
        element.src_bbox = (int(round(cx - bw / 2)), int(round(cy - bh / 2)),
                            int(round(bw)), int(round(bh)))

    def _auto_repair(self, elements):
        """[v1.43] 편집으로 바뀐 마디가 정상이 아니면 성부를 다시 나눠 본다(정상이 될 때만 반영) —
        예: 4분음표로 잘못 읽힌 G♯2를 8분음표로 고치면, 따로 떨어져 있던 E3가 같은 성부로 들어간다."""
        if self.score is None:
            return
        done = set()
        for el in elements or []:
            # 마디를 직접 넘길 수도 있다(지운 요소는 더 이상 어느 마디에도 없으므로)
            m = el if hasattr(el, "elements") else (self._find_measure(el) if el is not None else None)
            if m is None or m.number in done:
                continue
            done.add(m.number)
            # 마디가 정상이 아니거나, 한 보표에 '음 하나뿐인 성부'가 다른 성부 옆에 따로 떨어져 있으면
            # (박 합계는 맞아도 재생·내보내기에서 뒤 음들이 앞당겨진다) 성부를 다시 나눠 본다
            cnt = {}
            for e in m.elements:
                if not e.is_grace:
                    cnt.setdefault(e.staff_no, {}).setdefault(e.voice_no, 0)
                    cnt[e.staff_no][e.voice_no] += 1
            lone = any(len(vs) >= 2 and 1 in vs.values() for vs in cnt.values())
            if lone or ed.validate(self.score).get(m.number, ("ok",))[0] not in ("ok", "anacrusis", "time_change"):
                ed.repair_voices(self.score, m)
            # [v2.11] 이 마디에 남은 제안 음표를 다시 넣어 본다(편집으로 막힌 것이 풀렸을 수 있다)
            try:
                from omr_bridge.ink_check import auto_apply_measure
                added_ = auto_apply_measure(self.score, m.number)
                if added_:
                    self._set_status("빠진 음표를 함께 넣었습니다: " + ", ".join(q["text"] for q in added_))
            except Exception:
                pass

    def _after_edit(self, keep):
        self._diff_dirty = True                # [v4.10] 악보가 바뀜 — 붓 후보 자리는 붓을 다시 켤 때 새로 찾는다
        self._auto_repair([keep])
        self._resync_synthetic_pos(keep)
        self.selected = keep
        self.selected_multi = []
        self._rerender()                       # 모델 변경 → 재렌더(줌/팬 유지)
        if keep is not None:
            g = self._glyph_of(keep)
            if g is not None:
                self.on_select(g)
                return
        self.on_select(None)

    def _after_edit_targets(self, targets):
        """[3-B] 다중 편집 후: 편집된 요소 전부를 다시 선택 상태로 유지한
        채 재렌더한다(레이아웃이 바뀌어도 같은 Element 객체 참조는 그대로
        유효하므로 다시 찾아 강조할 수 있다)."""
        self._auto_repair(targets)
        self._rerender()
        if len(targets) >= 2:
            self.selected = targets[-1]
            self.selected_multi = list(targets)
            self._refresh_selection_ui()
        else:
            self.selected = targets[0] if targets else None
            self.selected_multi = []
            g = self._glyph_of(self.selected) if self.selected else None
            self.on_select(g) if g else self.on_select(None)

    # ------------------------------------------------------------- 네비/요약
    def on_next_error(self):
        if not self.score:
            return
        report = ed.validate(self.score)
        bad = [m for m in self.score.measures if report[m.number][0] != "ok"]
        if not bad:
            self._set_status("모든 마디 정상입니다 ✓")
            return
        m = bad[0]
        if m.elements:
            g = self._glyph_of(m.elements[0])
            if g:
                self.on_select(g)
        self._set_status(
            f"오류 마디 {report[m.number]} — 마디 {m.number} (남은 오류 {len(bad)}개)")

    def _refresh_summary(self):
        if not self.score:
            return
        report = ed.validate(self.score)
        ok = sum(1 for s, _, _ in report.values() if s == "ok")
        n = len(report)
        self.toolbar.set_stats(total=n, ok=ok, error=n - ok)
        self.bottom_bar.set_signatures(
            self.score.fifths, self.score.beats, self.score.beat_type,
            getattr(self.score, "time_sig_known", True),
            ts_symbol=getattr(self.score, "ts_symbol", None))

    def on_export(self):
        if not self.score:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "MusicXML 저장", "score.musicxml",
            "MusicXML (*.musicxml *.xml);;모든 파일 (*.*)")
        if not path:
            return
        try:
            mx.save_musicxml(self.score, path)
            self._set_status(f"내보냄: {path}")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"내보내기 실패:\n{e}")

    def on_export_midi(self):
        if not self.score:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "MIDI 저장", "score.mid", "MIDI (*.mid);;모든 파일 (*.*)")
        if not path:
            return
        try:
            score_midi.save_midi(self.score, path, bpm=PLAY_BPM)
            self._set_status(f"내보냄: {path}")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"MIDI 내보내기 실패:\n{e}")

    # ------------------------------------------------------------- 재생(2단계)
    # 소리로 편집하기: 내장 신스로 WAV 합성 → QSoundEffect 재생, 타이머로
    # 현재 음표를 따라가며 선택·하이라이트한다(원본↔편집 양쪽 모두).
    def on_play(self):
        if self._playing:
            self._stop_playback()
            return
        if not self.score:
            return
        try:
            events = score_midi.score_events(self.score, bpm=PLAY_BPM)
            if len(events) <= 1:
                return
            import tempfile
            # [재생 버그 수정 — 실사용 피드백] 예전엔 늘 같은 이름(omr_playback.wav)에 덮어썼는데,
            # QSoundEffect는 한 번 읽은 소리를 '파일 경로'를 열쇠로 메모리에 보관해 같은 경로면
            # 다시 읽지 않는다 → 다른 악보를 열거나 편집한 뒤 재생해도 처음 재생한 곡이 나왔다.
            # 재생할 때마다 새 이름으로 만들어 항상 지금 악보를 읽게 하고, 지난 파일은 지운다.
            self._play_seq = getattr(self, "_play_seq", 0) + 1
            old_wav = getattr(self, "_play_wav", None)
            self._play_wav = os.path.join(
                tempfile.gettempdir(), f"omr_playback_{os.getpid()}_{self._play_seq}.wav")
            syn = events
            if score_midi.CHORD_ACCOMP:                 # [v3.00] 코드 반주는 소리에만(하이라이트 목록은 그대로)
                syn = sorted(events[:-1] + score_midi.chord_events(self.score, events, PLAY_BPM),
                             key=lambda ev: ev[0]) + [events[-1]]
            score_midi.synth_wav(syn, self._play_wav,
                                 vmap=score_midi.velocity_map(self.score))
            if old_wav and old_wav != self._play_wav:
                try:
                    os.remove(old_wav)
                except OSError:
                    pass                  # 아직 재생기가 잡고 있으면 끝낼 때 정리
        except Exception as e:
            QMessageBox.critical(self, "오류", f"재생 준비 실패:\n{e}")
            return
        try:
            from PySide6.QtMultimedia import QSoundEffect
            from PySide6.QtCore import QUrl
            self._sound = QSoundEffect(self)
            self._sound.setSource(QUrl.fromLocalFile(self._play_wav))
            self._sound.setVolume(0.9)
            self._sound.play()
        except Exception:
            # QtMultimedia 미가용 환경(일부 리눅스 최소 설치 등): 파일로 안내
            QMessageBox.information(
                self, "재생", "이 환경에서 소리 재생 모듈을 열 수 없어\n"
                f"합성된 오디오를 파일로 저장했습니다:\n{self._play_wav}")
            return
        self._play_events = [ev for ev in events if ev[3] is not None]
        self._play_t0 = __import__("time").monotonic()
        self._play_idx = -1
        self._playing = True
        self._play_total = events[-1][0]
        if self.toolbar.play_btn:
            self.toolbar.play_btn.setText("■ 정지")
        self._play_timer.start(30)

    def closeEvent(self, event):
        """끝낼 때 재생용 임시 소리 파일을 정리한다. [v2.50] 저장하지 않은 편집이 있으면 물어본다."""
        if not self._confirm_discard():
            event.ignore()
            return
        try:
            self._stop_playback()
        except Exception:
            pass
        import glob
        import tempfile
        for f in glob.glob(os.path.join(tempfile.gettempdir(), f"omr_playback_{os.getpid()}_*.wav")):
            try:
                os.remove(f)
            except OSError:
                pass
        super().closeEvent(event)

    def _stop_playback(self):
        self._playing = False
        self._play_timer.stop()
        if getattr(self, "_sound", None) is not None:
            try:
                self._sound.stop()
            except Exception:
                pass
            self._sound = None
        if self.toolbar.play_btn:
            self.toolbar.play_btn.setText("▶ 재생")
        self._set_status("재생 정지")

    # ------------------------------------------------------------- [v2.50] 편집 저장·이어서 편집
    def on_save(self):
        """편집한 악보를 원본 옆에 같은 이름 + .sprouty로 저장(곡 정보 포함)."""
        src = getattr(self, "_source_path", None)
        if self.score is None or not src:
            self._set_status("저장할 악보가 없습니다 — 먼저 악보를 여세요")
            return False
        if getattr(self, "_book_state", None) is not None and \
                len(self._book_state.pages) < getattr(self._book_job, "n_pages", 0):
            self._set_status("아직 인식 중인 페이지가 있습니다 — 모든 페이지를 불러온 뒤 저장하세요")
            return False
        from score import project
        try:
            path = project.save(src, self.score, self.header_items, APP_VERSION)
        except Exception as ex:
            QMessageBox.warning(self, "저장하지 못했습니다", f"편집 파일을 저장하지 못했습니다.\n\n{ex}")
            return False
        self._dirty = False
        # [v2.54] 저장할 때마다 시각을 보여 새로 저장됐는지 알 수 있게, 확인 창도 띄운다(닫기 전 저장 확인 중에는 생략)
        t = time.strftime("%H:%M:%S")
        self._set_status(f"편집을 저장했습니다 ({t}): {path}")
        if not getattr(self, "_saving_quietly", False):
            QMessageBox.information(self, "저장", f"편집을 저장했습니다.\n\n저장 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}\n{path}")
        return True

    def _confirm_discard(self):
        """저장하지 않은 편집이 있으면 저장/버림/취소를 묻는다. 계속해도 되면 True."""
        if not getattr(self, "_dirty", False) or self.score is None:
            return True
        r = QMessageBox.question(self, "저장하지 않은 편집",
                                 "저장하지 않은 편집이 있습니다. 저장할까요?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                 QMessageBox.Save)
        if r == QMessageBox.Save:
            self._saving_quietly = True
            try:
                return bool(self.on_save())
            finally:
                self._saving_quietly = False
        return r == QMessageBox.Discard

    def _ask_resume(self, path):
        """원본 옆에 편집 파일이 있으면 이어서 편집할지 묻는다. 이어서면 편집 파일 경로, 아니면 None."""
        from score import project
        info = project.peek(path)
        if info is None:
            return None
        if info.get("broken"):
            QMessageBox.warning(self, "편집 파일", f"편집 파일을 읽을 수 없어 새로 인식합니다.\n{info['path']}")
            return None
        msg = (f"이 악보의 편집 파일이 있습니다.\n\n저장: {info['saved_at']}  (콩나물지기 v{info['app_version']})\n"
               f"{info['path']}\n\n이어서 편집할까요?")
        if info.get("changed"):
            msg += "\n\n※ 저장한 뒤 원본 파일이 바뀌었습니다 — 이어서 편집하면 음표 위치가 맞지 않을 수 있습니다."
        box = QMessageBox(self)
        box.setWindowTitle("이어서 편집")
        box.setText(msg)
        b_resume = box.addButton("이어서 편집", QMessageBox.AcceptRole)
        box.addButton("새로 인식", QMessageBox.RejectRole)
        box.setDefaultButton(b_resume)
        box.exec()
        return info["path"] if box.clickedButton() is b_resume else None

    def _apply_pending_project(self):
        """인식이 끝난 뒤, 고른 편집 파일의 악보·곡 정보로 바꾼다(되돌리기 기록은 새로 시작)."""
        path = getattr(self, "_pending_project", None)
        self._pending_project = None
        if not path:
            return
        from score import project
        try:
            data = project.load(path)
        except Exception as ex:
            QMessageBox.warning(self, "편집 파일", f"편집 파일을 읽지 못해 인식 결과로 시작합니다.\n\n{ex}")
            return
        recog = self.score                          # 방금 인식한 악보(원본 정보가 최신)
        self.score = data["score"]
        # [v2.56] 예전 저장 파일에 없는 '마디 단위 원본 정보'를 방금 인식한 결과에서 마디 번호로 옮겨 온다
        # (볼타 꺾쇠 높이 v2.54, 줄 첫 마디 도돌이 v2.51, 줄 머리 기호 자리 v2.30) — 사용자가 편집할 수 없는 정보
        try:
            rec_by = {m_.number: m_ for m_ in (recog.measures if recog is not None else [])}
            for m_ in self.score.measures:
                r_ = rec_by.get(m_.number)
                if r_ is None:
                    continue
                for attr in ("ending_ys", "start_repeat_x"):
                    if getattr(m_, attr, None) is None and getattr(r_, attr, None) is not None:
                        setattr(m_, attr, getattr(r_, attr))
                if getattr(r_, "start_repeat", False) and not getattr(m_, "start_repeat", False):
                    m_.start_repeat = True
            if recog is not None and not getattr(self.score, "header_x", None) and getattr(recog, "header_x", None):
                self.score.header_x = recog.header_x
        except Exception:
            pass
        # [v2.55] 예전 버전에서 저장한 편집 파일에는 그 뒤에 생긴 '불러올 때 바로잡기'가 빠져 있다(예: v2.54 화음 속
        # 먼 같은 음 떼기 — 저장 파일로 이어서 편집하면 인식을 다시 하지 않아 옛 데이터 그대로였다). 이어서 열 때 적용한다.
        fixed = []
        try:
            def _vt(v):
                return tuple(int(x) for x in str(v).split(".") if x.isdigit())
            if _vt(data.get("app_version", "0")) < _vt(APP_VERSION):
                from omr_bridge import ink_check as ik_
                if getattr(self.score, "ink_blobs", None) is None:
                    self.score.ink_blobs = []
                # 실제 목록을 넘긴다 — 증거로 쓴 빔 잉크를 미인식 표시에서 빼려면(예전엔 사본을 넘겨 주황 점선이 남았다)
                fixed = ik_.split_far_unisons(self.score, self.score.ink_blobs)
                for m_ in self.score.measures:
                    for e_ in m_.elements:
                        if getattr(e_, "_split_touched", False):
                            e_._orig = (e_.step, e_.octave, e_.duration_base, e_.src_x)
                            e_._split_touched = False
        except Exception:
            fixed = []
        if fixed:
            self._last_load_info = [f"저장 파일(v{data.get('app_version')})을 이 버전에 맞게 고쳤습니다: " + n
                                    for n in fixed] + list(self._last_load_info or [])
            self._dirty = True
        if getattr(self, "_book_state", None) is not None:
            self._book_state.score = self.score
        self.header_items = list(data.get("header_items") or [])
        self._sync_meta_from_items()
        self._undo = []
        self.selected, self.selected_multi = None, []
        self._rerender()
        self._refresh_ink_marks()
        self._dirty = bool(fixed)
        self._set_status(f"편집 파일을 불러와 이어서 편집합니다 (저장: {data.get('saved_at', '?')})"
                         + (f" — 이 버전에 맞게 {len(fixed)}곳을 고쳤습니다('정보'에서 확인)" if fixed else ""))

    def _score_context_menu(self, payload):
        """[v2.41] 편집 악보 오른쪽 버튼 메뉴 — 고른 것에 쓸 수 있는 동작.
        [v2.52] 이미 고른 음표 위에서 누른 경우에만 연다."""
        gpos, g = payload
        sel = self.selected_multi or ([self.selected] if self.selected is not None else [])
        if g is None or g.element not in sel:
            return
        menu = QMenu(self)
        act_align = menu.addAction("정렬 — 같은 박 음표를 세로로 가운데 맞추기")
        act_align.setEnabled(len(sel) >= 2)
        menu.addSeparator()
        act_del = menu.addAction("삭제")
        act_del.setEnabled(bool(sel))
        chosen = menu.exec(gpos)
        if chosen is act_align:
            self._align_selected()
        elif chosen is act_del:
            self._delete()

    def _align_selected(self):
        sel = list(self.selected_multi or ([self.selected] if self.selected is not None else []))
        if len(sel) < 2:
            return
        self._snapshot()
        n = ed.align_columns(self.score, sel)
        if not n:
            if self._undo:
                self._undo.pop()                      # 바뀐 게 없으면 되돌리기 기록도 남기지 않는다
            self._set_status("정렬: 서로 다른 보표에서 같은 박에 놓인 음표를 함께 고르세요")
            return
        keep = sel
        self._rerender()
        self.selected_multi = [e for e in keep if self._glyph_of(e) is not None]
        if len(self.selected_multi) >= 2:
            self._refresh_selection_ui_multi()
        self._set_status(f"같은 박의 음표를 세로로 맞췄습니다 ({n}개 옮김 · Ctrl+Z로 되돌리기)")

    # ------------------------------------------------------------- [v2.60] 붓으로 기호 넣기
    _BRUSH_LABELS = {"slur": "이음줄", "tie": "붙임줄", "cresc": "크레셴도 <", "decresc": "디크레셴도 >",
                     "staccato": "스타카토", "tenuto": "테누토", "dot": "점(부점)",
                     # [v2.70] 글꼴 모양 비교로 읽는 기호
                     "accent": "악센트", "marcato": "마르카토", "staccatissimo": "스타카티시모",
                     "fermata": "늘임표", "sharp": "올림표 ♯", "flat": "내림표 ♭", "natural": "제자리표 ♮",
                     "dyn:ppp": "ppp", "dyn:pp": "pp", "dyn:p": "p", "dyn:mp": "mp", "dyn:mf": "mf",
                     "dyn:f": "f", "dyn:ff": "ff", "dyn:fff": "fff", "dyn:fp": "fp", "dyn:sfz": "sfz",
                     # [v2.80] 꾸밈·숨표·아르페지오·이동·옥타브·페달
                     "trill": "트릴 tr", "mordent": "모르덴트", "turn": "턴", "breath": "숨표",
                     "arpeggio": "아르페지오", "segno": "세뇨", "coda": "코다",
                     "dc": "D.C.", "ds": "D.S.", "fine": "Fine", "dc_al_fine": "D.C. al Fine",
                     "dc_al_coda": "D.C. al Coda", "ds_al_fine": "D.S. al Fine", "ds_al_coda": "D.S. al Coda",
                     "to_coda": "To Coda", "8va": "8va", "8vb": "8vb", "15ma": "15ma",
                     "ped": "페달 Ped.", "ped_up": "페달 뗌 *",
                     # [v2.90]
                     "tremolo1": "트레몰로 /", "tremolo2": "트레몰로 //", "tremolo3": "트레몰로 ///",
                     "caesura": "카이주라 //", "ped_line": "페달 선 ⎣_⎦", "gliss": "글리산도",
                     "cresc_text": "cresc.", "dim_text": "dim.", "decresc_text": "decresc.",
                     "chord": "코드 이름…",
                     # [v3.40] 셋잇단음표·박자표
                     "tuplet3": "셋잇단음표 3", "ts:common": "박자표 C (4/4)", "ts:cut": "박자표 ₵ (2/2)",
                     "ts:2/4": "박자표 2/4", "ts:3/4": "박자표 3/4", "ts:4/4": "박자표 4/4", "ts:2/2": "박자표 2/2",
                     "ts:3/8": "박자표 3/8", "ts:6/8": "박자표 6/8", "ts:9/8": "박자표 9/8",
                     "ts:12/8": "박자표 12/8", "ts:5/4": "박자표 5/4", "ts:3/2": "박자표 3/2"}

    def _on_escape(self):
        """[v3.40] Esc: 칠한 자국·후보를 지우고 붓도 끈다(예전엔 자국만 지워 붓 아이콘이 켜진 채 남았다)."""
        ip = self.image_panel
        if getattr(ip, "_brush_on", False):
            ip.set_brush(False)
            self._set_status("붓을 껐습니다 (B로 다시 켜기)")
        elif ip._strokes:
            ip.clear_brush()

    def _brush_icon(self, key):
        try:
            from omr_bridge import brush as _br
            if key in _br.TEMPLATES:
                return _smufl_icon(_br.TEMPLATES[key][0], 120)
            if key in _br.OTTAVA:
                return _smufl_icon(_br.OTTAVA[key], 100)
            if key == "tuplet3":
                return _smufl_icon(_br.TUPLET3, 110)
            if key in ("ts:common", "ts:cut"):
                return _smufl_icon(_br._TS_GLYPH[key], 110)
            if key.startswith("ts:"):
                return None
            if key == "gliss":
                return _draw_icon(64, 64, lambda d, c: d.line([(8, 52), (56, 12)], fill=c, width=4))
            if key == "ped_line":
                return _draw_icon(64, 64, lambda d, c: d.line([(6, 20), (6, 46), (26, 46), (32, 34), (38, 46),
                                                               (58, 46), (58, 20)], fill=c, width=4))
            if key in _br.TEXTS:
                return None
            if key in ("slur", "tie"):
                return _slur_icon()
            if key == "staccato":
                return _smufl_icon("\uE4A2", 120)
            if key == "tenuto":
                return _smufl_icon("\uE4A4", 120)
            if key == "dot":
                return _smufl_icon("\uE1E7", 120)
            if key in ("cresc", "decresc"):
                def paint(d, c, k=key):
                    if k == "cresc":
                        d.line([(6, 32), (58, 18)], fill=c, width=4)
                        d.line([(6, 32), (58, 46)], fill=c, width=4)
                    else:
                        d.line([(6, 18), (58, 32)], fill=c, width=4)
                        d.line([(6, 46), (58, 32)], fill=c, width=4)
                return _draw_icon(64, 64, paint)
        except Exception:
            return None
        return None

    def _brush_context(self, strokes):
        """붓 자국 → (줄, 보표, 보표 오선 y, il, 페이지 그림(인식 좌표 회색), dy, 배율, 그 줄 보표들, 머리 상자들)."""
        sx, sy = (self.sx or 1.0), (self.sy or 1.0)
        pts = [(x / sx, y / sy) for s_ in strokes for x, y in s_]
        if not pts:
            return None
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        best = None
        for line, grp in (self.staff_groups or {}).items():
            for k, ys in enumerate(grp):
                il_ = (ys[4] - ys[0]) / 4.0
                if ys[0] - 6 * il_ <= cy <= ys[4] + 6 * il_:
                    d_ = abs(cy - (ys[0] + ys[4]) / 2.0)
                    if best is None or d_ < best[0]:
                        best = (d_, line, k + 1, list(ys), il_)
        if best is None:
            return None
        _d, line, sn, ys, il = best
        meas = [m for m in self.score.measures if m.system == line]
        if not meas:
            return None
        dy = 0.0
        if getattr(self, "_book_state", None) is not None:
            pg = getattr(meas[0], "page", 1)
            img = self._book_job.page_image(pg)
            dy = self._book_state.pages[pg - 1].dy_orig
        else:
            img = self.original_bgr
        from omr_bridge import ink_check as ik
        gray = ik._gray(img, 1.0 / sx)
        grp = self.staff_groups.get(line) or []
        heads = [(e.src_bbox[0], e.src_bbox[1] - dy, e.src_bbox[2], e.src_bbox[3])
                 for m in meas for e in m.elements if e.kind == "note" and e.src_bbox]
        return {"line": line, "sn": sn, "ys": ys, "il": il, "gray": gray, "dy": dy, "sx": sx, "sy": sy,
                "grp_page": [[v - dy for v in s_] for s_ in grp], "heads": heads, "meas": meas,
                "strokes_page": [[(x / sx, y / sy - dy) for x, y in s_] for s_ in strokes]}

    def _compute_diff_marks(self):
        """[v4.10] 붓 후보 자리 다시 찾기: 지금 악보를 그려 원본과 겹쳐 보고 원본에만 있는 잉크 덩어리를 표시한다.
        한 페이지 악보에서만(여러 페이지 악보는 인식 단계의 미인식 잉크 표시만 쓴다). 실패해도 편집은 계속."""
        self._diff_marks = []
        self._diff_dirty = False
        if self.score is None or self.original_bgr is None or getattr(self, "_book_state", None) is not None:
            return
        try:
            from omr_bridge import verify as _vf
            self._diff_marks = _vf.brush_candidates(
                self.score, self.original_bgr, self.by_sys, self.sx, self.sy, self.barlines, self.staff_groups,
                self.staff_left, getattr(self, "staff_clefs", None), getattr(self, "staff_parts", None),
                self.header_items)
        except Exception:
            import traceback
            traceback.print_exc()
            self._diff_marks = []

    def _on_brush_toggled_marks(self, on):
        """붓을 켤 때, 편집으로 악보가 바뀌었으면 후보 자리를 다시 찾는다."""
        if on and getattr(self, "_diff_dirty", True) and self.score is not None:
            self._compute_diff_marks()
            self._refresh_ink_marks()

    def _on_mark_clicked(self, x, y, w, h):
        """[v4.10] 주황 점선(읽히지 않은 잉크 자리)을 클릭 — 그 상자를 붓으로 칠한 것처럼 읽어 후보를 보여 준다."""
        if self.score is None or self.layout_obj is None:
            return
        ip = self.image_panel
        if not ip._brush_on:
            ip.set_brush(True)
        strokes, r = ip.fill_box_strokes(x, y, w, h)
        # 차이에서 나온 자리면 상자 전체가 아니라 '원본에만 있는 잉크'만 칠한다(옆 음표의 기둥이 섞이지 않게)
        sx_, sy_ = getattr(self, "sx", 1.0), getattr(self, "sy", 1.0)
        for d in getattr(self, "_diff_marks", None) or []:
            if abs(d[0] * sx_ - x) < 1.5 and abs(d[1] * sy_ - y) < 1.5 and len(d) > 5 and d[5]:
                strokes = [[(a_ * sx_, yy_ * sy_), (b_ * sx_, yy_ * sy_)] for a_, b_, yy_ in d[5]]
                r = 3.0 * sx_
                break
        ip._strokes = [list(s) for s in strokes]
        ip._painting = False
        ip.update()
        self._on_brush_done(strokes, r)

    def _on_brush_done(self, strokes, radius):
        if self.score is None or self.layout_obj is None:
            return
        ctx = self._brush_context(strokes)
        if ctx is None:
            self._set_status("붓: 오선 근처를 칠하세요")
            return
        from omr_bridge import brush
        res = brush.analyze(ctx["gray"], ctx["strokes_page"], radius / ctx["sx"], ctx["grp_page"], ctx["heads"],
                            ctx["il"])
        if res is None:
            self._set_status("붓: 칠한 곳에서 잉크를 찾지 못했습니다 — 기호 위를 덮어 칠하세요")
            self.image_panel.show_candidates([])
            return
        ctx["res"] = res
        self._brush_ctx = ctx
        top = res["cands"][0][1]
        cands = res["cands"][:8]
        # [v3.00] 글자처럼 보이면(또는 보표 위 글자) '코드 이름…'을 둘째 자리에 — 고르면 입력 칸
        if res["geom"].get("texty") or any(k in ("cresc_text", "dim_text", "decresc_text", "dc", "ds", "fine")
                                           for k, _ in cands[:3]):
            cands = cands[:1] + [("chord", 0.0)] + cands[1:7]
        else:
            cands = cands[:7] + [("chord", 0.0)]
        items = [(k, self._BRUSH_LABELS[k], self._brush_icon(k), sc_ >= top - 1e-6 and k != "chord")
                 for k, sc_ in cands]                            # 가능성 높은 것부터
        self.image_panel.show_candidates(items)
        self._set_status("붓: 원하는 기호를 고르세요(왼쪽일수록 가능성이 높음) — 계속 칠하면 영역이 넓어집니다, "
                         "Esc로 지우고 붓 끄기")

    def _brush_notes_on_line(self, ctx, staff_no=None):
        out = []
        for m in ctx["meas"]:
            for e in m.elements:
                if e.kind == "note" and e.src_bbox and (staff_no is None or e.staff_no == staff_no):
                    out.append((m, e, e.src_bbox[0] + e.src_bbox[2] / 2.0, e.src_bbox[1] + e.src_bbox[3] / 2.0))
        return out

    def _apply_brush(self, key):
        ctx = getattr(self, "_brush_ctx", None)
        if ctx is None or self.score is None:
            return
        res, il, dy = ctx["res"], ctx["il"], ctx["dy"]
        geom = res["geom"]
        cx, cy = geom["center"]
        cy_song = cy + dy
        name = self._BRUSH_LABELS.get(key, key)
        target = None
        if key in ("slur", "tie"):
            poly = geom.get("poly")
            if not poly:
                self._set_status(f"붓: {name}로 읽을 곡선을 찾지 못했습니다")
                return
            p0, p1 = (poly[0][0], poly[0][1] + dy), (poly[-1][0], poly[-1][1] + dy)
            # 보표는 '붙는 음'으로 정한다(기둥 아래 이음줄은 아래 보표에 더 가까울 수 있다) — 시작 음은 그 줄의 모든
            # 보표에서, 끝 음은 시작 음과 같은 보표에서
            notes = self._brush_notes_on_line(ctx)

            def near(p, staff=None):
                # 이음줄 끝은 머리 옆이나 기둥 끝에 붙는다 — 둘 다 붙는 자리로 보고 가장 가까운 음
                c = []
                for m, e, nx, ny in notes:
                    if staff is not None and e.staff_no != staff:
                        continue
                    anchors = [(nx, ny)]
                    st_ = getattr(e, "src_stem", None)
                    if st_:
                        anchors += [(st_[0], st_[1]), (st_[0], st_[2])]
                    dmin = min(abs(ax - p[0]) + 0.6 * abs(ay - p[1]) for ax, ay in anchors)
                    if dmin <= 3.5 * il:
                        c.append((dmin, m, e))
                return min(c, key=lambda t: t[0]) if c else None
            a_ = near(p0)
            b_ = near(p1, a_[2].staff_no) if a_ else None
            if not a_ or not b_ or a_[2] is b_[2]:
                self._set_status(f"붓: {name}의 양 끝에 가까운 음표 두 개를 찾지 못했습니다")
                return
            a, b = sorted([a_[2], b_[2]], key=lambda e: (self.score.measures.index(self._find_measure(e)), e.src_x))
            from score.model import Arc
            self._snapshot()
            arc = Arc(start=a, stop=b, kind=key)
            if key == "tie":
                a.tie_start, b.tie_stop = True, True
            else:
                a.slur_start, b.slur_stop = True, True
            arc.src_poly = [(x, y + dy) for x, y in poly]       # 원본 잉크 그대로(곡 좌표)
            self.score.arcs.append(arc)
            target = a
        elif key in ("cresc", "decresc"):
            x1, x2, yc, hh = geom["hairpin"]
            notes = self._brush_notes_on_line(ctx)
            if not notes:
                return
            start = min(notes, key=lambda t: abs(t[2] - x1))[1]
            stop = min(notes, key=lambda t: abs(t[2] - x2))[1]
            from score.model import Hairpin
            self._snapshot()
            self.score.hairpins.append(Hairpin(start=start, stop=stop, kind=key, auto=False,
                                               x1=x1, x2=x2, y=yc + dy, h=hh))
            target = start
        elif key in ("staccato", "tenuto", "accent", "marcato", "staccatissimo"):
            if key in ("accent", "marcato", "staccatissimo") and geom.get("tbox"):
                tb = geom["tbox"]                      # 글꼴 비교에 쓴 잉크(기둥을 지우지 않은 것)의 가운데
                cx, cy_song = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0 + dy
            notes = [t for t in self._brush_notes_on_line(ctx) if abs(t[2] - cx) <= 1.2 * il]
            if not notes:
                self._set_status(f"붓: {name}를 붙일 음표(같은 세로줄)를 찾지 못했습니다")
                return
            m, e, hx, hy = min(notes, key=lambda t: abs(t[3] - cy_song))
            self._snapshot()
            e.articulations = list(e.articulations or []) + [
                {"kind": key, "above": cy_song < hy, "dx": cx - hx, "dy": cy_song - hy}]
            target = e
        elif key == "chord":
            tb = geom.get("tbox") or geom["box"]
            tcx, tcy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0 + dy
            grp = self.staff_groups.get(ctx["line"]) or []
            below = [k + 1 for k, ys in enumerate(grp) if ys[0] >= tcy - 0.5 * il]
            sn_ = min(below) if below else ctx["sn"]
            notes = self._brush_notes_on_line(ctx, sn_) or self._brush_notes_on_line(ctx)
            if not notes:
                return
            m, e, hx, hy = min(notes, key=lambda t: abs(t[2] - tcx))
            guess = self._ocr_guess(ctx, tb)
            h = self._ask_chord(guess, e)
            if h is None:
                return
            self._snapshot()
            if h == "":
                e.harmony = None
            else:
                h.update({"x": tcx, "y": tcy, "src": "user"})
                e.harmony = h
            target = e
        elif key in ("tremolo1", "tremolo2", "tremolo3"):
            tb = geom.get("tbox") or geom["box"]
            tcx, tcy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0 + dy
            notes = [t for t in self._brush_notes_on_line(ctx) if abs(t[2] - tcx) <= 1.6 * il]
            if not notes:
                self._set_status("붓: 트레몰로를 붙일 음표(같은 세로줄)를 찾지 못했습니다")
                return
            m, e, hx, hy = min(notes, key=lambda t: abs(t[3] - tcy))
            self._snapshot()
            e.articulations = list(e.articulations or []) + [
                {"kind": "tremolo", "n": int(key[-1]), "above": tcy < hy, "dx": tcx - hx, "dy": tcy - hy}]
            target = e
        elif key in ("cresc_text", "dim_text", "decresc_text"):
            tb = geom.get("tbox") or geom["box"]
            tcx, tcy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0 + dy
            grp = self.staff_groups.get(ctx["line"]) or []
            above = [k + 1 for k, ys in enumerate(grp) if (ys[0] + ys[4]) / 2.0 <= tcy + 2 * il]
            sn_ = max(above) if above else 1
            notes = self._brush_notes_on_line(ctx, sn_) or self._brush_notes_on_line(ctx)
            if not notes:
                return
            m, e, hx, hy = min(notes, key=lambda t: abs(t[2] - tcx))
            # 효과 범위: 같은 보표에서 다음 셈여림 표시 앞까지(없으면 두 마디)
            seq = [(mm, x) for mm in self.score.measures for x in sorted(mm.elements, key=lambda q: q.src_x)
                   if x.staff_no == e.staff_no and x.kind == "note"]
            idx = next((i for i, (mm, x) in enumerate(seq) if x is e), 0)
            stop = e
            for mm, x in seq[idx + 1:]:
                if getattr(x, "dynamic", None) or mm.number > m.number + 2:
                    break
                stop = x
            from score.model import Hairpin
            self._snapshot()
            hp = Hairpin(start=e, stop=stop, kind="cresc" if key == "cresc_text" else "decresc", auto=False)
            hp.text = {"cresc_text": "cresc.", "dim_text": "dim.", "decresc_text": "decresc."}[key]
            hp.tx, hp.ty = tcx, tcy
            self.score.hairpins.append(hp)
            target = e
        elif key == "ped_line":
            if "ped_line" not in geom:
                self._set_status("붓: 페달 선(가로선과 양 끝 갈고리)으로 읽을 모양을 찾지 못했습니다")
                return
            x1_, x2_, ly_, notches = geom["ped_line"]
            notes = self._brush_notes_on_line(ctx)
            if not notes:
                return
            near = lambda x: min(notes, key=lambda t: abs(t[2] - x))[1]
            self._snapshot()
            if getattr(self.score, "pedals", None) is None:
                self.score.pedals = []
            self.score.pedals.append({"style": "line", "start": near(x1_), "stop": near(x2_),
                                      "changes": [near(c) for c in notches], "changes_x": list(notches),
                                      "x1": x1_, "y1": ly_ + dy, "x2": x2_, "y2": ly_ + dy, "line": ctx["line"]})
            target = near(x1_)
        elif key == "gliss":
            if "gliss" not in geom:
                self._set_status("붓: 글리산도로 읽을 곧은 선을 찾지 못했습니다")
                return
            gx1, gy1, gx2, gy2 = geom["gliss"]
            notes = self._brush_notes_on_line(ctx)
            a_ = min(notes, key=lambda t: abs(t[2] - gx1) + 0.5 * abs(t[3] - (gy1 + dy)))
            b_ = min(notes, key=lambda t: abs(t[2] - gx2) + 0.5 * abs(t[3] - (gy2 + dy)))
            if a_[1] is b_[1]:
                self._set_status("붓: 글리산도 양 끝의 음표 두 개를 찾지 못했습니다")
                return
            self._snapshot()
            if getattr(self.score, "glissandos", None) is None:
                self.score.glissandos = []
            self.score.glissandos.append({"start": a_[1], "stop": b_[1], "x1": gx1, "y1": gy1 + dy,
                                          "x2": gx2, "y2": gy2 + dy})
            target = a_[1]
        elif key in ("trill", "mordent", "turn", "breath", "caesura"):
            tb = geom.get("tbox") or geom["box"]
            tcx, tcy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0 + dy
            if key in ("breath", "caesura"):           # 숨표·카이주라는 그 앞(왼쪽) 음 뒤에
                notes = [t for t in self._brush_notes_on_line(ctx) if 0 < tcx - t[2] <= 4.0 * il]
                pick = (lambda ts: max(ts, key=lambda t: t[2] - abs(t[3] - tcy) * 0.01)) if notes else None
            else:
                notes = [t for t in self._brush_notes_on_line(ctx) if abs(t[2] - tcx) <= 1.5 * il]
                pick = (lambda ts: min(ts, key=lambda t: abs(t[3] - tcy))) if notes else None
            if not notes:
                self._set_status(f"붓: {name}를 붙일 음표를 찾지 못했습니다")
                return
            m, e, hx, hy = pick(notes)
            self._snapshot()
            e.articulations = list(e.articulations or []) + [
                {"kind": key, "above": tcy < hy, "dx": tcx - hx, "dy": tcy - hy}]
            target = e
        elif key == "arpeggio":
            tb = geom.get("tbox") or geom["box"]
            tcx = (tb[0] + tb[2]) / 2.0
            y0_, y1_ = tb[1] + dy, tb[3] + dy
            notes = [t for t in self._brush_notes_on_line(ctx)
                     if 0.2 * il <= t[2] - tcx <= 3.0 * il and y0_ - il <= t[3] <= y1_ + il]
            if not notes:
                self._set_status("붓: 아르페지오를 붙일 화음(오른쪽)을 찾지 못했습니다")
                return
            m, e, hx, hy = min(notes, key=lambda t: t[2] - tcx)
            self._snapshot()
            e.articulations = list(e.articulations or []) + [
                {"kind": "arpeggio", "above": True, "dx": tcx - hx, "dy": 0.0, "dy0": y0_ - hy, "dy1": y1_ - hy}]
            target = e
        elif key in ("segno", "coda", "dc", "ds", "fine", "dc_al_fine", "dc_al_coda", "ds_al_fine", "ds_al_coda",
                     "to_coda"):
            tb = geom.get("tbox") or geom["box"]
            tcx, tcy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0 + dy
            mb = [b_ for b_ in self.layout_obj.measure_boxes if b_.measure.system == ctx["line"]]
            sx_ = ctx["sx"]
            inside = [b_ for b_ in mb if b_.bbox[0] / sx_ - il <= tcx <= b_.bbox[2] / sx_ + il]
            if not inside:
                self._set_status(f"붓: {name}가 들어갈 마디를 찾지 못했습니다")
                return
            meas_ = min(inside, key=lambda b_: abs((b_.bbox[0] + b_.bbox[2]) / 2.0 / sx_ - tcx)).measure
            self._snapshot()
            meas_.nav = list(getattr(meas_, "nav", None) or []) + [{"kind": key, "x": tcx, "y": tcy}]
            target = next((e for e in meas_.elements if e.kind == "note"), None)
        elif key in ("8va", "8vb", "15ma"):
            tb = geom.get("tbox") or geom["box"]
            x1_, x2_ = tb[0], tb[2]
            tcy = (tb[1] + tb[3]) / 2.0 + dy
            grp = self.staff_groups.get(ctx["line"]) or []
            if key == "8vb":                           # 보표 아래 → 위 보표
                cand = [k + 1 for k, ys in enumerate(grp) if ys[4] <= tcy]
                sn_ = max(cand) if cand else 1
            else:                                      # 보표 위 → 아래 보표
                cand = [k + 1 for k, ys in enumerate(grp) if ys[0] >= tcy]
                sn_ = min(cand) if cand else 1
            notes = self._brush_notes_on_line(ctx, sn_)
            if not notes:
                return
            a_ = min(notes, key=lambda t: abs(t[2] - x1_))[1]
            b_ = min(notes, key=lambda t: abs(t[2] - x2_))[1]
            self._snapshot()
            if getattr(self.score, "ottavas", None) is None:
                self.score.ottavas = []
            self.score.ottavas.append({"kind": key, "start": a_, "stop": b_, "x1": x1_, "x2": x2_, "y": tcy})
            target = a_
        elif key in ("ped", "ped_up"):
            tb = geom.get("tbox") or geom["box"]
            tcx, tcy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0 + dy
            notes = self._brush_notes_on_line(ctx)
            if not notes:
                return
            near = min(notes, key=lambda t: abs(t[2] - tcx))[1]
            if getattr(self.score, "pedals", None) is None:
                self.score.pedals = []
            if key == "ped":
                self._snapshot()
                self.score.pedals.append({"start": near, "stop": None, "x1": tcx, "y1": tcy, "x2": None, "y2": None,
                                          "line": ctx["line"]})
            else:
                # 같은 줄(시스템)의 열린 페달 중 왼쪽에서 가장 가까운 것(높이는 따지지 않는다)
                open_ = [pd for pd in self.score.pedals if pd.get("stop") is None and pd["x1"] < tcx
                         and pd.get("line", ctx["line"]) == ctx["line"]]
                if not open_:
                    self._set_status("붓: 짝이 될 페달 시작(Ped.)을 왼쪽에서 찾지 못했습니다 — 먼저 Ped.를 넣으세요")
                    return
                self._snapshot()
                pd = max(open_, key=lambda q: q["x1"])
                pd.update({"stop": near, "x2": tcx, "y2": tcy})
            target = near
        elif key == "fermata":
            notes = [t for t in self._brush_notes_on_line(ctx) if abs(t[2] - cx) <= 1.5 * il]
            if not notes:
                self._set_status("붓: 늘임표를 붙일 음표(같은 세로줄)를 찾지 못했습니다")
                return
            m, e, hx, hy = min(notes, key=lambda t: abs(t[3] - cy_song))
            self._snapshot()
            e.fermata = True
            target = e
        elif key in ("sharp", "flat", "natural"):
            # 임시표는 오른쪽 같은 높이의 음에 붙는다(원본에 찍힌 가로 자리도 함께)
            tb = geom.get("tbox") or geom["box"]
            tcx, tcy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0 + dy
            notes = [t for t in self._brush_notes_on_line(ctx)
                     if 0.2 * il <= t[2] - tcx <= 3.5 * il and abs(t[3] - tcy) <= 0.8 * il]
            if not notes:
                self._set_status(f"붓: {name}를 붙일 음표(오른쪽 같은 높이)를 찾지 못했습니다")
                return
            m, e, hx, hy = min(notes, key=lambda t: (abs(t[3] - tcy) > 0.35 * il, t[2] - tcx))
            self._snapshot()
            e.alter = {"sharp": 1, "flat": -1, "natural": 0}[key]
            e.accidental = key
            e.acc_dx = tcx - hx
            target = e
        elif key.startswith("dyn:"):
            # 셈여림은 그 위 보표의 가장 가까운 음에 붙이고, 원본 자리에 그린다
            tb = geom.get("tbox") or geom["box"]
            tcx, tcy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0 + dy
            grp = self.staff_groups.get(ctx["line"]) or []
            above = [k + 1 for k, ys in enumerate(grp) if (ys[0] + ys[4]) / 2.0 <= tcy + 2 * il]
            sn_ = max(above) if above else 1
            notes = [t for t in self._brush_notes_on_line(ctx, sn_)] or self._brush_notes_on_line(ctx)
            if not notes:
                return
            m, e, hx, hy = min(notes, key=lambda t: abs(t[2] - tcx))
            self._snapshot()
            e.dynamic = {"key": key[4:], "x": tcx, "y": tcy, "auto": False}
            target = e
        elif key == "tuplet3":
            target = self._brush_tuplet(ctx, geom)
            if target is None:
                return
        elif key.startswith("ts:"):
            target = self._brush_timesig(ctx, geom, key)
            if target is None:
                return
        elif key == "dot":
            notes = [t for t in self._brush_notes_on_line(ctx)
                     if t[2] < cx and cx - t[2] <= 2.6 * il and abs(t[3] - cy_song) <= 0.8 * il]
            if not notes:
                self._set_status("붓: 점을 붙일 음표(바로 왼쪽)를 찾지 못했습니다")
                return
            m, e, hx, hy = max(notes, key=lambda t: t[2])
            self._snapshot()
            mem = [e] if e.chord_group == -1 else [x for x in m.elements if x.chord_group == e.chord_group]
            for x in mem:
                x.dotted = True
            target = e
        # [v4.10] 방금 붓으로 넣은 자리의 후보 표시는 지운다(칠한 영역과 겹치는 차이 표시)
        try:
            pts = [p_ for s_ in (self.image_panel._strokes or []) for p_ in s_]
            if pts and getattr(self, "_diff_marks", None):
                r_ = float(self.image_panel._brush_r)
                bx0, bx1 = min(p_[0] for p_ in pts) - r_, max(p_[0] for p_ in pts) + r_
                by0, by1 = min(p_[1] for p_ in pts) - r_, max(p_[1] for p_ in pts) + r_
                sx_, sy_ = getattr(self, "sx", 1.0), getattr(self, "sy", 1.0)
                self._diff_marks = [d for d in self._diff_marks
                                    if not (d[0] * sx_ < bx1 and bx0 < d[2] * sx_ and d[1] * sy_ < by1 and by0 < d[3] * sy_)]
        except Exception:
            pass
        self.image_panel.clear_brush()
        self._after_edit(target)
        extra, self._brush_extra = getattr(self, "_brush_extra", ""), ""
        self._set_status(f"붓으로 {name}을(를) 넣었습니다{extra} (Ctrl+Z로 되돌리기) — 붓은 켜져 있습니다(B 또는 Esc로 끄기)")

    def _brush_tuplet(self, ctx, geom):
        """[v3.40] 붓: 셋잇단음표. 괄호가 덮는 음들(괄호 없이 '3'만이면 숫자 가운데의 음 3개)을 셋잇단으로.
        보표는 괄호에 가까운 쪽(기둥 끝 포함) — 이미 셋잇단인 보표는 뒤로. 성부가 쪼개져 있었으면 다시 모은다."""
        tp = geom.get("tuplet")
        if not tp:
            self._set_status("붓: 셋잇단음표 숫자 3을 찾지 못했습니다 — 숫자와 괄호를 함께 칠하세요")
            return None
        il, dy = ctx["il"], ctx["dy"]
        x1_, x2_, dcx, dcy, bracket = tp
        dcy += dy
        notes = self._brush_notes_on_line(ctx)
        by_staff = {}
        for t in notes:
            by_staff.setdefault(t[1].staff_no, []).append(t)

        def pick(ts):
            # 같은 자리(화음)는 한 묶음
            grp = []
            for t in sorted(ts, key=lambda t: t[2]):
                if grp and abs(grp[-1][0][2] - t[2]) <= 0.6 * il and grp[-1][0][0] is t[0]:
                    grp[-1].append(t)
                else:
                    grp.append([t])
            if bracket:
                return [g for g in grp if x1_ - 0.8 * il <= g[0][2] <= x2_ + 0.8 * il]
            if len(grp) < 3:
                return []
            i = min(range(len(grp)), key=lambda k: abs(grp[k][0][2] - dcx))
            i = max(0, min(len(grp) - 3, i - 1))
            return grp[i:i + 3]
        best = None
        for sn_, ts in by_staff.items():
            g = pick(ts)
            if not g:
                continue
            ys_ = []
            for gg in g:
                for m, e, nx, ny in gg:
                    ys_ += [e.src_bbox[1], e.src_bbox[1] + e.src_bbox[3]]
                    st_ = getattr(e, "src_stem", None)
                    if st_:
                        ys_ += [st_[1], st_[2]]
            top_, bot_ = min(ys_), max(ys_)
            dist = (dcy - bot_) if dcy > bot_ else ((top_ - dcy) if dcy < top_ else 0.0)
            if all(e.in_triplet for gg in g for _m, e, _x, _y in gg):
                dist += 1e6                            # 이미 셋잇단인 보표는 마지막 후보
            if best is None or dist < best[0]:
                best = (dist, sn_, g)
        if best is None:
            self._set_status("붓: 셋잇단음표를 붙일 음표들을 찾지 못했습니다")
            return None
        _d, sn_, g = best
        self._snapshot()
        touched = []
        for gg in g:
            for m, e, _x, _y in gg:
                e.in_triplet = True
                if m not in touched:
                    touched.append(m)
        n_merge = sum(1 for m in touched if ed.merge_split_voices(self.score, m, sn_))
        if n_merge:
            self._brush_extra = " · 쪼개져 있던 성부를 하나로 모았습니다"
        return g[0][0][1]

    def _brush_timesig(self, ctx, geom, key):
        """[v3.40] 붓: 박자표. 칠한 자리의 마디(줄 첫머리면 그 줄 첫 마디)에 원본처럼 박자표를 둔다.
        곡 첫 마디면 곡 박자표, 앞 박자와 다르면 그 마디부터 박자가 바뀐다."""
        from omr_bridge import brush as _br
        il, sx_ = ctx["il"], ctx["sx"]
        tb = geom.get("tbox") or geom["box"]
        tcx = (tb[0] + tb[2]) / 2.0
        sym = {"ts:common": "common", "ts:cut": "cut"}.get(key)
        b, bt = {"ts:common": (4, 4), "ts:cut": (2, 2)}.get(key) or _br.TIMESIGS[key]
        mb = sorted([b_ for b_ in self.layout_obj.measure_boxes if b_.measure.system == ctx["line"]],
                    key=lambda b_: b_.bbox[0])
        if not mb:
            return None
        # 박자표는 마디 맨 앞에 있다 — 가운데가 들어 있는 마디, 줄 머리(첫 마디 앞)면 첫 마디
        meas_ = mb[0].measure
        for b_ in mb:
            if b_.bbox[0] / sx_ - il <= tcx:
                meas_ = b_.measure
        ms = sorted(self.score.measures, key=lambda m_: m_.number)
        self._snapshot()
        meas_.ts_printed = (b, bt, sym)
        if meas_ is ms[0]:
            self.score.beats, self.score.beat_type, self.score.ts_symbol = b, bt, sym
            self.score.time_sig_known = True
            meas_.time_sig_change = None
        else:
            prev = (self.score.beats, self.score.beat_type)
            for m_ in ms:
                if m_ is meas_:
                    break
                if getattr(m_, "time_sig_change", None):
                    prev = m_.time_sig_change
            meas_.time_sig_change = (b, bt) if (b, bt) != tuple(prev) else None
        # 줄 단위로 덮어 둔 실효 길이는 새 박자표를 따르게(다음 박자 바뀜 전까지)
        started = False
        for m_ in ms:
            if m_ is meas_:
                started = True
            elif started and getattr(m_, "time_sig_change", None):
                break
            if started:
                m_.target_capacity = None
        return next((e for e in meas_.elements if e.kind == "note"), None) or \
            next((e for m_ in ms for e in m_.elements), None)

    # ------------------------------------------------------------- [v3.10] 가사 흘려 넣기
    def _lyric_pos(self, e, row):
        """붙여 넣은 가사 자리(곡 좌표): 음 머리 가운데, 그 보표 맨 아랫줄 2.6칸 아래 + 절마다 1.9칸."""
        m = self._find_measure(e)
        grp = (self.staff_groups or {}).get(m.system) or []
        ys = grp[e.staff_no - 1] if e.staff_no - 1 < len(grp) else (grp[0] if grp else [0, 10, 20, 30, 40])
        il = (ys[4] - ys[0]) / 4.0
        x = (e.src_bbox[0] + e.src_bbox[2] / 2.0) if e.src_bbox else e.src_x
        return x, ys[4] + (2.6 + 1.9 * row) * il, 1.3 * il

    def _lyric_seq(self, e, voice_only=False, skip_tied=True, skip_slur=True):
        from score import lyrics_fill as LF
        m = self._find_measure(e)
        key = LF.staff_key(getattr(self, "staff_parts", None), m.system, e.staff_no)
        return LF.note_sequence(self.score, e.staff_no, (m, e), e.voice_no if voice_only else None,
                                skip_tied, skip_slur, staff_parts=getattr(self, "staff_parts", None), key=key)

    def _ocr_available(self):
        """이 PC의 Tesseract와 한국어 글자 데이터 위치(없으면 None)."""
        import shutil
        exe = shutil.which("tesseract") or next((p for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe", "/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract",
                                                              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe")
                                                if os.path.exists(p)), None)
        if not exe:
            return None
        dirs = [os.environ.get("TESSDATA_PREFIX"),
                os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "Sprouty", "tessdata"),
                os.path.join(os.path.dirname(exe), "tessdata"), "/usr/share/tesseract-ocr/5/tessdata"]
        for d in dirs:
            if d and os.path.exists(os.path.join(d, "kor.traineddata")):
                return exe, d
        return None

    def _ocr_lyrics(self, start_e, row):
        """[v3.10 B안] 원본의 가사 줄(보표 아래 띠)을 줄마다 잘라 Tesseract(kor+eng, 한 줄 모드)로 읽어 잇는다."""
        ocr = self._ocr_available()
        if not ocr:
            return ""
        exe, tdir = ocr
        import subprocess
        import tempfile
        import cv2
        from score import lyrics_fill as LF
        m0 = self._find_measure(start_e)
        key = LF.staff_key(getattr(self, "staff_parts", None), m0.system, start_e.staff_no)
        out = []
        lines = sorted({m.system for m in self.score.measures if m.system >= m0.system})
        for line in lines:
            meas = [m for m in self.score.measures if m.system == line]
            grp = (self.staff_groups or {}).get(line) or []
            sns = [k + 1 for k in range(len(grp)) if LF.staff_key(getattr(self, "staff_parts", None), line, k + 1) == key]
            if not sns:
                continue
            ys = grp[sns[0] - 1]
            il = (ys[4] - ys[0]) / 4.0
            dy = 0.0
            if getattr(self, "_book_state", None) is not None:
                pg = getattr(meas[0], "page", 1)
                img = self._book_job.page_image(pg)
                dy = self._book_state.pages[pg - 1].dy_orig
            else:
                img = self.original_bgr
            xs = [e.src_bbox[0] for m in meas for e in m.elements if e.src_bbox and e.staff_no in sns]
            x0 = int(min(xs) - il) if xs else 0
            if line == m0.system and start_e.src_bbox:
                x0 = int(start_e.src_bbox[0] - il)
            x1 = int(max(xs) + 3 * il) if xs else img.shape[1]
            # 보표 아래 띠에서 가사 줄(글자가 빽빽한 가로 줄)을 찾아 그 절의 줄만 자른다
            # (오선·기둥이 섞인 넓은 띠를 통째로 읽으면 글자 인식이 무너진다 — 교회 5마디 실측)
            b0, b1 = int(ys[4] + 0.8 * il - dy), int(ys[4] + (6.0 + 1.9 * row) * il - dy)
            g = img[max(0, b0):max(0, b1), max(0, x0):x1]
            if g.size == 0:
                continue
            g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY) if g.ndim == 3 else g
            ink = (g < 140).astype("uint8")
            vl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(0.9 * il)))))
            prof = (ink & (1 - vl)).mean(axis=1)
            h = max(4, int(1.7 * il))
            rows_found = []
            r = 0
            while r < len(prof) - h:
                win = prof[r:r + h].sum()
                if win > 0.04 * h and (not rows_found or r - rows_found[-1][0] >= int(1.4 * il)):
                    # 이 근처에서 가장 빽빽한 자리
                    best = max(range(r, min(len(prof) - h, r + int(1.2 * il)) + 1), key=lambda q: prof[q:q + h].sum())
                    rows_found.append((best, prof[best:best + h].sum()))
                    r = best + h
                else:
                    r += 1
            if row >= len(rows_found):
                continue
            top = rows_found[row][0]
            crop = g[max(0, top - 4):top + h + 4]
            crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            fn = os.path.join(tempfile.gettempdir(), "sprouty_lyric_ocr.png")
            cv2.imwrite(fn, crop)
            try:
                r_ = subprocess.run([exe, fn, "-", "-l", "kor", "--psm", "7", "--tessdata-dir", tdir],
                                    capture_output=True, text=True, timeout=15)
                t = (r_.stdout or "").strip()
            except Exception:
                t = ""
            # 늘임 표시(ㅡ - ~ _)는 뺀다 — 흘려 넣기가 그 음(이음줄·붙임줄 뒷음)을 이미 건너뛴다
            t = " ".join(w_ for w_ in t.split() if w_ not in ("ㅡ", "-", "—", "–", "―", "~", "_", "=", "|"))
            # 가사 줄이 아닌 곳(반주 보표 아래 등)에서 나온 엉뚱한 글자는 버린다: 한글이 60% 미만인 줄,
            # 숫자·기호만으로 된 조각
            toks = [w_ for w_ in t.split() if any("\uac00" <= c <= "\ud7a3" or c.isalpha() for c in w_)
                    and not w_.isdigit()]
            chars = "".join(toks)
            hangul = sum(1 for c in chars if "\uac00" <= c <= "\ud7a3")
            t = " ".join(toks) if chars and hangul / len(chars) >= 0.6 and len(toks) >= 3 else ""
            if t:
                out.append(t)
        return "\n".join(out)

    def _lyrics_fill(self):
        """[v3.10] L: 고른 음부터 붙여 넣은 가사를 차례로."""
        e = self.selected
        if e is None or self.score is None or e.kind != "note":
            self._set_status("가사 흘려 넣기: 먼저 가사를 시작할 음을 고르세요 (그다음 L)")
            return
        m = self._find_measure(e)
        rows_used = {r for mm in self.score.measures for x in mm.elements for r in (x.lyrics or {})}
        ocr_fn = (lambda row: self._ocr_lyrics(e, row)) if self._ocr_available() else None
        dlg = LyricsFillDialog(self, f"{m.number}마디 {e.step}{e.octave}", rows_used, ocr_fn)
        if dlg.exec() != QDialog.Accepted:
            return
        from score import lyrics_fill as LF
        syl = LF.tokenize(dlg.text.toPlainText())
        if not syl:
            return
        row = dlg.verse.currentData()
        seq = self._lyric_seq(e, dlg.voice_only.isChecked(), dlg.skip_tied.isChecked(), dlg.skip_slur.isChecked())
        self._snapshot()
        used, left = LF.fill(seq, syl, row, lambda x: self._lyric_pos(x, row), dlg.replace.isChecked())
        self._after_edit(e)
        msg = f"가사 {used}음절을 {row + 1}절에 넣었습니다"
        if left:
            msg += f" — 음이 모자라 {left}음절이 남았습니다"
        self._set_status(msg + " (어긋난 곳은 그 음을 고르고 Alt+→/←, Ctrl+Z로 되돌리기)")

    def _lyrics_shift(self, direction):
        """[v3.10] Alt+→/←: 고른 음부터 그 절 가사를 한 칸 밀기/당기기."""
        e = self.selected
        if e is None or self.score is None or e.kind != "note":
            return
        from score import lyrics_fill as LF
        rows = sorted((e.lyrics or {}).keys())
        if not rows:
            # 고른 음에 가사가 없으면 뒤쪽에서 처음 나오는 절
            seq0 = self._lyric_seq(e)
            rows = next((sorted((x.lyrics or {}).keys()) for _m, x in seq0 if x.lyrics), [0])
        row = rows[0]
        seq = self._lyric_seq(e)
        self._snapshot()
        LF.shift(seq, row, direction, lambda x: self._lyric_pos(x, row))
        self._after_edit(e)
        self._set_status(f"{row + 1}절 가사를 이 음부터 한 칸 {'뒤로 밀었습니다' if direction > 0 else '앞으로 당겼습니다'}")

    # ------------------------------------------------------------- [v3.00] 기타 코드
    def _ocr_guess(self, ctx, tb):
        """붓으로 칠한 코드 글자를 이 PC의 Tesseract(있을 때만)로 미리 읽어 본다. 없으면 빈 글자."""
        import shutil
        import subprocess
        import tempfile
        exe = shutil.which("tesseract") or next((p for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe", "/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract",)
                                                if os.path.exists(p)), None)
        if not exe:
            return ""
        try:
            import cv2
            x0, y0, x1, y1 = [int(v) for v in tb]
            g = ctx["gray"][max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4]
            g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            fn = os.path.join(tempfile.gettempdir(), "sprouty_chord_ocr.png")
            cv2.imwrite(fn, g)
            out = subprocess.run([exe, fn, "-", "--psm", "7", "-c",
                                  "tessedit_char_whitelist=ABCDEFGabdgijmnosu#+-/()0123456789"],
                                 capture_output=True, text=True, timeout=8)
            return (out.stdout or "").strip().replace(" ", "")
        except Exception:
            return ""

    def _ask_chord(self, initial, e):
        """코드 입력 칸. 반환: 코드 dict, ""(지우기), None(취소)."""
        from score import chords as _chd
        cur = _chd.display(getattr(e, "harmony", None)).replace("♯", "#").replace("♭", "b") if \
            getattr(e, "harmony", None) else ""
        text = initial or cur
        while True:
            t, ok = QInputDialog.getText(self, "기타 코드",
                                         "코드 이름 (예: C, Am7, F#m7b5, G/B, Dsus4) — 비우면 지웁니다:",
                                         text=text)
            if not ok:
                return None
            t = t.strip()
            if not t:
                return ""
            h = _chd.parse(t)
            if h:
                return h
            QMessageBox.information(self, "기타 코드", f"'{t}'은(는) 코드 이름으로 읽을 수 없습니다. 다시 입력해 주세요.")
            text = t

    def _edit_chord(self):
        """[v3.00] K: 고른 음(또는 쉼표)에 기타 코드 넣기·고치기·지우기(원본 자리를 모르면 보표 위에 자동 배치)."""
        e = self.selected
        if e is None or self.score is None:
            self._set_status("기타 코드: 먼저 코드를 붙일 음을 고르세요 (그다음 K)")
            return
        h = self._ask_chord("", e)
        if h is None:
            return
        self._snapshot()
        if h == "":
            e.harmony = None
            self._after_edit(e)
            self._set_status("기타 코드를 지웠습니다")
            return
        old = getattr(e, "harmony", None) or {}
        h.update({"x": old.get("x"), "y": old.get("y"), "src": "user"})
        e.harmony = h
        self._after_edit(e)
        from score import chords as _chd
        self._set_status(f"기타 코드 {_chd.display(h)}를 넣었습니다 (Ctrl+Z로 되돌리기)")

    def _pick_from_original(self, wx, wy):
        """[v2.40 원본에서 기호 넣기] 원본 이미지를 두 번 클릭하면, 그 자리의 음표 머리를 원본에서 읽어
        (음높이 = 오선 위치·음자리표·조표, 빈/검정 머리, 기둥 방향 → 성부, 박 위치 → 길이) 편집 악보에 넣는다.
        이미 인식된 음이면 그 음을 고른다. 되돌리기(Ctrl+Z)로 취소."""
        if self.score is None or self.layout_obj is None:
            return
        sx, sy = (self.sx or 1.0), (self.sy or 1.0)
        x, y = wx / sx, wy / sy                           # 원본(인식) 좌표
        # 이미 인식된 음 위면 그 음을 고른다
        for g in self.layout_obj.glyphs:
            e = g.element
            if e.kind == "note" and e.src_bbox:
                bx, by, bw, bh = e.src_bbox
                # 머리 가운데 가까이일 때만 '이미 인식된 음'(빽빽한 화음에서 이웃 칸까지 상자에 걸리지 않게)
                if bx - 2 <= x <= bx + bw + 2 and abs(y - (by + bh / 2.0)) <= bh * 0.3:
                    self.on_select(g)
                    self._set_status("이미 인식된 음입니다 — 골라 두었습니다")
                    return
        # 어느 줄·보표·마디인가
        best = None
        for line, grp in (self.staff_groups or {}).items():
            for k, ys in enumerate(grp):
                il_ = (ys[4] - ys[0]) / 4.0
                if ys[0] - 4.5 * il_ <= y <= ys[4] + 4.5 * il_:
                    dist = abs(y - (ys[0] + ys[4]) / 2.0)
                    if best is None or dist < best[0]:
                        best = (dist, line, k + 1, list(ys), il_)
        if best is None:
            self._set_status("원본에서 넣기: 오선 위나 가까이를 두 번 클릭하세요")
            return
        _d, line, sn, ys, il = best
        meas = None
        for mb in self.layout_obj.measure_boxes:
            if mb.measure.system == line and mb.bbox[0] <= wx <= mb.bbox[2]:
                meas = mb.measure
                break
        if meas is None:
            self._set_status("원본에서 넣기: 마디 안을 두 번 클릭하세요")
            return
        # [v2.56] 이미 있는 화음 바로 옆(같은 열)을 두 번 클릭하면 → 그 화음에 음을 더한다. 길이·성부·기둥은 화음이
        # 정하고, 원본에서 읽는 것은 높이(음높이)뿐 — 머리가 빽빽이 겹친 화음("교회" 41마디)에서도 빈/검정 판정 없이.
        col = [e for e in meas.elements if e.kind == "note" and e.staff_no == sn and e.src_bbox and not e.is_grace
               and abs((e.src_bbox[0] + e.src_bbox[2] / 2.0) - x) <= 0.8 * e.src_bbox[2]]
        if col:
            rep = min(col, key=lambda e: abs((e.src_bbox[1] + e.src_bbox[3] / 2.0) - y))
            ry = rep.src_bbox[1] + rep.src_bbox[3] / 2.0
            steps = int(round((ry - y) / (il / 2.0)))
            if steps == 0:
                self.on_select(self._glyph_of(rep))
                self._set_status("이미 인식된 음입니다 — 골라 두었습니다")
                return
            self._snapshot()
            new = ed.add_chord_note(meas, rep, step_delta=steps)
            if new is None:
                self._set_status("원본에서 넣기: 화음에 음을 더하지 못했습니다")
                return
            new.src_bbox = (rep.src_bbox[0], int(ry - steps * il / 2.0 - rep.src_bbox[3] / 2.0),
                            rep.src_bbox[2], rep.src_bbox[3])
            new.src_x = rep.src_x
            new.from_original = True
            self._after_edit(new)
            self._set_status(f"원본에서 읽어 화음에 넣었습니다: {meas.number}마디 · {new.step}{new.octave} "
                             f"({rep.duration_base}분, 화음과 같은 길이) — 임시표는 ♯♭♮ 버튼으로")
            return
        dy = 0.0
        if getattr(self, "_book_state", None) is not None:
            pg = getattr(meas, "page", 1)
            img = self._book_job.page_image(pg)
            dy = self._book_state.pages[pg - 1].dy_orig
        else:
            img = self.original_bgr
        from omr_bridge import ink_check as ik
        up = 1.0 / sx
        gray = ik._gray(img, up)
        got = ik.read_head_at(gray, x, y - dy, [v - dy for v in ys], il)
        if got is None:
            self._set_status("원본에서 넣기: 그 자리에서 음표 머리를 찾지 못했습니다 — 머리 가운데를 두 번 클릭하세요")
            return
        (x0, y0, x1, y1), hollow = got
        blob = {"kind": "head", "box": (x0, y0 + dy, x1, y1 + dy), "line": line, "staff_no": sn, "il": il,
                "hollow": hollow, "measure": meas.number}
        sugs = ik.make_suggestions(self.score, [blob], img, self.staff_groups, up, y_off=dy)
        if not sugs:
            self._set_status(f"원본에서 넣기: {meas.number}마디에서 이 음이 들어갈 박을 정하지 못했습니다")
            return
        sug = sugs[0]
        self._snapshot()
        el = ik.accept_suggestion(self.score, sug)
        el.from_original = True
        self._after_edit(el)
        self._set_status("원본에서 읽어 넣었습니다: " + sug["text"]
                         + f" — {'빈' if hollow else '검정'} 머리, 성부 {el.voice_no}"
                         + "  (틀리면 Ctrl+Z, 길이·음높이는 그대로 고칠 수 있음)")

    def _hl_rect(self, g):
        """[v2.32] 선택 테두리: 음표는 머리만(기둥 제외), 쉼표는 쉼표 기호 둘레. 클릭 칸(g.cell)은 넓게 두고
        보이는 테두리만 좁힌다 — 기둥까지 두른 큰 상자가 어느 음인지 헷갈렸다."""
        lay = self.layout_obj
        sp = lay.staff_space if lay is not None else 20.0
        e = g.element
        if e.kind == "note":
            hw = (e.src_bbox[2] * self.sx / 2.0) if e.src_bbox else sp * 0.62
            if e.is_grace:
                hw *= 0.75
            pad = sp * 0.18
            return (g.x - hw - pad, g.y - sp * 0.5 - pad, g.x + hw + pad, g.y + sp * 0.5 + pad)
        if e.kind == "rest":
            hh = sp * (1.6 if e.duration_base in (4, 8, 16, 32) else 0.7)
            return (g.x - sp * 0.75, g.y - hh, g.x + sp * 0.75, g.y + hh)
        return g.cell

    def _on_play_tick(self):
        if not self._playing:
            return
        t = __import__("time").monotonic() - self._play_t0
        if t >= self._play_total + 0.3:
            self._stop_playback()
            return
        # 현재 시각의 이벤트 찾기(선형 진행이므로 인덱스 전진만)
        idx = self._play_idx
        n = len(self._play_events)
        while idx + 1 < n and self._play_events[idx + 1][0] <= t:
            idx += 1
        if idx != self._play_idx and 0 <= idx < n:
            self._play_idx = idx
            # [v2.32] 같은 순간에 시작하는 음은 모든 파트를 함께 표시한다. 예전엔 '가장 나중 이벤트' 하나만
            # 골라, 합창·피아노가 동시에 시작하면 목록 순서에 따라, 피아노 8분음표처럼 자주 시작하는 쪽이
            # 있으면 그쪽으로 표시가 위아래로 오갔다.
            st0 = self._play_events[idx][0]
            same = [ev for ev in self._play_events if abs(ev[0] - st0) < 1e-3 and ev[3] is not None]
            glyphs = [g for ev in same if (g := self._glyph_of(ev[3])) is not None]
            mnum = self._play_events[idx][4]
            if glyphs:
                top = min(glyphs, key=lambda g: g.y)
                self.on_select(top)                           # 화면 따라가기(맨 위 음 기준)
                self.score_panel.set_multi_selection([self._hl_rect(g) for g in glyphs if g is not top])
            self._set_status(f"재생 중 · 마디 {mnum}  ({t:.1f}s)")

    def _set_status(self, text):
        self.status.setText(text)


class _FileOpenFilter(QObject):
    """[v3.30] 맥: 앱 아이콘에 악보를 끌어다 놓거나 '다음으로 열기'로 열면 실행 인자가 아니라
    FileOpen 이벤트로 들어온다 — 창이 준비되면 그 파일을 연다."""

    def __init__(self):
        super().__init__()
        self.window = None
        self.pending = None

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.FileOpen:
            path = ev.file()
            if self.window is not None and path:
                QTimer.singleShot(0, lambda p=path: self.window.load_image(p))
            else:
                self.pending = path
            return True
        return False


def main():
    if "--verify" in sys.argv:                 # [v3.60] 화면 없이 폴더 일괄 검증: --verify 폴더 [--limit N] [--out 폴더]
        i = sys.argv.index("--verify")
        args = sys.argv[i + 1:]
        if args and not args[0].startswith("--"):
            from omr_bridge import verify as vf
            limit = int(args[args.index("--limit") + 1]) if "--limit" in args else 0
            out = args[args.index("--out") + 1] if "--out" in args else None

            def prog(k, n, name, stage):
                try:
                    print(f"[{k}/{n}] {name} {stage}", flush=True)
                except Exception:
                    pass                        # 창 프로그램(콘솔 없음)에서는 출력할 곳이 없다
            res = vf.run_batch(args[0], out, limit=limit, progress=prog, app_version=APP_VERSION)
            try:
                print("보내기용 zip:", res["zip"])
            except Exception:
                pass
            sys.exit(0)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    _here = os.path.dirname(os.path.abspath(__file__))
    for _ic in ("app_icon.png", "app_icon.ico"):          # [v3.30] 맥은 .ico를 못 읽는 경우가 있어 png 먼저
        _icon_path = os.path.join(_here, "src", "icon", _ic)
        if os.path.exists(_icon_path):
            app.setWindowIcon(QIcon(_icon_path))
            break
    opener = _FileOpenFilter()
    app.installEventFilter(opener)
    window = ScoreEditorWindow()
    opener.window = window
    window.show()
    # 시작할 때 넘겨받은 악보(Windows: 아이콘에 끌어다 놓기·'연결 프로그램', 맥: FileOpen)
    first = opener.pending or next((a for a in sys.argv[1:]
                                    if a.lower().endswith(ScoreEditorWindow._DROP_EXT) and os.path.isfile(a)), None)
    if first:
        QTimer.singleShot(300, lambda: window.load_image(first))
    QTimer.singleShot(4000, lambda: window.check_update(manual=False))     # [v3.50] 새 버전 알림(하루 한 번)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
