# omr/engine.py
# ---------------------------------------------------------------------------
# [방향 2] 구조 우선 + 템플릿 매칭 + 음악 문법 검증 인식 엔진.
#
# 기존 파이프라인(전체 페이지 휴리스틱)의 근본 문제 — 가사·코드 기호가 음악
# 객체와 섞여 검출되는 것 — 를 구조적으로 제거한다.
#
#   1) 페이지 → 시스템(단) 분할: 오선 5줄 묶음을 먼저 찾고, 이후의 모든
#      처리는 '보표 밴드' 내부에서만 수행한다. 가사(아래)·코드 기호(위)는
#      애초에 파이프라인에 들어오지 않는다. 빨간색 코드 기호는 채도 마스크로
#      이진화 전에 제거한다.
#   2) 기호 분류 = Bravura(SMuFL) 글리프 템플릿 매칭: 음표 머리(검정/2분/온음),
#      꾸밈음(축소 머리), 쉼표(온/2분/4분/8분), 임시표(♯/♭/♮), 음자리표,
#      박자표(C·숫자)를 오선 간격 스케일로 렌더링해 정규화 상관으로 찾는다.
#      휴리스틱 면적/비율 판정보다 실사 악보에서 훨씬 안정적이다.
#   3) 음악 문법 검증: 마디선으로 마디를 나눈 뒤 "마디 박자 합 = 박자표"
#      제약으로 검산한다. 한 단의 마디들이 '일관되게 같은 값으로 부족'하면
#      악보에 표기되지 않은 박자 변화(예: 선린교가 5·6단의 2/4)로 추정해
#      오류가 아닌 정보(박자 변화 추정)로 표시한다. 그 외의 부족/초과는
#      보수적 자동 교정(점 복원·빔 단계 재해석)을 1~2회 시도한다.
#
# 출력 계약(기존 모델 계층과 호환):
#   elements : [(line, x, y, w, h, beat, stem_x, beam_group, accidental, is_grace)]
#              beat: 1/2/4/8/16/32, 점음표는 음수. accidental: "sharp"|"flat"|"natural"|None
#   pitches  : elements 와 정렬. 0~20 = F3..E6 인덱스, -1 = 쉼표, None = 미상
#   key      : 플랫 100×n / 샾 10×n / 0
#   time_signature : (분자, 분모) 또는 (0, 0) 미검출
#   barlines : {line: [x, ...]},  barline_types: {line: {x: 종류}}
# ---------------------------------------------------------------------------
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# SMuFL 코드포인트 (score/smufl.py 와 동일 규격 + 박자표 C 추가)
_GLYPH = {
    "gClef": "\uE050",
    "noteheadBlack": "\uE0A4",
    "noteheadHalf": "\uE0A3",
    "noteheadWhole": "\uE0A2",
    "restWhole": "\uE4E3",
    "restHalf": "\uE4E4",
    "restQuarter": "\uE4E5",
    "rest8th": "\uE4E6",
    "rest16th": "\uE4E7",
    "accidentalFlat": "\uE260",
    "accidentalNatural": "\uE261",
    "accidentalSharp": "\uE262",
    "timeSigCommon": "\uE08A",
    "timeSigCutCommon": "\uE08B",   # 알라브레베(¢, cut time)
    **{f"timeSig{i}": chr(0xE080 + i) for i in range(10)},
}

SS = 10.0          # 정규화 후 오선 간격(px)
_PITCH_LO = 0      # F3
_PITCH_HI = 20     # E6


def _font_path():
    """Bravura 폰트 파일 경로. PyInstaller(onefile)로 배포된 실행 파일에서도
    동작하도록 sys._MEIPASS(동결 시 임시 추출 폴더)를 우선 확인한다 —
    score/smufl.py 의 같은 처리와 동일한 이유(관례상 __file__ 기반 경로만
    쓰면 일부 PyInstaller 버전/모드 조합에서 어긋날 수 있어, 방어적으로
    _MEIPASS 를 명시적으로 먼저 시도한다).
    """
    bases = []
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bases.append(sys._MEIPASS)
    bases.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for base in bases:
        for name in ("Bravura.otf", "Bravura.ttf", "Leland.otf"):
            p = os.path.join(base, "src", "fonts", name)
            if os.path.exists(p):
                return p
    return None


# ------------------------------------------------------------------ 템플릿
_TEMPLATE_CACHE: dict = {}


_TEXT_DIGIT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"


def _render_text_digit(ch: str, ss: float, scale: float = 1.0) -> np.ndarray | None:
    """잇단음표 숫자("3" 등)를 이탤릭 텍스트 폰트로 렌더링한다.

    Bravura(음악 기호 폰트)의 tuplet 전용 숫자 글리프(SMuFL U+E880대)가
    이 프로젝트가 쓰는 폰트 서브셋에는 비어 있고, 설사 있어도 실제
    인쇄물의 잇단음표 숫자는 흔히 이탤릭 '텍스트' 폰트로 조판된다(시간
    표(time signature) 숫자와는 다른 글리프 계열) — 그래서 시간표용
    Bravura 숫자 템플릿을 그대로 재사용하면 매칭 점수가 실측 0.2~0.3대
    로 낮게 나온다. 이탤릭 세리프 텍스트로 바꾸니 0.57대로 크게 개선됨.
    """
    key = ("text_digit", ch, round(ss, 2), round(scale, 2))
    if key in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[key]
    size = max(6, int(round(ss * 1.6 * scale)))
    try:
        font = ImageFont.truetype(_TEXT_DIGIT_FONT, size)
    except Exception:
        return None
    pad = size
    img = Image.new("L", (pad * 3, pad * 3), 255)
    d = ImageDraw.Draw(img)
    d.text((pad, pad // 2), ch, font=font, fill=0)
    arr = 255 - np.array(img)
    ys, xs = np.where(arr > 50)
    if len(xs) == 0:
        _TEMPLATE_CACHE[key] = None
        return None
    t = (arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 50).astype(np.uint8) * 255
    _TEMPLATE_CACHE[key] = t
    return t


def _render_glyph(name: str, ss: float = SS, scale: float = 1.0) -> np.ndarray | None:
    """Bravura 글리프를 오선 간격 ss 기준 크기로 렌더링해 이진(잉크=255) 템플릿 반환."""
    key = (name, round(ss, 2), round(scale, 2))
    if key in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[key]
    path = _font_path()
    if path is None:
        return None
    px = int(round(ss * 4 * scale))          # SMuFL: em = 4 staff spaces
    try:
        font = ImageFont.truetype(path, px)
    except Exception:
        return None
    ch = _GLYPH[name]
    pad = px
    img = Image.new("L", (pad * 4, pad * 4), 0)
    d = ImageDraw.Draw(img)
    try:
        d.text((pad, pad * 2), ch, fill=255, font=font, anchor="lm")
    except TypeError:
        # 구형 Pillow: anchor 미지원 — 기본 앵커로 그려도 아래에서 잉크
        # bbox 로 크롭하므로 결과 템플릿은 동일하다.
        d.text((pad, pad), ch, fill=255, font=font)
    arr = np.array(img)
    ys, xs = np.where(arr > 96)
    if len(xs) == 0:
        return None
    t = (arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 96).astype(np.uint8) * 255
    # 속이 꽉 찬(=잉크 bbox 로 자르면 전체가 순수 255, 분산=0) 글리프에만
    # 배경 테두리 1px 을 남긴다. 온쉼표·2분쉼표가 이 경우다 — 이런 '평평한'
    # 템플릿은 cv2.matchTemplate 의 TM_CCOEFF_NORMED 가 0/0 이 되어 이미지의
    # 완전히 빈(흰) 영역에서도 무조건 1.000 을 반환한다(OpenCV 관례). 실측:
    # 이 버그로 온쉼표/2분쉼표 검출이 배경 전체에 가짜 만점 후보를 무더기로
    # 쏟아내, 위치 검증을 우연히 통과한 자리만 '검출'되고 진짜 쉼표는 다른
    # 가짜 후보에 밀려 누락됐다. 테두리를 '모든' 글리프에 붙이면 연결요소
    # 크기 검증(템플릿 크기 기준)이 틀어지므로, 필요한 경우에만 좁게 적용.
    if t.min() == t.max():
        t = np.pad(t, 1, mode="constant", constant_values=0)
    _TEMPLATE_CACHE[key] = t
    return t


def _match(img_bin: np.ndarray, tmpl: np.ndarray, thr: float):
    """정규화 상관 템플릿 매칭 → [(score, cx, cy, w, h)] (NMS 전).

    이진 이미지의 상관 응답은 넓은 '동점 평탄 지대'를 이뤄 임계 이상 후보가
    수십만 개까지 늘어난다(전체 인식 시간의 70%+). 후보를 '국소 최대 ∪
    3px 격자 표본'으로 줄인다 — 검증 우선 설계에서는 최강 후보가 구조
    검증에서 기각될 때 인접 후보가 기회를 얻어야 하므로 순수 국소최대만
    남기면 안 되고, 격자 표본이 평탄 지대 위 검증 기회를 보존한다.
    """
    th, tw = tmpl.shape
    ih, iw = img_bin.shape
    if ih < th or iw < tw:
        return []
    res = cv2.matchTemplate(img_bin, tmpl, cv2.TM_CCOEFF_NORMED)
    above = res >= thr
    kern = np.ones((3, 3), np.uint8)
    kern[1, 1] = 0
    neigh = cv2.dilate(res, kern)          # 중심 제외 이웃 최댓값
    strictmax = res > neigh                # 유일 봉우리만(동점 평탄 지대 제외)
    grid = np.zeros_like(above)
    grid[::2, ::2] = True
    ys, xs = np.where(above & (strictmax | grid))
    return [(float(res[y, x]), x + tw / 2.0, y + th / 2.0, tw, th)
            for y, x in zip(ys, xs)]


def _nms(cands, dx, dy):
    """점수 내림차순 비최대 억제.

    내부 판정을 numpy 로 벡터화했다(수십만 후보 × 유지목록의 순수 파이썬
    이중 루프가 인식 시간의 95%+ 를 차지하던 병목). 탐욕 순서가 같으므로
    결과는 종전 구현과 완전히 동일하다.
    """
    if not cands:
        return []
    out = []
    kx = np.empty(len(cands))
    ky = np.empty(len(cands))
    k = 0
    for c in sorted(cands, key=lambda t: -t[0]):
        if k and bool(np.any((np.abs(kx[:k] - c[1]) <= dx)
                             & (np.abs(ky[:k] - c[2]) <= dy))):
            continue
        out.append(c)
        kx[k] = c[1]
        ky[k] = c[2]
        k += 1
    return out


# ------------------------------------------------------------------ 자료형
@dataclass
class SystemInfo:
    index: int
    line_ys: list           # 5개 오선 y (정규화 좌표, float)
    x0: int = 0              # 보표 좌우 범위
    x1: int = 0
    thickness: float = 1.5   # 오선 두께

    @property
    def top(self):
        return self.line_ys[0]

    @property
    def bottom(self):
        return self.line_ys[4]

    @property
    def ss(self):
        return (self.line_ys[4] - self.line_ys[0]) / 4.0


@dataclass
class Note:
    x: float                 # 머리 중심 x
    y: float                 # 머리 중심 y
    kind: str                # "black" | "half" | "whole" | "grace"
    stem_x: float = -1.0
    stem_top: float = -1.0
    stem_bot: float = -1.0
    stem_dir: int = 0        # +1 위(오른쪽 기둥), -1 아래(왼쪽 기둥), 0 없음
    beams: int = 0           # 빔/깃발 수
    dotted: bool = False
    accidental: str | None = None
    beam_group: int = -1
    pitch: int | None = None
    beat: int = 4            # 최종 박자 코드(점이면 모델 전달 시 음수화)
    conf: float = 0.0
    dot_conf: float = 0.0    # 점 검출 신뢰도(문법 교정 시 참고)
    in_triplet: bool = False  # [잇단음표] "3" 표기가 확인된 셋잇단음표 멤버


@dataclass
class Rest:
    x: float
    y: float
    beat: int                # 1/2/4/8/16
    w: float = 0
    h: float = 0
    dotted: bool = False
    conf: float = 0.0


@dataclass
class SystemResult:
    info: SystemInfo
    notes: list = field(default_factory=list)
    rests: list = field(default_factory=list)
    barlines: list = field(default_factory=list)         # x 리스트
    barline_types: dict = field(default_factory=dict)    # {x: 종류}
    clef_box: tuple | None = None
    time_sig: tuple | None = None
    key_accidentals: int = 0     # +n 샾 / -n 플랫
    measures: list = field(default_factory=list)         # 검증 결과


# ------------------------------------------------------------------ 1. 이진화
def binarize(bgr: np.ndarray) -> np.ndarray:
    """색 기호(빨간 코드 등) 제거 + Otsu 이진화(잉크=255) + 테두리 노이즈 제거."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    colored = (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 60)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = gray.copy()
    gray[colored] = 255
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return _remove_border_noise(b)


def _remove_border_noise(b: np.ndarray) -> np.ndarray:
    """스캔 테두리(이미지 가장자리를 따라 도는 검은 선) 제거.

    가장자리 얇은 밴드(각 변 1.5%) 안에 '완전히 포함되는' 연결요소만 지운다.
    오선·마디선 등 실제 기호는 밴드 밖까지 뻗으므로 안전하다. 테두리를
    지우지 않으면 보표 좌우 범위(x0/x1)가 이미지 끝까지 붕괴해 조표·박자표
    탐색 존이 전부 어긋난다(music.jpg 퇴행의 1차 원인).
    """
    h, w = b.shape
    mx = max(3, int(round(w * 0.015)))
    my = max(3, int(round(h * 0.015)))
    n, labels, stats, _c = cv2.connectedComponentsWithStats((b > 0).astype(np.uint8))
    out = b.copy()
    for i in range(1, n):
        x, y, lw, lh, _a = stats[i]
        inside_lr = (x + lw <= mx) or (x >= w - mx)
        inside_tb = (y + lh <= my) or (y >= h - my)
        if inside_lr or inside_tb:
            out[labels == i] = 0
    return out


def _remove_frame_lines(b: np.ndarray) -> np.ndarray:
    """이미지 가장자리의 창 테두리/스캔 경계 직선 제거(레거시 1단계 이식).

    창 캡처·스캔 이미지는 가장자리를 따라 거의 꽉 찬 직선(테두리)이 남아
    보표 좌우 범위(x0)·조표 존 계산을 통째로 무너뜨린다(예: resource/music.jpg
    에서 x0=0 → 조표 4♭·박자표 6/8 전멸). 바깥 5% 띠 안에서 잉크 비율이
    85% 를 넘는 행/열만 지우므로 일반 악보 내용에는 영향이 없다.
    """
    out = b.copy()
    h, w = out.shape
    mh, mw = max(2, int(0.05 * h)), max(2, int(0.05 * w))
    for y in list(range(0, mh)) + list(range(h - mh, h)):
        if (out[y, :] > 0).mean() > 0.85:
            out[y, :] = 0
    for x in list(range(0, mw)) + list(range(w - mw, w)):
        if (out[:, x] > 0).mean() > 0.85:
            out[:, x] = 0
    return out


# ------------------------------------------------------------------ 2. 오선/단
def _staff_line_rows(b: np.ndarray):
    """수평 히스토그램에서 오선 후보 행 묶음 → [(중심 y, 두께)]."""
    hist = (b > 0).sum(axis=1).astype(np.float64)
    if hist.max() <= 0:
        return []
    thr = 0.45 * hist.max()
    rows = np.where(hist > thr)[0]
    lines = []
    if len(rows):
        s = p = rows[0]
        for r in rows[1:]:
            if r > p + 1:
                lines.append(((s + p) / 2.0, p - s + 1))
                s = r
            p = r
        lines.append(((s + p) / 2.0, p - s + 1))
    return lines


def _group_staves(lines):
    """오선 후보를 '간격이 균일한 5줄 묶음'으로 클러스터링(볼타 등 오검 배제)."""
    staves = []
    i = 0
    n = len(lines)
    while i + 4 < n:
        ys = [lines[j][0] for j in range(i, i + 5)]
        gaps = np.diff(ys)
        med = np.median(gaps)
        if med > 3 and np.all(np.abs(gaps - med) <= max(2.0, med * 0.25)):
            staves.append(list(range(i, i + 5)))
            i += 5
        else:
            i += 1          # 첫 줄이 오검(볼타 브래킷 등) → 한 줄 건너뜀
    return [[lines[j] for j in idxs] for idxs in staves]


def detect_systems(b: np.ndarray):
    groups = _group_staves(_staff_line_rows(b))
    systems = []
    for idx, g in enumerate(groups):
        ys = [c for c, _t in g]
        th = float(np.mean([t for _c, t in g]))
        systems.append(SystemInfo(index=idx, line_ys=ys, thickness=th))
    return systems


def normalize(b: np.ndarray, systems):
    """오선 간격 → SS(10px) 로 전체 스케일 정규화 후 오선 재검출.

    재이진화 임계값(100)은 실사진 잡음 억제를 위한 값이라 함부로 낮추면
    안 된다 — 실측: 100→40 로 전역 완화했더니 노트 인식 자체가 잡음에
    뭉개져 회귀가 72/84→56/84 로 대폭 악화됐다.

    문제는 '축소'(scale<1)할 때만 생긴다: INTER_AREA 축소 시 특정
    서브픽셀 정렬에서 오선 한두 줄이 여러 출력 행에 얇게 나뉘어 100을
    못 넘고 통째로 사라져, 그 시스템 전체가 5줄 그룹화에 실패해 누락된다
    (실측: 같은 악보를 PDF 왕복 렌더링해 픽셀 크기가 1px만 달라져도
    3단 중 1단만 인식되는 사고 재현). '확대'(scale>1, 지금까지의 실측
    4개 회귀 샘플이 전부 이 경로)는 이 희석이 원천적으로 생기지 않는
    경로라 건드릴 필요가 없다 — 그래서 축소일 때만 좁혀서 완화한다.
    """
    if not systems:
        return b, systems, 1.0
    ss = float(np.median([s.ss for s in systems]))
    scale = SS / ss if ss > 0 else 1.0
    if abs(scale - 1.0) > 0.02:
        h, w = b.shape
        nb = cv2.resize(b, (int(round(w * scale)), int(round(h * scale))),
                        interpolation=cv2.INTER_AREA)
        # 실질적 축소(scale<0.9)일 때만 완화한다. 선린교가(scale≈0.97,
        # 거의 축소 없음)는 임계값 100 그대로도 오선 6단 전부 정상
        # 검출되는데 완화를 걸면 실사진 잡음만 더 들어와 노트 인식이
        # 악화된다(실측 32→22마디) — 완화는 '진짜 필요한 경우'에만.
        thr = 40 if scale < 0.9 else 100
        _, nb = cv2.threshold(nb, thr, 255, cv2.THRESH_BINARY)
    else:
        nb = b.copy()
        scale = 1.0
    return nb, detect_systems(nb), scale


def _staff_x_extent(b: np.ndarray, sysinfo: SystemInfo):
    """오선 5줄 중 3줄 이상이 검은 열의 '최장 연속 구간' → 보표 x0..x1.

    첫/끝 열을 그대로 쓰면 잔여 노이즈 한 점이 범위를 이미지 끝까지
    끌어당긴다. 작은 틈(≤1ss)을 허용한 최장 연속 구간으로 강건화.
    """
    h, w = b.shape
    t = max(1, int(round(sysinfo.thickness)))
    cnt = np.zeros(w, np.int32)
    for y in sysinfo.line_ys:
        yy = int(round(y))
        band = b[max(0, yy - t):yy + t + 1, :]
        cnt += (band.max(axis=0) > 0).astype(np.int32)
    cols = np.where(cnt >= 3)[0]
    if len(cols) == 0:
        return 0, w - 1
    gap = max(2, int(round(sysinfo.ss)))
    best = (int(cols[0]), int(cols[0]))
    run_s = prev = int(cols[0])
    for x in cols[1:]:
        if x - prev > gap:
            if prev - run_s > best[1] - best[0]:
                best = (run_s, prev)
            run_s = int(x)
        prev = int(x)
    if prev - run_s > best[1] - best[0]:
        best = (run_s, prev)
    return best


def remove_staff_lines(b: np.ndarray, systems):
    """오선 위 '얇은 세로 런'만 지운다(기호를 관통하는 두꺼운 런은 보존)."""
    out = b.copy()
    h, w = out.shape
    col_ink = out > 0
    for s in systems:
        s.x0, s.x1 = _staff_x_extent(b, s)
        limit = int(round(s.thickness)) + 2
        ss = s.ss
        # 오선 5줄 + 덧줄(위/아래 3칸) 위치의 얇은 세로 런을 지운다.
        # 덧줄은 음표 머리를 관통해 hollow/채움 판정을 오염시키므로 함께 제거.
        real_line_ys = list(s.line_ys)
        ledger_ys = []
        for k in (1, 2, 3):
            ledger_ys.append(s.top - k * ss)
            ledger_ys.append(s.bottom + k * ss)
        for y in real_line_ys + ledger_ys:
            is_ledger = y in ledger_ys
            yy = int(round(y))
            if not (0 <= yy < h):
                continue
            # '가로로 넓게' 이어지는 잉크(빔)가 우연히 덧줄 후보 높이에
            # 걸치면, 아래 열별 검사가 빔의 각 단면을 '얇은 세로 런'으로
            # 오인해 통째로 지워버린다(실측: butterfly — 8분음표 빔이
            # 3번째 덧줄 높이와 겹쳐 사라짐). 가로 팽창으로 '폭이 넓은
            # 덩어리'를 찾아 그 열들은 건너뛴다 — 단, 이 보호는 '덧줄
            # 후보'에만 적용한다. 실제 오선 5줄 자체도 당연히 가로로
            # 넓으므로, 거기 이 보호를 걸면 오선 자체를 못 지운다.
            wide_cols = set()
            if is_ledger:
                row_band = col_ink[max(0, yy - limit):min(h, yy + limit + 1), s.x0:s.x1 + 1]
                if row_band.size:
                    any_ink = row_band.any(axis=0)
                    run_s = None
                    for xi, v in enumerate(any_ink):
                        if v and run_s is None:
                            run_s = xi
                        elif not v and run_s is not None:
                            if xi - run_s > 2.2 * ss:
                                wide_cols.update(range(run_s, xi))
                            run_s = None
                    if run_s is not None and len(any_ink) - run_s > 2.2 * ss:
                        wide_cols.update(range(run_s, len(any_ink)))
            for x in range(s.x0, s.x1 + 1):
                if (x - s.x0) in wide_cols:
                    continue
                if not col_ink[yy, x] and not (yy > 0 and col_ink[yy - 1, x]) \
                        and not (yy + 1 < h and col_ink[yy + 1, x]):
                    continue
                # 이 열에서 yy 를 포함(또는 인접)하는 세로 런의 길이 측정
                y0 = yy
                while y0 > 0 and col_ink[y0 - 1, x]:
                    y0 -= 1
                if not col_ink[yy, x]:
                    if yy + 1 < h and col_ink[yy + 1, x]:
                        y0 = yy + 1
                    elif yy > 0 and col_ink[yy - 1, x]:
                        y0 = yy - 1
                        while y0 > 0 and col_ink[y0 - 1, x]:
                            y0 -= 1
                    else:
                        continue
                y1 = y0
                while y1 + 1 < h and col_ink[y1 + 1, x]:
                    y1 += 1
                if (y1 - y0 + 1) <= limit and y0 >= yy - limit and y1 <= yy + limit:
                    out[y0:y1 + 1, x] = 0
    return out


def remove_floating_chord_symbols(clean: np.ndarray, systems):
    """오선 '완전히 위'에 독립적으로 떠 있는 기타 코드 기호(C, Dm7, Ab 등)를
    미리 지운다.

    사용자 제안(가사와 같은 원리): 코드 기호는 음정·박자 정보가 아니라
    텍스트이므로, 인식 단계 이전에 지워버리면 그로 인한 오검을 원천
    차단할 수 있다. 판정은 보수적으로: 연결요소 전체가 그 시스템의
    오선 첫 줄보다 확실히 위에 있고(요소가 오선에 조금이라도 닿아
    있으면 절대 안 지운다 — 덧줄 높은 음표·기둥과 혼동 방지), 크기가
    텍스트 범위 안일 때만 지운다. 코드 기호가 음표 기둥과 실제로 맞닿아
    한 덩어리가 된 경우(예: "Ab" 텍스트가 기둥에 닿음)는 이 조건에 안
    걸려 지워지지 않는다 — 그런 경우까지 잡으려면 별도 로직이 필요하며,
    여기서는 '확실한 것만' 안전하게 처리한다.
    """
    out = clean.copy()
    h, w = out.shape
    for s in systems:
        ss = s.ss
        top = s.top
        # 이 시스템 위쪽 탐색 범위: 바로 위 시스템의 아래쪽 여백 정도만
        # (그보다 위는 다른 시스템 소관이므로 안 건드림).
        y0 = max(0, int(top - 6.0 * ss))
        y1 = int(top - 1)
        if y1 <= y0:
            continue
        x0, x1 = int(max(0, s.x0 + 4.0 * ss)), int(min(w, s.x1 + 1))
        if x1 <= x0:
            continue
        band = out[y0:y1, x0:x1]
        n_lab, labels, stats, _c = cv2.connectedComponentsWithStats(
            (band > 0).astype(np.uint8))
        for i in range(1, n_lab):
            lx, ly, lw, lh, area = stats[i]
            # 조건 1: bbox 가 이 좁은 밴드 안에 완전히 들어옴(= 오선에
            # 안 닿음 — 밴드 자체가 오선 시작 전(y1=top-1)까지만이라
            # 별도 확인 불필요, band 밖은 애초에 포함 안 됨).
            # 조건 2: 텍스트 크기 범위(코드 기호는 최소 한 글자 이상—
            # 너무 작은 파편(점·부점·잡음)은 제외해 과대 삭제 방지).
            if not (0.7 * ss <= lh <= 3.0 * ss and 0.5 * ss <= lw <= 6.0 * ss):
                continue
            # 조건 3: 밴드 맨 아래 행(오선 바로 위)에 닿아 있지 않아야
            # 한다 — 닿아 있으면 기둥/덧줄과 물리적으로 이어졌을 위험이
            # 있어 텍스트만 골라 지우기 어려우므로 보수적으로 건너뛴다.
            if ly + lh >= band.shape[0] - 1:
                continue
            comp = labels[ly:ly + lh, lx:lx + lw] == i
            out[y0 + ly:y0 + ly + lh, x0 + lx:x0 + lx + lw][comp] = 0
    return out


def remove_floating_lyrics_below(clean: np.ndarray, systems):
    """오선 '아래' 독립적으로 떠 있는 가사 글자를 미리 지운다.

    remove_floating_chord_symbols 와 대칭인 아래쪽 버전. 가사 인식
    기능 자체는 삭제됐지만(인식률 저조), 가사 잉크가 여전히 이미지에는
    남아 있어 아래로 뻗는 음표 기둥·쉼표와 우연히 닿으면(실측: 191
    지존하신성체 — 반음표 아래-기둥이 가사 글자와 맞닿아 기둥 길이가
    과대 인식됨) 오검을 일으킨다. 정상적인 기둥이 닿을 수 없는 구간
    (오선 아래 4.0ss 이상)에서만, 완전히 독립된(오선·기둥에 안 닿은)
    작은 성분만 보수적으로 지운다.
    """
    out = clean.copy()
    h, w = out.shape
    for s in systems:
        ss = s.ss
        bottom = s.bottom
        y0 = int(bottom + 4.0 * ss)
        y1 = min(h, int(bottom + 9.0 * ss))
        if y1 <= y0:
            continue
        x0, x1 = int(max(0, s.x0)), int(min(w, s.x1 + 1))
        if x1 <= x0:
            continue
        band = out[y0:y1, x0:x1]
        n_lab, labels, stats, _c = cv2.connectedComponentsWithStats(
            (band > 0).astype(np.uint8))
        for i in range(1, n_lab):
            lx, ly, lw, lh, area = stats[i]
            if not (0.5 * ss <= lh <= 2.5 * ss and 0.3 * ss <= lw <= 3.0 * ss):
                continue
            # 밴드 맨 위 행(오선에 가장 가까운 쪽)에 닿아 있으면 기둥과
            # 이어졌을 위험이 있어 건너뛴다(보수적).
            if ly <= 0:
                continue
            comp = labels[ly:ly + lh, lx:lx + lw] == i
            out[y0 + ly:y0 + ly + lh, x0 + lx:x0 + lx + lw][comp] = 0
    return out


# ------------------------------------------------------------------ 3. 마디선
def detect_barlines_sys(clean: np.ndarray, s: SystemInfo, head_xs, stem_xs=()):
    """보표 5줄을 정확히 관통하는 얇은 세로선 = 마디선.

    기둥(stem)과의 구분: 기둥은 머리(음표)와 붙어 있고 보표 위/아래로 삐져
    나가거나 5줄 전체를 정확히 덮지 못한다. 마디선은 top-0.4ss ~ bottom+0.4ss
    범위 안에서 시작·끝나고, 주변 1.1ss 안에 음표 머리가 없다.
    """
    h, w = clean.shape
    top, bot = s.top, s.bottom
    ss = s.ss
    ys0 = int(round(top - 0.45 * ss))
    ys1 = int(round(bot + 0.45 * ss))
    cand_cols = []
    for x in range(s.x0, min(s.x1 + 1, w)):
        col = clean[max(0, ys0 - int(1.5 * ss)):min(h, ys1 + int(1.5 * ss)), x] > 0
        base = max(0, ys0 - int(1.5 * ss))
        # top~bottom 구간을 (2px 이하 틈 허용으로) 연속으로 덮는 런 찾기
        idx = np.where(col)[0]
        if len(idx) == 0:
            continue
        # 런 병합
        runs = []
        st = p = idx[0]
        for i in idx[1:]:
            if i > p + 3:
                runs.append((st, p))
                st = i
            p = i
        runs.append((st, p))
        for a, bb in runs:
            ya, yb = a + base, bb + base
            if ya <= top + 0.4 * ss and yb >= bot - 0.4 * ss \
                    and ya >= top - 0.8 * ss and yb <= bot + 0.8 * ss:
                cand_cols.append(x)
                break
    # 인접 열 병합 → 후보 마디선
    bars = []
    if cand_cols:
        st = p = cand_cols[0]
        for x in cand_cols[1:]:
            if x > p + 2:
                bars.append((st, p))
                st = x
            p = x
        bars.append((st, p))
    # 두께 제한 + 음표 머리 근접 배제
    out = []
    for a, bb in bars:
        wpx = bb - a + 1
        # 도돌이 시작(||:)·굵은 종지선의 두꺼운 획은 0.6ss 를 넘는다
        # (실측: guard 첫 마디선 0.7ss) — 이 상한을 넉넉히 올려 그런
        # 굵은 바도 후보로 잡히게 한다(분류는 아래 classify_barlines 가
        # 담당). 기둥과의 혼동은 아래 head_xs/stem_xs 근접 배제가 막는다.
        if wpx > 0.95 * ss:
            continue
        cx = (a + bb) / 2.0
        # 머리 타원과 겹치면 배제(0.8ss). 기둥은 '정확 일치(±3px)'만 배제해
        # 기둥 바로 옆(픽업 뒤 등)의 진짜 마디선을 살린다 — 머리 없는 고아
        # 기둥은 복원 패스가 이미 머리를 붙여 stem_xs 에 들어와 있다.
        if any(abs(nx - cx) < 0.8 * ss for nx in head_xs):
            continue
        if any(abs(sx2 - cx) <= 3 for sx2 in stem_xs):
            continue
        out.append((cx, wpx))
    return out


def classify_barlines(clean, s: SystemInfo, bars):
    """이중선/종지선/도돌이 판별. bars: [(x, w)] → (xs, {x: kind})."""
    ss = s.ss
    xs, types = [], {}
    used = [False] * len(bars)

    # 도돌이 점 검사: 잉크 비율만 보면 마지막 음표·가사 조각에도 오폭한다
    # (종지선이 repeat_end 로 뒤집힌 실측 사례). '2·3번째 칸 각각에
    # 소형(≤0.65ss) 원형 블롭이 하나씩' 있어야 도돌이다. 짝지어진 마디선
    # 뿐 아니라 단독 마디선(두꺼운 짝 선이 폭 필터에 걸려 후보에서 빠진
    # 경우)에도 써야 해서 바깥 스코프에 둔다.
    def dots(xa, xb):
        xa, xb = max(0, int(xa)), int(xb)
        if xb <= xa:
            return False
        found = 0
        for spot in (1.5, 2.5):        # top 기준 1.5ss·2.5ss = 두 칸
            cy = s.top + spot * ss
            seg = clean[int(cy - 0.45 * ss):int(cy + 0.45 * ss), xa:xb]
            if seg.size == 0:
                continue
            n2, _l2, st2, _c2 = cv2.connectedComponentsWithStats(
                (seg > 0).astype(np.uint8))
            for j in range(1, n2):
                _bx, _by, bw2, bh2, ba2 = st2[j]
                if 0.15 * ss <= bw2 <= 0.65 * ss \
                        and 0.15 * ss <= bh2 <= 0.65 * ss \
                        and ba2 >= 0.45 * bw2 * bh2:
                    found += 1
                    break
        return found == 2

    i = 0
    while i < len(bars):
        x, wpx = bars[i]
        if i + 1 < len(bars) and bars[i + 1][0] - x < 1.6 * ss:
            x2, w2 = bars[i + 1]
            thick_right = w2 > wpx * 1.6
            thick_left = wpx > w2 * 1.6
            has_dots_left = dots(x - 1.9 * ss, x - 0.6 * ss)
            has_dots_right = dots(x2 + 0.6 * ss, x2 + 1.9 * ss)
            pos = (x + x2) / 2.0
            if has_dots_left and not has_dots_right:
                kind = "repeat_end"
            elif has_dots_right and not has_dots_left:
                kind = "repeat_start"
            elif thick_right:
                kind = "final"
            elif thick_left:
                kind = "repeat_end"
            else:
                kind = "double"
            xs.append(pos)
            types[pos] = kind
            i += 2
            continue
        xs.append(x)
        # 단독 마디선도 도돌이 점 검사를 한다 — 두꺼운 짝 선이 폭 필터에
        # 걸려(0.6ss 초과) 별도 후보로 못 잡힌 경우, 이 얇은 선 하나만
        # 남아 항상 'normal' 로 떨어졌다(실측: 곡 시작 도돌이 시작표
        # "||:" 의 두꺼운 선이 통째로 누락). 점만으로도 도돌이 판정 가능.
        has_dots_right = dots(x + 0.6 * ss, x + 1.9 * ss)
        has_dots_left = dots(x - 1.9 * ss, x - 0.6 * ss)
        if has_dots_right and not has_dots_left:
            types[x] = "repeat_start"
        elif has_dots_left and not has_dots_right:
            types[x] = "repeat_end"
        else:
            types[x] = "normal"
        i += 1
    return xs, types


# ------------------------------------------------------------------ 4. 음표
def _find_stem(clean, s: SystemInfo, cx, cy, head_w):
    """머리 좌우에서 세로 기둥 탐색 → (stem_x, y0, y1, dir).

    양쪽을 모두 탐색해 '더 긴 세로 런'을 진짜 기둥으로 채택한다. 예전에는
    오른쪽에서 기준(≥2.2ss)을 넘기는 후보가 하나라도 나오면 왼쪽은 아예
    보지 않고 확정했는데, 노트헤드 우측 가장자리의 두꺼운 잉크가 우연히
    그 기준을 살짝 넘기면(실측: music.jpg) 왼쪽의 '진짜' 기둥(꼬리·깃발
    포함, 훨씬 긴 런)을 놓치고 짧은 오검을 기둥으로 삼아, 그 결과 깃발
    탐색 범위가 엉뚱한 쪽(반대편)을 보게 돼 8분음표가 4분음표로 오분류
    됐다.
    """
    h, w = clean.shape
    ss = s.ss
    best = None
    for side, xa, xb in (
            (+1, int(cx + head_w * 0.18), int(cx + head_w * 0.75)),
            (-1, int(cx - head_w * 0.75), int(cx - head_w * 0.18))):
        for x in range(max(0, xa), min(w, xb + 1)):
            col = clean[:, x] > 0
            yy = int(round(cy))
            if yy >= h:
                continue
            # cy 근처(±0.6ss)에서 잉크를 찾아 런 확장
            y_seed = None
            for dy in range(0, int(0.7 * ss)):
                if yy - dy >= 0 and col[yy - dy]:
                    y_seed = yy - dy
                    break
                if yy + dy < h and col[yy + dy]:
                    y_seed = yy + dy
                    break
            if y_seed is None:
                continue
            y0 = y1 = y_seed
            # 세로 확장에 길이 상한을 둔다 — 실측 정상 기둥 길이는 최대
            # 4.96ss(291개 표본)인데, 상한 없이 무한정 확장하면 기둥에
            # 가깝게 붙은 무관한 잉크(코드 기호 텍스트 등)와 우연히
            # 이어져 실제보다 길게 잡힐 수 있다. 국소 폭 검사로 사선
            # 글자 획을 걸러보려 했으나 대각선은 각 y행에서는 폭이
            # 좁게 측정돼 효과가 없었고 부작용도 있어 순수 길이 상한만
            # 안전망으로 남긴다 — 코드 기호가 기둥 끝과 픽셀 단위로
            # 맞닿는 경우(034 파일 'Ab')는 이 상한만으론 못 막는다.
            # 근본 해결은 인식 전 코드 기호/가사 밴드 사전 마스킹으로
            # 접근(아래 별도 함수) — 이 상한은 다른 경미한 케이스의
            # 보조 안전장치로만 유지한다.
            max_run = int(8.0 * ss)
            while y0 > 0 and col[y0 - 1] and (y_seed - y0) < max_run:
                y0 -= 1
            while y1 + 1 < h and col[y1 + 1] and (y1 - y_seed) < max_run:
                y1 += 1
            if (y1 - y0) >= 2.2 * ss:
                cand = (x, y0, y1, +1 if side > 0 else -1)
                if best is None or (y1 - y0) > (best[2] - best[1]):
                    best = cand
    return best


def _safe_beam_scan_cap(note, notes, ss):
    """`_count_beams` 의 max_off 계산: 가장 가까운 이웃 음표까지 거리의
    절반을 넘지 않도록 잡는다(그 이웃의 깃발/빔을 절대 건드리지 않을
    거리) — 최소 0.55ss 는 보장해 정상적인 단일 깃발(8분음표) 검출은
    방해하지 않는다."""
    if not notes or len(notes) < 2:
        return 1.00
    d = min((abs(nn.x - note.x) for nn in notes if nn is not note),
            default=999.0)
    return max(0.55, min(1.00, (d / ss) * 0.5))


def _count_beams(clean, s: SystemInfo, note: Note, max_off=1.00):
    """기둥 끝쪽에서 빔/깃발 교차 수 세기(수평 방향 짧은 스캔).

    max_off: 스캔할 최대 오프셋(ss 단위, 기본 1.00). 호출부가 이웃 음표
    까지의 거리로 좁혀 넘길 수 있다 — 자기 것에 잉크가 전혀 없는데 먼
    오프셋(0.85~1.0ss)에서 우연히 옆 음표의 깃발 끝을 건드려 '이 음표도
    깃발이 있다'고 오검하는 사고를 막기 위함이다(실측: 조밀하게 붙은
    16분음표+4분음표 조합에서, 깃발 없는 4분음표가 옆 16분음표의 깃발에
    걸려 8분음표로 오검됨). 진짜 자기 깃발/빔은 항상 기둥 바로 옆
    (0.4~0.55ss)에서 발견되므로, 이웃이 가까워 먼 오프셋을 못 쓰게 돼도
    실제 빔·깃발 인식에는 지장이 없다.
    """
    ss = s.ss
    h, w = clean.shape
    if note.stem_dir == 0:
        return 0
    # 기둥 끝(머리 반대편)에서 머리 쪽으로 2.6ss 구간을 검사
    if note.stem_dir > 0:      # 위 기둥: 위 끝이 빔
        y_end, y_in = note.stem_top, min(note.stem_top + 2.6 * ss, note.y - 0.9 * ss)
    else:
        y_end, y_in = note.stem_bot, max(note.stem_bot - 2.6 * ss, note.y + 0.9 * ss)
    ya, yb = int(min(y_end, y_in)), int(max(y_end, y_in))
    if yb <= ya:
        return 0
    def _runs_at(x):
        if x < 0 or x >= w:
            return -1
        col = clean[max(0, ya):min(h, yb + 1), x] > 0
        if not col.any():
            return -1
        lens = []
        run_len = 0
        for v in col:
            if v:
                run_len += 1
            else:
                if run_len >= 0.28 * ss:
                    lens.append(run_len)
                run_len = 0
        if run_len >= 0.28 * ss:
            lens.append(run_len)
        if not lens:
            return 0
        # 상대 두께 필터: 빔들은 두께가 균일하다. 최장 런의 60% 미만인 런은
        # 깃발 갈고리 끝 같은 부속 획이므로 계수하지 않는다(8분 깃발이
        # [4px, 9px] 두 런으로 갈라져 16분으로 과대 계수되는 것 차단).
        mx = max(lens)
        return sum(1 for L in lens if L >= 0.6 * mx)

    # 좌우 각 방향에서 '잉크가 있는 가장 가까운 오프셋'의 교차 수를 쓴다.
    # 팬 전체의 최댓값을 쓰면 8분음표 깃발의 갈고리가 먼 오프셋(0.7~1.0ss)
    # 에서 두 번 교차해 16분으로 과대 계수된다(자동교정이 다른 음표를
    # 4분으로 바꾸는 연쇄 오류의 뿌리). 기둥 바로 옆에서는 깃발 부착부=1런,
    # 16분 깃발의 두 컬=2런, 빔은 어느 오프셋에서나 정확하다.
    side_counts = []
    offs = [o for o in (0.40, 0.55, 0.70, 0.85, 1.00) if o <= max_off]
    for off_sign in (+1, -1):
        for off in offs:
            x = int(round(note.stem_x + off_sign * off * ss))
            rc = _runs_at(x)
            if rc >= 1:
                side_counts.append(rc)
                break
    return max(side_counts) if side_counts else 0


def _detect_dot_at(clean, s: SystemInfo, cx, cy, occupied, x_off0=0.75, x_off1=1.9,
                   y_off=0.85, parity_check=True):
    """기호(음표 머리/쉼표) 오른쪽의 부점 탐색 → 신뢰도 0..1.

    신뢰도는 '둥근 정도(채움비) × 크기 적정성'으로 계산한다. 종전의
    면적/0.35ss² 공식은 가는 판각의 점(면적 ~7px)을 0.2 로 깎아 부점을
    조직적으로 놓치게 했다(굵은 인쇄 기준 과적합).

    y_off: 세로 탐색 반경(ss 단위). 음표는 머리 중심 기준 ±0.85ss 로
    충분하지만, 갈고리형 쉼표(8·16·32분)는 글리프 자체가 세로로 길고
    쉼표의 '중심(cy)'이 점의 실제 높이(보통 글리프 상단부 근처)와 크게
    어긋난다(실측: 16분쉼표 점이 cy 보다 1.25ss 위에 있었음) — 이런
    쉼표는 호출부에서 더 큰 y_off 를 넘겨 탐색창을 넓힌다.
    parity_check: 참이면 점이 '칸(오선 사이 빈 공간)'에 있어야 한다는
    음표 전용 규칙을 적용한다. 쉼표는 이 규칙이 성립하지 않으므로
    호출부에서 끈다.
    """
    ss = s.ss
    h, w = clean.shape
    xa = int(cx + x_off0 * ss)
    xb = int(cx + x_off1 * ss)
    ya = int(cy - y_off * ss)
    yb = int(cy + y_off * ss)
    if xa >= w or ya >= h:
        return 0.0
    seg = clean[max(0, ya):min(h, yb), max(0, xa):min(w, xb)].copy()
    occ = occupied[max(0, ya):min(h, yb), max(0, xa):min(w, xb)]
    seg[occ > 0] = 0
    seg_raw = seg.copy()
    # 오선 위치 ±1px 행을 지운다 — 오선 제거가 남긴 두꺼운 교차 잔재가
    # 점 블롭과 세로로 병합되면 무게중심이 선 쪽으로 끌려 '칸 패리티'
    # 검사에서 진짜 점이 기각된다(실측: 선린 2단 317 부점).
    y0s = max(0, ya)
    line_rows = []
    for ly in s.line_ys:
        rr0 = int(round(ly)) - 1 - y0s
        for dr in range(3):
            r_i = rr0 + dr
            if 0 <= r_i < seg.shape[0]:
                seg[r_i, :] = 0
                line_rows.append(r_i)
    n, labels, stats, cents = cv2.connectedComponentsWithStats((seg > 0).astype(np.uint8))
    best = 0.0
    x_org, y_org = max(0, xa), max(0, ya)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        # 오선과 접촉해 꼬리가 잘린 블롭은 잔여 면적 기준을 높인다:
        # 진짜 점은 트림 후에도 ≥0.09ss²(≈9px)가 남지만, 오선에 눌린
        # 잔재 조각(가짜 점)은 2~3행짜리 얇은 띠만 남는다.
        touched = any(abs(r_i - (y + hh)) <= 1 or abs(r_i - (y - 1)) <= 1
                      for r_i in line_rows)
        min_area = 0.09 * ss * ss if touched else 4
        if 0.15 * ss <= ww <= 0.75 * ss and 0.15 * ss <= hh <= 0.75 * ss \
                and area >= 0.5 * ww * hh and area >= min_area \
                and 0.5 <= ww / max(1, hh) <= 2.0:
            # 조판 규칙: 부점은 항상 '칸(space)'에 놓인다. 선 위(짝수 위치)에
            # 있는 작은 잉크는 오선 제거 잔재·가사 조각일 가능성이 높다.
            # (쉼표는 이 규칙이 성립하지 않아 parity_check=False 로 끈다.)
            if parity_check:
                dot_cy = y_org + cents[i][1]
                pos = (s.bottom - dot_cy) / (ss / 2.0)
                near = round(pos)
                if abs(pos - near) > 0.45 or near % 2 == 0:
                    continue
            roundness = area / float(ww * hh)          # 둥근 채움비
            size_ok = min(1.0, area / (0.10 * ss * ss))  # 크기 적정성
            best = max(best, roundness * size_ok)
    return best


def _detect_dot(clean, s: SystemInfo, note: Note, occupied):
    return _detect_dot_at(clean, s, note.x, note.y, occupied)


def _stroke_comp_small(labels, stats, sx, sy, ss):
    """획 지점의 연결요소가 '임시표 크기'인지 — 음표의 깃발/기둥은 머리와
    이어져 거대 컴포넌트가 되므로, 그 획을 빌린 가짜 임시표를 걸러낸다."""
    h, w = labels.shape
    x, y = int(round(sx)), int(round(sy))
    lab = 0
    for dx in (0, -1, 1):
        if 0 <= y < h and 0 <= x + dx < w and labels[y, x + dx] > 0:
            lab = labels[y, x + dx]
            break
    if not lab:
        return False
    _lx, _ly, lw, lh, _la = stats[lab]
    return lw <= 2.6 * ss and lh <= 4.0 * ss


def _refine_accidental_kind(clean, cx, cy, ss, labels, stats, kind):
    """♯/♭/♮ 최종 판별 — 획 '구조'로 판정한다(템플릿 점수만으론 ♯↔♮ 혼동).

    ♭ = 긴 세로획 1개(중심선 왼쪽) + 볼은 아래-오른쪽에만(위-오른쪽은 비어
        있음 — 음표 머리+기둥 오검과의 결정적 차이).
    ♯ = 긴 세로획 2개, 오른획 위끝이 왼획과 비슷하거나 더 높음.
    ♮ = 긴 세로획 2개, 오른획이 뚜렷이 아래로 어긋남(왼획은 위로 돌출,
        오른획은 아래로 돌출 — SMuFL/판각 공통 특징).
    구조가 맞지 않으면 None 을 반환해 후보를 기각한다.
    """
    strokes = _acc_strokes(clean, cx, cy, ss)
    if kind == "flat" or len(strokes) == 1:
        if len(strokes) != 1:
            return None
        sx, st, sb = strokes[0]
        if sx > cx + 0.2 * ss:                       # ♭의 획은 중심선 왼쪽
            return None
        if (sb - st) > 2.9 * ss:                     # 기둥(≥3ss)은 ♭ 획이 아니다
            return None
        if not _stroke_comp_small(labels, stats, sx, (st + sb) / 2.0, ss):
            return None                              # 깃발/기둥을 빌린 가짜 ♭
        # ♭ 획의 '위쪽은 빈 공간'이다. 앞 음표의 깃발 꼬리가 오선 제거로
        # 끊겨 고립된 획은 바로 위에 깃발 본체 잉크가 남는다(실측) — 차단.
        hcl, wcl = clean.shape
        _ = (hcl, wcl)
        h, w = clean.shape
        def boxfill(x0, x1, y0, y1):
            x0, x1 = max(0, int(x0)), min(w, int(x1))
            y0, y1 = max(0, int(y0)), min(h, int(y1))
            if x1 <= x0 or y1 <= y0:
                return 0.0
            seg = clean[y0:y1, x0:x1]
            return float((seg > 0).mean())
        bowl = boxfill(sx + 2, sx + 0.7 * ss, cy - 0.1 * ss, cy + 0.7 * ss)
        upper_r = boxfill(sx + 0.3 * ss, sx + 0.8 * ss, st, cy - 0.6 * ss)
        left = boxfill(sx - 0.7 * ss, sx - 2, st + 0.2 * ss, sb - 0.2 * ss)
        if bowl >= 0.28 and upper_r <= 0.30 and left <= 0.22:
            return "flat"
        return None
    if len(strokes) < 2:
        return None
    # 이웃 기둥·부점이 창에 침입해 3개 이상 잡힐 수 있다 → 간격이 임시표
    # 범위(0.25~0.9ss)이고 중점이 후보 중심에 가장 가까운 '인접 쌍'을 고른다.
    pair = None
    best_d = 1e9
    for a, b2 in zip(strokes, strokes[1:]):
        gap = b2[0] - a[0]
        if 0.25 * ss <= gap <= 0.9 * ss:
            d = abs((a[0] + b2[0]) / 2.0 - cx)
            if d < best_d:
                best_d, pair = d, (a, b2)
    if pair is None or best_d > 0.8 * ss:
        return None
    (x1, t1, b1), (x2, t2, b2) = pair
    # 획 길이 상한: ♯/♮의 세로획은 ≤3.05ss. 음표 기둥(탐색창 클리핑 시
    # 3.2ss)이나 깃발+기둥 조합이 2획 구조를 흉내 내는 오검을 차단한다
    # (실측: 16분음표 깃발 뭉치가 ♯로 검출돼 A♯ 유령 임시표 발생).
    if (b1 - t1) > 3.4 * ss or (b2 - t2) > 3.4 * ss:
        return None
    # 채택된 두 획 모두 '독립 소형 컴포넌트'여야 한다 — 이웃 16분음표의
    # 기둥·깃발이 획으로 선택돼 가짜 ♯/♮ 가 되는 사례(실측) 차단.
    if not _stroke_comp_small(labels, stats, x1, (t1 + b1) / 2.0, ss) \
            or not _stroke_comp_small(labels, stats, x2, (t2 + b2) / 2.0, ss):
        return None
    if (t2 - t1) >= 0.22 * ss and (b2 - b1) >= 0.12 * ss:
        return "natural"
    if (t2 - t1) <= 0.18 * ss:
        return "sharp"
    return kind


def _verify_hollow(nb, clean, cx, cy, ss):
    """2분/온음 머리 검증: 원본(nb)에서 타원 테두리 잉크율 높고 중심은 비어야 함."""
    h, w = nb.shape
    ring_hit = 0
    ring_n = 0
    for k in range(24):
        th = 2 * math.pi * k / 24
        x = int(round(cx + 0.56 * ss * math.cos(th)))
        y = int(round(cy + 0.44 * ss * math.sin(th)))
        if 0 <= x < w and 0 <= y < h:
            ring_n += 1
            if nb[y, x] > 0 or (y + 1 < h and nb[y + 1, x] > 0) \
                    or (y > 0 and nb[y - 1, x] > 0):
                ring_hit += 1
    if ring_n == 0 or ring_hit / ring_n < 0.62:
        return False
    x0, x1 = int(cx - 0.20 * ss), int(cx + 0.20 * ss) + 1
    y0, y1 = int(cy - 0.16 * ss), int(cy + 0.16 * ss) + 1
    center = clean[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    return center.size == 0 or (center > 0).mean() <= 0.45


def _ellipse_fill_g(img, cx, cy, a, b):
    """타원 내부 잉크 비율(모듈 수준 헬퍼 — 복원 패스 등에서 사용)."""
    h, w = img.shape
    x0, x1 = max(0, int(cx - a)), min(w, int(cx + a) + 1)
    y0, y1 = max(0, int(cy - b)), min(h, int(cy + b) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0
    tot = int(mask.sum())
    if not tot:
        return 0.0
    return float((img[y0:y1, x0:x1][mask] > 0).sum()) / tot


def _run_len_v(clean, cx, cy):
    """(cx,cy)를 지나는 세로 잉크 런 길이(px). 호 획/머리 구분용."""
    h, w = clean.shape
    x, y = int(round(cx)), int(round(cy))
    if not (0 <= y < h and 0 <= x < w):
        return 0
    if clean[y, x] == 0:
        if y + 1 < h and clean[y + 1, x] > 0:
            y = y + 1
        elif y > 0 and clean[y - 1, x] > 0:
            y = y - 1
        else:
            return 0
    a = b = y
    col = clean[:, x]
    while a > 0 and col[a - 1] > 0:
        a -= 1
    while b + 1 < h and col[b + 1] > 0:
        b += 1
    return b - a + 1


def _run_len_h(clean, cx, cy):
    """(cx,cy)를 지나는 가로 잉크 런 길이(px). 빔 내부 오검 배제용."""
    h, w = clean.shape
    x, y = int(round(cx)), int(round(cy))
    if not (0 <= y < h and 0 <= x < w) or clean[y, x] == 0:
        return 0
    a = b = x
    row = clean[y]
    while a > 0 and row[a - 1] > 0:
        a -= 1
    while b + 1 < w and row[b + 1] > 0:
        b += 1
    return b - a + 1


def _head_shape_ok(clean, s: SystemInfo, cx, cy, hollow=False):
    """머리 후보의 국소 형태 검증: 빔·마디선·기둥 위 오검을 걸러낸다.

    오선 제거로 hollow 머리 테두리가 조각날 수 있으므로 '연속 런'이 아니라
    검사창(±1.35ss) 안의 잉크 스팬(최좌~최우)을 잰다. 빔은 창 양끝까지
    잉크가 닿으므로 구분되고, 마디선·기둥은 스팬이 좁아 걸러진다.
    """
    ss = s.ss
    h, w = clean.shape
    win = int(1.35 * ss)
    ok_rows = 0
    for dy in (-0.38 * ss, -0.2 * ss, 0, 0.2 * ss, 0.38 * ss):
        y = int(round(cy + dy))
        if not (0 <= y < h):
            continue
        x0, x1 = max(0, int(cx) - win), min(w, int(cx) + win + 1)
        seg = clean[y, x0:x1] > 0
        xs = np.where(seg)[0]
        if len(xs) == 0:
            continue
        touches_both = (xs[0] == 0 and xs[-1] == len(seg) - 1)
        if touches_both:
            continue                      # 빔 내부(창을 가로지르는 잉크)
        span = xs[-1] - xs[0] + 1
        if 0.72 * ss <= span <= 2.35 * ss:
            ok_rows += 1
    return ok_rows >= 1


def _lyric_cut_y(clean, s: SystemInfo):
    """보표 아래 가사 행의 시작 y를 찾는다(없으면 bottom+3.5ss).

    가사는 넓은 x범위에 걸쳐 잉크가 퍼진 '텍스트 행'을 이룬다. 빔·기둥은
    국소적이라 행 전체 잉크 비율이 낮다. 이 차이로 경계를 정한다.
    """
    h, _w = clean.shape
    ss = s.ss
    y_from = int(s.bottom + 1.1 * ss)
    y_to = min(h, int(s.bottom + 4.0 * ss))
    for y in range(y_from, y_to):
        row = clean[y, s.x0:s.x1 + 1]
        if row.size and (row > 0).mean() > 0.15:
            return max(y - 2, y_from)
    return y_to


def detect_notes(clean, nb, s: SystemInfo, rest_guard=()):
    """템플릿 매칭으로 음표 머리(일반/2분/온음/꾸밈음) 검출 + 기둥/빔/점 분석.

    rest_guard: 고신뢰 쉼표 x 목록. 쉼표의 곱슬 획이 머리(특히 꾸밈음)로
    선점되면 진짜 쉼표가 음표 근접 배제에 걸려 사라지므로, 이 구역에서는
    머리 후보를 받지 않는다(쉼표 우선 중재).

    nb: 오선 제거 '이전' 이진 이미지. 오선 제거가 hollow 머리(2분/온음)의
    얇은 테두리를 갉아먹어 템플릿 점수가 낮아지므로, hollow 머리는 낮은
    임계로 후보를 얻고 nb 위 '타원 테두리 링 검증'으로 확정한다.
    """
    ss = s.ss
    h, w = clean.shape
    band_y0 = max(0, int(s.top - 3.5 * ss))
    band_y1 = min(h, int(s.bottom + 3.5 * ss), _lyric_cut_y(clean, s))
    band = clean[band_y0:band_y1, :]

    t_grace = _render_glyph("noteheadBlack", ss, scale=0.66)

    cands = []
    for kind, name, thr in (("black", "noteheadBlack", 0.42),
                            ("half", "noteheadHalf", 0.30),
                            ("whole", "noteheadWhole", 0.48)):
        for scale in (0.9, 1.0, 1.1):
            tmpl = _render_glyph(name, ss, scale=scale)
            if tmpl is None:
                continue
            for sc, cx, cy, tw, th in _match(band, tmpl, thr):
                cands.append((sc, cx, cy + band_y0, tw, th, kind))

    # ── NMS 와 검증을 '한 루프'로 통합한다.
    #    (검증 전에 NMS 를 하면, 나중에 기각될 고점수 후보가 진짜 머리를
    #     억제해 둘 다 사라지는 문제가 생긴다 — 기둥 위 오검이 대표 사례)
    def _ellipse_fill(img, cx, cy, a, b):
        x0, x1 = max(0, int(cx - a)), min(w, int(cx + a) + 1)
        y0, y1 = max(0, int(cy - b)), min(h, int(cy + b) + 1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0
        tot = int(mask.sum())
        if not tot:
            return 0.0
        return float((img[y0:y1, x0:x1][mask] > 0).sum()) / tot

    notes = []
    import os as _os
    _DBG = _os.environ.get("OMR_DEBUG_HEAD")
    if _DBG:
        for sc, cx, cy, tw, th, kind0 in sorted(cands, key=lambda t: -t[0]):
            if 60 < cx < 90:
                print(f"[전체후보] sc={sc:.3f} cx={cx:.1f} cy={cy:.1f} kind={kind0}")
    for sc, cx, cy, tw, th, kind0 in sorted(cands, key=lambda t: -t[0]):
        _dbg_this = _DBG and 60 < cx < 90
        if any(abs(cx - gx) < 1.2 * ss for gx in rest_guard):
            continue                       # 쉼표 확정 구역 — 머리 금지
        # 단선율 제약: 한 x 위치에는 음표가 하나. 확정된 머리의 x범위와
        # 기둥 x는 이후 후보에서 통째로 배제한다(깃발·붙임줄 조각이 같은
        # 기둥 위/아래에서 머리로 오검되는 것을 차단).
        if any(abs(cx - n.x) <= 1.3 * ss
               or (n.stem_x > 0 and abs(cx - n.stem_x) <= 0.9 * ss)
               for n in notes):
            if _dbg_this:
                print(f"  [탈락:mono-x] sc={sc:.3f} cx={cx:.1f} cy={cy:.1f} kind={kind0}")
            continue
        # 보표에서 멀리 떨어진 후보(가사·코드 텍스트 조각)는 높은 확신을 요구
        if (cy > s.bottom + 1.9 * ss or cy < s.top - 0.6 * ss) and sc < 0.60:
            continue
        if not _head_shape_ok(clean, s, cx, cy, hollow=(kind0 != "black")):
            if _dbg_this:
                print(f"  [탈락:head_shape_ok] sc={sc:.3f} cx={cx:.1f} cy={cy:.1f} kind={kind0}")
            continue
        # hollow(2분/온음) 후보의 바깥 테두리는 오선 제거 '이전' 이미지에서
        # 측정한다 — 제거가 테두리를 조각내 outer 가 과소평가되는 것을 방지.
        outer = _ellipse_fill(nb if kind0 != "black" else clean,
                              cx, cy, 0.58 * ss, 0.44 * ss)
        if outer < 0.42:
            if _dbg_this:
                print(f"  [탈락:outer={outer:.3f}] sc={sc:.3f} cx={cx:.1f} cy={cy:.1f} kind={kind0}")
            continue
        if _dbg_this:
            print(f"  [통과:outer={outer:.3f}] sc={sc:.3f} cx={cx:.1f} cy={cy:.1f} kind={kind0}")
        fill = _ellipse_fill(clean, cx, cy, 0.38 * ss, 0.27 * ss)
        # 저신뢰(0.42~0.50) 검정 머리 후보의 2단계 게이트:
        # 붙임줄 호의 정점이 인접 두 음표 중간에서 0.43~0.49로 매칭되는
        # 오검 대역이다. 호 획(두께 0.3~0.45ss)과 진짜 머리(세로 ≈1ss)는
        # '중심 세로 잉크 런 길이'로 구분된다.
        if kind0 == "black" and sc < 0.56:
            # 3열(±2px) 중 잉크 있는 열들의 '최소' 세로 런: 피크가 기둥 열에
            # 앉아도 머리 전용 열이 최솟값을 대표한다(기둥 폭≈2px 회피).
            vrs = [v for v in (_run_len_v(clean, cx - 2, cy),
                               _run_len_v(clean, cx, cy),
                               _run_len_v(clean, cx + 2, cy)) if v > 0]
            if len(vrs) < 2 or not (0.65 * ss <= min(vrs) <= 1.7 * ss) \
                    or cy < s.top - 0.3 * ss or cy > s.bottom + 1.9 * ss:
                continue
        # 꾸밈음 크기 판별: 머리 폭이 뚜렷이 작으면(≤0.95ss) 일반 음표로
        # 선점하지 않고 꾸밈음 단계에 넘긴다 — 작은 머리를 일반 템플릿이
        # 0.6+ 점수로 먼저 차지해 꾸밈음(과 그 임시표)이 사라지는 문제 방지.
        if kind0 == "black":
            hr = _run_len_h(clean, cx, cy)
            if 0 < hr <= 0.95 * ss:
                vr_g = _run_len_v(clean, cx, cy)
                if 0 < vr_g <= 0.85 * ss:
                    continue
        st = _find_stem(clean, s, cx, cy, max(tw, 1.1 * ss))
        # 동일 기둥 배제: 단선율에서는 기둥 하나에 머리 하나다. 이미 확정된
        # 음표와 같은 기둥(±3px)을 공유하는 후보는 오검(보표 위 코드 글자가
        # 아래 음표의 기둥을 차지하는 사례 등)이다.
        if st is not None and any(n.stem_x > 0 and abs(st[0] - n.stem_x) <= 3
                                  for n in notes):
            continue

        if _dbg_this:
            print(f"  [stem탐색] sc={sc:.3f} cx={cx:.1f} cy={cy:.1f} kind0={kind0} st={st}")
        if st is None:
            # 기둥 없음 → 온음표만 가능(hollow + 보표 근처 + 높은 확신).
            # 보표 세로 범위 '밖'(위 0.3ss·아래 1.9ss 초과)의 후보는 제목·
            # 가사 한글의 둥근 획일 수 있으므로 요구 확신을 올린다 — 실제
            # 사례: 선린교가 제목 획이 conf 0.56 whole 로 오검돼 단선율
            # 배제 규칙으로 바로 아래 진짜 첫 음(C4)까지 차단했다.
            need = 0.5 if (s.top - 0.3 * ss <= cy <= s.bottom + 1.9 * ss) \
                else 0.62
            if fill <= 0.62 and outer >= 0.5 \
                    and s.top - 2.2 * ss <= cy <= s.bottom + 2.2 * ss \
                    and sc >= need:
                notes.append(Note(x=cx, y=cy, kind="whole", conf=sc))
            continue

        kind = "black" if fill >= 0.70 else "half"
        if kind == "black":
            # JPEG/스캔 블러로 half 머리의 위·아래 곡률(거의 수평)이 뭉개져
            # 채워진 것처럼 보이는 경우가 있다(중심 타원 fill 이 0.85 까지도
            # 나옴). '행별 채움비의 최솟값'이 훨씬 강한 판별자다 — 실측:
            # 진짜 검정 머리 2개는 최소 행 채움비가 0.89 밑으로 안 내려가지만
            # (전 행이 고르게 두꺼움), 진짜 2분음표는 0.40 까지 떨어지는
            # 행(허리가 가장 가는 지점)이 있었다 — 그 행이 어디인지는
            # 판각·기울어짐마다 달라 고정 위치로는 못 잡고 스캔해야 한다.
            hx0, hx1 = int(cx - 0.42 * ss), int(cx + 0.42 * ss) + 1
            hy0, hy1 = int(cy - 0.32 * ss), int(cy + 0.32 * ss) + 1
            head = clean[max(0, hy0):min(h, hy1), max(0, hx0):min(w, hx1)]
            if head.size and head.shape[0] >= 3:
                min_row_fill = min((row > 0).mean() for row in head)
                if min_row_fill < 0.65:
                    kind = "half"
        if kind == "half":
            if not (s.top - 2.2 * ss <= cy <= s.bottom + 2.2 * ss):
                continue
            # 2분음표는 빔을 가질 수 없다 → 빔 달린 hollow 후보는 오검
            probe = Note(x=cx, y=cy, kind="half", conf=sc)
            probe.stem_x, probe.stem_top, probe.stem_bot, probe.stem_dir = st
            if _count_beams(clean, s, probe) >= 1:
                continue
        n = Note(x=cx, y=cy, kind=kind, conf=sc)
        n.stem_x, n.stem_top, n.stem_bot, n.stem_dir = st[0], st[1], st[2], st[3]
        notes.append(n)
        if _dbg_this:
            print(f"  [최종추가] sc={sc:.3f} cx={cx:.1f} cy={cy:.1f} kind={kind} stem={st}")

    # 꾸밈음(축소 머리): 일반 머리와 겹치지 않는 위치에서 별도 매칭.
    if t_grace is not None:
        for sc, cx, cy, tw, th in _nms(_match(band, t_grace, 0.62), 0.6 * ss, 0.4 * ss):
            cy2 = cy + band_y0
            if any(abs(nn.x - cx) < 0.9 * ss and abs(nn.y - cy2) < 0.7 * ss
                   for nn in notes):
                continue
            if any(abs(cx - gx) < 1.2 * ss for gx in rest_guard):
                continue                    # 쉼표 확정 구역 — 꾸밈음 금지
            # 실제로 작아야 함: 주변 잉크 폭이 일반 머리보다 좁은지 확인
            x0, x1 = int(cx - 0.55 * ss), int(cx + 0.55 * ss)
            y0, y1 = int(cy2 - 0.45 * ss), int(cy2 + 0.45 * ss)
            seg = clean[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            if seg.size == 0 or (seg > 0).mean() < 0.30:
                continue
            if not _head_shape_ok(clean, s, cx, cy2):
                continue
            rl = _run_len_h(clean, cx, cy2)
            if rl > 1.35 * ss:          # 일반 머리 크기면 꾸밈음 아님
                continue
            g = Note(x=cx, y=cy2, kind="grace", conf=sc)
            st = _find_stem(clean, s, cx, cy2, tw)
            if st is None:
                continue                    # 꾸밈음은 반드시 기둥이 있다
            if any(n.stem_x > 0 and abs(st[0] - n.stem_x) <= 3 for n in notes):
                # 동일 기둥 배제: 이음줄 호 끝 등이 이웃 음표의 기둥을
                # '자기 기둥'으로 차지해 가짜 꾸밈음이 되는 것을 차단
                continue
            g.stem_x, g.stem_top, g.stem_bot, g.stem_dir = st
            if (g.stem_bot - g.stem_top) > 3.6 * ss:
                continue                    # 꾸밈음 기둥은 짧다(쉼표 오검 배제)
            # 깃발 검증: 기둥 끝(머리 반대편) 옆에 잉크가 있어야 한다
            fy = int(g.stem_top if g.stem_dir > 0 else g.stem_bot)
            fx = int(g.stem_x)
            box = clean[max(0, fy - int(0.2 * ss)):min(h, fy + int(1.2 * ss)),
                        max(0, fx + 2):min(w, fx + int(1.0 * ss))] \
                if g.stem_dir > 0 else \
                clean[max(0, fy - int(1.2 * ss)):min(h, fy + int(0.2 * ss)),
                      max(0, fx + 2):min(w, fx + int(1.0 * ss))]
            if box.size == 0 or (box > 0).mean() < 0.10:
                continue
            notes.append(g)

    # ── 고아 기둥 머리 복원: 템플릿이 놓친 머리(두꺼운 테두리·비표준 비율의
    #    hollow 등)도 '기둥'은 명확히 남는다. 마디선 조건(보표 top~bottom 을
    #    정확히 관통)에 해당하지 않는 긴 세로 런을 찾아, 그 양 끝에서 타원
    #    검증으로 머리를 복원한다. 임시표 획(≤2.7ss)과 구분하기 위해 런 길이
    #    하한은 2.9ss 로 둔다.
    top, bot = s.top, s.bottom
    x_lo = int(s.x0 + 4 * ss)
    col_runs = []
    for x in range(x_lo, min(s.x1, w - 1)):
        col = clean[max(0, int(top - 3.0 * ss)):min(h, int(bot + 3.0 * ss)), x] > 0
        base = max(0, int(top - 3.0 * ss))
        idx = np.where(col)[0]
        if len(idx) == 0:
            continue
        st_ = p = idx[0]
        for i in idx[1:]:
            if i > p + 2:
                if (p - st_) >= 2.9 * ss:
                    col_runs.append((x, st_ + base, p + base))
                st_ = i
            p = i
        if (p - st_) >= 2.9 * ss:
            col_runs.append((x, st_ + base, p + base))
    # 인접 열 병합(기둥 폭 1~3px)
    merged = []
    for x, a, bb in col_runs:
        if merged and x - merged[-1][0][-1] <= 1 \
                and abs(a - merged[-1][1]) < ss and abs(bb - merged[-1][2]) < ss:
            merged[-1][0].append(x)
            merged[-1][1] = min(merged[-1][1], a)
            merged[-1][2] = max(merged[-1][2], bb)
        else:
            merged.append([[x], a, bb])
    for xs_, a, bb in merged:
        if len(xs_) > 0.45 * ss:
            continue                              # 굵은 구조(빔 등) 배제
        sx = float(np.mean(xs_))
        # 마디선(보표를 정확히 관통) 배제
        if a <= top + 0.4 * ss and bb >= bot - 0.4 * ss \
                and a >= top - 0.8 * ss and bb <= bot + 0.8 * ss:
            continue
        if any(abs(sx - n.x) < 1.0 * ss
               or (n.stem_x > 0 and abs(sx - n.stem_x) < 1.0 * ss)
               for n in notes):
            continue
        if any(abs(sx - gx) < 1.2 * ss for gx in rest_guard):
            continue
        # 양 끝에서 머리 탐색(기둥의 좌우 어느 쪽이든).
        # 하행 기둥이면 머리는 위끝(a) 근처가 아니라 아래끝… 이 아니라:
        # 상행 기둥(sd=+1)은 머리가 '아래끝', 하행(sd=-1)은 머리가 '위끝'.
        found = None
        for end_y, sd in ((bb, +1), (a, -1)):
            for side in (-1, +1):
                hx = sx + side * 0.55 * ss
                for dy in (0.0, 0.25 * ss, 0.5 * ss):
                    cy2 = end_y - dy if sd > 0 else end_y + dy
                    if not (top - 3.0 * ss < cy2 < bot + 2.2 * ss):
                        continue
                    outer = _ellipse_fill_g(clean, hx, cy2, 0.58 * ss, 0.44 * ss)
                    if outer < 0.42:
                        continue
                    if not _head_shape_ok(clean, s, hx, cy2, hollow=True):
                        continue
                    fill = _ellipse_fill_g(clean, hx, cy2, 0.38 * ss, 0.27 * ss)
                    kind = "black" if fill >= 0.70 else "half"
                    if kind == "black":
                        # 본 후보 루프와 동일한 판별 보강(엔진.py 위쪽 주석
                        # 참고) — 블러로 뭉개진 half 머리가 여기서도 나온다.
                        hx0, hx1 = int(hx - 0.42 * ss), int(hx + 0.42 * ss) + 1
                        hy0, hy1 = int(cy2 - 0.32 * ss), int(cy2 + 0.32 * ss) + 1
                        headb = clean[max(0, hy0):min(h, hy1), max(0, hx0):min(w, hx1)]
                        if headb.size and headb.shape[0] >= 3:
                            if min((row > 0).mean() for row in headb) < 0.65:
                                kind = "half"
                    nn = Note(x=hx, y=cy2, kind=kind, conf=0.45)
                    nn.stem_x, nn.stem_top, nn.stem_bot = sx, float(a), float(bb)
                    nn.stem_dir = sd
                    if kind == "half" and _count_beams(clean, s, nn) >= 1:
                        continue
                    found = nn
                    break
                if found:
                    break
            if found:
                break
        if found:
            notes.append(found)

    # ── 음정 정밀화: 템플릿 매칭 피크는 판각·스케일 편차로 실제 머리
    #    중심에서 1~3px 어긋날 수 있고, 반음(=0.5ss=5px) 단위 판정에서
    #    이 오차가 음정 오독으로 이어진다.
    #
    # 단순 '창 안 잉크 무게중심'은 사선/블러가 있는 hollow 머리에서 크게
    # 틀릴 수 있다 — 실측 사례: 슬랜트(비스듬)한 2분음표 머리의 템플릿
    # 피크가 진짜 중심보다 아래쪽(아랫줄에 가까운 픽셀 밀도가 더 높은
    # 지점)에서 잡혀 D5 를 E5 로 오독했고, 창을 그 잘못된 피크 중심으로
    # 잡다 보니 머리 위쪽 절반이 통째로 창 밖으로 잘려 나가 무게중심
    # 보정도 오차를 절반만 회복했다. 연결요소의 '세로 폭(bbox)'은 밀도
    # 편향에 영향받지 않으므로, 머리를 포함하는 연결요소를 찾아 그
    # bbox 세로 중심으로 우선 보정하고, 못 찾으면 기존 잉크 무게중심으로
    # 대체한다(원.온음표처럼 독립 컴포넌트가 아닌 경우 대비).
    _n_lab, _labels, _stats, _ = cv2.connectedComponentsWithStats(
        (clean > 0).astype(np.uint8))
    for n in notes:
        rx = 0.55 * ss if n.kind != "grace" else 0.38 * ss
        ry = 0.55 * ss if n.kind != "grace" else 0.38 * ss
        x0c, x1c = int(n.x - rx), int(n.x + rx) + 1
        y0c, y1c = int(n.y - ry), int(n.y + ry) + 1
        seg = clean[max(0, y0c):min(h, y1c), max(0, x0c):min(w, x1c)]
        if seg.size == 0:
            continue
        ysn, xsn = np.where(seg > 0)
        if len(ysn) < 6:
            continue
        cy_new = max(0, y0c) + float(ysn.mean())

        # 연결요소 bbox 기반 보정(가능하면 우선 사용)
        yy, xx = int(round(n.y)), int(round(n.x))
        lab = 0
        if 0 <= yy < h and 0 <= xx < w:
            for dy in (0, -1, 1, -2, 2, -3, 3):
                y2 = yy + dy
                if not (0 <= y2 < h):
                    continue
                for dx in (0, -1, 1, -2, 2):
                    x2 = xx + dx
                    if 0 <= x2 < w and _labels[y2, x2] > 0:
                        lab = _labels[y2, x2]
                        break
                if lab:
                    break
        if lab:
            lx, ly, lw, lh, _la = _stats[lab]
            # 이 컴포넌트가 '머리 크기'(기둥·빔이 안 섞인 순수 머리)일
            # 때만 신뢰한다 — 기둥까지 뭉친 큰 컴포넌트는 bbox 중심이
            # 기둥 길이만큼 아래로 쏠려 오히려 더 틀린다.
            if lw <= 1.6 * ss and lh <= 1.7 * ss:
                cy_new = ly + lh / 2.0

        if abs(cy_new - n.y) <= 0.6 * ss:
            n.y = cy_new

    notes.sort(key=lambda n: n.x)
    return notes


# ------------------------------------------------------------------ 5. 쉼표/임시표
def detect_rests(clean, s: SystemInfo, notes, boost=0.0, only=None):
    ss = s.ss
    h, w = clean.shape
    band_y0 = max(0, int(s.top - 1.5 * ss))
    band_y1 = min(h, int(s.bottom + 1.5 * ss))
    band = clean[band_y0:band_y1, :]
    mid = (s.top + s.bottom) / 2.0
    # 쉼표는 '독립된 기호'다: 주변 1.6ss 안에 음표 머리·기둥이 없어야 하고,
    # 후보 자리의 연결요소 크기가 템플릿 크기와 비슷해야 한다(기둥·깃발 배제).
    n_lab, labels, stats, _cents = cv2.connectedComponentsWithStats(
        (clean > 0).astype(np.uint8))
    note_xs = [n.x for n in notes] + [n.stem_x for n in notes if n.stem_x > 0]

    out = []
    spec = [("restQuarter", 4, 0.66), ("rest8th", 8, 0.67),
            ("rest16th", 16, 0.65), ("restHalf", 2, 0.74), ("restWhole", 1, 0.74)]
    for name, beat, thr0 in spec:
        if only is not None and name not in only:
            continue
        thr = thr0 + boost
        for r_scale in (0.9, 1.0, 1.1):
            tmpl = _render_glyph(name, ss, scale=r_scale)
            if tmpl is None:
                continue
            th_t, tw_t = tmpl.shape
            for sc, cx, cy, tw, th in _nms(_match(band, tmpl, thr),
                                           1.2 * ss, 1.2 * ss):
                cy2 = cy + band_y0
                if any(abs(nx - cx) < 1.6 * ss for nx in note_xs):
                    continue
                if beat == 1 and abs((cy2 - 0.25 * ss) - (s.line_ys[1])) > 0.9 * ss:
                    continue        # 온쉼표: 둘째 줄 아래 매달림
                if beat == 2 and abs((cy2 + 0.25 * ss) - (s.line_ys[2])) > 0.9 * ss:
                    continue        # 2분쉼표: 셋째 줄 위에 얹힘
                if beat in (4, 8, 16) and abs(cy2 - mid) > 1.4 * ss:
                    continue
                if beat in (1, 2):
                    # 블록형(2분/온) 쉼표 고립성 검증: 같은 높이의 좌우 측면이
                    # 비어 있어야 한다. 빔·잉크 띠가 중간 줄 근처를 지나면
                    # 사각 블록 템플릿이 conf 1.0 으로 대량 오검되는데(가늘고
                    # 조밀한 판각에서 마디합 폭증의 주범), 빔은 좌우로 잉크가
                    # 이어지므로 이 검사로 걸러진다.
                    fy0 = max(0, int(cy2 - 1.0 * ss))
                    fy1 = min(clean.shape[0], int(cy2 + 1.0 * ss))
                    side_ok = True
                    for side_i, (fx0, fx1) in enumerate(
                            ((cx - 1.8 * ss, cx - 0.8 * ss),
                             (cx + 0.8 * ss, cx + 1.8 * ss))):
                        fx0 = max(0, int(fx0))
                        fx1 = min(clean.shape[1], int(fx1))
                        if fx1 <= fx0:
                            continue
                        win = clean[fy0:fy1, fx0:fx1]
                        ink = win > 0
                        frac = ink.mean()
                        if frac <= 0.06:
                            continue
                        # 오른쪽 창(점 자리)의 침범 잉크가 '점 크기'의 작고
                        # 동그란 덩어리 하나뿐이면 부점(augmentation dot)
                        # 일 가능성이 높다 — 빔은 창을 가로지르는 넓은
                        # 띠라 이 조건을 못 만족한다. 부점이면 통과시키고,
                        # 뒤에서 _detect_dot_at 이 실제로 부점을 붙인다.
                        if side_i == 1:
                            n2, _l2, st2, _c2 = cv2.connectedComponentsWithStats(
                                ink.astype(np.uint8))
                            # 잡티(면적<6px, 예: 스캔 노이즈·1px 파편)는 무시
                            # 하고, 남은 조각이 전부 점 크기여야 통과한다.
                            comps = [st2[j] for j in range(1, n2) if st2[j][4] >= 6]
                            if comps and all(
                                    0.15 * ss <= bw2 <= 0.85 * ss
                                    and 0.15 * ss <= bh2 <= 0.85 * ss
                                    and ba2 >= 0.4 * bw2 * bh2
                                    for _bx, _by, bw2, bh2, ba2 in comps):
                                continue        # 점(들)으로 판단 → 통과
                        side_ok = False
                        break
                    if not side_ok:
                        continue
                # 연결요소 크기 검증: 쉼표는 자기 컴포넌트가 템플릿과 비슷한
                # 크기여야 한다. 단, 오선 제거가 갈고리형 쉼표의 얇은 연결
                # 획을 끊어 하나의 글리프가 조각(갈고리부/꼬리부)으로 쪼개질
                # 때가 있다 — 매칭 중심이 하필 작은 꼬리 조각에 떨어지면
                # 그 조각만으로는 크기 검증에 항상 실패해 진짜 쉼표를
                # 통째로 놓친다(실측: music.jpg 8분쉼표 다수). 매칭 중심
                # 주변 '템플릿 크기 범위' 안의 모든 조각을 합쳐 전체 글리프
                # 크기를 복원한 뒤 검증한다.
                sx0 = int(cx - 0.9 * tw_t); sx1 = int(cx + 0.9 * tw_t)
                sy0 = int(cy2 - 0.9 * th_t); sy1 = int(cy2 + 0.9 * th_t)
                sx0, sy0 = max(0, sx0), max(0, sy0)
                sx1 = min(clean.shape[1], sx1)
                sy1 = min(clean.shape[0], sy1)
                nearby_labs = set(np.unique(labels[sy0:sy1, sx0:sx1])) - {0}
                if not nearby_labs:
                    continue
                ux0 = min(stats[lb][0] for lb in nearby_labs)
                uy0 = min(stats[lb][1] for lb in nearby_labs)
                ux1 = max(stats[lb][0] + stats[lb][2] for lb in nearby_labs)
                uy1 = max(stats[lb][1] + stats[lb][3] for lb in nearby_labs)
                lw, lh = ux1 - ux0, uy1 - uy0
                # 세로 폭 허용범위: 갈고리형 쉼표(4·8·16분)는 좁게(0.75~1.35배)
                # 조인다 — 느슨하면(0.6~1.6배) 8분쉼표 템플릿이 '16분쉼표
                # 하단 갈고리+꼬리'(갈고리 하나 더 있는 더 큰 컴포넌트)에도
                # 헐겁게 들어맞아 16분쉼표를 8분쉼표로 오분류한다(실측:
                # music.jpg 여러 곳). 사각 블록형(온·2분쉼표)은 갈고리가
                # 없어 이 문제가 없고, 오히려 얇은 스캔에서 세로폭이 살짝
                # 작게 나오는 경우가 있어 기존 느슨한 범위를 유지한다.
                # 세로 상한 1.35→1.45: 실측(여정 악보) 8분쉼표가 꼬리가 길어
                # 1.40배로 나와 기존 상한에 근소하게 걸려 통째로 누락됐다.
                # 하한(0.75)은 그대로 둔다 — 짧게 잘린 조각(오검) 배제 목적은
                # 유지해야 하므로.
                lo_r, hi_r = (0.75, 1.45) if beat in (4, 8, 16) else (0.6, 1.6)
                if not (0.5 * tw_t <= lw <= 1.8 * tw_t
                        and lo_r * th_t <= lh <= hi_r * th_t):
                    continue
                if beat in (1, 2):
                    # 블록형 쉼표는 좌우가 비어 있어야 한다(0.35~1.0ss 링).
                    # 블록 '가장자리'에 템플릿이 걸친 중복 매칭은 링에 자기
                    # 블록 잉크가 들어와 기각된다 → 중심 매칭만 생존.
                    xl0, xl1 = int(cx - 1.0 * ss), int(cx - 0.5 * tw_t - 0.35 * ss)
                    xr0, xr1 = int(cx + 0.5 * tw_t + 0.35 * ss), int(cx + 1.0 * ss + 0.5 * tw_t)
                    ya2, yb2 = int(cy2 - 0.6 * ss), int(cy2 + 0.6 * ss)

                    def _ring_ink(x0r, x1r, y0r, y1r):
                        x0r, x1r = max(0, x0r), min(clean.shape[1], x1r)
                        y0r, y1r = max(0, y0r), min(clean.shape[0], y1r)
                        if x1r <= x0r or y1r <= y0r:
                            return None
                        return clean[y0r:y1r, x0r:x1r] > 0

                    def _ring_blocked(x0r, x1r, y0r, y1r, allow_dot=False):
                        ink = _ring_ink(x0r, x1r, y0r, y1r)
                        if ink is None or ink.mean() <= 0.12:
                            return False
                        if allow_dot:
                            # 오른쪽 링(점 자리)에 걸리는 잉크가 '점 크기'의
                            # 작고 동그란 덩어리 하나뿐이면 부점으로 보고
                            # 통과시킨다(위 side_ok 검사와 동일 원리) —
                            # 이게 없으면 부점 붙은 온/2분쉼표가 전부
                            # 오검·소실된다(실측: music.jpg 다발 누락의
                            # 주요 원인).
                            n3, _l3, st3, _c3 = cv2.connectedComponentsWithStats(
                                ink.astype(np.uint8))
                            comps3 = [st3[j] for j in range(1, n3)]
                            if len(comps3) == 1:
                                _bx3, _by3, bw3, bh3, ba3 = comps3[0]
                                if 0.15 * ss <= bw3 <= 0.85 * ss \
                                        and 0.15 * ss <= bh3 <= 0.85 * ss \
                                        and ba3 >= 0.4 * bw3 * bh3:
                                    return False
                        return True

                    if _ring_blocked(xl0, xl1, ya2, yb2) \
                            or _ring_blocked(xr0, xr1, ya2, yb2, allow_dot=True):
                        continue
                    # 수직 고립: 8분쉼표의 볼(●)은 아래로 획이 이어져 블록형
                    # 쉼표로 오검(conf≈1.0)되고, 그 오검이 진짜 8분쉼표를
                    # NMS 로 억제한다. 블록 위/아래 0.25~0.9ss 도 비어야 한다.
                    bt, bb2 = uy0, uy0 + lh          # 합친 컴포넌트 상/하단
                    xw0, xw1 = int(cx - 0.55 * tw_t), int(cx + 0.55 * tw_t)
                    if _ring_blocked(xw0, xw1, int(bb2 + 0.25 * ss), int(bb2 + 0.9 * ss)) \
                            or _ring_blocked(xw0, xw1, int(bt - 0.9 * ss), int(bt - 0.25 * ss)):
                        continue
                    # half/whole 판별: 온쉼표는 줄 '아래 매달림'(상단이 줄에 접),
                    # 2분쉼표는 줄 '위에 얹힘'(하단이 줄에 접).
                    d_top = min(abs(bt - ly_) for ly_ in s.line_ys)
                    d_bot = min(abs(bb2 - ly_) for ly_ in s.line_ys)
                    beat = 1 if d_top < d_bot else 2
                out.append(Rest(x=cx, y=cy2, beat=beat, w=tw, h=th, conf=sc))
    final = []
    for r in sorted(out, key=lambda r: -r.conf):
        if all(abs(r.x - f.x) > 1.5 * ss for f in final):
            final.append(r)
    final.sort(key=lambda r: r.x)
    return final


def _acc_strokes(clean, cx, cy, ss):
    """임시표 후보의 '긴 세로획' 목록 → [(x, top, bottom)] (획 구조 판별용)."""
    h, w = clean.shape
    x0, x1 = int(cx - 0.75 * ss), int(cx + 0.75 * ss)
    # 세로 창 ±1.9ss: ♯ 획은 오선 교차 부풀림 포함 ~3.2ss 까지 자랄 수 있어
    # 창이 그보다 커야 '창 관통(클리핑된 기둥)' 판정과 구분된다. 기둥
    # (≥3.5ss)은 관통 검사 또는 길이 상한(3.4ss)에 걸린다.
    y0, y1 = int(cy - 1.9 * ss), int(cy + 1.9 * ss)
    strokes = []
    wy0, wy1 = max(0, y0), min(h, y1)
    for x in range(max(0, x0), min(w, x1)):
        col = clean[wy0:wy1, x] > 0
        idx = np.where(col)[0]
        if len(idx) == 0:
            continue
        st = p = idx[0]
        best = None
        span = wy1 - wy0
        def _ok(a, bb):
            # 창 위·아래를 '동시에' 관통하는 런은 클리핑된 음표 기둥이다 —
            # 진짜 임시표 획은 항상 창 안에서 끝난다(오선 교차로 길이가
            # 다소 부풀어도 무방). 길이 상한 대신 이 관통 검사를 쓴다.
            return not (a <= 1 and bb >= span - 2)
        for i in idx[1:]:
            if i > p + 1:
                if (p - st) >= 1.3 * ss and _ok(st, p) \
                        and (best is None or p - st > best[2] - best[1]):
                    best = (x, st + wy0, p + wy0)
                st = i
            p = i
        if (p - st) >= 1.3 * ss and _ok(st, p) \
                and (best is None or p - st > best[2] - best[1]):
            best = (x, st + wy0, p + wy0)
        if best:
            strokes.append(best)
    # 인접 열 병합(획 폭 1~3px) — 폭이 0.35ss 넘으면 획이 아니라 덩어리
    merged = []
    for x, a, bb in strokes:
        if merged and x - merged[-1][0][-1] <= 1:
            merged[-1][0].append(x)
            merged[-1][1] = min(merged[-1][1], a)
            merged[-1][2] = max(merged[-1][2], bb)
        else:
            merged.append([[x], a, bb])
    return sorted((float(np.mean(xs)), a, bb) for xs, a, bb in merged
                  if len(xs) <= 0.4 * ss)


def detect_accidentals(clean, s: SystemInfo):
    """♯/♭/♮ 전체 검출 → [(kind, x, y, conf)] (조표/임시표 구분은 상위에서).

    임시표는 '독립된 연결요소'라는 구조적 사실로 오검을 배제한다:
    음표 머리는 기둥·빔과 붙어 거대한 컴포넌트가 되므로, 후보 자리의
    컴포넌트가 임시표 크기 범위를 넘으면 기각한다. 인쇄 폰트가 Bravura 와
    비율이 다를 수 있어 3개 스케일로 매칭한다.
    """
    ss = s.ss
    h, w = clean.shape
    band_y0 = max(0, int(s.top - 3.5 * ss))
    # 보표 아래 덧줄 음역(C4 부근)의 임시표까지 포함하되 가사행은 배제
    band_y1 = min(h, int(s.bottom + 3.5 * ss), int(_lyric_cut_y(clean, s)))
    band = clean[band_y0:band_y1, :]
    _n, labels, stats, _c = cv2.connectedComponentsWithStats(
        (clean > 0).astype(np.uint8))

    def comp_ok(cx, cy, max_w, max_h):
        """후보 창(±0.4ss) 안의 컴포넌트 중 하나라도 임시표 크기면 통과.

        종전의 '가장 가까운 잉크 1점' 방식은 ♯ 템플릿 중심이 두 세로획
        사이 빈 공간에 떨어질 때 옆 음표 가장자리를 먼저 잡아(2px vs 3px)
        거대 컴포넌트로 오판·기각했다. 음표/빔 위의 오검은 창에 큰
        컴포넌트만 있으므로 여전히 걸러진다.
        """
        yy, xx = int(round(cy)), int(round(cx))
        r_ = max(3, int(0.4 * ss))
        y0c, y1c = max(0, yy - r_), min(h, yy + r_ + 1)
        x0c, x1c = max(0, xx - r_), min(w, xx + r_ + 1)
        labs = np.unique(labels[y0c:y1c, x0c:x1c])
        for lab in labs:
            if lab == 0:
                continue
            _lx, _ly, lw, lh, _la = stats[lab]
            if lw <= max_w and lh <= max_h and lh >= 1.0 * ss:
                return True
        return False

    out = []
    # ♯ 임계는 낮게 둔다: 인쇄 폰트의 ♯ 비율 편차가 크고(예: 좁고 긴 판각),
    # 오검은 _refine_accidental_kind 의 획 구조 검증이 걸러 준다.
    for name, kind, thr in (("accidentalSharp", "sharp", 0.38),
                            ("accidentalFlat", "flat", 0.53),
                            ("accidentalNatural", "natural", 0.56)):
        for scale in (0.85, 1.0, 1.15):
            tmpl = _render_glyph(name, ss, scale=scale)
            if tmpl is None:
                continue
            for sc, cx, cy, tw, th in _match(band, tmpl, thr):
                cy2 = cy + band_y0
                if not comp_ok(cx, cy2, 2.4 * ss, 3.8 * ss):
                    continue
                out.append((sc, cx, cy2, tw, th, kind))
    # 검증(refine)을 NMS '앞'에 둔다 — 나중에 기각될 고점수 오검이 진짜
    # 임시표를 억제해 둘 다 사라지는 문제 방지(머리 검출과 같은 원리).
    # 비용 억제: 종류별 소반경 사전 NMS 로 후보 수를 먼저 줄인다.
    pre = []
    for c in sorted(out, key=lambda t: -t[0]):
        if all(c[5] != p[5] or abs(c[1] - p[1]) > 0.3 * ss
               or abs(c[2] - p[2]) > 0.4 * ss for p in pre):
            pre.append(c)
    refined = []
    for sc, x, y, tw, th, k in pre:
        k2 = _refine_accidental_kind(clean, x, y, ss, labels, stats, k)
        if k2 is None:
            continue                        # 획 구조 불일치 → 오검 기각
        refined.append((sc, x, y, tw, th, k2))
    picked = []
    for c in sorted(refined, key=lambda t: -t[0]):
        if all(abs(c[1] - p[1]) > 0.7 * ss or abs(c[2] - p[2]) > 0.9 * ss
               for p in picked):
            picked.append(c)
    return [(k, x, y, sc) for sc, x, y, _tw, _th, k in picked]


def detect_clef_key_time(clean, s: SystemInfo, accidentals):
    """줄 머리 영역에서 음자리표·조표·박자표 검출."""
    ss = s.ss
    h, w = clean.shape
    # 음자리표: x0 근처 (x0 .. x0+4ss) 의 큰 세로 구조
    clef_end = s.x0 + 3.2 * ss
    t_clef = _render_glyph("gClef", ss)
    clef_box = None
    if t_clef is not None:
        band_y0 = max(0, int(s.top - 3 * ss))
        seg = clean[band_y0:min(h, int(s.bottom + 3 * ss)),
                    s.x0:min(w, int(s.x0 + 5 * ss))]
        m = _match(seg, t_clef, 0.45)
        if m:
            sc, cx, cy, tw, th = max(m, key=lambda t: t[0])
            clef_box = (s.x0 + cx - tw / 2, band_y0 + cy - th / 2, tw, th)
            clef_end = s.x0 + cx + tw / 2

    # ── 박자표 먼저 위치 확정(조표 오검이 박자표 'C'를 삼키는 순환 의존 차단)
    #    탐색 존: 음자리표 끝 ~ +9ss (조표가 있으면 박자표는 더 오른쪽에 있다)
    ts = None
    ts_x = None
    ts_end = clef_end
    zone_x0 = int(max(s.x0, clef_end - 0.3 * ss))
    zone_x1 = int(min(w, clef_end + 9.5 * ss))
    seg = clean[int(s.top - 0.5 * ss):int(s.bottom + 0.5 * ss), zone_x0:zone_x1]
    ts_symbol = None
    if seg.size:
        # C(공통박자)·¢(알라브레베)와 숫자 박자표를 모두 매칭해 '점수가
        # 높은 쪽'을 채택한다. (숫자 2/4 위에 C 템플릿이 0.5 이상으로
        # 걸리는 오검 사례가 있음)
        best_c = (0.0, None)
        best_cut = (0.0, None)
        for scale in (0.9, 1.0, 1.1):
            tc = _render_glyph("timeSigCommon", ss, scale=scale)
            m = _match(seg, tc, 0.45) if tc is not None else []
            if m:
                b = max(m, key=lambda t: t[0])
                if b[0] > best_c[0]:
                    best_c = (b[0], b)
            tcut = _render_glyph("timeSigCutCommon", ss, scale=scale)
            m2 = _match(seg, tcut, 0.45) if tcut is not None else []
            if m2:
                b2 = max(m2, key=lambda t: t[0])
                if b2[0] > best_cut[0]:
                    best_cut = (b2[0], b2)
        mid_y = seg.shape[0] / 2.0
        num_cands, den_cands = [], []
        for d in (2, 3, 4, 6, 8, 9):
            for scale in (0.9, 1.0, 1.1):
                tmpl = _render_glyph(f"timeSig{d}", ss, scale=scale)
                if tmpl is None:
                    continue
                # 임계값을 0.48→0.36 으로 완화했다 — 실측(3/2 박자 악보)
                # 에서 분자(3)는 0.6 안팎으로 잘 잡히는데 분모(2)는 0.38
                # 안팎으로 종전 임계값(0.48)에 못 미쳐 통째로 후보에서
                # 빠지는 사례가 있었다. 정밀도는 아래 '정렬된 쌍' 요구
                # (같은 x, 분모 종류 제한)와 최종적으로 이 조합 점수를
                # C/¢ 기호 점수와 비교하는 단계에서 확보한다 — 단순
                # 임계값 완화만으로 오검이 늘지 않는지 실측 회귀로 확인함.
                for sc, cx, cy, tw, th in _match(seg, tmpl, 0.48):
                    if cy < mid_y:
                        num_cands.append((sc, d, cx))
                    else:
                        den_cands.append((sc, d, cx))
        # 분모 "2" 보조 탐색: 실측(3/2 박자 악보)에서 분자(예: 3)는 0.6
        # 안팎으로 잘 잡히는데 분모 "2"는 0.38 안팎으로 종전 임계값(0.48)
        # 에 못 미쳐 후보에서 통째로 빠지는 사례가 있었다. 전역 임계값을
        # 낮추면 무관한 숫자가 늘어나 다른 박자표 오검을 유발하므로
        # (실측: 회귀 발생, 되돌림), **이미 확정된 분자 후보 바로 아래
        # 좁은 x 구간(±0.4ss)만** 낮은 임계값으로 보조 탐색한다 — 탐색
        # 범위 자체가 좁아 무관한 오검 유입 위험이 낮다.
        if num_cands:
            best_num_x = max(num_cands, key=lambda t: t[0])[2]
            for scale in (0.9, 1.0, 1.1):
                t2 = _render_glyph("timeSig2", ss, scale=scale)
                if t2 is None:
                    continue
                for sc, cx, cy, tw, th in _match(seg, t2, 0.32):
                    if cy >= mid_y and abs(cx - best_num_x) <= 0.4 * ss:
                        den_cands.append((sc, 2, cx))
        # 분자·분모는 같은 x 에 세로로 쌓인다 — '정렬된 쌍'의 합산 점수로
        # 선택한다(독립 최고점끼리 무관 위치에서 (8,4)처럼 결합하는 오검 방지).
        # 정렬 관용도는 1.2ss→0.6ss 로 좁혔다: 진짜 박자표 숫자쌍은 거의
        # 정확히 같은 x(오차 1px 안팎)에 있는데, 관용도가 넓으면 6px 넘게
        # 떨어진 무관한 숫자(예: 근처 코드 기호의 일부)가 개별 점수만
        # 높다는 이유로 진짜 짝(더 낮은 점수지만 완벽히 정렬됨)을 이기는
        # 사례가 있었다(실측: 3/2 박자 악보에서 '2' 대신 어긋난 '4'가 선택됨).
        best_pair = None
        for sn, dn, xn in num_cands:
            for sd, dd, xd in den_cands:
                if dd not in (2, 4, 8) or abs(xn - xd) > 1.2 * ss:
                    continue
                tot = sn + sd
                if best_pair is None or tot > best_pair[0]:
                    best_pair = (tot, dn, dd, (xn + xd) / 2.0, min(sn, sd))
        digit_ok = best_pair is not None
        digit_score = best_pair[4] if digit_ok else 0.0
        best_symbol_score = max(best_c[0], best_cut[0])
        if digit_ok and digit_score >= best_symbol_score:
            ts = (best_pair[1], best_pair[2])
            ts_x = zone_x0 + best_pair[3]
            ts_end = ts_x + 1.2 * ss
        elif best_cut[0] >= 0.65 and best_cut[0] >= best_c[0]:
            # 알라브레베(¢) = 2/2박자. 임계값을 공통박자(C)의 0.54보다
            # 높게 잡는다 — 실측(합성 다중 시스템 테스트)에서 박자표가
            # 반복 인쇄되지 않는 뒤쪽 시스템의 음자리표·잔여 잉크가 이
            # 글리프와 0.56 안팎으로 우연히 매칭돼, 실제로 없는 알라브레베
            # 표기가 여러 시스템에 걸쳐 오검되며 다수결로 진짜 박자표를
            # 뒤엎는 사례가 있었다(진짜 알라브레베는 0.72 안팎으로 훨씬
            # 높게 나와 0.65 상향으로도 정탐엔 지장 없음을 확인).
            ts = (2, 2)
            ts_symbol = "cut"
            ts_x = zone_x0 + best_cut[1][1]
            ts_end = ts_x + best_cut[1][3] / 2.0
        elif best_c[0] >= 0.54:
            # 임계 0.50 은 뒤 시스템에서 음표·빔 잉크가 C 글리프와 우연히
            # 0.50~0.52 로 매칭돼(실측: guard L3) 진짜 박자표가 없는 줄의
            # head_end 를 밀어내 첫 음표를 통째로 잘라먹었다. 진짜 C 매칭은
            # 보통 0.55 이상이라 이 정도 상향은 정탐에 영향이 없다.
            ts = (4, 4)
            ts_symbol = "common"        # C 기호로 표기된 4/4 — 표기 보존
            ts_x = zone_x0 + best_c[1][1]
            ts_end = ts_x + best_c[1][3] / 2.0

    # ── 조표: 음자리표 끝 ~ (박자표 직전 | +6ss) 사이의 ♯/♭.
    # 조표는 '등간격 사슬'로 나열된다 — x 정렬 후 간격 ≤1.4ss 로 이어지는
    # 선두 사슬만 조표로 인정한다. 사슬에서 떨어진 후보(박자표 글리프가 ♭로
    # 오검된 것 등)를 조표로 세면 key_end 가 박자표를 지나쳐 박자표 존이
    # 어긋난다(music.jpg 의 6/8 미검출 원인).
    key_hi = (ts_x - 0.9 * ss) if ts_x is not None else (clef_end + 6.0 * ss)
    key_zone = (clef_end - 0.3 * ss, key_hi)
    zone_acc = sorted((x, k) for k, x, _y, _c in accidentals
                      if key_zone[0] <= x <= key_zone[1] and k in ("sharp", "flat"))
    chain = []
    for x, k in zone_acc:
        if not chain:
            if x <= clef_end + 2.2 * ss:      # 사슬 시작은 음자리표 바로 뒤
                chain.append((x, k))
        elif x - chain[-1][0] <= 2.2 * ss:   # 사슬 중 1개 미검출 허용
            chain.append((x, k))
        else:
            break
    n_sharp = sum(1 for _x, k in chain if k == "sharp")
    n_flat = sum(1 for _x, k in chain if k == "flat")
    key = 0
    key_end = clef_end
    if n_sharp and n_sharp >= n_flat:
        key = 10 * min(n_sharp, 7)
        key_end = max(x for x, k in chain if k == "sharp") + 0.6 * ss
    elif n_flat:
        key = 100 * min(n_flat, 7)
        key_end = max(x for x, k in chain if k == "flat") + 0.6 * ss

    head_end = max(clef_end, key_end, ts_end)
    return clef_box, key, ts, head_end, ts_symbol



# ------------------------------------------------------------------ 6. 문법 검증
def _beat_value(beat_code: int, dotted: bool) -> float:
    base = {1: 4.0, 2: 2.0, 4: 1.0, 8: 0.5, 16: 0.25, 32: 0.125}.get(abs(beat_code), 0)
    return base * 1.5 if dotted else base


def _fix_plan(m, target, msum):
    """단일 요소 재해석으로 마디 합을 target 에 정확히 맞출 계획을 찾는다.

    반환: ("exact", None) | ("dot", obj) | ("undot", obj) | ("beat", obj, 새박자)
          | None(불가). 실제 적용은 호출자가 한다(가설 검정 중에는 적용 금지).
    """
    diff = target - msum(m)
    if abs(diff) < 0.01:
        return ("exact", None)
    if abs(diff) > 2.01:
        return None
    # 점 복원(버려진 저신뢰 점): diff == +0.5×값
    for _x, o, isn in m["items"]:
        if not isn or o.kind == "grace" or o.dotted:
            continue
        if abs(diff - _beat_value(o.beat, False) * 0.5) < 0.01 and o.dot_conf > 0.15:
            return ("dot", o)
    # 점 제거(저신뢰 점 오검): diff == −0.5×값
    for _x, o, isn in m["items"]:
        if not isn or o.kind == "grace" or not o.dotted:
            continue
        if abs(diff + _beat_value(o.beat, False) * 0.5) < 0.01 and o.dot_conf < 0.55:
            return ("undot", o)
    # 빔/머리 1단계 재해석(16↔8, 8↔4, 4↔2) — 점 인지 델타, 저신뢰만.
    # 후보가 여럿이면 '신뢰도가 가장 낮은' 요소를 고른다 — x 순서 첫
    # 후보를 고치면 멀쩡한 음표(conf 0.76)를 바꾸고 진짜 오독(conf 0.45)
    # 을 놔두는 역효과가 실측됐다(선린 1단 4마디 4분↔2분 역전 사례).
    best = None
    for _x, o, isn in m["items"]:
        if not isn or o.kind == "grace":
            continue
        for nb2 in (o.beat // 2, o.beat * 2):
            if nb2 in (1, 2, 4, 8, 16, 32) and nb2 != o.beat:
                nd = _beat_value(nb2, o.dotted) - _beat_value(o.beat, o.dotted)
                if abs(diff - nd) < 0.01 and o.conf < 0.8:
                    if best is None or o.conf < best[1].conf:
                        best = ("beat", o, nb2)
    if best is not None:
        return best
    # 제한적 2요소 탐색: 단일 재해석으로 도달 불가한 작은 차이(|diff|≤0.75)
    # 를 저신뢰 요소 2개의 조합으로 맞춘다 — 실측: 점8분+16분 쌍이
    # [16., 8]로 뒤집혀 +0.125 가 남는 사례(단일 변경 합으로는 불가능).
    if abs(diff) <= 0.75:
        cands2 = []
        for _x, o, isn in m["items"]:
            if not isn or o.kind == "grace" or o.conf >= 0.78:
                continue
            for nb2 in (o.beat // 2, o.beat * 2):
                if nb2 in (1, 2, 4, 8, 16, 32) and nb2 != o.beat:
                    nd = _beat_value(nb2, o.dotted) - _beat_value(o.beat, o.dotted)
                    cands2.append((o, nb2, nd))
        best2 = None
        for i1 in range(len(cands2)):
            for i2 in range(i1 + 1, len(cands2)):
                o1, n1, d1 = cands2[i1]
                o2, n2, d2 = cands2[i2]
                if o1 is o2:
                    continue
                if abs(diff - (d1 + d2)) < 0.01:
                    csum = o1.conf + o2.conf
                    if best2 is None or csum < best2[0]:
                        best2 = (csum, o1, n1, o2, n2)
        if best2 is not None:
            return ("beat2", best2[1], best2[2], best2[3], best2[4])
    return None


# 마디합 자동 교정 스위치. 실전 검증에서 리듬 재해석이 이웃 요소를 잘못
# 건드리는 부작용이 반복 관찰되어(예: 저신뢰 8분음표가 16분으로 잘못
# 재해석되며 그 옆 음표를 4분으로 바꾸는 연쇄) 기본값을 끔. 사용자가
# 에디터에서 직접 확인·수정하는 편이 원본 보존과 예측 가능성 면에서
# 더 안전하다는 판단(마디 상태 배지는 그대로 표시되어 어디를 봐야 할지는
# 안내한다). 필요 시 이 값을 True 로 바꾸면 종전의 자동 교정이 복원된다.
AUTOFIX_ENABLED = False


def grammar_pass(sysres: SystemResult, capacity: float, autofix: bool = True):
    """마디 분할 + 검산 + 용량 가설 검정 + 보수적 자동 교정.

    용량 가설 검정: 표기 박자표 용량과 더 작은 후보 용량(2/4=2.0 등)들을
    놓고 '정확히 맞는 마디 ×2 + 단일 교정으로 맞는 마디 ×1' 점수를 비교한다.
    후보 용량이 표기 용량을 2점 이상 이길 때만 실효 용량으로 채택한다 —
    표기되지 않은 박자 변화(선린교가 5·6단의 2/4)를 오류가 아닌 정보로
    표시하기 위함이다.
    """
    s = sysres.info
    items = []
    for n in sysres.notes:
        items.append((n.stem_x if n.stem_x > 0 else n.x, n, True))
    for r in sysres.rests:
        items.append((r.x, r, False))
    items.sort(key=lambda t: t[0])

    bars = sorted(sysres.barlines)
    bounds = [-1e9] + bars + [1e9]
    measures = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        seg = [(x, o, isn) for x, o, isn in items if lo <= x < hi]
        if not seg:
            continue
        measures.append({"lo": lo, "hi": hi, "items": seg})

    def msum(m):
        tot = 0.0
        for _x, o, isn in m["items"]:
            if isn and o.kind == "grace":
                continue
            if not isn and o.beat == 1:
                # 기보법: 온쉼표는 박자표와 무관하게 '온마디' 쉼표다
                tot += capacity
                continue
            tot += _beat_value(o.beat, o.dotted)
        return tot

    # ── 용량 가설 검정
    cands = [capacity] + [v for v in (1.0, 1.5, 2.0, 3.0)
                          if v < capacity - 0.01]
    scores = {}
    for cap_h in cands:
        sc = 0
        for m in measures:
            plan = _fix_plan(m, cap_h, msum)
            if plan is None:
                continue
            sc += 2 if plan[0] == "exact" else 1
        scores[cap_h] = sc
    eff_capacity = capacity
    for cap_h in cands[1:]:
        if scores[cap_h] >= scores[eff_capacity] + 2 \
                and scores[cap_h] > scores.get(capacity, 0):
            if scores[cap_h] > scores.get(eff_capacity, 0) + 1 \
                    or eff_capacity == capacity:
                eff_capacity = cap_h
    time_change = eff_capacity != capacity

    # ── 자동 교정: 채택된 용량을 향해 단일 재해석 적용(기본 비활성 — 위
    #    AUTOFIX_ENABLED 참고). 꺼져 있어도 위의 용량 가설 검정(time_change
    #    판정)은 그대로 동작해 '박자 변화 추정' 안내는 유지된다.
    if autofix and AUTOFIX_ENABLED:
        for m in measures:
            if abs(msum(m) - capacity) < 0.01:      # 표기 용량에 맞으면 보존
                continue
            plan = _fix_plan(m, eff_capacity, msum)
            if plan is None or plan[0] == "exact":
                continue
            if plan[0] == "dot":
                plan[1].dotted = True
            elif plan[0] == "undot":
                plan[1].dotted = False
            elif plan[0] == "beat":
                plan[1].beat = plan[2]
            elif plan[0] == "beat2":
                plan[1].beat = plan[2]
                plan[3].beat = plan[4]

    results = []
    for i, m in enumerate(measures):
        tot = msum(m)
        if abs(tot - capacity) < 0.01:
            status = "OK"
        elif time_change and abs(tot - eff_capacity) < 0.01:
            status = "time_change"      # 박자 변화 추정(오류 아님·정보)
        elif tot > eff_capacity:
            status = "over"
        else:
            status = "under"
        results.append((s.index, i + 1, round(tot, 3), capacity, status))
        m["status"] = status
    sysres.measures = results
    return results, eff_capacity, time_change


# ------------------------------------------------------------------ 호(이음줄/붙임줄)
def detect_triplets(clean, s: SystemInfo, notes):
    """셋잇단음표 검출 — 인접 음표 3개 위의 가로 브래킷("⌐──⌐") 또는 "3" 표기.

    문법(마디 합)만으로 셋잇단음표를 "추론"하지 않는다 — 이 프로젝트의
    보수적 자동교정이 겪은 것과 같은 부작용(엉뚱한 요소를 재해석해
    연쇄 오류를 냄) 위험이 있고, 애초에 셋잇단음표는 실제 악보에
    그려진 명시적 기호이므로 굳이 추측할 이유가 없다 — 그 표기 자체를
    찾으면 된다.

    실측(가톨릭성가 057번)에서 확인한 실제 표기 형태: 셋잇단음표
    3개가 **빔으로 묶이지 않고 각자 개별 깃발**을 단 채로(8분음표
    3개가 나란히), 그 위에 **양 끝이 아래로 짧게 꺾인 가로 브래킷
    선**과 그 중앙에 "3" 숫자가 함께 표시된다. 처음엔 '빔그룹이 같은
    음표 3개'만 후보로 삼았는데, 이 표기 형태에서는애초에 빔이 없어
    beam_group 이 셋 다 제각각이라 후보 자체가 안 만들어졌다(실측:
    18마디 중 겨우 1그룹만 검출).

    절차: beam_group 과 무관하게, **x축으로 인접하고(간격이 고르고
    사이에 다른 요소 없음) 같은 종류(black)·같은 박자·논-도트인
    음표 3개**를 후보로 삼는다. 그 위쪽에서 가로 브래킷(넓은 가로
    직선)을 찾거나, 실패하면 "3" 숫자만으로도 인정한다(브래킷 없이
    "3"만 표기하는 판각도 있음 — 실측: 057번 곡의 뒷부분 그룹들).
    """
    ss = s.ss
    h, w = clean.shape
    cands = sorted(
        [n for n in notes if n.kind == "black" and not n.dotted and n.stem_x > 0],
        key=lambda n: n.x)

    out_notes = []
    used = set()
    i = 0
    while i + 2 < len(cands):
        a, b, c = cands[i], cands[i + 1], cands[i + 2]
        if id(a) in used:
            i += 1
            continue
        # 같은 박자(깃발 수)·고른 간격(등간격에서 크게 벗어나지 않음)
        gap1, gap2 = b.x - a.x, c.x - b.x
        same_beat = a.beams == b.beams == c.beams
        even_gap = (gap1 > 0 and gap2 > 0
                   and 0.55 < gap1 / gap2 < 1.82
                   and gap1 < 4.5 * ss and gap2 < 4.5 * ss)
        if not (same_beat and even_gap):
            i += 1
            continue
        top = int(min(n.stem_top if n.stem_top > 0 else n.y for n in (a, b, c)))
        x0 = int(a.x - 0.5 * ss)
        x1 = int(c.x + 0.5 * ss)
        y0c = max(0, top - int(2.2 * ss))
        y1c = max(0, top - 1)
        found = False
        if y1c > y0c and x1 > x0:
            band = clean[y0c:y1c, max(0, x0):min(w, x1)]
            # 1) 가로 브래킷: 폭의 60% 이상을 가로지르는 얇은(≤0.6ss) 가로 띠.
            row_ink = (band > 0).sum(axis=1)
            band_w = band.shape[1]
            if band_w > 0:
                wide_rows = np.where(row_ink >= 0.6 * band_w)[0]
                if len(wide_rows) > 0:
                    found = True
            # 2) 브래킷이 안 보이면 "3" 숫자 단독 표기로도 인정.
            if not found:
                for scale in (0.8, 1.0, 1.2):
                    t3 = _render_text_digit("3", ss, scale=scale)
                    if t3 is None:
                        continue
                    if _match(band, t3, 0.50):
                        found = True
                        break
        if found:
            out_notes.extend([a, b, c])
            used.update({id(a), id(b), id(c)})
            i += 3
        else:
            i += 1
    return out_notes




def detect_arcs(clean, s: SystemInfo, notes, barline_xs):
    """이음줄/붙임줄 호 검출 — 구조물 소거 + 두께 기반 빔 분리.

    호는 머리·빔과 맞닿아 한 컴포넌트를 이루는 일이 많아 '잔여 컴포넌트'
    만으로는 찾을 수 없다. 절차:
      1) 알려진 구조물 소거: 머리(원판), 기둥(열), 마디선(열), 쉼표 부근.
      2) 두께로 빔 분리: 빔(두께 ≈0.5ss)은 세로 침식(0.38ss)을 견디고
         호(2~3px 획)는 사라진다. 침식→팽창으로 빔 마스크를 만들어 뺀다.
      3) 남은 성분 중 '넓고(≥2ss) 낮고(≤2.2ss) 가는(중앙값 획≤0.5ss)'
         것을 호로 보고, 중앙 대 끝점 잉크 높이로 위/아래를 판정한다.
    반환: [{"line", "x1", "x2", "above"}] (model._attach_arcs 계약).
    """
    ss = s.ss
    h, w = clean.shape
    y0 = max(0, int(s.top - 2.8 * ss))
    y1 = min(h, int(_lyric_cut_y(clean, s)))
    if y1 <= y0:
        return []
    work = (clean[y0:y1, :] > 0).astype(np.uint8) * 255

    # 1) 구조물 소거
    for n in notes:
        cv2.circle(work, (int(n.x), int(n.y - y0)), int(0.85 * ss), 0, -1)
        if n.stem_x > 0:
            cv2.line(work, (int(n.stem_x), int(n.stem_top - y0)),
                     (int(n.stem_x), int(n.stem_bot - y0)), 0,
                     thickness=max(3, int(0.3 * ss)))
    for bx in barline_xs:
        cv2.line(work, (int(bx), 0), (int(bx), y1 - y0), 0,
                 thickness=max(3, int(0.3 * ss)))

    # 2) 구조 기반 빔/깃발 소거: 기둥 위치를 이미 알므로 빔이 놓일 자리
    #    (같은 빔 그룹의 인접 기둥 끝 사이 사각형)와 깃발 자리(외톨이
    #    8분/16분 기둥 끝 옆)를 직접 지운다. 두께 기준 형태학은 굵은
    #    붙임줄까지 지워버리므로 쓰지 않는다.
    beamed = [n for n in notes if n.stem_x > 0 and n.beams >= 1
              and n.kind in ("black", "grace")]
    by_group = {}
    for n in beamed:
        by_group.setdefault(n.beam_group, []).append(n)
    for g, grp in by_group.items():
        grp.sort(key=lambda n: n.stem_x)
        depth = int((max(n.beams for n in grp) * 0.62 + 0.35) * ss)
        if g != -1 and len(grp) >= 2:
            for a, b2 in zip(grp, grp[1:]):
                ya = int(a.stem_top if a.stem_dir > 0 else a.stem_bot)
                yb = int(b2.stem_top if b2.stem_dir > 0 else b2.stem_bot)
                lo = min(ya, yb) - 2
                hi = max(ya, yb) + 2
                if a.stem_dir > 0:
                    hi += depth
                else:
                    lo -= depth
                cv2.rectangle(work, (int(a.stem_x) - 2, lo - y0),
                              (int(b2.stem_x) + 2, hi - y0), 0, -1)
        for n in grp if (g == -1 or len(grp) == 1) else []:
            ye = int(n.stem_top if n.stem_dir > 0 else n.stem_bot)
            lo, hi = (ye - 2, ye + int(2.0 * ss)) if n.stem_dir > 0 \
                else (ye - int(2.0 * ss), ye + 2)
            cv2.rectangle(work, (int(n.stem_x) - 2, lo - y0),
                          (int(n.stem_x + 1.4 * ss), hi - y0), 0, -1)

    # 3) 호 후보 성분 분석
    # 가로로 작은 틈(안티에일리어싱·구조물 소거 경계에서 호가 끊기는
    # 경우)을 메운다 — 실측: 진짜 붙임줄이 정확히 이 이유로 두 조각으로
    # 끊겨, 어느 조각도 양쪽 음표에 다 안 닿아 통째로 누락됨(여정 악보
    # 마디17→18, C5 붙임줄). 세로 커널은 1px 로 얇게 유지해 다른
    # 성분끼리 잘못 합쳐지는 부작용은 최소화한다.
    close_kernel = np.ones((1, 7), np.uint8)
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, close_kernel)
    n_lab, labels, stats, _c = cv2.connectedComponentsWithStats((work > 0).astype(np.uint8))
    out = []
    for i in range(1, n_lab):
        lx, ly, lw, lh, area = stats[i]
        # 짧은 붙임줄(인접한 두 음표 사이)은 1.1ss 정도로도 나타난다ㅡ
        # 2.0ss 고정 하한은 이런 짧은 호를 전부 놓친다(실측: guard 다수).
        if lw < 1.1 * ss or lh > 2.2 * ss or lw < 1.5 * max(lh, 1):
            continue
        comp = (labels[ly:ly + lh, lx:lx + lw] == i)
        # 채움비 필터: 진짜 호(가는 곡선)는 bbox 대비 잉크가 성기다
        # (실측 0.2~0.5). 노트헤드 잔재(타원 덩어리)가 구조 소거를 뚫고
        # 남으면 폭·높이 조건은 통과해도 채움비가 0.65 이상으로 뚜렷이
        # 높다(실측: guard 다수) — 이를 걸러 '없던 이음줄'을 없앤다.
        if area > 0.55 * lw * lh:
            continue
        col_ink = comp.sum(axis=0)
        cols_pos = col_ink[col_ink > 0]
        if cols_pos.size < 0.7 * lw or np.median(cols_pos) > 0.5 * ss:
            continue
        def ink_cy(cs):
            ys = np.where(comp[:, cs])[0]
            return float(ys.mean()) if len(ys) else None
        mid = ink_cy(lw // 2)
        e1, e2 = ink_cy(0), ink_cy(lw - 1)
        ends = [v for v in (e1, e2) if v is not None]
        if mid is None or not ends:
            continue
        bow = min(ends) - mid if mid < min(ends) else mid - max(ends)
        # 짧은 붙임줄(인접 음표 사이)은 획 자체가 3px 안팎이라 raw 곡률도
        # 작게 측정된다(실측: 0.10ss). 0.18ss 고정 하한은 이런 것들을
        # 전부 직선으로 오판해 배제했다.
        if abs(bow) < 0.04 * ss:            # 곡률 없는 직선(볼타·잔여 오선) 배제
            continue
        # 이음줄/붙임줄은 음표를 잇는다 — 양 끝점이 각각 어떤 머리의
        # x±1.3ss 안에 닿아야 한다. 임의 잉크 조각으로 생기는 '유령
        # 이음줄'(원본에 없는 호)을 차단한다.
        ex1, ex2 = lx, lx + lw
        near1 = [n for n in notes if abs(ex1 - n.x) <= 1.6 * ss]
        near2 = [n for n in notes if abs(ex2 - n.x) <= 1.6 * ss]
        if not near1 or not near2:
            continue
        # 세로 위치도 확인한다: 진짜 호는 항상 두 음표 '머리 옆'에 붙어
        # 있어 그 y 근방(±3ss)에서 나타난다. 실측(여정 악보): 가사 줄의
        # 장음 표시("ㅡ")나 코드 기호 파편이 음표에서 5ss 넘게 떨어진
        # 곳에 있는데도, x 위치만 맞으면 이 필터를 통과해 '없던 이음줄'
        # 로 오검됐다 — x 근접만으론 부족하고 y 근접도 함께 봐야 한다.
        arc_y = ly + y0 + (comp.shape[0] / 2.0)   # 성분의 세로 중심(절대좌표)
        if not any(abs(arc_y - n.y) <= 2.0 * ss for n in near1) \
                or not any(abs(arc_y - n.y) <= 2.0 * ss for n in near2):
            continue
        above = mid < min(ends)
        out.append({"line": s.index, "x1": float(lx), "x2": float(lx + lw),
                    "above": bool(above)})
    return out


# ------------------------------------------------------------------ 7. 엔진 본체
@dataclass
class EngineResult:
    norm_bin: np.ndarray
    clean: np.ndarray
    systems: list
    sys_results: list
    scale: float
    key: int = 0
    time_signature: tuple = (0, 0)
    elements: list = field(default_factory=list)
    pitches: list = field(default_factory=list)
    barlines: dict = field(default_factory=dict)
    barline_types: dict = field(default_factory=dict)
    measure_results: list = field(default_factory=list)
    staves: list = field(default_factory=list)
    slurs: list = field(default_factory=list)
    ts_symbol: str | None = None       # "common" = C 기호로 표기된 4/4
    sys_capacity: dict = field(default_factory=dict)  # {줄: 이 줄의 실효 용량}
    # (박자 변화가 없으면 전역 용량과 같은 값. 못갖춘마디로 인한 마디별 예외는
    #  measure_results 의 "anacrusis" 상태로 별도 표시되며 여기 포함되지 않는다.)


def _pitch_index(s: SystemInfo, y: float):
    pos = (s.bottom - y) / (s.ss / 2.0)
    idx = int(round(pos)) + 6           # bottom line(E4) = index 6
    if idx < _PITCH_LO - 2 or idx > _PITCH_HI + 2:
        return None
    return max(_PITCH_LO, min(_PITCH_HI, idx))


def run(bgr: np.ndarray) -> EngineResult:
    b = binarize(bgr)
    systems0 = detect_systems(b)
    nb, systems, scale = normalize(b, systems0)
    clean = remove_staff_lines(nb, systems)
    clean = remove_floating_chord_symbols(clean, systems)
    # remove_floating_lyrics_below 는 아직 연결하지 않는다 — 실측 회귀에서
    # 심각한 손상이 발견됨(선린교가 32/32→11/23, 마디 수 자체가 줄어듦).
    # 함수 내부에 버그가 있는 것으로 보이나 원인 미파악 — 다음 라운드
    # 과제로 남긴다(PLAN.md 23장 참고).

    res = EngineResult(norm_bin=nb, clean=clean, systems=systems,
                       sys_results=[], scale=scale)
    res.staves = [y for s in systems for y in s.line_ys]

    all_meas = []
    global_ts = None
    global_key = None
    key_votes = []
    ts_votes = []
    beam_key = 0

    for s in systems:
        sr = SystemResult(info=s)
        accs = detect_accidentals(clean, s)
        clef_box, key, ts, head_end, ts_sym = detect_clef_key_time(clean, s, accs)
        if ts_sym and res.ts_symbol is None:
            res.ts_symbol = ts_sym
        sr.clef_box, sr.key_accidentals, sr.time_sig = clef_box, key, ts
        # 조표·박자표는 모든 단에 반복 표기되므로 '단별 다수결'로 정한다
        # (한 단의 검출 실패/오검이 전체를 오염시키지 않도록).
        key_votes.append(key)
        if ts:
            ts_votes.append(ts)

        # 쉼표 우선 중재: 고신뢰 쉼표 위치를 먼저 확정해 머리 검출에서 배제
        strong_rest_xs = [r.x for r in detect_rests(
            clean, s, [], boost=0.06,
            only=("restQuarter", "rest8th"))]   # 블록형(2분/온) 쉼표는
        # 음표 머리와 혼동될 수 있어 가드에서 제외한다(곱슬형만 가드).
        notes = [n for n in detect_notes(clean, nb, s, rest_guard=strong_rest_xs)
                 if n.x > head_end + 0.4 * s.ss]
        # 박자표 숫자를 머리로 오인한 잔재 제거: head_end 이후만 유지(위에서 처리)
        rests = [r for r in detect_rests(clean, s, notes) if r.x > head_end + 0.4 * s.ss]
        # 쉼표의 부점(점2분쉼표 등): 쉼표 오른쪽의 둥근 점 탐색.
        # 사각 블록형(온·2분쉼표)은 폭이 넓어(≈1.3ss) 고정 x_off0=0.6ss
        # 로는 탐색창이 블록 자기 자신의 오른쪽 가장자리 '안'에서 시작돼
        # 그 가장자리 잉크를 점으로 오검한다(실측: 도돌이 없는 온쉼표가
        # 항상 부점으로 오판). 쉼표 자신의 실측 반폭(rr.w/2)만큼 더
        # 물려서 시작한다.
        #
        # 갈고리형 쉼표(8·16·32분)는 글리프가 세로로 길고(예: 16분쉼표
        # ≈2.7ss) 점은 '중심(cy)'이 아니라 글리프 상단부 근처에 찍힌다
        # (실측: 점이 cy 보다 1.25ss 위). 그래서 이런 쉼표는 탐색 기준
        # y를 상단 쪽으로 옮기고 창도 넓힌다. 또한 부점의 '칸(space)
        # 안착' 규칙은 음표 전용이라 쉼표에는 끈다(parity_check=False).
        _zero = np.zeros_like(clean)
        for rr in rests:
            half_w_ss = (rr.w / 2.0) / s.ss if s.ss else 0.0
            x_off0 = max(0.6, half_w_ss + 0.25)
            if rr.beat in (4, 8, 16, 32):
                dot_cy = rr.y - rr.h / 2.0 + 0.5 * s.ss
                dc = _detect_dot_at(clean, s, rr.x, dot_cy, _zero,
                                    x_off0=x_off0, x_off1=x_off0 + 1.1,
                                    y_off=0.9, parity_check=False)
            else:
                dc = _detect_dot_at(clean, s, rr.x, rr.y, _zero,
                                    x_off0=x_off0, x_off1=x_off0 + 1.1)
            if dc >= 0.35:
                rr.dotted = True
        sr.notes, sr.rests = notes, rests

        # 마디선(음표 머리 위치를 알아야 기둥 오검을 배제할 수 있다)
        bars_raw = detect_barlines_sys(clean, s, [n.x for n in notes],
                                       [n.stem_x for n in notes if n.stem_x > 0])
        bar_xs, bar_types = classify_barlines(clean, s, bars_raw)
        # 줄 머리(조표·박자표 이전) 마디선 제거
        bar_pairs = [(x, k) for x, k in zip(bar_xs, [bar_types[x] for x in bar_xs])
                     if x > head_end + 0.5 * s.ss]
        sr.barlines = [x for x, _k in bar_pairs]
        sr.barline_types = {x: k for x, k in bar_pairs}

        # 임시표 → 가장 가까운 오른쪽 머리에 부착(조표 구간 제외).
        # 4분/8분 쉼표의 곱슬 획이 ♭로 오검될 수 있으므로 쉼표 근처는 버린다.
        for k, ax, ay, _c in accs:
            if ax <= head_end + 0.2 * s.ss:
                continue
            if any(abs(rr.x - ax) < 1.2 * s.ss for rr in rests):
                continue
            if k == "flat" and any(
                    nn.kind != "grace"
                    and 0.1 * s.ss <= (nn.x - ax) <= 0.75 * s.ss
                    and abs(nn.y - ay) <= 0.55 * s.ss for nn in notes):
                # ♭의 '볼' 자리에 확정 음표 머리가 있다 = 앞 음표의 깃발
                # 꼬리(획) + 다음 머리(볼) 조합의 가짜 ♭ — 기각.
                continue
            cand = [n for n in notes
                    if 0.35 * s.ss <= (n.x - ax) <= 2.4 * s.ss
                    and abs(n.y - ay) <= 0.9 * s.ss]
            if cand:
                cand.sort(key=lambda n: n.x - ax)
                cand[0].accidental = k

        # 기둥/빔 분석 → 박자
        occupied = np.zeros_like(clean)
        for rr in rests:
            # 음표의 부점 탐색이 바로 옆(≤2ss) 쉼표 글리프의 잉크를 점으로
            # 오검하는 사례가 있다(실측: music.jpg 점32분음표 — 사실은
            # 인접한 16분쉼표의 시작 획이었음). 쉼표 영역도 가려 둔다.
            rw = max(rr.w, 1.2 * s.ss)
            cv2.rectangle(occupied,
                         (int(rr.x - rw / 2), int(rr.y - rr.h)),
                         (int(rr.x + rw / 2), int(rr.y + rr.h)), 255, -1)
        for n in notes:
            cv2.circle(occupied, (int(n.x), int(n.y)), int(0.75 * s.ss), 255, -1)
            if n.stem_dir != 0:
                cv2.line(occupied, (int(n.stem_x), int(n.stem_top)),
                         (int(n.stem_x), int(n.stem_bot)), 255,
                         thickness=max(2, int(0.25 * s.ss)))
                # 깃발 구역도 마스킹: 위기둥 깃발은 오른쪽 아래로 말려 내려와
                # (down-stem 은 오른쪽 위로) 그 꼬리끝이 이웃/자기 점 창을
                # 침범해 가짜 부점(dotc 0.6~0.75)을 만든다. 다만 폭을 너무
                # 넓게 잡으면(예전 1.35ss) 머리 폭이 넓은 음표에서는 부점
                # 탐색 구역과 아예 겹쳐 '진짜' 부점까지 지워버린다(실측:
                # guard 8분음표). 갈고리 1~2개(8·16분)는 기둥 바로 옆
                # 0.55ss 로 충분하지만, '외따로 떨어진' 32분음표(갈고리
                # 3개)는 그보다 훨씬 크게 퍼져 이 폭을 넘긴다.
                #
                # 단, 빔으로 묶인 그룹의 일원이면 갈고리가 아니라 '직선
                # 빔'을 공유하므로 이렇게 넓게 퍼지지 않는다 — 이 경우
                # 좁은 폭을 그대로 쓴다. 그룹 여부는 이 시점에 아직
                # beam_group 이 없어(뒤에서 배정) 근접한 동일 방향 이웃
                # 유무로 추정한다(실측: music.jpg 점32분음표 빔 쌍의
                # 진짜 부점이 넓은 마스크에 지워졌었음).
                _nb = _count_beams(clean, s, n, _safe_beam_scan_cap(n, notes, s.ss))
                # '이웃이 있으니 깃발이 좁다'는 가정은 그 이웃이 실제로
                # 빔으로 묶일 수 있는 음표(검은 머리, 즉 8분음표 이하)일
                # 때만 성립한다. 2분·온음표는 빔이 없으니 아무리 가까워도
                # 이 음표의 깃발 폭에 영향을 줄 이유가 없다 — 예전엔 종류
                # 구분 없이 '방향 같고 가까우면' 전부 이웃으로 쳐서, 옆에
                # 2분음표가 있다는 이유만으로 깃발 마스크가 좁게 그려지고
                # 깃발 꼬리 일부가 그대로 남아 부점으로 오검됐다(실측:
                # Bravura 판각 8분음표+2분음표 인접 배치).
                _has_sibling = any(
                    nn is not n and nn.kind == "black"
                    and nn.stem_dir == n.stem_dir
                    and abs(nn.x - n.x) <= 5.5 * s.ss
                    for nn in notes)
                _extra = 0 if _has_sibling else max(0, _nb - 1)
                fx0 = int(n.stem_x)
                fx1 = int(n.stem_x + (0.55 + 0.35 * _extra) * s.ss)
                fh = 2.6 + 0.5 * _extra
                if n.stem_dir > 0:
                    fy0, fy1 = int(n.stem_top), int(n.stem_top + fh * s.ss)
                else:
                    fy0, fy1 = int(n.stem_bot - fh * s.ss) - 3, int(n.stem_bot)
                cv2.rectangle(occupied, (fx0, fy0), (fx1, fy1), 255, -1)
        for n in notes:
            if n.kind == "whole":
                # 온음표도 부점을 가질 수 있다(점온음표) — 종전에는 여기서
                # continue 로 빠져 점 검사 자체가 생략되는 버그가 있었다.
                n.beat = 1
                n.dot_conf = _detect_dot(clean, s, n, occupied)
                n.dotted = n.dot_conf >= 0.35
                continue
            beams = _count_beams(clean, s, n, _safe_beam_scan_cap(n, notes, s.ss)) if n.stem_dir != 0 else 0
            n.beams = beams
            if n.kind == "half":
                n.beat = 2
            elif n.kind == "grace":
                n.beat = 8
            else:
                n.beat = {0: 4, 1: 8, 2: 16, 3: 32}.get(min(beams, 3), 4)
            n.dot_conf = _detect_dot(clean, s, n, occupied)
            n.dotted = n.dot_conf >= 0.35 and n.kind != "grace"

        # 빔 그룹: 간격·기둥 방향 휴리스틱만으로 묶으면 '개별 깃발 8분음표'가
        # 이웃과 억지로 묶이거나(원본과 다른 빔), 진짜 빔 쌍이 끊긴다.
        # 구조 검증을 추가한다 — 두 기둥의 끝(빔 쪽)을 잇는 선분을 따라
        # 실제로 잉크가 이어져 있어야 같은 그룹이다.
        def _stems_connected(a, b):
            ya = a.stem_top if a.stem_dir > 0 else a.stem_bot
            yb = b.stem_top if b.stem_dir > 0 else b.stem_bot
            x0i, x1i = int(a.stem_x) + 2, int(b.stem_x) - 2
            if x1i <= x0i:
                return True
            tol = 0.65 * s.ss
            hh, ww = clean.shape
            hit = tot = 0
            for x in range(x0i, x1i + 1, 2):
                t = (x - x0i) / max(1, (x1i - x0i))
                ye = ya + (yb - ya) * t
                lo = max(0, int(ye - tol))
                hi = min(hh, int(ye + tol) + 1)
                tot += 1
                if x < ww and (clean[lo:hi, x] > 0).any():
                    hit += 1
            return tot > 0 and hit / tot >= 0.75

        prev = None
        for n in notes:
            if n.kind == "grace":
                continue                    # 꾸밈음은 빔 체인을 끊지 않는다
            if n.kind in ("black",) and n.beams >= 1:
                if prev is not None and n.x - prev.x < 5.5 * s.ss \
                        and prev.stem_dir == n.stem_dir \
                        and _stems_connected(prev, n):
                    n.beam_group = prev.beam_group
                else:
                    beam_key += 1
                    n.beam_group = beam_key
                prev = n
            else:
                prev = None

        # ── 개별 깃발 음표의 부점8분+16분 관용 문법 판별.
        # 빔 없이 깃발로만 적는 가창 악보에서, 점8분음표 바로 다음의 개별
        # 깃발 음표는 실제 갈고리 수와 무관하게 거의 항상 그 점8분의
        # 16분음표 짝이다(관례적 리듬). 이 경우만 16분으로 승격한다 —
        # 직접 측정된 갈고리 수(n.beams)를 무조건 깎아내리는 하향 규칙은
        # 두지 않는다(과거엔 그랬으나, 상대두께 필터 도입으로 갈고리 계수
        # 자체가 신뢰할 만해진 뒤로는 그 하향 규칙이 오히려 정확히 측정된
        # 16분음표를 8분으로 오분류시켰다 — 실측: music.jpg).
        grp_size = {}
        for n in notes:
            if n.beam_group != -1:
                grp_size[n.beam_group] = grp_size.get(n.beam_group, 0) + 1
        prev_n = None
        for n in notes:
            if n.kind == "grace":
                continue
            solo_flag = (n.kind == "black" and n.beams >= 1
                         and grp_size.get(n.beam_group, 1) <= 1)
            if solo_flag and prev_n is not None and prev_n.beat == 8 \
                    and prev_n.dotted and (n.x - prev_n.x) <= 4.5 * s.ss:
                n.beat = 16
                n.dotted = False
            prev_n = n

        res.slurs.extend(detect_arcs(clean, s, notes, sr.barlines))
        for _n in detect_triplets(clean, s, notes):
            _n.in_triplet = True
        res.sys_results.append(sr)

    # 다수결 집계(동률이면 앞 단 우선). 조표 0(없음)도 유효한 투표다.
    if key_votes:
        from collections import Counter as _Counter
        global_key = _Counter(k for k in key_votes if k is not None).most_common(1)
        global_key = global_key[0][0] if global_key else 0
    if ts_votes:
        from collections import Counter as _Counter
        global_ts = _Counter(ts_votes).most_common(1)[0][0]
    res.key = global_key or 0
    res.time_signature = global_ts or (0, 0)
    capacity = 4.0
    if global_ts and global_ts[0] and global_ts[1]:
        capacity = global_ts[0] * 4.0 / global_ts[1]

    # 문법 검증 — 3단계로 진행한다.
    #   1) 교정 없이 검산만 → 단별 실효 박자(eff) 추정
    #   2) '전 마디 만장일치'면 전역 박자표 자체를 재추정(글리프 오인 대비)
    #   3) 단별 실효 박자를 목표로 보수적 자동 교정 + 최종 상태 산출
    pass1 = []
    sys_eff = {}
    for sr in res.sys_results:
        results, eff, tc = grammar_pass(sr, capacity, autofix=False)
        pass1.extend(results)
        sys_eff[sr.info.index] = (eff, tc)

    if pass1:
        # 분자 교차검증·만장일치 재추정 투표에서 '박자 변화 추정' 단은 제외
        # 한다 — 선린교가처럼 절반의 단이 2/4 인 악보에서 그 마디들이 표를
        # 몰아줘 멀쩡한 분자(4)를 2로 뒤집는 오폭을 막는다.
        tc_lines = {li for li, (eff_, tc_) in sys_eff.items() if tc_}
        vote = [m for m in pass1 if m[0] not in tc_lines] or pass1
        sums = [m[2] for m in vote]
        # ── 분자 문법 교차검증: 분모 글리프는 확실한데(4·8은 형태가 단순)
        #    분자가 인쇄 폰트 차이로 오독되는 사례(music.jpg 의 6 → 8).
        #    후보 분자별로 '정확 일치×2 + 1.0 이내 부족×1'(부점·쉼표 누락은
        #    부족으로 나타난다) 점수를 매겨, 검출값을 3점 이상 이길 때만
        #    분자를 교정한다. 3개 검증 샘플에서는 동률이라 발동하지 않는다.
        if res.time_signature and res.time_signature[0] and res.time_signature[1]:
            n0, d0 = res.time_signature

            def _num_score(n_):
                cap_ = n_ * 4.0 / d0
                sc_ = 0
                for v in sums:
                    if abs(v - cap_) < 0.01:
                        sc_ += 2
                    elif 0 < cap_ - v <= 1.0:
                        sc_ += 1
                return sc_

            base = _num_score(n0)
            best_n, best_s = n0, base
            for n_ in (2, 3, 4, 6, 8, 9, 12):
                if n_ == n0:
                    continue
                s_ = _num_score(n_)
                if s_ > best_s:
                    best_n, best_s = n_, s_
            if best_n != n0 and best_s >= base + 3:
                res.time_signature = (best_n, d0)
                capacity = best_n * 4.0 / d0
                # 새 용량으로 단별 실효값 재산출
                sys_eff = {}
                for sr in res.sys_results:
                    _r2, eff2, tc2 = grammar_pass(sr, capacity, autofix=False)
                    sys_eff[sr.info.index] = (eff2, tc2)
                sums = [m[2] for sr in res.sys_results for m in sr.measures]

        vals, counts = np.unique(np.round(sums, 2), return_counts=True)
        mode_v = float(vals[np.argmax(counts)])
        share = counts.max() / len(sums)
        # 박자표 만장일치 보정: 글리프 인식과 달리 거의 모든 마디가 일관되게
        # 같은 합 V(≠capacity)면 실제 박자표를 (V,4)로 재추정한다.
        # (예: 나비야 — 2/4 숫자 폰트가 표준 글리프와 달라 4/4로 오인)
        # 일부 단만 다른 경우(선린교가 5·6단)는 박자 변화 추정으로만 표시.
        if share >= 0.8 and abs(mode_v - capacity) > 0.01 \
                and abs(mode_v - round(mode_v)) < 0.01 and 1 <= mode_v <= 12:
            res.time_signature = (int(round(mode_v)), 4)
            capacity = mode_v
            sys_eff = {k: (capacity, False) for k in sys_eff}

    all_meas = []
    for sr in res.sys_results:
        eff, tc = sys_eff.get(sr.info.index, (capacity, False))
        target = eff if (tc and eff < capacity) else capacity
        res.sys_capacity[sr.info.index] = target
        results, _e, _t = grammar_pass(sr, target, autofix=True)
        if target != capacity:
            # 상태 재라벨: 목표(실효 박자)에 맞으면 time_change 로 표시
            results = [(li, mi, tot, capacity,
                        "time_change" if st == "OK" else st)
                       for (li, mi, tot, _cap, st) in results]
        all_meas.extend(results)
    res.measure_results = all_meas

    # ── 못갖춘마디(여린내기): 곡 전체의 첫 마디와 마지막 마디 합이 정확히
    #    한 마디를 이루면 둘 다 정상으로 간주한다(관례적 표기).
    if len(res.measure_results) >= 2:
        first = res.measure_results[0]
        last = res.measure_results[-1]
        if first[4] == "under" and abs(first[2] + last[2] - capacity) < 0.01:
            res.measure_results[0] = first[:4] + ("anacrusis",)
            if last[4] == "under":
                res.measure_results[-1] = last[:4] + ("anacrusis",)

    # 모델 계약 형태로 변환
    for sr in res.sys_results:
        s = sr.info
        line = s.index
        objs = []
        for n in sr.notes:
            pit = _pitch_index(s, n.y)
            w0 = 1.2 * s.ss
            # 선택 하이라이트용 bbox: 머리만이 아니라 기둥(과 깃발 여유)까지
            # 포함해 '음표 전체'가 원본 패널에서 강조되게 한다.
            x0b = n.x - w0 / 2
            x1b = n.x + w0 / 2
            y0b = n.y - 0.6 * s.ss
            y1b = n.y + 0.6 * s.ss
            if n.stem_dir != 0 and n.stem_x > 0:
                x0b = min(x0b, n.stem_x - 1.5)
                x1b = max(x1b, n.stem_x + 1.5)
                y0b = min(y0b, n.stem_top)
                y1b = max(y1b, n.stem_bot)
                if n.beams >= 1 and n.beam_group == -1:
                    x1b = max(x1b, n.stem_x + 1.1 * s.ss)   # 깃발 폭 여유
            bbox = (int(x0b), int(y0b), int(x1b - x0b), int(y1b - y0b))
            beat = -n.beat if n.dotted else n.beat
            sx = int(n.stem_x if n.stem_x > 0 else n.x)
            objs.append(((line, bbox[0], bbox[1], bbox[2], bbox[3], beat, sx,
                          n.beam_group, n.accidental, n.kind == "grace",
                          n.in_triplet), pit))
        for r in sr.rests:
            bbox = (int(r.x - r.w / 2), int(r.y - r.h / 2), int(r.w), int(r.h))
            beat = -r.beat if r.dotted else r.beat
            objs.append(((line, bbox[0], bbox[1], bbox[2], bbox[3], beat,
                          int(r.x), -1, None, False), -1))
        objs.sort(key=lambda t: t[0][6])
        for el, p in objs:
            res.elements.append(el)
            res.pitches.append(p)
        res.barlines[line] = sorted(int(round(x)) for x in sr.barlines)
        res.barline_types[line] = {int(round(x)): k for x, k in sr.barline_types.items()}

    return res


# ------------------------------------------------------------------ 시각화
_PITCH_NAMES = [
    "F3", "G3", "A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4", "B4",
    "C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6", "D6", "E6",
]


def draw_overlay(res: EngineResult) -> np.ndarray:
    """인식 결과 오버레이(음표=초록, 쉼표=보라, 마디선=주황, 마디상태 배경색)."""
    vis = cv2.cvtColor(255 - res.norm_bin, cv2.COLOR_GRAY2BGR)
    status_color = {"OK": (200, 255, 200), "time_change": (255, 235, 190),
                    "over": (200, 200, 255), "under": (255, 220, 220)}
    for sr in res.sys_results:
        s = sr.info
        ss = s.ss
        y0, y1 = int(s.top - 3 * ss), int(s.bottom + 2.2 * ss)
        bounds = [s.x0] + sorted(int(x) for x in sr.barlines) + [s.x1]
        # 마디 배경
        seg_items = sorted([(n.stem_x if n.stem_x > 0 else n.x) for n in sr.notes]
                           + [r.x for r in sr.rests])
        mi = 0
        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]
            if not any(lo <= x < hi for x in seg_items):
                continue
            mi += 1
            st = next((t[4] for t in sr.measures if t[1] == mi), "OK")
            col = status_color.get(st, (230, 230, 230))
            ov = vis.copy()
            cv2.rectangle(ov, (int(lo), y0), (int(hi), y1), col, -1)
            vis = cv2.addWeighted(ov, 0.35, vis, 0.65, 0)
        for x in sr.barlines:
            cv2.line(vis, (int(x), int(s.top - 0.6 * ss)),
                     (int(x), int(s.bottom + 0.6 * ss)), (0, 140, 255), 2)
        for n in sr.notes:
            color = (0, 160, 0) if n.kind != "grace" else (160, 0, 160)
            cv2.circle(vis, (int(n.x), int(n.y)), int(0.62 * ss), color, 2)
            pit = _pitch_index(s, n.y)
            lab = _PITCH_NAMES[pit] if pit is not None else "?"
            beat = f"{n.beat}{'.' if n.dotted else ''}"
            if n.accidental:
                lab = {"sharp": "#", "flat": "b", "natural": "n"}[n.accidental] + lab
            if n.kind == "grace":
                beat = "g" + beat
            cv2.putText(vis, f"{beat}", (int(n.x - 0.6 * ss), int(s.top - 1.4 * ss)),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, (200, 0, 0), 1)
            cv2.putText(vis, lab, (int(n.x - 0.8 * ss), int(s.bottom + 1.8 * ss)),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 200), 1)
        for r in sr.rests:
            cv2.rectangle(vis, (int(r.x - r.w / 2), int(r.y - r.h / 2)),
                          (int(r.x + r.w / 2), int(r.y + r.h / 2)), (200, 0, 200), 2)
            cv2.putText(vis, f"r{r.beat}", (int(r.x - 0.6 * ss), int(s.top - 1.4 * ss)),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, (200, 0, 200), 1)
    return vis
