# score_render.py
# ---------------------------------------------------------------------------
# Score 배치(원본 좌표계 Layout) → PIL 이미지로 렌더링.
#
# 우측 화면을 '원본과 같은 좌표계의 이미지'로 만들어, 좌측 원본과 동일한
# 확대/축소/패닝 변환을 공유할 수 있게 한다(좌우 동기화).
#
# 렌더는 원본 프레임을 factor(F)배 확대한 캔버스에 그린다(확대 시 선명도 확보).
# 좌표/클릭 판정은 원본 프레임(layout)에서 처리하고, 그릴 때만 ×F 한다.
# ---------------------------------------------------------------------------
import math
import os
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import layout as sl
from . import smufl

INK = (17, 17, 17)
STAFF = (17, 17, 17)   # 오선 색상 — 이전에 (90,90,90)→(55,55,55)로 한 차례
                       # 진하게 했으나 "아직도 많이 부족하다"는 재확인 지적으로
                       # 이번엔 INK(음표와 동일한 거의 검정)로 완전히 맞춤.
                       # 오선·음표를 색으로 구분할 이유가 없다 — 원본 스캔도
                       # 둘 다 같은 진한 잉크다.
LABEL = (90, 90, 90)
TINT = {"over": (246, 213, 213), "under": (213, 226, 246), "ok": None}

# 플랫/샾 조표의 세로 위치(음이름 기준 다이아토닉 값). 높은음자리표 표준 배열.
_FLAT_DIA = {"B": 34, "E": 37, "A": 33, "D": 36, "G": 32, "C": 35, "F": 31}
_FLAT_SEQ = ["B", "E", "A", "D", "G", "C", "F"]
_SHARP_DIA = {"F": 38, "C": 35, "G": 39, "D": 36, "A": 33, "E": 37, "B": 34}
_SHARP_SEQ = ["F", "C", "G", "D", "A", "E", "B"]


def _smufl(d, glyph_char, cx, baseline_y, ss, fill=INK, anchor="mm"):
    """SMuFL 글리프 하나를 (cx, baseline_y)에 그린다. 폰트 없으면 False.

    SMuFL 글리프는 대부분 y원점(오선 기준선)에 정렬되도록 설계돼 있어
    PIL anchor 로 세로 중앙(mm) 정렬하면 대체로 자연스럽다. 개별 글리프의
    미세 보정은 호출부에서 baseline_y 를 조정해 맞춘다.
    """
    font = smufl.get_font(smufl.size_for_staff_space(ss))
    if font is None:
        return False
    d.text((cx, baseline_y), glyph_char, font=font, fill=fill, anchor=anchor)
    return True



_FONT_KO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "src", "fonts", "NanumGothic-Regular.ttf")


# [글자 크기 — 실사용 피드백] PDF에서 읽은 높이(h)는 '글자 자체의 실제 높이'인데, 그 값을
# 그대로 글꼴 크기로 쓰면 화면 글자가 원본보다 작아진다(나눔고딕 한글은 실제 높이가 글꼴
# 크기의 약 92% — 실측 0.91~0.94). 원본과 같은 높이가 되도록 나눠서 키운다.
KO_GLYPH_HEIGHT_RATIO = 0.92


_FONT_MJ_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "src", "fonts", "NanumMyeongjo-Regular.ttf")


@lru_cache(maxsize=32)
def _font_header(size):
    """[v2.31] 곡 정보(제목·작사작곡·빠르기말) 기본 글꼴 = 명조(번들 나눔명조, OFL) → Windows 바탕 → 고딕."""
    size = max(6, int(size))
    for path in (_FONT_MJ_PATH, "C:/Windows/Fonts/batang.ttc", "batang.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return _font_ko(size)


@lru_cache(maxsize=32)
def _font_ko(size):
    """한글이 들어간 텍스트(제목·작사작곡·빠르기말)용 글꼴. DejaVu에는 한글 글리프가
    없어 □로 깨진다. 번들 나눔고딕(OFL) → Windows 맑은 고딕 → DejaVu 순으로 시도."""
    size = max(6, int(size))
    for path in (_FONT_KO_PATH, "C:/Windows/Fonts/malgun.ttf", "malgun.ttf",
                 "NanumGothic.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return _font(size)


_FONT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "src", "fonts", "DejaVuSans.ttf")


@lru_cache(maxsize=64)
def _font_chord(size):
    """[v3.00] 기타 코드 글꼴: 번들 DejaVu Sans Bold(없으면 보통 굵기)."""
    import os
    size = max(6, int(size))
    p = os.path.join(os.path.dirname(_FONT_PATH), "DejaVuSans-Bold.ttf")
    for path in (p, _FONT_PATH):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return _font(size)


def _font(size):
    # [폰트 크기 무시 버그 수정 — 실사용 검증 후] "DejaVuSans.ttf"를 이름
    # 만으로 요청하면 PIL은 시스템에 설치된 폰트만 찾는다 — 리눅스
    # 개발환경엔 흔히 있지만 Windows에는 기본으로 없어서, 실사용자
    # 환경에서는 항상 예외가 나 ImageFont.load_default()로 폴백되고
    # 있었다. load_default()는 고정 크기 비트맵 폰트라 size 인자를
    # 완전히 무시한다 — 그래서 폰트 크기를 아무리 키워도(잇단음표 "3"
    # 등) 화면에 전혀 반영되지 않는 것처럼 보였다(실사용 검증 중 발견).
    # matplotlib에 포함된 자유 라이선스(Bitstream Vera 기반) DejaVu Sans를
    # 프로젝트에 직접 번들링해 절대경로로 로드하므로, 어떤 OS에서
    # 실행하든 항상 정확한 크기로 렌더링된다.
    size = max(6, int(size))
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def render(layout, orig_w, orig_h, F=3, selected=None, header_strips=None,
           return_margin=False, lyric_y_off=0.0):
    """Layout(원본 좌표계) → PIL 이미지. selected=강조할 Element.
    header_strips={sid:(PIL_RGB, x0, y0, w, h)} 있으면 그 시스템의 음자리표/조표/
    박자표는 원본 조각을 붙여 표시하고 직접 그리지 않는다(좌표는 원본 프레임 기준).

    오선이 원본 왼쪽 끝(x=0)에서 시작하면 음자리표·조표·박자표가 음수 x로
    캔버스 밖에 그려져 잘린다. 이를 막기 위해 왼쪽에 헤더 여백(left_margin)을
    두고 모든 내용을 그만큼 오른쪽으로 시프트한다. return_margin=True 면
    (이미지, left_margin_px) 튜플을 돌려줘, 패널이 원본 좌표계와의 오프셋을
    맞출 수 있게 한다.
    """
    header_strips = header_strips or {}
    ss0 = layout.staff_space
    # 헤더(음자리표·조표·박자표)는 이제 오선 '안쪽'에 그리므로 큰 좌측 여백이
    # 필요 없다. 다만 마디 번호가 오선 위 왼쪽에 붙고, 아래로 향한 기둥·꼬리가
    # 첫 음표에서 살짝 왼쪽으로 나갈 수 있어 작은 여백만 둔다.
    left_margin = int(ss0 * 1.2 * F)

    W = int(orig_w * F) + left_margin
    H = int(orig_h * F)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    ss = layout.staff_space * F                      # 렌더 좌표 오선 간격

    def X(wx): return wx * F + left_margin
    def Y(wy): return wy * F

    # 마디 배경(초과=빨강/부족=파랑)
    for mb in layout.measure_boxes:
        tint = TINT.get(mb.status)
        if tint:
            x0, y0, x1, y1 = mb.bbox
            d.rectangle([X(x0), Y(y0), X(x1), Y(y1)], fill=tint)

    # 시스템(오선 + 음자리표 + 조표 + 박자표)
    for sysL in layout.systems:
        _draw_system(d, sysL, layout.header, F, ss, left_margin,
                     skip_header=sysL.index in header_strips)

    # [Tier2] 브레이스(중괄호): brace_group이 같은 보표들(그랜드 스태프 등)을
    # 왼쪽에 묶어 그린다 — "이 보표들이 한 악기"라는 표준 조판 관례.
    from collections import defaultdict
    brace_groups = defaultdict(list)
    for sysL in layout.systems:
        if sysL.brace_group is not None and sysL.brace_group != -1:
            brace_groups[sysL.brace_group].append(sysL)
    for members in brace_groups.values():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda s: s.staff_top)
        y0 = members[0].lines_y[0] * F
        y1 = members[-1].lines_y[-1] * F
        x_brace = members[0].x_left * F + left_margin - ss * 0.55
        _draw_brace(img, d, x_brace, y0, y1, ss)

    # 마디선: 종류별로 그린다(일반 / 도돌이 시작 / 도돌이 끝 / 종료선)
    def _thin(bx, y0, y1):
        d.line([bx, y0, bx, y1], fill=INK, width=max(3, int(ss * 0.15)))

    def _thick(bx, y0, y1):
        tw = max(3, int(ss * 0.4))
        d.rectangle([bx - tw / 2, y0, bx + tw / 2, y1], fill=INK)

    def _dots(bx, sys_members_list):
        """도돌이 점 2개: 각 보표의 가운뎃줄 위·아래 칸에.

        [Tier2 버그 수정] 이전에는 보표 하나(sys_members[0], 첫 보표)만
        받아 그 보표에만 점을 찍었다 — 그랜드 스태프(2단보표)에서 아래
        보표에는 도돌이 점이 안 그려지는 문제였다(실사용 검증 중 지적
        받음). 이제 그 시스템의 모든 보표를 순회하며 각각 점을 찍는다.
        """
        r = max(2, ss * 0.16)
        for sysL in sys_members_list:
            L = _sysline(sysL, F)
            for cy in ((L[1] + L[2]) / 2, (L[2] + L[3]) / 2):
                d.ellipse([bx - r, cy - r, bx + r, cy + r], fill=INK)

    # [Tier2 버그 수정] 이전에는 next(...)로 같은 시스템의 "첫 번째
    # 보표(SystemLayout)"만 찾아 그 보표 오선(위~아래 5줄) 안에서만
    # 마디선을 그렸다 — 다중 보표(그랜드 스태프)에서 마디선이 아래
    # 보표까지 이어지지 않고 위 보표에서 끊겨 보이는 원인이었다(실사용
    # 검증 중 발견). 같은 index(시스템 번호)를 가진 보표 전체를 모아
    # 맨 위 보표의 첫 줄부터 맨 아래 보표의 마지막 줄까지 긋는다
    # (브레이스를 그릴 때 쓴 것과 동일한 방식).
    systems_by_index = {}
    for s in layout.systems:
        systems_by_index.setdefault(s.index, []).append(s)

    # [Tier2 신규] 시스템(줄) 맨 시작(브레이스 바로 오른쪽, 음자리표 앞)에
    # 여는 마디선을 하나 그린다 — 표준 악보 관례. 그동안은 "마디와 마디
    # 사이 경계선"만 그렸을 뿐, 시스템이 시작되는 지점 자체의 마디선이
    # 빠져 있었다(실사용 검증 중 지적받음).
    for sid, s_list in systems_by_index.items():
        s_sorted = sorted(s_list, key=lambda s: s.staff_top)
        y0s, y1s = Y(s_sorted[0].lines_y[0]), Y(s_sorted[-1].lines_y[4])
        bxs = X(s_sorted[0].x_left)
        _thin(bxs, y0s, y1s)

    # [v2.51] 줄 첫 마디의 도돌이 시작(굵은 + 얇은 + 점2) — 원본 자리에. 줄 중간 도돌이는 앞 마디선에 그린다.
    for mb in layout.measure_boxes:
        m_ = mb.measure
        if getattr(m_, "start_repeat", False) and getattr(m_, "start_repeat_x", None) is not None:
            mem_ = sorted(systems_by_index[m_.system], key=lambda s: s.staff_top)
            pg_ = []
            for sm_ in mem_:
                if pg_ and getattr(pg_[-1][-1], "part", 0) == getattr(sm_, "part", 0):
                    pg_[-1].append(sm_)
                else:
                    pg_.append([sm_])
            bx0 = X(m_.start_repeat_x * getattr(layout, "src_sx", 1.0)) + ss * 0.25
            for grp_ in pg_:
                ya, yb = Y(grp_[0].lines_y[0]), Y(grp_[-1].lines_y[4])
                _thick(bx0, ya, yb)
                _thin(bx0 + ss * 0.6, ya, yb)
                _dots(bx0 + ss * 1.0, grp_)

    for mb in layout.measure_boxes:
        sys_members = sorted(systems_by_index[mb.measure.system],
                             key=lambda s: s.staff_top)
        bx = X(mb.barline_x)
        kind = getattr(mb.measure, "end_barline", "normal")
        # [여러 파트] 마디선은 파트마다 따로 긋는다(노래 보표와 피아노 보표 사이는 끊김 —
        # 일반 악보 표기). 같은 파트의 보표끼리는 위아래로 이어진다(그랜드 스태프).
        part_groups = []
        for sm_ in sys_members:
            if part_groups and getattr(part_groups[-1][-1], "part", 0) == getattr(sm_, "part", 0):
                part_groups[-1].append(sm_)
            else:
                part_groups.append([sm_])
        for grp_members in part_groups:
            y0, y1 = Y(grp_members[0].lines_y[0]), Y(grp_members[-1].lines_y[4])

            if kind == "final":                     # 얇은 + 굵은
                _thin(bx - ss * 0.6, y0, y1)
                _thick(bx, y0, y1)
            elif kind == "double":                  # [신규] 겹세로줄(얇은 + 얇은)
                _thin(bx - ss * 0.35, y0, y1)
                _thin(bx, y0, y1)
            elif kind == "repeat_end":              # 점2 + 얇은 + 굵은
                _dots(bx - ss * 1.0, grp_members)
                _thin(bx - ss * 0.6, y0, y1)
                _thick(bx, y0, y1)
            elif kind == "repeat_start":            # 굵은 + 얇은 + 점2
                _thick(bx, y0, y1)
                _thin(bx + ss * 0.6, y0, y1)
                _dots(bx + ss * 1.0, grp_members)
            else:
                _thin(bx, y0, y1)

    # [Tier2 신규] 시스템(줄) 중간에 박자표가 바뀌는 마디 — 그 마디가
    # 속한 시스템의 모든 보표(높은음자리/낮은음자리 등)에 새 박자표를
    # 그린다. Audiveris는 이런 변화를 정확히 인식하는데(실측 확인),
    # 기존 렌더러는 "시스템 맨 처음에만 박자표를 그린다"는 전제라 이
    # 정보가 화면에 전혀 반영되지 않았다(실사용 검증 중 지적받음).
    seen_ts_measures = set()
    for mb in layout.measure_boxes:
        m = mb.measure
        tsc = getattr(m, "time_sig_change", None)
        if tsc is None or m.number in seen_ts_measures:
            continue
        seen_ts_measures.add(m.number)
        sys_members = sorted(systems_by_index[m.system], key=lambda s: s.staff_top)
        beats_val, beat_type_val = tsc
        # 이 마디가 도돌이 시작(start_repeat)이면 굵은선+점2개가 이미
        # bbox[0] 근처를 차지하므로, 박자표를 그 너머로 밀어야 겹치지
        # 않는다(실사용 검증 중 실제로 겹쳐 점이 안 보이는 걸 발견).
        base_off = ss * 2.6 if getattr(m, "start_repeat", False) else ss * 1.0
        tx = X(mb.bbox[0]) + base_off
        # [박자표-음자리표 겹침 수정 — 실사용 검증 후] 시스템 맨 앞 마디에서
        # 박자가 바뀌는 경우(교가 17마디 2/4, 21마디 4/4)에는 마디 경계가 곧
        # 오선 시작점이라 위 계산대로 그리면 높은음자리표 위에 그대로
        # 겹친다. 이때는 시스템 헤더(음자리표→조표) 바로 뒤, 즉 첫 시스템의
        # 처음 박자표와 같은 자리에 그려야 한다(_draw_system_header 규칙).
        if getattr(m, "first_in_system", False):
            f_h = getattr(sys_members[0], "fifths", None)
            n_acc_h = abs(layout.header["fifths"] if f_h is None else f_h)
            clef_x_h = X(sys_members[0].x_left) + ss * 1.9
            acc_gap_h = ss * 0.72
            tx = (clef_x_h + ss * 2.6 + max(0, n_acc_h - 1) * acc_gap_h + ss * 1.7) \
                if n_acc_h else (clef_x_h + ss * 2.8)
        for sm in sys_members:
            sm_lines = _sysline(sm, F)
            sm_top = sm.staff_top * F
            _draw_time_sig_at(d, tx, sm_lines, sm_top, ss, beats_val, beat_type_val,
                              (getattr(m, "ts_printed", None) or (0, 0, None))[2])   # [v3.40] C·₵ 기호

    # [v1.10 조표 바뀜] 줄 중간에서 바뀐 조표 — 그 줄의 모든 보표에(각 보표의 음자리표 기준)
    for km in getattr(layout, "key_marks", []):
        for sm in systems_by_index.get(km["sid"], []):
            _draw_key_sig(d, X(km["x"]), sm.staff_top * F, ss, km["fifths"],
                          getattr(sm, "clef", "G"), prev=km["prev"])
    # [v1.10 음자리표 변경] 줄 중간의 음자리표 — 바뀌는 음 바로 앞에 작게(표준 관례 약 75%)
    for cm in getattr(layout, "clef_marks", []):
        sm = next((s_ for s_ in systems_by_index.get(cm["sid"], [])
                   if s_.staff_no == cm["staff_no"]), None)
        if sm is None or not smufl.font_available():
            continue
        cx = X(cm["x"]) - ss * (3.3 if cm.get("acc") else 2.3)
        _draw_clef_glyph(d, cm["clef"], cx, _sysline(sm, F), ss * 0.75)

    # 마디 번호: 각 보표(시스템)의 첫 마디 번호를 오선 위 왼쪽에 작게 표시.
    # 곡 전체의 첫 마디(=1번)는 관례상 생략한다.
    num_font = _font(ss * 0.95)
    seen_nums = set()
    for mb in layout.measure_boxes:
        m = mb.measure
        if m.number in seen_nums:
            continue
        seen_nums.add(m.number)
        if not getattr(m, "first_in_system", False) or m.number == 1:
            continue
        sysL = next(s for s in layout.systems if s.index == m.system)
        nx = X(sysL.x_left) + ss * 0.3
        ny = Y(sysL.lines_y[0]) - ss * 1.3
        d.text((nx, ny), str(m.number), fill=INK, font=num_font, anchor="lm")

    # 글리프(그룹/단독)
    groups, singles = {}, []
    for g in layout.glyphs:
        (groups.setdefault(g.group_id, []).append(g) if g.group_id > 0 else singles.append(g))

    # [Tier2 버그 수정] 화음(chord_group) 구성음들이 각자 독립적으로
    # _draw_element -> _draw_note 를 호출하면, 각 노트가 자기 y좌표만
    # 보고 stem 방향(위/아래)을 따로 정해버려 같은 화음 안에서 방향이
    # 어긋나는 문제가 있었다(실사용 검증 중 지적받음). 화음은 표준
    # 기보법상 반드시 하나의 공통 방향(보통 오선 중심에서 가장 먼 음
    # 기준)을 써야 한다. 먼저 (마디, chord_group)별로 묶어 공통 방향을
    # 계산한 뒤, 그 값을 모든 멤버에 동일하게 전달한다.
    chord_map = {}
    plain_singles = []
    for g in singles:
        cg = getattr(g.element, "chord_group", -1)
        if cg != -1:
            chord_map.setdefault((id(g.measure), cg), []).append(g)
        else:
            plain_singles.append(g)

    for g in plain_singles:
        _draw_element(d, g, layout, F, ss, left_margin)

    for members in chord_map.values():
        if len(members) == 1:
            _draw_element(d, members[0], layout, F, ss, left_margin)
            continue
        sysL0 = _sys_of(layout, members[0])
        mid0 = (sysL0.staff_top + 2 * sysL0.staff_space) * F
        # 오선 중심에서 가장 멀리 떨어진 멤버를 기준으로 방향 결정
        # (표준 규칙: 그 음이 중심보다 아래(y가 큼)면 기둥은 위로).
        # 가운데 줄에서 가장 먼 음 쪽의 반대로 기둥을 낸다. 위아래로 똑같이 멀면 아래로(기보 관례 —
        # 예전엔 목록 순서에 따라 위로 가기도 해 데모곡 1마디 A·C 화음이 원본과 반대였다).
        below = max([mg.y * F - mid0 for mg in members if mg.y * F > mid0] or [0])
        above = max([mid0 - mg.y * F for mg in members if mg.y * F < mid0] or [0])
        chord_up = below > above + 0.5
        voice_dirs = [mg.stem_up for mg in members if getattr(mg, "stem_up", None) is not None]
        if voice_dirs:                                    # [v1.10 다성부] 성부 규칙이 우선
            chord_up = voice_dirs[0]
        # [화음 기둥 버그 수정 — 실사용 검증] 예전엔 구성음마다 자기 기둥을 그려, 2도 때문에 옆으로
        # 비킨 머리("봄이 오면" 14마디 E♭4)에 기둥이 따로 생겨 별개 음표처럼 보였다(Audiveris와
        # 모델은 같은 화음). 기둥은 화음에 하나만 — 비키지 않은 머리 열(가장 많은 x)의 음에서
        # 반대쪽 끝 음을 넘어 뻗게 그리고, 나머지 구성음은 머리만 그린다.
        notes_c = [mg for mg in members if mg.element.kind == "note"]
        for g in members:
            if g.element.kind == "rest":
                _draw_rest(d, g, layout, F, ss, left_margin)
        if notes_c:
            anchor = _chord_anchor(notes_c, chord_up, layout)
            n_fl = sl.beam_count(notes_c[0].element.duration_base)
            st_len = ss * (3.5 + 0.35 * (n_fl - 1)) if n_fl >= 1 else ss * 3.5
            if chord_up:
                tip_y = min(mg.y for mg in notes_c) * F - st_len
            else:
                tip_y = max(mg.y for mg in notes_c) * F + st_len
            for mg in notes_c:                              # [v2.30] 원본 기둥 끝(화음 음 중 하나라도)
                st_ = _src_stem_tip(mg.element, chord_up, layout, F)
                if st_ is not None and all(_lay_pristine(x.element) for x in notes_c):
                    tip_y = st_
                    break
            for g in notes_c:
                _draw_note(d, g, layout, F, ss, left_margin, stem_up=chord_up,
                           draw_flags=(g is anchor), draw_stem=(g is anchor),
                           stem_tip_y=(tip_y if g is anchor else None))

    for gs in groups.values():
        _draw_group(d, gs, layout, F, ss, left_margin)

    # [잇단음표 "3" 표시] 이전에는 in_triplet 값이 계산에만 쓰이고 화면에
    # 아무 표시가 없어, 실제로 셋잇단음표로 정확히 인식된 경우에도
    # 사용자가 "그냥 계산이 틀렸나?"라고 오해하기 쉬웠다(실사용 검증 중
    # 지적받음). 마디 안에서 연속으로 이어지는 in_triplet 요소들을 한
    # 그룹으로 보고, 그 위(또는 아래, 기둥 방향에 맞춰)에 작은 "3"을 그린다.
    #
    # [버그 수정 — 실사용 검증 후] m.elements는 오른손·왼손 등 여러
    # 보표의 요소를 원본 인식 순서 그대로(보표 무관하게 인터리브되어)
    # 담고 있다(실측: staff1, staff2, staff1, staff2, ... 순서로 번갈아
    # 저장됨). 예전 코드는 "m.elements 안에서 물리적으로 바로 다음
    # 요소"만 같은 그룹으로 묶었는데, 그 사이에 다른 보표의 음표가 끼면
    # (staff_no 체크로 걸러지긴 하지만) 그 지점에서 그룹이 끊겨버려,
    # 같은 보표의 셋잇단음표 3개가 실제로는 하나의 그룹인데도 각각
    # 독립된 1개짜리 그룹으로 쪼개지는 문제가 있었다(브라켓이 안 그려지고
    # [브라켓 그룹핑 재수정 — 실사용 재검증 후] 처음엔 (staff_no,voice_no)
    # 둘 다로 나눠서 묶었는데, 이러면 정확히 계산 버그(2/4 오표시)의
    # 원인이었던 것과 같은 현상 — Audiveris가 셋잇단음표 안의 각 음마다
    # 서로 다른 voice_no를 매기는 경우(실사용 확인: A5=voice1, G5=voice2,
    # F5=voice3, 셋 다 같은 잇단음표인데) — 를 브라켓 그룹핑에서도 그대로
    # 겪어, 3개 음이 전혀 안 묶이고 작은 "3"이 개별로 3번 찍히는 문제가
    # 있었다(실사용 검증 중 발견 — 이전에 "정상적으로 그려진다"고 잘못
    # 보고했었는데, 실제로 이 case를 렌더링해서 확인하지 않고 성급하게
    # 결론 내린 것이었다). voice_no는 무시하고 staff_no만 같으면 묶는다
    # (한 보표 안에서 동시에 서로 다른 잇단음표가 독립 진행되는 경우는
    # 극히 드물고, Audiveris의 voice 오배정이 훨씬 흔하다).
    glyph_by_el = {id(g.element): g for g in layout.glyphs}
    tri_font = _font(ss * 1.3)
    # [잇단음표 일반화] 묶음 판정은 score.model.tuplet_groups가 한다(MusicXML 내보내기와
    # 같은 기준). 숫자는 비율의 actual(3·5·6·7·2·4), 덜 채워진 그룹은 빨간 숫자로 경고.
    from .model import tuplet_groups
    for mb in layout.measure_boxes:
        m = mb.measure
        for group_els, complete in tuplet_groups(m.elements):
            if True:
                glyphs_in_group = [glyph_by_el[id(e2)] for e2 in group_els
                                  if id(e2) in glyph_by_el]
                if glyphs_in_group:
                    # [위치 규칙 재설계 — 실사용 요청] 일반 기보 관례를 따른다.
                    #  · 빔으로 한데 묶인 그룹 → 괄호 없이 숫자만, 빔 쪽(기둥 끝 바깥)
                    #  · 빔이 없는 그룹(꼬리 음표·쉼표 섞임) → 괄호+숫자, 기둥 쪽. 다만 그 보표에
                    #    가사가 있으면 가사와 부딪치지 않게 항상 위쪽
                    #  · 음표머리·기둥 끝·오선 중 가장 바깥 것에 바짝 붙인다
                    # (예전엔 늘 기둥 반대편 = 음표머리 쪽이라, 노래 보표에서 가사 쪽에 붙었다)
                    gx = sum(g2.x for g2 in glyphs_in_group) / len(glyphs_in_group)
                    sysL = _sys_of(layout, glyphs_in_group[0])
                    sp_w = sysL.staff_space
                    mid_w = sysL.staff_top + 2 * sp_w
                    notes_g = [g2 for g2 in glyphs_in_group if g2.element.kind == "note"] or glyphs_in_group
                    avg_y = sum(g2.y for g2 in notes_g) / len(notes_g)
                    stem_up = avg_y > mid_w  # _draw_note와 같은 판정 기준
                    if getattr(notes_g[0], "stem_up", None) is not None:
                        stem_up = notes_g[0].stem_up          # [v1.10 다성부]
                    bgs = {getattr(g2.element, "beam_group", -1) for g2 in glyphs_in_group}
                    beamed = (len(glyphs_in_group) >= 2 and len(bgs) == 1 and -1 not in bgs
                              and all(g2.element.kind == "note" for g2 in glyphs_in_group))
                    staff_has_lyrics = any(getattr(e3, "lyrics", None) for e3 in m.elements
                                           if e3.staff_no == group_els[0].staff_no)
                    if beamed:
                        above = stem_up
                    else:
                        above = True if staff_has_lyrics else stem_up
                    STEM = 3.5 * sp_w
                    top_note = min(g2.y for g2 in notes_g)
                    bot_note = max(g2.y for g2 in notes_g)
                    if above:
                        obstacle = min(sysL.lines_y[0],
                                       (top_note - STEM) if stem_up else (top_note - 0.6 * sp_w))
                        ty = Y(obstacle) - ss * (0.7 if beamed else 0.8)
                    else:
                        obstacle = max(sysL.lines_y[4],
                                       (bot_note + STEM) if not stem_up else (bot_note + 0.6 * sp_w))
                        ty = Y(obstacle) + ss * (0.7 if beamed else 0.8)
                    tx = X(gx)
                    x0 = X(glyphs_in_group[0].x)
                    x1 = X(glyphs_in_group[-1].x)
                    bw = max(2, int(ss * 0.05))
                    gap = ss * 0.9  # 숫자 좌우로 선을 끊어줄 여백
                    tick = ss * 0.35 * (1 if above else -1)   # 양 끝 눈금은 음표 쪽으로 꺾인다
                    if len(glyphs_in_group) >= 2 and not beamed:
                        if x1 - x0 > gap * 1.1:
                            d.line([x0, ty, tx - gap / 2, ty], fill=INK, width=bw)
                            d.line([tx + gap / 2, ty, x1, ty], fill=INK, width=bw)
                        else:
                            d.line([x0, ty, x1, ty], fill=INK, width=bw)
                        d.line([x0, ty, x0, ty + tick], fill=INK, width=bw)
                        d.line([x1, ty, x1, ty + tick], fill=INK, width=bw)
                    d.text((tx, ty), str(group_els[0].tup_actual),
                           fill=INK if complete else (220, 60, 50), font=tri_font, anchor="mm")

    # [곡 정보 표시 — 실사용 피드백] 제목(가운데)·작사작곡(오른쪽 정렬)·
    # 빠르기말(왼쪽 정렬). 좌표는 원본 이미지 픽셀(= world 좌표) 기준이라
    # 원본 패널의 같은 위치에 그려진다.
    for it in getattr(layout, "header_items", None) or []:
        if not it.get("text"):
            continue
        fnt = _font_header(it["h"] * F / KO_GLYPH_HEIGHT_RATIO)
        # 원본이 굵은 글꼴이면 테두리를 살짝 둘러 굵게(번들 글꼴은 보통 굵기 하나뿐)
        sw = max(1, int(it["h"] * F * 0.045)) if it.get("bold") else 0
        kw = dict(fill=INK, font=fnt, stroke_width=sw, stroke_fill=INK)
        if any(ch in it["text"] for ch in _TEMPO_GLYPH):
            # [v1.10 빠르기 ♩] 음표가 섞인 줄은 음표만 Bravura 메트로놈 글리프로 그린다
            ax = {"title": "m", "credit": "r"}.get(it["kind"], "l")
            xr = {"m": X((it["x0"] + it["x1"]) / 2), "r": X(it["x1"])}.get(ax, X(it["x0"]))
            _draw_text_with_notes(d, xr, Y(it["y"]), it["text"], it["h"] * F, ax, kw)
        elif it["kind"] == "title":
            d.text((X((it["x0"] + it["x1"]) / 2), Y(it["y"])), it["text"], anchor="mm", **kw)
        elif it["kind"] == "credit":
            d.text((X(it["x1"]), Y(it["y"])), it["text"], anchor="rm", **kw)
        else:
            d.text((X(it["x0"]), Y(it["y"])), it["text"], anchor="lm", **kw)

    # [가사 — PDF 인식분만] 음표에 붙은 가사를 원본에서 읽은 그 위치에 그린다
    # (음표 위치는 원본 좌표 그대로 유지되므로 가사도 원본과 같은 자리에 온다).
    for mb in layout.measure_boxes:
        for e in mb.measure.elements:
            for ly in (getattr(e, "lyrics", None) or {}).values():
                if ly.get("text"):
                    d.text((X(ly["x"]), Y(ly["y"] - lyric_y_off)), ly["text"], fill=INK,
                           font=_font_ko(ly["h"] * F / KO_GLYPH_HEIGHT_RATIO), anchor="mm")

    # [유령 요소 표시 — 실사용 검증 후] Audiveris가 단선율 악보에서 실제로는
    # 없는 쉼표를 보조 성부(voice 2 등)에 혼자 만들어 같은 위치의 음표와
    # 겹쳐 보이게 하는 경우가 있다(교가 5번 마디의 16분쉼표). 박자 계산에서는
    # 이미 무시되지만 화면에서는 음표를 가려서, "한 보표 안에서 혼자뿐인
    # 쉼표 성부 + 다른 성부 음표와 x가 거의 같음"인 요소를 주황 테두리로
    # 표시해 사용자가 바로 알아보고 삭제(Delete)할 수 있게 한다.
    ghost_by_el = {id(g.element): g for g in layout.glyphs}
    for mb in layout.measure_boxes:
        els = mb.measure.elements
        counts = {}
        for e in els:
            counts[(e.staff_no, e.voice_no)] = counts.get((e.staff_no, e.voice_no), 0) + 1
        for e in els:
            if e.kind != "rest" or counts[(e.staff_no, e.voice_no)] != 1:
                continue
            others = [o for o in els if o.staff_no == e.staff_no and o.voice_no != e.voice_no
                      and o.kind == "note"]
            gg = ghost_by_el.get(id(e))
            if gg is None or not others:
                continue
            clash = any(abs((ghost_by_el[id(o)].x if id(o) in ghost_by_el else 1e9) - gg.x)
                        < layout.staff_space * 1.2 for o in others)
            if clash:
                r = ss * 0.95
                d.rectangle([X(gg.x) - r, Y(gg.y) - r, X(gg.x) + r, Y(gg.y) + r],
                            outline=(230, 126, 34), width=max(3, int(ss * 0.1)))

    # 이음줄/붙임줄 호 — 음표 위에 그린다(2차 베지어 근사)
    for arc in getattr(layout, "arcs", []):
        _draw_arc(d, arc, X, Y, ss, F)

    # 볼타(1절/2절 등) 브래킷 — 마디 상단, 코드 기호보다 위에 그린다
    for end in getattr(layout, "endings", []):
        _draw_ending(d, end, X, Y, ss, F)

    # 크레센도/디크레센도 쐐기 — 보표 아래에 그린다
    for hp in getattr(layout, "hairpins", []):
        _draw_hairpin(d, hp, X, Y, ss, F)
    for dshape in getattr(layout, "dynamics", []):
        _draw_dynamic(d, dshape, X, Y, ss)
    for mk in getattr(layout, "marks", []):            # [v2.80] 이동·옥타브·페달 기호
        _draw_mark(d, mk, X, Y, ss, F)

    # [늘임표 — 실사용 버그 수정] Audiveris가 늘임표를 정확히 인식하고(Element.fermata) U 단축키로
    # 켜고 끌 수도 있었는데, 렌더러에 그리는 코드가 없어 화면에 전혀 안 나왔다. 화음은 한 번만,
    # 맨 위 음(또는 위로 뻗은 기둥 끝)과 오선 윗줄 중 더 높은 곳 위에 그린다.
    seen_f = set()
    for g in layout.glyphs:
        e = g.element
        if not getattr(e, "fermata", False):
            continue
        cg = getattr(e, "chord_group", -1)
        key = (id(g.measure), cg) if cg != -1 else (id(g.measure), id(e))
        if key in seen_f:
            continue
        seen_f.add(key)
        sysL = _sys_of(layout, g)
        mem = [gg for gg in layout.glyphs if cg != -1 and gg.measure is g.measure
               and getattr(gg.element, "chord_group", -1) == cg] or [g]
        top_y = min(gg.y for gg in mem)
        bot_y = max(gg.y for gg in mem)
        sp = layout.staff_space
        mid = sysL.staff_top + 2 * sysL.staff_space
        stem_up = e.kind == "note" and e.duration_base != 1 and (bot_y + top_y) / 2 > mid
        if getattr(g, "stem_up", None) is not None and e.kind == "note" and e.duration_base != 1:
            stem_up = g.stem_up
            if not stem_up:
                # [v1.10 다성부] 아래 성부의 늘임표는 관례대로 아래쪽(오선 아래)에 거꾸로
                fy = max(sysL.lines_y[4], bot_y + sp * 3.4) + sp * 0.75
                _smufl(d, "\uE4C1", X(g.x), Y(fy), ss)
                continue
        # [간격 조정 — 실사용 요청] 가장 바깥 장애물(머리 위·기둥 끝·오선 윗줄) 바로 위에 붙인다
        reach = top_y - sp * 3.4 if stem_up else top_y - sp * 0.5
        fy = min(sysL.lines_y[0], reach) - sp * 0.75
        _smufl(d, "\uE4C0", X(g.x), Y(fy), ss)

    # [v1.40 미인식 보완] 제안 음표 — 주황 머리(+덧줄) + 둥근 테두리. 받아들이면 보통 음표가 된다.
    SUG = (230, 126, 34)
    for g in getattr(layout, "suggestion_glyphs", []):
        x, y = X(g.x), Y(g.y)
        for ly in g.ledger:
            d.line([x - ss * 0.9, Y(ly), x + ss * 0.9, Y(ly)], fill=SUG, width=max(2, int(ss * 0.11)))  # 머리0.59+0.3
        gs_ = ss * (0.65 if getattr(g.element, "is_grace", False) else 1.0)   # 꾸밈음은 작게
        if smufl.font_available():
            _smufl(d, _head_glyph(g.element), x, y, gs_, fill=SUG)
            if getattr(g.element, "is_grace", False):        # 꾸밈음 표시: 짧은 기둥 + 빗금
                d.line([x + gs_ * 0.55, y, x + gs_ * 0.55, y - gs_ * 2.6], fill=SUG, width=max(2, int(ss * 0.1)))
                d.line([x - gs_ * 0.3, y - gs_ * 0.6, x + gs_ * 1.2, y - gs_ * 1.9], fill=SUG,
                       width=max(2, int(ss * 0.08)))
        r = gs_ * 1.05
        d.ellipse([x - r, y - r, x + r, y + r], outline=SUG, width=max(2, int(ss * 0.1)))

    # 원본에서 잘라온 음자리표/조표/박자표 조각 붙이기
    for sid, (pil, x0, y0, w, h) in header_strips.items():
        tw, th = max(1, int(w * F)), max(1, int(h * F))
        img.paste(pil.resize((tw, th)), (int(x0 * F) + left_margin, int(y0 * F)))

    # 선택 강조
    if selected is not None:
        for g in layout.glyphs:
            if g.element is selected:
                x0, y0, x1, y1 = g.cell
                d.rectangle([X(x0), Y(y0), X(x1), Y(y1)],
                            outline=(29, 158, 117), width=max(2, F))
                break
    if return_margin:
        return img, left_margin
    return img


# [v1.10 빠르기 ♩] 화면 문자 → Bravura 메트로놈 글리프(SMuFL metNote…Up)
_TEMPO_GLYPH = {"\u2669": "\uECA5", "\u266A": "\uECA7", "\U0001D15E": "\uECA3",
                "\U0001D15D": "\uECA2"}


def _draw_text_with_notes(d, x, y, text, h_px, anchor, kw):
    """글자 사이의 음표 문자(♩ 등)를 악보 글꼴로 그린다. anchor: l/m/r(가로), y는 글자 가운데."""
    note_font = smufl.get_font(max(8, int(h_px * 1.75))) if smufl.font_available() else None
    segs, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch in _TEMPO_GLYPH and note_font is not None:
            g = _TEMPO_GLYPH[ch]
            if i + 1 < len(text) and text[i + 1] == ".":
                g += "\uECB7"                               # metAugmentationDot
                i += 1
            segs.append(("n", g))
        elif segs and segs[-1][0] == "t":
            segs[-1] = ("t", segs[-1][1] + ch)
        else:
            segs.append(("t", ch))
        i += 1

    def _w(seg):
        f = note_font if seg[0] == "n" else kw["font"]
        bb = d.textbbox((0, 0), seg[1], font=f, anchor="ls")
        return bb[2] - bb[0] + (h_px * 0.15 if seg[0] == "n" else 0)
    widths = [_w(sg) for sg in segs]
    total = sum(widths)
    cx = x - (total if anchor == "r" else total / 2 if anchor == "m" else 0)
    base_y = y + h_px * 0.5                                  # 글자 아랫선
    for sg, w in zip(segs, widths):
        if sg[0] == "n":
            sp = h_px * 1.75 / 4.0                           # 이 크기의 오선 한 칸
            d.text((cx + h_px * 0.05, base_y - sp * 0.5), sg[1], font=note_font, fill=INK, anchor="ls")
        else:
            d.text((cx, y), sg[1], anchor="lm", **kw)
        cx += w


def _draw_arc(d, arc, X, Y, ss, F):
    """이음줄/붙임줄 호를 2차 베지어 곡선으로 그린다.

    양 끝점 사이 거리에 비례해 아치 깊이를 정하되(너무 평평하거나 과하게
    부풀지 않도록 오선 간격 기준으로 상·하한을 둔다), 위/아래 볼록은
    arc.above 로 정한다. 가운데를 살짝 굵게 그려 실제 악보의 호처럼 보이게 한다.
    """
    x1, y1 = X(arc.x1), Y(arc.y1)
    x2, y2 = X(arc.x2), Y(arc.y2)
    if getattr(arc, "poly", None):
        # [v2.31] 원본 잉크를 따라간 점 목록 그대로, 가운데를 굵게
        pts = [(X(px), Y(py)) for px, py in arc.poly]
        wmax = max(3, int(round(ss * 0.26)))
        wmin = max(2, int(round(ss * 0.10)))
        for i in range(len(pts) - 1):
            t = i / max(1, len(pts) - 1)
            w_ = int(round(wmin + (wmax - wmin) * (1 - abs(2 * t - 1))))
            d.line([pts[i], pts[i + 1]], fill=INK, width=max(1, w_))
        return
    if getattr(arc, "c1x", None) is not None:
        # [v2.30 원본 곡선] 원본 3차 베지어 그대로(제어점 2개), 가운데를 굵게
        c1 = (X(arc.c1x), Y(arc.c1y))
        c2 = (X(arc.c2x), Y(arc.c2y))
        pts = []
        for i in range(41):
            t = i / 40
            u = 1 - t
            pts.append((u ** 3 * x1 + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t ** 3 * x2,
                        u ** 3 * y1 + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t ** 3 * y2))
        wmax = max(3, int(round(ss * 0.26)))
        wmin = max(2, int(round(ss * 0.10)))
        for i in range(len(pts) - 1):
            t = i / (len(pts) - 1)
            w_ = int(round(wmin + (wmax - wmin) * (1 - abs(2 * t - 1))))
            d.line([pts[i], pts[i + 1]], fill=INK, width=max(1, w_))
        return
    span = abs(x2 - x1)
    # [곡률 재조정 — 실사용 검증 후] 긴 이음줄(여러 박자에 걸친 것)이
    # 상한(2.2칸)까지 그대로 부풀어 "원형에 가깝고 투박해 보인다"는
    # 지적을 받았다. 원본 악보처럼 납작한 타원형이 되도록 span에 따른
    # 증가율과 상한을 함께 낮췄다.
    depth = min(max(span * 0.12, ss * 0.6), ss * 1.5)   # 아치 깊이(상·하한)
    if arc.above:
        depth = -depth
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + depth * 2   # 제어점(베지어)

    pts = []
    for i in range(33):
        t = i / 32
        u = 1 - t
        px = u * u * x1 + 2 * u * t * cx + t * t * x2
        py = u * u * y1 + 2 * u * t * cy + t * t * y2
        pts.append((px, py))
    # [두께 재조정 — 실사용 검증 후] 오선 굵기(ss*0.09 기준)보다 확실히
    # 두꺼워 보이도록 ss 비례로 다시 계산했다(양 끝도 원본처럼 가늘지
    # 않고 어느 정도 두께를 유지, 중앙은 오선보다 눈에 띄게 두껍게).
    wmax = max(3, int(round(ss * 0.26)))
    wmin = max(2, int(round(ss * 0.10)))
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        t = (i + 0.5) / (len(pts) - 1)
        wid = int(round(wmin + (wmax - wmin) * (1 - abs(2 * t - 1))))
        d.line([a, b], fill=INK, width=max(wmin, wid))


def _lay_pristine(e):
    from score.layout import _pristine
    return _pristine(e)


def _src_y(layout, F, y_src):
    return (y_src * getattr(layout, "src_sy", 1.0) - getattr(layout, "y_shift", 0.0)) * F


def _src_stem_tip(e, up, layout, F):
    """[v2.30] 원본 기둥의 끝(위로면 위 끝, 아래로면 아래 끝). 편집한 음·방향이 다르면 None."""
    st = getattr(e, "src_stem", None)
    if st is None or not _lay_pristine(e) or not e.src_bbox:
        return None
    _x, top, bot = st
    head_cy = e.src_bbox[1] + e.src_bbox[3] / 2.0
    if up and top < head_cy - 0.5 * e.src_bbox[3]:
        return _src_y(layout, F, top)
    if (not up) and bot > head_cy + 0.5 * e.src_bbox[3]:
        return _src_y(layout, F, bot)
    return None


def _src_beam_line(anchors, members, stem_up, layout, F, xoff):
    """[v2.30] 원본 빔: 묶음 음표들의 기둥에 붙은 빔 중 바깥(위로 기둥이면 가장 위) 빔의 바깥 가장자리 선을
    그리기 좌표 (X1,Y1,X2,Y2)로. 묶음 안에 편집한 음이 있거나, 빔이 기둥 쪽에 있지 않으면 None."""
    if not anchors or any(not _lay_pristine(g.element) for mem in members for g in mem):
        return None
    beams = []
    for g in anchors:
        for b in (getattr(g.element, "src_beams", None) or []):
            if b not in beams:
                beams.append(b)
    if not beams:
        return None
    heads_y = [g.element.src_bbox[1] + g.element.src_bbox[3] / 2.0 for g in anchors if g.element.src_bbox]
    if not heads_y:
        return None
    mean_b = lambda b: (b[1] + b[3]) / 2.0
    outer = min(beams, key=mean_b) if stem_up else max(beams, key=mean_b)
    if stem_up and mean_b(outer) > min(heads_y):
        return None
    if (not stem_up) and mean_b(outer) < max(heads_y):
        return None
    same = [b for b in beams if abs(mean_b(b) - mean_b(outer)) < max(2.0, outer[4])]
    b = max(same, key=lambda q: abs(q[2] - q[0]))           # 같은 층에서 가장 긴 빔
    sx = getattr(layout, "src_sx", 1.0)
    edge = -b[4] / 2.0 if stem_up else b[4] / 2.0
    return (b[0] * sx * F + xoff, _src_y(layout, F, b[1] + edge),
            b[2] * sx * F + xoff, _src_y(layout, F, b[3] + edge))


def _measure_left(layout, measure):
    """마디 상자의 왼쪽(마디선) x — 곡 좌표. 레이아웃마다 한 번 표로 만든다."""
    tab = getattr(layout, "_mb_left_tab", None)
    if tab is None:
        tab = {id(mb.measure): mb.bbox[0] for mb in layout.measure_boxes}
        layout._mb_left_tab = tab
    return tab.get(id(measure))


# [v2.03] SMuFL 아티큘레이션 글리프(위·아래)
_ARTIC_GLYPH = {
    ("accent", True): "\uE4A0", ("accent", False): "\uE4A1",
    ("staccato", True): "\uE4A2", ("staccato", False): "\uE4A3",
    ("tenuto", True): "\uE4A4", ("tenuto", False): "\uE4A5",
    ("staccatissimo", True): "\uE4A6", ("staccatissimo", False): "\uE4A7",
    ("marcato", True): "\uE4AC", ("marcato", False): "\uE4AD",
    # [v2.80] 꾸밈·숨표
    ("trill", True): "\uE566", ("trill", False): "\uE566",
    ("mordent", True): "\uE56C", ("mordent", False): "\uE56C",
    ("turn", True): "\uE567", ("turn", False): "\uE567",
    ("breath", True): "\uE4CE", ("breath", False): "\uE4CE",
    ("caesura", True): "\uE4D1", ("caesura", False): "\uE4D1",          # [v2.90]
}
_TREMOLO_GLYPH = {1: "\uE220", 2: "\uE221", 3: "\uE222"}

# [v2.80] 이동·옥타브·페달 기호(글리프 또는 글자)
_MARK_GLYPH = {"segno": "\uE047", "coda": "\uE048", "8va": "\uE511", "8vb": "\uE51C", "15ma": "\uE514",
               "ped": "\uE650", "ped_up": "\uE655"}
_MARK_TEXT = {"dc": "D.C.", "ds": "D.S.", "fine": "Fine", "dc_al_fine": "D.C. al Fine",
              "dc_al_coda": "D.C. al Coda", "ds_al_fine": "D.S. al Fine", "ds_al_coda": "D.S. al Coda",
              "to_coda": "To Coda"}


def _draw_mark(d, mk, X, Y, ss, F):
    """[v2.80] 이동 기호(세뇨·코다·D.C.…)·옥타브 기호(8va…, 점선과 끝 갈고리)·페달(Ped. … *)."""
    kind = mk["kind"]
    if kind == "chord":                                 # [v3.00] 기타 코드
        anchor = "mm" if mk.get("orig") else "lm"
        d.text((X(mk["x"]), Y(mk["y"])), mk["text"], font=_font_chord(ss * 1.55), fill=INK, anchor=anchor)
        return
    if kind == "text":                                  # [v2.90] cresc. 같은 글자 셈여림(기울임 없는 명조)
        d.text((X(mk["x"]), Y(mk["y"])), mk["text"], font=_font_header(ss * 1.5), fill=INK, anchor="mm")
        return
    if kind == "ped_line":                              # [v2.90] 계단형 페달 선 |____^____|
        x1, x2, y = X(mk["x1"]), X(mk["x2"]), Y(mk["y"])
        w = max(2, int(round(ss * 0.12)))
        pts = [(x1, y - ss * 1.0), (x1, y)]
        for cx in sorted(mk.get("changes") or []):
            c = X(cx)
            pts += [(c - ss * 0.35, y), (c, y - ss * 0.8), (c + ss * 0.35, y)]
        pts += [(x2, y), (x2, y - ss * 1.0)]
        d.line(pts, fill=INK, width=w, joint="curve")
        return
    if kind == "gliss":                                 # [v2.90] 글리산도 선
        d.line([(X(mk["x1"]), Y(mk["y1"])), (X(mk["x2"]), Y(mk["y2"]))], fill=INK,
               width=max(2, int(round(ss * 0.12))))
        return
    if kind in _MARK_TEXT:
        fnt = _font_header(ss * 1.6)
        d.text((X(mk["x"]), Y(mk["y"])), _MARK_TEXT[kind], font=fnt, fill=INK, anchor="mm")
        return
    if kind in ("segno", "coda"):
        _smufl(d, _MARK_GLYPH[kind], X(mk["x"]), Y(mk["y"]), ss)
        return
    if kind in ("8va", "8vb", "15ma"):
        x1, x2, y = X(mk["x1"]), X(mk["x2"]), Y(mk["y"])
        _smufl(d, _MARK_GLYPH[kind], x1 + ss * 0.9, y, ss)
        w = max(1, int(round(ss * 0.1)))
        xs = x1 + ss * 2.2
        while xs < x2:                                   # 점선
            d.line([(xs, y), (min(x2, xs + ss * 0.5), y)], fill=INK, width=w)
            xs += ss * 0.9
        hook = ss * 0.9 if kind == "8vb" else -ss * 0.9
        d.line([(x2, y), (x2, y - hook)], fill=INK, width=w)   # 8va는 아래로, 8vb는 위로 갈고리
        return
    if kind == "pedal":
        _smufl(d, _MARK_GLYPH["ped"], X(mk["x1"]), Y(mk["y1"]), ss)
        if mk.get("x2") is not None:
            _smufl(d, _MARK_GLYPH["ped_up"], X(mk["x2"]), Y(mk["y2"]), ss)


def _draw_arpeggio(d, x, y_top, y_bot, ss):
    """[v2.80] 아르페지오: 화음 왼쪽 세로 물결선."""
    import math
    pts = []
    n = max(4, int((y_bot - y_top) / 2))
    for i in range(n + 1):
        yy = y_top + (y_bot - y_top) * i / n
        pts.append((x + math.sin((yy - y_top) / (ss * 0.5) * math.pi) * ss * 0.22, yy))
    d.line(pts, fill=INK, width=max(2, int(round(ss * 0.12))))


def _draw_ending(d, end, X, Y, ss, F):
    """볼타(1절/2절 등 엔딩) 브래킷을 그린다.

    구조: 왼쪽에 아래로 향한 짧은 훅 + 긴 가로선 + (닫힌 경우) 오른쪽에도 훅.
    번호는 왼쪽 훅 바로 오른쪽 위에 "1." 형식으로 쓴다.

    인접한 두 볼타(예: 1절의 닫는 훅과 2절의 여는 훅)는 같은 x(공유 마디선)에
    걸리는 경우가 있어, 훅을 그대로 그리면 겹쳐 보인다. 가로선은 원래 x범위
    그대로 두고, 훅만 살짝 안쪽으로 당겨 두 훅 사이에 틈을 준다.
    """
    x1, x2 = X(end.x1), X(end.x2)
    y = Y(end.y)
    # [v2.02 — 실사용] 원본처럼: 다리 길이 약 2칸, 선 굵기 0.1칸, 번호 높이 약 2칸.
    # (예전 번호 크기 ss*2.6*3 은 옛 엔진의 작은 ss 기준이라 지금은 7칸 넘게 커져 오선을 덮었다)
    hook = ss * 2.0
    w = max(3, int(round(ss * 0.2)))     # [v2.54] 원본보다 얇아 보여 2배(0.1 → 0.2칸)
    hook_gap = ss * 0.35            # 인접 볼타 훅이 겹치지 않도록 안쪽으로 당기는 양
    lx = x1 + hook_gap
    # 가로선은 왼쪽 다리에서 시작하고(마디선 쪽으로 삐져나오지 않게), 열린 끝은 마디선 조금 앞에서 멈춘다
    rx = x2 - hook_gap if not end.open_end else x2 - ss * 0.6
    d.line([lx, y, rx, y], fill=INK, width=w)               # 가로선
    d.line([lx, y, lx, y + hook], fill=INK, width=w)         # 왼쪽 다리
    if not end.open_end:
        d.line([rx, y, rx, y + hook], fill=INK, width=w)     # 오른쪽 다리(닫힌 경우)
    f = _font_header(ss * 2.2)          # [v2.54] 10% 작게, 가는 명조(원본의 '1.'과 같은 세리프 숫자)
    d.text((lx + ss * 0.4, y + ss * 0.25), f"{end.number}.",
           fill=INK, font=f, anchor="lt")


# [셈여림] 내부 키 → SMuFL(Bravura) 정식 셈여림 글리프. 화면 문구와는 별개인 내부 키다.
DYN_GLYPHS = {
    "ppp": "\uE52A", "pp": "\uE52B", "p": "\uE520", "mp": "\uE52C", "mf": "\uE52D",
    "f": "\uE522", "ff": "\uE52F", "fff": "\uE530", "fp": "\uE534", "fz": "\uE535",
    "sf": "\uE536", "sfp": "\uE537", "sfz": "\uE539", "rfz": "\uE53C",
}


def _draw_dynamic(d, dshape, X, Y, ss):
    glyph = DYN_GLYPHS.get(dshape.key)
    if glyph and smufl.font_available():
        _smufl(d, glyph, X(dshape.x), Y(dshape.y), ss)
    else:                                   # 글꼴이 없으면 기울임 굵은 글자로 대체
        d.text((X(dshape.x), Y(dshape.y)), dshape.key, fill=INK, font=_font(ss * 1.4), anchor="mm")


def _draw_hairpin(d, hp, X, Y, ss, F):
    """크레센도(<)/디크레센도(>) 쐐기를 그린다.

    두 개의 직선이 한쪽 끝(점)에서 만나고 반대쪽 끝(열린 쪽)에서 벌어지는
    모양. cresc.=왼쪽이 점·오른쪽이 열림, decresc.=오른쪽이 점·왼쪽이 열림.
    """
    x1, x2 = X(hp.x1), X(hp.x2)
    y = Y(hp.y)
    half = (hp.half * F) if getattr(hp, "half", None) else ss * 0.55   # 열린 쪽 반높이
    w = max(2, int(round(ss * 0.12)))                   # 오선 굵기에 맞춘 선 두께
    if hp.kind == "cresc":
        pt_x, open_x = x1, x2
    else:
        pt_x, open_x = x2, x1
    d.line([pt_x, y, open_x, y - half], fill=INK, width=w)
    d.line([pt_x, y, open_x, y + half], fill=INK, width=w)


def _sysline(sysL, F):
    return [y * F for y in sysL.lines_y]


def _draw_system(d, sysL, header, F, ss, xoff, skip_header=False):
    lines = _sysline(sysL, F)
    x0, x1 = sysL.x_left * F + xoff, sysL.x_right * F + xoff

    def X(wx):                                          # world x → 그리기 x
        return wx * F + xoff
    for ly in lines:
        # [오선 선명도 — 실사용 피드백] 맞춤 보기에서는 비트맵이 약 1/6로
        # 축소돼 두께 0.09칸(F=2에서 4px)의 선이 0.6px로 뭉개져 흐리게 보였다.
        # 0.15칸(약 8px)이면 같은 축소에서도 1px 이상으로 또렷하다.
        # [v1.10] 축소 피라미드(진한 픽셀 보존) 덕분에 굵힐 필요가 없어져 원래 굵기로 되돌린다.
        d.line([x0, ly, x1, ly], fill=STAFF, width=max(2, int(round(ss * 0.09))))
    if skip_header:
        return                                          # 음자리표/조표/박자표는 원본 조각으로 대체

    top = sysL.staff_top * F
    g4_line = lines[3]                                   # 아래에서 둘째 줄(G4)
    use_font = smufl.font_available()

    # 헤더(음자리표 → 조표 → 박자표)는 오선 '안쪽'(x0 오른쪽)에 배치한다.
    # 원본 악보도 오선 시작 후 헤더 자리를 비워두고 첫 음표를 놓으므로,
    # 안쪽에 그려도 음표와 겹치지 않는다(실측 여백 10ss 이상, 헤더는 ~5ss).
    fifths = sysL.fifths if getattr(sysL, "fifths", None) is not None else header["fifths"]
    n_acc = abs(fifths)

    # --- 음자리표 (오선 시작 직후) ---
    # [Tier2] 보표2 이상(그랜드 스태프의 아래 보표 등)은 낮은음자리표를 쓴다.
    # 실제 3개 이상 보표(예: 오르간)를 지원하게 되면 여기 규칙을 다시 봐야
    # 하지만, 현재 Tier2 범위(피아노류 2단보표)에서는 "보표1=높은음자리,
    # 보표2=낮은음자리"가 표준 관례와 일치한다.
    clef_k = getattr(sysL, "clef", "F" if getattr(sysL, "staff_no", 1) >= 2 else "G")
    is_bass = clef_k == "F"
    clef_x = x0 + ss * 1.9
    sh_ = getattr(sysL, "src_head", None) or {}
    if sh_.get("clef") is not None:
        clef_x = X(sh_["clef"])                          # [v2.30] 원본 음자리표 자리
    if use_font:
        _draw_clef_glyph(d, clef_k, clef_x, lines, ss)
    elif is_bass:
        _draw_bass_clef(d, clef_x, sysL.staff_top * F, ss)
    else:
        _draw_treble_clef(d, clef_x, sysL.staff_top * F, ss)

    # --- 조표 (음자리표 오른쪽) ---
    # [Tier2 버그 수정] 이전에는 diatonic 기준값(38=높은음자리 맨위줄)과
    # _FLAT_DIA/_SHARP_DIA(둘 다 높은음자리 절대 위치 기준)를 낮은음자리
    # 조표에도 그대로 써서, 조표(플랫/샵) 위치가 완전히 엉뚱한 줄에
    # 그려지는 문제가 있었다(실사용 검증 중 지적받음 — 원본은 낮은음자리
    # 조표가 "아래에서 2번째 줄"인데 렌더링은 "위에서 3번째 줄"이었음).
    # 표준 악보 관례: 낮은음자리 조표는 높은음자리와 정확히 2옥타브
    # (diatonic 14칸) 아래에 그려진다. 이 보정을 반영한다.
    # 원본 악보의 여백감에 맞춰 음자리표–조표–박자표 사이를 넉넉히 띄운다.
    acc_gap = ss * 0.72                                  # 조표 기호 사이 간격
    kx = clef_x + ss * 2.6                               # 음자리표 → 조표
    key_gap = None
    if sh_.get("key") is not None:
        half_ = ss * 0.5                                 # 조표 기호 반폭(♯·♭ 약 1칸)
        kx = X(sh_["key"]) + half_                       # [v2.30] 원본 조표 상자 안에 고르게
        if sh_.get("key_w") and n_acc >= 2:
            key_gap = (sh_["key_w"] * F - 2 * half_) / (n_acc - 1)
            # [v3.20 조바꿈] 기호 수가 원본과 다르면(♭1 자리에 ♯4 등) 원본 상자에 끼우면 겹치거나 흩어진다
            # → 보통 간격 범위로
            key_gap = min(max(key_gap, ss * 0.62), ss * 1.0)
    _draw_key_sig(d, kx, top, ss, fifths, clef_k, gap=key_gap)

    # --- 박자표 (조표 오른쪽 / 첫 시스템, [v3.40] 원본이 줄 첫머리에 다시 찍은 줄) ---
    sys_ts = getattr(sysL, "time_sig", None)
    if sysL.index == 0 or sys_ts:
        # 조표가 없으면(다장조) 음자리표 바로 뒤에, 있으면 마지막 기호 뒤에 띄운다.
        last_acc = kx + max(0, n_acc - 1) * (key_gap if key_gap else acc_gap)
        tx = (last_acc + ss * 1.7) if n_acc else (clef_x + ss * 2.8)
        if sh_.get("time") is not None:
            # [v2.30] 원본 박자표 가운데 — [v3.20] 조바꿈으로 조표가 길어져 닿으면 조표 뒤로
            tx = max(X(sh_["time"]), (last_acc + ss * 1.7) if n_acc else 0)
        if sys_ts and sysL.index != 0:
            _draw_time_sig_at(d, tx, lines, top, ss, sys_ts[0], sys_ts[1], sys_ts[2])
        else:
            _draw_time_sig_at(d, tx, lines, top, ss, header["beats"], header["beat_type"],
                              header.get("ts_symbol"))


def _draw_clef_glyph(d, clef_k, cx, lines, ss):
    """[v1.10] 음자리표 글리프 하나(G=둘째 줄, F=넷째 줄, C=가운데 줄 기준). lines는 렌더 좌표 5줄."""
    if clef_k == "F":
        _smufl(d, smufl.GLYPHS["fClef"], cx, lines[1], ss)
    elif clef_k == "C":
        _smufl(d, smufl.GLYPHS["cClef"], cx, lines[2], ss)
    else:
        _smufl(d, smufl.GLYPHS["gClef"], cx, lines[3], ss)


_KEY_SHIFT = {"G": 0, "F": 14, "C": 7}   # 높은음자리 기준 조표 위치에서 내릴 diatonic 칸


def _key_positions(fifths, clef_k):
    """조표 기호 [(내부 키, 높은음자리 기준 dia)] — 플랫/샵 순서대로."""
    if fifths < 0:
        return [("accidentalFlat", _FLAT_DIA[n]) for n in _FLAT_SEQ[:(-fifths)]]
    if fifths > 0:
        return [("accidentalSharp", _SHARP_DIA[n]) for n in _SHARP_SEQ[:fifths]]
    return []


def _draw_key_sig(d, kx, top, ss, fifths, clef_k, prev=None, gap=None):
    """[v1.10] 조표를 kx부터 그린다. 반환값 = 마지막 기호 다음 x.
    prev가 있고 새 조표가 다장조(0)이면, 앞 조표 자리에 제자리표를 그려 취소를 보인다.
    (그 밖의 바뀜은 새 조표만 그린다 — 요즘 출판 악보의 일반 관례)"""
    use_font = smufl.font_available()
    dia_top_ref = sl.CLEF_DIA.get(clef_k, sl.CLEF_DIA["G"])[0]
    shift = _KEY_SHIFT.get(clef_k, 0)
    acc_gap = gap if gap else ss * 0.72                 # [v2.30] 원본 조표 상자에 맞춘 간격(있으면)
    items = _key_positions(fifths, clef_k)
    if not items and prev:
        items = [("accidentalNatural", dia) for _k, dia in _key_positions(prev, clef_k)]
    for i, (gk, dia) in enumerate(items):
        fy = top + (dia_top_ref - (dia - shift)) * (ss / 2)
        x = kx + i * acc_gap
        if use_font:
            _smufl(d, smufl.GLYPHS[gk], x, fy, ss)
        elif gk == "accidentalFlat":
            _draw_flat(d, x, fy, ss)
        elif gk == "accidentalSharp":
            _draw_sharp(d, x, fy, ss)
        else:
            _draw_natural(d, x, fy, ss)
    return kx + len(items) * acc_gap


def _draw_time_sig_at(d, tx, lines, top, ss, beats, beat_type, ts_symbol=None):
    """박자표 하나를 (tx, 오선 기준) 위치에 그린다.

    [Tier2 신규] 기존에는 이 로직이 _draw_system() 안에 있어 시스템
    맨 처음에서만 호출됐다 — 시스템(줄) 중간에 박자표가 바뀌는 마디
    에도 그릴 수 있도록 별도 함수로 분리했다(render()에서 그 마디
    위치에 직접 호출).
    """
    use_font = smufl.font_available()
    if use_font and ts_symbol in ("common", "cut"):
        font = smufl.get_font(smufl.size_for_staff_space(ss))
        if font is not None:
            d.text((tx, lines[2]), smufl.GLYPHS["timeSigCommon" if ts_symbol == "common" else "timeSigCutCommon"],
                   font=font, fill=INK, anchor="mm")
            return
    if use_font:
        _sig_num(d, str(beats), tx, lines[1], ss)
        _sig_num(d, str(beat_type), tx, lines[3], ss)
    else:
        f = _font(ss * 2.0)
        d.text((tx, top + ss * 0.1), str(beats), fill=INK, font=f, anchor="mm")
        d.text((tx, top + ss * 3.0), str(beat_type), fill=INK, font=f, anchor="mm")


def _sig_num(d, s, cx, center_y, ss):
    """박자표 여러 자리 숫자를 SMuFL timeSig 글리프로 가로 배치."""
    font = smufl.get_font(smufl.size_for_staff_space(ss))
    if font is None:
        return
    chars = [smufl.GLYPHS["timeSig"][ch] for ch in s if ch.isdigit()]
    text = "".join(chars)
    d.text((cx, center_y), text, font=font, fill=INK, anchor="mm")


def _cubic_bezier_points(p0, p1, p2, p3, n=24):
    """3차 베지어 곡선을 n개 점으로 샘플링(부드러운 곡선용)."""
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        x = (mt**3 * p0[0] + 3 * mt**2 * t * p1[0] +
             3 * mt * t**2 * p2[0] + t**3 * p3[0])
        y = (mt**3 * p0[1] + 3 * mt**2 * t * p1[1] +
             3 * mt * t**2 * p2[1] + t**3 * p3[1])
        pts.append((x, y))
    return pts


def _draw_brace(img, d, x, y0, y1, ss):
    """왼쪽 중괄호({): 두 보표(그랜드 스태프)를 세로로 묶어 표시.

    [4차 재설계 — 실사용 검증] 3차 설계(anchor="mm" 중앙 정렬)로 실제
    렌더링해보니, 브레이스가 목표 범위를 위아래로 크게(최대 800px 이상)
    벗어나 그려지는 문제를 발견했다. 원인을 직접 실측 조사한 결과,
    PIL의 anchor="mm"은 "실제 잉크의 중앙"이 아니라 "폰트 메트릭
    (ascent/descent) 기준 중앙"에 맞춘다는 것을 확인했다 — brace처럼
    표준 em 박스를 크게 벗어나는 특수 글리프(SMuFL 옵티컬 글리프)에서는
    이 둘의 차이가 매우 커서(실측 826px), anchor="mm"으로 그리면 실제
    잉크 위치를 예측할 수 없다. 이제 캔버스를 넉넉히 크게 잡고
    anchor="la"(좌상단 기준)로 그린 뒤, 실제 잉크 bbox를 찾아 정밀
    크롭하는 방식으로 바꿨다 — 처음 그리는 위치가 어디든 상관없이
    "잉크가 캔버스 안에 온전히 들어있기만 하면" 정확하다.
    """
    target_h = max(4, int(y1 - y0))
    fsize = int(target_h * 1.15)
    font = smufl.get_font(fsize)
    if font is not None:
        # 여유를 넉넉히(폰트 크기의 3배 높이, 2배 너비) 잡아 절대 잘리지
        # 않게 한다 — 실측으로 확인한 최대 치우침(폰트크기의 절반 정도)
        # 보다 충분히 크다.
        canvas_w, canvas_h = fsize * 2, fsize * 3
        tmp = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        tmp_draw = ImageDraw.Draw(tmp)
        tmp_draw.text((canvas_w // 2, fsize), smufl.GLYPHS["brace"],
                      font=font, fill=(*INK, 255), anchor="la")
        bbox = tmp.getbbox()
        if bbox:
            cropped = tmp.crop(bbox)
            cw, ch = cropped.size
            scale = target_h / ch
            new_w = max(1, int(cw * scale))
            resized = cropped.resize((new_w, target_h), Image.LANCZOS)
            # 브레이스는 왼쪽으로 볼록한 모양 — 오른쪽 끝(뾰족한 첨점 쪽)이
            # 오선 시작 x(=여기 인자 x)에 맞닿도록 왼쪽으로 new_w만큼 당겨 붙인다.
            img.paste(resized, (int(x - new_w), int(y0)), resized)
            return

    # ------------------------------------------------------------ 폴백
    # SMuFL 폰트를 못 구했을 때만 쓰는 손그림 근사(사인 곡선 기반).
    import math
    n = 60
    w = max(2, int(ss * 0.24))
    bulge = max(ss * 1.8, (y1 - y0) * 0.11)

    pts = []
    for i in range(n + 1):
        t = i / n
        y = y0 + t * (y1 - y0)
        shape = math.sin(math.pi * t) ** 1.15
        xt = x - bulge * shape
        pts.append((xt, y))
    d.line(pts, fill=INK, width=w, joint="curve")

    r = w / 2
    for p in (pts[0], pts[-1]):
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=INK)
    center = pts[n // 2]
    r2 = ss * 0.15
    d.ellipse([center[0] - r2, center[1] - r2, center[0] + r2, center[1] + r2], fill=INK)


def _draw_treble_clef(d, cx, staff_top, ss):
    """높은음자리표: 등뼈(세로)+상단 고리+G4줄 나선+아래 꼬리로 구성해 알아보기 쉽게."""
    gline = staff_top + 3 * ss                         # G4 = 아래에서 둘째 줄
    w = max(2, int(ss * 0.2))
    # 등뼈(위 끝 → 아래 꼬리), 살짝 휘어짐
    backbone = [
        (cx + 0.18 * ss, staff_top - 1.5 * ss),
        (cx + 0.28 * ss, staff_top + 0.6 * ss),
        (cx + 0.22 * ss, gline - 0.1 * ss),
        (cx + 0.15 * ss, gline + 1.7 * ss),
        (cx - 0.12 * ss, gline + 2.35 * ss),
        (cx - 0.6 * ss, gline + 2.05 * ss),
    ]
    d.line(backbone, fill=INK, width=w, joint="curve")
    d.ellipse([cx - 0.72 * ss, gline + 1.9 * ss,
               cx - 0.36 * ss, gline + 2.26 * ss], fill=INK)     # 꼬리 끝 점
    # 상단 고리 → 왼쪽으로 크게 돌아 내려와 G줄을 감는 나선
    loop = [
        (cx + 0.18 * ss, staff_top - 1.5 * ss),
        (cx - 0.55 * ss, staff_top - 1.0 * ss),
        (cx - 0.9 * ss, staff_top + 0.2 * ss),
        (cx - 0.5 * ss, staff_top + 1.4 * ss),
        (cx + 0.35 * ss, gline - 1.4 * ss),
        (cx + 0.85 * ss, gline - 0.7 * ss),
        (cx + 0.72 * ss, gline + 0.2 * ss),
        (cx + 0.05 * ss, gline + 0.55 * ss),
        (cx - 0.55 * ss, gline + 0.1 * ss),
        (cx - 0.55 * ss, gline - 0.5 * ss),
        (cx - 0.02 * ss, gline - 0.55 * ss),
        (cx + 0.28 * ss, gline - 0.12 * ss),
    ]
    d.line(loop, fill=INK, width=w, joint="curve")


def _draw_bass_clef(d, cx, staff_top, ss):
    """낮은음자리표: 점(head) + F3줄을 감싸는 갈고리 + 우측 점 2개."""
    fline = staff_top + ss                              # F3 = 위에서 둘째 줄
    w = max(2, int(ss * 0.2))
    # 갈고리(왼쪽 위에서 시작해 오른쪽으로 크게 돌아 아래로)
    hook = [
        (cx - 0.15 * ss, fline - 1.6 * ss),
        (cx + 0.55 * ss, fline - 1.5 * ss),
        (cx + 0.95 * ss, fline - 0.7 * ss),
        (cx + 0.85 * ss, fline + 0.3 * ss),
        (cx + 0.35 * ss, fline + 0.75 * ss),
        (cx - 0.25 * ss, fline + 0.55 * ss),
    ]
    d.line(hook, fill=INK, width=w, joint="curve")
    d.ellipse([cx - 0.55 * ss, fline - 1.85 * ss,
               cx - 0.1 * ss, fline - 1.4 * ss], fill=INK)   # 머리(점)
    r = ss * 0.13
    d.ellipse([cx + 1.05 * ss - r, fline - 0.55 * ss - r,
               cx + 1.05 * ss + r, fline - 0.55 * ss + r], fill=INK)
    d.ellipse([cx + 1.05 * ss - r, fline + 0.15 * ss - r,
               cx + 1.05 * ss + r, fline + 0.15 * ss + r], fill=INK)


def _draw_flat(d, x, y, ss):
    w = max(2, int(ss * 0.16))
    d.line([(x, y - ss * 1.1), (x, y + ss * 0.35)], fill=INK, width=w)      # 세로 대
    d.line([(x, y - ss * 0.15), (x + ss * 0.34, y - ss * 0.45),
            (x + ss * 0.30, y + ss * 0.1), (x, y + ss * 0.35)],
           fill=INK, width=w, joint="curve")                               # 배


def _draw_sharp(d, x, y, ss):
    w = max(2, int(ss * 0.14))
    d.line([(x - ss * 0.25, y - ss * 0.55), (x - ss * 0.25, y + ss * 0.55)], fill=INK, width=w)
    d.line([(x + ss * 0.15, y - ss * 0.7), (x + ss * 0.15, y + ss * 0.4)], fill=INK, width=w)
    d.line([(x - ss * 0.45, y - ss * 0.15), (x + ss * 0.35, y - ss * 0.3)], fill=INK, width=w)
    d.line([(x - ss * 0.45, y + ss * 0.3), (x + ss * 0.35, y + ss * 0.15)], fill=INK, width=w)


def _draw_natural(d, x, y, ss):
    w = max(2, int(ss * 0.13))
    d.line([(x - ss * 0.22, y - ss * 0.65), (x - ss * 0.22, y + ss * 0.25)], fill=INK, width=w)
    d.line([(x + ss * 0.22, y - ss * 0.25), (x + ss * 0.22, y + ss * 0.65)], fill=INK, width=w)
    d.line([(x - ss * 0.22, y - ss * 0.25), (x + ss * 0.22, y - ss * 0.42)], fill=INK, width=w)
    d.line([(x - ss * 0.22, y + ss * 0.08), (x + ss * 0.22, y - 0.08 * ss)], fill=INK, width=w)


# --------------------------------------------------------------------- 요소
def _sys_of(layout, g):
    """이 글리프(g)가 실제로 속한 보표의 SystemLayout을 반환한다.

    [Tier2 버그 수정] 이전에는 시스템 번호(index)만 보고 첫 번째로
    찾은 보표를 무조건 반환했다 — 다중 보표(그랜드 스태프)에서 보표2
    (베이스 등)에 속한 음표를 그릴 때도 항상 보표1(높은음자리)의
    SystemLayout을 잘못 참조해, 기둥 방향 판정 등에 쓰이는 "가운뎃줄"
    기준선이 엉뚱한 보표 것으로 계산되는 문제가 있었다(실사용 검증
    중 발견). element.staff_no로 정확한 보표를 먼저 찾고, 실패하면
    (단선율 등 staff_no 정보가 없는 경우) 기존처럼 첫 보표로 폴백한다
    — Tier1 회귀 없음.
    """
    staff_no = getattr(g.element, "staff_no", 1)
    sid = g.measure.system
    for s in layout.systems:
        if s.index == sid and s.staff_no == staff_no:
            return s
    return next(s for s in layout.systems if s.index == sid)


def _dot(d, x, y, ss):
    r = max(1.4, ss * 0.17)
    d.ellipse([x - r, y - r, x + r, y + r], fill=INK)


def _dot_y(y, sysL, ss, F):
    """부점 세로 위치: 음표 머리가 오선 '줄 위'에 있으면 바로 윗칸으로 올린다.

    표준 기보법에서 부점은 항상 칸(space)에 놓인다. 음표가 줄에 걸쳐 있으면
    점이 그 줄에 겹쳐 가려지므로 반 칸(ss/2) 위로 올려 윗칸에 찍는다.
    """
    top = sysL.lines_y[0] * F
    half = ss / 2.0
    # 머리 중심이 top 기준 몇 번째 반칸인지(짝수=줄, 홀수=칸)
    steps = round((y - top) / half)
    if steps % 2 == 0:            # 줄 위 → 윗칸으로
        return y - half
    return y                      # 이미 칸 안 → 그대로


_ACC_TXT = {0: "", 1: "#", -1: "b"}


def _draw_element(d, g, layout, F, ss, xoff=0):
    if g.element.kind == "rest":
        _draw_rest(d, g, layout, F, ss, xoff)
    else:
        _draw_note(d, g, layout, F, ss, xoff, draw_flags=True,
                   stem_up=getattr(g, "stem_up", None))


def _head_glyph(e):
    if e.duration_base == 1:
        return smufl.GLYPHS["noteheadWhole"]
    if e.duration_base == 2:
        return smufl.GLYPHS["noteheadHalf"]
    return smufl.GLYPHS["noteheadBlack"]


@lru_cache(maxsize=32)
def _head_side_widths(glyph, px):
    """머리 글리프의 좌·우 '진짜 잉크' 반폭(px)을 각각 잰다.

    SMuFL 노트헤드 글리프는 문자 원점(anchor) 기준으로 좌우가 완전히
    대칭이 아니다(실측: Bravura 검정/2분 머리 모두 좌측 반폭이 우측보다
    1px 가량 더 크다). 예전에는 `min(좌,우)` 하나만 써서 기둥을 좌우
    동일 오프셋으로 그렸는데, 그 결과 —
      · 오른쪽에 붙는 위기둥(stem up): min 값이 우측 실제값과 비슷해
        거의 맞았지만 그래도 살짝 짧아 머리가 아주 미세하게 삐져나왔다.
      · 왼쪽에 붙는 아래기둥(stem down): min 값이 좌측 실제값보다
        1px 가까이 작아 기둥이 머리 안쪽으로 눈에 띄게 파고들었다.
    좌우를 각각 재서 기둥 방향에 맞는 쪽을 쓰면 두 증상이 모두 없어진다.
    """
    font = smufl.get_font(px)
    if font is None:
        return None, None
    size = px * 3
    img = Image.new("L", (size, size), 255)
    dd = ImageDraw.Draw(img)
    cx = cy = size // 2
    dd.text((cx, cy), glyph, font=font, fill=0, anchor="mm")
    arr = np.array(img)
    ink = arr < 128
    if not ink.any():
        return None, None
    rows = np.where(ink.any(axis=1))[0]
    right_w = max(np.where(ink[ry])[0].max() for ry in rows) - cx
    left_w = cx - min(np.where(ink[ry])[0].min() for ry in rows)
    return float(left_w), float(right_w)


def _head_half_width(glyph, px):
    """(레거시 호환) 좌우 중 작은 쪽 — 덧줄 폭 등 방향 무관 용도에 사용."""
    lw, rw = _head_side_widths(glyph, px)
    if lw is None:
        return None
    return min(lw, rw)


def _draw_note(d, g, layout, F, ss, xoff=0, beam_y=None, stem_up=None, draw_flags=True,
               draw_stem=True, stem_tip_y=None):
    sysL = _sys_of(layout, g)
    e = g.element
    x, y = g.x * F + xoff, g.y * F
    # 꾸밈음(장식음)은 본음표보다 작게 그린다(표준 관례: 약 65% 크기).
    # ss 를 축소해 두면 이 함수 안의 반경·기둥길이·꼬리 등 ss 기반 크기가
    # 전부 자동으로 따라 줄어든다(위치는 g.x/g.y 그대로라 영향 없음).
    if getattr(e, "is_grace", False):
        ss = ss * 0.65
    rx, ry = ss * 0.66, ss * 0.5
    # 가운뎃줄(B4) 기준 y: 음표 자신의 y(g.y)는 실측 오선 5줄의 평균
    # 간격(staff_space)으로 계산되는데(layout.py), 여기서 실제 스캔의
    # '원시' 3번째 줄 좌표(lines_y[2])와 비교하면 오선이 완벽히 균일하지
    # 않은 스캔(실측 다수)에서 최대 0.75px 어긋나 B4 음표의 기둥 방향이
    # 들쭉날쭉해진다 — 반드시 g.y 와 '같은 공식'으로 middle 을 구해야
    # 비교가 일관된다.
    middle = (sysL.staff_top + 2 * sysL.staff_space) * F
    use_font = smufl.font_available()

    # 기둥이 붙는 x 오프셋. 폰트를 쓸 때는 벡터용 rx(=0.66ss)가 Bravura 머리보다
    # 살짝 커서 기둥이 머리 바깥에 떠 보인다. 실제 글리프 반폭을 재서 맞춘다.
    #
    # 노트헤드 글리프는 좌우가 완전 대칭이 아니므로(윗 _head_side_widths
    # 참고) 기둥이 붙는 '쪽'(위기둥=오른쪽, 아래기둥=왼쪽)의 실측값을 따로
    # 써야 한다 — 이제 기둥 방향(up)을 여기서 먼저 정하고 그에 맞는 폭을
    # 고른다. 점(dot)은 방향과 무관하게 항상 오른쪽에 찍으므로 우측 폭을
    # 별도로 유지한다(dot_rx).
    stem_w = max(2, int(ss * 0.14))
    stem_rx = rx
    dot_rx = rx
    if stem_up is not None:
        up = stem_up
    else:
        up = y > middle
        if getattr(e, "is_grace", False):
            up = True        # [v1.44] 꾸밈음 기둥은 음높이와 상관없이 위로(표준 조판 관례 — 성부 지정이 없을 때)
    if use_font:
        lw, rwid = _head_side_widths(_head_glyph(e), smufl.size_for_staff_space(ss))
        if lw and rwid:
            # 안쪽 당김은 안티앨리어싱만 보정하면 되므로 작은 고정값(0.15px).
            stem_rx = (rwid if up else lw) - stem_w / 2.0 - 0.15
            dot_rx = rwid - stem_w / 2.0 - 0.15

    # [v2.03 덧줄 길이 기준] 머리 폭 + 양쪽 0.3칸(SMuFL 기본 0.4칸, Gould『Behind Bars』는 머리의 약 1/4 —
    # 그 사이). 예전 ±1.9×머리 반폭(전체 약 2.5칸)은 원본보다 길었다. 검정 머리면 전체 약 1.8칸.
    hw_ = rx
    if use_font:
        lw_, rw_ = _head_side_widths(_head_glyph(e), smufl.size_for_staff_space(ss))
        if lw_ and rw_:
            hw_ = (lw_ + rw_) / 2.0
    for ly in g.ledger:                                    # 덧줄
        yy = ly * F
        d.line([x - hw_ - ss * 0.3, yy, x + hw_ + ss * 0.3, yy], fill=STAFF,
               width=max(2, int(round(ss * 0.11))))

    # --- 음표 머리 ---
    if use_font:
        _smufl(d, _head_glyph(e), x, y, ss)
    else:
        filled = e.duration_base >= 4
        if filled:
            d.ellipse([x - rx, y - ry, x + rx, y + ry], fill=INK)
        else:
            d.ellipse([x - rx, y - ry, x + rx, y + ry],
                      outline=INK, width=max(2, int(ss * 0.16)))

    # --- 임시표(#,♭,♮): 마디 중간에 인식된 것이 있으면 머리 왼쪽에 그린다 ---
    acc = getattr(e, "accidental", None)
    if acc:
        ax = x - rx * 2.1
        if getattr(e, "acc_dx", None) is not None:
            # [v2.03] 원본에 찍힌 자리(머리 가운데 기준 거리) — 화음·2도에서 계산 위치가 마디선에 걸치던 문제
            ax = x + e.acc_dx * F
        # 마디선과는 0.7칸 넘게 떨어뜨린다(임시표가 마디선에 걸치지 않게)
        mb_left = _measure_left(layout, g.measure)
        if mb_left is not None:
            ax = max(ax, mb_left * F + (x - g.x * F) + ss * 0.7)
        if use_font:
            glyph = {"sharp": "accidentalSharp", "flat": "accidentalFlat",
                     "natural": "accidentalNatural"}.get(acc)
            if glyph:
                _smufl(d, smufl.GLYPHS[glyph], ax, y, ss)
        elif acc == "sharp":
            _draw_sharp(d, ax, y, ss)
        elif acc == "flat":
            _draw_flat(d, ax, y, ss)
        elif acc == "natural":
            _draw_natural(d, ax, y, ss)

    # [v2.03 아티큘레이션] 악센트·스타카토·테누토 등 — 원본에 찍힌 자리(머리 가운데 기준 거리)에 그린다
    for ar in getattr(e, "articulations", None) or []:
        if ar["kind"] == "arpeggio":                     # [v2.80] 화음 높이만큼 물결선
            _draw_arpeggio(d, x + ar["dx"] * F, y + ar.get("dy0", -ss / F) * F, y + ar.get("dy1", ss / F) * F, ss)
            continue
        cp = _ARTIC_GLYPH.get((ar["kind"], bool(ar["above"])))
        if ar["kind"] == "tremolo":                     # [v2.90] 빗금 수에 따라
            cp = _TREMOLO_GLYPH.get(int(ar.get("n", 1)))
        if cp and use_font:
            _smufl(d, cp, x + ar["dx"] * F, y + ar["dy"] * F, ss)

    if e.dotted:
        # 점은 음표 머리 오른쪽. 머리가 줄 '위'에 있으면(줄 사이 칸이 아니면)
        # 점이 오선에 겹쳐 가리므로, 바로 윗칸으로 살짝 올려 찍는다.
        _dot(d, x + dot_rx + ss * 0.5, _dot_y(y, sysL, ss, F), ss)
        if getattr(e, "double_dot", False):               # [v1.10 겹점]
            _dot(d, x + dot_rx + ss * 1.0, _dot_y(y, sysL, ss, F), ss)

    if e.duration_base == 1:                               # 온음표: 기둥 없음
        return
    if not draw_stem:                                      # 화음의 다른 구성음: 머리만(기둥은 하나)
        return

    if beam_y is not None:                                 # 빔 그룹: 방향은 그룹이 결정
        xs = x + stem_rx if up else x - stem_rx
        d.line([xs, y, xs, beam_y], fill=INK, width=stem_w)
        if getattr(e, "is_grace", False):
            _draw_grace_slash(d, xs, y, up, ss)
    else:
        n = sl.beam_count(e.duration_base)
        # 기둥 길이: 표준 3.5 staff-space. 꼬리(깃발)가 있으면 꼬리 높이만큼
        # 여유가 필요하다(Bravura 8분 꼬리 높이 ~3.3ss). 기둥이 짧으면 꼬리가
        # 음표 머리까지 내려와 겹친다. 꼬리 수가 늘수록(16·32분) 더 길게.
        stem_len = ss * 3.5
        if n >= 1:
            stem_len = ss * (3.5 + 0.35 * (n - 1))
        if up:
            xs, tip = x + stem_rx, y - stem_len
        else:
            xs, tip = x - stem_rx, y + stem_len
        if stem_tip_y is not None:                         # 화음: 끝 음을 넘어 뻗는 공통 기둥
            tip = stem_tip_y
        else:
            st_ = _src_stem_tip(e, up, layout, F)          # [v2.30] 원본 기둥 끝
            if st_ is not None:
                tip = st_
        d.line([xs, y, xs, tip], fill=INK, width=stem_w)
        if getattr(e, "is_grace", False):
            _draw_grace_slash(d, xs, y, up, ss)
        if draw_flags:
            if use_font and n >= 1:
                _smufl_flags(d, xs, tip, up, e.duration_base, ss, stem_w)
            elif n >= 1:
                _draw_flags(d, xs, tip, up, n, ss)


def _draw_grace_slash(d, xs, y, up, ss):
    """꾸밈음(장식음, acciaccatura) 표시용 기둥 위 짧은 사선."""
    w = max(2, int(ss * 0.18))
    cy = y - ss * 1.1 if up else y + ss * 1.1     # 기둥의 머리 쪽 1/3 지점 부근
    d.line([xs - ss * 0.55, cy + ss * 0.5, xs + ss * 0.55, cy - ss * 0.5],
           fill=INK, width=w)


_FLAG_GLYPHS = {
    (8, True): "flag8thUp", (8, False): "flag8thDown",
    (16, True): "flag16thUp", (16, False): "flag16thDown",
    (32, True): "flag32ndUp", (32, False): "flag32ndDown",
}


def _smufl_flags(d, xs, tip, stem_up, base, ss, stem_w=None):
    """개별 꼬리(깃발)를 SMuFL 글리프로. 16·32분의 여러 겹은 글리프가 이미 포함.

    SMuFL 깃발 글리프의 원점(origin)은 '기둥에 닿는 지점'이며 텍스트 baseline
    에 맞춰 설계돼 있다. 따라서 PIL anchor 는 좌측+baseline("ls")로 원점을
    기둥 끝(tip)에 올린다.

    가로 위치: 깃발은 기둥의 '바깥 가장자리'(위 기둥이면 오른쪽 면, 아래 기둥
    이면 왼쪽 면)에 붙는다. xs 는 기둥의 중심선이므로 기둥 굵기의 절반만큼
    보정해야 깃발이 기둥에 딱 맞물린다.
    """
    key = _FLAG_GLYPHS.get((base, stem_up))
    if key is None:
        return
    half = (stem_w / 2.0) if stem_w else 0.0
    fx = xs - half if stem_up else xs - half   # 글리프는 원점에서 오른쪽으로 그려짐
    _smufl(d, smufl.GLYPHS[key], fx, tip, ss, anchor="ls")


def _stem_tip_for_flag(tip, stem_up, ss):
    """깃발이 붙는 기둥 끝 y. 위 기둥은 그대로, 아래 기둥은 baseline 보정."""
    return tip


def _draw_flags(d, xs, tip, stem_up, n, ss):
    w = max(2, int(ss * 0.2))
    for k in range(n):
        if stem_up:
            fy = tip + k * ss * 0.6
            d.line([(xs, fy), (xs + ss * 0.55, fy + ss * 0.4),
                    (xs + ss * 0.9, fy + ss * 0.95)], fill=INK, width=w, joint="curve")
        else:
            fy = tip - k * ss * 0.6
            d.line([(xs, fy), (xs + ss * 0.55, fy - ss * 0.4),
                    (xs + ss * 0.9, fy - ss * 0.95)], fill=INK, width=w, joint="curve")


_REST_GLYPHS = {1: "restWhole", 2: "restHalf", 4: "restQuarter",
                8: "rest8th", 16: "rest16th", 32: "rest32nd"}


def _draw_rest(d, g, layout, F, ss, xoff=0):
    sysL = _sys_of(layout, g)
    e = g.element
    L = _sysline(sysL, F)
    dy = getattr(g, "rest_dy", 0.0) * F                  # [v1.10 다성부] 성부별 쉼표 높이
    if dy:
        L = [v + dy for v in L]
    x = g.x * F + xoff
    w = ss * 0.62
    base = e.duration_base

    if smufl.font_available():
        # SMuFL 쉼표는 대부분 가운뎃줄(3번째 줄)에 기준선이 오도록 설계됨.
        # 단, 온쉼표(넷째 줄 아래 매달림)·2분쉼표(가운뎃줄 위)는 세로 위치가
        # 달라 기준 y를 글리프별로 지정한다.
        glyph = smufl.GLYPHS[_REST_GLYPHS.get(base, "restQuarter")]
        if base == 1:
            ry = L[1]            # 온쉼표: 넷째 줄(위에서 둘째)에 매달림
        elif base == 2:
            ry = L[2]            # 2분쉼표: 가운뎃줄 위에 얹힘
        else:
            ry = L[2]            # 4/8/16/32분: 가운뎃줄 기준
        _smufl(d, glyph, x, ry, ss, anchor="mm")
        if e.dotted:
            _dot(d, x + w + ss * 0.4, L[2] - ss * 0.5, ss)
        if getattr(e, "double_dot", False):
            _dot(d, x + w + ss * 0.9, L[2] - ss * 0.5, ss)
        return

    # ---- 폴백: 벡터 드로잉 ----
    if base == 1:                                          # 온쉼표: 넷째 줄에 매달림
        d.rectangle([x - w, L[1], x + w, L[1] + ss * 0.5], fill=INK)
    elif base == 2:                                        # 2분쉼표: 가운뎃줄 위
        d.rectangle([x - w, L[2] - ss * 0.5, x + w, L[2]], fill=INK)
    elif base == 4:                                        # 4분쉼표: 지그재그
        tw = max(2, int(ss * 0.28))
        top = L[1] + ss * 0.1
        d.line([(x - ss * 0.28, top), (x + ss * 0.3, top + ss * 0.72),
                (x - ss * 0.2, top + ss * 1.28), (x + ss * 0.34, top + ss * 1.98)],
               fill=INK, width=tw, joint="curve")
        d.line([(x + ss * 0.02, top + ss * 1.5), (x - ss * 0.3, top + ss * 2.15)],
               fill=INK, width=tw)
    else:                                                  # 8/16/32분쉼표: 사선 + 머리 n개
        n = sl.beam_count(base)
        top = L[1] + ss * 0.15
        bot = L[3] + ss * 0.15
        d.line([(x + ss * 0.4, top), (x - ss * 0.24, bot)],
               fill=INK, width=max(2, int(ss * 0.26)))
        for k in range(n):
            hx = x + ss * 0.4 - k * ss * 0.3
            hy = top + k * ss * 0.62
            d.ellipse([hx - ss * 0.78, hy - ss * 0.04,
                       hx - ss * 0.02, hy + ss * 0.52], fill=INK)
            d.line([(hx, hy + ss * 0.28), (hx - ss * 0.82, hy - ss * 0.02)],
                   fill=INK, width=max(2, int(ss * 0.16)))
    if e.dotted:
        _dot(d, x + w + ss * 0.4, L[2] - ss * 0.5, ss)
        if getattr(e, "double_dot", False):
            _dot(d, x + w + ss * 0.9, L[2] - ss * 0.5, ss)


def _chord_anchor(notes_c, stem_up, layout):
    """화음에서 기둥을 그릴 음(기준 음). 2도 때문에 옆으로 비킨 머리는 소수이므로, 가장 많은
    머리가 모인 x열(기본 열)에서 기둥 반대쪽 끝 음을 고른다(위로 뻗으면 맨 아래 음)."""
    tol = layout.staff_space * 0.4
    xs_ = sorted(mg.x for mg in notes_c)
    normal_x = max(xs_, key=lambda xv: (sum(1 for o in xs_ if abs(o - xv) <= tol),
                                        -xv if stem_up else xv))
    normal = [mg for mg in notes_c if abs(mg.x - normal_x) <= tol]
    return max(normal, key=lambda mg: mg.y) if stem_up else min(normal, key=lambda mg: mg.y)


def _draw_group(d, gs, layout, F, ss, xoff=0):
    notes = [g for g in gs if g.element.kind == "note"]
    for g in gs:
        if g.element.kind == "rest":
            _draw_rest(d, g, layout, F, ss, xoff)
    if not notes:
        return
    sysL = _sys_of(layout, notes[0])
    # (윗 _draw_note 와 동일 이유로 g.y 와 같은 공식 사용 — 실측 오선의
    #  raw 3번째 줄이 아니라 staff_top+2*staff_space 로 middle 을 구한다.)
    middle = (sysL.staff_top + 2 * sysL.staff_space) * F
    stem_len = ss * 3.2

    # 빔 기둥 방향: 그룹 전체를 하나로 판단(표준 규보법).
    # 가운뎃줄(B4)에서 가장 먼 음표가 방향을 결정하고, 동률이면 아래로.
    # 위로(stem_up)=빔이 음표 위, 아래로=빔이 음표 아래.
    ys = [g.y * F for g in notes]
    far_above = max(ys) - middle          # 가장 낮은 음(y 큼)이 중앙보다 아래로 얼마나
    far_below = middle - min(ys)          # 가장 높은 음(y 작음)이 중앙보다 위로 얼마나
    # 동률(예: 그룹 전체가 정확히 B4)이면 아래로 — 단일 음표의 B4 관례와
    # 통일한다. 이전에는 '>=' 라 동률에서 위로 그려져, 빔그룹 전체가 B4인
    # 경우 기둥이 위로 나오는 불일치가 있었다(사용자 실측 재확인 반영).
    stem_up = far_above > far_below        # 아래쪽이 더 멀면 위로, 동률·그 외엔 아래로
    vd = [g.stem_up for g in notes if getattr(g, "stem_up", None) is not None]
    if vd:                                 # [v1.10 다성부] 위 성부는 위로, 아래 성부는 아래로
        stem_up = vd[0]

    # [화음 기둥] 화음(chord_group)마다 기준 음 하나만 기둥을 긋고, 빔은 그 기둥들을 잇는다.
    slots = {}
    for g in notes:
        cg = getattr(g.element, "chord_group", -1)
        slots.setdefault(("c", cg) if cg != -1 else ("n", id(g)), []).append(g)
    groups_ = []
    for mem in slots.values():
        anc = _chord_anchor(mem, stem_up, layout) if len(mem) > 1 else mem[0]
        groups_.append((anc, mem))
    groups_.sort(key=lambda t: t[0].x)

    # 빔의 x는 각 음표의 기둥 위치와 같아야 한다(머리 반폭 - 기둥굵기/2).
    stem_w_g = max(2, int(ss * 0.14))

    def _srx(e):
        if smufl.font_available():
            lw, rwid = _head_side_widths(_head_glyph(e), smufl.size_for_staff_space(ss))
            if lw and rwid:
                side = rwid if stem_up else lw
                return side - stem_w_g / 2.0 - 0.15
        return ss * 0.66

    xs = [anc.x * F + xoff + (_srx(anc.element) if stem_up else -_srx(anc.element))
          for anc, _m in groups_]
    # [v1.40 빔 기울기 — 실사용 요청] 예전엔 빔이 늘 수평이라 딱딱해 보였다. 표준 조판처럼
    # 첫 음과 끝 음의 높이 차이 쪽으로 기울이되(차이의 3/4, 최대 2.5칸), 가운데 음이 양 끝보다 빔
    # 쪽으로 더 튀어나오면(오목한 모양) 수평으로 둔다. 기둥은 모두 최소 길이(stem_len) 이상.
    near_y = [(min if stem_up else max)(g.y * F for g in mem) for _a, mem in groups_]
    slope = 0.0
    if len(groups_) >= 2 and xs[-1] - xs[0] > 1:
        dy = near_y[-1] - near_y[0]
        inner = near_y[1:-1]
        lo, hi = min(near_y[0], near_y[-1]), max(near_y[0], near_y[-1])
        concave = any(v < lo - 0.5 for v in inner) if stem_up else any(v > hi + 0.5 for v in inner)
        if not concave:
            dy = max(-2.5 * ss, min(2.5 * ss, dy * 0.75))   # 높이 차이의 3/4, 최대 2.5칸
            slope = dy / (xs[-1] - xs[0])
            # [v1.42 최대 기울기 — 실사용] 음 사이가 좁은 묶음은 높이 차이가 커도 지나치게 가팔라졌다
            # ("교회" 13마디 오른손). 기울기 각도를 약 11°(tan 0.2)로 제한한다.
            slope = max(-0.2, min(0.2, slope))
    if stem_up:
        base = min(y - stem_len - slope * (x - xs[0]) for x, y in zip(xs, near_y))
    else:
        base = max(y + stem_len - slope * (x - xs[0]) for x, y in zip(xs, near_y))

    src_line = _src_beam_line([anc for anc, _m in groups_], [m for _a, m in groups_], stem_up, layout, F, xoff)
    if src_line is not None:                    # [v2.30] 원본 빔 선(바깥 빔의 바깥쪽 가장자리)
        X1, Y1, X2, Y2 = src_line
        slope = (Y2 - Y1) / (X2 - X1) if abs(X2 - X1) > 1 else 0.0
        base = Y1 + slope * (xs[0] - X1)

    def beam_at(x):
        return base + slope * (x - xs[0])

    for (anc, mem), xs_i in zip(groups_, xs):
        by = beam_at(xs_i)
        for g in mem:
            _draw_note(d, g, layout, F, ss, xoff, beam_y=by, stem_up=stem_up,
                       draw_flags=False, draw_stem=(g is anc))
    notes = [anc for anc, _m in groups_]          # 아래 빔 계산은 화음 단위(기준 음)로

    counts = [sl.beam_count(g.element.duration_base) for g in notes]
    segs = sl.beam_segments(counts)
    thick = max(3, int(round(ss * 0.462)))        # [v2.32] 원본보다 얇아 보여 10% 더 굵게(0.42 → 0.462칸)
    gap = ss * 0.68
    for level, spans in segs.items():
        # 위로는 빔이 아래로 쌓이고(+), 아래로는 위로 쌓인다(-)
        off = (level - 1) * gap * (1 if stem_up else -1)
        for (i, j) in spans:
            if i == j:
                # [v2.32] 반쪽 빔: 이웃 음까지 거리의 45%(정확히 절반보다 조금 짧아야 자연스럽다)
                if i > 0:
                    a, b = xs[i] - (xs[i] - xs[i - 1]) * 0.45, xs[i]
                elif i < len(xs) - 1:
                    a, b = xs[i], xs[i] + (xs[i + 1] - xs[i]) * 0.45
                else:
                    a, b = xs[i] - ss * 0.7, xs[i]
            else:
                a, b = xs[i], xs[j]
            ya, yb = beam_at(a) + off, beam_at(b) + off
            if stem_up:                            # 기울어진 빔 = 평행사변형
                d.polygon([(a, ya), (b, yb), (b, yb + thick), (a, ya + thick)], fill=INK)
            else:
                d.polygon([(a, ya - thick), (b, yb - thick), (b, yb), (a, ya)], fill=INK)
