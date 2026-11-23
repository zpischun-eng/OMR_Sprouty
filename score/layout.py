# score_layout.py
# ---------------------------------------------------------------------------
# Score 모델 → '화면에 그릴 좌표'로 변환하는 순수 계산 모듈(tkinter 비의존).
#
# GUI(editor_app)는 여기서 나온 배치(Layout)를 받아 캔버스에 그리기만 하고,
# 클릭 판정(hit_test)도 여기서 처리한다. 그리기 코드와 좌표 계산을 분리해,
# 좌표/클릭 로직을 GUI 없이 단독으로 테스트할 수 있게 한다.
#
# 좌표계: 스태프는 높은음자리표 기준.
#   맨 아래줄 = E4, 맨 위줄 = F5. 다이아토닉 1칸 = 스태프 반 칸(staff_space/2).
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field

from . import editing as ed

_STEP_IDX = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
_DIA_TOP = 38     # F5 (높은음자리 맨 위줄)
_DIA_BOTTOM = 30  # E4 (높은음자리 맨 아래줄)
# [Tier2 버그 수정] 낮은음자리(베이스)는 오선 구성 자체가 다른데
# (맨 위줄 A3, 맨 아래줄 G2), 여태 모든 보표에 위 높은음자리 기준값을
# 그대로 썼다 — Tier1(단선율 보컬)은 항상 높은음자리만 다뤄서 드러나지
# 않았던 문제다. 실사용 검증 중 "낮은음자리 음표가 전부 일정하게 아래로
# 밀려 그려진다"는 것을 발견했고, 정확히 diatonic 12칸(높은음자리
# F5=38 vs 낮은음자리 A3=26) 차이만큼 어긋난다는 것을 계산으로 확인했다.
_DIA_TOP_BASS = 26     # A3 (낮은음자리 맨 위줄) = 3*7 + _STEP_IDX['A']
_DIA_BOTTOM_BASS = 18  # G2 (낮은음자리 맨 아래줄) = 2*7 + _STEP_IDX['G']
# [v1.10 음자리표 변경] 음자리표별 (맨 위줄, 맨 아래줄) diatonic 값. C=가온음자리(알토, 가운데 줄 C4)
CLEF_DIA = {"G": (_DIA_TOP, _DIA_BOTTOM), "F": (_DIA_TOP_BASS, _DIA_BOTTOM_BASS), "C": (32, 24)}


def diatonic(step, octave):
    return octave * 7 + _STEP_IDX[step]


@dataclass
class Glyph:
    element: object            # 모델의 Element (선택/편집 대상)
    measure: object            # 소속 Measure
    x: float                   # 중심 x
    y: float                   # 음표 머리 중심 y (쉼표는 기준 y)
    cell: tuple                # (x0, y0, x1, y1) 클릭 판정 영역
    group_id: int              # 같은 빔 그룹이면 같은 값(-1이면 단독)
    ledger: list = field(default_factory=list)   # 덧줄 y좌표들
    stem_up: object = None     # [v1.10 다성부] None=음높이로 정함, True/False=성부 규칙으로 정해진 기둥 방향
    rest_dy: float = 0.0       # [v1.10 다성부] 쉼표를 기본 자리에서 위(-)/아래(+)로 옮길 양(world)
    clef: str = None           # [v1.10] 이 요소에 적용되는 음자리표("G"/"F"/"C")


@dataclass
class MeasureBox:
    measure: object
    bbox: tuple                # (x0, y0, x1, y1) 마디 배경 영역
    status: str                # 'ok'|'over'|'under'
    barline_x: float


@dataclass
class SystemLayout:
    index: int
    staff_top: float           # 맨 위줄 y
    staff_space: float
    x_left: float              # 스태프 시작 x(헤더 뒤)
    x_right: float
    lines_y: list              # 5개 오선 y
    staff_no: int = 1           # [Tier2] 같은 시스템 안 몇 번째 보표인가(1=위)
    brace_group: int = -1       # [Tier2] 같은 값이면 왼쪽에 중괄호로 묶어 그림(-1=단독)
    clef: str = "G"             # [음자리표] 이 보표의 음자리표("G"/"F"/"C") — 음표 높이 계산과 그리기 공용
    part: int = 0               # [여러 파트] 이 보표의 논리 파트(마디선·중괄호를 파트 단위로)
    fifths: int = None          # [v1.10 조표 바뀜] 이 줄 머리에 그릴 조표(None이면 layout.header 값)
    src_head: dict = None       # [v2.30] 원본 줄 머리 기호 x(world): {"clef","key","time"}
    time_sig: tuple = None      # [v3.40] 이 줄 머리에 원본이 다시 찍은 박자표 (beats, beat_type, symbol)


@dataclass
class ArcShape:
    """렌더용 호 하나: 배치가 끝난 좌표계(world)에서의 시작·끝점과 방향."""
    x1: float
    y1: float
    x2: float
    y2: float
    above: bool                # 위로 볼록
    kind: str = "slur"         # "tie" | "slur"
    # [v2.30 원본 곡선] 있으면 원본 3차 곡선 그대로(제어점 2개, world 좌표)
    c1x: float = None
    c1y: float = None
    c2x: float = None
    c2y: float = None
    poly: list = None          # [v2.31] 원본 잉크를 따라간 점 목록(world) — 있으면 그대로


@dataclass
class EndingShape:
    """렌더용 볼타(1절/2절 등 엔딩) 브래킷 하나."""
    x1: float
    x2: float
    y: float                   # 브래킷 가로선의 y(세계좌표)
    number: int                # 표시할 번호(예: 1 → "1.")
    open_end: bool = True      # 오른쪽 끝을 닫는 훅 없이 열어 둘지


@dataclass
class HairpinShape:
    """렌더용 크레센도/디크레센도 쐐기 하나(음표 아래에 그린다)."""
    x1: float
    x2: float
    y: float                  # 쐐기 중심 y(세계좌표)
    kind: str = "cresc"       # "cresc"(<) | "decresc"(>)
    half: float = None        # 열린 쪽 반높이(world). None이면 렌더러 기본값


@dataclass
class DynamicShape:
    """렌더용 셈여림 기호 하나(pp·p·mf…). key는 내부 키, 화면 문구와 분리."""
    x: float
    y: float
    key: str


@dataclass
class Layout:
    width: float
    height: float
    staff_space: float
    systems: list = field(default_factory=list)
    glyphs: list = field(default_factory=list)
    measure_boxes: list = field(default_factory=list)
    header: dict = field(default_factory=dict)   # {fifths, beats, beat_type}
    arcs: list = field(default_factory=list)     # ArcShape 리스트
    endings: list = field(default_factory=list)  # EndingShape 리스트
    hairpins: list = field(default_factory=list)  # HairpinShape 리스트
    marks: list = field(default_factory=list)     # [v2.80] 이동·옥타브·페달 기호(world 좌표 dict)
    dynamics: list = field(default_factory=list)  # DynamicShape 리스트
    key_marks: list = field(default_factory=list)   # [v1.10] 줄 중간 조표 바뀜 {x, sid, fifths, prev}
    clef_marks: list = field(default_factory=list)  # [v1.10] 줄 중간 음자리표 변경 {x, sid, staff_no, clef}
    suggestion_glyphs: list = field(default_factory=list)  # [v1.40] 미인식 보완 '제안 음표' Glyph(.suggestion)
    # [v2.30 원본 기하] 원본 좌표 → world 배율, 그리기용 사본의 세로 이동량(page_view)
    src_sx: float = 1.0
    src_sy: float = 1.0
    y_shift: float = 0.0

    def hit_test(self, px, py):
        """화면 좌표에서 클릭된 Glyph 를 찾는다(없으면 None). 제안 음표를 먼저 본다(작은 머리 칸)."""
        for g in self.suggestion_glyphs:
            x0, y0, x1, y1 = g.cell
            if x0 <= px <= x1 and y0 <= py <= y1:
                return g
        for g in self.glyphs:
            x0, y0, x1, y1 = g.cell
            if x0 <= px <= x1 and y0 <= py <= y1:
                return g
        return None


def layout_score(score, width, staff_space=12, margin=24,
                 header_w=64, min_cell=34, gap=22):
    """Score → Layout. width(캔버스 폭)에 맞춰 각 시스템을 가로로 채운다."""
    cap = score.capacity()

    # 시스템(원본 보표 줄) 단위로 마디를 묶는다.
    systems_measures = {}
    for m in score.measures:
        systems_measures.setdefault(m.system, []).append(m)
    system_ids = sorted(systems_measures)

    ss = staff_space
    staff_h = 4 * ss
    top_pad = 3 * ss + 6         # 높은 음 덧줄 여유
    bot_pad = 3 * ss + 40        # 낮은 음 덧줄 + 라벨 2줄 여유
    row_h = top_pad + staff_h + bot_pad

    layout = Layout(width=width, height=0, staff_space=ss,
                    header={"fifths": score.fifths, "beats": score.beats,
                            "ts_symbol": getattr(score, "ts_symbol", None),
                            "time_sig_known": getattr(score, "time_sig_known", True),
                            "beat_type": score.beat_type})

    gid_counter = 0
    for si, sid in enumerate(system_ids):
        measures = systems_measures[sid]
        staff_top = margin + si * (row_h + gap) + top_pad
        x_left = margin + header_w
        x_right = width - margin
        avail = x_right - x_left

        # 마디별 요소 수에 비례해 폭 배분(최소 폭 보장)
        counts = [max(1, len(m.elements)) for m in measures]
        weights = [c * min_cell for c in counts]
        total_w = sum(weights)
        scale = avail / total_w if total_w > 0 else 1.0
        widths = [w * scale for w in weights]

        lines_y = [staff_top + i * ss for i in range(5)]
        layout.systems.append(SystemLayout(
            index=sid, staff_top=staff_top, staff_space=ss,
            x_left=x_left, x_right=x_right, lines_y=lines_y))

        mx = x_left
        for m, mw in zip(measures, widths):
            n = max(1, len(m.elements))
            cell_w = mw / n
            status = ed.measure_status(m, cap)
            layout.measure_boxes.append(MeasureBox(
                measure=m, status=status,
                bbox=(mx, staff_top - top_pad + 4, mx + mw, staff_top + staff_h + bot_pad - 4),
                barline_x=mx + mw))

            # 빔 그룹 식별(같은 bbox 2개 이상)
            group_map = {}
            for grp in ed.beam_groups(m):
                if len(grp) >= 2:
                    gid_counter += 1
                    for e in grp:
                        group_map[id(e)] = gid_counter

            for j, e in enumerate(m.elements):
                cx = mx + cell_w * (j + 0.5)
                ledger = []
                if e.kind == "note" and e.step is not None:
                    dia = diatonic(e.step, e.octave)
                    cy = staff_top + (_DIA_TOP - dia) * (ss / 2)
                    # 덧줄: 스태프 밖의 '줄' 위치(짝수 dia)마다
                    if dia > _DIA_TOP:
                        for d in range(_DIA_TOP + 2, dia + 1, 2):
                            ledger.append(staff_top + (_DIA_TOP - d) * (ss / 2))
                    elif dia < _DIA_BOTTOM:
                        for d in range(_DIA_BOTTOM - 2, dia - 1, -2):
                            ledger.append(staff_top + (_DIA_TOP - d) * (ss / 2))
                else:
                    cy = staff_top + 2 * ss   # 쉼표는 가운데줄 기준

                cell = (mx + cell_w * j, staff_top - top_pad,
                        mx + cell_w * (j + 1), staff_top + staff_h + bot_pad)
                layout.glyphs.append(Glyph(
                    element=e, measure=m, x=cx, y=cy, cell=cell,
                    group_id=group_map.get(id(e), -1), ledger=ledger))
            mx += mw

    n_sys = len(system_ids)
    layout.height = margin * 2 + n_sys * row_h + max(0, n_sys - 1) * gap
    return layout


# ---------------------------------------------------------------------------
# 빔(beam) 렌더링용 순수 계산 — 그룹 안에서 몇 번째 빔이 어디까지 이어지는지.
# ---------------------------------------------------------------------------
def beam_count(base):
    """음길이 → 빔(꼬리) 개수. 8분=1, 16분=2, 32분=3, 그 외=0."""
    return {8: 1, 16: 2, 32: 3}.get(base, 0)


def beam_segments(counts):
    """그룹 내 각 음표의 빔 개수 리스트 → 레벨별로 이어지는 구간을 계산.

    반환: {레벨: [(시작index, 끝index), ...]}  (시작==끝이면 홑빔 stub)
    예) [1,2,1] → {1:[(0,2)], 2:[(1,1)]}   [2,2,1] → {1:[(0,2)], 2:[(0,1)]}
    """
    counts = [max(1, c) for c in counts]     # 그룹에 있으면 최소 1빔으로 이어 준다
    maxlevel = max(counts) if counts else 0
    out = {}
    n = len(counts)
    for level in range(1, maxlevel + 1):
        spans, i = [], 0
        while i < n:
            if counts[i] >= level:
                j = i
                while j + 1 < n and counts[j + 1] >= level:
                    j += 1
                spans.append((i, j))
                i = j + 1
            else:
                i += 1
        out[level] = spans
    return out


# ---------------------------------------------------------------------------
# 원본 이미지 좌표계 기준 배치 — 좌/우 화면 동기화(확대·축소·패닝)용.
#   각 요소를 '원본에서의 x 위치'와 '오선 기준 음높이 y'에 배치한다.
#   → 우측 렌더가 좌측 원본과 같은 좌표 프레임을 공유하게 된다.
# ---------------------------------------------------------------------------
def layout_score_world(score, staves_by_system, sx, sy, barlines=None, staff_groups=None,
                       staff_left_by_line=None, staff_clefs=None, staff_parts=None):
    """원본 좌표계 Layout 생성.

    staves_by_system : [[y0..y4(정규화좌표)], ...]  시스템별 오선 5줄(보표1 기준,
        staff_groups 미지정 시 이 값만 쓰는 기존 동작 그대로 — 하위호환).
    sx, sy           : 정규화→원본 좌표 배율
    barlines         : {line_index:[x...]} 정규화 마디선(있으면 오선/마디 경계 정확히)
    staff_groups     : [Tier2] {line_index: [[y0..y4], [y0..y4], ...]} 시스템별
        보표 여러 개(그랜드 스태프 등)의 오선 좌표. 있으면 각 보표를 실제
        감지된 좌표 그대로 세로로 배치한다(보표 간 간격도 원본과 자동 일치).
        생략하면 기존처럼 시스템당 보표 1개로 동작(Tier1 회귀 없음).
    """
    barlines = barlines or {}
    staff_groups = staff_groups or {}
    staff_left_by_line = staff_left_by_line or {}
    layout = Layout(width=0, height=0, staff_space=0,
                    header={"fifths": score.fifths, "beats": score.beats,
                            "ts_symbol": getattr(score, "ts_symbol", None),
                            "time_sig_known": getattr(score, "time_sig_known", True),
                            "beat_type": score.beat_type})
    layout.src_sx, layout.src_sy = sx, sy          # [v2.30] 원본 기하 → world
    # 마디 상태(못갖춘마디 인식 포함)를 한 번에 계산해 틴트/배지에 사용
    status_map = {num: st for num, (st, _, _) in ed.validate(score).items()}
    sys_measures = {}
    for m in score.measures:
        sys_measures.setdefault(m.system, []).append(m)

    gid = 0
    space_acc = []
    for sid, measures in sorted(sys_measures.items()):
        if sid >= len(staves_by_system):
            continue

        # [Tier2] 이 시스템의 보표별 (lines, top, space) 목록을 만든다.
        # staff_groups에 이 시스템 항목이 있으면 그 전체 보표를 쓰고,
        # 없으면 기존처럼 보표1(staves_by_system[sid])만 쓴다.
        groups_norm = staff_groups.get(sid) or [staves_by_system[sid]]
        staff_infos = []  # [(staff_no, lines, top, space, dia_top, dia_bottom), ...]
        clef_by_sn, part_by_sn = {}, {}
        for i, grp in enumerate(groups_norm):
            g_lines = [v * sy for v in grp]
            g_top = g_lines[0]
            g_space = (g_lines[-1] - g_lines[0]) / 4.0
            sn = i + 1
            # [음자리표 — 실사용 버그 수정] 예전엔 "보표1=높은음, 보표2 이상=낮은음"으로
            # 고정해, 노래+피아노(3보표) 악보에서 피아노 오른손(보표2)이 낮은음자리표로
            # 그려지고 음 높이도 틀렸다. Audiveris가 보표마다 인식한 음자리표를 쓴다
            # (정보가 없을 때만 예전 규칙). render.py도 같은 SystemLayout.clef를 쓴다.
            clefs_here = (staff_clefs or {}).get(sid) or []
            clef_i = clefs_here[i] if i < len(clefs_here) else ("F" if sn >= 2 else "G")
            clef_by_sn[sn] = clef_i
            parts_here = (staff_parts or {}).get(sid) or []
            part_by_sn[sn] = parts_here[i] if i < len(parts_here) else 0
            if clef_i == "F":
                d_top, d_bot = _DIA_TOP_BASS, _DIA_BOTTOM_BASS
            else:
                d_top, d_bot = _DIA_TOP, _DIA_BOTTOM
            staff_infos.append((sn, g_lines, g_top, g_space, d_top, d_bot))
        # 기존 코드 호환용 대표값(보표1 기준) — 아래 마디 상자·헤더 폭 계산 등
        # 여러 곳에서 "이 시스템 전체"를 대표하는 하나의 오선 세트로 쓰인다.
        lines = staff_infos[0][1]
        top = staff_infos[0][2]
        space = staff_infos[0][3]
        space_acc.append(space)
        by_staff_no = {sn: (ln, tp, sp, dt, db) for sn, ln, tp, sp, dt, db in staff_infos}

        xe_all = [e.src_bbox[0] + e.src_bbox[2] for m in measures for e in m.elements]
        bl_sys = sorted(barlines.get(sid, []))              # 이 시스템 마디선(정규화)
        # x_left: 원본 이미지의 실제 좌표를 그대로 쓴다(더 이상 압축하지 않음).
        # 예전엔 첫 요소가 '렌더 헤더 폭' 바로 뒤에 오도록 전체 콘텐츠를 왼쪽
        # 으로 당겼는데(shf 보정), 그러면 편집 악보의 좌표계가 원본 이미지의
        # 좌표계와 달라져 두 패널을 같은 배율로 나란히 놓고 패닝해도 내용이
        # 어긋나 보이는 문제가 있었다(실측 지적: 우측 패널을 왼쪽 끝까지
        # 밀어도 원본처럼 여백이 남지 않음). 좌우 패널이 정확히 같은 좌표계를
        # 공유해야 팬·줌이 일치하므로, 압축을 포기하고 원본 좌표를 그대로
        # 쓴다 — 그 대신 원본 헤더 여백이 넉넉하면 편집 악보 쪽 첫 음표 앞에
        # 약간의 여유 공간이 남을 수 있지만, 좌우 정렬 일치가 더 중요하다.
        xs_first_ = min((e.src_bbox[0] for m2 in measures for e in m2.elements),
                        default=None)
        hdr_w = (3.4 + 0.95 * abs(score.fifths) + 3.0) * space
        # [Tier2 버그 수정] "첫 음표 x - 추정 헤더폭"으로 x_left를 역산하는
        # 기존 방식은, Audiveris가 이미 정확히 알려주는 실제 오선 시작
        # x(staff.left)와 최대 오선간격의 2~3배까지 어긋났다(실사용 검증
        # 중 발견 — 헤더 요소가 첫 음표와 겹쳐 보이는 문제와, 좌우 패널
        # 원점이 어긋나는 문제 둘 다의 공통 원인이었음). 실측값이 있으면
        # (Tier2, omr_bridge 경로) 그걸 그대로 쓰고, 없으면(Tier1 구엔진
        # 경로) 기존 추정 방식으로 폴백해 회귀 없이 호환한다.
        real_left = staff_left_by_line.get(sid)
        if real_left is not None:
            x_left = real_left * sx
        else:
            x_left = (max(0.0, xs_first_ * sx - hdr_w)
                     if xs_first_ is not None else 0.0)
        x_right = ((bl_sys[-1] * sx) if bl_sys
                   else ((max(xe_all) * sx) if xe_all else 0))
        # [Tier2] 보표가 여러 개(그랜드 스태프 등)면 각 보표마다 SystemLayout을
        # 하나씩 등록하고, brace_group=sid로 묶어 렌더러가 왼쪽에 중괄호를
        # 그릴 수 있게 한다. 보표가 1개면 기존과 동일(brace_group=-1).
        # [중괄호 — 실사용 버그 수정] 중괄호는 '같은 파트의 보표가 둘 이상'일 때만(피아노),
        # 노래 보표처럼 혼자인 파트는 묶지 않는다(예전엔 시스템 전체 3보표를 묶었다).
        part_count = {}
        for sn_ in part_by_sn:
            part_count[part_by_sn[sn_]] = part_count.get(part_by_sn[sn_], 0) + 1
        key_at = getattr(score, "key_at", None)
        sys_fifths = key_at(min(m_.number for m_ in measures)) if key_at else score.fifths
        m_first_ = min(measures, key=lambda m_: m_.number)
        sys_ts = getattr(m_first_, "ts_printed", None)
        if sys_ts is not None and getattr(m_first_, "time_sig_change", None):
            sys_ts = None                     # 박자가 바뀌는 줄 첫 마디는 render의 박자 바뀜 그리기가 맡는다
        for sn, s_lines, s_top, s_space, _dt, _db in staff_infos:
            pt = part_by_sn.get(sn, 0)
            layout.systems.append(SystemLayout(
                index=sid, staff_top=s_top, staff_space=s_space,
                x_left=x_left, x_right=x_right, lines_y=list(s_lines),
                staff_no=sn, brace_group=(sid * 100 + pt if part_count.get(pt, 1) > 1 else -1),
                clef=clef_by_sn.get(sn, "G"), part=pt, fifths=sys_fifths, time_sig=sys_ts,
                src_head={k: v * sx for k, v in
                          ((getattr(score, "header_x", None) or {}).get(sid, {}).get(sn, {}) or {}).items()} or None))

        # [v1.10 음자리표 변경] 보표마다 줄 머리 음자리표에서 시작해, clef_change가 붙은 요소의
        # 자리(마디 번호, x)부터 새 음자리표를 쓴다. 화음 구성음은 같은 x라 함께 바뀐다.
        def _ex_of(e):
            v = getattr(e, "src_x", None)
            if not v and e.src_bbox:
                v = e.src_bbox[0] + e.src_bbox[2] / 2
            if getattr(e, "disp_x", None) is not None:
                v = e.disp_x
            return v or 0
        clef_of = {}
        changes_by_sn = {}
        for sn_ in by_staff_no:
            els_ = [(mm.number, _ex_of(e), e) for mm in measures for e in mm.elements
                    if getattr(e, "staff_no", 1) == sn_]
            tol_ = by_staff_no[sn_][2] * 0.4 / max(sx, 1e-9)
            changes = sorted(((n_, x_, e.clef_change, e) for n_, x_, e in els_
                              if getattr(e, "clef_change", None) in CLEF_DIA), key=lambda t: (t[0], t[1]))
            changes_by_sn[sn_] = (changes, tol_)
            for n_, x_, e in els_:
                cur = clef_by_sn.get(sn_, "G")
                for cn, cx, cc, _ce in changes:
                    if cn < n_ or (cn == n_ and x_ >= cx - tol_):
                        cur = cc
                clef_of[id(e)] = cur
            prev_c = clef_by_sn.get(sn_, "G")
            for cn, cx, cc, ce in changes:
                if cc != prev_c:
                    layout.clef_marks.append({"x": cx * sx, "sid": sid, "staff_no": sn_,
                                              "clef": cc, "acc": bool(getattr(ce, "accidental", None)),
                                              "measure": cn})
                prev_c = cc

        cap = score.capacity()
        ordered = sorted(measures, key=lambda mm: mm.number)
        use_bl = len(bl_sys) == len(ordered)                # 마디선 개수가 맞으면 경계로 사용
        prev = x_left
        mbox_by_num = {}
        # [Tier2] 마디 배경 상자는 시스템의 마지막 보표 아래까지 포괄한다
        # (보표 1개면 기존과 동일 — bottom_lines가 곧 lines).
        bottom_lines = staff_infos[-1][1]
        for mi, m in enumerate(ordered):
            g_start = len(layout.glyphs)
            xs = [e.src_bbox[0] for e in m.elements]
            xe = [e.src_bbox[0] + e.src_bbox[2] for e in m.elements]
            if use_bl:
                right = bl_sys[mi] * sx
                mb = (prev, top - space * 3, right, bottom_lines[-1] + space * 3)
                prev = right
            else:
                mb = ((min(xs) * sx - 6 * sx) if xs else 0, top - space * 3,
                      (max(xe) * sx + 6 * sx) if xe else 0,
                      bottom_lines[-1] + space * 3)
            layout.measure_boxes.append(MeasureBox(
                measure=m, status=status_map.get(m.number, ed_status(m, cap)),
                bbox=mb, barline_x=mb[2]))
            mbox_by_num[m.number] = mb
            # [v1.10 조표 바뀜] 줄 중간에서 바뀌면 그 마디 앞에 새 조표(줄 처음이면 머리에 그려진다)
            if getattr(m, "key_change", None) is not None and mi > 0:
                prev_f = score.key_at(m.number - 1) if key_at else score.fifths
                if prev_f != m.key_change:
                    off = space * (2.2 if getattr(m, "start_repeat", False) else 0.9)
                    layout.key_marks.append({"x": mb[0] + off, "sid": sid,
                                             "fifths": m.key_change, "prev": prev_f})

            # 빔그룹: beam_group 키로 판정(이음줄로만 붙은 음표는 다른 키 → 분리).
            bg_count = {}
            for e in m.elements:
                k = getattr(e, "beam_group", -1)
                if k is not None and k != -1:
                    bg_count[k] = bg_count.get(k, 0) + 1
            bg_gid = {}

            # 위치: 각 음표의 '실제 기둥 x'(src_x)를 쓴다.
            # bbox 기준으로 분산시키면, 이음줄로 앞 마디 음표와 한 연결요소가 된
            # 음표가 앞 마디 위치에 그려진다(모델의 마디 배정은 맞는데 그림만
            # 엉뚱한 곳에 나오는 문제). src_x 는 인식 단계에서 각 기둥의 x 이므로
            # 원본과 같은 자리에 정확히 놓인다.
            # 기둥이 없는 요소(쉼표·온음표)는 src_x 가 bbox x 이므로 머리 폭의
            # 절반을 더해 중앙에 오게 한다.
            for e in m.elements:
                # [Tier2] 이 요소가 속한 보표의 오선을 쓴다(보표1개 악보는
                # 항상 staff_no=1 → by_staff_no[1] == (lines, top, space) 그대로).
                e_lines, e_top, e_space, e_dia_top, e_dia_bottom = by_staff_no.get(
                    getattr(e, "staff_no", 1), (lines, top, space, _DIA_TOP, _DIA_BOTTOM))

                bk = getattr(e, "beam_group", -1)
                if bk is not None and bk != -1 and bg_count.get(bk, 0) >= 2:
                    if bk not in bg_gid:
                        gid += 1
                        bg_gid[bk] = gid
                    grp_id = bg_gid[bk]
                else:
                    grp_id = -1

                sxw = getattr(e, "src_x", None)
                if sxw is None or sxw == 0:
                    bx, by, bw, bh = e.src_bbox
                    sxw = bx + bw / 2
                elif e.kind == "rest" or e.duration_base == 1:
                    bx, by, bw, bh = e.src_bbox
                    sxw = bx + bw / 2          # 기둥 없는 요소는 bbox 중앙
                elif e.src_bbox and abs(sxw - (e.src_bbox[0] + e.src_bbox[2] / 2)) <= max(e.src_bbox[2], 1):
                    # [v2.42] 'src_x가 상자 왼쪽 끝과 같을 때만'이던 조건을 '머리 상자 가운데에서 머리 폭 안이면'으로 —
                    # (멀리 떨어진 값은 이웃 상자를 빌려 쓴 새 음이므로 그대로 둔다) 원본에서 넣은
                    # 음은 박 계산용으로 src_x가 기준 음 자리에 맞춰져 상자 왼쪽 끝과 달라, 그 값을 가운데로 써
                    # 머리가 반 머리 넘게 왼쪽에 그려졌다("교회" 9마디 G♯4 — 정렬해도 안 맞아 보임).
                    # 머리 자리는 원본 머리 상자(원본이 정한다)의 가운데.
                    # [v2.03 머리 가로 위치 — 실사용] Audiveris 경로의 src_x는 '머리 왼쪽 끝'인데(옛 엔진은 기둥 x),
                    # 그리기는 이 점을 머리 '가운데'로 써서 모든 머리가 반 머리만큼 왼쪽에 그려졌다(임시표가 머리에
                    # 붙고, 2도로 비킨 머리가 기둥에서 떨어져 보이던 원인). 머리 상자의 가운데를 쓴다.
                    sxw = e.src_bbox[0] + e.src_bbox[2] / 2
                if getattr(e, "disp_x", None) is not None:
                    sxw = e.disp_x             # 편집으로 재배치된 화면 위치(원본 위치는 유지)
                wx = sxw * sx

                ledger = []
                e_clef = clef_of.get(id(e))
                if e_clef in CLEF_DIA:          # [v1.10] 줄 중간에 바뀐 음자리표 반영
                    e_dia_top, e_dia_bottom = CLEF_DIA[e_clef]
                if e.kind == "note" and e.step is not None:
                    dia = diatonic(e.step, e.octave)
                    wy = e_top + (e_dia_top - dia) * (e_space / 2)
                    if dia > e_dia_top:
                        for d in range(e_dia_top + 2, dia + 1, 2):
                            ledger.append(e_top + (e_dia_top - d) * (e_space / 2))
                    elif dia < e_dia_bottom:
                        for d in range(e_dia_bottom - 2, dia - 1, -2):
                            ledger.append(e_top + (e_dia_top - d) * (e_space / 2))
                else:
                    wy = e_top + 2 * e_space
                half = e_space * 1.1
                # [Tier2] 화음 구성음(chord_group != -1)은 같은 x에 여러 음이
                # 세로로 쌓이므로, 클릭 판정을 wy 중심으로 좁혀야 서로 구분된다.
                # 단독 음(기존 Tier1)은 오선 전체를 덮는 기존 관대한 판정을
                # 그대로 유지한다(회귀 없음 — 사용자가 대충 눌러도 걸리는
                # 기존 UX를 단선율에서는 굳이 바꿀 이유가 없다).
                if getattr(e, "chord_group", -1) != -1:
                    vhalf = e_space * 0.5
                    cell = (wx - half, wy - vhalf, wx + half, wy + vhalf)
                else:
                    # [v2.24] 오선에서 3칸보다 멀리 있는 음(덧줄 많은 낮은·높은 음)도 머리를 누르면 고르도록
                    cell = (wx - half, min(e_top - e_space * 3, wy - e_space),
                            wx + half, max(e_lines[-1] + e_space * 3, wy + e_space))
                layout.glyphs.append(Glyph(element=e, measure=m, x=wx, y=wy,
                                           cell=cell, group_id=grp_id, ledger=ledger,
                                           clef=e_clef))
            _narrow_shared_cells(layout.glyphs[g_start:], space)
            _align_chord_columns(layout.glyphs[g_start:], space)
            _assign_voice_stems(layout.glyphs[g_start:], by_staff_no, sy)
            _apply_stem_sources(layout.glyphs[g_start:])
            _head_cells(layout.glyphs[g_start:], space, sx)
            # [v1.40] 이 마디의 '제안 음표'(미인식 보완) — 주황색으로 따로 그리고 먼저 클릭되게 둔다
            for sug in [q for q in (getattr(score, "suggestions", None) or []) if q["measure"] == m.number]:
                e = sug["element"]
                sn_ = getattr(e, "staff_no", 1)
                if sn_ not in by_staff_no:
                    continue
                s_lines, s_top, s_space, _dt, _db = by_staff_no[sn_]
                cl_ = clef_by_sn.get(sn_, "G")
                chg, tol_c = changes_by_sn.get(sn_, ([], 0))
                for cn, cx, cc, _ce in chg:
                    if cn < m.number or (cn == m.number and e.src_x >= cx - tol_c):
                        cl_ = cc
                d_top, d_bot = CLEF_DIA.get(cl_, CLEF_DIA["G"])
                dia = diatonic(e.step, e.octave)
                bx, by_, bw, bh = sug["blob"]["box"][0], 0, 0, 0
                wx = (sug["blob"]["box"][0] + sug["blob"]["box"][2]) / 2.0 * sx
                wy = s_top + (d_top - dia) * (s_space / 2)
                led = []
                if dia > d_top:
                    led = [s_top + (d_top - d) * (s_space / 2) for d in range(d_top + 2, dia + 1, 2)]
                elif dia < d_bot:
                    led = [s_top + (d_top - d) * (s_space / 2) for d in range(d_bot - 2, dia - 1, -2)]
                hw = s_space * 0.9
                g = Glyph(element=e, measure=m, x=wx, y=wy, cell=(wx - hw, wy - hw, wx + hw, wy + hw),
                          group_id=-1, ledger=led, clef=cl_)
                g.suggestion = sug
                layout.suggestion_glyphs.append(g)

        # --- 볼타(1절/2절 등) 브래킷: 같은 ending 번호가 연속되는 마디들을
        #     하나의 구간으로 묶어 그 위에 걸치는 브래킷 하나를 만든다.
        #     구간 뒤에 다른 마디(같은 시스템)가 이어지면 오른쪽을 훅으로 닫고,
        #     그 구간이 시스템의 마지막이면 열어 둔다(표준 조판 관례). ---
        runs = []
        cur_num, cur_start = None, None
        for m in ordered:
            num = getattr(m, "ending", None)
            if num != cur_num:
                if cur_num is not None:
                    runs.append((cur_start, prev_m, cur_num))
                cur_num, cur_start = num, m
            prev_m = m
        if cur_num is not None:
            runs.append((cur_start, prev_m, cur_num))

        for start_m, end_m, num in runs:
            x1 = mbox_by_num[start_m.number][0]
            x2 = mbox_by_num[end_m.number][2]
            if start_m.number == ordered[0].number:
                # [v2.02] 줄 첫 마디면 음자리표·조표 뒤(첫 음 조금 앞)에서 시작한다(원본 조판과 같게)
                xs_ = [g.x for g in layout.glyphs if g.measure is start_m]
                if xs_:
                    x1 = max(x1, min(xs_) - space * 1.2)
            # 오른쪽 다리: 되돌이(:|)로 끝나는 볼타만 닫고, 나머지(마지막 볼타·줄을 넘어가는 볼타)는 연다
            open_end = getattr(end_m, "end_barline", "normal") != "repeat_end"
            # [v2.54] 파트마다 하나씩(원본처럼 합창 위에도, 피아노 위에도) — 각 파트 첫 보표 위 같은 높이
            ys_src = getattr(start_m, "ending_ys", None)
            if ys_src:
                # 원본이 정한다: Audiveris가 읽은 꺾쇠마다 그 원본 높이에(합창 위·피아노 위 등)
                for y_ in ys_src:
                    layout.endings.append(EndingShape(
                        x1=x1, x2=x2, y=y_ * sy, number=num, open_end=open_end))
            else:
                part_tops = {}
                for s_ in sorted((s for s in layout.systems if s.index == start_m.system),
                                 key=lambda s: s.staff_top):
                    part_tops.setdefault(getattr(s_, "part", 0), s_.staff_top)
                for ptop in (sorted(part_tops.values())[:1] or [top]):
                    layout.endings.append(EndingShape(
                        x1=x1, x2=x2, y=ptop - space * 4, number=num, open_end=open_end))

    layout.staff_space = (sum(space_acc) / len(space_acc)) if space_acc else 10.0

    # --- 이음줄/붙임줄: 모델의 Arc(요소 참조)를 배치 좌표의 ArcShape 로 변환 ---
    # 편집으로 삭제된 요소를 참조하는 호는 글리프가 없으므로 건너뛴다(방어).
    gmap = {id(g.element): g for g in layout.glyphs}
    ss = layout.staff_space

    def _find_sys_for_arc(sid, staff_no):
        for s in layout.systems:
            if s.index == sid and s.staff_no == staff_no:
                return s
        for s in layout.systems:
            if s.index == sid:
                return s
        return None

    for arc in getattr(score, "arcs", []):
        g1 = gmap.get(id(arc.start))
        g2 = gmap.get(id(arc.stop))
        if g1 is None or g2 is None:
            continue
        # [줄 끝 붙임줄 — 실사용 요청] 시작·끝 음표가 서로 다른 시스템(줄)에
        # 있으면(줄 끝에서 다음 줄로 이어지는 붙임줄/이음줄) 표준 표기대로
        # 반쪽 호 두 개로 나눠 그린다: ① 시작 음표 → 그 줄 오른쪽 끝,
        # ② 다음 줄 첫 음표 앞(짧은 시작부) → 끝 음표. 한 개로 그리면 두 줄을
        # 가로지르는 엉뚱하게 긴 호가 된다.
        if g1.measure.system != g2.measure.system:
            staff_no = getattr(arc.start, "staff_no", 1)
            s1 = _find_sys_for_arc(g1.measure.system, staff_no)
            s2 = _find_sys_for_arc(g2.measure.system, getattr(arc.stop, "staff_no", 1))
            if s1 is None or s2 is None:
                continue
            for gg, sysx, x_a, x_b, cv in (
                    (g1, s1, g1.x, s1.x_right - ss * 0.4, getattr(arc, "src_curve_a", None)),
                    (g2, s2, max(s2.x_left + ss * 1.0, g2.x - ss * 3.0), g2.x, getattr(arc, "src_curve_b", None))):
                if cv is not None and _pristine(gg.element):
                    layout.arcs.append(_arc_from_curve(cv, sx, sy, arc.kind))   # [v2.30] 원본 반쪽 곡선
                    continue
                up = gg.y > sysx.staff_top + 2 * sysx.staff_space
                above = not up
                off = -ss * 0.9 if above else ss * 0.9
                layout.arcs.append(ArcShape(x1=x_a, y1=gg.y + off, x2=x_b, y2=gg.y + off,
                                            above=above, kind=arc.kind))
            continue
        pl = getattr(arc, "src_poly", None)
        if pl and _pristine(arc.start) and _pristine(arc.stop):
            pts = [(x * sx, y * sy) for x, y in pl]
            mid_y = pts[len(pts) // 2][1]
            layout.arcs.append(ArcShape(x1=pts[0][0], y1=pts[0][1], x2=pts[-1][0], y2=pts[-1][1],
                                        above=mid_y < (pts[0][1] + pts[-1][1]) / 2.0, kind=arc.kind, poly=pts))
            continue
        cv = getattr(arc, "src_curve", None)
        if cv is not None and _pristine(arc.start) and _pristine(arc.stop):
            layout.arcs.append(_arc_from_curve(cv, sx, sy, arc.kind))           # [v2.30] 원본 곡선 그대로
            continue
        # [방향 결정 — 실사용 검증 후 추가] 이전에는 항상 Arc.above(고정
        # 기본값 True)를 그대로 써서, 모든 이음줄/붙임줄이 예외 없이
        # 음표 위쪽에만 그려졌다. 표준 표기는 "기둥(stem) 반대편"에
        # 그리는 것 — 음이 오선 가운데보다 낮아 기둥이 위로 향하면
        # 이음줄은 아래에, 기둥이 아래로 향하면 이음줄은 위에 그린다
        # (_draw_note의 stem 방향 판정과 동일한 기준을 그대로 재사용).
        staff_no = getattr(arc.start, "staff_no", 1)
        sysL = _find_sys_for_arc(g1.measure.system, staff_no)
        if getattr(arc, "src_above", None) is not None:
            arc_above = arc.src_above            # [v2.03] 원본 곡선이 휜 방향 그대로(Audiveris SLUR_ABOVE/BELOW)
        elif g1.stem_up is not None:              # [v1.10 다성부] 성부 쪽(기둥 쪽)으로
            arc_above = g1.stem_up
        elif sysL is not None:
            mid = sysL.staff_top + 2 * sysL.staff_space
            avg_y = (g1.y + g2.y) / 2
            stem_up = avg_y > mid
            arc_above = not stem_up
        else:
            arc_above = arc.above
        off = -ss * 0.9 if arc_above else ss * 0.9      # 머리에서 위/아래로 띄움
        # [v2.31] 기둥 쪽에 그리면(다성부·사용자 뒤집기) 원본처럼 기둥 끝에 붙인다
        if g1.stem_up is not None and bool(g1.stem_up) == bool(arc_above) \
                and g1.element.duration_base != 1:
            off = -ss * 3.9 if arc_above else ss * 3.9
        layout.arcs.append(ArcShape(
            x1=g1.x, y1=g1.y + off, x2=g2.x, y2=g2.y + off,
            above=arc_above, kind=arc.kind))

    # --- 셈여림·크레센도/디크레센도 ---
    # [셈여림 — 실사용 요청] 자동 인식분은 원본에서 읽은 자리 그대로 그리고, 편집으로
    # 새로 붙인 것은 기본 자리에 둔다: 보통은 오선 아래, 바로 아래에 가사가 있는
    # 보표(성악 악보)는 가사와 겹치지 않게 오선 위에 둔다.
    def _sys_of_el(sid, staff_no):
        return _find_sys_for_arc(sid, staff_no)

    above_cache = {}

    def _place_above(sysL):
        key_ = (sysL.index, sysL.staff_no)
        if key_ not in above_cache:
            lim = sysL.lines_y[4] + ss * 3.2
            above_cache[key_] = any(
                ly.get("y") is not None and ly["y"] * sy < lim
                for m in score.measures if m.system == sysL.index
                for e in m.elements if getattr(e, "staff_no", 1) == sysL.staff_no
                for ly in (getattr(e, "lyrics", None) or {}).values())
        return above_cache[key_]

    def _default_y(sysL):
        return sysL.lines_y[0] - ss * 2.4 if _place_above(sysL) else sysL.lines_y[4] + ss * 2.6

    for m in score.measures:
        for e in m.elements:
            dyn = getattr(e, "dynamic", None)
            g = gmap.get(id(e))
            if not dyn or g is None:
                continue
            sysL = _sys_of_el(m.system, getattr(e, "staff_no", 1))
            if sysL is None:
                continue
            x = dyn["x"] * sx if dyn.get("x") is not None else g.x
            y = dyn["y"] * sy if dyn.get("y") is not None else _default_y(sysL)
            layout.dynamics.append(DynamicShape(x=x, y=y, key=dyn["key"]))

    for hp in getattr(score, "hairpins", []):
        g1 = gmap.get(id(hp.start))
        g2 = gmap.get(id(hp.stop))
        if g1 is None or g2 is None:
            continue
        if getattr(hp, "text", None):                    # [v2.90] 글자 셈여림은 쐐기를 그리지 않는다
            continue
        if getattr(hp, "x1", None) is not None:          # 자동 인식분: 원본 좌표 그대로
            layout.hairpins.append(HairpinShape(
                x1=hp.x1 * sx, x2=hp.x2 * sx, y=hp.y * sy, kind=hp.kind,
                half=(hp.h * sy / 2.0) if hp.h else None))
            continue
        s1 = _sys_of_el(g1.measure.system, getattr(hp.start, "staff_no", 1))
        s2 = _sys_of_el(g2.measure.system, getattr(hp.stop, "staff_no", 1))
        if s1 is None or s2 is None:
            continue
        if g1.measure.system == g2.measure.system:
            x1, x2 = sorted((g1.x, g2.x))
            if x2 - x1 < ss * 1.5:
                x2 = x1 + ss * 1.5
            layout.hairpins.append(HairpinShape(x1=x1, x2=x2, y=_default_y(s1), kind=hp.kind))
        else:                                             # 줄을 넘는 쐐기는 두 조각으로
            layout.hairpins.append(HairpinShape(x1=g1.x, x2=s1.x_right - ss * 0.5,
                                                y=_default_y(s1), kind=hp.kind))
            layout.hairpins.append(HairpinShape(x1=s2.x_left + ss * 1.0, x2=g2.x,
                                                y=_default_y(s2), kind=hp.kind))
    # [v3.00] 기타 코드 → world. 원본 자리(PDF)는 그대로, 모르면(Audiveris·직접 넣음) 그 보표 2.6칸 위
    try:
        from score import chords as _chd
        tops = {}
        for s_ in layout.systems:
            tops.setdefault(s_.index, []).append(s_.staff_top)
        for g in layout.glyphs:
            h = getattr(g.element, "harmony", None)
            if not h:
                continue
            txt = _chd.display(h)
            if h.get("x") is not None and h.get("y") is not None:
                layout.marks.append({"kind": "chord", "text": txt, "x": h["x"] * sx, "y": h["y"] * sy,
                                     "orig": True, "el": id(g.element)})
            else:
                st_tops = sorted(tops.get(g.measure.system, []))
                k = (getattr(g.element, "staff_no", 1) or 1) - 1
                top = st_tops[k] if 0 <= k < len(st_tops) else (st_tops[0] if st_tops else g.y)
                layout.marks.append({"kind": "chord", "text": txt, "x": g.x - space * 0.5,
                                     "y": top - space * 2.6, "orig": False, "el": id(g.element)})
    except Exception:
        pass
    # [v2.80] 이동 기호(마디)·옥타브·페달(곡) → world
    for m in score.measures:
        for nv in (getattr(m, "nav", None) or []):
            layout.marks.append({"kind": nv["kind"], "x": nv["x"] * sx, "y": nv["y"] * sy})
    for ot in (getattr(score, "ottavas", None) or []):
        layout.marks.append({"kind": ot["kind"], "x1": ot["x1"] * sx, "x2": ot["x2"] * sx, "y": ot["y"] * sy})
    for pd in (getattr(score, "pedals", None) or []):
        if pd.get("style") == "line":                  # [v2.90] 계단형 페달 선
            layout.marks.append({"kind": "ped_line", "x1": pd["x1"] * sx, "x2": pd["x2"] * sx, "y": pd["y1"] * sy,
                                 "changes": [c * sx for c in (pd.get("changes_x") or [])]})
            continue
        layout.marks.append({"kind": "pedal", "x1": pd["x1"] * sx, "y1": pd["y1"] * sy,
                             "x2": (pd["x2"] * sx) if pd.get("x2") is not None else None,
                             "y2": (pd["y2"] * sy) if pd.get("y2") is not None else None})
    for gl in (getattr(score, "glissandos", None) or []):     # [v2.90]
        layout.marks.append({"kind": "gliss", "x1": gl["x1"] * sx, "y1": gl["y1"] * sy,
                             "x2": gl["x2"] * sx, "y2": gl["y2"] * sy})
    for hp in getattr(score, "hairpins", []):                  # [v2.90] 글자 셈여림(cresc. 등)은 글자로
        if getattr(hp, "text", None):
            layout.marks.append({"kind": "text", "text": hp.text, "x": hp.tx * sx, "y": hp.ty * sy})
    return layout


def _assign_voice_stems(glyphs, by_staff_no, sy):
    """[v1.10 다성부] 한 마디 안에서, 같은 보표의 서로 다른 성부 음이 같은 자리(x)에 겹치면 그
    성부들은 '동시에 진행하는 성부'다. 평균 음높이가 가장 높은 성부는 기둥을 위로, 나머지는
    아래로 낸다(빔·잇단 숫자·이음줄도 이 방향을 따른다). 겹치는 자리가 없는 성부(예: Audiveris가
    잇단음표 안 음마다 성부 번호를 달리 매긴 경우)는 지금처럼 음높이로 정한다.
    그 성부의 쉼표는 원본에 찍힌 높이로 옮긴다(가운데 줄에서 서로 겹치지 않게)."""
    by_staff = {}
    for g in glyphs:
        by_staff.setdefault(getattr(g.element, "staff_no", 1), []).append(g)
    for sn, gs in by_staff.items():
        info = by_staff_no.get(sn)
        if info is None:
            continue
        _lines, top, space = info[0], info[1], info[2]
        tol = space * 0.8

        def _together(a, b):
            # 같은 자리(0.8칸). 단, 1도·2도는 머리가 옆으로 비켜 찍히므로 2.6칸까지 같은 자리로 본다
            # (v1.40 — "교회" 3마디 왼손: 8분음표 A2와 온음표 A2가 나란히 찍혀 두 성부를 못 알아봄)
            dx = abs(a.x - b.x)
            return dx <= tol or (abs(a.y - b.y) <= space * 0.55 and dx <= space * 2.6)
        poly = set()
        notes = [g for g in gs if g.element.kind == "note"]
        for i, a in enumerate(notes):
            for b in notes[i + 1:]:
                va, vb = a.element.voice_no, b.element.voice_no
                if va == vb or not _together(a, b):
                    continue
                cga, cgb = a.element.chord_group, b.element.chord_group
                if cga != -1 and cga == cgb:
                    continue
                poly.update((va, vb))
        if len(poly) < 2:
            continue
        mean_y = {}
        for v in poly:
            ys = [g.y for g in notes if g.element.voice_no == v]
            if ys:
                mean_y[v] = sum(ys) / len(ys)
        if len(mean_y) < 2:
            continue
        # [v1.10 실사용 보정] 방향은 '실제로 겹치는 상대 성부'와 비교해 정한다. Audiveris는 위 선율이
        # 마디 중간부터 새 성부 번호(예: 3)로 이어지기도 한다("기도의 능력" 2마디) — 이 성부를 곡 전체
        # 평균으로만 비교하면 아래로 잘못 간다. 같은 자리 음끼리 위/아래 표를 모으고, 동률이면 평균 높이.
        votes = {v: 0 for v in mean_y}
        for i, a in enumerate(notes):
            for b in notes[i + 1:]:
                va, vb = a.element.voice_no, b.element.voice_no
                if va == vb or va not in votes or vb not in votes or not _together(a, b):
                    continue
                if abs(a.y - b.y) < 1e-6:
                    continue
                hi, lo = (va, vb) if a.y < b.y else (vb, va)
                votes[hi] += 1
                votes[lo] -= 1
        upper = min(mean_y, key=lambda v: mean_y[v])
        up_of = {v: (votes[v] > 0 if votes[v] != 0 else v == upper) for v in mean_y}
        mid = top + 2 * space
        for g in gs:
            v = g.element.voice_no
            if v not in mean_y:
                continue
            up = up_of[v]
            if g.element.kind == "note":
                g.stem_up = up
            else:
                g.stem_up = up
                nat = {1: -0.75, 2: -0.25}.get(g.element.duration_base, 0.0)
                bb = g.element.src_bbox
                if bb and bb[3]:
                    cy = (bb[1] + bb[3] / 2.0) * sy
                    k = round((cy - mid) / space - nat)
                else:
                    k = -2 if up else 2
                if up and k > 0 or (not up) and k < 0:
                    k = -2 if up else 2          # 원본 위치가 반대쪽이면 관례 자리로
                g.rest_dy = max(-5, min(5, k)) * space
                g.y = mid + g.rest_dy


def ed_status(measure, capacity):
    """status_map 에 없는 마디의 방어적 폴백(정상 경로에서는 거의 안 쓰임).

    measure.target_capacity(박자 변화 인식 줄의 실효 용량)가 있으면 그것을
    우선한다 — 전역 capacity 하나만 쓰면 그 줄 전체가 항상 '부족'으로
    잘못 판정된다(ed.validate() 의 동일 버그와 같은 원인).
    """
    cap = measure.target_capacity if measure.target_capacity is not None \
        else capacity
    total = measure.quarter_sum(cap)
    if abs(total - cap) < 0.01:
        return "ok"
    return "over" if total > cap else "under"


def _head_cells(glyphs, space, sx):
    """[v2.52 클릭 범위 — 실사용] 음표는 머리를 두른 사각형 안을 눌러야만 골라지게(예전엔 오선 위아래까지
    넓어 옆 음·다른 성부가 잘못 골라졌다). 쉼표는 쉼표 기호 둘레."""
    for g in glyphs:
        e = g.element
        if e.kind == "note":
            hw = (e.src_bbox[2] * sx / 2.0) if e.src_bbox else space * 0.6
            if e.is_grace:
                hw *= 0.75
            pad = space * 0.15
            g.cell = (g.x - hw - pad, g.y - space * 0.6, g.x + hw + pad, g.y + space * 0.6)
        elif e.kind == "rest":
            hh = space * (1.6 if e.duration_base in (4, 8, 16, 32) else 0.7)
            g.cell = (g.x - space * 0.8, g.y - hh, g.x + space * 0.8, g.y + hh)


def _narrow_shared_cells(glyphs, space):
    """[v2.31] 같은 자리에 다른 성부 음이 있으면(화음이 아닌 두 성부 — "교회" 7마디 E4·B3 2분음표) 오선 전체
    높이 클릭 칸이 겹쳐 어느 음을 고르는지 알 수 없었다 → 그 음들은 머리 높이(±0.55칸)로 좁힌다."""
    notes = [g for g in glyphs if g.element.kind == "note"]
    for g in notes:
        if any(o is not g and o.element.staff_no == g.element.staff_no and abs(o.x - g.x) <= space * 1.2
               for o in notes):
            c = g.cell
            g.cell = (c[0], g.y - space * 0.55, c[2], g.y + space * 0.55)


def _pristine(e) -> bool:
    """[v2.30] 불러온 뒤 음높이·길이·자리를 바꾸지 않은 음인가 — 그렇다면 원본 기하(곡선·빔·기둥)를 쓴다.
    편집했으면 원본 모양이 더는 맞지 않으므로 계산으로 그린다."""
    o = getattr(e, "_orig", None)
    return o is not None and o == (e.step, e.octave, e.duration_base, e.src_x)


def _arc_from_curve(cv, sx, sy, kind):
    x1, y1, c1x, c1y, c2x, c2y, x2, y2 = cv
    mid = 0.125 * (y1 + y2) + 0.375 * (c1y + c2y)
    return ArcShape(x1=x1 * sx, y1=y1 * sy, x2=x2 * sx, y2=y2 * sy, above=mid < (y1 + y2) / 2.0, kind=kind,
                    c1x=c1x * sx, c1y=c1y * sy, c2x=c2x * sx, c2y=c2y * sy)


def _apply_stem_sources(glyphs):
    """[v2.25 → v2.40 원칙대로] 기둥 방향을 정하는 차례:
    ① 사용자가 뒤집은 방향(e.stem_override, Shift+X) ② 원본 그림에서 읽은 방향(e.stem_dir_img)
    ③ 다성부 규칙(_assign_voice_stems — 원본 방향을 모르는 음만) ④ 음높이 규칙(그리기 단계).
    (v2.25~2.32는 ③을 ②보다 앞에 둬, 원본에서 넣은 음 하나 때문에 성부 평균이 바뀌어 그 성부 기둥이
    통째로 뒤집혔다 — "교회" 9마디 오른손 위 성부 16분음표) 화음·빔 묶음은 한 방향이어야 하므로
    ①·②를 묶음 안 다수결로 맞춘다."""
    notes = [g for g in glyphs if g.element.kind == "note" and g.element.duration_base != 1
             and not g.element.is_grace]
    known = set()
    for g in notes:
        ov = getattr(g.element, "stem_override", None)
        dimg = getattr(g.element, "stem_dir_img", None)
        if ov in ("up", "down"):
            g.stem_up = (ov == "up")
            known.add(id(g))
        elif dimg in ("up", "down"):
            g.stem_up = (dimg == "up")
            known.add(id(g))
    groups = {}
    for g in notes:
        e = g.element
        if getattr(e, "chord_group", -1) != -1:
            groups.setdefault(("c", e.staff_no, e.chord_group), []).append(g)
        bg = getattr(e, "beam_group", -1)
        if bg is not None and bg != -1:
            groups.setdefault(("b", e.staff_no, bg), []).append(g)
    for key, mem in groups.items():
        if len(mem) < 2:
            continue
        ov = [g.element.stem_override for g in mem if getattr(g.element, "stem_override", None)]
        votes = [g.stem_up for g in mem if id(g) in known]
        if ov:
            d = ov[0] == "up"
        elif votes:
            d = sum(1 for v in votes if v) >= len(votes) - sum(1 for v in votes if v)
        else:
            continue
        for g in mem:
            g.stem_up = d


def _narrow_shared_cells(glyphs, space):
    """[v2.31] 같은 자리에 다른 성부 음이 있으면(화음이 아닌 두 성부 — "교회" 7마디 E4·B3 2분음표) 오선 전체
    높이 클릭 칸이 겹쳐 어느 음을 고르는지 알 수 없었다 → 그 음들은 머리 높이(±0.55칸)로 좁힌다."""
    notes = [g for g in glyphs if g.element.kind == "note"]
    for g in notes:
        if any(o is not g and o.element.staff_no == g.element.staff_no and abs(o.x - g.x) <= space * 1.2
               for o in notes):
            c = g.cell
            g.cell = (c[0], g.y - space * 0.55, c[2], g.y + space * 0.55)


def _pristine(e) -> bool:
    """[v2.30] 불러온 뒤 음높이·길이·자리를 바꾸지 않은 음인가 — 그렇다면 원본 기하(곡선·빔·기둥)를 쓴다.
    편집했으면 원본 모양이 더는 맞지 않으므로 계산으로 그린다."""
    o = getattr(e, "_orig", None)
    return o is not None and o == (e.step, e.octave, e.duration_base, e.src_x)


def _arc_from_curve(cv, sx, sy, kind):
    x1, y1, c1x, c1y, c2x, c2y, x2, y2 = cv
    mid = 0.125 * (y1 + y2) + 0.375 * (c1y + c2y)
    return ArcShape(x1=x1 * sx, y1=y1 * sy, x2=x2 * sx, y2=y2 * sy, above=mid < (y1 + y2) / 2.0, kind=kind,
                    c1x=c1x * sx, c1y=c1y * sy, c2x=c2x * sx, c2y=c2y * sy)


def _align_chord_columns(glyphs, space):
    """[v2.24 화음 머리 정렬 — 실사용] 화음 머리는 한 줄로 서고, 2도로 붙은 머리만 기둥 반대쪽으로 비킨다.
    · 2도 이웃이 없는데 옆으로 비킨 머리는 기본 열로 되돌린다("교회" 53마디 온음표 화음: Audiveris가
      맨 위 머리를 30px 오른쪽에 둠 → 어긋나 보였다).
    · 2도로 비킨 머리는 기본 열에서 정확히 '머리 폭 − 기둥 굵기'만큼 떨어뜨린다(기둥에 딱 붙게 —
      원본 좌표대로 두면 어떤 머리는 기둥을 침범하고 어떤 머리는 떨어져 보였다, 52마디)."""
    groups = {}
    for g in glyphs:
        e = g.element
        if e.kind == "note" and getattr(e, "chord_group", -1) != -1 and not e.is_grace:
            groups.setdefault((e.staff_no, e.chord_group), []).append(g)
    tol = space * 0.4
    for mem in groups.values():
        if len(mem) < 2:
            continue
        xs = [g.x for g in mem]
        # 머리가 가장 많이 모인 x열. 같으면 왼쪽 열(Audiveris가 잘못 둔 머리는 오른쪽으로 밀려 있었다)
        col = max(xs, key=lambda xv: (sum(1 for o in xs if abs(o - xv) <= tol), -xv))
        whole = all(g.element.duration_base == 1 for g in mem)
        head_w = space * (1.69 if whole else 1.18)
        offset = head_w if whole else head_w - space * 0.12
        for g in mem:
            dxv = g.x - col
            if abs(dxv) > 1.5 * head_w:
                continue                  # [v2.54] 머리 폭 1.5배보다 멀면 화음 머리가 아니라 다른 음(원본 자리 그대로)
            if abs(dxv) <= tol:
                if dxv:
                    g.cell = (g.cell[0] - dxv, g.cell[1], g.cell[2] - dxv, g.cell[3])
                    g.x = col
                continue
            second = any(o is not g and abs(o.y - g.y) <= space * 0.6 for o in mem)
            nx = col + (offset if dxv > 0 else -offset) if second else col
            d = nx - g.x
            g.cell = (g.cell[0] + d, g.cell[1], g.cell[2] + d, g.cell[3])
            g.x = nx


def page_view(layout, y0: float):
    """[v2.00 여러 페이지] 한 페이지만 그리기 위한 레이아웃 사본: 모든 세로 좌표를 y0만큼 위로 옮긴다
    (Element·Measure는 그대로 공유 — 선택 강조가 그대로 맞는다). 화면 클릭은 원래 레이아웃(곡 전체
    좌표)으로 하므로 이 사본은 그리기에만 쓴다."""
    import dataclasses as _dc

    def gl(g):
        c = g.cell
        ng = _dc.replace(g, y=g.y - y0, cell=(c[0], c[1] - y0, c[2], c[3] - y0),
                         ledger=[v - y0 for v in g.ledger])
        if hasattr(g, "suggestion"):
            ng.suggestion = g.suggestion
        return ng
    out = _dc.replace(layout)
    out.glyphs = [gl(g) for g in layout.glyphs]
    out.suggestion_glyphs = [gl(g) for g in getattr(layout, "suggestion_glyphs", [])]
    out.systems = [_dc.replace(s, staff_top=s.staff_top - y0, lines_y=[v - y0 for v in s.lines_y])
                   for s in layout.systems]
    out.measure_boxes = [_dc.replace(mb, bbox=(mb.bbox[0], mb.bbox[1] - y0, mb.bbox[2], mb.bbox[3] - y0))
                         for mb in layout.measure_boxes]
    out.arcs = [_dc.replace(a, y1=a.y1 - y0, y2=a.y2 - y0,
                            poly=[(px, py - y0) for px, py in a.poly] if a.poly else None,
                            c1y=(a.c1y - y0) if a.c1y is not None else None,
                            c2y=(a.c2y - y0) if a.c2y is not None else None) for a in layout.arcs]
    out.y_shift = layout.y_shift + y0
    out.endings = [_dc.replace(e, y=e.y - y0) for e in layout.endings]
    out.hairpins = [_dc.replace(h, y=h.y - y0) for h in layout.hairpins]
    out.marks = [{k: (v - y0 if k in ("y", "y1", "y2") and v is not None else v) for k, v in mk.items()}
                 for mk in getattr(layout, "marks", [])]
    out.dynamics = [_dc.replace(d, y=d.y - y0) for d in layout.dynamics]
    out.key_marks = list(getattr(layout, "key_marks", []))
    out.clef_marks = list(getattr(layout, "clef_marks", []))
    out.header_items = [dict(it, y=it["y"] - y0) for it in (getattr(layout, "header_items", None) or [])]
    return out
