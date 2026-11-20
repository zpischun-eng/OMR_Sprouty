# omr_bridge/geometry.py
# ---------------------------------------------------------------------------
# [1단계] Audiveris의 .omr 프로젝트 파일(zip 내부 sheet#N.xml)에서
# 원본 이미지 픽셀 좌표 + 신뢰도(grade)를 추출한다.
#
# 설계 근거: HYBRID_OMR_PLAN_v6.md 3장(좌우 100% 시각적 일치 핵심 아키텍처).
#
# sheet#N.xml 구조(이번 대화에서 실측 확인):
#   <sheet>
#     <scale><interline main=".."/></scale>          오선 간격(px)
#     <page>
#       <system id="1">
#         <sig><inters>                                전역 기호 목록(id 포함)
#           <head id=".." pitch=".." shape=".." grade=".." ctx-grade="..">
#             <bounds x=".." y=".." w=".." h=".."/>
#           </head>
#           <rest id=".." shape=".." .../>
#           <barline id=".." shape=".." .../>
#           <staff-barline id=".." .../>
#           <head-chord id=".."><bounds .../></head-chord>   (동시발음 그룹)
#           <rest-chord id=".."><bounds .../></rest-chord>
#         </inters>
#         <relations>
#           <relation source="<chord_id>" target="<head_or_rest_id>">
#             <containment/>                                 chord가 포함하는 멤버
#           </relation>
#         </relations></sig>
#         <part id="1">
#           <staff id="1" left=".." right="..">
#             <lines><line><point x=".." y=".."/>..</line>×5</lines>
#             <notes>id id id ...</notes>                     이 보표의 전체 head id (연주순)
#           </staff>
#           <left-barline><staff-barlines>id</staff-barlines></left-barline>
#           <measure id="1">
#             <right-barline><staff-barlines>id</staff-barlines></right-barline>
#             <voice id="1"><slots>
#               <entry><key>1</key><value chord="<chord_id>" status="BEGIN"/></entry>
#               ...
#             </slots></voice>
#           </measure>
#           ...
# ---------------------------------------------------------------------------
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path


@dataclass
class GlyphBox:
    """기호 하나(음표머리/쉼표/마디선 등)의 원본 픽셀 좌표 + 신뢰도."""
    kind: str            # "head" | "rest" | "barline" | "staff-barline"
    x: float
    y: float
    w: float
    h: float
    grade: float = 0.0
    ctx_grade: float = 0.0
    staff: int = 1
    pitch: float = None   # head 전용: 오선 기준 상대 위치(정수 반음 아님, 라인/스페이스 단위)
    shape: str = ""
    gid: str = ""         # .omr 안의 inter id(붙임줄↔음표머리 relation 대조용)


@dataclass
class VoiceSlotEntry:
    """마디 안 한 성부의 슬롯 하나 — chord(음표 또는 쉼표 그룹) 참조."""
    slot_key: int
    chord_id: str


@dataclass
class MeasureGeometry:
    measure_id: int             # 이 보표 안에서의 마디 순번(1부터, .omr 전역 번호)
    right_barline_x: float = None
    repeat: str = None          # None | "LEFT"(이 마디 시작=도돌이 시작) | "RIGHT"(이 마디 끝=도돌이 끝) | "BOTH"
    duration_frac: str = None   # <stack duration="..">: Audiveris가 실제로 센 이 마디의
                                 # 실효 박자량(온마디=1 기준 분수). 표기 박자표와 다르면
                                 # (예: "1/2") 그 줄 전체가 실질적으로 다른 박자로 인식된
                                 # 것일 가능성이 높다 — merge.py에서 sys_capacity로 반영.
    voices: dict = field(default_factory=dict)  # {voice_id(int): [VoiceSlotEntry, ...]}


@dataclass
class StaffGeometry:
    staff_id: int
    left: float
    right: float
    line_ys_left: list = field(default_factory=list)   # 5개, staff 왼쪽 끝 y
    line_ys_right: list = field(default_factory=list)  # 5개, staff 오른쪽 끝 y (기울기 보정용)
    left_barline_x: float = None
    measures: list = field(default_factory=list)  # [MeasureGeometry, ...]
    part_index: int = 0   # [여러 파트 지원] 이 보표의 논리 파트(0=첫 파트: 노래, 1=피아노 …)
    clef: str = "G"       # [음자리표] 이 보표 머리의 음자리표("G"=높은음, "F"=낮은음, "C"=가온음)


@dataclass
class SystemGeometry:
    system_id: int
    staves: list = field(default_factory=list)   # [StaffGeometry, ...] (파트 순서대로)


@dataclass
class SheetGeometry:
    interline_px: float
    systems: list = field(default_factory=list)  # [SystemGeometry, ...]
    glyphs: dict = field(default_factory=dict)    # {id(str): GlyphBox}
    chord_members: dict = field(default_factory=dict)  # {chord_id(str): [member_id, ...]}
    # [붙임줄 신뢰도 필터] .omr의 <slur> inter(붙임줄·이음줄 곡선)들:
    # [{"tie": bool, "grade": float, "x":..,"y":..,"w":..,"h":..}, ...]
    slurs: list = field(default_factory=list)
    # [셈여림 신뢰도 필터] .omr의 <dynamics> inter: [{"key","grade","x","y","w","h"}]
    dynamics: list = field(default_factory=list)
    # [쐐기 실제 좌표] .omr의 <wedge> inter: [{"kind","grade","x","y","w","h"}]
    wedges: list = field(default_factory=list)
    # [v1.32] .omr의 <articulation> inter(스타카토 등): [{"shape","grade","x","y","w","h"}]
    articulations: list = field(default_factory=list)
    # [v1.43] .omr의 <alter> inter(임시표 ♯·♭·♮): [{"shape","grade","x","y","w","h"}]
    alters: list = field(default_factory=list)
    # [v2.00] .omr의 <ending> inter(볼타 꺾쇠): [{"x1","x2","y","staff"}]
    endings: list = field(default_factory=list)
    # [v2.30 원본 기하] 머리 inter id → 그 머리의 기둥 (x, y위, y아래), 그 기둥에 붙은 빔들 [(x1,y1,x2,y2,두께)]
    stem_of_head: dict = field(default_factory=dict)
    # [v2.30] 줄 머리 기호(음자리표·조표·박자표): [{"kind","staff","x","w"}] — 원본 자리
    header_syms: list = field(default_factory=list)
    beams_of_head: dict = field(default_factory=dict)
    # [잇단음표 신뢰도 필터] 화음(chord inter id) → 그 화음에 걸린 잇단음표 숫자 기호의 신뢰도
    tuplet_grade_of_chord: dict = field(default_factory=dict)
    fermata_grade_of_chord: dict = field(default_factory=dict)   # [v1.41] {chord id: 늘임표 신뢰도}
    articulations_of_chord: dict = field(default_factory=dict)   # [v2.03] {chord id: [articulation dict]}


def _parse_bounds(el):
    b = el.find("bounds")
    if b is None:
        return None
    return (float(b.get("x")), float(b.get("y")), float(b.get("w")), float(b.get("h")))


def _parse_curve(el):
    """<curve x1 y1 ctrlx1 ctrly1 ctrlx2 ctrly2 x2 y2/> → 튜플(없으면 None)."""
    c = el.find("curve")
    if c is None:
        return None
    try:
        return tuple(float(c.get(k)) for k in ("x1", "y1", "ctrlx1", "ctrly1", "ctrlx2", "ctrly2", "x2", "y2"))
    except (TypeError, ValueError):
        return None


def load_sheet_geometry(omr_path: str, sheet_index: int = 1) -> SheetGeometry:
    """.omr(zip) 안의 sheet#{sheet_index}.xml을 파싱해 SheetGeometry로 반환."""
    with zipfile.ZipFile(omr_path) as z:
        member = f"sheet#{sheet_index}/sheet#{sheet_index}.xml"
        with z.open(member) as f:
            root = ET.fromstring(f.read())

    scale_el = root.find("scale")
    interline = float(scale_el.find("interline").get("main")) if scale_el is not None else 0.0

    geo = SheetGeometry(interline_px=interline)

    # ── 1) 전역 기호(glyph) 인덱스 + chord 소속(containment) 관계 구축 ──
    for tag in ("head", "rest", "barline", "staff-barline", "head-chord", "rest-chord"):
        for el in root.iter(tag):
            gid = el.get("id")
            if gid is None:
                continue
            bounds = _parse_bounds(el)
            if bounds is None:
                continue
            x, y, w, h = bounds
            geo.glyphs[gid] = GlyphBox(
                kind=tag, x=x, y=y, w=w, h=h,
                grade=float(el.get("grade", 0) or 0),
                ctx_grade=float(el.get("ctx-grade", 0) or 0),
                staff=int(el.get("staff", 1) or 1),
                pitch=(float(el.get("pitch")) if el.get("pitch") is not None else None),
                shape=el.get("shape", ""),
                gid=gid,
            )

    for sl_el in root.iter("slur"):
        if sl_el.get("grade") is None:      # <slurs> 목록 태그 등 곡선 자체가 아닌 것 제외
            continue
        b = _parse_bounds(sl_el)
        if b is None:
            continue
        geo.slurs.append({"id": sl_el.get("id"), "left": None, "right": None,
                          "tie": sl_el.get("tie") == "true",
                          "curve": _parse_curve(sl_el),     # [v2.30] 원본 곡선(x1,y1,c1x,c1y,c2x,c2y,x2,y2)
                          "above": (sl_el.get("shape") or "").endswith("ABOVE"),   # [v2.03] 실제로 휜 방향
                          "grade": float(sl_el.get("grade") or 0),
                          "x": b[0], "y": b[1], "w": b[2], "h": b[3]})

    for dy_el in root.iter("dynamics"):
        if dy_el.get("grade") is None:
            continue
        b = _parse_bounds(dy_el)
        if b is None:
            continue
        key = (dy_el.get("shape") or "").replace("DYNAMICS_", "").lower()
        geo.dynamics.append({"key": key, "grade": float(dy_el.get("grade") or 0),
                             "x": b[0], "y": b[1], "w": b[2], "h": b[3]})

    for wg_el in root.iter("wedge"):
        if wg_el.get("grade") is None:
            continue
        b = _parse_bounds(wg_el)
        if b is None:
            continue
        geo.wedges.append({"kind": "cresc" if wg_el.get("shape") == "CRESCENDO" else "decresc",
                           "grade": float(wg_el.get("grade") or 0),
                           "x": b[0], "y": b[1], "w": b[2], "h": b[3]})

    for en_el in root.iter("ending"):
        b = _parse_bounds(en_el)
        if b is not None and float(en_el.get("grade") or 0) >= 0.5:
            geo.endings.append({"x1": b[0], "x2": b[0] + b[2], "y": b[1],
                                "staff": int(en_el.get("staff") or 0)})

    art_by_id = {}
    for ar_el in root.iter("articulation"):
        b = _parse_bounds(ar_el)
        if b is not None and ar_el.get("id"):
            art_by_id[ar_el.get("id")] = {"shape": ar_el.get("shape") or "", "grade": float(ar_el.get("grade") or 0),
                                          "x": b[0], "y": b[1], "w": b[2], "h": b[3]}

    for al_el in root.iter("alter"):
        if al_el.get("grade") is None:
            continue
        b = _parse_bounds(al_el)
        if b is not None:
            geo.alters.append({"shape": al_el.get("shape"), "grade": float(al_el.get("grade") or 0),
                               "x": b[0], "y": b[1], "w": b[2], "h": b[3]})

    for ar_el in root.iter("articulation"):
        if ar_el.get("grade") is None:
            continue
        b = _parse_bounds(ar_el)
        if b is None:
            continue
        geo.articulations.append({"shape": ar_el.get("shape"), "grade": float(ar_el.get("grade") or 0),
                                  "x": b[0], "y": b[1], "w": b[2], "h": b[3]})

    fermata_grade = {}
    for fm_el in root.iter("fermata"):
        if fm_el.get("grade") is not None and fm_el.get("id"):
            fermata_grade[fm_el.get("id")] = float(fm_el.get("grade") or 0)
    for rel in root.iter("relation"):
        if rel.find("chord-articulation") is not None:
            geo._art_rel = getattr(geo, "_art_rel", []) + [(rel.get("source"), rel.get("target"))]
        if rel.find("fermata-chord") is not None and rel.get("source") in fermata_grade:
            geo.fermata_grade_of_chord[rel.get("target")] = fermata_grade[rel.get("source")]

    # 화음 ↔ 아티큘레이션 짝(관계를 다 읽은 뒤)
    for src, tgt in getattr(geo, "_art_rel", []):
        if tgt in art_by_id:
            geo.articulations_of_chord.setdefault(src, []).append(art_by_id[tgt])

    for tag, kind in (("clef", "clef"), ("key", "key"), ("time-pair", "time"), ("time-whole", "time")):
        for el in root.iter(tag):
            b = _parse_bounds(el)
            if b is None or el.get("staff") is None or el.get("grade") is None:
                continue
            geo.header_syms.append({"kind": kind, "staff": int(el.get("staff")), "x": b[0], "w": b[2]})

    # [v2.30 원본 기하] 기둥·빔: 머리 → 기둥(head-stem) → 빔(beam-stem)
    stems, beams = {}, {}
    for st_el in root.iter("stem"):
        b = _parse_bounds(st_el)
        if b is not None and st_el.get("id"):
            stems[st_el.get("id")] = (b[0] + b[2] / 2.0, b[1], b[1] + b[3])
    for bm_el in root.iter("beam"):
        med = bm_el.find("median")
        if med is None or not bm_el.get("id"):
            continue
        p1, p2 = med.find("p1"), med.find("p2")
        try:
            beams[bm_el.get("id")] = (float(p1.get("x")), float(p1.get("y")), float(p2.get("x")),
                                      float(p2.get("y")), float(bm_el.get("height") or 0))
        except (TypeError, ValueError, AttributeError):
            continue
    head_stem, stem_beams = {}, {}
    for rel in root.iter("relation"):
        src, tgt = rel.get("source"), rel.get("target")
        if rel.find("head-stem") is not None:
            head_stem[src] = tgt
        elif rel.find("beam-stem") is not None:
            stem_beams.setdefault(tgt, []).append(src)
    for hid, sid in head_stem.items():
        if sid in stems:
            geo.stem_of_head[hid] = stems[sid]
            bl = [beams[b] for b in stem_beams.get(sid, []) if b in beams]
            if bl:
                geo.beams_of_head[hid] = bl

    tuplet_grade = {}
    for tp_el in root.iter("tuplet"):
        if tp_el.get("grade") is not None and tp_el.get("id"):
            # 맥락 신뢰도(ctx-grade)를 우선 쓴다 — 실측: 진짜 0.845~0.914 / 가짜 "3" 0.736
            # (단순 grade는 진짜 0.649~0.787 / 가짜 0.494로 간격이 더 좁고 진짜 6잇단이 낮게 나옴)
            tuplet_grade[tp_el.get("id")] = float(tp_el.get("ctx-grade") or tp_el.get("grade") or 0)
    for rel in root.iter("relation"):
        if rel.find("chord-tuplet") is not None and rel.get("target") in tuplet_grade:
            geo.tuplet_grade_of_chord[rel.get("source")] = tuplet_grade[rel.get("target")]

    # 붙임줄/이음줄 곡선이 실제로 어느 음표머리(왼쪽/오른쪽)에 걸려 있는지
    slur_by_id = {sl["id"]: sl for sl in geo.slurs}
    for rel in root.iter("relation"):
        sh = rel.find("slur-head")
        if sh is not None and rel.get("source") in slur_by_id:
            side = "left" if sh.get("side") == "LEFT" else "right"
            slur_by_id[rel.get("source")][side] = rel.get("target")

    for rel in root.iter("relation"):
        if rel.find("containment") is None:
            continue
        src, tgt = rel.get("source"), rel.get("target")
        if src is None or tgt is None:
            continue
        geo.chord_members.setdefault(src, []).append(tgt)

    # ── 2) 시스템 → 파트 → 보표 → 마디 구조 ──
    # [음자리표] 보표 머리의 <clef>는 inter id만 적혀 있어, 먼저 id → 종류를 모아 둔다
    _ck = {"TREBLE": "G", "BASS": "F", "ALTO": "C", "TENOR": "C"}
    clef_kind_by_id = {}
    for c_el in root.iter("clef"):
        if c_el.get("id") and c_el.get("kind"):
            clef_kind_by_id[c_el.get("id")] = _ck.get(c_el.get("kind"), "G")
    for sys_el in root.iter("system"):
        sys_geo = SystemGeometry(system_id=int(sys_el.get("id")))
        # 이 시스템의 stack(=마디) 순서대로 repeat 속성 수집 — 위치 기준으로
        # 뒤에서 각 보표의 measure와 순서대로 짝지어진다(한 시스템 안에서
        # stack 순서와 measure 순서는 항상 좌→우로 일치한다).
        stack_repeats = [st.get("repeat") for st in sys_el.findall("stack")]
        stack_durations = [st.get("duration") for st in sys_el.findall("stack")]
        for p_idx, part_el in enumerate(sys_el.findall("part")):
            # [파트 대응 버그 수정] 시스템 안의 '순서'가 아니라 논리 파트 id로 정한다 —
            # 피아노만 있는 도입부 시스템("봄이 오면" 1줄)은 그 시스템의 첫 파트가 피아노
            # (id=2)인데, 순서로 짝지으면 노래 파트로 잘못 읽혀 음표가 전부 어긋났다.
            try:
                logical = int(part_el.get("id")) - 1
            except (TypeError, ValueError):
                logical = p_idx
            for staff_el in part_el.findall("staff"):
                staff_geo = _parse_staff(staff_el, part_el, geo, stack_repeats, stack_durations)
                staff_geo.part_index = logical
                hdr = staff_el.find("header")
                cid = hdr.findtext("clef") if hdr is not None else None
                if cid and cid.strip() in clef_kind_by_id:
                    staff_geo.clef = clef_kind_by_id[cid.strip()]
                sys_geo.staves.append(staff_geo)
        geo.systems.append(sys_geo)

    return geo


def _parse_staff(staff_el, part_el, geo: SheetGeometry, stack_repeats=None,
                 stack_durations=None) -> StaffGeometry:
    staff_id = int(staff_el.get("id"))
    left = float(staff_el.get("left"))
    right = float(staff_el.get("right"))

    line_ys_left, line_ys_right = [], []
    lines_el = staff_el.find("lines")
    if lines_el is not None:
        for line_el in lines_el.findall("line"):
            points = line_el.findall("point")
            if not points:
                continue
            line_ys_left.append(float(points[0].get("y")))
            line_ys_right.append(float(points[-1].get("y")))

    staff_geo = StaffGeometry(
        staff_id=staff_id, left=left, right=right,
        line_ys_left=line_ys_left, line_ys_right=line_ys_right,
    )

    # left-barline: part 레벨에 있고 여러 staff가 공유할 수 있으므로
    # part의 직계 자식 중 staff 태그 앞뒤가 아니라 흩어져 있을 수 있음 —
    # 여기서는 staff 바로 다음 형제 left-barline을 찾는 대신, part 전체에서
    # 첫 left-barline을 이 보표의 left_barline으로 사용(단일 보표 전제 MVP).
    lb_el = part_el.find("left-barline")
    if lb_el is not None:
        sb = lb_el.find("staff-barlines")
        if sb is not None and sb.text:
            first_id = sb.text.strip().split()[0]
            gb = geo.glyphs.get(first_id)
            if gb:
                staff_geo.left_barline_x = gb.x + gb.w / 2

    # 이 파트에 속한 <measure> 들을 순서대로 수집(보표가 1개라고 가정하는
    # Tier1 MVP에서는 모든 measure가 이 staff에 속한다고 본다).
    for i, meas_el in enumerate(part_el.findall("measure"), start=1):
        raw_id = meas_el.get("id", str(i))
        has_content = meas_el.find("voice") is not None
        try:
            mid = int(raw_id)
        except ValueError:
            if not has_content:
                # "7C" 처럼 접미사 붙은 특수 마디 중 <voice>가 없는 경우
                # (실측: 음표/성부 없이 마디선 정보만 있는 "여백/장식용"
                # 마디, 접미사 C는 cautionary로 추정) — 실제 음악 내용이
                # 없으므로 건너뛴다.
                continue
            # "X18" 처럼 접미사가 붙었지만 <voice>(실제 음표/성부)가 있는
            # 마디는 건너뛰면 데이터 손실이므로, id에서 숫자 부분만 뽑아
            # 쓴다(예: "X18" → 18). 숫자를 하나도 못 찾으면 그때만 건너뛴다
            # (이 두 경우 모두 545개 배치 실측에서 실제로 발견해 수정함).
            import re as _re
            digits = _re.sub(r"[^0-9]", "", raw_id)
            if not digits:
                continue
            mid = int(digits)
        mg = MeasureGeometry(measure_id=mid)
        if stack_repeats and i - 1 < len(stack_repeats):
            mg.repeat = stack_repeats[i - 1]
        if stack_durations and i - 1 < len(stack_durations):
            mg.duration_frac = stack_durations[i - 1]
        rb_el = meas_el.find("right-barline")
        if rb_el is not None:
            sb = rb_el.find("staff-barlines")
            if sb is not None and sb.text:
                first_id = sb.text.strip().split()[0]
                gb = geo.glyphs.get(first_id)
                if gb:
                    mg.right_barline_x = gb.x + gb.w / 2
        for voice_el in meas_el.findall("voice"):
            vid = int(voice_el.get("id"))
            entries = []
            slots_el = voice_el.find("slots")
            if slots_el is not None:
                for entry_el in slots_el.findall("entry"):
                    key = int(entry_el.findtext("key"))
                    value_el = entry_el.find("value")
                    if value_el is None:
                        continue
                    chord_id = value_el.get("chord")
                    status = value_el.get("status", "BEGIN")
                    if chord_id is None or status != "BEGIN":
                        # CONTINUE 상태는 이전 슬롯에서 이미 시작된 화음이
                        # 계속되는 표시이므로 새 음표로 세지 않는다.
                        continue
                    entries.append(VoiceSlotEntry(slot_key=key, chord_id=chord_id))
            mg.voices[vid] = entries
        staff_geo.measures.append(mg)

    return staff_geo


def chord_to_members(geo: SheetGeometry, chord_id: str):
    """chord(head-chord 또는 rest-chord) id -> 멤버 glyph(GlyphBox) 목록.

    화음(동시발음)이면 멤버가 여러 개, 단음/쉼표면 1개.
    y좌표 오름차순(위에서 아래) 정렬해 반환한다.
    """
    member_ids = geo.chord_members.get(chord_id, [])
    members = [geo.glyphs[mid] for mid in member_ids if mid in geo.glyphs]
    members.sort(key=lambda g: g.y)
    return members
