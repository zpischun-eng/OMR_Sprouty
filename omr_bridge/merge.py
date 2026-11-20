# omr_bridge/merge.py
# ---------------------------------------------------------------------------
# geometry.py(.omr 좌표) + mxl_reader.py(.mxl 음악 의미)를 결합해
# 기존 score/model.py가 기대하는 ctx 계약(elements/pitches/barlines/
# barline_types/staves)을 그대로 조립한다.
#
# 핵심 설계: 기존 omr/engine.py가 만들던 것과 "똑같은 모양"의 데이터를
# 만들어내므로, score/model.py·score/layout.py·score/render.py는
# 전혀 수정하지 않고 그대로 재사용한다(계획서 3.3절 아키텍처).
#
# elements 튜플 형식 — [Tier2] 15개 필드로 확장(앞 11개는 omr/engine.py와
# 완전히 동일해 Tier1 회귀 없음, 뒤 4개는 신규):
#   (line, x, y, w, h, beat, stem_x, beam_group, acc, is_grace, in_triplet,
#    staff_no, voice_no, chord_group, fermata)
#   - line: 0부터 시작하는 시스템(줄) 번호
#   - x,y,w,h: 이 요소(음표머리/쉼표)의 원본 좌표계 bbox
#   - beat: duration_base(1/2/4/8/16/32), 부점이면 음수
#   - stem_x: 마디 내 정렬 기준 x(기둥 위치) — head bbox 중심 x로 대체
#   - beam_group: 빔 묶음 키(-1=단독, 자동 빔 묶기는 후속 과제)
#   - acc: "sharp"|"flat"|"natural"|None
#   - is_grace, in_triplet: bool
#   - staff_no: 보표 번호(1=위, 2=아래 등)
#   - voice_no: 같은 보표 내 성부 번호
#   - chord_group: 동시발음(화음) 그룹id, -1=단독음
#   - fermata: bool
#
# pitches: elements와 같은 길이의 리스트. score/model.py의 _PITCH_NAMES
# 인덱스(정수) 또는 쉼표는 -1/None.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
from dataclasses import dataclass
from fractions import Fraction
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omr_bridge.geometry import load_sheet_geometry, chord_to_members, SheetGeometry
from omr_bridge.mxl_reader import load_score_music, ScoreMusic


# score/model.py의 _PITCH_NAMES와 정확히 동일한 순서/내용이어야 한다.
# (model.py에서 그대로 가져옴 — 이번 MVP는 참고용으로 로컬 재정의)
_PITCH_NAMES = [
    "C0", "C#0", "D0", "D#0", "E0", "F0", "F#0", "G0", "G#0", "A0", "A#0", "B0",
]
# 실제로는 score/model.py의 _PITCH_NAMES를 직접 import해서 쓰는 것이 안전
# (색인 불일치 위험 제거) — 아래 build_ctx에서 그렇게 처리한다.


@dataclass
class MergeResult:
    ctx: dict
    warnings: list


def _step_octave_to_pitch_index(step: str, octave: int, pitch_names: list) -> int:
    """'C4' 같은 이름을 score/model.py._PITCH_NAMES 색인으로 변환."""
    name = f"{step}{octave}"
    if name in pitch_names:
        return pitch_names.index(name)
    # 이름표에 없는 극단적 음역(옥타브 이탈) 등은 가장 가까운 값으로 근사.
    return None


def _group_by_chord(events):
    """[Tier2] NoteEvent 리스트를 화음 단위로 그룹핑.

    MusicXML 규약: chord 태그 없는 음표가 새 화음(또는 단독음)을 열고,
    바로 뒤에 chord 태그 있는 음표들이 그 화음에 계속 추가된다.
    반환: [[ev, ev, ...], [ev], ...] (그룹 하나 = 동시에 울리는 음표들)
    """
    groups = []
    for ev in events:
        if not ev.is_chord_member or not groups:
            groups.append([ev])
        else:
            groups[-1].append(ev)
    return groups


def _filter_real_time_changes(music):
    """[Tier2 버그 수정] MusicXML은 새 시스템(줄) 시작 마디마다 <attributes>
    (clef/key/time)를 값이 같아도 다시 선언하는 경우가 흔하다(표준 관행).
    music.time_changes는 <time> 출현 위치를 전부 담고 있으므로, "그 값이
    직전과 실제로 다른 지점"만 걸러야 진짜 박자표 변경이다 — 그동안은
    마디1만 제외하고 나머지는 전부 "변경"으로 취급해서, 값이 그대로인
    새 시스템 시작 마디에서도 박자표를 다시 그려 음자리표와 겹쳐 보이는
    문제가 있었다(실사용 검증 중 발견).
    """
    result = {}
    cur = (music.beats, music.beat_type)
    for mnum in sorted(music.time_changes):
        val = music.time_changes[mnum]
        if mnum != 1 and val != cur:
            result[mnum] = val
        cur = val
    return result


def build_ctx(omr_path: str, mxl_path: str, original_w: int, original_h: int,
             upscale_factor: float = 1.0, sheet: int = 1) -> MergeResult:
    """geometry + mxl_reader 결과를 score/model.py 호환 ctx로 조립.
    sheet: [v2.00] 여러 페이지 책(.omr)에서 몇 번째 시트인지(mxl은 그 시트만 내보낸 것)."""
    from score.model import _PITCH_NAMES as MODEL_PITCH_NAMES

    geo = load_sheet_geometry(omr_path, sheet)
    music = load_score_music(mxl_path)
    warnings = []

    # music.parts[0] 하나만 다룬다(Tier1 MVP: 단일 파트 전제).
    music_part = music.parts[0] if music.parts else None
    music_by_number = {m.number: m for m in (music_part.measures if music_part else [])}
    # [여러 파트 지원 — 실사용 버그] 노래+피아노처럼 파트가 둘 이상인 악보에서 예전엔 첫
    # 파트(노래)만 읽어 피아노 음표가 통째로 빠졌다("봄이 오면"). 파트마다 마디 표를 둔다.
    music_by_number_parts = [{m.number: m for m in p.measures} for p in music.parts] or [{}]
    TUPLET_MIN_GRADE = 0.8     # ctx-grade 기준(진짜 ≥0.845, 교가 가짜 0.736)
    # [v1.41 늘임표 신뢰도] "교회" 7마디: 가사 '랑'을 FERMATA_BELOW(0.224)로 오인 → 원본에 없는 늘임표.
    FERMATA_MIN_GRADE = 0.5
    dropped_fermatas = [0]
    dropped_tuplets = [0]

    staves_flat = []
    staff_groups = {}
    staff_clefs = {}
    staff_parts = {}  # [Tier2] {line_idx: [[y0..y4]_보표1, [y0..y4]_보표2, ...]}
    staff_left_by_line = {}  # [Tier2 버그 수정] {line_idx: 실제 감지된 오선 시작 x}
    barlines = {}
    barline_types = {}
    sys_capacity = {}
    elements = []
    pitches = []
    next_chord_gid = [0]  # 리스트로 감싸 클로저 없이도 함수 내에서 증가 가능
    # [이음줄/붙임줄 지원] 붙임줄(tie)은 Element.tie_start 필드로 직접
    # 표시(같은 성부 안 다음 음이 같은 음정이면 자동 병합, score/midi.py
    # 기존 로직 그대로 재사용). 이음줄(slur)은 좌표 기반 Score.slurs
    # 리스트로 만들어 score/model.py._attach_arcs()(기존 Tier1 인프라)가
    # 그대로 처리하게 한다 — 성부별로 "열린 이음줄의 시작 좌표"를
    # 추적하다가 짝이 되는 종료를 만나면 (line, x1, x2)로 확정한다.
    slurs = []
    open_slur_by_voice = {}  # {voice_id: (line_idx, x)}
    # [붙임줄 화면 표시 — 실사용 검증 후] 이전에는 붙임줄(<tie>)을 Element.
    # tie_start 플래그(MIDI 재생용)로만 기록하고 화면에 그릴 호(arc)는 전혀
    # 만들지 않아, 원본에 있는 붙임줄이 편집 악보에서는 통째로 사라졌다
    # (교가 7마디의 G.⌒G). 붙임줄은 "같은 성부·같은 음정"의 start→stop
    # 짝이므로 (voice, 음이름, 옥타브)로 짝을 찾아 이음줄과 같은 방식으로
    # ID 쌍을 만든다(화음이면 구성음마다 따로 짝을 지음).
    open_tie_by_key = {}     # {(voice_id, step, octave): elem_id}
    tie_pairs = []           # 신뢰도 필터를 거치기 전의 붙임줄 짝
    dyn_raw = []             # [셈여림] 신뢰도 필터 전 후보 [{elem_id,key,nx,ny}]
    harm_raw = []            # [v3.00] Audiveris가 읽은 기타 코드 [{elem_id, text}]
    wedges = []              # [쐐기] [{start_id, stop_id, kind}]
    open_wedge = {}          # {staff: (elem_id, kind)} — 쐐기는 줄을 넘어갈 수 있어 마디·줄과 무관하게 유지
    elem_pos = {}            # {elem_id: (x, y중심, line_idx)} — 붙임줄 곡선과 위치 대조용
    elem_mid = {}            # [v1.43] {elem_id: 마디 번호}
    src_geo = {}             # [v2.30] {elem_id: ((기둥 x, y위, y아래), [빔 (x1,y1,x2,y2,두께)] 또는 None)}
    art_raw = []             # [v2.03] (화음 대표 elem_id, 화음 id, 구성음 수)
    # [빔 그룹 지원] MusicXML의 <beam number="1">begin/continue/end</beam>을
    # 그대로 따라간다 — "begin"에서 새 그룹 id를 발급하고, 다음 "end"를
    # 만날 때까지 그 마디+성부 안의 같은 그룹으로 묶는다. 전역 유일 id를
    # 쓴다(chord_gid와 동일한 패턴).
    next_beam_gid = [0]
    open_beam_by_voice = {}  # {voice_id: 현재 진행 중인 beam group id}
    # [재설계] "몇 번째 요소인지" 인덱스는 model.py에서 마디별로 x좌표
    # 기준 재정렬이 일어나 안전하지 않다(정렬 후 인덱스가 달라짐) —
    # 대신 요소 자체에 전역 고유 ID를 심어(elements 튜플의 새 필드),
    # 정렬과 무관하게 항상 정확히 같은 요소를 찾을 수 있게 한다.
    next_elem_id = [0]

    line_idx = 0
    for sys_geo in geo.systems:
        if not sys_geo.staves:
            continue
        staff = sys_geo.staves[0]
        # [실사용 검증 후 버그 수정] Audiveris는 보표 번호를 시스템마다
        # 새로 1부터 매기지 않고, **문서 전체를 통틀어 계속 증가하는
        # 절대 번호**를 매긴다(예: 1번째 시스템=보표1,2 / 2번째 시스템=
        # 보표3,4). 이전 방어 코드가 "gb.staff가 1~len(sys_geo.staves)
        # 범위를 벗어나면 이상값으로 보고 1로 보정"했는데, 이 절대번호를
        # 이상값으로 잘못 판단해 2번째 시스템의 진짜 왼손 음표(staff=4)
        # 를 전부 오른손(1)으로 뭉개버리는 사고로 이어졌다(실사용 검증
        # 중 발견 — 2개 이상 시스템이 있는 악보에서만 드러남). 절대
        # staff_id를 "이 시스템 안에서 몇 번째 보표인지"(1-based)로
        # 정확히 변환하는 매핑을 만들어 그대로 쓴다.
        staff_id_to_relative = {st.staff_id: i + 1 for i, st in enumerate(sys_geo.staves)}

        def _line_ys(st):
            if len(st.line_ys_left) == 5 and len(st.line_ys_right) == 5:
                return [(l + r) / 2 for l, r in zip(st.line_ys_left, st.line_ys_right)]
            return st.line_ys_left or [0, 0, 0, 0, 0]

        # [Tier2] 이 시스템의 보표 전체(피아노 등 그랜드 스태프면 2개 이상)
        # 오선 좌표를 staff_groups에 담는다 — score/layout.py가 이걸로
        # 여러 보표를 세로로 정확히 배치한다(실제 감지된 좌표를 그대로
        # 쓰므로 보표 간 간격도 원본과 자동으로 일치한다).
        staff_groups[line_idx] = [_line_ys(st) for st in sys_geo.staves]
        staff_clefs[line_idx] = [st.clef for st in sys_geo.staves]        # [음자리표] 보표별
        staff_parts[line_idx] = [st.part_index for st in sys_geo.staves]  # [중괄호·마디선] 보표별 파트
        # [Tier2 버그 수정] 실사용 검증 중 발견: x_left를 "첫 음표 x -
        # 추정 헤더폭"으로 역산하던 기존 방식이 실제 오선 시작점과 최대
        # 오선간격의 2~3배까지 어긋났다(헤더 요소가 겹쳐 보이고, 좌우
        # 패널 원점도 어긋나는 두 문제의 공통 원인이었음). Audiveris가
        # 이미 정확히 알려주는 실제 오선 시작 x(staff.left)를 그대로
        # 전달해, layout.py가 추정 대신 이 실측값을 직접 쓰게 한다.
        staff_left_by_line[line_idx] = staff.left

        # 오선 5줄 y좌표(원본 이미지 좌표) — 기존 ctx["staves"](flat list,
        # 보표1 기준) 하위호환을 위해 그대로 유지.
        line_ys = _line_ys(staff)
        staves_flat.extend(line_ys)

        bars_this_line = []
        types_this_line = {}

        # ── 이 줄(시스템)의 실효 박자 용량 계산 ──
        # Audiveris는 <stack duration=".."> 로 각 마디에서 실제로 센 박자량을
        # (온마디=1 기준 분수로) 이미 알려준다. 표기 박자표와 다른 값이
        # 일관되게 나오면 실효 박자로 반영한다(model.py 기존 설계 의도 —
        # 마디별로 들쭉날쭉한 값은 무시하고 최빈값만 채택해 노이즈에 흔들리지
        # 않게 한다). [Tier2 보강] 단, 한 줄 안에서 박자표 자체가 바뀌는
        # 경우(mxl의 time_changes)는 그 구간별로 따로 최빈값을 구해야
        # 정확하다 — 데모곡 검증(2단보표+박자표 변화 조합) 중 발견해 보강함.
        time_bounds = sorted(music.time_changes.keys())

        def _nominal_capacity_for(measure_id):
            beats, beat_type = music.beats, music.beat_type
            for mnum in time_bounds:
                if measure_id >= mnum:
                    beats, beat_type = music.time_changes[mnum]
                else:
                    break
            return _time_sig_capacity(beats, beat_type)

        # 마디를 "표기 박자 구간"별로 묶는다(구간 = 같은 nominal_capacity가
        # 연속되는 마디들).
        segments = []  # [(nominal_cap, [mg, ...]), ...]
        for mg in staff.measures:
            nc = _nominal_capacity_for(mg.measure_id)
            if segments and segments[-1][0] == nc:
                segments[-1][1].append(mg)
            else:
                segments.append((nc, [mg]))

        measure_capacity_override = {}  # {measure_id: capacity}
        global_default_capacity = _time_sig_capacity(music.beats, music.beat_type)
        for nc, mgs in segments:
            # [Tier2 핵심 수정] 이 구간의 "표기 박자"(nc) 자체가 악보 전체의
            # 기본값(global_default_capacity, score.capacity()로 넘어가는 값)과
            # 다르면, Audiveris가 실제로 어떻게 인식했는지와 무관하게 그 구간의
            # 모든 마디를 nc로 강제해야 한다 — 그래야 3/4 구간의 마디가 4/4
            # 기본값으로 잘못 검증되지 않는다(데모곡 검증 중 발견한 버그:
            # 처음엔 "인식값이 표기박자와 같으면 오버라이드 불필요"로 판단했지만,
            # 실제로 필요한 기준은 "표기박자가 전역 기본값과 다른가"였다).
            if abs(nc - global_default_capacity) > 0.01:
                for mg in mgs:
                    measure_capacity_override[mg.measure_id] = nc

            durations = []
            for mg in mgs:
                if mg.duration_frac:
                    try:
                        durations.append(Fraction(mg.duration_frac) * 4)
                    except (ValueError, ZeroDivisionError):
                        pass
            if not durations:
                continue
            most_common_val, count = Counter(durations).most_common(1)[0]
            if count >= max(2, len(durations) * 0.6) and \
                    abs(float(most_common_val) - nc) > 0.01:
                # 표기 박자(nc)와도 다른 값이 일관되게 인식됨 — 원래 있던
                # "박자표 오독 의심" 케이스(이 구간 자체를 Audiveris가 다르게
                # 읽었을 가능성). nc 오버라이드보다 우선한다.
                cap = float(most_common_val)
                for mg in mgs:
                    measure_capacity_override[mg.measure_id] = cap
                warnings.append(
                    f"시스템 {line_idx} 마디{mgs[0].measure_id}~{mgs[-1].measure_id}: "
                    f"표기 박자({nc}) 대신 실효 박자 {cap}로 처리"
                    f"(Audiveris stack duration 최빈값 기준)"
                )

        # [여러 파트 지원] 파트마다 그 파트 첫 보표의 마디를 돈다(파트 순서 = MusicXML 파트 순서).
        part_staves, seen_parts = [], set()
        for st_ in sys_geo.staves:
            if st_.part_index not in seen_parts:
                seen_parts.add(st_.part_index)
                part_staves.append(st_)
        for pi, pstaff in enumerate(part_staves):
            # 파트는 '논리 id'로 MusicXML 파트와 짝짓는다(시스템 안 순서와 다를 수 있음)
            lp = pstaff.part_index
            music_by_number = music_by_number_parts[lp] if 0 <= lp < len(music_by_number_parts) else {}
            for mg in pstaff.measures:
                if pi == 0 and mg.right_barline_x is not None:   # 마디선은 첫 파트 기준
                    if mg.repeat == "LEFT":
                        # [Tier2 버그 수정] "이 마디에서 도돌이가 시작"이므로
                        # 그 앞 경계(=지금까지 쌓인 bars_this_line의 마지막
                        # 원소 = 직전 마디의 오른쪽 마디선 = 이 마디의 왼쪽
                        # 경계)를 repeat_start로 표시해야 build_score_model()의
                        # 판정 규칙(각 마디의 "왼쪽 경계"가 repeat_start인지로
                        # 그 마디가 도돌이로 열리는지 판정)과 맞는다. 실제
                        # Audiveris는 repeat=LEFT를 정확히 인식했는데(실측
                        # 확인) 이 반영 로직이 "후처리 예정"으로 미구현 상태로
                        # 남아있어 화면에 안 그려졌었다(실사용 검증 중 발견).
                        if bars_this_line:
                            types_this_line[bars_this_line[-1]] = "repeat_start"
                        else:
                            # [v2.51] 줄 첫 마디의 도돌이 시작: 앞 마디선이 이 줄에 없어 기록되지 않았다
                            # ("교회" 21마디 — Audiveris는 굵은+얇은 마디선·점 모두 인식, MusicXML에도
                            # forward repeat가 있었다). 줄 첫머리는 키 -1로 적는다.
                            types_this_line[-1] = "repeat_start"
                            # 원본 자리(원본이 정한다): 이 줄 보표들의 마디선 중 오선 왼쪽 끝보다 오른쪽이고
                            # 이 마디 오른쪽 마디선보다 왼쪽인 것의 가장 왼쪽(굵은 선) x
                            sids_ = {st_.staff_id for st_ in geo.systems[line_idx].staves}
                            lefts_ = [st_.left for st_ in geo.systems[line_idx].staves if st_.left]
                            x_left_ = min(lefts_) if lefts_ else 0
                            cand_ = [gb_.x for gb_ in geo.glyphs.values()
                                     if gb_.kind in ("barline", "staff-barline") and gb_.staff in sids_
                                     and x_left_ + 2 * geo.interline_px < gb_.x < mg.right_barline_x]
                            if cand_:
                                types_this_line[-2] = min(cand_)          # 굵은 선 왼쪽 끝 x
                    bars_this_line.append(mg.right_barline_x)
                    if mg.repeat == "RIGHT":
                        types_this_line[mg.right_barline_x] = "repeat_end"

                mmusic = music_by_number.get(mg.measure_id)
                if mmusic is None:
                    warnings.append(f"마디 {mg.measure_id}: mxl에 대응 마디 없음")
                    continue

                # [종지선/겹세로줄 지원] repeat_end로 이미 표시된 경우는
                # 그대로 두고(도돌이 끝이 종지선을 겸하는 경우는 드물고,
                # repeat 표시가 시각적으로 더 중요하다), 그 외에는 bar_style을
                # 반영한다. light-heavy=종지선(얇은+굵은), light-light=
                # 겹세로줄(얇은+얇은, 곡 끝이 아니라 단락 구분 등에 쓰임 —
                # 실사용 검증: Audiveris가 정확히 인식하는데도 이 타입 자체를
                # 다루는 코드가 없어 화면에 그냥 일반 마디선으로만 나왔었다).
                if pi == 0 and types_this_line.get(mg.right_barline_x) != "repeat_end":
                    if mmusic.bar_style == "light-heavy":
                        types_this_line[mg.right_barline_x] = "final"
                    elif mmusic.bar_style == "light-light":
                        types_this_line[mg.right_barline_x] = "double"

                open_beam_by_voice.clear()  # 빔은 마디를 넘어가지 않는다(표준 표기)

                for voice_id, slot_entries in mg.voices.items():
                    mxl_events = mmusic.voices.get(voice_id, [])
                    # [Tier2] mxl_events는 화음 멤버까지 전부 포함하므로, .omr의
                    # 슬롯(화음 단위) 개수와 비교하려면 먼저 화음 단위로 묶어야 한다.
                    mxl_groups = _group_by_chord(mxl_events)
                    if len(slot_entries) != len(mxl_groups):
                        warnings.append(
                            f"마디 {mg.measure_id} voice{voice_id}: "
                            f".omr 슬롯 {len(slot_entries)}개 vs .mxl 화음그룹 "
                            f"{len(mxl_groups)}개 — 개수 불일치(순서 매칭 위험)"
                        )
                    # [온마디 쉼표 — 실사용 검증] Audiveris가 MusicXML에는 '이 마디 전체 쉼표'를
                    # 적었지만 .omr에는 그 기호의 위치(슬롯)가 없는 경우가 있다("봄이 오면"
                    # 7~8마디 노래 보표). 위치는 그 마디 가운데(두 마디선 사이)로 정해 만든다.
                    if (not slot_entries and len(mxl_groups) == 1
                            and all(e_.kind == "rest" for e_ in mxl_groups[0])
                            and mg.right_barline_x is not None):
                        rev = mxl_groups[0][0]
                        x_r = mg.right_barline_x
                        lefts = [bx_ for bx_ in bars_this_line if bx_ < x_r - 1]
                        x_l = max(lefts) if lefts else pstaff.left
                        part_st = [st_ for st_ in sys_geo.staves if st_.part_index == pstaff.part_index]
                        st_i = max(0, min(len(part_st) - 1, (getattr(rev, "staff", 1) or 1) - 1))
                        rst = part_st[st_i]
                        ly_ = rst.line_ys_left or [0, 0, 0, 0, 0]
                        il_ = (ly_[-1] - ly_[0]) / 4.0 if len(ly_) >= 5 else 20.0
                        cx = (x_l + x_r) / 2.0
                        elem_id = next_elem_id[0]
                        next_elem_id[0] += 1
                        elements.append((
                            line_idx, cx - il_ * 0.6, ly_[1], il_ * 1.2, il_ * 0.5,
                            1, cx, -1, None, False, False,
                            staff_id_to_relative.get(rst.staff_id, 1), voice_id, -1, False,
                            False, elem_id, 3, 2,
                        ))
                        pitches.append(-1)
                        continue
                    for slot, group in zip(slot_entries, mxl_groups):
                        members = chord_to_members(geo, slot.chord_id)
                        if not members:
                            continue
                        # [화음 짝짓기 버그 수정 — 실사용 검증] 화음 구성음은 .omr 음표머리 목록과
                        # MusicXML 음 목록을 '순서대로' 짝짓는데, 두 목록의 정렬 방향이 서로 반대인
                        # 경우가 있었다("봄이 오면" 4마디: 맨 아래 머리에 가장 높은 음이 붙음).
                        # 화면은 음높이로 세로, 짝지은 머리로 가로 위치를 정하므로 옆으로 비킨
                        # 머리(2도 음정)가 엉뚱한 음에 붙었다(임시표도 같은 위험). 둘 다 '낮은 음부터'
                        # (머리는 아래쪽부터, 음은 음높이 오름차순) 정렬해 짝짓는다.
                        if len(members) > 1 and len(group) > 1 and all(e_.step for e_ in group):
                            members = sorted(members, key=lambda b: -(b.y + b.h / 2.0))
                            _stp = "CDEFGAB"
                            group = sorted(group, key=lambda e_: (e_.octave or 0) * 7 + _stp.index(e_.step))
                        # 화음 단위 표시(셈여림·쐐기·이음줄)는 MusicXML에서 화음의 '첫 음'에 붙어 오므로,
                        # 정렬로 대표 음이 바뀌어도 놓치지 않게 화음 전체에서 모은다.
                        g_dyn = next((e_.dynamic for e_ in group if e_.dynamic), None)
                        g_harm = next((getattr(e_, "harmony", None) for e_ in group
                                       if getattr(e_, "harmony", None)), None)        # [v3.00]
                        g_wstart = next((e_.wedge_start for e_ in group if e_.wedge_start), None)
                        g_wstop = any(e_.wedge_stop for e_ in group)
                        g_sstart = any(e_.slur_start for e_ in group)
                        g_sstop = any(e_.slur_stop for e_ in group)
                        # [잇단음표 신뢰도 필터 — 실측 근거] 가짜 "3"(교가 23마디, 맥락 신뢰도 0.736)과
                        # 진짜 잇단음표 숫자(시험 악보 0.845~0.914)가 갈린다. 이 화음에 걸린
                        # 잇단음표 기호의 신뢰도가 기준 미만이면 잇단음표로 보지 않는다.
                        tg = geo.tuplet_grade_of_chord.get(str(slot.chord_id))
                        tup_ok = group[0].in_triplet and not (tg is not None and tg < TUPLET_MIN_GRADE)
                        if group[0].in_triplet and not tup_ok:
                            dropped_tuplets[0] += 1
                        # 대표(첫 멤버) 이벤트가 이 화음 전체의 리듬(길이/점/
                        # 잇단음표/꾸밈음/늘임표)을 대표한다 — 화음 구성음들은
                        # 항상 같은 리듬으로 동시에 울리므로 이는 표기법상 항상
                        # 성립한다.
                        rep_ev = group[0]
                        fg = geo.fermata_grade_of_chord.get(str(slot.chord_id))
                        ferm_ok = rep_ev.fermata and not (fg is not None and fg < FERMATA_MIN_GRADE)
                        if rep_ev.fermata and not ferm_ok:
                            dropped_fermatas[0] += 1
                        beat = rep_ev.duration_base if not rep_ev.dotted else -rep_ev.duration_base
                        is_chord = len(group) > 1 and len(members) > 1
                        # [빔 그룹 지원] "begin"에서 새 그룹 id 발급, "continue"는
                        # 그 id를 그대로 이어받고, "end"에서 그 그룹을 닫는다
                        # (다음 "begin"에서 새 id를 받도록). 빔 표시가 없는
                        # 음표(4분음표 이상, 또는 인식이 놓친 경우)는 -1(단독).
                        if rep_ev.beam_state == "begin":
                            beam_gid = next_beam_gid[0]
                            next_beam_gid[0] += 1
                            open_beam_by_voice[voice_id] = beam_gid
                        elif rep_ev.beam_state in ("continue", "end"):
                            beam_gid = open_beam_by_voice.get(voice_id, -1)
                            if rep_ev.beam_state == "end":
                                open_beam_by_voice.pop(voice_id, None)
                        else:
                            beam_gid = -1
                        chord_gid = -1
                        if is_chord:
                            chord_gid = next_chord_gid[0]
                            next_chord_gid[0] += 1
                        n = min(len(members), len(group))
                        if len(members) != len(group):
                            warnings.append(
                                f"마디 {mg.measure_id} voice{voice_id}: 화음 멤버 수 "
                                f"불일치(.omr={len(members)} vs .mxl={len(group)}) "
                                f"— 앞 {n}개만 사용"
                            )
                        if str(slot.chord_id) in geo.articulations_of_chord:
                            art_raw.append((next_elem_id[0], str(slot.chord_id), n))
                        for i in range(n):
                            gb = members[i]
                            ev = group[i]
                            # [실사용 검증 후 방어 코드] Audiveris가 아주 드물게
                            # 글리프에 이 시스템에 실제로 존재하는 보표 수를
                            # 벗어난 staff 번호(예: 단선율 악보인데 staff=4)를
                            # 잘못 매기는 경우를 발견했다 — 원본 MusicXML에는
                            # <staff> 태그 자체가 없는 음표였는데도 .omr 글리프
                            # 쪽에 잘못된 값이 있었다. 이런 값을 그대로 쓰면
                            # 후속 처리(다성부 MIDI 재생 등)가 "존재하지 않는
                            # 보표"를 독립 타임라인으로 취급해 극단적으로
                            # 이상한 결과(수십 초짜리 음, 등)를 만들어낼 수
                            # 있다. 이 시스템에 실제로 있는 보표 수를 벗어나면
                            # 안전하게 보표1로 보정한다.
                            gb_staff = staff_id_to_relative.get(gb.staff, 1)
                            elem_id = next_elem_id[0]
                            next_elem_id[0] += 1
                            elem_pos[elem_id] = (gb.x, gb.y + gb.h / 2.0, line_idx, gb.gid, gb_staff)
                            if gb.gid in geo.stem_of_head:            # [v2.30] 원본 기둥·빔
                                src_geo[elem_id] = (geo.stem_of_head[gb.gid], geo.beams_of_head.get(gb.gid))
                            elem_mid[elem_id] = mg.measure_id
                            elements.append((
                                line_idx, gb.x, gb.y, gb.w, gb.h,
                                beat, gb.x,  # stem_x 대체값(head 중심 x)
                                beam_gid,
                                rep_ev.accidental if i == 0 else ev.accidental,
                                rep_ev.is_grace, tup_ok,
                                gb_staff, voice_id, chord_gid, ferm_ok,
                                ev.tie_start, elem_id,
                                rep_ev.tup_actual, rep_ev.tup_normal,
                                # [v1.10] 19: 점 개수(겹점), 20: 이 요소부터 바뀌는 음자리표(대표 음에만)
                                rep_ev.dots, (rep_ev.clef_change if i == 0 else None),
                            ))
                            # 화음 대표(첫 멤버) 기준으로만 이음줄을 추적한다
                            # (화음 구성음 각각에 개별 이음줄이 걸리는 경우는
                            # 드물고, 대표 하나로도 실사용에 충분하다).
                            #
                            # [재설계 — 실사용 검증 후] 이전에는 slur를 "좌표
                            # (x1,x2)"로만 기록해뒀다가, 나중에 model.py의
                            # _attach_arcs()가 "그 시스템(줄) 전체(보통 4~5개
                            # 마디 전부)에서 가장 가까운 음표"를 다시 찾는
                            # 방식이었다. 코드 기호가 많고 음표가 촘촘한 가요
                            # 악보(실사용 "여정" 샘플)에서, 여러 마디의 음표
                            # x좌표가 우연히 가까워 **엉뚱한 마디의 음표에
                            # 잘못 매칭**되는 사고가 실제로(사용자 환경에서
                            # 결정적으로) 재현됐다 — 좌표 근사 매칭이라는
                            # 설계 자체의 근본적 한계였다. 이제 요소 고유
                            # ID를 직접 기록해, 나중에 좌표 재탐색 없이 정확히
                            # 그 요소를 곧바로 연결한다(오매칭 원천 차단).
                            if ev.kind == "note" and ev.step is not None:
                                tkey = (pi, voice_id, ev.step, ev.octave)
                                if ev.tie_stop and tkey in open_tie_by_key:
                                    tie_pairs.append({"start_id": open_tie_by_key.pop(tkey),
                                                      "stop_id": elem_id})
                                if ev.tie_start:
                                    open_tie_by_key[tkey] = elem_id
                            if i == 0:
                                # [셈여림·쐐기] 화음은 대표음 하나에만 붙인다(화음 전체에 걸리는 표시).
                                if g_harm:
                                    harm_raw.append({"elem_id": elem_id, "text": g_harm})
                                if g_dyn:
                                    dyn_raw.append({"elem_id": elem_id, "key": g_dyn,
                                                    "nx": gb.x + gb.w / 2.0, "ny": gb.y + gb.h / 2.0})
                                if g_wstop and gb_staff in open_wedge:
                                    w_start, w_kind = open_wedge.pop(gb_staff)
                                    if w_start != elem_id:
                                        wedges.append({"start_id": w_start, "stop_id": elem_id,
                                                       "kind": w_kind})
                                if g_wstart:
                                    open_wedge[gb_staff] = (elem_id, g_wstart)
                                if g_sstart:
                                    open_slur_by_voice[(pi, voice_id)] = elem_id
                                if g_sstop and (pi, voice_id) in open_slur_by_voice:
                                    start_id = open_slur_by_voice.pop((pi, voice_id))
                                    slurs.append({"start_id": start_id, "stop_id": elem_id})
                            if ev.kind == "rest" or ev.step is None:
                                pitches.append(-1)
                            else:
                                pidx = _step_octave_to_pitch_index(
                                    ev.step, ev.octave, MODEL_PITCH_NAMES)
                                pitches.append(pidx if pidx is not None else -1)

        barlines[line_idx] = bars_this_line
        barline_types[line_idx] = types_this_line
        for mid, cap in measure_capacity_override.items():
            sys_capacity[(line_idx, mid)] = cap
        line_idx += 1

    # [붙임줄 신뢰도 필터 — 실사용 검증 후] Audiveris는 8분·16분음표의 꼬리를
    # 붙임줄로 오인하는 경우가 많다(교가 샘플: 붙임줄 4개 중 진짜는 1개).
    # .omr에는 붙임줄 곡선마다 신뢰도(grade)가 있어 진짜 0.65 / 허위 0.29~0.37로
    # 갈린다. 각 짝의 (시작x, 끝x)와 가장 가까운 tie 곡선을 찾아 그 grade가
    # TIE_MIN_GRADE 미만이면 버린다. 버린 짝의 시작 음표는 tie_start 플래그도
    # 꺼야 한다(안 그러면 화면엔 없는데 MIDI 재생에서만 두 음이 이어져 버린다).
    TIE_MIN_GRADE = 0.5
    tie_inters = [t for t in geo.slurs if t["tie"]]
    il = geo.interline_px or 20.0

    # [v1.42 이음줄 신뢰도 — 실사용] "교회" 13→14마디 합창: 가사 글자 조각을 SLUR_BELOW(0.193)로 읽어 원본에
    # 없는 이음줄이 생겼다. 붙임줄처럼 이음줄도 짝(시작·끝 음)에 가장 잘 맞는 slur inter의 신뢰도를 보고
    # 0.3 미만이면 버린다(이 악보의 진짜 이음줄은 0.42 이상).
    SLUR_MIN_GRADE = 0.3
    slur_inters = [t for t in geo.slurs if not t["tie"]]
    kept_slurs, dropped_slurs = [], 0
    for pr in slurs:
        ps, pe = elem_pos.get(pr["start_id"]), elem_pos.get(pr["stop_id"])
        keep = True
        if ps is not None and pe is not None and slur_inters:
            if ps[2] == pe[2]:
                # 짝의 가로 범위(시작 음 ~ 끝 음, 앞뒤 1칸)와 겹치는 이음줄 inter
                lo_, hi_ = min(ps[0], pe[0]) - il, max(ps[0], pe[0]) + 2 * il
                cands = [t for t in slur_inters if t["x"] < hi_ and t["x"] + t["w"] > lo_]
            else:
                cands = [t for t in slur_inters if abs(t["x"] - ps[0]) < 3 * il
                         or abs(t["x"] + t["w"] - pe[0]) < 3 * il]
            cands = [t for t in cands if abs((t["y"] + t["h"] / 2.0) - (ps[1] + pe[1]) / 2.0) < 6 * il]
            if cands:
                best = min(cands, key=lambda t: abs(t["x"] - ps[0]) + abs(t["x"] + t["w"] - pe[0]))
                keep = best["grade"] >= SLUR_MIN_GRADE
                if keep and best.get("above") is not None:
                    pr["above"] = best["above"]           # [v2.03] 원본 곡선이 휜 방향(위·아래)
        if keep:
            _attach_curves(pr, slur_inters, ps, pe, il)
            kept_slurs.append(pr)
        else:
            dropped_slurs += 1
    slurs = kept_slurs
    if dropped_slurs:
        warnings.append(f"신뢰도가 낮은 이음줄 {dropped_slurs}개(글자 조각 등)는 표시하지 않았습니다"
                        f"(필요하면 '이음' 버튼으로 직접 추가)")

    def _near_y(t, p):
        return abs((t["y"] + t["h"] / 2.0) - p[1]) < 4 * il

    dropped_start_ids = set()
    for pr in tie_pairs:
        ps, pe = elem_pos.get(pr["start_id"]), elem_pos.get(pr["stop_id"])
        keep = True
        # 1순위: .omr의 relation(곡선→음표머리)으로 정확히 대조 — 이 짝의 시작·끝
        # 음표머리에 실제로 걸린 tie 곡선의 신뢰도를 본다(줄을 넘는 반쪽 곡선도 그대로 잡힘).
        linked = [t for t in tie_inters
                  if ps is not None and pe is not None and
                  ((t["left"] and t["left"] == ps[3]) or (t["right"] and t["right"] == pe[3]))]
        if linked:
            keep = max(t["grade"] for t in linked) >= TIE_MIN_GRADE
        elif tie_inters and ps is not None and pe is not None:
            # 2순위(relation이 없을 때): 위치 근사
            if ps[2] == pe[2]:
                # 같은 줄: 그 음표 높이 근처의 곡선 중 (시작x, 끝x)가 가장 맞는 것
                cands = [t for t in tie_inters if _near_y(t, ps)]
                if cands:
                    best = min(cands, key=lambda t: abs(t["x"] - ps[0]) + abs(t["x"] + t["w"] - pe[0]))
                    keep = best["grade"] >= TIE_MIN_GRADE
            else:
                # 줄을 넘는 붙임줄: Audiveris는 반쪽 곡선 두 개(줄 끝/다음 줄 앞)로
                # 인식한다. 둘 중 하나라도 신뢰도가 충분하면 진짜로 본다.
                c1 = [t for t in tie_inters if _near_y(t, ps) and abs(t["x"] - ps[0]) < 3 * il]
                c2 = [t for t in tie_inters if _near_y(t, pe) and abs(t["x"] + t["w"] - pe[0]) < 3 * il]
                grades = [max(t["grade"] for t in c) for c in (c1, c2) if c]
                if grades:
                    keep = max(grades) >= TIE_MIN_GRADE
        if keep:
            _attach_curves(pr, tie_inters, ps, pe, il)
            slurs.append(pr)
        else:
            dropped_start_ids.add(pr["start_id"])
    # [셈여림 신뢰도 필터 — 실측 근거] 깨끗한 시험 악보의 진짜 셈여림은 신뢰도 0.63~0.79,
    # 실제 교가·데모곡에서 가사·코드기호를 오인한 허위는 0.28~0.46이었다. 같은 모양의
    # 가장 가까운 셈여림 inter를 찾아 0.6 미만이면 버린다(진짜는 모두 남고 허위 대부분 제거).
    DYN_MIN_GRADE = 0.6
    dynamics = []
    dropped_dyn = 0
    for dr in dyn_raw:
        cands = [g for g in geo.dynamics if g["key"] == dr["key"]
                 and abs(g["x"] + g["w"] / 2 - dr["nx"]) < 6 * il
                 and abs(g["y"] + g["h"] / 2 - dr["ny"]) < 10 * il]
        best = min(cands, key=lambda g: abs(g["x"] + g["w"] / 2 - dr["nx"])
                   + 0.3 * abs(g["y"] + g["h"] / 2 - dr["ny"])) if cands else None
        if best is not None and best["grade"] < DYN_MIN_GRADE:
            dropped_dyn += 1
            continue
        if best is None:
            # [v1.42] 원본에서 그 기호를 찾지 못한 셈여림은 근거가 없다("교회" 13마디 f: 가사를 0.231로 오인)
            dropped_dyn += 1
            continue
        # [v1.42 노래 보표 아래 = 가사 자리] 노래·합창 파트 보표의 '아래'에 있는 셈여림은 가사 글자를
        # 오인한 것으로 본다("교회" 9~11·14마디 합창 p — 신뢰도 0.6~0.7이라 신뢰도로는 못 거름).
        # 노래 파트의 셈여림은 가사와 겹치지 않게 보표 위에 쓰는 것이 관례다.
        pinfo = elem_pos.get(dr["elem_id"])
        if pinfo is not None and len(pinfo) > 4:
            ln_, sn_ = pinfo[2], pinfo[4]
            parts_ln = staff_parts.get(ln_) or []
            pidx = parts_ln[sn_ - 1] if 0 < sn_ <= len(parts_ln) else None
            pname = (music.part_names[pidx] if pidx is not None and pidx < len(music.part_names) else "")
            grp_ = staff_groups.get(ln_) or []
            if _is_vocal(pname) and 0 < sn_ <= len(grp_):
                bottom = grp_[sn_ - 1][4]
                if best["y"] + best["h"] / 2.0 > bottom + 0.5 * il:
                    dropped_dyn += 1
                    continue
        dynamics.append({"elem_id": dr["elem_id"], "key": dr["key"],
                         "x": (best["x"] + best["w"] / 2) if best else None,
                         "y": (best["y"] + best["h"] / 2) if best else None,
                         "w": best["w"] if best else None, "h": best["h"] if best else None,
                         "grade": best["grade"] if best else None})
    if dropped_tuplets[0]:
        warnings.append(f"신뢰도가 낮은 잇단음표 표시 {dropped_tuplets[0]}개(글자를 숫자 3으로 오인 등)는 "
                        f"적용하지 않았습니다(필요하면 '잇단' 버튼으로 직접 추가)")
    if dropped_fermatas[0]:
        warnings.append(f"신뢰도가 낮은 늘임표 {dropped_fermatas[0]}개(가사 글자를 늘임표로 오인 등)는 "
                        f"표시하지 않았습니다(필요하면 U 키로 직접 추가)")
    if dropped_dyn:
        warnings.append(f"신뢰도가 낮은 셈여림 {dropped_dyn}개는 표시하지 않았습니다"
                        f"(필요하면 '셈여림' 메뉴로 직접 추가)")

    # [끝 없는 쐐기 보완 — 실사용 검증] Audiveris가 쐐기 '시작'만 적고 '끝(stop)'을 빠뜨린
    # 경우("봄이 오면" 22마디 노래 보표)는 .omr의 실제 쐐기 길이 안에 있는 같은 보표의 마지막
    # 음표를 끝으로 잡는다.
    il_w = geo.interline_px or 20.0
    for st_key, (w_start, w_kind) in list(open_wedge.items()):
        ps = elem_pos.get(w_start)
        if ps is None:
            continue
        cands = [g for g in geo.wedges if g["kind"] == w_kind
                 and abs(g["x"] - ps[0]) < 3 * il_w and abs(g["y"] + g["h"] / 2 - ps[1]) < 9 * il_w]
        if not cands:
            continue
        g = min(cands, key=lambda g: abs(g["x"] - ps[0]))
        inside = [(p[0], eid) for eid, p in elem_pos.items()
                  if p[2] == ps[2] and len(p) > 4 and p[4] == st_key and ps[0] < p[0] <= g["x"] + g["w"] + il_w]
        if inside:
            wedges.append({"start_id": w_start, "stop_id": max(inside)[1], "kind": w_kind})

    # [쐐기 실제 좌표] Audiveris는 쐐기 '끝'을 마지막 음 앞에 두는 등 음표 기준으로는
    # 원본 길이와 어긋난다(시험 악보: 마디선까지 이어진 쐐기가 마지막 음에서 끝남).
    # 같은 종류·시작 음표 근처의 wedge inter를 찾아 원본 그대로의 x범위·높이를 붙인다.
    # [배정 방식 개선 — 실사용 검증] Audiveris가 쐐기의 보표를 잘못 적는 경우("봄이 오면" 22마디
    # 왼손 디크레셴도가 오른손 소속으로 옴)에는 '시작 음표 높이 근처'에서 찾으면 실패한다.
    # 같은 줄(시스템) 안에서 같은 종류·가까운 x의 쐐기를 후보로 두고, 비용(가로 거리 + 세로
    # 거리 일부)이 낮은 짝부터 하나씩만 배정한다(노래·피아노가 같은 x에 각자 쐐기를 가질 때
    # 서로 빼앗지 않게). 화면에는 배정된 실제 좌표로 그린다.
    def _line_span(li):
        grp = staff_groups.get(li) or []
        if not grp:
            return None
        return grp[0][0] - 8 * il, grp[-1][4] + 10 * il
    cand = []
    for wi, wg in enumerate(wedges):
        ps = elem_pos.get(wg["start_id"])
        if ps is None:
            continue
        span = _line_span(ps[2])
        for gi, g in enumerate(geo.wedges):
            gy = g["y"] + g["h"] / 2
            if g["kind"] != wg["kind"] or abs(g["x"] - ps[0]) >= 3 * il:
                continue
            if span is not None and not (span[0] <= gy <= span[1]):
                continue
            cand.append((abs(g["x"] - ps[0]) + 0.2 * abs(gy - ps[1]), wi, gi))
    used_w, used_g = set(), set()
    for _cost, wi, gi in sorted(cand):
        if wi in used_w or gi in used_g:
            continue
        used_w.add(wi)
        used_g.add(gi)
        g = geo.wedges[gi]
        wedges[wi].update({"x1": g["x"], "x2": g["x"] + g["w"], "y": g["y"] + g["h"] / 2, "h": g["h"]})

    # [v1.32 스타카토로 오인한 점 — 실사용] Audiveris가 부점(augmentation dot)을 스타카토로 읽는 경우
    # ("기도의 능력" 2마디 아래 성부 F4: 머리 바로 오른쪽 같은 높이의 점이 STACCATO 0.779). 스타카토는
    # 자기 머리의 위나 아래 '가운데'에 붙고, 부점은 머리 '오른쪽 같은 높이'에 붙는다. 머리 오른쪽 끝에서
    # 0.1~1.6칸 오른쪽, 세로 ±0.6칸 안에 있고, 다른 머리의 가로 범위 안(그 머리의 스타카토)이 아니면
    # 부점으로 고친다(점이 없는 음만, 화음이면 구성음 모두).
    il_a = geo.interline_px or 20.0
    heads = [(eid, p[0], p[1]) for eid, p in elem_pos.items()]
    idx_by_id = {e[16]: i for i, e in enumerate(elements) if len(e) > 16}
    head_w = {e[16]: e[3] for e in elements if len(e) > 16}
    fixed_dots = 0
    for ar in geo.articulations:
        if ar["shape"] != "STACCATO":
            continue
        cx, cy = ar["x"] + ar["w"] / 2.0, ar["y"] + ar["h"] / 2.0
        if any(hx - 0.3 * il_a <= cx <= hx + head_w.get(eid, il_a) + 0.3 * il_a
               and 0.6 * il_a < abs(cy - hy) <= 2.0 * il_a for eid, hx, hy in heads):
            continue                              # 어느 머리 바로 위·아래(2칸 안) 가운데 → 진짜 스타카토
        cands = []
        for eid, hx, hy in heads:
            right = hx + head_w.get(eid, il_a)
            if right + 0.1 * il_a <= cx <= right + 1.6 * il_a and abs(cy - hy) <= 0.6 * il_a:
                cands.append((cx - right + abs(cy - hy), eid))
        if not cands:
            continue
        eid = min(cands)[1]
        i = idx_by_id.get(eid)
        if i is None or elements[i][5] < 0:
            continue
        cg = elements[i][13]
        for j, e in enumerate(elements):
            if j == i or (cg != -1 and e[13] == cg and e[0] == elements[i][0]):
                e = list(e)
                e[5] = -abs(e[5])
                if len(e) > 19:
                    e[19] = max(1, e[19])
                elements[j] = tuple(e)
        fixed_dots += 1
    if fixed_dots:
        warnings.append(f"스타카토로 잘못 읽은 부점 {fixed_dots}개를 부점으로 바로잡았습니다")

    # [v1.43 약한 임시표 — 실사용] "교회" 16마디 왼손 마지막 B3의 ♮: 너비 2px의 잉크 조각을 NATURAL(0.367)로
    # 읽음. 신뢰도가 0.5 미만이고, 조표만으로도 같은 음높이가 되는(소리에 영향 없는) 임시표는 지운다.
    # 같은 마디에서 앞서 같은 음에 다른 임시표가 있었으면(제자리표가 실제로 필요한 경우) 남긴다.
    from score.model import _PITCH_NAMES as _PN
    _acc_alt = {"sharp": 1, "flat": -1, "natural": 0}
    dropped_acc = 0
    for idx, e in enumerate(elements):
        acc = e[8] if len(e) > 8 else None
        if not acc or len(e) <= 16 or pitches[idx] in (-1, None):
            continue
        x, y, w, h = e[1], e[2], e[3], e[4]
        cyh = y + h / 2.0
        cands = [a for a in geo.alters if x - 3 * il <= a["x"] + a["w"] / 2.0 <= x + 0.2 * il
                 and abs(a["y"] + a["h"] / 2.0 - cyh) <= 1.2 * il]
        if not cands:
            continue
        best = min(cands, key=lambda a: abs(a["x"] + a["w"] / 2.0 - (x - 0.8 * il)))
        if best["grade"] >= 0.5:
            continue
        mid = elem_mid.get(e[16], 1)
        f = music.fifths
        for kn in sorted(music.key_changes):
            if kn <= mid:
                f = music.key_changes[kn]
        name = _PN[pitches[idx]]
        step = name[0]
        key_alt = {}
        if f > 0:
            key_alt = {s_: 1 for s_ in "FCGDAEB"[:f]}
        elif f < 0:
            key_alt = {s_: -1 for s_ in "BEADGCF"[:-f]}
        if _acc_alt.get(acc) != key_alt.get(step, 0):
            continue                              # 소리를 바꾸는 임시표는 남긴다
        prior = [o for j, o in enumerate(elements) if o is not e and len(o) > 16 and o[8]
                 and elem_mid.get(o[16]) == mid and o[0] == e[0] and o[11] == e[11] and o[1] < x
                 and pitches[j] not in (-1, None) and _PN[pitches[j]][0] == step]
        if prior:
            continue
        elements[idx] = e[:8] + (None,) + e[9:]
        dropped_acc += 1
    # [v2.03] 남은 임시표는 원본에 찍힌 x를 기억한다(머리 가운데 기준 거리) — 화음·2도에서 계산 위치가
    # 마디선에 걸치던 문제("교회" 29마디 오른손 ♮)
    acc_dx = {}
    for idx, e in enumerate(elements):
        if not (len(e) > 16 and e[8]):
            continue
        x, y, w, h = e[1], e[2], e[3], e[4]
        cands = [a for a in geo.alters if x - 3 * il <= a["x"] + a["w"] / 2.0 <= x + 0.2 * il
                 and abs(a["y"] + a["h"] / 2.0 - (y + h / 2.0)) <= 1.2 * il and a["grade"] >= 0.5]
        if cands:
            a = min(cands, key=lambda a: abs(a["x"] + a["w"] / 2.0 - (x - 0.8 * il)))
            acc_dx[e[16]] = (a["x"] + a["w"] / 2.0) - (x + w / 2.0)

    # [v2.03 아티큘레이션 — 실사용] 악센트(>)·스타카토 등을 화음 대표 음에 붙인다. Audiveris가 화음과
    # 짝지어 준 것(chord-articulation) 중: 신뢰도 0.5 이상, 머리에서 4.5칸 안, 노래 파트 보표 '아래'(가사 자리)는
    # 제외("교회" 3쪽 합창 가사의 '—'·글자를 TENUTO·ACCENT로 26개 오인), v1.32에서 부점으로 바꾼 점은 제외.
    _ART = {"ACCENT": "accent", "STACCATO": "staccato", "TENUTO": "tenuto",
            "STACCATISSIMO": "staccatissimo", "STRONG_ACCENT": "marcato"}
    articulations = []
    head_w_ = {e[16]: e[3] for e in elements if len(e) > 16}
    for eid0, cid, nmem in art_raw:
        ids = list(range(eid0, eid0 + nmem))
        heads_ = [elem_pos[i] for i in ids if i in elem_pos]
        if not heads_:
            continue
        rep = elem_pos[eid0]
        for ar in geo.articulations_of_chord.get(cid, []):
            kind = _ART.get(ar["shape"].replace("_BELOW", "").replace("_ABOVE", ""))
            if kind is None or ar["grade"] < 0.5:
                continue
            cx, cy = ar["x"] + ar["w"] / 2.0, ar["y"] + ar["h"] / 2.0
            near = min(abs(cy - hp[1]) for hp in heads_)
            if near > 4.5 * il:
                continue
            ln0, sn0 = elem_pos[eid0][2], elem_pos[eid0][4]
            grp0 = (staff_groups.get(ln0) or [])
            if kind in ("staccato", "tenuto", "staccatissimo"):
                # 스타카토·테누토는 머리 바로 옆(2.2칸 안)에 붙는다 — 볼타 번호 "1."의 마침표 등 제외
                if near > 2.2 * il:
                    continue
                if kind == "tenuto" and 0 < sn0 <= len(grp0):
                    # 덧줄을 테누토로 읽은 것: 오선 줄 자리(위·아래로 정수 칸)에 놓인 가로선은 덧줄이다
                    ys0 = grp0[sn0 - 1]
                    # 오선 밖 머리와 오선 사이(또는 머리를 지나는) 가로선 = 덧줄. 진짜 테누토는 머리 바깥쪽에 붙는다
                    # (오선 밖에서는 테누토도 줄 자리에 올 수 있어 '줄 자리'만으로 덧줄이라 하면 안 된다)
                    if any((hp[1] < ys0[0] and hp[1] - 0.35 * il <= cy <= ys0[0] + 0.3 * il) or
                           (hp[1] > ys0[4] and ys0[4] - 0.3 * il <= cy <= hp[1] + 0.35 * il)
                           for hp in heads_):
                        continue
            if kind == "staccato" and any(hp[0] + head_w_.get(i, il) + 0.1 * il <= cx <= hp[0] + head_w_.get(i, il) + 1.6 * il
                                          and abs(cy - hp[1]) <= 0.6 * il for i, hp in zip(ids, heads_)):
                continue                                 # 부점으로 바꾼 점
            ln_, sn_ = rep[2], rep[4]
            parts_ln = staff_parts.get(ln_) or []
            pidx = parts_ln[sn_ - 1] if 0 < sn_ <= len(parts_ln) else None
            pname = music.part_names[pidx] if pidx is not None and pidx < len(music.part_names) else ""
            grp_ = staff_groups.get(ln_) or []
            if _is_vocal(pname) and 0 < sn_ <= len(grp_) and cy > grp_[sn_ - 1][4] + 0.5 * il:
                continue                                 # 노래 보표 아래 = 가사 자리
            hx = rep[0] + head_w_.get(eid0, il) / 2.0
            above = cy < min(hp[1] for hp in heads_)
            articulations.append({"elem_id": eid0, "kind": kind, "above": above,
                                  "dx": cx - hx, "dy": cy - rep[1]})
    if dropped_acc:
        warnings.append(f"신뢰도가 낮고 소리에 영향이 없는 임시표 {dropped_acc}개는 표시하지 않았습니다")

    if dropped_start_ids:
        elements = [(e[:15] + (False,) + e[16:]) if (len(e) > 16 and e[16] in dropped_start_ids) else e
                    for e in elements]
        warnings.append(f"신뢰도가 낮은 붙임줄 {len(dropped_start_ids)}개는 표시하지 않았습니다"
                        f"(필요하면 '이음' 버튼으로 직접 추가)")

    ctx = {
        "staves": staves_flat,
        "staff_groups": staff_groups,  # [Tier2] {line_idx: [[y0..y4], ...]} 다중 보표 배치용
        "staff_left": staff_left_by_line,  # [Tier2 버그 수정] {line_idx: 실측 오선 시작 x}
        "recognition": {
            "key": _fifths_to_key(music.fifths),
            # [버그 수정] music.beats는 마지막 <time> 값이라 곡 중간에 박자가 바뀌면
            # (4/4 … 6/8) 첫 줄 박자표까지 6/8로 잘못 나왔다. 첫 마디의 박자를 쓴다.
            "time_signature": (music.time_changes[min(music.time_changes)]
                               if music.time_changes else (music.beats, music.beat_type)),
            "time_symbol": None,
            "beats": [e[5] for e in elements],
            "pitches": pitches,
            "elements": elements,
            # [Tier2 신규] {마디번호: (beats,beat_type)} — 마디1(악보 전체
            # 시작 박자표, 이미 첫 시스템 헤더에 그려짐)은 제외하고, 중간에
            # 실제로 바뀌는 지점만 넘긴다.
            "time_sig_changes": _filter_real_time_changes(music),
            # [v2.21] 인쇄된 박자표 전부(값이 같아도) — 놓친 박자표 추론에 쓴다
            "time_printed": dict(music.time_changes),
            # [v1.10 조표 바뀜] {마디번호: fifths} — 곡 중간에서 실제로 조표가 달라지는 마디
            "key_changes": dict(music.key_changes),
        },
        "barlines": barlines,
        "barline_types": barline_types,
        "measure_info": {"barlines": barlines, "measure_results": [], "sys_capacity": sys_capacity},
        "slurs": slurs, "voltas": _voltas_from_geo(geo, music), "hairpins": [],
        "articulations": articulations, "acc_dx": acc_dx, "src_geo": src_geo,
        "header_x": _header_x(geo, elements),
        "staff_clefs": staff_clefs,   # {line_idx: ["G","G","F"]} 보표별 음자리표
        "staff_parts": staff_parts,   # {line_idx: [0,1,1]} 보표별 논리 파트
        "part_names": list(music.part_names),   # [v1.20] 논리 파트 순서의 파트 이름(Choir, Piano …)
        "dynamics": dynamics,    # [셈여림] [{elem_id,key,x,y,w,h,grade}] — 음표에 붙는 셈여림
        "harmonies": harm_raw,   # [v3.00] 기타 코드 [{elem_id, text}]
        "wedges": wedges,        # [쐐기] [{start_id,stop_id,kind}] — 크레셴도/디크레셴도
        # 원본 이미지 크기 및 업스케일 배율 — 편집기가 sx,sy 계산에 사용.
        "_source_size": (original_w, original_h),
        "_upscale_factor": upscale_factor,
    }
    return MergeResult(ctx=ctx, warnings=warnings)


def _header_x(geo, elements):
    """[v2.30] 줄마다·보표마다 줄 머리 기호의 원본 x: {줄: {보표: {"clef": 가운데, "key": 왼쪽, "time": 가운데}}}.
    그 줄 첫 음표보다 왼쪽에 있는 기호만(줄 가운데의 음자리표·조표 바뀜은 제외)."""
    where = {}
    for li, sysg in enumerate(geo.systems):
        for k, st in enumerate(sysg.staves):
            where[st.staff_id] = (li, k + 1)
    first_x = {}
    for e in elements:
        first_x[e[0]] = min(first_x.get(e[0], 1e9), e[1])
    out = {}
    for hs in sorted(geo.header_syms, key=lambda h: h["x"]):
        loc = where.get(hs["staff"])
        if loc is None or hs["x"] >= first_x.get(loc[0], 1e9):
            continue
        d = out.setdefault(loc[0], {}).setdefault(loc[1], {})
        if hs["kind"] in d:
            continue
        d[hs["kind"]] = hs["x"] if hs["kind"] == "key" else hs["x"] + hs["w"] / 2.0
        if hs["kind"] == "key":
            d["key_w"] = hs["w"]
    return out


def _attach_curves(pr, inters, ps, pe, il):
    """[v2.30 원본 곡선] 짝(시작·끝 음)에 맞는 원본 곡선을 pr에 적는다.
    같은 줄: pr["curve"] — 두 머리에 relation으로 걸린 곡선, 없으면 위치가 가장 맞는 곡선.
    줄을 넘음: pr["curve_a"](시작 음 쪽 반쪽), pr["curve_b"](끝 음 쪽 반쪽)."""
    if ps is None or pe is None or not inters:
        return
    near = [t for t in inters if t.get("curve") and abs(t["y"] + t["h"] / 2.0 - (ps[1] + pe[1]) / 2.0) < 6 * il]
    if ps[2] == pe[2]:
        both = [t for t in inters if t.get("curve") and t.get("left") == ps[3] and t.get("right") == pe[3]]
        if both:
            pr["curve"] = both[0]["curve"]
            return
        lo_, hi_ = min(ps[0], pe[0]) - il, max(ps[0], pe[0]) + 2 * il
        c = [t for t in near if t["x"] < hi_ and t["x"] + t["w"] > lo_]
        if c:
            best = min(c, key=lambda t: abs(t["x"] - ps[0]) + abs(t["x"] + t["w"] - pe[0]))
            pr["curve"] = best["curve"]
        return
    a = [t for t in inters if t.get("curve") and t.get("left") == ps[3]] or \
        [t for t in inters if t.get("curve") and abs(t["x"] - ps[0]) < 2 * il
         and abs(t["y"] + t["h"] / 2.0 - ps[1]) < 6 * il]
    b = [t for t in inters if t.get("curve") and t.get("right") == pe[3]] or \
        [t for t in inters if t.get("curve") and abs(t["x"] + t["w"] - pe[0]) < 2.5 * il
         and abs(t["y"] + t["h"] / 2.0 - pe[1]) < 6 * il]
    if a:
        pr["curve_a"] = min(a, key=lambda t: abs(t["x"] - ps[0]))["curve"]
    if b:
        pr["curve_b"] = min(b, key=lambda t: abs(t["x"] + t["w"] - pe[0]))["curve"]


def _voltas_from_geo(geo, music):
    """[v2.00 볼타] .omr의 볼타 꺾쇠(보표마다 하나씩 그려진다)를 줄마다 하나로 묶고, MusicXML에서 읽은
    볼타 번호를 나오는 차례대로 붙인다 → model의 voltas 형식 [{line, x1, x2, number}]."""
    staff_line = {}
    for li, sysg in enumerate(geo.systems):
        for st in sysg.staves:
            staff_line[st.staff_id] = li
    seen = []
    for en in sorted(geo.endings, key=lambda e: (staff_line.get(e["staff"], 99), e["x1"])):
        li = staff_line.get(en["staff"])
        if li is None:
            continue
        same = [v for v in seen if v["line"] == li and abs(v["x1"] - en["x1"]) < 20]
        if same:
            same[0]["ys"].append(en["y"])             # [v2.54] 같은 볼타의 다른 파트 꺾쇠 — 원본 높이 모두
            continue
        seen.append({"line": li, "x1": en["x1"], "x2": en["x2"], "number": None, "ys": [en["y"]]})
    nums = list(getattr(music, "ending_numbers", []) or [])
    for i, v in enumerate(seen):
        v["number"] = nums[i] if i < len(nums) else (i + 1)
    return seen


_VOCAL_RE = None


def _is_vocal(name: str) -> bool:
    """[v1.42] 노래·합창 파트 이름인가(Audiveris 기본 이름 'Voice' 포함)."""
    global _VOCAL_RE
    if _VOCAL_RE is None:
        import re
        _VOCAL_RE = re.compile(r"voice|vocal|choir|chorus|soprano|alto|tenor|bass(?!oon)|baritone|"
                               r"sop\b|^s\.?a\.?|^t\.?b\.?|solo|노래|합창|성악|소프라노|알토|테너|베이스", re.I)
    return bool(name) and bool(_VOCAL_RE.search(name))


def _time_sig_capacity(beats: int, beat_type: int) -> float:
    """박자표 -> 4분음표 단위 마디 용량(예: 4/4=4.0, 3/8=1.5)."""
    if not beat_type:
        return 4.0
    return beats * 4.0 / beat_type


def _fifths_to_key(fifths: int) -> int:
    """score/model.py의 key_to_fifths 역함수 — model.py 인코딩(#=10n, b=100n) 사용."""
    if fifths >= 0:
        return fifths * 10
    return -fifths * 100
