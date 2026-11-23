# score_model.py
# ---------------------------------------------------------------------------
# 인식 결과(흩어진 리스트들)를 '편집 가능한 하나의 문서 모델'로 재구성한다.
#
# 이 모델이 프로그램의 중심(single source of truth)이다.
#   - 화면(편집기)은 이 모델을 그린다.
#   - 사용자의 수정은 이 모델을 바꾼다.
#   - MusicXML 내보내기는 이 모델을 직렬화한다.
#   - 각 요소는 원본 이미지 좌표(src_bbox)를 들고 있어, 편집 중 원본과 대조할 수 있다.
#
# 입력(모두 modules.recognition / detect_barlines 이 이미 만들어 준 것):
#   key            : 조표 값(플랫 100/개, 샾 10/개, 0=다장조)
#   time_signature : (분자, 분모)  예: (8, 4)
#   elements       : [(line, x, y, w, h, beat), ...]  ← 원본 위치 포함
#   pitches        : elements 와 같은 순서/길이. -1=쉼표, 그 외 0~20=음정 인덱스
#   barlines       : {보표번호: [마디선 x, ...]}
#
# 산출: Score → System(보표) → Measure(마디) → Element(음표/쉼표/온음표)
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Optional

# 음정 인덱스 → (음이름 문자, 옥타브).
# [Tier2 버그 수정, 재설계] 원래는 "F3"~"E6"(21개, 성가·동요 같은
# 단선율 보컬 음역만 상정)만 있었다. 이 상태에서 피아노 등 넓은 음역
# 악기를 인식하면, 왼손 저음부(B2·F2·F1·G1·D1 등)가 이 테이블에 없어
# _step_octave_to_pitch_index()가 매번 실패 → 전부 쉼표로 잘못
# 치환되는 심각한 버그가 있었다(실사용 검증 중 발견).
#
# **주의**: 처음에는 이 테이블을 통째로 새로 만들어(C0부터 순서대로)
# 확장을 시도했으나, 이러면 기존 인덱스 값(예: 0=F3)이 완전히
# 달라져서 -- 기존 omr/engine.py(Tier1 구엔진)가 자기 자신의 동일한
# 21개 테이블 기준으로 이미 계산해 둔 pitch 정수 인덱스와 더 이상
# 맞지 않게 되는 위험한 회귀가 있었다(tools/regression.py는 마디
# 개수만 확인해 이 문제를 못 잡아냈다 -- 직접 음이름을 대조해보고
# 발견함). 그래서 **기존 21개(인덱스 0~20)는 절대 순서를 바꾸지
# 않고 그대로 유지**하고, 피아노 전체 음역 중 이 21개에 없는 음들만
# 뒤에(인덱스 21부터) 추가하는 방식으로 안전하게 확장한다. 인덱스
# "정렬 순서"는 이 프로젝트 어디에서도 산술적으로 쓰이지 않음을
# 확인했다(조옮김 등은 step/octave/alter 필드를 직접 쓰지 이
# 정수 인덱스를 쓰지 않는다 -- score/editing.py transpose_score
# 참고) -- 그래서 "값이 유일하기만 하면" 순서와 무관하게 안전하다.
_PITCH_NAMES_BASE = [
    "F3", "G3", "A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4", "B4",
    "C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6", "D6", "E6",
]

_NOTE_LETTERS = ["C", "D", "E", "F", "G", "A", "B"]


def _build_extended_pitch_names(base, lo_octave=0, hi_octave=8):
    """base(기존 21개, 인덱스 0~20 그대로 고정)에 없는 음만 뒤에 추가."""
    names = list(base)
    seen = set(base)
    for octv in range(lo_octave, hi_octave + 1):
        for letter in _NOTE_LETTERS:
            name = f"{letter}{octv}"
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names


_PITCH_NAMES = _build_extended_pitch_names(_PITCH_NAMES_BASE, 0, 8)

# 조표(플랫 나열 순서 B E A D G C F / 샾 나열 순서 F C G D A E B)
_FLAT_ORDER = ["B", "E", "A", "D", "G", "C", "F"]
_SHARP_ORDER = ["F", "C", "G", "D", "A", "E", "B"]

# 박자 코드(절댓값) → MusicXML note type
_TYPE = {1: "whole", 2: "half", 4: "quarter", 8: "eighth", 16: "16th", 32: "32nd"}


def key_to_fifths(key: int) -> int:
    """조표 값 → MusicXML fifths(+샾수 / -플랫수 / 0)."""
    if key >= 100:
        return -(key // 100)      # 플랫
    if 0 < key < 100:
        return key // 10          # 샾
    return 0


def _alter_map(fifths: int) -> dict:
    """조표에 따라 각 음이름 문자가 받는 반음(alter)을 미리 계산해 둔다.

    임시표(개별 음표의 ♯♭♮)는 현재 인식되지 않으므로, 우선 조표만 반영한다.
    (임시표는 편집 화면에서 사용자가 지정 → 그때 Element.alter 를 직접 덮어쓴다.)
    """
    m = {c: 0 for c in "ABCDEFG"}
    if fifths < 0:
        for c in _FLAT_ORDER[:(-fifths)]:
            m[c] = -1
    elif fifths > 0:
        for c in _SHARP_ORDER[:fifths]:
            m[c] = +1
    return m


@dataclass
class Element:
    """마디 안의 한 요소(음표/쉼표/온음표). 편집의 최소 단위."""
    kind: str                 # "note" | "rest"
    duration_base: int        # 1/2/4/8/16/32 (박자 코드 절댓값)
    dotted: bool              # 점(부점) 여부
    step: Optional[str] = None    # "C".."B" (음표만)
    octave: Optional[int] = None  # 4, 5 ... (음표만)
    alter: int = 0                # -1 플랫, +1 샾, 0 (조표+임시표 반영 결과)
    src_bbox: tuple = None        # (x, y, w, h) 원본 이미지 좌표
    src_x: int = 0                # 정렬/마디배정용 x
    beam_group: int = -1          # 빔 하위그룹키(같은 값이면 한 빔그룹, -1=단독)
    # 이음줄/붙임줄 플래그 — MusicXML <tied>/<slur> 출력과 편집 표시에 쓴다.
    tie_start: bool = False       # 이 음표에서 붙임줄이 시작(다음 음과 같은 음정)
    tie_stop: bool = False        # 이 음표에서 붙임줄이 끝남
    slur_start: bool = False      # 이 음표에서 이음줄이 시작(다른 음정)
    slur_stop: bool = False       # 이 음표에서 이음줄이 끝남
    accidental: Optional[str] = None  # "sharp"|"flat"|"natural" — 마디 중간에
                                      # 붙어 인식된 임시표(있으면 alter 를 덮어씀).
                                      # MusicXML <accidental> 표시에도 쓰인다.
    is_grace: bool = False        # 꾸밈음(장식음) 여부. True면 박자를 차지하지
                                  # 않는다(마디 박자 합계·검증에서 제외).
    in_triplet: bool = False      # 잇단음표 그룹 멤버(이름은 호환용 — 3잇단만이 아니라 모든 비율)
    tup_actual: int = 3           # [잇단음표 일반화] 비율 actual:normal. 3:2=셋잇단, 5:4, 6:4, 7:4,
    tup_normal: int = 2           #   2:3(복합박자 2잇단), 4:3(복합박자 4잇단)

    # ── [Tier2 확장] HYBRID_OMR_PLAN_v6.md 5.4절 — 기존 필드는 그대로
    # 두고 신규 필드만 추가한다. 전부 기본값(staff_no=1,voice_no=1,
    # chord_group=-1)인 악보는 Tier1과 100% 동일하게 동작해야 한다
    # (회귀 없음이 원칙).
    staff_no: int = 1              # 몇 번째 보표 소속인가(1=위, 2=아래 등)
    voice_no: int = 1              # 같은 보표 내 몇 번째 성부인가
    chord_group: int = -1          # 동시발음(화음) 그룹 id. -1=단독음.
                                  # 같은 chord_group 값을 가진 Element들은
                                  # 같은 시점(같은 기둥)에 울리는 화음이다.
    fermata: bool = False          # 늘임표
    articulations: list = field(default_factory=list)   # [v2.03] [{kind, above, dx, dy}] 악센트·스타카토 등
    acc_dx: Optional[float] = None  # [v2.03] 임시표가 원본에 찍힌 x(머리 가운데 기준 거리, 원본 픽셀)
    # [v2.30 원본 기하] 원본 기둥 (x, y위, y아래)·빔 [(x1,y1,x2,y2,두께)] — 원본 좌표
    src_stem: Optional[tuple] = None
    src_beams: Optional[list] = None
    dynamic: dict = None           # [셈여림] {"key": "mf", "x","y": 원본좌표 또는 None, "auto": 자동인식 여부}
                                    # key는 내부 키(pp/p/mp/mf/f/ff/sfz/fp…)이고 화면 문구와 분리된다.
    disp_x: float = None           # [화면 표시 전용 x] 편집으로 음표를 쪼개 원본 간격이 모자랄 때만
                                    # 쓴다. src_x/src_bbox(원본 이미지 위치)는 그대로 두어 왼쪽
                                    # 원본 패널의 강조 사각형이 실제 음표머리와 계속 맞는다.
    lyrics: dict = field(default_factory=dict)  # [가사] {절번호(0부터): {"text","x","y","h"}}
                                    # PDF 텍스트 레이어에서 인식된 경우만 채워진다. 음표에
                                    # 붙어 있으므로 음표를 지우면 가사도 함께 사라진다.
    double_dot: bool = False       # [v1.10 겹점] True면 점 두 개(dotted도 True). 길이 = 기본 × 1.75
    clef_change: Optional[str] = None  # [v1.10 음자리표 변경] 이 요소부터 그 보표에 적용되는 새 음자리표
                                    # ("G"/"F"/"C" 내부 키). 마디 처음·중간 어디든 올 수 있다.
    synthetic: bool = False        # [편집 기능] 원본에 실제 잉크가 없는,
                                    # "앞/뒤에 삽입"으로 사람이 새로 만든
                                    # 요소인지. True면 음정이 바뀔 때마다
                                    # src_bbox의 y를 오선 좌표 기준으로
                                    # 다시 계산해줘야 한다(editor_app_qt.py
                                    # 참고) — 원본 스캔에 실제로 없는
                                    # 위치이므로 좌표를 자동 추정할 수밖에
                                    # 없다.

    @property
    def note_type(self) -> str:
        return _TYPE.get(self.duration_base, "quarter")

    def quarter_length(self) -> float:
        """4분음표를 1로 하는 길이(마디 박자 검산용). 꾸밈음은 박자를 차지하지 않는다."""
        if self.is_grace:
            return 0.0
        base = {1: 4.0, 2: 2.0, 4: 1.0, 8: 0.5, 16: 0.25, 32: 0.125}.get(
            self.duration_base, 0.0)
        val = base * dot_factor(self)
        if self.in_triplet and self.tup_actual:
            return val * self.tup_normal / self.tup_actual
        return val


@dataclass
class Measure:
    number: int                       # 곡 전체에서 1부터 이어지는 마디 번호
    system: int                       # 소속 보표(줄) 번호
    elements: list = field(default_factory=list)
    first_in_system: bool = False     # 줄바꿈 표시용
    end_barline: str = "normal"       # 이 마디를 닫는 마디선 종류
                                      # "normal"|"repeat_start"|"repeat_end"|"final"
    start_repeat: bool = False        # 이 마디가 도돌이 시작(:|| 다음)인가
    start_repeat_x: Optional[float] = None  # [v2.51] 줄 첫 마디 도돌이 시작의 원본 x(굵은 선 왼쪽 끝)
    ending: Optional[int] = None      # 볼타(1절/2절 등) 번호. 없으면 None.
    target_capacity: Optional[float] = None  # 이 마디가 속한 줄의 실효 박자
        # 용량(4분음표 단위). 표기 박자표와 다르면(예: 표기 4/4 지만 이 줄만
        # 2/4로 인식된 경우) 여기 그 실효값이 들어간다. None 이면 악보 전체
        # 박자표(Score.capacity())를 그대로 쓴다. editing.validate() 가
        # 마디 상태(OK/over/under) 판정에 사용 — 이 필드가 없던 과거에는
        # 모든 마디를 전역 용량 하나로만 검사해, 박자 변화가 인식된 줄
        # 전체가 항상 '박자 부족'으로 잘못 표시되는 문제가 있었다.
    time_sig_change: Optional[tuple] = None  # [Tier2 신규] (beats, beat_type).
        # 이 마디"앞"에 새 박자표 기호를 그려야 하면 그 값이 들어간다(악보
        # 전체 시작 박자표와 같으면 None — 다시 그릴 필요 없음). 시스템
        # (줄) 맨 처음이 아니라 줄 "중간"에서 박자가 바뀌는 경우, 기존
        # 렌더러는 줄 시작에만 박자표를 그리는 전제라 이 변경을 표시할
        # 방법이 없었다(실사용 검증 중 지적받음 — Audiveris 인식 자체는
        # 정확했는데 화면에 반영되지 않았음).
    ts_printed: Optional[tuple] = None  # [v3.40] 원본에 찍힌 박자표 (beats, beat_type, "common"|"cut"|None) —
        # 값이 바뀌지 않아도(줄 첫머리에 다시 찍은 것) 원본처럼 그리고, C·₵ 기호를 숫자로 바꾸지 않는다
    key_change: Optional[int] = None  # [v1.10 조표 바뀜] 이 마디부터 새 조표(fifths). 곡 처음 조표는
        # Score.fifths 이고, 여기에는 곡 중간에서 실제로 바뀌는 마디만 들어간다.

    def quarter_sum(self, capacity: float = 4.0) -> float:
        """마디 전체 박자 합(4분음표 단위) — Tier1(단일 성부) 호환용.

        Tier2(다성부)에서는 이 메서드가 모든 성부를 뒤섞어 합산하므로
        부정확할 수 있다 -- 성부가 둘 이상이면 quarter_sum_by_voice()를
        써야 한다. 단일 성부(voice_no 전부 1)인 경우에는 결과이 동일하다.

        기보법 특례: 온쉼표(점 없는)는 박자표와 무관하게 '온마디 쉼표'다
        (표준 표기 관례). capacity 를 넘겨받으면 그 값으로 계산하고, 생략
        하면 4.0 으로 가정한다(레거시 호출부 호환). 이 특례가 없으면 8/4
        같은 비표준 박자표에서 온쉼표가 실제보다 짧게(4.0) 집계돼, 편집기
        검증이 마디를 '박자 부족'으로 오판한다 — 엔진(omr/engine.py
        grammar_pass·msum)과 반드시 같은 규칙을 써야 한다(실제로 이 불일치
        때문에 엔진과 편집기의 판정이 갈리는 문제가 있었다).
        """
        total = 0.0
        seen_chords = set()
        for e in self.elements:
            if e.chord_group != -1:
                if e.chord_group in seen_chords:
                    continue  # 화음 구성음은 대표(첫 멤버)만 박자에 반영
                seen_chords.add(e.chord_group)
            if e.kind == "rest" and e.duration_base == 1 and not e.dotted:
                total += capacity
            else:
                total += e.quarter_length()
        return total

    def quarter_sum_by_voice(self, capacity: float = 4.0) -> dict:
        """[Tier2] (staff_no, voice_no)별 박자 합. {(staff,voice): 합계}.

        4.3절 (안 B) 마디 배지 설계 근거: 마디 배지 하나는 이 결과 중
        하나라도 capacity와 다르면 '문제 있음'으로 표시하고, 클릭 시
        이 딕셔너리 전체를 펼쳐 보여준다.
        """
        sums: dict = {}
        seen_chords = set()
        for e in self.elements:
            key = (e.staff_no, e.voice_no)
            if e.chord_group != -1:
                if e.chord_group in seen_chords:
                    continue
                seen_chords.add(e.chord_group)
            if e.kind == "rest" and e.duration_base == 1 and not e.dotted:
                sums[key] = sums.get(key, 0.0) + capacity
            else:
                sums[key] = sums.get(key, 0.0) + e.quarter_length()
        return sums


@dataclass
class Arc:
    """이음줄/붙임줄 하나(두 요소를 잇는 호).

    start/stop 은 Element 객체 자체를 가리킨다(인덱스가 아니라 참조).
    편집으로 요소가 삭제되면 그 요소를 참조하는 호는 렌더·내보내기에서
    자동으로 건너뛰도록 각 단계에서 존재 여부를 확인한다.
    """
    start: object            # 시작 Element
    stop: object             # 끝 Element
    kind: str = "slur"       # "tie"(같은 음정) | "slur"(다른 음정)
    above: bool = True       # 호가 위로 볼록한가


@dataclass
class Hairpin:
    """크레센도(<)/디크레센도(>) 하나. start/stop 은 Element 참조(Arc와 동일 방식)."""
    start: object             # 시작 지점에 가장 가까운 요소
    stop: object              # 끝 지점에 가장 가까운 요소
    kind: str = "cresc"       # "cresc" | "decresc"
    auto: bool = False        # Audiveris 자동 인식분인지(“자동 인식 셈여림 모두 지우기” 대상)
    x1: float = None          # 자동 인식분의 원본 좌표(있으면 그대로 그린다) — 편집으로 만든 것은 None
    x2: float = None
    y: float = None
    h: float = None


@dataclass
class Score:
    fifths: int = 0
    beats: int = 4          # 박자표 분자
    beat_type: int = 4      # 박자표 분모
    time_sig_known: bool = True   # False면 박자표를 못 찾아 4/4로 임시 대체한 것
    implied_capacity: Optional[float] = None   # [v3.63] 박자표 없음 + 마디 길이가 대부분 같을 때의 숨은 박자 길이
    measures: list = field(default_factory=list)
    arcs: list = field(default_factory=list)   # Arc 리스트(이음줄/붙임줄)
    hairpins: list = field(default_factory=list)  # Hairpin 리스트(크레센도/디크레센도)
    meta: dict = field(default_factory=dict)  # 곡 정보 {title, lyricist, composer, credit, tempo}

    def capacity(self) -> float:
        """한 마디의 정상 박자 용량(4분음표 단위)."""
        return self.beats * 4.0 / self.beat_type

    def key_at(self, measure_number: int) -> int:
        """[v1.10 조표 바뀜] 그 마디에서 유효한 조표(fifths)."""
        f = self.fifths
        for m in sorted(self.measures, key=lambda mm: mm.number):
            if m.number > measure_number:
                break
            if m.key_change is not None:
                f = m.key_change
        return f


def _attach_hairpins(score, hairpins):
    """검출된 하프핀(크레센도·디크레센도)의 양 끝점을 가장 가까운 요소에 매칭한다.

    이음줄(Arc)과 매칭 방식은 같다(같은 보표에서 x가 가장 가까운 요소를 선택).
    쉼표에도 붙을 수 있으므로(셈여림은 쉼표 구간에도 걸릴 수 있다) 이음줄과
    달리 음표로 한정하지 않고 마디 안의 모든 요소를 후보로 삼는다.
    """
    if not hairpins:
        return
    by_system = {}
    for m in score.measures:
        for e in m.elements:
            by_system.setdefault(m.system, []).append(e)

    for hp in hairpins:
        elems = by_system.get(hp["line"], [])
        if len(elems) < 1:
            continue
        start = min(elems, key=lambda e: abs(e.src_x - hp["x1"]))
        stop = min(elems, key=lambda e: abs(e.src_x - hp["x2"]))
        if start.src_x > stop.src_x:
            start, stop = stop, start
        score.hairpins.append(Hairpin(start=start, stop=stop, kind=hp["kind"]))


def _attach_arcs(score, slurs, elem_by_id=None):
    """검출된 호(slurs)를 실제 음표 쌍(Arc)으로 복원한다.

    [재설계 — 실사용 검증 후] Tier2(omr_bridge)는 이제 slurs 항목에
    "start_id"/"stop_id"(요소 고유 ID)를 직접 담아 보낸다 — merge.py가
    Audiveris의 <slur>/<tie> 태그를 실제로 어느 음표에서 봤는지 이미
    정확히 알고 있으므로, 좌표로 다시 찾을 필요가 전혀 없다.

    이전에는 좌표(x1,x2)로 "그 시스템(줄) 전체(보통 4~5개 마디 전부)
    에서 가장 가까운 음표"를 찾는 근사 매칭이었는데, 코드 기호가 많고
    음표가 촘촘한 가요 악보(실사용 "여정" 샘플)에서 여러 마디의 음표
    x좌표가 우연히 가까워 **엉뚱한 마디의 음표에 잘못 매칭**되는
    사고가 실제 사용자 환경에서 결정적으로 재현됐다 — 좌표 근사
    매칭이라는 설계 자체의 근본적 한계였다.

    Tier1(구엔진, omr/pipeline.py)은 아직 좌표 형태("line"/"x1"/"x2")로
    보내므로, 그 경우에는 기존 좌표 근사 매칭으로 폴백한다(회귀 없음).
    """
    if not slurs:
        return
    elem_by_id = elem_by_id or {}
    # [v2.24] 순서 보정은 '마디 순서'가 먼저다 — 줄 끝에서 다음 줄로 이어지는 붙임줄은 끝 음이 x가 더 작아
    # x만 비교하면 거꾸로 뒤집혔다("교회" 53→54마디 온음표 붙임줄이 54→53이 되어 엉뚱한 반쪽 호)
    order = {}
    for mi, m in enumerate(score.measures):
        for e in m.elements:
            order[id(e)] = mi

    def _make_arc(start, stop, src_above=None, curves=None):
        if start is stop or start is None or stop is None:
            return
        ka = (order.get(id(start), 0), start.src_x)
        kb = (order.get(id(stop), 0), stop.src_x)
        curves = dict(curves or {})
        if ka > kb:                            # 순서 보정(마디 순서 → 같은 마디면 x)
            start, stop = stop, start
            curves["curve_a"], curves["curve_b"] = curves.get("curve_b"), curves.get("curve_a")
        if any(a.start is start and a.stop is stop for a in score.arcs):
            return                              # [v2.24] 같은 붙임줄이 두 번 오면 하나만(53마디)
        same = (start.step == stop.step and start.octave == stop.octave)
        kind = "tie" if same else "slur"
        if kind == "tie":
            start.tie_start = True
            stop.tie_stop = True
        else:
            start.slur_start = True
            stop.slur_stop = True
        arc_ = Arc(start=start, stop=stop, kind=kind, above=True if src_above is None else bool(src_above))
        if src_above is not None:
            arc_.src_above = bool(src_above)      # [v2.03] 원본 방향을 알면 그대로 그린다
        for k in ("curve", "curve_a", "curve_b"):  # [v2.30] 원본 곡선
            if curves.get(k):
                setattr(arc_, "src_" + k, tuple(curves[k]))
        score.arcs.append(arc_)

    # ID 기반(정확) 항목과 좌표 기반(Tier1 폴백) 항목을 분리
    id_based = [s for s in slurs if "start_id" in s]
    coord_based = [s for s in slurs if "start_id" not in s]

    for s in id_based:
        start = elem_by_id.get(s["start_id"])
        stop = elem_by_id.get(s["stop_id"])
        _make_arc(start, stop, s.get("above"), {k: s.get(k) for k in ("curve", "curve_a", "curve_b")})

    if not coord_based:
        return
    # --- 이하 Tier1용 좌표 근사 매칭(기존 로직 그대로 유지) ---
    by_system = {}
    for m in score.measures:
        for e in m.elements:
            if e.kind == "note":
                by_system.setdefault(m.system, []).append(e)

    for s in coord_based:
        notes = by_system.get(s["line"], [])
        if len(notes) < 2:
            continue
        mid_x = (s["x1"] + s["x2"]) / 2.0
        left_notes = [e for e in notes if e.src_x <= mid_x]
        right_notes = [e for e in notes if e.src_x > mid_x]
        if left_notes:
            start = min(left_notes, key=lambda e: abs(e.src_x - s["x1"]))
        else:
            start = min(notes, key=lambda e: abs(e.src_x - s["x1"]))
        if right_notes:
            stop = min(right_notes, key=lambda e: abs(e.src_x - s["x2"]))
        else:
            stop = min(notes, key=lambda e: abs(e.src_x - s["x2"]))
        _make_arc(start, stop)


def build_score_model(key, time_signature, elements, pitches, barlines,
                      barline_types=None, slurs=None, voltas=None, hairpins=None,
                      sys_capacity=None, time_sig_changes=None,
                      dynamics=None, wedges=None, key_changes=None, harmonies=None,
                      articulations=None, acc_dx=None, src_geo=None) -> Score:
    """인식 결과를 구조화된 Score 모델로 재구성한다(결정론적).

    time_sig_changes: [Tier2 신규] {마디번호: (beats, beat_type)} — 그
        마디부터 표기 박자표가 실제로 바뀌는 지점(중간에 여러 번 바뀔 수
        있음). 시스템(줄) 맨 처음이 아닌 마디에도 정확히 표시된다 —
        기존에는 줄 시작에만 박자표를 그릴 수 있어 줄 중간의 박자 변화가
        화면에 반영되지 않는 한계가 있었다(실사용 검증 중 발견).

    barline_types: {보표: {마디선x: 종류}} — detect_barline_types() 결과.
        각 마디의 '끝 마디선' 종류(normal/repeat_start/repeat_end/final)를 기록해
        편집 악보 렌더에서 도돌이표·종료선을 그릴 수 있게 한다.
    slurs: [{line, x1, y1, x2, y2, above}, ...] — detect_and_remove_slurs() 결과.
        각 호의 끝점을 음표에 매칭해 score.arcs(붙임줄/이음줄)로 복원한다.
    voltas: [{line, x1, x2, number}, ...] — detect_voltas() 결과(정규화 배율).
        각 마디의 x범위가 어느 볼타 구간에 들어가는지 확인해 Measure.ending 에
        번호를 매긴다.
    hairpins: [{line, x1, x2, y, kind}, ...] — detect_hairpins() 결과.
        각 쐐기의 끝점을 요소에 매칭해 score.hairpins(크레센도/디크레센도)로 복원한다.
    sys_capacity: {줄: 실효 용량} — 엔진이 계산한 줄별 박자 용량(engine.py
        EngineResult.sys_capacity). 표기 박자표와 다른 줄(박자 변화 인식)의
        마디에 Measure.target_capacity 로 심어, 편집기 검증이 그 줄만은
        올바른 용량으로 검사하게 한다. 생략하면 모든 마디가 전역 박자표
        용량 하나로 검사된다(과거 동작과 동일).
    """
    fifths = key_to_fifths(key or 0)
    ts_known = bool(time_signature and time_signature[0] and time_signature[1])
    if ts_known:
        beats_num, beat_type = time_signature
    else:
        beats_num, beat_type = 4, 4     # 미인식 시 기본값(편집에서 수정 가능)

    score = Score(fifths=fifths, beats=beats_num, beat_type=beat_type,
                 time_sig_known=ts_known)
    alter = _alter_map(fifths)
    barline_types = barline_types or {}
    voltas_by_line = {}
    for v in (voltas or []):
        voltas_by_line.setdefault(v["line"], []).append(v)

    n_systems = (max(e[0] for e in elements) + 1) if elements else 0
    measure_no = 0
    # [재설계] 요소 고유 ID → 생성된 Element 매핑. 좌표 재탐색 없이
    # slurs(elem_id 기반)를 정확히 연결하기 위함(아래 _attach_arcs 참고).
    elem_by_id = {}

    for line in range(n_systems):
        # 이 줄의 요소만 뽑아 '각 음표의 기둥 x'(el[6]) 순으로 정렬.
        # 이음줄로 여러 음표가 한 연결요소가 되면 bbox x(el[1])가 모두 같아지므로,
        # bbox 기준으로 마디를 나누면 마디선을 넘어간 음표까지 앞 마디에 들어간다.
        # 각 음표의 실제 기둥 위치로 판단해야 마디 경계가 정확하다.
        def _ex(el):
            return el[6] if len(el) > 6 else el[1]

        line_items = sorted(
            [(el, p) for el, p in zip(elements, pitches) if el[0] == line],
            key=lambda t: _ex(t[0]))
        if not line_items:
            continue

        bars = sorted(barlines.get(line, []))
        types_l = barline_types.get(line, {})
        first_x = _ex(line_items[0][0])
        # 마디 경계 x좌표: [줄 시작] + 마디선들 + [끝]
        bounds = [first_x - 10] + bars
        last_x = _ex(line_items[-1][0])
        if not bars or last_x >= bars[-1]:
            bounds = bounds + [10 ** 9]   # 마지막 마디선 이후 요소도 담기게 끝을 크게

        first_in_system = True
        for m in range(len(bounds) - 1):
            lo, hi = bounds[m], bounds[m + 1]
            in_measure = [(el, p) for el, p in line_items if lo <= _ex(el) < hi]
            if not in_measure:
                continue
            measure_no += 1
            # 이 마디를 닫는 마디선(=경계 hi)의 종류
            end_kind = types_l.get(hi, "normal") if hi in types_l else "normal"
            # 이 마디를 여는 마디선(=경계 lo)이 도돌이 시작인가
            start_rep = types_l.get(lo) == "repeat_start" or \
                (first_in_system and types_l.get(-1) == "repeat_start")      # [v2.51] 줄 첫 마디
            start_rep_x = types_l.get(-2) if (first_in_system and types_l.get(-1) == "repeat_start") else None
            # [Tier2 보강] sys_capacity는 두 가지 키 형태를 지원한다:
            #   - 정수 line: 그 줄(시스템) 전체에 동일 용량 적용(기존 방식,
            #     하위호환 유지 — 546개 배치의 대다수 케이스가 이에 해당).
            #   - (line, measure_no) 튜플: 그 마디 하나만 다른 용량(한 줄
            #     안에서 박자표가 바뀌는 드문 경우, 데모곡 검증 중 발견).
            #   튜플 키가 있으면 우선, 없으면 정수 키로 폴백한다.
            sc = sys_capacity or {}
            tcap = sc.get((line, measure_no), sc.get(line))
            tsc = (time_sig_changes or {}).get(measure_no)
            kch = (key_changes or {}).get(measure_no)
            measure = Measure(number=measure_no, system=line,
                              first_in_system=first_in_system,
                              end_barline=end_kind, start_repeat=start_rep, start_repeat_x=start_rep_x,
                              target_capacity=tcap, time_sig_change=tsc,
                              key_change=kch)
            if kch is not None:                 # [v1.10] 이 마디부터 새 조표의 음높이 규칙
                alter = _alter_map(kch)
            first_in_system = False

            # 이 마디가 어느 볼타(1절/2절 등) 구간에 들어가는지 확인.
            # 훅 폭 등으로 브래킷 경계가 마디선과 몇 px 어긋날 수 있으므로,
            # 마디 폭의 20%(최소 8px)를 허용 오차로 두고 '마디 전체가 거의
            # 볼타 구간 안에 있는지'를 기준으로 판정한다.
            for v in voltas_by_line.get(line, []):
                tol = max(8.0, (hi - lo) * 0.2) if hi < 10 ** 8 else 8.0
                if lo >= v["x1"] - tol and (hi < 10 ** 8 and hi <= v["x2"] + tol
                                            or hi >= 10 ** 8):
                    measure.ending = v["number"]
                    measure.ending_ys = sorted(v.get("ys") or [])    # [v2.54] 원본 꺾쇠 높이(파트마다)
                    break

            for item, pitch in in_measure:
                l, x, y, w, h, beat = item[0], item[1], item[2], item[3], item[4], item[5]
                sx_note = item[6] if len(item) > 6 else x
                gkey = item[7] if len(item) > 7 else -1
                acc = item[8] if len(item) > 8 else None
                is_grace = item[9] if len(item) > 9 else False
                in_triplet = item[10] if len(item) > 10 else False
                # [Tier2] 이하 4개 필드는 omr_bridge.merge가 채워주는 신규
                # 항목이다. 기존(omr/engine.py) 11개 필드 튜플에는 없으므로
                # 길이 체크로 안전하게 기본값 처리한다 — Tier1 회귀 없음.
                staff_no = item[11] if len(item) > 11 else 1
                voice_no = item[12] if len(item) > 12 else 1
                chord_group = item[13] if len(item) > 13 else -1
                fermata = item[14] if len(item) > 14 else False
                tie_start_v = item[15] if len(item) > 15 else False
                elem_id_v = item[16] if len(item) > 16 else None
                tup_a = item[17] if len(item) > 17 else 3
                tup_n = item[18] if len(item) > 18 else 2
                n_dots = item[19] if len(item) > 19 else (1 if beat < 0 else 0)
                clef_ch = item[20] if len(item) > 20 else None
                dur = abs(int(beat))
                dotted = beat < 0
                bbox = (int(x), int(y), int(w), int(h))
                # pitch is None: 음표 머리는 찾았지만 오선 대비 위치가 음정
                # 판정 허용오차 밖이라 정확한 음이름을 못 정한 경우다(예: 이미지
                # 손상·기울어짐이 심한 발췌 스캔). 크래시 대신 쉼표로 대체해
                # 마디 슬롯은 보존하고, 사용자가 편집기의 '음표/쉼표' 전환으로
                # 직접 바로잡을 수 있게 한다.
                if pitch == -1 or pitch is None:
                    new_el = Element(
                        kind="rest", duration_base=dur, dotted=dotted,
                        src_bbox=bbox, src_x=int(sx_note), beam_group=gkey,
                        staff_no=staff_no, voice_no=voice_no,
                        chord_group=chord_group, fermata=bool(fermata),
                        # 잇단음표 안의 쉼표(예: 셋잇단 8분 중 가운데가 쉼표)도 그룹의 일원
                        in_triplet=bool(in_triplet), tup_actual=tup_a, tup_normal=tup_n,
                        double_dot=bool(dotted and n_dots >= 2), clef_change=clef_ch)
                    measure.elements.append(new_el)
                    if elem_id_v is not None:
                        elem_by_id[elem_id_v] = new_el
                else:
                    name = _PITCH_NAMES[pitch]
                    step, octave = name[0], int(name[1:])
                    # 인식된 임시표가 있으면 조표 기반 alter 대신 그 값을 쓴다
                    # (같은 마디 안에서 이 음부터 마디 끝까지 적용되는 게 표준
                    # 표기 규칙이지만, 지금은 '이 음표 하나'에만 반영한다 —
                    # 같은 마디에서 같은 음이 반복될 때 자동 전파는 향후 보강).
                    note_alter = {"sharp": 1, "flat": -1, "natural": 0}.get(
                        acc, alter.get(step, 0))
                    new_el = Element(
                        kind="note", duration_base=dur, dotted=dotted,
                        step=step, octave=octave, alter=note_alter,
                        src_bbox=bbox, src_x=int(sx_note), beam_group=gkey,
                        accidental=acc, is_grace=bool(is_grace),
                        in_triplet=bool(in_triplet), tup_actual=tup_a, tup_normal=tup_n,
                        staff_no=staff_no, voice_no=voice_no,
                        chord_group=chord_group, fermata=bool(fermata),
                        tie_start=bool(tie_start_v),
                        double_dot=bool(dotted and n_dots >= 2), clef_change=clef_ch)
                    measure.elements.append(new_el)
                    if elem_id_v is not None:
                        elem_by_id[elem_id_v] = new_el
            score.measures.append(measure)

    _attach_arcs(score, slurs, elem_by_id)  # 검출된 호 → 붙임줄/이음줄 관계로 복원
    # [셈여림·쐐기] omr_bridge가 요소 ID로 짝지어 넘긴 자동 인식분을 그대로 연결한다.
    for dy in (dynamics or []):
        e = elem_by_id.get(dy["elem_id"])
        if e is not None:
            e.dynamic = {"key": dy["key"], "x": dy.get("x"), "y": dy.get("y"), "auto": True}
    # [v3.00] Audiveris가 읽은 기타 코드 — 원본 자리는 모름(레이아웃이 보표 위에 둔다)
    if harmonies:
        from score import chords as _ch
        for hr in harmonies:
            e = elem_by_id.get(hr["elem_id"])
            h_ = _ch.parse(hr["text"]) if e is not None else None
            if h_:
                h_.update({"x": None, "y": None, "src": "omr"})
                e.harmony = h_
    for wg in (wedges or []):
        a, b = elem_by_id.get(wg["start_id"]), elem_by_id.get(wg["stop_id"])
        if a is not None and b is not None:
            score.hairpins.append(Hairpin(start=a, stop=b, kind=wg["kind"], auto=True,
                                          x1=wg.get("x1"), x2=wg.get("x2"),
                                          y=wg.get("y"), h=wg.get("h")))
    _attach_hairpins(score, hairpins)  # 검출된 쐐기 → 크레센도/디크레센도로 복원
    for ar in (articulations or []):            # [v2.03]
        e = elem_by_id.get(ar["elem_id"])
        if e is not None:
            e.articulations.append({k: ar[k] for k in ("kind", "above", "dx", "dy")})
    for eid, (stem_, beams_) in (src_geo or {}).items():      # [v2.30] 원본 기둥·빔
        e = elem_by_id.get(eid)
        if e is not None:
            e.src_stem = tuple(stem_)
            e.src_beams = [tuple(b) for b in beams_] if beams_ else None
    for eid, dx in (acc_dx or {}).items():
        e = elem_by_id.get(eid)
        if e is not None and e.accidental:
            e.acc_dx = dx
    return score


def build_from_context(ctx, barlines, barline_types=None) -> Score:
    """파이프라인 ctx(+detect_barlines 결과)에서 곧바로 모델을 만든다(편의 함수)."""
    rec = ctx["recognition"]
    sys_cap = ctx.get("measure_info", {}).get("sys_capacity")
    score = build_score_model(rec["key"], rec["time_signature"],
                             rec["elements"], rec["pitches"], barlines,
                             barline_types, ctx.get("slurs"), ctx.get("voltas"),
                             ctx.get("hairpins"), sys_capacity=sys_cap,
                             time_sig_changes=rec.get("time_sig_changes"),
                             dynamics=ctx.get("dynamics"), wedges=ctx.get("wedges"),
                             key_changes=rec.get("key_changes"),
                             articulations=ctx.get("articulations"), acc_dx=ctx.get("acc_dx"),
                             src_geo=ctx.get("src_geo"), harmonies=ctx.get("harmonies"))
    score.ts_symbol = rec.get("time_symbol")   # "common" = C 기호 4/4
    # [v1.20 MusicXML 내보내기] 파트·보표 구성을 Score에 함께 둔다(내보내기가 화면 없이도 쓸 수 있게)
    score.staff_parts = dict(ctx.get("staff_parts") or {})    # {줄: [보표별 논리 파트]}
    score.staff_clefs = dict(ctx.get("staff_clefs") or {})    # {줄: [보표별 줄 머리 음자리표]}
    score.part_names = list(ctx.get("part_names") or [])
    return score


def attach_lyrics(score, tokens, interline):
    """[가사 — PDF 인식분만] extract_lyric_tokens()의 토큰을 같은 시스템(줄)의 가장
    가까운 음표에 붙인다. 쉼표·꾸밈음에는 붙이지 않고, 한 음표의 같은 절에 토큰이
    둘 이상 몰리면 더 가까운 쪽만 남긴다. (붙인 개수, 못 붙인 개수)를 돌려준다."""
    by_line = {}
    for m in score.measures:
        for e in m.elements:
            if e.kind == "note" and not e.is_grace:
                # 가사는 그 가사 바로 위 보표의 음표에만 붙인다(노래+피아노 악보)
                by_line.setdefault((m.system, getattr(e, "staff_no", 1)), []).append(e)
    attached = missed = 0
    leftover = []
    for tk in tokens:
        cands = by_line.get((tk["line"], tk.get("staff", 1)), [])
        best, best_d = None, None
        for e in cands:
            if e.src_bbox:
                cx = e.src_bbox[0] + e.src_bbox[2] / 2.0
            else:
                cx = e.src_x
            dd = abs(cx - tk["x"])
            if best_d is None or dd < best_d:
                best, best_d = e, dd
        if best is None or best_d > 1.6 * interline:
            leftover.append(tk)
            continue
        prev = best.lyrics.get(tk["row"])
        if prev is not None and prev["d"] <= best_d:
            leftover.append(tk)
            continue
        if prev is not None:
            leftover.append(prev["_tk"])
        best.lyrics[tk["row"]] = {"text": tk["text"], "x": tk["x"], "y": tk["y"],
                                  "h": tk["h"], "d": best_d, "_tk": tk}
        if tk.get("hdy") is not None:
            best.lyrics[tk["row"]].update(hdy=tk["hdy"], hw=tk.get("hw"))
        attached += 1
    # [v3.63] 낭송음(그레고리오 성가의 온음표 하나에 여러 낱말 — "그리스도와 함께 다시 살게 되었으니"):
    # 붙지 못한 음절이 온음표(또는 다음 음까지 6칸 넘게 빈 음) 오른쪽 빈자리에 있으면 그 음의 가사에 이어 붙인다.
    # 예전엔 음마다 가장 가까운 음절 하나만 남아 "그 스 와 함"처럼 나머지가 사라졌다.
    reci = {}
    for tk in leftover:
        cands = by_line.get((tk["line"], tk.get("staff", 1)), [])

        def cx_(e):
            return (e.src_bbox[0] + e.src_bbox[2] / 2.0) if e.src_bbox else e.src_x
        left = [e for e in cands if cx_(e) <= tk["x"]]
        right = [e for e in cands if cx_(e) > tk["x"]]
        if not left:
            missed += 1
            continue
        a = max(left, key=cx_)
        b = min(right, key=cx_) if right else None
        gap = (cx_(b) - cx_(a)) if b is not None else 1e9
        # 같은 낱말의 이어지는 글자(음표 하나에 "그리"·"스도"처럼 두 글자) — 붙은 음절 바로 옆이면 그 음에
        near = None
        for e in cands:
            ly = e.lyrics.get(tk["row"])
            t0 = ly.get("_tk") if ly else None
            if t0 is not None and abs(t0["x"] - tk["x"]) < 1.25 * tk["h"]:
                if near is None or abs(t0["x"] - tk["x"]) < abs(near[1]["x"] - tk["x"]):
                    near = (e, t0)
        if near is not None:
            reci.setdefault((id(near[0]), tk["row"]), (near[0], []))[1].append(tk)
            attached += 1
        elif (a.duration_base == 1 or gap > 6 * interline) and tk["x"] - cx_(a) < gap - 1.0 * interline \
                and tk["row"] in a.lyrics:
            reci.setdefault((id(a), tk["row"]), (a, []))[1].append(tk)
            attached += 1
        else:
            missed += 1
    for (_k, row), (a, tks) in reci.items():
        base = a.lyrics[row].get("_tk")
        allt = sorted(([base] if base else []) + tks, key=lambda t: t["x"])

        def w_(t):
            return t["h"] * (0.95 if any("\uac00" <= c <= "\ud7a3" for c in t["text"]) else 0.55 * len(t["text"]))
        text = allt[0]["text"]
        for p_, t in zip(allt, allt[1:]):
            gap_ = (t["x"] - w_(t) / 2) - (p_["x"] + w_(p_) / 2)
            text += (" " if gap_ > 0.45 * t["h"] else "") + t["text"]
        x0 = allt[0]["x"] - w_(allt[0]) / 2
        x1 = allt[-1]["x"] + w_(allt[-1]) / 2
        a.lyrics[row].update({"text": text, "x": (x0 + x1) / 2})
    for e in (x for v in by_line.values() for x in v):
        for ly in e.lyrics.values():
            ly.pop("_tk", None)
    return attached, missed


def raw_quarter(e) -> float:
    """잇단 비율을 적용하기 '전'의 길이(4분음표=1)."""
    base = {1: 4.0, 2: 2.0, 4: 1.0, 8: 0.5, 16: 0.25, 32: 0.125}.get(e.duration_base, 0.0)
    return base * dot_factor(e)


def dot_factor(e) -> float:
    """[v1.10 겹점] 점 개수에 따른 길이 배수: 없음 1, 점 1.5, 겹점 1.75."""
    if not e.dotted:
        return 1.0
    return 1.75 if getattr(e, "double_dot", False) else 1.5


def tuplet_groups(elements):
    """[잇단음표 일반화] 한 마디의 요소들에서 잇단음표 그룹을 찾는다.
    반환: [(구성 요소 리스트, 완전한 그룹인지), ...]
    같은 보표에서 같은 비율로 이어지는 요소를 모은 뒤, 그 구간의 가장 짧은 음가를
    단위로 'actual × 단위' 길이가 찰 때마다 한 그룹으로 끊는다 — 그래서 셋잇단 두 개가
    붙어 있어도(3+3) 따로 묶이고, 4분+8분 같은 섞인 셋잇단도 한 그룹이 된다.
    화음 구성음은 대표 하나만 길이에 넣는다. 성부(voice) 번호는 Audiveris가 그룹 안에서도
    제각각 매기는 경우가 있어(실측) 구분하지 않는다."""
    by_staff = {}
    for e in elements:
        by_staff.setdefault(getattr(e, "staff_no", 1), []).append(e)
    groups = []
    for els in by_staff.values():
        slots, seen = [], set()
        for e in els:
            cg = getattr(e, "chord_group", -1)
            if cg != -1:
                if cg in seen:
                    continue
                seen.add(cg)
            slots.append(e)
        i = 0
        while i < len(slots):
            e = slots[i]
            if not e.in_triplet or e.is_grace:
                i += 1
                continue
            r = (e.tup_actual, e.tup_normal)
            # [다성부 보정 — 실사용 검증] 같은 자리에 다른 성부의 일반 음표가 끼어 있어도(예: 위
            # 성부 4분음표가 셋잇단 첫 음과 같은 x) 그룹이 끊기지 않게, '그룹에 들어온 성부가 아닌'
            # 일반 음표는 건너뛴다. 같은 성부의 일반 음표가 나오면 그룹이 끝난다.
            run, voices, j = [], set(), i
            while j < len(slots):
                x = slots[j]
                if x.in_triplet and not x.is_grace and (x.tup_actual, x.tup_normal) == r:
                    run.append(x)
                    voices.add(x.voice_no)
                elif x.voice_no in voices or x.is_grace:
                    break
                j += 1
            unit = min(raw_quarter(x) for x in run) or 0.5
            target = r[0] * unit
            cur, acc = [], 0.0
            for x in run:
                cur.append(x)
                acc += raw_quarter(x)
                if acc >= target - 1e-6:
                    groups.append((cur, abs(acc - target) < 1e-6))
                    cur, acc = [], 0.0
            if cur:
                groups.append((cur, False))
            i = j
    return groups
