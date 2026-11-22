# score_editing.py
# ---------------------------------------------------------------------------
# Score 모델을 '수정'하고 '검증'하는 순수 로직 계층(GUI 비의존 → 단독 테스트 가능).
#
# 편집기 화면(editor_app)은 사용자의 클릭·버튼을 여기 함수 호출로 바꾸기만 한다.
# 모든 수정은 Score 모델을 직접 바꾸고, 그 즉시 validate() 로 마디 상태가 갱신된다.
#
# 핵심 편집 조작
#   - set_duration / toggle_dot         : 박자(음길이) 교정
#   - shift_step / shift_octave / cycle_accidental : 음높이·임시표 교정
#   - toggle_note_rest                  : 음표 ↔ 쉼표
#   - delete_element / insert_after     : 요소 추가·삭제
#   - set_beam_group_duration           : ★ 빔 그룹을 한 박자로 일괄 지정(가장 잦은 교정)
# ---------------------------------------------------------------------------
import dataclasses
from .model import Element, Measure, Score, Arc, Hairpin, raw_quarter

STEPS = ["C", "D", "E", "F", "G", "A", "B"]
DURATION_CHOICES = [1, 2, 4, 8, 16, 32]     # 온/2/4/8/16/32분


# --------------------------------------------------------------------- 검증
def _capacity_of(score: Score, measure: Measure) -> float:
    """이 마디를 검사할 때 쓸 용량. 박자 변화가 인식된 줄은 그 줄의 실효
    용량(Measure.target_capacity)을, 그 외에는 악보 전체 박자표 용량을 쓴다.

    이 구분이 없던 과거에는 모든 마디를 전역 용량 하나로만 검사해, 표기
    박자표와 다르게 인식된 줄(예: 4/4 곡 안의 2/4 구간) 전체가 항상 '박자
    부족'으로 잘못 표시됐다(엔진 자체 판정은 정확했는데 편집기 검증만
    그 정보를 몰랐던 것 — 실제로 확인된 버그).
    """
    if measure.target_capacity is not None:
        return measure.target_capacity
    # [버그 수정 — 시험 악보에서 발견] 곡 중간에 박자가 바뀌면(4/4 → 6/8) 그 뒤 마디들도
    # 바뀐 박자로 검사해야 한다. 예전엔 그 줄에 별도 용량이 없으면 곡 전체 박자만 봐서,
    # 6/8 구간이 모두 '박자 부족'으로 잘못 표시됐다.
    ts = None
    for mm in score.measures:
        if getattr(mm, "time_sig_change", None):
            ts = mm.time_sig_change
        if mm is measure:
            break
    if ts:
        return ts[0] * 4.0 / ts[1]
    return score.capacity()


def _real_voice_pool(measure: Measure, by_voice: dict, capacity: float = None) -> dict:
    """성부별 박자 합(보정 후). 아래 _real_voice_pool_core의 보정에 더해 [생략 쉼표 관례]를
    적용한다: 같은 보표에 마디를 정확히 채운 성부가 있으면, 그보다 짧은 다른 성부는 앞이나
    뒤의 쉼표가 생략된 것으로 본다(일반 기보 관례 — "봄이 오면" 22마디 왼손 위 성부는 2박째부터
    시작해 2박뿐인데, 원본도 1박째 쉼표를 생략했다). 넘치는 성부는 그대로 오류로 남긴다."""
    pool = _real_voice_pool_core(measure, by_voice, capacity)
    if capacity is None:
        return pool
    full = [k for k, v in pool.items() if abs(v - capacity) < 0.01]
    if not full:
        return pool
    # [v1.21 생략 쉼표 판정 재설계 — 실사용] 짧은 성부를 '쉼표를 생략한 성부'로 볼지는, 그 성부의
    # 음들이 같은 줄(시스템)의 '마디를 채운 성부' 음들과 같은 박 위치에 맞아떨어지는지로 정한다.
    # 보표가 달라도 같은 줄의 세로 정렬은 같으므로 다른 보표의 성부도 기준이 된다.
    #   · "봄이 오면" 22마디 왼손(2박째부터 시작), "기도의 능력" 2마디 왼손(3박째 2분음표 하나),
    #     같은 마디 오른손 아래 성부(1~2박 뒤 쉼표 생략) → 모두 맞아떨어짐 → 정상
    #   · "교회" 7마디 오른손 아래 성부(가운데 음 누락: 3박 자리의 2분음표가 2박째로 계산됨)
    #     → 어긋남 → 오류. 점이 빠진 경우도 뒤 음이 어긋나 오류로 남는다.
    # (v1.20은 '마디 처음보다 늦게 시작하는 성부만' 봐줘서 뒤쪽 쉼표 생략과 다른 보표 기준을 놓쳤다)
    ok = _omitted_rest_voices(measure, pool, by_voice, capacity, full)
    return {k: (capacity if (v < capacity - 0.01 and k in ok) else v) for k, v in pool.items()}


def _omitted_rest_voices(measure, pool, by_voice, capacity, full_keys):
    from .musicxml import _slots, _place_voices, DIVISIONS
    cap = int(round(capacity * DIVISIONS))
    by_staff = {}
    for e in measure.elements:
        by_staff.setdefault(e.staff_no, {}).setdefault(e.voice_no, []).append(e)
    placed = {}                                    # (staff, voice) → 슬롯(시작 위치 포함)
    for sn, voices in by_staff.items():
        vs = {v: _slots(es) for v, es in voices.items()}
        _place_voices(vs, cap)
        for v, sl in vs.items():
            placed[(sn, v)] = sl
    # 목록에서 빠진 성부: '이어지는 성부'로 합쳐진 것(기준으로 씀)과, 혼자뿐인 음이라 버려진 것(그
    # 보표는 내용이 빠진 셈이라 생략 쉼표로 봐주지 않는다 — 데모곡 11마디: 오른손 4분음표 셋이
    # 성부 1·2·3에 하나씩 흩어져 버려졌는데, 남은 2박짜리 성부를 '생략 쉼표'로 봐주던 문제)
    xs_of = {}
    for e in measure.elements:
        xs_of.setdefault((e.staff_no, e.voice_no), []).append(e.src_x)
    merged_keys, dropped_staffs = set(), set()
    for kb in by_voice:
        if kb in pool:
            continue
        cont = any(ka[0] == kb[0] and abs(pool[ka] - capacity) < 0.01
                   and abs(by_voice.get(ka, 0) + by_voice[kb] - capacity) < 0.01
                   and xs_of.get(ka) and xs_of.get(kb) and max(xs_of[ka]) < min(xs_of[kb])
                   for ka in pool)
        if cont:
            merged_keys.add(kb)
        else:
            dropped_staffs.add(kb[0])
    anchor_keys = set(full_keys) | merged_keys
    anchors = [s for k in anchor_keys for s in placed.get(k, []) if not s["grace"]]
    ok = set()
    ink = getattr(measure, "unexplained_heads", None) or {}
    for k, v in pool.items():
        if v >= capacity - 0.01 or k not in placed:
            continue
        if k[0] in dropped_staffs:
            continue
        if ink.get(k[0]):
            # 원본의 이 보표에 인식 안 된 음표 머리가 있다 → 빈 박은 '생략'이 아니라 '누락'
            # ("교회" 9마디 오른손 1박: 박 위치는 맞지만 원본엔 음표가 있었다)
            continue

        sl = [s for s in placed[k] if not s["grace"]]
        if not sl:
            continue

        def _anchor(s):
            # 같은 자리: 머리 폭 1.1배 이내(2도 때문에 옆으로 비킨 머리 포함)
            near = [a for a in anchors if abs(a["x"] - s["x"]) <= 1.1 * max(a["w"], s["w"])]
            return min(near, key=lambda a: abs(a["x"] - s["x"])) if near else None
        # 시작 위치: 첫 음과 같은 자리의 기준 음 박 위치(다른 보표 포함). 없으면 보표 안 배치값.
        a0 = _anchor(sl[0])
        pos = a0["on"] if a0 is not None else sl[0]["on"]
        matched, bad = 0, False
        for s in sl:
            a = _anchor(s)
            if a is not None:
                matched += 1
                if abs(a["on"] - pos) > 1:
                    bad = True
                    break
            pos += s["dur"]
        if matched and not bad and pos <= cap:
            ok.add(k)
    return ok


def _real_voice_pool_core(measure: Measure, by_voice: dict, capacity: float = None) -> dict:
    """[공통 헬퍼] "유령 성부"(요소 1개뿐) 문제를 다룬다.

    [재설계, 실사용 검증 후] 처음엔 요소 1개짜리 성부를 "무조건 무시"
    했는데, 이러면 정반대의 사고가 난다 — 실제로 존재하는 쉼표가
    Audiveris 인식 과정에서 실수로 별도 성부(voice)로 떨어져 나간
    경우(예: 화음은 voice5, 그 화음 뒤 쉼표만 voice6으로 분리), 그
    쉼표를 통째로 무시해버려서 진짜 존재하는 마디 내용이 계산에서
    빠지고, 오히려 정상인 마디를 "박자 부족"으로 잘못 표시했다(실사용
    검증 중 발견 — 왼손 쉼표가 무시되어 3/4로 잘못 표시됨).

    이제는: 같은 보표 안에서 요소 1개짜리 성부(orphan)를, 그 보표의
    "진짜 성부"(요소 2개 이상)에 합쳐봤을 때 **정확히 capacity와
    맞아떨어지면 합치고**, 안 맞으면(즉 정말 무관한 유령 데이터일
    가능성이 높으면) 기존처럼 무시한다. "우연히 딱 맞아떨어질" 확률은
    낮으므로, 이 판별 기준은 두 상황(진짜 분리된 요소 vs 진짜 유령)을
    실질적으로 잘 구분해낸다.

    [추가 개선 — 실사용 검증 후] 한 보표가 유령 성부 하나가 아니라
    **여러 개로(voice 1,2,3,4처럼) 쪼개지는 경우**도 실사용 중 발견됐다
    (셋잇단음표+일반박이 섞인 복잡한 리듬에서 Audiveris가 음표마다
    다른 성부를 매기는 극단적 사례). 이런 경우 orphan을 하나씩만
    시도하면 "부분적으로만 맞는 조합"을 못 찾아 전부 버려진다 — 먼저
    "그 보표의 real+모든 orphan을 통째로 다 합친 값"이 정확히 목표와
    맞는지 확인하고, 맞으면 그걸 채택한다(안 맞을 때만 기존처럼
    orphan 하나씩 개별 시도로 폴백).
    """
    counts = {}
    for e in measure.elements:
        key = (e.staff_no, e.voice_no)
        counts[key] = counts.get(key, 0) + 1
    # [성부 쪼개짐 보정 — 잇단음표 시험에서 발견] Audiveris가 한 줄 선율을 성부 둘
    # 이상으로 나누는 경우(예: 6잇단은 성부1, 뒤의 4분·2분은 성부2)에는 성부마다
    # 요소가 여러 개라 '유령 성부' 규칙에 안 걸린다. 같은 보표의 성부들이 하나도
    # 박자에 안 맞는데 '전부 합치면' 정확히 맞으면 한 성부로 본다.
    if capacity is not None:
        by_staff = {}
        # [v2.70 P-1] 원본에 숨은 음 증거(인식 안 된 머리)가 있는 보표는 성부를 이어 붙여 '꽉 찼다'고 보지 않는다
        # ("교회" 9마디: 아래 성부(1~2박)와 위 성부(3~4박)를 한 성부로 이어 4박 — 실제로는 위 성부 1박 G♯4 누락)
        _ink = getattr(measure, "unexplained_heads", None) or {}
        for k, v in by_voice.items():
            by_staff.setdefault(k[0], []).append((k, v))
        # [v1.20 이어지는 성부] 마디를 채운 성부가 없을 때, 한 성부가 끝난 오른쪽에서 다른 성부가
        # 시작하고 둘을 합치면 마디가 꼭 차면 한 성부로 본다(Audiveris가 위 선율을 마디 중간부터
        # 다른 번호로 이은 경우 — "기도의 능력" 2마디 성부 1 + 성부 3).
        xs = {}
        for e in measure.elements:
            xs.setdefault((e.staff_no, e.voice_no), []).append(e.src_x)
        for sn, items in by_staff.items():
            if any(abs(v - capacity) < 0.01 for _, v in items) or _ink.get(sn):
                continue                       # (숨은 음 증거가 있는 보표는 이어 붙이지 않는다)
            done = False
            for ka, va in items:
                for kb, vb in items:
                    if done or ka == kb or abs(va + vb - capacity) >= 0.01:
                        continue
                    if xs.get(ka) and xs.get(kb) and max(xs[ka]) < min(xs[kb]):
                        by_voice = dict(by_voice)
                        by_voice[ka] = capacity
                        by_voice.pop(kb, None)
                        counts[ka] = counts.get(ka, 0) + counts.get(kb, 0)
                        done = True
            if done:
                by_staff[sn] = [(k, by_voice[k]) for k, _ in items if k in by_voice]
        collapsed = {}
        for sn, items in by_staff.items():
            if len(items) >= 2 and not any(abs(v - capacity) < 0.01 for _, v in items) \
                    and abs(sum(v for _, v in items) - capacity) < 0.01 and not _ink.get(sn):
                collapsed[items[0][0]] = capacity
                counts[items[0][0]] = max(2, counts.get(items[0][0], 0))
            else:
                collapsed.update(dict(items))
        by_voice = collapsed

    real = {k: v for k, v in by_voice.items() if counts.get(k, 0) >= 2}
    orphans = {k: v for k, v in by_voice.items() if counts.get(k, 0) < 2}

    if not orphans:
        return real or by_voice
    if not real:
        return by_voice  # 전부 1개짜리뿐이면 어쩔 수 없이 그대로 다 보여줌

    merged = dict(real)
    orphans_by_staff = {}
    for (sn, vn), val in orphans.items():
        orphans_by_staff.setdefault(sn, []).append(((sn, vn), val))

    for sn, items in orphans_by_staff.items():
        same_staff_keys = [k for k in merged if k[0] == sn]
        if not same_staff_keys:
            for k, v in items:
                merged[k] = v   # 합칠 곳이 없으면 그대로 별개 성부로 유지
            continue
        target_key = same_staff_keys[0]
        base = merged[target_key]
        total_all = base + sum(v for _, v in items)
        if capacity is not None and abs(total_all - capacity) < 0.01:
            merged[target_key] = total_all   # 전부 합쳐야 맞는 경우 한번에 채택
            continue
        for k, v in items:
            candidate = merged[target_key] + v
            if capacity is not None and abs(candidate - capacity) < 0.01:
                merged[target_key] = candidate   # 정확히 맞아떨어지면 합침
            # 안 맞으면 이 orphan은 버린다(기존처럼 무시 — 진짜 유령일 가능성)
    return merged


def measure_status(measure: Measure, capacity: float) -> str:
    """마디 상태: 'ok' | 'over' | 'under'  (단일 마디 기준, 못갖춘마디 미고려).

    [Tier2] 마디 안에 성부(staff_no, voice_no 조합)가 둘 이상이면, 그 중
    하나라도 어긋나면 이 마디 전체를 문제 있음으로 판정한다 — 계획서
    4.3절 "(안 B)" 설계: 마디 배지 자체는 하나로 유지하고, 성부별 상세는
    validate_by_voice()가 별도로 제공한다. 단일 성부(기존 Tier1)는 결과가
    완전히 동일하다(회귀 없음).
    """
    by_voice = measure.quarter_sum_by_voice(capacity)
    if len(by_voice) <= 1:
        total = measure.quarter_sum(capacity)
        if abs(total - capacity) < 0.01:
            return "ok"
        return "over" if total > capacity else "under"

    # 다성부: 어긋난 성부 중 대표 하나를 골라 그 방향(over/under)으로 표시
    pool = _real_voice_pool(measure, by_voice, capacity)

    worst_key, worst_total, worst_diff = None, None, -1.0
    for key, total in pool.items():
        diff = abs(total - capacity)
        if diff > worst_diff:
            worst_key, worst_total, worst_diff = key, total, diff
    if worst_diff < 0.01:
        return "ok"
    return "over" if worst_total > capacity else "under"


def validate_by_voice(score: Score) -> dict:
    """[Tier2] {마디번호: {(staff_no,voice_no): (status, 합, 용량)}}.

    4.3절 "(안 B)" 마디 배지 클릭 시 펼치는 성부별 상세 패널이 이 함수를
    쓴다. 단일 성부 악보에서도 호출 가능하지만(딕셔너리에 항목 1개),
    UI는 성부가 2개 이상일 때만 상세 패널을 보여주면 된다.
    """
    result = {}
    for m in score.measures:
        cap = _capacity_of(score, m)
        by_voice = m.quarter_sum_by_voice(cap)
        detail = {}
        for key, total in by_voice.items():
            if abs(total - cap) < 0.01:
                st = "ok"
            else:
                st = "over" if total > cap else "under"
            detail[key] = (st, round(total, 3), cap)
        result[m.number] = detail
    return result


def detect_anacrusis(score: Score) -> bool:
    """이 악보가 못갖춘마디(anacrusis, 여린내기)로 시작하는지 추정한다.

    판정: 첫 마디 합이 용량보다 작고(부족), 그 부족분을 마지막 마디가 정확히
    채우는가(첫 마디 합 + 마지막 마디 합 == 용량). 못갖춘마디로 시작하는 곡은
    관례상 마지막 마디에서 박자를 보충해 한 마디를 완성한다.
    """
    if len(score.measures) < 2:
        return False
    if not score.measures:
        return False
    first_m, last_m = score.measures[0], score.measures[-1]
    cap = _capacity_of(score, first_m)
    first = first_m.quarter_sum(cap)
    last = last_m.quarter_sum(_capacity_of(score, last_m))
    if first <= 0 or first >= cap - 0.01:
        return False
    return abs((first + last) - cap) < 0.01


def _same_stem_lone_conflict(measure, cap_q) -> bool:
    """[v1.43 숨은 박자 오류 — 실사용] "교회" 15마디 왼손: 원본은 8분음표 G♯2·E3를 빔으로 이었는데, Audiveris가
    빔을 놓쳐 G♯2를 4분음표로, E3를 혼자뿐인 다른 성부로 읽었다. 4분음표 G♯2 덕분에 그 성부가 마디를 꼭
    채워 정상으로 보였다(혼자뿐인 E3는 '유령 성부'로 빠짐). 원본 그림의 기둥 방향(stem_dir_img)을 보면
    E3와 G♯2는 같은 방향 — 진짜 두 성부라면 기둥이 서로 반대다. 그래서 '음 하나뿐인 성부의 음이, 마디를
    채운 성부의 같은 방향 기둥 음이 울리는 도중에 시작하면' 박자 오류로 본다."""
    from .musicxml import element_onsets, _dur_div, DIVISIONS
    cap = int(round(cap_q * DIVISIONS))
    sums = measure.quarter_sum_by_voice(cap_q)
    by_key = {}
    for e in measure.elements:
        if not e.is_grace:
            by_key.setdefault((e.staff_no, e.voice_no), []).append(e)
    lone = [(k, es[0]) for k, es in by_key.items() if len(es) == 1 and es[0].kind == "note"
            and getattr(es[0], "stem_dir_img", None)]
    if not lone:
        return False
    on = None
    for (sn, v), e in lone:
        fulls = [k for k in by_key if k[0] == sn and k != (sn, v) and abs(sums.get(k, 0) - cap_q) < 0.01]
        if not fulls:
            continue
        if on is None:
            on = element_onsets(measure, cap)
        oe = on.get(id(e))
        if oe is None:
            continue
        for k in fulls:
            for w in by_key[k]:
                if w.kind != "note" or getattr(w, "stem_dir_img", None) != e.stem_dir_img:
                    continue
                ow = on.get(id(w))
                if ow is not None and ow < oe < ow + _dur_div(w):
                    return True
    return False


def validate(score: Score) -> dict:
    """모든 마디의 상태를 {마디번호: ('ok'|'over'|'under', 합, 용량)} 로 반환.

    못갖춘마디(anacrusis)로 시작하는 악보를 포괄적으로 지원한다: 이 경우
    첫 마디와 마지막 마디는 단독으로는 용량에 못 미치지만 '첫+마지막=용량'이면
    정상으로 간주한다(짝을 이루는 여린내기/마침마디).
    """
    result = {}
    anac = detect_anacrusis(score)
    n = len(score.measures)
    for idx, m in enumerate(score.measures):
        cap = _capacity_of(score, m)
        by_voice = m.quarter_sum_by_voice(cap)
        if len(by_voice) <= 1:
            total = m.quarter_sum(cap)
        else:
            # [실사용 검증 후 버그 수정] 이전에는 여기서 유령 성부를 걸러
            # 내지 않아, measure_status()/editor_app_qt.py의 라벨 표시와는
            # 다른(더 오래된) 기준으로 total을 계산했다 — 편집(예: 부점
            # 추가)으로 실제 상태가 좋아져도 이 함수가 돌려주는 total/
            # status가 갱신 안 된 것처럼 보이는 버그의 실제 원인이었다
            # (다중 선택 편집 후 배지가 그대로 남아있던 문제). 공통 헬퍼로
            # 다른 두 곳(measure_status, editor_app_qt.py)과 기준을 통일한다.
            pool = _real_voice_pool(m, by_voice, cap)
            _, total = max(pool.items(), key=lambda kv: abs(kv[1] - cap))
        if anac and (idx == 0 or idx == n - 1):
            # 못갖춘마디 짝: 각자는 부족하지만 정상으로 처리
            status = "ok"
        else:
            status = measure_status(m, cap)
            if status == "ok" and _same_stem_lone_conflict(m, cap):
                status = "over"
        result[m.number] = (status, round(total, 3), cap)
    return result


# ----------------------------------------------------------------- 빔 그룹
def beam_groups(measure: Measure):
    """한 마디 안에서 실제 빔으로 연결된 음표들을 묶어 반환.

    인식 단계에서 각 음표에 beam_group 키를 부여한다(같은 객체라도 빔이 끊긴
    구간 — 예: 이음줄로만 붙은 음표 — 은 다른 키). 같은 키끼리 묶는다.
    beam_group 이 없거나(-1) 다르면 별개 그룹.
    반환: [[el, el, ...], ...]  (2개 이상 묶인 것만이 '빔 그룹')
    """
    groups = []
    seen = {}
    for e in measure.elements:
        key = getattr(e, "beam_group", -1)
        if key is None or key == -1:
            groups.append([e])
            continue
        if key in seen:
            seen[key].append(e)
        else:
            grp = [e]
            seen[key] = grp
            groups.append(grp)
    return groups


def group_of(measure: Measure, element: Element):
    """element 이 속한 (빔)그룹 요소 리스트를 돌려준다(단독이면 자기 하나)."""
    for g in beam_groups(measure):
        if element in g:
            return g
    return [element]


# ------------------------------------------------------------- 박자 편집
def set_duration(element: Element, base: int, dotted=None):
    element.duration_base = base
    if dotted is not None:
        element.dotted = dotted
        if not dotted:
            element.double_dot = False


def toggle_dot(element: Element):
    """[v1.10 겹점] 점 순환: 없음 → 점 → 겹점 → 없음."""
    if not element.dotted:
        element.dotted, element.double_dot = True, False
    elif not element.double_dot:
        element.double_dot = True
    else:
        element.dotted, element.double_dot = False, False


# --------------------------------------------------------- 이음줄/붙임줄/셋잇단음표
# [편집 기능 추가] Audiveris 인식이 음정·기본 박자는 매우 정확한 반면,
# 부점 다음으로 tie/slur/셋잇단음표에서도 오인식이 실사용 중 확인됐다
# (실사용 검증: "여정" 샘플에서 원본에 없는 tie/slur가 결정적으로 생김,
# 원본에 없는 셋잇단음표 오인식). 커스텀 엔진 하이브리드 재인식 대신,
# 애초 설계 철학(계획서 "(안 B)" — 자동 인식 + 사람이 편집으로 보정)을
# 그대로 따라 편집 기능으로 보완한다.

def _by_staff_pairs(elements: list):
    """[여러 보표 동시 편집 지원] 선택된 요소들을 staff_no별로 나눠,
    각 보표 그룹이 정확히 2개씩인지 확인한다. 그랜드 스태프에서 "위/
    아래 성부를 잇는 붙임줄 4개(오른손 2개+왼손 2개)"를 한 번에
    선택해도, 보표를 섞어 하나의 tie로 취급하면 안 되고 각 보표 안에서
    따로따로 적용돼야 한다(실사용 요청). staff_no가 하나뿐이면 기존과
    동일하게 동작한다. 그룹 중 하나라도 2개가 아니면 빈 리스트를
    반환한다(적용 불가 판정용).
    """
    by_staff = {}
    for e in elements:
        by_staff.setdefault(e.staff_no, []).append(e)
    if any(len(v) != 2 for v in by_staff.values()):
        return []
    return list(by_staff.values())


def can_toggle_tie(elements: list) -> bool:
    """붙임줄 토글 가능 여부: 보표별로 나눴을 때 각 그룹이 정확히 2개,
    모두 음표, 같은 성부, 같은 음정(음이름+옥타브)이어야 한다(다른
    음정이면 tie 개념 자체가 성립하지 않음)."""
    if len(elements) < 2:
        return False
    groups = _by_staff_pairs(elements)
    if not groups:
        return False
    for a, b in groups:
        if a.kind != "note" or b.kind != "note":
            return False
        if a.voice_no != b.voice_no:
            return False
        if a.step != b.step or a.octave != b.octave:
            return False
    return True


def can_toggle_slur(elements: list) -> bool:
    """이음줄 토글 가능 여부: 보표별로 나눴을 때 각 그룹이 정확히 2개,
    모두 음표, 같은 성부여야 한다."""
    if len(elements) < 2:
        return False
    groups = _by_staff_pairs(elements)
    if not groups:
        return False
    for a, b in groups:
        if a.kind != "note" or b.kind != "note":
            return False
        if a.voice_no != b.voice_no:
            return False
    return True


def can_toggle_triplet(elements: list) -> bool:
    """셋잇단음표 토글 가능 여부: 1개 이상(음표/쉼표 모두 가능)."""
    return len(elements) >= 1


def _find_arc(score, a, b, kind):
    for arc in score.arcs:
        if arc.kind == kind and {id(arc.start), id(arc.stop)} == {id(a), id(b)}:
            return arc
    return None


def toggle_tie(score, elements: list):
    """선택된 음표들 사이의 붙임줄을 보표별로 각각 토글한다(있으면
    제거, 없으면 생성) — 여러 보표를 동시에 선택해도 보표마다 독립
    적용된다."""
    for a, b in _by_staff_pairs(elements):
        if a.src_x > b.src_x:
            a, b = b, a
        existing = _find_arc(score, a, b, "tie")
        if existing:
            score.arcs.remove(existing)
            a.tie_start = False
            b.tie_stop = False
        else:
            score.arcs.append(Arc(start=a, stop=b, kind="tie"))
            a.tie_start = True
            b.tie_stop = True


def toggle_slur(score, elements: list):
    """선택된 음표들 사이의 이음줄을 보표별로 각각 토글한다(있으면
    제거, 없으면 생성) — 여러 보표를 동시에 선택해도 보표마다 독립
    적용된다."""
    for a, b in _by_staff_pairs(elements):
        if a.src_x > b.src_x:
            a, b = b, a
        existing = _find_arc(score, a, b, "slur")
        if existing:
            score.arcs.remove(existing)
            a.slur_start = False
            b.slur_stop = False
        else:
            score.arcs.append(Arc(start=a, stop=b, kind="slur"))
            a.slur_start = True
            b.slur_stop = True


def can_toggle_bond(elements: list) -> bool:
    """[UI 통합 — 실사용 요청] "붙임줄"과 "이음줄" 버튼을 하나("이음")
    로 합쳤다 — 선택된 두 음의 음정이 같으면 자동으로 붙임줄, 다르면
    이음줄로 판정한다. 활성화 조건 자체는 이음줄 쪽(음정 제약 없음)과
    동일 — 붙임줄이 되려면 음정이 같아야 하지만, 그건 toggle_bond가
    실행 시점에 판별하지 활성화 여부와는 무관하다."""
    return can_toggle_slur(elements)


def align_columns(score, elements: list) -> int:
    """[v2.41 같은 박 세로 정렬] 고른 음·쉼표를 마디마다 '시작 박'으로 묶어, 묶음마다 가로 가운데(중앙값)에
    맞춘다. 보표마다는 통째로 옮긴다(2도로 옆에 비킨 머리 등 그 보표 안의 좌우 관계는 그대로) — 그 보표에서
    가운데에 가장 가까운 음이 정확히 맞고 나머지는 같은 만큼 따라간다. 한 마디를 통째로 골라도 박마다
    나눠 정렬된다. 정렬은 표시 자리만 바꾸므로 원본 기하(곡선·빔)는 계속 쓴다. 반환: 옮긴 요소 수."""
    from .musicxml import measure_anchors, _slots, _place_voices, DIVISIONS
    sel = {id(e) for e in elements if e is not None and not getattr(e, "is_grace", False)}
    moved = 0
    for m in score.measures:
        mine = [e for e in m.elements if id(e) in sel and e.src_bbox]
        if len(mine) < 2:
            continue
        cap = int(round(_capacity_of(score, m) * DIVISIONS))
        anchors = measure_anchors(m.elements, cap)
        on_of = {}
        by_sv = {}
        for e in m.elements:
            by_sv.setdefault(e.staff_no, {}).setdefault(e.voice_no, []).append(e)
        for sn, voices in by_sv.items():
            vv = {v: _slots(es) for v, es in voices.items()}
            _place_voices(vv, cap, anchors)
            for sl in vv.values():
                for s in sl:
                    for e in s["els"]:
                        on_of[id(e)] = s["on"]
        groups = {}
        for e in mine:
            if id(e) in on_of:
                groups.setdefault(on_of[id(e)], []).append(e)
        for on, es in groups.items():
            if len({e.staff_no for e in es}) < 2:
                continue                             # 한 보표 안끼리는 일부러 비킨 것(2도 등)일 수 있어 그대로
            cx = lambda e: e.src_bbox[0] + e.src_bbox[2] / 2.0
            xs = sorted(cx(e) for e in es)
            target = xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2.0
            by_staff = {}
            for e in es:
                by_staff.setdefault(e.staff_no, []).append(e)
            for sn, part in by_staff.items():
                ref = min(part, key=lambda e: abs(cx(e) - target))
                d = int(round(target - cx(ref)))
                if not d:
                    continue
                # 같은 보표의 같은 화음 구성음도 함께(고르지 않았어도 — 화음이 쪼개지지 않게)
                cgs = {e.chord_group for e in part if e.chord_group != -1}
                group_els = set(id(e) for e in part)
                group_els |= {id(x) for x in m.elements if x.staff_no == sn and x.chord_group in cgs and cgs}
                for x in m.elements:
                    if id(x) not in group_els:
                        continue
                    pristine = getattr(x, "_orig", None) == (x.step, x.octave, x.duration_base, x.src_x)
                    x.src_x += d
                    bx, by, bw, bh = x.src_bbox
                    x.src_bbox = (bx + d, by, bw, bh)
                    if pristine:                     # 자리 맞춤은 '편집'이 아니다 — 원본 곡선·빔을 계속 쓴다
                        x._orig = (x.step, x.octave, x.duration_base, x.src_x)
                    moved += 1
    return moved


def bond_pairs(elements: list) -> list:
    """[v2.31] 이음 버튼용 짝 고르기: 보표마다 정확히 2개면 그대로. 더 많이 골랐으면(두 성부가 같은 자리에
    있어 함께 골라진 경우 등) 가장 앞 음과, 가장 뒤 자리 음들 중 '같은 성부'(없으면 음높이가 가장 가까운) 음."""
    by_staff = {}
    for e in elements:
        if e.kind == "note":
            by_staff.setdefault(e.staff_no, []).append(e)
    pairs = []
    for es in by_staff.values():
        if len(es) < 2:
            return []
        es = sorted(es, key=lambda e: e.src_x)
        a = es[0]
        last_x = es[-1].src_x
        tail = [e for e in es if abs(e.src_x - last_x) <= 20 and e is not a]
        same = [e for e in tail if e.voice_no == a.voice_no]
        dia = lambda e: e.octave * 7 + STEPS.index(e.step)
        b = same[0] if same else min(tail, key=lambda e: abs(dia(e) - dia(a)))
        pairs.append([a, b])
    return pairs


def toggle_bond(score, elements: list):
    """보표별로 그룹핑해 각각 독립 토글하되, 그 쌍의 음정이 같으면
    붙임줄(tie), 다르면 이음줄(slur)로 자동 판별해서 적용한다."""
    for a, b in _by_staff_pairs(elements):
        if a.src_x > b.src_x:
            a, b = b, a
        kind = "tie" if (a.kind == "note" and b.kind == "note" and
                         a.step == b.step and a.octave == b.octave) else "slur"
        existing = _find_arc(score, a, b, kind)
        if existing:
            score.arcs.remove(existing)
            if kind == "tie":
                a.tie_start = False
                b.tie_stop = False
            else:
                a.slur_start = False
                b.slur_stop = False
        else:
            score.arcs.append(Arc(start=a, stop=b, kind=kind))
            if kind == "tie":
                a.tie_start = True
                b.tie_stop = True
            else:
                a.slur_start = True
                b.slur_stop = True


def toggle_triplet(element: Element):
    """셋잇단음표 표시를 켜고 끈다(단일 선택 시에만 쓰인다 — 다중
    선택은 toggle_triplet_group이 대신 처리한다)."""
    element.in_triplet = not element.in_triplet


def toggle_triplet_group(elements: list):
    """[동작 재설계 — 실사용 요청] 다중 선택된 요소들의 셋잇단음표
    상태가 뒤섞여 있을 때(예: Audiveris가 3개 중 2개만 셋잇단으로
    인식해 0.66666, 1, 0.66666처럼 박자가 들쭉날쭉한 경우), 이전에는
    선택된 요소 각각을 독립적으로 반전시켜서 여전히 뒤섞인 상태가 될
    수 있었다. 이제는 먼저 "전부 같은 상태로 맞추는" 것을 우선한다:
    - 상태가 뒤섞여 있으면(일부만 셋잇단) → 전부 꺼서(원래 박자로)
      맞춘다.
    - 이미 전부 같은 상태면(다 켜져 있거나 다 꺼져 있으면) → 그 반대
      값으로 다같이 뒤집는다(다 꺼져 있었으면 다같이 셋잇단으로).
    이렇게 하면 항상 "한 번 누르면 가지런한 상태"가 보장된다."""
    vals = {bool(e.in_triplet) for e in elements}
    new_val = False if len(vals) > 1 else (not vals.pop())
    for e in elements:
        e.in_triplet = new_val


def set_beam_group_duration(measure: Measure, element: Element, base: int, dotted=False):
    """element 이 속한 빔 그룹 전체를 같은 박자로 일괄 지정(★ 핵심 편의 기능)."""
    for e in group_of(measure, element):
        e.duration_base = base
        e.dotted = dotted
        e.double_dot = False


# ------------------------------------------------------------- 음높이 편집
def shift_step(element: Element, delta: int):
    """음이름을 다이아토닉으로 delta 칸 올리거나(양수) 내린다(음수). 옥타브 자동 처리."""
    if element.kind != "note":
        return
    idx = STEPS.index(element.step) + delta
    octave = element.octave + (idx // 7)
    element.step = STEPS[idx % 7]
    element.octave = octave


def shift_octave(element: Element, delta: int):
    if element.kind == "note":
        element.octave += delta


def cycle_accidental(element: Element, fifths: int = None):
    """임시표 순환: 제자리(0) → 샾(+1) → 플랫(-1) → 제자리.

    [버그 수정 — 실사용 검증 후] alter(소리, MusicXML <alter>)뿐 아니라
    accidental(화면에 그려지는 기호)도 함께 갱신한다 — render.py의
    임시표 그리기는 accidental 필드만 보고 판단하는데, 이 함수가
    alter만 바꾸다 보니 버튼을 눌러도 값은 바뀌어도 화면엔 전혀
    반영이 안 되는 버그가 있었다(실사용 검증 중 발견 — 특히 "뒤에
    삽입"으로 새로 만든 음표에 임시표를 붙이려 할 때 눈에 띔)."""
    if element.kind != "note":
        return
    # [v1.43] 순환: 없음 → ♯ → ♭ → ♮ → 없음(조표대로). '없음'이 없어서 잘못 인식된 임시표(예: 소리에
    # 영향 없는 ♮)를 지울 수 없었다. '없음'은 기호를 지우고 음높이를 조표대로 돌린다.
    nxt = {None: "sharp", "sharp": "flat", "flat": "natural", "natural": None}[element.accidental]
    element.accidental = nxt
    if nxt is None:
        element.alter = _key_alter_map(fifths).get(element.step, 0) if fifths is not None else 0
    else:
        element.alter = {"sharp": 1, "flat": -1, "natural": 0}[nxt]


# ------------------------------------------------------- 음표/쉼표·추가·삭제
def toggle_note_rest(element: Element):
    """음표 ↔ 쉼표 전환. 음표로 바꿀 때 음높이가 없으면 기본값(B4)을 준다."""
    if element.kind == "note":
        element.kind = "rest"
    else:
        element.kind = "note"
        if element.step is None:
            element.step, element.octave, element.alter = "B", 4, 0


def hand_over_clef(measure: Measure, element: Element):
    """[v1.10 음자리표 변경] 지우는 요소에 음자리표 변경이 붙어 있으면 같은 보표의 바로 다음
    요소(같은 마디)로 넘겨, 요소를 지워도 뒤 음들의 높이 기준이 바뀌지 않게 한다."""
    ch = getattr(element, "clef_change", None)
    if not ch:
        return
    nxt = [x for x in measure.elements if x is not element and x.staff_no == element.staff_no
           and x.src_x >= element.src_x]
    if nxt:
        tgt = min(nxt, key=lambda x: x.src_x)
        if not tgt.clef_change:
            tgt.clef_change = ch


def delete_element(measure: Measure, element: Element):
    if element in measure.elements:
        hand_over_clef(measure, element)
        measure.elements.remove(element)


# --------------------------------------------------------- [Tier2] 신규 편집
# HYBRID_OMR_PLAN_v6.md 4.2절 근거: 기존 UI 감사에서 "새로 필요"로 표시된
# 기능들. 기존 함수와 동일한 순수 함수 패턴(Element/Measure를 직접 조작,
# 반환값 없음 또는 새 Element 반환)을 따른다.

def toggle_fermata(element: Element):
    """늘임표 토글."""
    element.fermata = not getattr(element, "fermata", False)


def toggle_grace(element: Element):
    """꾸밈음(장식음) 토글. 꾸밈음으로 바꾸면 박자 계산에서 제외된다
    (Element.quarter_length()가 is_grace=True인 요소는 0을 반환하는
    기존 로직을 그대로 따른다 — 이 함수는 플래그만 뒤집는다)."""
    element.is_grace = not getattr(element, "is_grace", False)


def add_chord_note(measure: Measure, element: Element, step_delta: int = 2):
    """element 가 속한 화음에 새 음 하나를 추가(간격은 diatonic step_delta,
    기본 3도=2칸 위)하고 그 새 Element를 반환한다.

    element 가 아직 단독음(chord_group=-1)이면 새로 화음 그룹을 만든다 —
    화음은 항상 2개 이상의 멤버가 있어야 성립하므로, 이 시점부터 원래
    element 도 같은 chord_group에 편입된다.
    """
    if element.kind != "note" or element.step is None:
        return None
    gid = element.chord_group
    if gid == -1:
        gid = (max((e.chord_group for e in measure.elements), default=-1) + 1)
        element.chord_group = gid

    idx = STEPS.index(element.step) + step_delta
    new_octave = element.octave + (idx // 7)
    new_step = STEPS[idx % 7]

    new_el = Element(
        kind="note", duration_base=element.duration_base, dotted=element.dotted,
        double_dot=element.double_dot,
        step=new_step, octave=new_octave, alter=element.alter,
        src_bbox=element.src_bbox, src_x=element.src_x,
        beam_group=element.beam_group, is_grace=element.is_grace,
        in_triplet=element.in_triplet, tup_actual=element.tup_actual,
        tup_normal=element.tup_normal, staff_no=element.staff_no,
        voice_no=element.voice_no, chord_group=gid, fermata=element.fermata,
    )
    idx_in_list = measure.elements.index(element)
    measure.elements.insert(idx_in_list + 1, new_el)
    return new_el


def remove_chord_note(measure: Measure, element: Element):
    """element 를 화음에서 제거. 화음 멤버가 1개만 남으면 그 남은 음의
    chord_group도 -1(단독음)로 되돌린다(더 이상 화음이 아니므로)."""
    gid = element.chord_group
    delete_element(measure, element)
    if gid == -1:
        return
    remaining = [e for e in measure.elements if e.chord_group == gid]
    if len(remaining) == 1:
        remaining[0].chord_group = -1


def cycle_staff(element: Element, n_staves: int = 2):
    """이 요소가 속한 보표를 다음 번호로 전환(1→2→1→...).

    n_staves: 이 악보의 전체 보표 수(피아노=2). 그 이상으로 순환하지 않는다.
    """
    if n_staves < 2:
        return
    element.staff_no = (element.staff_no % n_staves) + 1


def cycle_voice(element: Element, available_voices: list):
    """이 요소의 성부를 available_voices 리스트 안에서 다음 값으로 전환.

    available_voices 는 그 마디(또는 보표)에 실제로 등장하는 성부 번호
    목록을 호출부(editor_app_qt.py)가 넘겨준다 — 임의의 정수로 바꾸면
    다른 성부와 뒤섞여 편집이 꼬일 수 있어, 실제 존재하는 값 사이에서만
    순환하게 제한한다.
    """
    if not available_voices:
        return
    vs = sorted(set(available_voices))
    if element.voice_no not in vs:
        element.voice_no = vs[0]
        return
    idx = vs.index(element.voice_no)
    element.voice_no = vs[(idx + 1) % len(vs)]


def insert_after(measure: Measure, element: Element, min_gap: float = None):
    """element 뒤에 같은 박자의 쉼표를 하나 삽입하고 그 새 요소를 반환.

    새 요소의 x좌표(src_x/src_bbox)를 element 와 다르게 잡는다 — 이전에는
    element 의 좌표를 그대로 복사해, 편집 악보에서 새 요소가 기존 요소와
    정확히 같은 자리에 겹쳐 그려져 '버튼을 눌러도 아무 변화가 없는 것처럼'
    보이는 버그가 있었다(실제로는 모델에는 추가돼 있었지만 시각적으로
    구분이 안 됨). 다음 요소가 있으면 그 중간 지점에, 마지막이면 이전
    간격만큼 오른쪽에 놓아 항상 눈에 보이는 별도 위치를 갖게 한다.

    [간격 버그 수정 — 실사용 검증 후] 기본 간격을 20px로 고정해뒀었는데,
    400dpi로 스캔한 고해상도 원본(오선 간격이 27px 이상)에서는 음표
    머리 지름보다도 좁아 여전히 시각적으로 거의 겹쳐 보였다(마디에 요소가
    하나뿐이라 앞뒤 참고점이 없는 경우 특히 심함 — 실사용 검증 중 발견).
    min_gap(호출부가 실제 오선 간격 기준으로 넘겨주는 값)이 있으면 그
    이상을 보장한다.
    """
    i = measure.elements.index(element) if element in measure.elements else -1
    nxt = measure.elements[i + 1] if 0 <= i and i + 1 < len(measure.elements) else None
    prv = measure.elements[i - 1] if i > 0 else None
    default_gap = min_gap if min_gap else 20.0
    # [겹침 버그 추가 수정 — 실사용 검증 후] 참조 요소가 부점(dotted)이면
    # 렌더링에서 부점 자체가 요소 폭보다 더 오른쪽(+0.4칸 이상)까지
    # 튀어나온다 — min_gap이 그 부점 영역보다 좁으면, 새로 삽입한
    # 요소가 시각적으로 원래 요소의 "부점 자리"와 겹쳐 보였다(실사용
    # 검증 중 발견: 새 쉼표가 점4분쉼표의 부점과 겹쳐 보임). 부점이
    # 있으면 최소 간격을 한 칸 더 확보한다.
    if default_gap and element.dotted:
        default_gap = default_gap * 1.5

    if nxt is not None and nxt.src_x != element.src_x:
        new_x = (element.src_x + nxt.src_x) / 2.0
        if min_gap and abs(nxt.src_x - element.src_x) < min_gap * 2:
            new_x = element.src_x + min_gap  # 너무 좁으면 최소 간격 보장
    else:
        gap = (element.src_x - prv.src_x) if prv is not None \
            and element.src_x != prv.src_x else default_gap
        gap = gap if abs(gap) >= default_gap else default_gap
        new_x = element.src_x + gap

    bx, by, bw, bh = element.src_bbox if element.src_bbox else (element.src_x, 0, 12, 20)
    new_bbox = (int(new_x - bw / 2), by, bw, bh)

    # [버그 수정 — 실사용 검증] 보표·성부를 기준 음표에서 물려받는다(예전엔 항상 보표1·성부1로
    # 만들어져, 피아노 왼손 옆에 넣어도 노래 보표 쉼표가 되어 버렸다).
    new = Element(kind="rest", duration_base=element.duration_base,
                  dotted=False, src_bbox=new_bbox, src_x=int(round(new_x)),
                  synthetic=True, staff_no=element.staff_no, voice_no=element.voice_no)
    pos = (i + 1) if i >= 0 else len(measure.elements)
    measure.elements.insert(pos, new)
    return new


def insert_before(measure: Measure, element: Element, min_gap: float = None):
    """[신규 추가 — 실사용 요청] element 앞에 같은 박자의 쉼표를 하나
    삽입하고 그 새 요소를 반환한다. Audiveris가 마디의 첫 음표를 통째로
    놓친 경우(실사용 검증: 원본 XML 자체에 그 음표가 없음을 확인 —
    Audiveris의 순수 인식 누락이지 파싱 문제가 아니었다), "뒤에 삽입"만
    있으면 그 누락된 자리를 만들 방법이 없었다. insert_after와 대칭되는
    위치·간격 로직을 그대로 쓴다.
    """
    i = measure.elements.index(element) if element in measure.elements else -1
    prv = measure.elements[i - 1] if i > 0 else None
    nxt = measure.elements[i + 1] if 0 <= i and i + 1 < len(measure.elements) else None
    default_gap = min_gap if min_gap else 20.0
    if default_gap and prv is not None and prv.dotted:
        default_gap = default_gap * 1.5

    if prv is not None and prv.src_x != element.src_x:
        new_x = (element.src_x + prv.src_x) / 2.0
        if min_gap and abs(element.src_x - prv.src_x) < min_gap * 2:
            new_x = element.src_x - min_gap
    else:
        gap = (nxt.src_x - element.src_x) if nxt is not None \
            and element.src_x != nxt.src_x else default_gap
        gap = gap if abs(gap) >= default_gap else default_gap
        new_x = element.src_x - gap

    bx, by, bw, bh = element.src_bbox if element.src_bbox else (element.src_x, 0, 12, 20)
    new_bbox = (int(new_x - bw / 2), by, bw, bh)

    # [버그 수정 — 실사용 검증] 보표·성부를 기준 음표에서 물려받는다(예전엔 항상 보표1·성부1로
    # 만들어져, 피아노 왼손 옆에 넣어도 노래 보표 쉼표가 되어 버렸다).
    new = Element(kind="rest", duration_base=element.duration_base,
                  dotted=False, src_bbox=new_bbox, src_x=int(round(new_x)),
                  synthetic=True, staff_no=element.staff_no, voice_no=element.voice_no)
    pos = i if i >= 0 else 0
    measure.elements.insert(pos, new)
    return new

# ------------------------------------------------------------------ 조옮김
# 음이름 → 반음 피치클래스(C=0). 조옮김의 기준 좌표계.
_STEP_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# fifths(-7..+7) → 장조 으뜸음 피치클래스: 5도권을 반음으로 환산.
#   예) fifths=+1(G장조) → 7,  fifths=-1(F장조) → 5
_FLAT_ORDER = ["B", "E", "A", "D", "G", "C", "F"]
_SHARP_ORDER = ["F", "C", "G", "D", "A", "E", "B"]

# 사용자에게 보여줄 15개 장조(fifths, 표시 이름)
KEY_CHOICES = [
    (0, "다장조 (C, 조표 없음)"),
    (1, "사장조 (G, ♯1)"), (2, "라장조 (D, ♯2)"), (3, "가장조 (A, ♯3)"),
    (4, "마장조 (E, ♯4)"), (5, "나장조 (B, ♯5)"), (6, "올림바장조 (F♯, ♯6)"),
    (7, "올림다장조 (C♯, ♯7)"),
    (-1, "바장조 (F, ♭1)"), (-2, "내림나장조 (B♭, ♭2)"),
    (-3, "내림마장조 (E♭, ♭3)"), (-4, "내림가장조 (A♭, ♭4)"),
    (-5, "내림라장조 (D♭, ♭5)"), (-6, "내림사장조 (G♭, ♭6)"),
    (-7, "내림다장조 (C♭, ♭7)"),
]


def _tonic_pc(fifths: int) -> int:
    """장조 조표 fifths → 으뜸음 피치클래스(반음, C=0)."""
    return (fifths * 7) % 12


def _key_alter_map(fifths: int) -> dict:
    m = {c: 0 for c in "ABCDEFG"}
    if fifths < 0:
        for c in _FLAT_ORDER[:(-fifths)]:
            m[c] = -1
    elif fifths > 0:
        for c in _SHARP_ORDER[:fifths]:
            m[c] = +1
    return m


def _spell_midi(midi: int, fifths: int):
    """MIDI 음 → 목표 조표에 맞는 (step, octave, alter) 철자 선택.

    우선순위: ① 조표 그대로(diatonic) 표기 가능한 철자
             ② 제자리표(♮)로 적을 수 있으면 제자리표 — 예) F장조의 B♮을
                C♭로 적지 않는다
             ③ 조표가 샾 계열이면 샾 표기, 플랫 계열이면 플랫 표기
    """
    pc = midi % 12
    keymap = _key_alter_map(fifths)
    candidates = []
    for step, base in _STEP_PC.items():
        for alt in (-1, 0, 1):
            if (base + alt) % 12 == pc:
                octave = (midi - base - alt) // 12 - 1
                candidates.append((step, octave, alt))

    def rank(c):
        step, _oct, alt = c
        if keymap[step] == alt:
            return 0                       # 조표 그대로 = 최우선
        if alt == 0:
            return 1                       # 제자리표 표기 가능 = 차선
        if fifths >= 0 and alt == 1:
            return 2                       # 샾 계열 조 → 샾 임시표 선호
        if fifths < 0 and alt == -1:
            return 2                       # 플랫 계열 조 → 플랫 임시표 선호
        return 3
    candidates.sort(key=rank)
    return candidates[0]


def _respell_accidentals(m, keymap, score=None):
    """[v3.20 조바꿈 임시표 — 실사용 피드백] 마디 안 박 순서대로, 보표·음이름·옥타브마다 '지금 적용되는 변화'
    (처음엔 조표)를 따라가며 실제 변화와 다를 때만 임시표를 붙인다. 같은 마디에서 앞 음의 임시표는 뒤 음에도
    이어지므로, 예전처럼 조표하고만 견주면 '♮ 뒤 다시 ♯'이 빠지거나 겹쳐 붙었다(교회 E→C장조 표기 오류 51→70).
    마디를 넘는 붙임줄의 뒷음은 다시 적지 않는다(표준 표기). 화음은 같은 박이라 함께."""
    try:
        from .musicxml import element_onsets, DIVISIONS
        cap = _capacity_of(score, m) if score is not None else 4.0
        on = element_onsets(m, int(round(cap * DIVISIONS)))
    except Exception:
        on = {}
    state = {}
    notes = [e for e in m.elements if e.kind == "note" and e.step is not None]
    for e in sorted(notes, key=lambda x: (on.get(id(x), 0), x.src_x or 0)):
        k = (e.staff_no, e.step, e.octave)
        cur = state.get(k, keymap[e.step])
        alt = e.alter or 0
        first = on.get(id(e), 0) == min((on.get(id(x), 0) for x in notes), default=0)
        if getattr(e, "tie_stop", False) and first:
            e.accidental = None                       # 앞 마디에서 이어진 음
            continue
        if alt != cur:
            e.accidental = {1: "sharp", -1: "flat", 0: "natural", 2: "double-sharp", -2: "flat-flat"}.get(alt)
            state[k] = alt
        else:
            e.accidental = None


def _shift_key(fifths: int, delta: int, old_first: int, new_first: int) -> int:
    """조표 하나를 delta 반음 옮긴 조표. 같은 음이름 후보(예: F#=+6, Gb=-6) 중 첫 조표가 옮겨진
    방향(샵 쪽/플랫 쪽)을 따르고, 그다음은 조표 수가 적은 쪽을 고른다."""
    pc = (_tonic_pc(fifths) + delta) % 12
    cands = [f for f in range(-7, 8) if _tonic_pc(f) == pc]
    shift = new_first - old_first
    return min(cands, key=lambda f: (abs(f - (fifths + shift)), abs(f)))


def transpose_score(score: Score, target_fifths: int, direction: str = "nearest"):
    """악보 전체를 목표 조(장조 기준)로 조옮김한다.

    Parameters
    ----------
    score : Score
    target_fifths : int
        목표 조표(-7..+7). 예) 사장조 = +1, 바장조 = -1.
    direction : str
        "nearest"(가까운 쪽) | "up"(올림) | "down"(내림) — 반음 이동 방향.

    동작: 현재 조표와 목표 조표의 으뜸음 간 반음 차를 구해 모든 음표의
    MIDI 값을 이동한 뒤, 목표 조표 기준으로 철자를 다시 정한다(조표에 있는
    음은 임시표 없이, 반음계 음은 조 성격에 맞는 임시표로). 쉼표·박자는
    그대로 두고 score.fifths 만 갱신한다.
    """
    delta = (_tonic_pc(target_fifths) - _tonic_pc(score.fifths)) % 12
    off = getattr(score, "transpose_offset", 0) or 0       # [v3.20] 원래 높이에서 지금까지 옮긴 반음
    if direction == "nearest":
        # 위·아래 중 원래 높이에 가까운 쪽(증4도 동률이면 되돌아오는 쪽) — 예전엔 F→B(+6)→F(+6)로
        # 한 옥타브가 쌓였다(기도의 능력 시험)
        delta = min((delta, delta - 12), key=lambda d: (abs(off + d), d))
    elif direction == "down":
        if delta > 0:
            delta -= 12
    # "up"이면 0..11 그대로(같은 조면 0 → 이동 없음)
    if delta == 0 and target_fifths == score.fifths:
        return score

    # [v1.10 조표 바뀜] 곡 중간의 조표도 같은 음정만큼 옮기고, 각 마디는 그 마디의 조표로 철자한다.
    old_first = score.fifths
    cur = target_fifths
    for m in sorted(score.measures, key=lambda mm: mm.number):
        if getattr(m, "key_change", None) is not None:
            m.key_change = _shift_key(m.key_change, delta, old_first, target_fifths)
            cur = m.key_change
        for e in m.elements:
            if e.kind != "note" or e.step is None:
                continue
            midi = 12 * (e.octave + 1) + _STEP_PC[e.step] + (e.alter or 0)
            step, octave, alt = _spell_midi(midi + delta, cur)
            e.step, e.octave, e.alter = step, octave, alt
        _respell_accidentals(m, _key_alter_map(cur), score)
    score.transpose_offset = off + delta
    # [v3.00] 기타 코드도 같은 음정만큼 — 철자는 목표 조(내림표 조면 ♭, 올림표 조면 ♯)
    from . import chords as _chd
    for m in score.measures:
        for e in m.elements:
            if getattr(e, "harmony", None):
                _chd.transpose(e.harmony, delta, target_fifths)
    score.fifths = target_fifths
    return score


# ---------------------------------------------------------------- 셈여림
# [셈여림 편집 — 실사용 요청] 내부 키만 저장·비교한다(화면 문구는 editor_app_qt가
# 표시 시점에 변환). 자동 인식은 가사·코드기호를 셈여림으로 오인하는 경우가 많아
# (실측: 교과서 PDF 15개 중 대부분 허위) 개별 수정과 "자동 인식분 일괄 삭제"를 함께 둔다.
DYNAMIC_KEYS = ("pp", "p", "mp", "mf", "f", "ff", "sfz", "fp")
HAIRPIN_KEYS = ("cresc", "decresc")


def _order_index(score):
    """요소의 악보상 순서(마디 순 → 같은 마디 안 x 순) — 쐐기의 시작/끝 판정용."""
    idx = {}
    for mi, m in enumerate(score.measures):
        for i, e in enumerate(m.elements):
            idx[id(e)] = (mi, e.src_x, i)
    return idx


def can_set_dynamic(elements: list) -> bool:
    return len(elements) >= 1


def set_dynamic(elements: list, key: str):
    """선택한 요소(들)에 셈여림을 붙인다. 모두 이미 같은 셈여림이면 떼어낸다(토글)."""
    if key not in DYNAMIC_KEYS or not elements:
        return
    if all(e.dynamic and e.dynamic.get("key") == key for e in elements):
        for e in elements:
            e.dynamic = None
        return
    for e in elements:
        e.dynamic = {"key": key, "x": None, "y": None, "auto": False}


def can_toggle_hairpin(elements: list) -> bool:
    """같은 보표에서 2개 이상 선택돼 있어야 쐐기(시작~끝)를 만들 수 있다."""
    counts = {}
    for e in elements:
        counts[e.staff_no] = counts.get(e.staff_no, 0) + 1
    return any(c >= 2 for c in counts.values())


def hairpins_covering(score, elements: list, kind: str = None) -> list:
    """선택한 요소 중 하나라도 그 구간(시작~끝) 안에 드는 쐐기들(같은 보표, kind 지정 시 그 종류만)."""
    idx = _order_index(score)
    found = []
    for h in score.hairpins:
        if kind is not None and h.kind != kind:
            continue
        i0, i1 = idx.get(id(h.start)), idx.get(id(h.stop))
        if i0 is None or i1 is None:
            continue
        lo, hi = min(i0, i1), max(i0, i1)
        for e in elements:
            ie = idx.get(id(e))
            if ie is not None and e.staff_no == h.start.staff_no and lo <= ie <= hi:
                found.append(h)
                break
    return found


def toggle_hairpin(score, elements: list, kind: str):
    """크레셴도/디크레셴도 토글.
    - 선택한 음표가 같은 종류의 기존 쐐기 안에 있으면 → 그 쐐기를 지운다(시작·끝 음표를 정확히
      고르지 않아도 됨. 음표 하나만 골라도 된다).
    - 아니면 보표별로 선택 범위의 첫 요소~마지막 요소에 새로 만든다(그 구간에 다른 종류 쐐기가
      있으면 바꾼다). 새로 만들려면 같은 보표에서 2개 이상 선택해야 한다."""
    if kind not in HAIRPIN_KEYS:
        return "none"
    covering = hairpins_covering(score, elements, kind)
    if covering:
        score.hairpins[:] = [h for h in score.hairpins if h not in covering]
        return "removed"
    idx = _order_index(score)
    by_staff = {}
    for e in elements:
        by_staff.setdefault(e.staff_no, []).append(e)
    made = False
    for els in by_staff.values():
        if len(els) < 2:
            continue
        els = sorted(els, key=lambda e: idx.get(id(e), (0, 0, 0)))
        a_, b_ = els[0], els[-1]
        other = hairpins_covering(score, [a_, b_], None)
        score.hairpins[:] = [h for h in score.hairpins if h not in other]
        score.hairpins.append(Hairpin(start=a_, stop=b_, kind=kind))
        made = True
    return "added" if made else "none"


def clear_dynamics(score, elements: list):
    """선택한 요소의 셈여림과, 그 요소에서 시작하거나 끝나는 쐐기를 지운다."""
    ids = {id(e) for e in elements}
    for e in elements:
        e.dynamic = None
    score.hairpins[:] = [h for h in score.hairpins
                         if id(h.start) not in ids and id(h.stop) not in ids]


def clear_auto_dynamics(score) -> int:
    """자동 인식된 셈여림·쐐기를 모두 지운다(편집으로 직접 넣은 것은 남긴다). 지운 개수 반환."""
    n = 0
    for m in score.measures:
        for e in m.elements:
            if e.dynamic and e.dynamic.get("auto"):
                e.dynamic = None
                n += 1
    before = len(score.hairpins)
    score.hairpins[:] = [h for h in score.hairpins if not getattr(h, "auto", False)]
    return n + (before - len(score.hairpins))


# ---------------------------------------------------------------- 잇단음표(일반화)
# [잇단음표 일반화 — 실사용 요청] 내부 키 → (actual, normal). 화면 문구는 editor_app_qt에서
# 표시 시점에만 붙인다. 측정 결과 Audiveris는 3잇단·6잇단만 자동 인식하고 5·7잇단,
# 복합박자의 2·4잇단은 못 읽으므로(시험 악보 0/4) 편집으로 만드는 경로가 필수다.
TUPLET_SPECS = {"t3": (3, 2), "t5": (5, 4), "t6": (6, 4), "t7": (7, 4),
                "t2": (2, 3), "t4": (4, 3)}


def tuplet_split_spec(e, key):
    """음표 1개를 key 잇단음표로 쪼갤 때 새 음가 (duration_base, dotted). 불가능하면 None.
    원래 길이 D를 normal로 나눈 값이 표준 음가(또는 점음가)여야 한다
    (예: 4분음표→셋잇단 8분 3개, 점4분음표→2잇단 8분 2개, 4분음표→2잇단은 불가)."""
    spec = TUPLET_SPECS.get(key)
    if spec is None or e.is_grace or e.in_triplet:
        return None
    q_b = raw_quarter(e) / spec[1]
    for base in (1, 2, 4, 8, 16, 32):
        bq = 4.0 / base
        if abs(bq - q_b) < 1e-9:
            return base, False
        if abs(bq * 1.5 - q_b) < 1e-9:
            return base, True
    return None


def apply_tuplet(score, measure, elements: list, key: str, min_gap: float = None):
    """잇단음표 적용. 반환: (결과 코드, 포커스 요소)
    결과 코드(내부 키): "cleared" 해제 / "marked" 표시만 / "split" 쪼갬 /
    "unsupported" 이 길이로는 불가 / "chord" 화음 단일 선택(쪼개기 미지원)."""
    if key == "clear":
        for e in elements:
            e.in_triplet = False
        return "cleared", (elements[0] if elements else None)
    spec = TUPLET_SPECS.get(key)
    if spec is None or not elements:
        return "unsupported", None
    if len(elements) == 1:
        e = elements[0]
        if e.in_triplet:                       # 잘못 인식된 잇단음표 하나를 해제
            e.in_triplet = False
            return "cleared", e
        if e.chord_group != -1:
            return "chord", e
        ns = tuplet_split_spec(e, key)
        if ns is None:
            return "unsupported", e
        base, dot = ns
        actual, normal = spec
        idx = measure.elements.index(e)
        nxt = next((x for x in measure.elements[idx + 1:]
                    if x.staff_no == e.staff_no and x.src_x > e.src_x), None)
        span = (nxt.src_x - e.src_x) if nxt is not None else (min_gap or 20.0) * actual
        new_bg = e.beam_group
        if base >= 8:                          # 8분음표 이하는 한 빔으로 묶는다
            used = [x.beam_group for m in score.measures for x in m.elements]
            new_bg = max([b for b in used if isinstance(b, int)] + [0]) + 1
        e.duration_base, e.dotted, e.double_dot = base, dot, False
        e.in_triplet, e.tup_actual, e.tup_normal = True, actual, normal
        e.beam_group = new_bg
        for k in range(1, actual):
            dx = span * k / actual
            bb = e.src_bbox
            new = dataclasses.replace(
                e, clef_change=None, src_x=int(round(e.src_x + dx)),
                src_bbox=(int(round(bb[0] + dx)), bb[1], bb[2], bb[3]) if bb else None,
                accidental=None, tie_start=False, tie_stop=False, slur_start=False,
                slur_stop=False, lyrics={}, dynamic=None, synthetic=True, fermata=False)
            measure.elements.insert(idx + k, new)
        if min_gap and span / actual < min_gap * 0.8:
            _respace_staff(measure, e.staff_no, min_gap)
        return "split", e
    if all(x.in_triplet and (x.tup_actual, x.tup_normal) == spec for x in elements):
        for x in elements:
            x.in_triplet = False
        return "cleared", elements[0]
    for x in elements:
        x.in_triplet, x.tup_actual, x.tup_normal = True, spec[0], spec[1]
    return "marked", elements[0]


def _respace_staff(measure, staff_no, min_gap):
    """[쪼개기 후 재배치] 원본 악보의 음표 간격이 좁아 쪼갠 음표들이 겹칠 때, 그 마디의
    해당 보표 요소들을 화면 표시 위치(disp_x)만 길이에 비례해 다시 벌린다. 첫 요소 자리에서
    시작해 마지막 요소 자리(필요하면 오른쪽으로 조금 더)까지 쓴다. 화음 구성음은 같은 x."""
    slots, seen = [], {}
    for x in measure.elements:
        if x.staff_no != staff_no:
            continue
        cg = x.chord_group
        if cg != -1 and cg in seen:
            seen[cg].append(x)
            continue
        grp = [x]
        if cg != -1:
            seen[cg] = grp
        slots.append(grp)
    if len(slots) < 2:
        return
    def _cx(x):
        return x.disp_x if x.disp_x is not None else x.src_x
    x0 = _cx(slots[0][0])
    # 촘촘한 원본에서는 길이 비례보다 '겹치지 않는 것'이 우선이라 거의 균등하게 벌린다
    # (지수 0.25: 긴 음표가 조금만 더 넓게).
    widths = [max(g[0].quarter_length(), 0.125) ** 0.25 for g in slots]
    need = x0 + min_gap * 1.0 * (len(slots) - 1)
    x1 = max(_cx(slots[-1][0]), need)
    total = sum(widths[:-1]) or 1.0
    acc = 0.0
    for g, w in zip(slots, widths):
        pos = x0 + (x1 - x0) * acc / total
        for x in g:
            x.disp_x = pos
        acc += w


def remove_ghost_rests(score, interline: float, staff_groups: dict = None) -> int:
    """[유령 쉼표 자동 정리 — 실사용 요청] 한 마디·한 보표에서 '혼자뿐인 쉼표 성부'이면서
    다른 성부의 음표와 거의 같은 x에 있는 쉼표를 지운다. 그런 성부는 마디를 채우지도
    못해(예: 16분쉼표 하나) 음악적으로 성립하지 않는 Audiveris 오인식이다(교가 5마디).
    지운 개수를 돌려준다."""
    n = 0
    for m in score.measures:
        counts = {}
        for e in m.elements:
            counts[(e.staff_no, e.voice_no)] = counts.get((e.staff_no, e.voice_no), 0) + 1
        ghosts = []
        for e in m.elements:
            if e.kind != "rest" or counts[(e.staff_no, e.voice_no)] != 1:
                continue
            if any(o.kind == "note" and o.staff_no == e.staff_no and o.voice_no != e.voice_no
                   and abs(o.src_x - e.src_x) < interline * 1.2 for o in m.elements):
                ghosts.append(e)
        # [꼬리 오인 쉼표 — 실사용 검증] Audiveris가 아래로 뻗은 8분음표의 꼬리를 8분쉼표로
        # 읽는 경우("봄이 오면" 19마디). 진짜 쉼표는 오선 안쪽에 오는데, 이런 쉼표는 오선
        # 바깥으로 한 칸 이상 벗어나 있고 같은 보표 음표의 기둥 바로 옆에 붙어 있다.
        if staff_groups:
            grp = staff_groups.get(m.system) or []
            for e in m.elements:
                if e.kind != "rest" or e in ghosts or not e.src_bbox or e.staff_no > len(grp):
                    continue
                lines = grp[e.staff_no - 1]
                cy = e.src_bbox[1] + e.src_bbox[3] / 2.0
                outside = cy > lines[4] + interline * 1.0 or cy < lines[0] - interline * 1.0
                near_stem = any(o.kind == "note" and o.staff_no == e.staff_no
                                and abs(o.src_x - e.src_x) < interline * 1.5 for o in m.elements)
                if outside and near_stem and e.duration_base >= 8:
                    ghosts.append(e)
        for g in ghosts:
            # 지우기 전에, 이 쉼표에 붙어 있던 쐐기·셈여림을 같은 보표의 가장 가까운 요소로
            # 옮긴다(Audiveris가 쐐기 시작을 유령 쉼표에 붙인 경우 — "봄이 오면" 22마디 왼손
            # 디크레셴도가 쉼표와 함께 사라지던 문제).
            rest_ = [o for o in m.elements if o is not g and o.staff_no == g.staff_no]
            if rest_:
                near = min(rest_, key=lambda o: abs(o.src_x - g.src_x))
                for h in getattr(score, "hairpins", []):
                    if h.start is g:
                        h.start = near
                    if h.stop is g:
                        h.stop = near
                if g.dynamic and not near.dynamic:
                    near.dynamic = g.dynamic
            hand_over_clef(m, g)
            m.elements.remove(g)
            n += 1
    return n


def fix_tie_rests(score, interline: float) -> int:
    """[v1.21 붙임줄을 쉼표로 오인 — 실사용] Audiveris가 짧은 붙임줄을 2분쉼표로 읽는 경우("교회" 14마디
    합창: E3 16분 ⌒ E3 16분 사이의 붙임줄이 HALF_REST, 신뢰도 0.52). 다음을 모두 만족하면 쉼표를 지우고
    붙임줄로 바꾼다: 2분·온쉼표 / 납작함(높이 0.7칸 이하) / 같은 보표·성부의 바로 앞뒤 음이 같은 음높이 /
    쉼표를 빼면 그 성부가 마디를 꼭 채운다(쉼표 때문에 넘친다). 바꾼 개수를 돌려준다."""
    from .model import Arc
    n = 0
    for m in score.measures:
        cap = _capacity_of(score, m)
        for e in list(m.elements):
            if e.kind != "rest" or e.duration_base not in (1, 2) or not e.src_bbox:
                continue
            if e.src_bbox[3] > 0.7 * interline:
                continue
            same = sorted((x for x in m.elements if x.staff_no == e.staff_no and x.voice_no == e.voice_no
                           and x is not e and (x.chord_group == -1 or x is next(
                               y for y in m.elements if y.chord_group == x.chord_group))),
                          key=lambda x: x.src_x)
            prv = [x for x in same if x.src_x < e.src_x]
            nxt = [x for x in same if x.src_x > e.src_x]
            if not prv or not nxt:
                continue
            a, b = prv[-1], nxt[0]
            if a.kind != "note" or b.kind != "note" or (a.step, a.octave, a.alter) != (b.step, b.octave, b.alter):
                continue
            total = m.quarter_sum_by_voice(cap).get((e.staff_no, e.voice_no), 0)
            # [v2.11] 예전엔 '빼면 마디가 꼭 참'만 받았다. 같은 마디에 음이 하나 더 빠져 있으면("교회" 33마디
            # 합창: 붙임줄을 2분쉼표로 + 16분음표 B2 누락) 빼도 조금 모자라 바로잡지 못했고, 그 쉼표 때문에
            # 빠진 음의 자동 보완도 막혔다. 납작함·같은 음 사이라는 근거가 확실하므로 '이 쉼표 때문에 넘치고,
            # 빼면 넘치지 않음'이면 바로잡는다(모자란 부분은 뒤의 자동 보완이 채운다).
            if not (total > cap + 0.01 and total - e.quarter_length() <= cap + 0.01):
                continue
            hand_over_clef(m, e)
            m.elements.remove(e)
            if not any(ar.start is a and ar.stop is b for ar in score.arcs):
                score.arcs.append(Arc(start=a, stop=b, kind="tie", above=True))
                a.tie_start, b.tie_stop = True, True
            n += 1
    return n


def flip_stems(measure, elements, current_up: dict):
    """[v2.25] 고른 음의 기둥 방향을 뒤집는다(화음·빔 묶음은 묶음 전체). current_up: {id(e): 지금 그려진
    방향(True=위)}. 뒤집은 방향을 e.stem_override('up'/'down')에 적는다."""
    targets = set()
    for e in elements:
        if e.kind != "note" or e.duration_base == 1:
            continue
        mem = [e]
        if e.chord_group != -1:
            mem += [x for x in measure.elements if x.chord_group == e.chord_group]
        if e.beam_group is not None and e.beam_group != -1:
            mem += [x for x in measure.elements if x.beam_group == e.beam_group and x.staff_no == e.staff_no]
            for x in list(mem):
                if x.chord_group != -1:
                    mem += [y for y in measure.elements if y.chord_group == x.chord_group]
        targets.update(id(x) for x in mem)
    by_id = {id(x): x for x in measure.elements}
    seen_dir = None
    for i in targets:
        x = by_id.get(i)
        if x is None:
            continue
        if seen_dir is None:
            seen_dir = not current_up.get(i, True)       # 묶음 안 첫 음 기준으로 뒤집은 방향
        x.stem_override = "up" if seen_dir else "down"
    return seen_dir


def insert_grace(score, measure, main, space: float = 20.0):
    """[v2.21 앞꾸밈음 넣기] main 음 앞에 작은 8분 꾸밈음(한 음 위, 조표대로)을 넣고 main과 이음줄로 잇는다.
    음높이는 넣은 뒤 ↑↓로 맞춘다. 넣은 꾸밈음을 돌려준다."""
    from .model import Element, Arc, _alter_map
    if main is None or main.kind != "note" or main.is_grace:
        return None
    idx = STEPS.index(main.step) + 1
    step, octave = STEPS[idx % 7], main.octave + idx // 7
    alter = _alter_map(score.key_at(measure.number)).get(step, 0)
    bx, by, bw, bh = main.src_bbox if main.src_bbox else (main.src_x, 0, int(space), int(space))
    gx = int(bx - 1.6 * space)
    el = Element(kind="note", duration_base=8, dotted=False, step=step, octave=octave, alter=alter,
                 src_bbox=(gx, int(by - space / 2), int(bw * 0.65), int(bh * 0.65)), src_x=gx,
                 staff_no=main.staff_no, voice_no=main.voice_no, is_grace=True, synthetic=True)
    measure.elements.append(el)
    score.arcs.append(Arc(start=el, stop=main, kind="slur", above=True))
    el.slur_start, main.slur_stop = True, True
    return el


def infer_missed_time_changes(score, printed: dict) -> list:
    """[v2.21 놓친 박자표 — 실사용] Audiveris가 박자표를 못 읽은 경우("교회" 50마디 2/4 — .omr에 2/4 기호
    자체가 없음). 그러면 53마디에 다시 찍힌 4/4는 '바뀌지 않은 박자'로 보여 함께 사라진다.
    곡 중간(줄 첫 마디가 아닌 곳)에 찍힌 박자표는 반드시 박자가 바뀐다는 뜻이므로, 지금 박자와 같게 보이면
    그 앞에 놓친 바뀜이 있다. 그 앞으로 거슬러 '모든 성부가 같은 다른 길이로 꽉 찬' 마디들을 찾아, 그 첫
    마디에 그 길이의 박자표(x/4)를 넣고 찍힌 박자표로 돌아오게 한다. 그 줄 전체에 걸려 있던 줄 단위 길이
    (target_capacity)는 지우고 박자표를 따르게 한다. printed: {마디 번호: (beats, beat_type)}.
    반환: [(마디 번호, (beats, beat_type)), ...] 넣은 박자표."""
    added = []
    by_num = {m.number: m for m in score.measures}
    for k in sorted(printed or {}):
        mk = by_num.get(k)
        if mk is None or k == 1 or getattr(mk, "first_in_system", False) or mk.time_sig_change:
            continue
        b, bt = printed[k]
        cap_t = b * 4.0 / bt
        run, run_c = [], None
        j = k - 1
        while j >= 1 and j in by_num:
            mj = by_num[j]
            sums = mj.quarter_sum_by_voice(cap_t)
            if not sums:
                break
            c = max(sums.values())
            if abs(c - cap_t) < 0.01 or (run_c is not None and abs(c - run_c) > 0.01):
                break
            if any(abs(v - c) > 0.01 for v in sums.values() if v > 0.5 * c):
                break                                # 꽉 찬 성부끼리 길이가 다르면 근거가 약하다
            run_c = c
            run.append(mj)
            if mj.time_sig_change:
                break
            j -= 1
        if not run or run_c is None or run_c <= 0 or abs(run_c * 2 - round(run_c * 2)) > 0.01:
            continue
        start = run[-1]
        new_ts = (int(round(run_c)), 4) if abs(run_c - round(run_c)) < 0.01 else (int(round(run_c * 2)), 8)
        start.time_sig_change = new_ts
        mk.time_sig_change = (b, bt)
        lines = {m.system for m in run} | {mk.system}
        for m in score.measures:
            if m.system in lines:
                m.target_capacity = run_c if m in run else None
        added += [(start.number, new_ts), (k, (b, bt))]
    return added


def merge_split_voices(score, measure, staff_no) -> bool:
    """[v3.40] Audiveris가 셋잇단음표를 못 읽으면 한 줄로 이어진 음들의 박 위치가 맞지 않아 성부를 여러 개로
    쪼갠다(데모 11마디 윗보표: A G F E D가 성부 1·2·3·4). 셋잇단을 바로잡은 뒤, 그 보표 요소들이 서로 겹치지
    않고(같은 자리에는 한 성부만) 차례로 이어 붙인 길이가 마디 길이와 같으면 한 성부로 모은다. 모았으면 True."""
    els = [e for e in measure.elements if e.staff_no == staff_no and not e.is_grace]
    voices = {e.voice_no for e in els}
    if len(voices) < 2 or not els:
        return False
    cap = _capacity_of(score, measure)

    def xo(e):
        return (e.src_bbox[0] + e.src_bbox[2] / 2.0) if e.src_bbox else (e.src_x or 0.0)
    tol = max([0.6 * e.src_bbox[2] for e in els if e.src_bbox] or [6.0])
    groups = []
    for e in sorted(els, key=xo):
        if groups and abs(xo(groups[-1][0]) - xo(e)) <= tol:
            groups[-1].append(e)
        else:
            groups.append([e])
    total = 0.0
    for g in groups:
        if len({x.voice_no for x in g}) > 1 or len({x.quarter_length() for x in g}) > 1:
            return False                     # 같은 자리에 성부가 둘 — 진짜 다성부
        if g[0].kind == "rest" and g[0].duration_base == 1 and not g[0].dotted:
            return False
        total += g[0].quarter_length()
    if abs(total - cap) > 0.01:
        return False
    v = min(voices)
    for e in els:
        e.voice_no = v
    return True


def repair_voices(score, measure) -> bool:
    """[v1.40 성부 다시 나누기] Audiveris가 한 보표 안의 성부를 섞어 매긴 마디(예: "교회" 5마디 —
    아래 성부의 2박째 E4가 위 성부의 2분음표 뒤에 붙어 위 성부 3박·아래 성부 구멍)를 고친다.
    음마다 박 위치를 '같은 줄의 마디를 채운 줄'의 같은 자리에서 정하고, 시간 순으로 성부에 다시
    배정한다(빈틈 없이 이어지는 성부 → 원래 성부 → 비어 있는 성부 순으로). 그 결과 마디 검사가
    '정상'이 될 때만 반영하고 True를 돌려준다(아니면 원래대로 둔다)."""
    from .musicxml import measure_anchors, _slots, DIVISIONS
    cap_q = _capacity_of(score, measure)
    cap = int(round(cap_q * DIVISIONS))
    anchors = measure_anchors(measure.elements, cap)
    if not anchors:
        return False
    before = {id(e): e.voice_no for e in measure.elements}
    changed = False
    for sn in sorted({e.staff_no for e in measure.elements}):
        els = [e for e in measure.elements if e.staff_no == sn and not e.is_grace]
        if not els:
            continue
        slots = _slots(els)                       # 화음은 한 슬롯
        for sl in slots:
            near = [a for a in anchors if abs(a["x"] - sl["x"]) <= 1.3 * max(a["w"], sl["w"])]
            if not near:
                break
            sl["on"] = min(near, key=lambda a: abs(a["x"] - sl["x"]))["on"]
        else:
            orig_voices = sorted({e.voice_no for e in els})
            ends = {}
            plan = []
            for sl in sorted(slots, key=lambda q: (q["on"], -max(
                    (e.octave * 7 + "CDEFGAB".index(e.step)) if e.kind == "note" and e.step else -99
                    for e in sl["els"]))):
                ov = sl["els"][0].voice_no
                cont = [v for v, end in ends.items() if end == sl["on"]]
                if ov in cont:
                    v = ov
                elif cont:
                    v = cont[0]
                elif ends.get(ov, 0) <= sl["on"]:
                    v = ov
                else:
                    free = [v for v in orig_voices if ends.get(v, 0) <= sl["on"]]
                    v = free[0] if free else max(orig_voices + list(ends)) + 1
                ends[v] = sl["on"] + sl["dur"]
                plan.append((sl, v))
            for sl, v in plan:
                for e in sl["els"]:
                    if e.voice_no != v:
                        e.voice_no = v
                        changed = True
    if not changed:
        return False
    if validate(score).get(measure.number, ("",))[0] in ("ok", "anacrusis", "time_change"):
        return True
    for e in measure.elements:                    # 나아지지 않으면 되돌린다
        if id(e) in before:
            e.voice_no = before[id(e)]
    return False


def toggle_beam(measure, elements) -> str:
    """[v1.44 빔으로 묶기/풀기] 고른 음표(8분음표 이하, 같은 보표) 2개 이상을 한 빔으로 묶는다. 이미 모두 한
    빔이면 푼다. 화음은 구성음 모두 함께. 반환: "joined" | "split" | "unsupported".
    (Audiveris가 빔을 놓친 경우 — "교회" 15마디 왼손 G♯2–E3: 이음줄이 빔 위를 지나 빔을 못 읽음)"""
    notes = [e for e in elements if e.kind == "note" and not e.is_grace]
    if len(notes) < 2 or len({e.staff_no for e in notes}) != 1 or any(e.duration_base < 8 for e in notes):
        return "unsupported"
    groups = {e.beam_group for e in notes}
    members = list(notes)
    for e in notes:                                  # 화음 구성음도 함께
        if e.chord_group != -1:
            members += [x for x in measure.elements if x.chord_group == e.chord_group and x not in members]
    if len(groups) == 1 and -1 not in groups:
        for e in members:
            e.beam_group = -1
        return "split"
    used = [x.beam_group for mm in [measure] for x in mm.elements if isinstance(x.beam_group, int)]
    new = max(used + [0]) + 1000                     # 다른 마디의 번호와도 겹치지 않게 크게
    for e in members:
        e.beam_group = new
    return "joined"


ARTIC_KINDS = ("accent", "staccato", "tenuto", "marcato")


def toggle_articulation(measure, element, kind: str, space: float = 20.0, above: bool = True) -> str:
    """[v2.20] 음표(화음이면 화음 전체)에 아티큘레이션을 붙이거나 뗀다. 이미 있으면 지우고("removed"),
    없으면 붙인다("added"). 붙이는 자리는 above(기둥 반대쪽 — 호출부가 정함)에 따라 화음의 맨 위 음 위
    또는 맨 아래 음 아래, 머리 가운데에서 1.4칸(같은 쪽에 기호가 이미 있으면 1칸씩 더 바깥).
    space: 원본 좌표 기준 오선 한 칸(px)."""
    if element.kind != "note":
        return "unsupported"
    members = [element]
    if element.chord_group != -1:
        members = [e for e in measure.elements if e.chord_group == element.chord_group and e.kind == "note"]
    if any(a["kind"] == kind for e in members for a in (e.articulations or [])):
        for e in members:
            e.articulations = [a for a in (e.articulations or []) if a["kind"] != kind]
        return "removed"

    def dia(e):
        return e.octave * 7 + STEPS.index(e.step)
    target = max(members, key=dia) if above else min(members, key=dia)
    same_side = [a for e in members for a in (e.articulations or []) if bool(a["above"]) == bool(above)]
    off = (1.4 + len(same_side)) * space
    target.articulations = list(target.articulations or []) + [
        {"kind": kind, "above": bool(above), "dx": 0.0, "dy": -off if above else off}]
    return "added"


def beam_members(measure, element) -> list:
    """같은 마디·같은 보표에서 이 음표와 한 빔으로 묶인 요소들(화음 구성음 포함). 없으면 []."""
    bg = getattr(element, "beam_group", -1)
    if bg is None or bg == -1:
        return []
    mem = [x for x in measure.elements if x.beam_group == bg and x.staff_no == element.staff_no]
    return mem if len({(x.chord_group if x.chord_group != -1 else id(x)) for x in mem}) >= 2 else []
