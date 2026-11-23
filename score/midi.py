# midi.py
# ---------------------------------------------------------------------------
# [업그레이드 2단계] 소리로 편집하기 — Score → MIDI 이벤트 → 재생/저장.
#
#   score_events(score, bpm)   : 재생 이벤트 목록(요소 참조 포함 — 하이라이트용)
#   save_midi(score, path)     : 표준 MIDI 파일(SMF type 0) 저장 — 외부 의존성 없음
#   synth_wav(events, path)    : 내장 신디사이저로 WAV 합성(numpy만 사용)
#
# 설계 노트(PLAN.md 7장):
#   * 음높이→MIDI 변환은 조옮김(transpose_score)이 이미 쓰는 것과 동일한
#     반음 산법을 쓴다(step 반음 오프셋 + alter + 옥타브).
#   * 꾸밈음은 박자를 차지하지 않으므로(모델 규칙) 본음표 직전에서 고정
#     0.10초를 '본음표 길이에서 훔쳐' 연주한다(표준 관행 근사).
#   * 붙임줄(tie)로 이어진 같은 음은 한 번의 발음으로 합친다.
#   * 재생 검수 목적이므로 소리는 단순 명료한 순음 계열(배음 3개 + 감쇠).
# ---------------------------------------------------------------------------
import struct
import wave

import numpy as np

_STEP_SEMI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

GRACE_SEC = 0.10        # 꾸밈음 실연주 길이(초)


def element_midi(e) -> int | None:
    """모델 Element(음표) → MIDI 번호. 쉼표/미상은 None."""
    if e.kind != "note" or e.step is None or e.octave is None:
        return None
    return 12 * (e.octave + 1) + _STEP_SEMI[e.step] + (e.alter or 0)


def score_events(score, bpm: float = 100.0):
    """Score → [(t_start초, dur초, midi|None, element, measure_number), ...]

    [Tier2] 다성부 지원: 요소를 (staff_no, voice_no)별로 묶어 각각
    독립적인 시간축으로 처리하되, 마디 시작 절대시각은 모든 성부가
    공유한다(그래야 오른손·왼손이 같은 순간에 시작해서 함께 진행된다).
    화음(chord_group) 멤버는 같은 슬롯이므로 대표(첫 멤버)만 시간을
    진행시키고 나머지는 같은 시각에 추가로 발음한다.

    단일 성부 악보(Tier1)는 (staff_no,voice_no)가 항상 (1,1) 하나뿐이라
    기존과 동일하게 동작한다(회귀 없음).

    쉼표는 midi=None 으로 포함한다(하이라이트 커서가 쉼표에도 멈추도록).
    반환 목록은 시간순이며 마지막에 (총길이, 0, None, None, None) 센티널을 둔다.
    """
    spq = 60.0 / max(20.0, min(300.0, bpm))     # 4분음표 1개의 초

    # 1) 마디별 시작 절대시각(모든 성부가 공유하는 기준선)
    ordered_measures = sorted(score.measures, key=lambda m: m.number)
    measure_start_t = {}
    t_cursor = 0.0
    for meas in ordered_measures:
        measure_start_t[meas.number] = t_cursor
        cap = _cap_of(score, meas)
        t_cursor += cap * spq
    total_len = t_cursor

    # 2) (staff_no, voice_no)별로 그룹핑(마디 순서 유지)
    # [v1.41 재생 시각 — 실사용] 예전엔 성부마다 '목록 순서대로 길이를 더해' 시각을 정해, 편집으로
    # 넣은 음(목록 끝에 붙음)이나 마디 처음보다 늦게 시작하는 성부(앞 쉼표 생략)가 엉뚱한 때에 울렸다.
    # 이제 내보내기·마디 검사와 같은 박 배치(element_onsets)로 각 음의 시작 시각을 정한다.
    from collections import defaultdict
    from .musicxml import element_onsets, DIVISIONS as _DIV
    onset = {}
    from .editing import _capacity_of
    for meas in ordered_measures:
        onset.update(element_onsets(meas, int(round(_capacity_of(score, meas) * _DIV))))
    by_key = defaultdict(list)
    for meas in ordered_measures:
        for e in sorted(meas.elements, key=lambda e: (onset.get(id(e), 0), e.src_x)):
            key = (getattr(e, "staff_no", 1), getattr(e, "voice_no", 1))
            by_key[key].append((meas, e))

    events = []
    for key, items in by_key.items():
        t = 0.0
        cur_meas_num = None
        pending_tie = None
        graces = []
        seen_chords = set()
        chord_start = {}      # {chord_group: 대표 음이 울린 시각} — 나머지 구성음도 같은 시각에

        def flush_tie():
            nonlocal pending_tie
            if pending_tie:
                events.append(pending_tie)
                pending_tie = None

        for meas, e in items:
            if meas.number != cur_meas_num:
                cur_meas_num = meas.number
                seen_chords.clear()   # 화음 그룹id는 마디마다 재사용될 수 있음
                chord_start.clear()
            gid0 = getattr(e, "chord_group", -1)
            if not (gid0 != -1 and gid0 in seen_chords) and not getattr(e, "is_grace", False):
                # 이 음의 박 위치(앞 음과의 사이가 비면 그만큼 쉰다 — 생략 쉼표)
                t = measure_start_t[meas.number] + onset.get(id(e), 0) / _DIV * spq

            # 화음 구성음: 대표(그룹의 첫 발견 멤버)만 시간을 진행시키고,
            # 나머지는 같은 시각(t 진행 전)에 추가로 발음만 낸다.
            gid = getattr(e, "chord_group", -1)
            is_extra_chord_member = (gid != -1 and gid in seen_chords)
            if gid != -1:
                seen_chords.add(gid)

            if getattr(e, "is_grace", False) and not is_extra_chord_member:
                graces.append((e, meas.number))
                continue

            dur = e.quarter_length() * spq
            if dur <= 0 and not is_extra_chord_member:
                continue
            midi = element_midi(e)

            if is_extra_chord_member:
                # [화음 재생 버그 수정 — 실사용 피드백 "화음이 한 번에 안 들려요"] 예전엔 여기서
                # 현재 t를 썼는데, 대표 음을 처리하면서 t가 이미 그 음 길이만큼 넘어가 있어서
                # 나머지 구성음이 대표 음이 '끝난 뒤'에 울렸다. 대표 음의 시작 시각을 쓴다.
                if midi is not None:
                    events.append((chord_start.get(gid, t), dur, midi, e, meas.number))
                continue

            # 꾸밈음: 본음표 길이 앞부분을 잘라 짧게 연주
            if graces and midi is not None:
                g_total = min(GRACE_SEC * len(graces), dur * 0.4)
                g_each = g_total / len(graces)
                gt = t
                for ge, gmn in graces:
                    gm = element_midi(ge)
                    if gm is not None:
                        flush_tie()
                        events.append((gt, g_each, gm, ge, gmn))
                    gt += g_each
                t += g_total
                dur -= g_total
            graces.clear()
            if gid != -1:
                chord_start[gid] = t              # 이 화음의 다른 구성음도 이 시각에 울린다

            if midi is None:                      # 쉼표
                flush_tie()
                events.append((t, dur, None, e, meas.number))
            elif pending_tie and pending_tie[2] == midi and \
                    getattr(pending_tie[3], "tie_start", False):
                # [버그 수정] pending_tie는 (t, dur, midi, element, measure)
                # 순서인데, 예전 코드가 이 언패킹 변수명을 (midi, start,
                # dur, ...)처럼 잘못 붙여서 t/dur/midi 값이 완전히 뒤섞이는
                # 사고가 있었다(실사용 검증 중 발견 — 이상한 63초짜리 음,
                # 소수점 MIDI 번호로 나타남). 실제 튜플 순서에 정확히
                # 맞춘 변수명으로 고친다.
                t0, dur0, midi0, el0, mn0 = pending_tie
                pending_tie = (t0, dur0 + dur, midi0, el0, mn0)
            else:
                flush_tie()
                pending_tie = (t, dur, midi, e, meas.number)
            t += dur
        flush_tie()

    events.sort(key=lambda ev: ev[0])
    # [v2.20] 스타카토 등은 짧게 울린다(박 위치는 그대로)
    arts = _art_kinds(score)
    if arts:
        for i, ev in enumerate(events):
            st, dur, mi, el, mn = ev
            ks = arts.get(id(el), ()) if el is not None else ()
            r = min([_ART_LEN.get(k, 1.0) for k in ks] + [1.0])
            if mi is not None and r < 1.0:
                events[i] = (st, dur * r, mi, el, mn)
    events = _apply_marks(score, events, ordered_measures, spq)       # [v2.80] 옥타브·페달·꾸밈·숨표·아르페지오
    # [v2.80] 연주 순서(도돌이·볼타·D.C./D.S.·Fine·Coda)로 마디를 다시 늘어놓는다
    from .playorder import performance_order
    order = performance_order(score)
    if [m.number for m in order] != [m.number for m in ordered_measures]:
        by_m = {}
        for ev in events:
            by_m.setdefault(ev[4], []).append(ev)
        out, T = [], 0.0
        for meas in order:
            cap = _cap_of(score, meas)
            base = measure_start_t.get(meas.number, 0.0)
            for st, dur, mi, el, mn in by_m.get(meas.number, []):
                out.append((T + (st - base), dur, mi, el, mn))
            T += cap * spq
        events = sorted(out, key=lambda ev: ev[0])
        total_len = T
    events.append((total_len, 0.0, None, None, None))
    return events


# ------------------------------------------------------------------ SMF 저장
# [셈여림 → 재생 세기] 내부 키 → MIDI velocity. 셈여림이 없으면 mf(80) 정도로 연주한다.
DYN_VELOCITY = {"ppp": 25, "pp": 36, "p": 50, "mp": 64, "mf": 80, "f": 96, "ff": 110, "fff": 122}
_ACCENT_KEYS = {"sfz", "sf", "fz", "rfz"}       # 그 음만 강하게(이후 세기는 그대로)
DEFAULT_VELOCITY = 80


_STEPS7 = ["C", "D", "E", "F", "G", "A", "B"]


def _neighbor_midi(score, meas, e, delta):
    """[v2.80] 조표를 따른 위(+1)·아래(-1) 이웃음의 MIDI 번호(꾸밈 재생용)."""
    try:
        from .model import _alter_map
        idx = _STEPS7.index(e.step) + delta
        step, octave = _STEPS7[idx % 7], e.octave + idx // 7
        alt = _alter_map(score.key_at(meas.number)).get(step, 0)
        return 12 * (octave + 1) + _STEP_SEMI[step] + alt
    except Exception:
        return None


def _apply_marks(score, events, measures, spq):
    """[v2.80] 재생에 반영: 옥타브 기호(범위 안 음 ±12/24), 페달(누른 동안 시작한 음은 뗄 때까지 울림),
    트릴·모르덴트·턴(꾸밈음으로 쪼갬), 숨표(길이 ¾), 아르페지오(화음 구성음을 아래부터 조금씩 늦게)."""
    if not events:
        return events
    pos = {}
    for mi_, m in enumerate(measures):
        for e in m.elements:
            pos[id(e)] = (mi_, e.src_x, m)
    # 옥타브
    shift = {}
    for ot in (getattr(score, "ottavas", None) or []):
        a, b = ot.get("start"), ot.get("stop")
        if a is None or id(a) not in pos:
            continue
        pa = pos[id(a)][:2]
        pb = pos[id(b)][:2] if (b is not None and id(b) in pos) else (10 ** 9, 0)
        d = {"8va": 12, "8vb": -12, "15ma": 24}.get(ot["kind"], 0)
        for m in measures:
            for e in m.elements:
                p_ = pos[id(e)][:2]
                if e.staff_no == a.staff_no and pa <= p_ <= pb:
                    shift[id(e)] = d
    # 페달 구간(초)
    t_of = {id(ev[3]): ev[0] for ev in events if ev[3] is not None}
    ped = []
    for pd in (getattr(score, "pedals", None) or []):
        a, b = pd.get("start"), pd.get("stop")
        if a is not None and id(a) in t_of:
            t0 = t_of[id(a)]
            t1 = t_of.get(id(b)) if b is not None else None
            t1 = t1 if t1 is not None else t0 + 4 * spq
            # [v2.90] 다시 밟기(계단형 페달 선의 꼭지): 그 자리에서 떼었다 다시 밟는다
            cuts = sorted(t_of[id(c)] for c in (pd.get("changes") or []) if id(c) in t_of and t0 < t_of[id(c)] < t1)
            bounds = [t0] + cuts + [t1]
            for u, v in zip(bounds, bounds[1:]):
                ped.append((u, v))
    gliss = {}
    for gl in (getattr(score, "glissandos", None) or []):        # [v2.90]
        from_, to_ = gl.get("start"), gl.get("stop")
        if from_ is not None and to_ is not None:
            tm = element_midi(to_)
            if tm is not None:
                gliss[id(from_)] = tm + shift.get(id(to_), 0)
    arts = {}
    own = {}                                  # 꾸밈(트릴·모르덴트·턴)은 그 기호가 붙은 음에만
    trem = {}                                 # [v2.90] 트레몰로 빗금 수
    for m in measures:
        for e in m.elements:
            ks = [a["kind"] for a in (getattr(e, "articulations", None) or [])]
            own[id(e)] = set(ks)
            for a in (getattr(e, "articulations", None) or []):
                if a["kind"] == "tremolo":           # 화음이면 구성음 모두
                    mem_ = [e] if e.chord_group == -1 else [x for x in m.elements if x.chord_group == e.chord_group]
                    for x in mem_:
                        trem[id(x)] = int(a.get("n", 1))
            if e.chord_group != -1 and ks:
                for x in m.elements:
                    if x.chord_group == e.chord_group:
                        arts.setdefault(id(x), set()).update(ks)
            elif ks:
                arts.setdefault(id(e), set()).update(ks)
    out = []
    chord_seen = {}
    for st, dur, mi, el, mn in events:
        if mi is None or el is None:
            out.append((st, dur, mi, el, mn))
            continue
        mi = mi + shift.get(id(el), 0)
        for t0, t1 in ped:
            if t0 - 1e-6 <= st < t1:
                dur = max(dur, t1 - st)
        ks = arts.get(id(el), set())
        meas = pos.get(id(el), (0, 0, None))[2]
        if "breath" in ks:
            dur *= 0.75
        if "caesura" in ks:                    # [v2.90] 끊고 쉬기
            dur *= 0.5
        if id(el) in gliss and mi is not None:  # [v2.90] 글리산도: 앞 절반은 그대로, 뒤 절반은 반음씩 미끄러짐
            tgt = gliss[id(el)]
            half = dur / 2.0
            out.append((st, half, mi, el, mn))
            steps = list(range(mi + (1 if tgt > mi else -1), tgt, 1 if tgt > mi else -1))
            if steps:
                u = half / len(steps)
                for k_, p_ in enumerate(steps):
                    out.append((st + half + k_ * u, u, p_, el, mn))
            continue
        tn = trem.get(id(el))
        if tn:                                 # [v2.90] 트레몰로: 빗금 1·2·3개 = 8·16·32분음표로 잘게 반복
            u = spq / (2 ** tn)
            t = st
            while t < st + dur - 1e-6:
                out.append((t, min(u, st + dur - t) * 0.9, mi, el, mn))
                t += u
            continue
        if "arpeggio" in ks and el.chord_group != -1:
            key = (mn, el.staff_no, el.chord_group)
            k_ = chord_seen.get(key, 0)
            chord_seen[key] = k_ + 1
            dly = min(0.04 * k_, dur * 0.5)
            out.append((st + dly, dur - dly, mi, el, mn))
            continue
        oks = own.get(id(el), set())
        if ("trill" in oks or "mordent" in oks or "turn" in oks) and meas is not None:
            ks = oks
            up_ = _neighbor_midi(score, meas, el, +1)
            lo_ = _neighbor_midi(score, meas, el, -1)
            if up_ is not None:
                up_ += shift.get(id(el), 0)
            if lo_ is not None:
                lo_ += shift.get(id(el), 0)
            if "trill" in ks and up_ is not None:
                u = max(0.06, min(0.12, spq / 8))
                t, k_ = st, 0
                while t < st + dur - 1e-6:
                    out.append((t, min(u, st + dur - t), mi if k_ % 2 == 0 else up_, el, mn))
                    t += u
                    k_ += 1
                continue
            if "mordent" in ks and up_ is not None:
                u = min(0.08, dur / 4)
                out += [(st, u, mi, el, mn), (st + u, u, up_, el, mn), (st + 2 * u, dur - 2 * u, mi, el, mn)]
                continue
            if "turn" in ks and up_ is not None and lo_ is not None:
                q = dur / 4
                out += [(st, q, up_, el, mn), (st + q, q, mi, el, mn), (st + 2 * q, q, lo_, el, mn),
                        (st + 3 * q, q, mi, el, mn)]
                continue
        out.append((st, dur, mi, el, mn))
    return sorted(out, key=lambda ev: ev[0])


def _art_kinds(score) -> dict:
    """[v2.20] {id(Element): 아티큘레이션 종류 집합} — 화음은 구성음 모두 같은 집합."""
    out = {}
    for m in score.measures:
        by_cg = {}
        for e in m.elements:
            ks = {a["kind"] for a in (getattr(e, "articulations", None) or [])}
            if e.chord_group != -1:
                by_cg.setdefault(e.chord_group, set()).update(ks)
            elif ks:
                out[id(e)] = ks
        for e in m.elements:
            if e.chord_group != -1 and by_cg.get(e.chord_group):
                out[id(e)] = by_cg[e.chord_group]
    return out


# 아티큘레이션 → 세기 더하기 / 길이 비율(재생·MIDI)
_ART_VEL = {"accent": 20, "marcato": 25}
_ART_LEN = {"staccato": 0.5, "staccatissimo": 0.35, "marcato": 0.75}


def velocity_map(score) -> dict:
    """{id(Element): velocity}. 보표별로 악보 순서대로 셈여림을 따라가며 세기를 정하고,
    크레셴도/디크레셴도 구간은 시작 세기→목표 세기로 고르게 바꾼다(목표 = 끝 바로 다음의
    셈여림, 없으면 시작보다 ±24)."""
    seq = {}                                   # {staff: [Element, ...]} 악보 순서
    for m in score.measures:
        for e in m.elements:
            seq.setdefault(getattr(e, "staff_no", 1), []).append(e)
    vmap = {}
    arts = _art_kinds(score)
    for staff, els in seq.items():
        level = DEFAULT_VELOCITY
        base = []
        for e in els:
            key = (getattr(e, "dynamic", None) or {}).get("key")
            v = level
            if key in DYN_VELOCITY:
                level = v = DYN_VELOCITY[key]
            elif key in _ACCENT_KEYS:
                v = max(level + 24, 105)
            elif key == "fp":
                v, level = 100, DYN_VELOCITY["p"]
            base.append(v)
        pos = {id(e): i for i, e in enumerate(els)}
        for hp in getattr(score, "hairpins", []):
            i0, i1 = pos.get(id(hp.start)), pos.get(id(hp.stop))
            if i0 is None or i1 is None or i1 <= i0:
                continue
            v0 = base[i0]
            nxt = els[i1 + 1] if i1 + 1 < len(els) else None
            nkey = (getattr(nxt, "dynamic", None) or {}).get("key") if nxt is not None else None
            if nkey in DYN_VELOCITY:
                v1 = DYN_VELOCITY[nkey]
            else:
                v1 = v0 + (24 if hp.kind == "cresc" else -24)
                # 다음 셈여림이 나올 때까지는 바뀐 세기를 유지
                k = i1 + 1
                while k < len(els) and not (getattr(els[k], "dynamic", None) or {}).get("key"):
                    base[k] = v1
                    k += 1
            for k in range(i0, i1 + 1):
                base[k] = v0 + (v1 - v0) * (k - i0) / (i1 - i0)
        for e, v in zip(els, base):
            v += max([_ART_VEL.get(k, 0) for k in arts.get(id(e), ())] + [0])   # [v2.20] 악센트·마르카토
            vmap[id(e)] = int(max(15, min(127, round(v))))
    return vmap


def _vlq(n: int) -> bytes:
    """MIDI 가변 길이 수량 인코딩."""
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


CHORD_ACCOMP = False      # [v3.00] 재생·MIDI에 기타 코드 반주를 넣을지(툴바 '코드 반주')


def chord_events(score, events, bpm: float = 100.0):
    """[v3.00] 코드 반주: 코드가 붙은 음이 울리는 때(도돌이로 여러 번이면 매번)부터 다음 코드 전까지(최대 두 마디)
    근음 자리 화음(가온 C 아래)과 베이스(분수 코드면 '/' 뒤 음)를 부드럽게. 하이라이트와 섞이지 않게 요소는 None."""
    from . import chords as _chd
    spq = 60.0 / max(20.0, min(300.0, bpm))
    starts = []
    seen = set()
    for st, dur, mi, el, mn in events:
        h = getattr(el, "harmony", None) if el is not None else None
        if h and (id(el), round(st, 4)) not in seen:
            seen.add((id(el), round(st, 4)))
            starts.append((st, h, mn))
    starts.sort(key=lambda t: t[0])
    total = events[-1][0] if events else 0.0
    out = []
    for i, (st, h, mn) in enumerate(starts):
        nxt = starts[i + 1][0] if i + 1 < len(starts) else total
        dur = max(0.05, min(nxt - st, 8 * spq))
        root_pc, tns, bass_pc = _chd.tones(h)
        base = 48 + root_pc                       # C3~B3
        for iv in tns:
            out.append((st, dur * 0.95, base + iv, None, mn))
        out.append((st, dur * 0.95, 36 + (bass_pc if bass_pc is not None else root_pc), None, mn))
    return out


def save_midi(score, path: str, bpm: float = 100.0, program: int = 0):
    """표준 MIDI 파일(type 0, 480tpq) 저장. program: GM 악기(0=피아노)."""
    TPQ = 480
    spq = 60.0 / max(20.0, min(300.0, bpm))
    events = score_events(score, bpm)
    if CHORD_ACCOMP:
        events = sorted(events + chord_events(score, events, bpm), key=lambda ev: ev[0])
    vmap = velocity_map(score)
    msgs = []   # (tick, priority, bytes)  priority: note_off(0) < note_on(1)
    for st, dur, midi, _el, _mn in events:
        if midi is None or dur <= 0:
            continue
        t0 = int(round(st / spq * TPQ))
        t1 = int(round((st + dur * 0.98) / spq * TPQ))   # 미세 갭(재발음 구분)
        vel = vmap.get(id(_el), DEFAULT_VELOCITY)
        msgs.append((t0, 1, bytes((0x90, midi, vel))))
        msgs.append((max(t0 + 1, t1), 0, bytes((0x80, midi, 0))))
    msgs.sort(key=lambda m: (m[0], m[1]))

    track = bytearray()
    tempo = int(round(60_000_000 / bpm))
    track += _vlq(0) + bytes((0xFF, 0x51, 0x03)) + tempo.to_bytes(3, "big")
    track += _vlq(0) + bytes((0xFF, 0x58, 0x04,
                              score.beats, {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}
                              .get(score.beat_type, 2), 24, 8))
    track += _vlq(0) + bytes((0xC0, program))
    prev = 0
    for tick, _pri, data in msgs:
        track += _vlq(tick - prev) + data
        prev = tick
    track += _vlq(0) + bytes((0xFF, 0x2F, 0x00))

    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, TPQ))
        f.write(b"MTrk" + struct.pack(">I", len(track)) + bytes(track))


# ------------------------------------------------------------------ WAV 합성
SR = 22050

def synth_wav(events, path: str, sr: int = SR, vmap=None):
    """이벤트 → 간단한 피아노풍 합성(배음 3개 + 지수 감쇠) WAV 저장.

    외부 사운드폰트 없이 검수용으로 충분한 명료한 음색을 목표로 한다.
    """
    total = max((st + dur for st, dur, *_ in events), default=0.0) + 0.4
    buf = np.zeros(int(total * sr) + 1, np.float32)
    for st, dur, midi, _el, _mn in events:
        if midi is None or dur <= 0:
            continue
        f0 = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        # 잔향 포함 실제 발음 길이(다음 음과 살짝 겹쳐 자연스럽게)
        play = min(dur + 0.15, dur * 1.25 + 0.05)
        n = int(play * sr)
        tt = np.arange(n, dtype=np.float32) / sr
        wavef = (1.00 * np.sin(2 * np.pi * f0 * tt)
                 + 0.35 * np.sin(2 * np.pi * 2 * f0 * tt)
                 + 0.15 * np.sin(2 * np.pi * 3 * f0 * tt))
        env = np.exp(-tt * 3.2).astype(np.float32)
        atk = int(0.005 * sr)
        if atk > 0 and n > atk:
            env[:atk] *= np.linspace(0.0, 1.0, atk, dtype=np.float32)
        rel = int(0.03 * sr)
        if rel > 0 and n > rel:
            env[-rel:] *= np.linspace(1.0, 0.0, rel, dtype=np.float32)
        i0 = int(st * sr)
        amp = 0.22
        if vmap is not None:                      # 셈여림에 따른 세기(mf=80 기준)
            amp *= (vmap.get(id(_el), DEFAULT_VELOCITY) / DEFAULT_VELOCITY) ** 1.3
        seg = (wavef * env * amp).astype(np.float32)
        buf[i0:i0 + n] += seg[:len(buf) - i0]
    peak = float(np.max(np.abs(buf))) or 1.0
    if peak > 0.98:
        buf *= 0.98 / peak
    pcm = (buf * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _cap_of(score, meas):
    """마디 길이(4분음표=1) — 편집기 검사와 같은 규칙(줄 실효 길이·박자 바뀜·[v3.63] 자유 리듬)."""
    from .editing import _capacity_of
    return _capacity_of(score, meas)
