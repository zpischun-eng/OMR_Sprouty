# score/musicxml.py  (v1.20 재작성)
# ---------------------------------------------------------------------------
# score.model.Score → MusicXML 3.1 partwise. 외부 라이브러리 없이 직접 만든다.
#
# v1.00까지는 곡 전체를 '파트 1개 · 보표 1개 · 성부 1개'로 이어 붙여, 노래+피아노처럼 보표가
# 여럿인 곡을 내보내면 모든 음이 한 줄로 이어졌다. v1.20은 편집 화면과 같은 구조로 내보낸다.
#
#   파트   : Audiveris의 논리 파트(score.staff_parts)마다 <part>. 이름도 Audiveris 것을 쓴다.
#   보표   : 파트 안 보표 수(<staves>)와 보표별 음자리표. 줄마다 보표 수가 다르면(예: 합창이
#            한 줄은 SA, 다른 줄은 TB) 음자리표가 맞는 자리에 배정하고, 없는 보표는 온마디 쉼표.
#   성부   : 보표·성부별로 <backup>/<forward>로 시간 위치를 맞춘다. 성부가 마디 처음에서 시작하지
#            않으면(생략 쉼표) 같은 자리(x)에 있는 다른 성부 음의 박 위치로 시작점을 정한다.
#            시간이 겹치지 않는 성부는 한 성부로 합친다(Audiveris가 위 선율을 마디 중간부터 다른
#            번호로 잇거나, 잇단음표 안 음마다 번호를 달리 매기는 경우). 겹치는 성부는 위 성부가
#            먼저(MusicXML voice 1 = 기둥 위, 2 = 기둥 아래 — 다른 악보 프로그램의 관례).
#   그 밖  : 화음(<chord/>), 잇단(<time-modification>·<tuplet>), 겹점, 꾸밈음, 붙임줄·이음줄,
#            빔, 늘임표, 셈여림·쐐기, 가사, 조표·박자표·음자리표 바뀜(마디 중간 포함),
#            줄바꿈(<print new-system>), 도돌이·볼타·종지선.
#
# 마디 박자가 박자표와 안 맞아도(=인식 오류) 그대로 내보낸다(편집 화면에서 고치는 것이 전제).
# ---------------------------------------------------------------------------
import xml.etree.ElementTree as ET
from fractions import Fraction
from xml.dom import minidom

from .model import tuplet_groups, dot_factor

# 4분음표 = DIVISIONS. 8·3·5·7의 공배수(점 32분, 3·5·7잇단 모두 정수).
DIVISIONS = 1680
_DUR = {1: 6720, 2: 3360, 4: 1680, 8: 840, 16: 420, 32: 210}
_CLEF_XML = {"G": ("G", 2), "F": ("F", 4), "C": ("C", 3)}
_LYRIC_EXTEND = {"-", "\u2013", "\u2014", "_", "~", "\u2015", "\u2012"}
_GRID = 48          # 성부 시작점을 맞출 박 격자(4분음표의 1/48 — 16분·셋잇단·32분 모두 표현)


def _sub(parent, tag, text=None, **attrs):
    e = ET.SubElement(parent, tag, {k.replace("_", "-"): str(v) for k, v in attrs.items()})
    if text is not None:
        e.text = str(text)
    return e


def _dur_div(el) -> int:
    """요소 하나의 길이(divisions). 꾸밈음은 0."""
    if getattr(el, "is_grace", False):
        return 0
    d = Fraction(_DUR.get(el.duration_base, DIVISIONS)) * Fraction(dot_factor(el)).limit_denominator(8)
    if getattr(el, "in_triplet", False) and getattr(el, "tup_actual", 0):
        d = d * el.tup_normal / el.tup_actual
    return int(round(d))


def _capacity_div(score, measure) -> int:
    cap = measure.target_capacity if measure.target_capacity is not None else None
    if cap is None:
        cap = _time_at(score, measure)[0] * 4.0 / _time_at(score, measure)[1]
    return int(round(cap * DIVISIONS))


def _time_at(score, measure):
    t = (score.beats, score.beat_type)
    for m in score.measures:
        if m.number > measure.number:
            break
        if getattr(m, "time_sig_change", None):
            t = m.time_sig_change
    return t


def _x_of(e):
    if getattr(e, "disp_x", None) is not None:
        return e.disp_x
    if e.src_x:
        return e.src_x
    if e.src_bbox:
        return e.src_bbox[0] + e.src_bbox[2] / 2.0
    return 0


def _dia(e):
    return e.octave * 7 + "CDEFGAB".index(e.step) if e.kind == "note" and e.step else None


# --------------------------------------------------------------- 파트·보표 구성
def _part_layout(score):
    """반환: (파트 순서 [p...], {p: 보표 수}, {(줄, 화면보표번호): (p, 파트안보표)}, {p: 이름})."""
    staff_parts = getattr(score, "staff_parts", None) or {}
    staff_clefs = getattr(score, "staff_clefs", None) or {}
    lines = sorted({m.system for m in score.measures})
    max_sn = {}
    for m in score.measures:
        for e in m.elements:
            max_sn[m.system] = max(max_sn.get(m.system, 1), getattr(e, "staff_no", 1))
    line_parts = {}
    for ln in lines:
        pl = list(staff_parts.get(ln) or [])
        n = max(len(pl), max_sn.get(ln, 1))
        while len(pl) < n:                       # 정보가 없으면 같은 파트(피아노 등)로 본다
            pl.append(pl[-1] if pl else 0)
        line_parts[ln] = pl
    parts = sorted({p for pl in line_parts.values() for p in pl})
    n_staves = {p: max(pl.count(p) for pl in line_parts.values()) for p in parts}
    # 보표 수가 가장 많은 줄의 음자리표를 그 파트의 '자리별 대표 음자리표'로 삼는다
    typical = {}
    for p in parts:
        for ln in lines:
            pl = line_parts[ln]
            if pl.count(p) == n_staves[p]:
                cl = list(staff_clefs.get(ln) or [])
                typical[p] = [cl[i] if i < len(cl) else "G" for i, q in enumerate(pl) if q == p]
                break
    mapping = {}
    for ln in lines:
        pl = line_parts[ln]
        cl = list(staff_clefs.get(ln) or [])
        for p in parts:
            sns = [i + 1 for i, q in enumerate(pl) if q == p]
            if len(sns) == n_staves[p]:
                for k, sn in enumerate(sns):
                    mapping[(ln, sn)] = (p, k + 1)
                continue
            free = list(range(1, n_staves[p] + 1))
            for sn in sns:                       # 음자리표가 맞는 자리 → 없으면 앞자리부터
                c = cl[sn - 1] if sn - 1 < len(cl) else "G"
                pick = next((k for k in free if typical.get(p, [])[k - 1:k] == [c]), free[0])
                free.remove(pick)
                mapping[(ln, sn)] = (p, pick)
    names = list(getattr(score, "part_names", None) or [])
    pname = {p: (names[p] if 0 <= p < len(names) and names[p] else
                 ("Melody" if len(parts) == 1 else f"Part {i + 1}")) for i, p in enumerate(parts)}
    return parts, n_staves, mapping, pname, line_parts


# --------------------------------------------------------------- 성부 시간 배치
def _slots(elements):
    """성부 하나의 요소 → 슬롯(화음은 한 슬롯) 목록, x 순. 꾸밈음은 따로 표시."""
    by_key, order = {}, []
    for e in sorted(elements, key=_x_of):
        cg = getattr(e, "chord_group", -1)
        key = ("c", cg) if cg != -1 else ("e", id(e))
        if key not in by_key:
            by_key[key] = []
            order.append(key)
        by_key[key].append(e)
    out = []
    for k in order:
        mem = by_key[k]
        rep = mem[0]
        out.append({"els": mem, "x": min(_x_of(e) for e in mem), "dur": _dur_div(rep),
                    "grace": bool(getattr(rep, "is_grace", False)), "w":
                    max((e.src_bbox[2] for e in mem if e.src_bbox), default=20)})
    return out


def _snap(v):
    q = Fraction(v, DIVISIONS)
    q = Fraction(round(q * _GRID), _GRID)
    return int(q * DIVISIONS)


def measure_anchors(elements, cap):
    """[v1.32] 마디 전체(모든 보표)에서 '마디를 꼭 채운 성부'의 슬롯 = 박 위치 기준점.
    보표가 달라도 같은 줄의 세로 정렬은 같으므로, 혼자 짧은 보표(예: 왼손 3박째 2분음표 하나)의
    시작 위치를 정하는 데 쓴다."""
    by_staff = {}
    for e in elements:
        by_staff.setdefault(e.staff_no, {}).setdefault(e.voice_no, []).append(e)
    out = []
    for sn, vs in by_staff.items():
        vv = {v: _slots(es) for v, es in vs.items()}
        order = _place_voices(vv, cap)
        for lane in _lanes(vv, order):           # 이어지는 성부는 합쳐서 본다
            sl = [x for x in lane if not x["grace"]]
            if not sl or sl[0]["on"] != 0:
                continue
            t, ok = 0, True
            for x in sl:                          # 빈틈 없이 마디를 꼭 채운 줄만
                if x["on"] != t:
                    ok = False
                    break
                t += x["dur"]
            if ok and t == cap:
                out.extend(sl)
    return out


def element_onsets(measure, cap):
    """[v1.41] 마디 안 요소마다 시작 위치(divisions). 내보내기·마디 검사와 같은 배치(같은 줄의 마디를
    채운 줄을 기준으로 한 박 위치)를 재생에도 쓰도록 한다. {id(Element): 시작}"""
    anchors = measure_anchors(measure.elements, cap)
    by_staff = {}
    for e in measure.elements:
        by_staff.setdefault(e.staff_no, {}).setdefault(e.voice_no, []).append(e)
    out = {}
    for sn, vs in by_staff.items():
        vv = {v: _slots(es) for v, es in vs.items()}
        _place_voices(vv, cap, anchors)
        for sl in vv.values():
            for s in sl:
                for e in s["els"]:
                    out[id(e)] = s["on"]
    return out


def _place_voices(voices, cap, ext=None):
    """voices: {voice_no: [slot...]} → 각 슬롯에 'on'(마디 안 시작 위치, divisions)을 매긴다.
    가장 긴 성부를 마디 처음부터 놓고, 나머지는 같은 자리(x)의 음에 맞춰 시작점을 정한다."""
    # 마디 처음(가장 왼쪽)에서 시작하는 성부들을 먼저 0에 놓고, 나머지는 왼쪽부터 차례로 놓는다
    firsts = {v: voices[v][0] for v in voices if voices[v]}
    x_min = min((f["x"] for f in firsts.values()), default=0)
    starters = [v for v, f in firsts.items() if f["x"] - x_min <= f["w"] * 0.8]
    total_of = {v: sum(s["dur"] for s in voices[v]) for v in voices}
    order = sorted(starters, key=lambda v: (-total_of[v], v)) + \
        sorted((v for v in firsts if v not in starters), key=lambda v: firsts[v]["x"])
    anchors = []                                  # (x, on, w)
    placed = []                                   # (마지막 x, 끝 위치)
    for vi, v in enumerate(order):
        slots = voices[v]
        if not slots:
            continue
        cont = [end for last_x, end in placed
                if last_x < slots[0]["x"] and end + total_of[v] == cap]
        ea = None
        if ext and total_of[v] < cap:
            near_e = [a for a in ext if abs(a["x"] - slots[0]["x"]) <= 1.1 * max(a["w"], slots[0]["w"])]
            if near_e:
                ea = min(near_e, key=lambda a: abs(a["x"] - slots[0]["x"]))["on"]
        if ea is not None and ea + total_of[v] <= cap:
            pos = ea                              # 다른 보표의 같은 자리 박 위치
        elif v in starters or not anchors:
            pos = 0
        elif cont:
            # [이어지는 성부] 앞 성부가 끝난 오른쪽에서 시작하고 둘을 합치면 마디가 꼭 찬다 →
            # 그 성부의 뒷부분(Audiveris가 선율 중간부터 다른 성부 번호를 매긴 경우, "기도의 능력" 2마디)
            pos = max(cont)
        else:
            x0 = slots[0]["x"]
            near = min(anchors, key=lambda a: abs(a[0] - x0))
            tol = max(slots[0]["w"], near[2]) * 0.8      # 온음표처럼 넓은 머리도 같은 자리로
            if abs(near[0] - x0) <= tol:
                pos = near[1]
            else:                                 # 앞뒤 기준점 사이를 x 비율로 나눈다
                left = [a for a in anchors if a[0] < x0]
                right = [a for a in anchors if a[0] > x0]
                if left and right:
                    a, b = max(left), min(right)
                    pos = a[1] + (b[1] - a[1]) * (x0 - a[0]) / max(1e-9, b[0] - a[0])
                elif left:
                    a = max(left)
                    pos = a[1] + (cap - a[1]) * 0.5
                else:
                    pos = 0
                pos = max(0, min(cap, _snap(int(pos))))
            # 성부 전체가 마디 안에 들어가는 길이면, 마디를 넘지 않게 시작점을 당긴다
            # (x로 어림한 시작점이 조금 늦으면 마디선을 넘는다 — "교회" 3마디 왼손 온음표)
            total = sum(s["dur"] for s in slots)
            if total <= cap:
                pos = min(pos, cap - total)
        for s in slots:
            s["on"] = pos
            if not s["grace"]:
                anchors.append((s["x"], pos, s["w"]))
            pos += s["dur"]
        placed.append((slots[-1]["x"], pos))
    return order


def _lanes(voices, order):
    """시간이 겹치지 않는 성부를 한 줄(lane)로 합친다. 겹치는 성부는 높은 음 쪽이 앞 lane."""
    def mean_dia(v):
        ds = [_dia(e) for s in voices[v] for e in s["els"] if _dia(e) is not None]
        return sum(ds) / len(ds) if ds else -999
    lanes = []                                    # [[slots...], ...]
    for v in sorted(order, key=lambda v: -mean_dia(v)):
        iv = [(s["on"], s["on"] + max(s["dur"], 1)) for s in voices[v] if not s["grace"]]
        for lane in lanes:
            occ = [(s["on"], s["on"] + max(s["dur"], 1)) for s in lane if not s["grace"]]
            if all(b1 <= a2 or b2 <= a1 for a1, b1 in iv for a2, b2 in occ):
                lane.extend(voices[v])
                break
        else:
            lanes.append(list(voices[v]))
    for lane in lanes:
        lane.sort(key=lambda s: (s["on"], 0 if s["grace"] else 1, s["x"]))
    return lanes


# --------------------------------------------------------------- 본체
def score_to_element(score, title="악보 인식 결과"):
    root = ET.Element("score-partwise", version="3.1")
    meta = getattr(score, "meta", None) or {}
    _sub(_sub(root, "work"), "work-title", meta.get("title") or title)
    if any(meta.get(k) for k in ("lyricist", "composer", "translator", "credit")):
        ident = _sub(root, "identification")
        for k, typ in (("lyricist", "lyricist"), ("composer", "composer"), ("translator", "translator")):
            if meta.get(k):
                _sub(ident, "creator", meta[k], type=typ)
        if not (meta.get("lyricist") or meta.get("composer")) and meta.get("credit"):
            _sub(ident, "creator", meta["credit"], type="composer")

    parts, n_staves, mapping, pname, line_parts = _part_layout(score)
    part_list = _sub(root, "part-list")
    for p in parts:
        sp = _sub(part_list, "score-part", id=f"P{p + 1}")
        _sub(sp, "part-name", pname[p])

    live = {id(e) for m in score.measures for e in m.elements}
    marks = {}
    # [v1.42] 한 음에서 이음줄 둘이 시작할 수 있다("교회" 5마디 G♯4 — 위 긴 곡선과 아래 곡선). 이음줄마다
    # 번호(1~6)를 따로 매겨 시작·끝 짝이 섞이지 않게 한다(예전엔 성부 번호 하나라 하나로 합쳐졌다).
    slur_no = 0
    for arc in getattr(score, "arcs", []):
        if id(arc.start) in live and id(arc.stop) in live:
            if arc.kind == "tie":
                marks.setdefault(id(arc.start), set()).add(("tied", "start", 0))
                marks.setdefault(id(arc.stop), set()).add(("tied", "stop", 0))
            else:
                slur_no = slur_no % 6 + 1
                marks.setdefault(id(arc.start), set()).add(("slur", "start", slur_no))
                marks.setdefault(id(arc.stop), set()).add(("slur", "stop", slur_no))
    hp_start, hp_stop = {}, set()
    _text_hp = []
    for hp in getattr(score, "hairpins", []):
        if getattr(hp, "text", None):
            _text_hp.append(hp)                         # [v2.90] 글자 셈여림은 쐐기 대신 words
            continue
        if id(hp.start) in live and id(hp.stop) in live:
            hp_start[id(hp.start)] = hp.kind
            hp_stop.add(id(hp.stop))
    # [v2.80] 옥타브(8va·8vb·15ma)·페달 — 시작/끝 음 앞에 direction
    global _EXTRA_DIRS
    _EXTRA_DIRS = {}
    for ot in (getattr(score, "ottavas", None) or []):
        size = "15" if ot["kind"] == "15ma" else "8"
        typ = "up" if ot["kind"] == "8vb" else "down"      # MusicXML: 8va(높게 연주) = 기보를 내려 적음 → down
        if ot.get("start") is not None:
            _EXTRA_DIRS.setdefault(id(ot["start"]), []).append(("octave-shift", {"type": typ, "size": size}))
        if ot.get("stop") is not None:
            _EXTRA_DIRS.setdefault(id(ot["stop"]), []).append(("octave-shift", {"type": "stop", "size": size}))
    for pd in (getattr(score, "pedals", None) or []):
        line_ = "yes" if pd.get("style") == "line" else "no"
        if pd.get("start") is not None:
            _EXTRA_DIRS.setdefault(id(pd["start"]), []).append(("pedal", {"type": "start", "line": line_}))
        for ch_ in (pd.get("changes") or []):          # [v2.90] 다시 밟기
            _EXTRA_DIRS.setdefault(id(ch_), []).append(("pedal", {"type": "change", "line": line_}))
        if pd.get("stop") is not None:
            _EXTRA_DIRS.setdefault(id(pd["stop"]), []).append(("pedal", {"type": "stop", "line": line_}))
    for hp in _text_hp:
        _EXTRA_DIRS.setdefault(id(hp.start), []).append(("words", {"_text": hp.text}))
    global _GLISS
    _GLISS = {}
    for gl in (getattr(score, "glissandos", None) or []):
        _GLISS.setdefault(id(gl["start"]), []).append("start")
        _GLISS.setdefault(id(gl["stop"]), []).append("stop")

    measures = sorted(score.measures, key=lambda m: m.number)
    ending_start, ending_stop = _endings(measures)

    for p in parts:
        part = _sub(root, "part", id=f"P{p + 1}")
        cur_clef = {}                          # {파트안보표: 음자리표}
        cur_key = None
        cur_time = None
        for mi, measure in enumerate(measures):
            m = _sub(part, "measure", number=measure.number)
            if mi != 0 and getattr(measure, "page", 1) != getattr(measures[mi - 1], "page", 1):
                _sub(m, "print", new_page="yes")      # [v2.00] 원본 페이지 나눔 그대로
            elif measure.first_in_system and mi != 0:
                _sub(m, "print", new_system="yes")
            _left_barline(m, measure, ending_start.get(mi))

            # ---- 이 마디에서 이 파트의 보표별 요소
            per_staff = {k: [] for k in range(1, n_staves[p] + 1)}
            line_staff_clef = {}
            for e in measure.elements:
                pk = mapping.get((measure.system, getattr(e, "staff_no", 1)))
                if pk is None or pk[0] != p:
                    continue
                per_staff[pk[1]].append(e)
            for (ln, sn), (pp, k) in mapping.items():
                if ln == measure.system and pp == p:
                    cl = (getattr(score, "staff_clefs", None) or {}).get(ln) or []
                    line_staff_clef[k] = cl[sn - 1] if sn - 1 < len(cl) else ("F" if k >= 2 else "G")

            # ---- 마디 머리 속성(조표·박자표·음자리표)
            key_now = score.fifths if mi == 0 else measure.key_change
            time_now = _time_at(score, measure)
            free_ = not getattr(score, "time_sig_known", True) and not any(
                getattr(mm, "time_sig_change", None) for mm in score.measures if mm.number <= measure.number)
            if free_:
                # [v3.63] 자유 리듬: 마디마다 실제 길이에 맞춘 '보이지 않는 박자표'(MuseScore 등이 마디 길이를
                # 그대로 읽게). 8분음표 단위로, 안 되면 16분음표 단위로.
                from .editing import free_capacity
                q_ = free_capacity(measure)
                time_now = (int(round(q_ * 2)), 8) if abs(q_ * 2 - round(q_ * 2)) < 1e-6 \
                    else (max(1, int(round(q_ * 4))), 16)
            attr = None

            def _attr():
                nonlocal attr
                if attr is None:
                    attr = _sub(m, "attributes")
                return attr
            if mi == 0:
                _sub(_attr(), "divisions", DIVISIONS)
            if key_now is not None and key_now != cur_key:
                _sub(_sub(_attr(), "key"), "fifths", key_now)
                cur_key = key_now
            if time_now != cur_time:
                t = _sub(_attr(), "time")
                sym_ = (getattr(measure, "ts_printed", None) or (0, 0, None))[2]      # [v3.40] 마디마다 C·₵
                if mi == 0 and getattr(score, "ts_symbol", None) in ("common", "cut"):
                    sym_ = score.ts_symbol
                if sym_ in ("common", "cut") and tuple(time_now) == ((4, 4) if sym_ == "common" else (2, 2)):
                    t.set("symbol", sym_)
                if free_:
                    t.set("print-object", "no")
                _sub(t, "beats", time_now[0])
                _sub(t, "beat-type", time_now[1])
                cur_time = time_now
            if mi == 0 and n_staves[p] > 1:
                _sub(_attr(), "staves", n_staves[p])
            for k in range(1, n_staves[p] + 1):
                # 줄 처음은 그 줄 머리 음자리표, 아니면 직전 값 유지. 그 마디 첫 요소에 붙은
                # 변경(clef_change)은 마디 처음 위치이므로 여기서 함께 처리한다.
                want = cur_clef.get(k)
                if measure.first_in_system or k not in cur_clef:
                    want = line_staff_clef.get(k, want or ("F" if k >= 2 else "G"))
                first = min(per_staff[k], key=_x_of) if per_staff[k] else None
                if first is not None and getattr(first, "clef_change", None) \
                        and _is_first_slot(first, per_staff[k]):
                    want = first.clef_change
                if want != cur_clef.get(k):
                    c = _sub(_attr(), "clef", number=k) if n_staves[p] > 1 else _sub(_attr(), "clef")
                    sign, line = _CLEF_XML.get(want, ("G", 2))
                    _sub(c, "sign", sign)
                    _sub(c, "line", line)
                    cur_clef[k] = want

            if mi == 0 and p == parts[0] and meta.get("tempo"):
                dr = _sub(m, "direction", placement="above")
                _sub(_sub(dr, "direction-type"), "words", meta["tempo"])
            if p == parts[0]:
                _nav_directions(m, measure, at_start=True)     # [v2.80] 세뇨·코다(마디 처음)

            # ---- 보표·성부별 음표
            cap = _capacity_div(score, measure)
            beam_pos = _beam_positions(measure.elements)
            tup_mark = {}
            for g_els, _c in tuplet_groups(measure.elements):
                tup_mark.setdefault(id(g_els[0]), []).append("start")
                tup_mark.setdefault(id(g_els[-1]), []).append("stop")
            pos = 0
            vnum = 0
            ext_anchors = measure_anchors(measure.elements, cap)
            for k in range(1, n_staves[p] + 1):
                els = per_staff[k]
                if not els:
                    vnum += 1
                    if pos:
                        _sub(_sub(m, "backup"), "duration", pos)
                        pos = 0
                    _measure_rest(m, cap, vnum, k, n_staves[p])
                    pos = cap
                    continue
                voices = {}
                for e in els:
                    voices.setdefault(getattr(e, "voice_no", 1), []).append(e)
                voices = {v: _slots(es) for v, es in voices.items()}
                order = _place_voices(voices, cap, ext_anchors)
                lanes = _lanes(voices, order)
                for li, lane in enumerate(lanes):
                    vnum += 1
                    stem = None if len(lanes) < 2 else ("up" if li == 0 else "down")
                    if pos:
                        _sub(_sub(m, "backup"), "duration", pos)
                        pos = 0
                    for s in lane:
                        if not s["grace"] and s["on"] > pos:
                            fw = _sub(m, "forward")
                            _sub(fw, "duration", s["on"] - pos)
                            _sub(fw, "voice", vnum)
                            if n_staves[p] > 1:
                                _sub(fw, "staff", k)
                            pos = s["on"]
                        _emit_slot(m, s, vnum, k, n_staves[p], cap, marks, hp_start, hp_stop,
                                   beam_pos, tup_mark, cur_clef, s is lane[0] and
                                   _is_first_slot(s["els"][0], els), stem)
                        pos += s["dur"]
            if p == parts[0]:
                _nav_directions(m, measure, at_start=False)    # [v2.80] D.C.·D.S.·Fine·To Coda(마디 끝)
            _right_barline(m, measure, mi == len(measures) - 1, ending_stop.get(mi))
    return root


_EXTRA_DIRS = {}
_GLISS = {}
_NAV_WORDS = {"dc": "D.C.", "ds": "D.S.", "fine": "Fine", "dc_al_fine": "D.C. al Fine", "dc_al_coda": "D.C. al Coda",
              "ds_al_fine": "D.S. al Fine", "ds_al_coda": "D.S. al Coda", "to_coda": "To Coda"}


def _nav_directions(m, measure, at_start):
    """[v2.80] 이동 기호를 direction + sound(재생 프로그램이 실제로 뛰도록)로."""
    for nv in (getattr(measure, "nav", None) or []):
        k = nv["kind"]
        if at_start and k in ("segno", "coda"):
            d = _sub(m, "direction", placement="above")
            _sub(_sub(d, "direction-type"), k)
            _sub(d, "sound", **({"segno": "segno1"} if k == "segno" else {"coda": "coda1"}))
        elif not at_start and k in _NAV_WORDS:
            d = _sub(m, "direction", placement="above")
            _sub(_sub(d, "direction-type"), "words", _NAV_WORDS[k])
            snd = {}
            if k.startswith("dc"):
                snd["dacapo"] = "yes"
            if k.startswith("ds"):
                snd["dalsegno"] = "segno1"
            if k == "fine":
                snd["fine"] = "yes"
            if k == "to_coda":
                snd["tocoda"] = "coda1"
            if snd:
                _sub(d, "sound", **snd)


def _is_first_slot(e, els):
    x0 = min(_x_of(x) for x in els)
    return abs(_x_of(e) - x0) < 1e-6


def _emit_slot(m, s, vnum, staff, n_staves, cap, marks, hp_start, hp_stop, beam_pos, tup_mark,
               cur_clef, is_measure_first, stem=None):
    els = s["els"]
    rep = els[0]
    # 마디 중간의 음자리표 변경 — 이 음 '앞'에(마디 처음 변경은 머리 속성에서 이미 처리)
    ch = next((getattr(e, "clef_change", None) for e in els if getattr(e, "clef_change", None)), None)
    if ch and not is_measure_first and ch != cur_clef.get(staff):
        a = _sub(m, "attributes")
        c = _sub(a, "clef", number=staff) if n_staves > 1 else _sub(a, "clef")
        sign, line = _CLEF_XML.get(ch, ("G", 2))
        _sub(c, "sign", sign)
        _sub(c, "line", line)
        cur_clef[staff] = ch
    for e in els:
        h = getattr(e, "harmony", None)
        if h and h.get("root"):                                 # [v3.00] 기타 코드 — 그 음 앞에
            he = _sub(m, "harmony", **{"print-frame": "no"})
            r = _sub(he, "root")
            _sub(r, "root-step", h["root"])
            if h.get("alter"):
                _sub(r, "root-alter", h["alter"])
            kd = _sub(he, "kind", h.get("kind") or "major")
            kd.set("text", h.get("kind_text") or "")          # (악보에 적힌 꼬리 — _sub의 text는 내용이라 따로)
            if h.get("bass"):
                b_ = _sub(he, "bass")
                _sub(b_, "bass-step", h["bass"])
                if h.get("bass_alter"):
                    _sub(b_, "bass-alter", h["bass_alter"])
            # 기본 종류에 음을 더한 코드(7sus4·add9·7♭9·7♯9) — degree로 적어야 다른 프로그램이 같은 코드로 읽는다
            tail = (h.get("kind_text") or "").replace("♭", "b").replace("♯", "#")
            adds = []
            if h.get("kind") == "suspended-fourth" and "7" in tail:
                adds.append((7, 0))
            if "add9" in tail or "add2" in tail:
                adds.append((9, 0))
            if "b9" in tail:
                adds.append((9, -1))
            if "#9" in tail:
                adds.append((9, 1))
            for val, alt in adds:
                dg = _sub(he, "degree")
                _sub(dg, "degree-value", val)
                _sub(dg, "degree-alter", alt)
                _sub(dg, "degree-type", "add")
            if n_staves > 1:
                _sub(he, "staff", staff)
        for tag, attrs in _EXTRA_DIRS.get(id(e), []):          # [v2.80] 옥타브·페달
            if tag == "words":
                _direction(m, staff, n_staves, vnum, lambda dt, a=attrs: _sub(dt, "words", a["_text"]))
                continue
            _direction(m, staff, n_staves, vnum, lambda dt, t=tag, a=attrs: _sub(dt, t, **a))
        if id(e) in hp_stop:
            _direction(m, staff, n_staves, vnum, lambda dt: _sub(dt, "wedge", type="stop"))
        kind = hp_start.get(id(e))
        if kind:
            _direction(m, staff, n_staves, vnum, lambda dt, kd=kind: _sub(
                dt, "wedge", type="crescendo" if kd == "cresc" else "diminuendo"))
        dyn = getattr(e, "dynamic", None)
        if dyn and dyn.get("key"):
            from .midi import DYN_VELOCITY
            vel = DYN_VELOCITY.get(dyn["key"])
            d = _direction(m, staff, n_staves, vnum,
                           lambda dt, kk=dyn["key"]: _sub(_sub(dt, "dynamics"), kk))
            if vel:
                _sub(d, "sound", dynamics=str(round(vel * 100 / 90)))
    # 화음은 낮은 음부터(대표 음이 맨 앞), 쉼표는 하나만
    notes = sorted(els, key=lambda e: (_dia(e) if _dia(e) is not None else -1, e.alter or 0))
    if rep.kind == "rest":
        notes = [rep]
    for i, e in enumerate(notes):
        note = _sub(m, "note")
        if getattr(e, "is_grace", False):
            _sub(note, "grace")
        if i > 0:
            _sub(note, "chord")
        whole_rest = (e.kind == "rest" and e.duration_base == 1 and not e.dotted
                      and not getattr(e, "in_triplet", False))
        if e.kind == "rest":
            if whole_rest:
                _sub(note, "rest", measure="yes")
            else:
                _sub(note, "rest")
        else:
            pt = _sub(note, "pitch")
            _sub(pt, "step", e.step)
            if e.alter:
                _sub(pt, "alter", e.alter)
            _sub(pt, "octave", e.octave)
        mk = marks.get(id(e), set())
        tie_types = {t[1] for t in mk if t[0] == "tied"}
        if not getattr(e, "is_grace", False):
            _sub(note, "duration", cap if whole_rest else s["dur"])
        # <tie>는 <duration> 뒤(MusicXML 규격 순서 — v1.00은 앞에 써서 엄격한 프로그램이 거부할 수 있었다)
        if "stop" in tie_types:
            _sub(note, "tie", type="stop")
        if "start" in tie_types:
            _sub(note, "tie", type="start")
        _sub(note, "voice", vnum)
        if not whole_rest:
            _sub(note, "type", e.note_type)
            if e.dotted:
                _sub(note, "dot")
                if getattr(e, "double_dot", False):
                    _sub(note, "dot")
        acc = getattr(e, "accidental", None)
        if acc and e.kind == "note":
            _sub(note, "accidental", acc)
        if getattr(e, "in_triplet", False) and not getattr(e, "is_grace", False):
            tm = _sub(note, "time-modification")
            _sub(tm, "actual-notes", e.tup_actual)
            _sub(tm, "normal-notes", e.tup_normal)
        ov_ = getattr(e, "stem_override", None) or getattr(rep, "stem_override", None)
        if ov_ in ("up", "down") and e.kind == "note" and e.duration_base != 1:
            _sub(note, "stem", ov_)                 # [v2.25] 사용자가 정한 기둥 방향
        elif stem and e.kind == "note" and e.duration_base != 1:
            _sub(note, "stem", stem)                # 두 성부 이상: 위 성부 위로, 아래 성부 아래로
        if n_staves > 1:
            _sub(note, "staff", staff)
        bp = beam_pos.get(id(rep))
        if bp and e.kind == "note" and i == 0:          # 화음은 첫 음에만(Verovio 등이 구성음 빔을 거부)
            _sub(note, "beam", bp, number=1)
        tmark = next((tup_mark[id(x)] for x in els if id(x) in tup_mark), None) if i == 0 else None
        ferm = i == 0 and any(getattr(x, "fermata", False) for x in els)
        slur_mk = sorted(((t[1], t[2]) for t in mk if t[0] == "slur"), key=lambda q: (q[0] != "stop", q[1]))
        tie_mk = [("tied", t) for t in ("stop", "start") if t in tie_types]
        arts = [a for x in els for a in (getattr(x, "articulations", None) or [])] if i == 0 else []
        arp = any(a["kind"] == "arpeggio" for x in els for a in (getattr(x, "articulations", None) or []))
        gls = _GLISS.get(id(e), [])
        if tie_mk or slur_mk or tmark or ferm or arts or arp or gls:
            nots = _sub(note, "notations")
            if ferm:
                _sub(nots, "fermata", type="upright")
            if arp:
                _sub(nots, "arpeggiate")             # [v2.80] 화음 구성음마다
            for gt in gls:                           # [v2.90] 글리산도
                _sub(nots, "glissando", type=gt, **{"line-type": "solid"})
            _tag = {"accent": "accent", "staccato": "staccato", "tenuto": "tenuto",
                    "staccatissimo": "staccatissimo", "marcato": "strong-accent", "breath": "breath-mark",
                    "caesura": "caesura"}
            if any(a["kind"] in _tag for a in arts):  # [v2.03] 악센트·스타카토·테누토 등
                ae = _sub(nots, "articulations")
                for a in arts:
                    if a["kind"] in _tag:
                        _sub(ae, _tag[a["kind"]], placement="above" if a["above"] else "below")
            _orn = {"trill": "trill-mark", "mordent": "mordent", "turn": "turn", "tremolo": "tremolo"}
            if any(a["kind"] in _orn for a in arts):  # [v2.80] 꾸밈
                oe = _sub(nots, "ornaments")
                for a in arts:
                    if a["kind"] == "tremolo":           # [v2.90] 빗금 수
                        _sub(oe, "tremolo", str(int(a.get("n", 1))), type="single")
                    elif a["kind"] in _orn:
                        _sub(oe, _orn[a["kind"]], placement="above" if a["above"] else "below")
            for tt in (tmark or ()):
                _sub(nots, "tuplet", type=tt, bracket="yes")
            for _t, typ in tie_mk:
                _sub(nots, "tied", type=typ)
            for typ, no in slur_mk:
                _sub(nots, "slur", type=typ, number=no)
        if i == 0:
            lyr = {}
            for x in els:                           # 화음이면 가사가 붙은 구성음에서 모은다
                for row, ly_ in (getattr(x, "lyrics", {}) or {}).items():
                    lyr.setdefault(row, ly_)
            for row in sorted(lyr):
                txt = lyr[row].get("text", "")
                if not txt or txt.strip() in _LYRIC_EXTEND:   # 늘임 표시(–)는 음절이 아니다
                    continue
                ly = _sub(note, "lyric", number=str(row + 1))
                _sub(ly, "syllabic", "single")
                _sub(ly, "text", txt)


def _direction(m, staff, n_staves, vnum, fill):
    d = _sub(m, "direction", placement="below")
    fill(_sub(d, "direction-type"))
    _sub(d, "voice", vnum)
    if n_staves > 1:
        _sub(d, "staff", staff)
    return d


def _measure_rest(m, cap, vnum, staff, n_staves):
    note = _sub(m, "note")
    _sub(note, "rest", measure="yes")
    _sub(note, "duration", cap)
    _sub(note, "voice", vnum)
    if n_staves > 1:
        _sub(note, "staff", staff)


def _beam_positions(elements):
    out, groups = {}, {}
    for e in elements:
        bg = getattr(e, "beam_group", -1)
        if bg is None or bg == -1 or e.kind != "note":
            continue
        cg = getattr(e, "chord_group", -1)
        groups.setdefault(bg, {})
        key = ("c", cg) if cg != -1 else ("e", id(e))
        groups[bg].setdefault(key, []).append(e)
    for bg, slots in groups.items():
        ordered = sorted(slots.values(), key=lambda mem: min(_x_of(x) for x in mem))
        if len(ordered) < 2:
            continue
        for i, mem in enumerate(ordered):
            st = "begin" if i == 0 else "end" if i == len(ordered) - 1 else "continue"
            for e in mem:
                out[id(e)] = st
    return out


def _endings(measures):
    start, stop = {}, {}
    i = 0
    while i < len(measures):
        num = getattr(measures[i], "ending", None)
        if num is None:
            i += 1
            continue
        j = i
        while j + 1 < len(measures) and getattr(measures[j + 1], "ending", None) == num:
            j += 1
        start[i] = num
        nxt = j + 1 < len(measures) and getattr(measures[j + 1], "ending", None) is not None
        stop[j] = (num, "stop" if nxt else "discontinue")
        i = j + 1
    return start, stop


def _left_barline(m, measure, ending_num):
    rep = getattr(measure, "start_repeat", False)
    if rep or ending_num is not None:
        bl = _sub(m, "barline", location="left")
        if rep:
            _sub(bl, "bar-style", "heavy-light")
        if ending_num is not None:
            _sub(bl, "ending", type="start", number=str(ending_num))
        if rep:
            _sub(bl, "repeat", direction="forward")


def _right_barline(m, measure, is_last, ending_stop):
    kind = getattr(measure, "end_barline", "normal")
    style = {"repeat_end": "light-heavy", "final": "light-heavy",
             "double": "light-light"}.get(kind)
    if is_last and style is None:
        style = "light-heavy"
    if style or ending_stop:
        bl = _sub(m, "barline", location="right")
        if style:
            _sub(bl, "bar-style", style)
        if ending_stop:
            _sub(bl, "ending", type=ending_stop[1], number=str(ending_stop[0]))
        if kind == "repeat_end":
            _sub(bl, "repeat", direction="backward")


def score_to_string(score, title="악보 인식 결과") -> str:
    root = score_to_element(score, title)
    rough = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    lines = pretty.split("\n", 1)
    doctype = ('<!DOCTYPE score-partwise PUBLIC '
               '"-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
               '"http://www.musicxml.org/dtds/partwise.dtd">')
    return lines[0] + "\n" + doctype + "\n" + lines[1]


def save_musicxml(score, path, title="악보 인식 결과"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(score_to_string(score, title))
    return path
