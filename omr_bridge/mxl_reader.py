# omr_bridge/mxl_reader.py
# ---------------------------------------------------------------------------
# [1단계] Audiveris가 만든 .mxl(MusicXML)에서 "음악적 의미"(음정·박자·
# 점·임시표·꾸밈음·잇단음표 등)를 추출한다. 위치(좌표)는 이 모듈이 아니라
# geometry.py(.omr 기반)가 담당한다 — 역할 분리는 계획서 3.3절 아키텍처.
# ---------------------------------------------------------------------------
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from fractions import Fraction


# omr_34 score/model.py의 duration_base 코드값(1=온음표...32=32분음표)과
# MusicXML <type> 텍스트의 매핑.
_TYPE_TO_BASE = {
    "whole": 1, "half": 2, "quarter": 4, "eighth": 8,
    "16th": 16, "32nd": 32, "64th": 64,
}


@dataclass
class NoteEvent:
    kind: str                  # "note" | "rest"
    duration_base: int         # 1/2/4/8/16/32 (0이면 알 수 없음 — 편집기에서 보정 필요)
    dotted: bool
    step: str = None
    octave: int = None
    accidental: str = None     # "sharp"|"flat"|"natural"|None
    is_grace: bool = False
    in_triplet: bool = False   # 잇단음표 그룹의 일원인지(3잇단만이 아니라 모든 비율)
    tup_actual: int = 3        # [잇단음표 일반화] 비율 actual:normal (3:2, 5:4, 6:4, 7:4, 2:3, 4:3 …)
    tup_normal: int = 2
    fermata: bool = False      # Tier2 신규 필드(계획서 5.3절)
    voice: int = 1
    staff: int = 1
    is_chord_member: bool = False  # [Tier2] True면 바로 앞 이벤트와 동시발음(화음)
    tie_start: bool = False    # [이음줄/붙임줄 지원] 이 음에서 붙임줄(같은 음정) 시작
    tie_stop: bool = False     # 이 음에서 붙임줄 종료(없으면 호를 그릴 짝을 못 찾는다)
    slur_start: bool = False   # 이 음에서 이음줄(다른 음정 연결) 시작
    slur_stop: bool = False    # 이 음에서 이음줄 종료
    beam_state: str = None     # [빔 그룹 지원] "begin"|"continue"|"end"|None
    dynamic: str = None        # [셈여림] 내부 키("p","mf","sfz"…) — 이 음 바로 앞의 셈여림 기호
    wedge_start: str = None    # [쐐기] "cresc" | "decresc" — 이 음에서 시작
    wedge_stop: bool = False   # [쐐기] 이 음에서 끝
    dots: int = 0              # [v1.10 겹점] 점 개수(0·1·2). dotted는 dots>=1과 같다
    harmony: str = None        # [v3.00] 이 음 앞의 코드 이름(Audiveris <harmony>를 글자로: "C#m7/G#")
    clef_change: str = None    # [v1.10 음자리표 변경] 이 음(쉼표 포함)부터 그 보표에 적용되는 새 음자리표
                               #   내부 키 "G"/"F"/"C" — 직전과 실제로 달라진 경우만 채운다


@dataclass
class MeasureMusic:
    number: int
    voices: dict = field(default_factory=dict)  # {voice(int): [NoteEvent, ...]}
    bar_style: str = None  # [종지선 지원] 오른쪽 마디선 스타일(light-heavy 등)


@dataclass
class PartMusic:
    part_id: str
    measures: list = field(default_factory=list)  # [MeasureMusic, ...] (마디번호 순)


@dataclass
class ScoreMusic:
    fifths: int = 0
    beats: int = 4
    beat_type: int = 4
    parts: list = field(default_factory=list)
    # [Tier2 보강] {마디번호: (beats, beat_type)} — 박자표가 실제로 몇 번째
    # 마디부터 바뀌는지 기록. merge.py가 "줄 전체 최빈값" 대신 이 구간
    # 정보로 마디별 target_capacity를 더 정확히 계산하는 데 쓴다.
    time_changes: dict = field(default_factory=dict)
    time_symbols: dict = field(default_factory=dict)   # [v3.40] {마디: "common"|"cut"} — C·₵ 기호로 찍힌 박자표
    # [v1.10 조표 바뀜] {마디번호: fifths} — 직전과 실제로 다른 조표가 시작되는 마디만.
    # fifths 는 곡 '처음' 조표다(예전엔 마지막 <key>로 덮어써 곡 전체가 끝 조표로 보였다).
    key_changes: dict = field(default_factory=dict)
    # [v2.00] 첫 파트의 볼타 시작 번호(나오는 차례대로) — 꺾쇠 위치(.omr)와 차례로 짝짓는다
    ending_numbers: list = field(default_factory=list)
    part_names: list = field(default_factory=list)   # [v1.20] <score-part><part-name> (파트 순서)


def _load_root(mxl_path: str):
    if mxl_path.endswith(".mxl"):
        with zipfile.ZipFile(mxl_path) as z:
            names = [n for n in z.namelist() if n.endswith(".xml") and "META-INF" not in n]
            with z.open(names[0]) as f:
                return ET.parse(f).getroot()
    else:
        return ET.parse(mxl_path).getroot()


def _majority_keys(root):
    """파트마다 각 마디 시작의 조표를 모아 마디별 다수결 → (처음 조표, {바뀌는 마디: 조표}).
    파트가 하나뿐이거나 조표가 없으면 None(예전 방식)."""
    import re as _re
    per_part = []
    for part_el in root.findall(".//part"):
        cur, seq = None, {}
        for meas_el in part_el.findall("measure"):
            digits = _re.sub(r"[^0-9]", "", meas_el.get("number", "") or "")
            if not digits:
                continue
            k = meas_el.find("attributes/key/fifths")
            if k is not None and k.text:
                cur = int(k.text)
            if cur is not None:
                seq[int(digits)] = cur
        if seq:
            per_part.append(seq)
    if len(per_part) < 2:
        return None
    nums = sorted(set().union(*per_part))
    first, prev, changes = None, None, {}
    for n in nums:
        votes = {}
        for seq in per_part:
            if n in seq:
                votes[seq[n]] = votes.get(seq[n], 0) + 1
        best = max(votes.values())
        cands = [k for k, v in votes.items() if v == best]
        kf = prev if prev in cands else cands[0]       # 동률이면 앞 조표 유지
        if first is None:
            first = kf
        elif kf != prev:
            changes[n] = kf
        prev = kf
    return first, changes


def load_score_music(mxl_path: str) -> ScoreMusic:
    root = _load_root(mxl_path)
    sm = ScoreMusic()

    divisions = 4
    for spart in root.findall("part-list/score-part"):
        sm.part_names.append((spart.findtext("part-name") or "").strip())
    first_key, cur_key = [None], [None]
    # [v2.00 조표 다수결 — 실사용] 조표는 예전엔 첫 파트에서만 읽었다. "교회" 2쪽 셋째 줄은 Audiveris가
    # 맨 위 합창 보표의 조표만 놓쳐(나머지 3보표는 ♯4개) 첫 파트가 다장조로 바뀐 것처럼 나왔고, 곡 전체가
    # 다장조로 바뀌어 ♯이 빠졌다. 이제 마디마다 파트들의 조표를 모아 '다수'를 따른다.
    _maj = _majority_keys(root)
    if _maj is not None:
        sm.fifths, sm.key_changes = _maj
        first_key[0] = "done"
    _CLEF_SIGN = {"G": "G", "F": "F", "C": "C"}
    _first_part = root.find(".//part")
    if _first_part is not None:
        for _bl in _first_part.iter("barline"):
            _en = _bl.find("ending")
            if _en is not None and _en.get("type") == "start":
                import re as _re2
                _d = _re2.sub(r"[^0-9]", "", (_en.get("number") or "1").split(",")[0])
                sm.ending_numbers.append(int(_d) if _d else 1)
    for part_el in root.findall(".//part"):
        part = PartMusic(part_id=part_el.get("id", "P1"))
        if first_key[0] is not None:       # 조표는 첫 파트 기준(파트마다 같은 값이 반복된다)
            first_key[0], cur_key[0] = "done", None
        cur_clef = {}              # [v1.10] {보표: 현재 음자리표}
        carry_clef = {}            # 마디 끝에 바뀐 음자리표 → 다음 마디 그 보표 첫 요소에
        carry_wedge_stop = set()   # [쐐기] 마디 끝에 남은 쐐기 끝 → 다음 마디로 넘김
        for meas_el in part_el.findall("measure"):
            raw_num = meas_el.get("number", "0") or "0"
            try:
                mnum = int(raw_num)
            except ValueError:
                # geometry.py와 동일한 이유(Audiveris가 "X18" 같은 접미사
                # 붙은 마디 번호를 쓰는 경우가 있음, 545개 배치 실측으로
                # 발견) — 숫자 부분만 추출해 geometry.py 쪽 처리와 번호를
                # 일치시킨다. 숫자가 전혀 없으면 이 마디는 건너뛴다.
                import re as _re
                digits = _re.sub(r"[^0-9]", "", raw_num)
                if not digits:
                    continue
                mnum = int(digits)
            mm = MeasureMusic(number=mnum)

            attrs = meas_el.find("attributes")
            if attrs is not None:
                d = attrs.find("divisions")
                if d is not None and d.text:
                    divisions = int(d.text)
                k = attrs.find("key")
                if k is not None and k.find("fifths") is not None and first_key[0] != "done":
                    kf = int(k.find("fifths").text)
                    if first_key[0] is None:
                        first_key[0] = kf
                        sm.fifths = kf
                    elif kf != cur_key[0]:
                        sm.key_changes[mnum] = kf
                    cur_key[0] = kf
                t = attrs.find("time")
                if t is not None and t.find("beats") is not None:
                    sm.beats = int(t.find("beats").text)
                    sm.beat_type = int(t.find("beat-type").text)
                    sm.time_changes[mnum] = (sm.beats, sm.beat_type)
                    if t.get("symbol") in ("common", "cut"):
                        sm.time_symbols[mnum] = t.get("symbol")

            # [셈여림·쐐기] <direction>은 음표 사이사이에 순서대로 끼어 있다. 마디의
            # 자식들을 순서대로 훑으며 "보표별 대기열"에 쌓았다가 그 보표의 다음
            # 음표(쉼표 포함, 화음 구성음 제외)에 붙인다. 마디 끝까지 남은 것은 그
            # 보표의 마지막 음표에 붙인다(곡선이 마지막 음 뒤에서 끝나는 경우).
            dir_info = {}            # {id(note_el): {"dynamic","wedge_start","wedge_stop"}}
            # 이전 마디 끝에 남았던 "쐐기 끝"은 이 마디 첫 음에 붙인다(마디선까지 이어지는 쐐기).
            pending = {st: {"wedge_stop": True} for st in carry_wedge_stop}
            carry_wedge_stop.clear()
            last_note_by_staff = {}
            clef_info = {}             # [v1.10] {id(note_el): 새 음자리표}
            pend_clef = dict(carry_clef)
            carry_clef.clear()
            for ch_el in list(meas_el):
                st_el = ch_el.find("staff")
                st_no = int(st_el.text) if st_el is not None and st_el.text else 1
                if ch_el.tag == "attributes":
                    # [v1.10 음자리표 변경] <clef number="n">의 위치(마디 처음·중간)를 그대로 따른다.
                    # 첫 선언·같은 값의 재선언(새 줄 머리)은 변경이 아니다.
                    for c_el in ch_el.findall("clef"):
                        sign = _CLEF_SIGN.get((c_el.findtext("sign") or "").strip())
                        if sign is None:
                            continue
                        cn = int(c_el.get("number", "1") or 1)
                        if cn in cur_clef and cur_clef[cn] != sign:
                            pend_clef[cn] = sign
                        elif cn in cur_clef:
                            pend_clef.pop(cn, None)
                        cur_clef[cn] = sign
                        # [v4.00] 높은음자리표 8va(clef-octave-change +1)는 오독으로 본다 — 줄 첫머리로 넘어온
                        # 붙임줄 끝이 음자리표 위에 걸려 '8'로 읽힌 것(성가 445번 3번째 줄: 한 줄이 통째로 한 옥타브 위).
                        # 그 음자리표가 유효한 동안 음을 한 옥타브 내려 읽는다. (8vb(-1) 테너 음자리표는 실제로 쓰여 그대로.)
                        try:
                            oc = int((c_el.findtext("clef-octave-change") or "0").strip() or 0)
                        except ValueError:
                            oc = 0
                        cur_clef[("oct", cn)] = 1 if (sign == "G" and oc == 1) else 0
                    continue
                if ch_el.tag == "note" and ch_el.find("chord") is None and pend_clef.get(st_no):
                    clef_info[id(ch_el)] = pend_clef.pop(st_no)
                if ch_el.tag == "harmony":         # [v3.00] 기타 코드 → 다음 음에
                    txt = _harmony_text(ch_el)
                    if txt:
                        pending.setdefault(st_no, {})["harmony"] = txt
                    continue
                if ch_el.tag == "direction":
                    pd = pending.setdefault(st_no, {})
                    for dt in ch_el.findall("direction-type"):
                        dyn_el = dt.find("dynamics")
                        if dyn_el is not None and len(dyn_el):
                            pd["dynamic"] = dyn_el[0].tag
                        wg = dt.find("wedge")
                        if wg is not None:
                            wt = wg.get("type")
                            if wt == "crescendo":
                                pd["wedge_start"] = "cresc"
                            elif wt == "diminuendo":
                                pd["wedge_start"] = "decresc"
                            elif wt == "stop":
                                pd["wedge_stop"] = True
                elif ch_el.tag == "note" and ch_el.find("chord") is None \
                        and ch_el.find("grace") is None:
                    last_note_by_staff[st_no] = ch_el
                    if pending.get(st_no):
                        dir_info[id(ch_el)] = pending.pop(st_no)
            carry_clef.update(pend_clef)     # 마디 끝(뒤에 음이 없는) 변경은 다음 마디로
            for st_no, pd in pending.items():
                if pd.pop("wedge_stop", False):
                    carry_wedge_stop.add(st_no)      # 다음 마디 첫 음에서 끝내기
                if pd and st_no in last_note_by_staff:
                    dir_info.setdefault(id(last_note_by_staff[st_no]), {}).update(pd)

            for note_el in meas_el.findall("note"):
                is_chord_member = note_el.find("chord") is not None
                # [Tier2] 화음 구성음도 스킵하지 않고 그대로 읽는다. merge.py가
                # geometry.py의 chord_to_members()와 순서를 맞춰 결합할 때,
                # "화음의 몇 번째 멤버인가"를 is_chord_member로 구분해 처리한다.

                is_grace = note_el.find("grace") is not None
                is_rest = note_el.find("rest") is not None
                voice_el = note_el.find("voice")
                voice = int(voice_el.text) if voice_el is not None and voice_el.text else 1
                staff_el = note_el.find("staff")
                staff = int(staff_el.text) if staff_el is not None and staff_el.text else 1

                type_el = note_el.find("type")
                dur_base = _TYPE_TO_BASE.get(type_el.text, 0) if type_el is not None else 0
                n_dots = len(note_el.findall("dot"))
                dotted = n_dots > 0

                # [잇단음표 일반화 — 실사용 버그 수정] 이전에는 <notations><tuplet>이 붙은
                # 음만 잇단음표로 봤는데, 이 표시는 그룹의 첫 음(start)과 끝 음(stop)에만
                # 붙는다 — 가운데 음은 늘 일반 음으로 처리돼 "0.667, 1, 0.667"처럼 박수가
                # 섞이는 문제가 있었다. 그룹의 모든 음에 붙는 <time-modification>(실제
                # 비율)을 기준으로 삼는다.
                tup = False
                tup_actual, tup_normal = 3, 2
                tm_el = note_el.find("time-modification")
                if tm_el is not None:
                    try:
                        tup_actual = int(tm_el.findtext("actual-notes"))
                        tup_normal = int(tm_el.findtext("normal-notes"))
                        tup = tup_actual != tup_normal
                    except (TypeError, ValueError):
                        tup_actual, tup_normal = 3, 2
                notations = note_el.find("notations")
                fermata = False
                if notations is not None:
                    for t_el in notations.findall("tuplet"):
                        if t_el.get("type") == "start":
                            tup = True
                    if notations.find("tuplet") is not None:
                        # start/stop 상관없이 잇단음표 구간 내부면 True로 취급
                        # (구간 판정은 추후 병합 단계에서 정교화 가능 — MVP는
                        #  각 note가 직접 갖는 tuplet 태그 유무만 반영)
                        tup = True
                    if notations.find("fermata") is not None:
                        fermata = True

                # [이음줄/붙임줄 지원] Audiveris 실측(545개 배치 중 83%가
                # slur, 55%가 tie를 담고 있었음)으로 확인된 실제 정보를
                # 활용한다. <tie type="start">는 note 최상위에(음향적
                # 연결), <notations><slur type="start"/></notations>는
                # 시각 정보로 따로 있다.
                tie_start = note_el.find("tie[@type='start']") is not None
                tie_stop = note_el.find("tie[@type='stop']") is not None
                slur_start = False
                slur_stop = False
                if notations is not None:
                    for sl_el in notations.findall("slur"):
                        t = sl_el.get("type")
                        if t == "start":
                            slur_start = True
                        elif t == "stop":
                            slur_stop = True

                # [빔 그룹 지원] number="1"(1차 빔, 8분음표 기준)만 본다 —
                # 16분음표 이상의 2차/3차 빔은 렌더러가 duration_base로
                # 자동으로 겹줄 개수를 정하므로(beam_count()), 어느 음표들이
                # "하나의 빔 덩어리"로 묶이는지만 알면 충분하다.
                beam_state = None
                beam_el = note_el.find("beam[@number='1']")
                if beam_el is not None:
                    beam_state = beam_el.text

                ev = NoteEvent(
                    kind="rest" if is_rest else "note",
                    duration_base=dur_base, dotted=dotted,
                    is_grace=is_grace, in_triplet=tup, fermata=fermata,
                    tup_actual=tup_actual, tup_normal=tup_normal,
                    voice=voice, staff=staff, is_chord_member=is_chord_member,
                    tie_start=tie_start, tie_stop=tie_stop,
                    slur_start=slur_start, slur_stop=slur_stop,
                    beam_state=beam_state,
                    dynamic=dir_info.get(id(note_el), {}).get("dynamic"),
                    harmony=dir_info.get(id(note_el), {}).get("harmony"),
                    wedge_start=dir_info.get(id(note_el), {}).get("wedge_start"),
                    wedge_stop=bool(dir_info.get(id(note_el), {}).get("wedge_stop")),
                    dots=min(n_dots, 2), clef_change=clef_info.get(id(note_el)),
                )

                if not is_rest:
                    pitch_el = note_el.find("pitch")
                    if pitch_el is not None:
                        ev.step = pitch_el.findtext("step")
                        ev.octave = int(pitch_el.findtext("octave")) - cur_clef.get(("oct", st_no), 0)
                    acc_el = note_el.find("accidental")
                    if acc_el is not None and acc_el.text:
                        ev.accidental = acc_el.text  # "sharp"|"flat"|"natural"

                mm.voices.setdefault(voice, []).append(ev)

            # [종지선 지원] 이 마디의 오른쪽 마디선 스타일(light-heavy=
            # 종지선, light-light=더블바 등). Audiveris가 실제로 정확히
            # 인식하는 것을 확인했다(545개 배치 다수에서 light-heavy
            # 존재) — 그동안 렌더러(render.py)와 모델(model.py)에는 이미
            # "final" 타입 처리 로직이 있었는데, 이 정보 자체를 채워
            # 넘기는 코드가 없어 활용이 안 되고 있었다.
            for bl_el in meas_el.findall("barline"):
                if bl_el.get("location") == "right":
                    style_el = bl_el.find("bar-style")
                    if style_el is not None and style_el.text:
                        mm.bar_style = style_el.text

            part.measures.append(mm)
        sm.parts.append(part)

    return sm



_KIND_TAIL = {"major": "", "minor": "m", "dominant": "7", "major-seventh": "maj7", "minor-seventh": "m7",
              "diminished": "dim", "diminished-seventh": "dim7", "half-diminished": "m7b5", "augmented": "aug",
              "augmented-seventh": "aug7", "major-sixth": "6", "minor-sixth": "m6", "dominant-ninth": "9",
              "major-ninth": "maj9", "minor-ninth": "m9", "dominant-11th": "11", "dominant-13th": "13",
              "suspended-fourth": "sus4", "suspended-second": "sus2", "major-minor": "mM7", "power": "5"}


def _harmony_text(h_el):
    """[v3.00] <harmony> → 코드 글자. kind의 text 속성(악보에 적힌 꼬리)이 있으면 그것을 쓴다."""
    step = h_el.findtext("root/root-step")
    if not step:
        return None
    alt = h_el.findtext("root/root-alter")
    acc = {"1": "#", "-1": "b"}.get((alt or "").strip(), "")
    k = h_el.find("kind")
    tail = ""
    if k is not None:
        tail = k.get("text") if k.get("text") is not None else _KIND_TAIL.get((k.text or "").strip(), "")
    bass = h_el.findtext("bass/bass-step")
    s = step + acc + (tail or "")
    if bass:
        balt = h_el.findtext("bass/bass-alter")
        s += "/" + bass + {"1": "#", "-1": "b"}.get((balt or "").strip(), "")
    return s
