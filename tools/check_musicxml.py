# tools/check_musicxml.py
# ---------------------------------------------------------------------------
# [v1.20] MusicXML 내보내기 자동 검사 — 내보낸 파일을 다시 읽어 편집 모델과 맞춰 본다.
#   python tools/check_musicxml.py 악보.pdf|png [...]     (개발용, music21 필요: pip install music21)
#
#  ① 시간 배치: 파트·마디마다 <backup>/<forward>/<duration>을 따라가며 위치가 음수가 되거나
#     마디 길이를 넘는 성부가 없는지(모델에서 '정상'인 마디 기준).
#  ② 내용 일치: 마디별 (음높이·길이·점) 목록이 모델과 같은지(빈 보표용 온마디 쉼표는 제외).
#  ③ 붙임줄·이음줄·가사 개수가 모델과 같은지.
#  ④ music21(독립 구현)로 읽어 파트·마디 수, 정상 마디의 길이가 박자와 같은지.
# ---------------------------------------------------------------------------
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from score import musicxml as mx  # noqa: E402
from score import editing as ed  # noqa: E402
from score.model import dot_factor  # noqa: E402

GOOD = ("ok", "anacrusis", "time_change")


def model_items(score):
    out = {}
    for m in score.measures:
        c = out.setdefault(m.number, Counter())
        for e in m.elements:
            if e.kind == "rest":
                if e.duration_base == 1 and not e.dotted and not e.in_triplet:
                    c[("rest", "measure")] += 1
                else:
                    c[("rest", mx._dur_div(e), e.dotted + e.double_dot)] += 1
            else:
                c[(e.step, e.octave, e.alter or 0, mx._dur_div(e), e.dotted + e.double_dot)] += 1
    return out


def check(score, label=""):
    xml = mx.score_to_string(score)
    root = ET.fromstring(xml.split("\n", 2)[2] if xml.startswith("<?xml") else xml)
    problems = []
    status = {k: v[0] for k, v in ed.validate(score).items()}
    caps = {m.number: mx._capacity_div(score, m) for m in score.measures}
    got = {}
    n_tie = n_slur = n_lyr = 0
    parts = root.findall("part")
    for part in parts:
        for meas in part.findall("measure"):
            num = int(meas.get("number"))
            pos, mx_pos = 0, 0
            c = got.setdefault(num, Counter())
            for ch in meas:
                if ch.tag == "backup":
                    pos -= int(ch.findtext("duration"))
                    if pos < 0:
                        problems.append(f"{part.get('id')} 마디{num}: backup이 마디 앞을 넘음")
                elif ch.tag == "forward":
                    pos += int(ch.findtext("duration"))
                elif ch.tag == "note":
                    d = int(ch.findtext("duration") or 0)
                    if ch.find("chord") is None:
                        pos += d
                    mx_pos = max(mx_pos, pos)
                    dots = len(ch.findall("dot"))
                    r = ch.find("rest")
                    if r is not None:
                        if r.get("measure") == "yes":
                            # 빈 보표용으로 만든 쉼표인지는 아래에서 모델 개수와 맞춰 본다
                            c[("rest", "measure")] += 1
                        else:
                            c[("rest", d, dots)] += 1
                    else:
                        c[(ch.findtext("pitch/step"), int(ch.findtext("pitch/octave")),
                           int(ch.findtext("pitch/alter") or 0), d, dots)] += 1
                    n_tie += len(ch.findall("notations/tied[@type='start']"))
                    n_slur += len(ch.findall("notations/slur[@type='start']"))
                    n_lyr += len(ch.findall("lyric"))
            if status.get(num) in GOOD and mx_pos > caps[num]:
                problems.append(f"{part.get('id')} 마디{num}: 정상 마디인데 내보낸 길이 {mx_pos} > {caps[num]}")
    want = model_items(score)
    for num, wc in want.items():
        gc = Counter(got.get(num, Counter()))
        extra_rests = gc[("rest", "measure")] - wc[("rest", "measure")]
        if extra_rests > 0:
            gc[("rest", "measure")] -= extra_rests       # 빈 보표 채움
        if gc != wc:
            problems.append(f"마디{num}: 내용 불일치 빠짐={dict(wc - gc)} 더함={dict(gc - wc)}")
    live = {id(e) for m in score.measures for e in m.elements}
    w_tie = sum(1 for a in score.arcs if a.kind == "tie" and id(a.start) in live and id(a.stop) in live)
    w_slur = sum(1 for a in score.arcs if a.kind != "tie" and id(a.start) in live and id(a.stop) in live)
    w_lyr = 0
    for m in score.measures:
        seen = set()
        for e in m.elements:
            cg = e.chord_group
            key = ("c", cg) if cg != -1 else ("e", id(e))
            for row, ly in (e.lyrics or {}).items():
                if ly.get("text") and ly["text"].strip() not in mx._LYRIC_EXTEND and (key, row) not in seen:
                    seen.add((key, row))
                    w_lyr += 1
    if (n_tie, n_slur, n_lyr) != (w_tie, w_slur, w_lyr):
        problems.append(f"붙임줄/이음줄/가사 개수 {n_tie}/{n_slur}/{n_lyr} ≠ 모델 {w_tie}/{w_slur}/{w_lyr}")
    # ④ music21
    m21 = ""
    try:
        import music21
        sc = music21.converter.parse(xml, format="musicxml")
        nm = [len(p.getElementsByClass("Measure")) for p in sc.parts]
        bad = []
        for p in sc.parts:
            for mm in p.getElementsByClass("Measure"):
                if status.get(mm.number) in GOOD:
                    q = float(mm.highestTime)
                    if abs(q * mx.DIVISIONS - caps[mm.number]) > 1:
                        bad.append((p.partName, mm.number, q))
        m21 = f"music21: 파트 {len(sc.parts)}개 {[p.partName for p in sc.parts]}, 마디 {nm}"
        if bad:
            problems.append(f"music21 길이 불일치(정상 마디): {bad[:6]}")
    except Exception as ex:                       # noqa: BLE001
        problems.append(f"music21 읽기 실패: {ex}")
    n_ok = sum(1 for v in status.values() if v in GOOD)
    print(f"[{label}] 마디 {len(score.measures)}(정상 {n_ok}) · 파트 {len(parts)} · {m21}")
    for pmsg in problems:
        print("   ✗", pmsg)
    if not problems:
        print("   ✓ 통과")
    return problems, xml


if __name__ == "__main__":
    from compare_render import build
    for src in sys.argv[1:]:
        _doc, sc, _la, _hd = build(src)
        check(sc, os.path.basename(src))
