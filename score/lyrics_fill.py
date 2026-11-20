# score/lyrics_fill.py
# ---------------------------------------------------------------------------
# [v3.10] 가사 흘려 넣기 — 글자 인식 없이, 사용자가 붙여 넣은 가사를 음표에 차례로 배정한다.
# (PDF는 텍스트 층에서 가사를 읽지만, JPG 등 그림 악보에는 글자 정보가 없다.)
#   · 음절 나누기: 한글은 한 글자 = 한 음절, 라틴 문자는 띄어쓰기 단위 + 단어 안 '-'로 나눈 음절.
#     문장부호(, . ! ? ; : ' " ))는 앞 음절에 붙인다. '_' = 이 음은 건너뛴다. 혼자 쓴 '-' = 늘임 표시(–).
#   · 음 차례: 정한 보표(·성부)의 음을 박 순서대로, 화음은 한 번만(맨 위 음). 붙임줄로 이어진 뒷음,
#     이음줄 안의 뒷음(멜리스마 — 한 음절을 여러 음으로)은 건너뛸 수 있다.
# 가사 저장 형태는 PDF 가사와 같다: e.lyrics[절] = {"text", "x", "y", "h", "d"} (곡 좌표).
# ---------------------------------------------------------------------------
import re

_PUNCT = set(",.!?;:'\")]}…·~")


def _is_hangul(ch):
    return "\uac00" <= ch <= "\ud7a3" or "\u3131" <= ch <= "\u318e"


def tokenize(text):
    """가사 글 → 음절 목록. 각 항목은 글자(str) 또는 None(건너뛰기 '_')."""
    out = []
    for word in re.split(r"\s+", (text or "").strip()):
        if not word:
            continue
        if word == "_" or set(word) == {"_"}:
            out.extend([None] * len(word))
            continue
        if word in ("-", "–", "—", "―"):
            out.append("–")
            continue
        if any(_is_hangul(c) for c in word):
            cur = []
            for c in word:
                if c == "_":
                    cur.append(None)
                elif c in _PUNCT and cur and cur[-1] is not None:
                    cur[-1] += c
                elif c in "-–—―":
                    cur.append("–")
                else:
                    cur.append(c)
            out.extend(cur)
            continue
        parts = [p for p in re.split(r"(?<=\w)-(?=\w)", word) if p != ""]
        out.extend(parts)
    return out


def staff_key(staff_parts, line, staff_no):
    """보표 번호는 줄마다 새로 매겨진다(첫 줄 1번 = 피아노, 다음 줄 1번 = 노래일 수 있음) →
    (파트, 그 파트 안에서 몇 번째 보표)로 같은 줄기를 잇는다. 파트 정보가 없으면 보표 번호 그대로."""
    parts = (staff_parts or {}).get(line)
    if not parts or staff_no - 1 >= len(parts):
        return ("staff", staff_no)
    p = parts[staff_no - 1]
    j = sum(1 for k in range(staff_no - 1) if parts[k] == p)
    return (p, j)


def note_sequence(score, staff_no, start=None, voice=None, skip_tied=True, skip_slur=True, staff_parts=None,
                  key=None):
    """가사를 받을 음들의 차례. key=(파트, 파트 안 보표 순번)이면 그 줄기(줄마다 보표 번호가 달라도),
    없으면 staff_no. start=(마디, 요소)부터. voice가 주어지면 그 성부만. 반환: [(measure, element), ...]"""
    from .musicxml import element_onsets, DIVISIONS
    from .editing import _capacity_of
    ms = sorted(score.measures, key=lambda m: m.number)
    slur_inside = set()
    ordered = []
    for m in ms:
        on = element_onsets(m, int(round(_capacity_of(score, m) * DIVISIONS)))
        if key is not None:
            sn_here = [k + 1 for k in range(len((staff_parts or {}).get(m.system) or [])) if
                       staff_key(staff_parts, m.system, k + 1) == key] or ([key[1]] if key[0] == "staff" else [])
        else:
            sn_here = [staff_no]
        cands = [e for e in m.elements if e.kind == "note" and not e.is_grace and e.staff_no in sn_here
                 and (voice is None or e.voice_no == voice)]
        groups = {}
        for e in cands:
            k = (on.get(id(e), 0), e.voice_no if voice is not None else 0)
            groups.setdefault(k, []).append(e)
        for k in sorted(groups):
            g = groups[k]
            top = max(g, key=lambda e: (e.octave or 0) * 7 + "CDEFGAB".index(e.step or "C"))
            ordered.append((m, top, g))
    # 이음줄 안의 뒷음(멜리스마)
    if skip_slur:
        pos = {}
        for i, (m, e, g) in enumerate(ordered):
            for x in g:
                pos[id(x)] = i
        for arc in getattr(score, "arcs", []):
            if getattr(arc, "kind", "slur") != "slur":
                continue
            a, b = pos.get(id(arc.start)), pos.get(id(arc.stop))
            if a is not None and b is not None and b > a:
                for i in range(a + 1, b + 1):
                    slur_inside.add(i)
    # 이어진 뒷음: ①붙임 끝 표시 ②붙임줄 호의 끝 ③마디를 넘는 당김음(마디 끝 짧은 음(8분 이하) → 다음 마디
    # 첫 음이 같은 음높이이고 더 김) — ③은 붙임줄 인식이 빠진 곳(기도의 능력 23·45·53·59·63마디)에서 나왔고,
    # 이 곡 가사로 채점해 13곳 모두 맞음(오탐 0)
    tie_end = {id(a.stop) for a in getattr(score, "arcs", []) if getattr(a, "kind", "") == "tie"}
    synco = set()
    for i in range(1, len(ordered)):
        pm, pe, pg = ordered[i - 1]
        m, e, g = ordered[i]
        if pm is m or (pe.step, pe.octave, pe.alter) != (e.step, e.octave, e.alter):
            continue
        last = max((x for x in pm.elements if x.kind == "note" and x.staff_no == pe.staff_no), key=lambda x: x.src_x)
        if last in pg and pe.duration_base >= 8 and e.duration_base <= 4:
            synco.add(i)
    seq = []
    begun = start is None
    for i, (m, e, g) in enumerate(ordered):
        if not begun:
            if start[1] in g or (start[1] is None and m is start[0]):
                begun = True
            else:
                continue
        if skip_tied and (any(getattr(x, "tie_stop", False) or id(x) in tie_end for x in g) or i in synco):
            continue
        if i in slur_inside:
            continue
        seq.append((m, e))
    return seq


def fill(seq, syllables, row, pos_of, replace=True):
    """seq의 음에 syllables를 차례로. pos_of(e) → (x, y, h)(곡 좌표). replace면 쓴 범위의 그 절 가사를 먼저 비운다.
    반환: (넣은 음절 수, 남은 음절 수)"""
    n = min(len(seq), len(syllables))
    if replace:
        for m, e in seq[:max(n, 1)]:
            (getattr(e, "lyrics", None) or {}).pop(row, None)
    used = 0
    for (m, e), syl in zip(seq, syllables):
        if syl is None:
            continue
        x, y, h = pos_of(e)
        if getattr(e, "lyrics", None) is None:
            e.lyrics = {}
        e.lyrics[row] = {"text": syl, "x": x, "y": y, "h": h, "d": 0.0, "src": "typed"}
        used += 1
    return used, max(0, len(syllables) - len(seq))


def shift(seq, row, direction, pos_of):
    """seq[0]부터 그 절 가사를 한 칸 오른쪽(+1: 빈칸을 끼움) 또는 왼쪽(-1: 첫 음 가사를 빼고 당김)."""
    texts = [((e.lyrics or {}).get(row) or {}).get("text") for _m, e in seq]
    if direction > 0:
        texts = [None] + texts[:-1]
    else:
        texts = texts[1:] + [None]
    for (m, e), t in zip(seq, texts):
        if t is None:
            (e.lyrics or {}).pop(row, None)
        else:
            x, y, h = pos_of(e)
            old = (e.lyrics or {}).get(row) or {}
            if getattr(e, "lyrics", None) is None:
                e.lyrics = {}
            e.lyrics[row] = {"text": t, "x": x, "y": old.get("y", y), "h": old.get("h", h), "d": 0.0,
                             "src": "typed"}
