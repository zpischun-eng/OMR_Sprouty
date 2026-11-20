# score/chords.py
# ---------------------------------------------------------------------------
# [v3.00] 기타 코드(코드 이름) — 해석·표시·조옮김·구성음.
# 저장 형태(음 요소의 e.harmony, 언어와 무관한 내부 값):
#   {"root": "C"..."B", "alter": -1|0|1, "kind": MusicXML kind 값, "kind_text": 표시용 꼬리("m7","sus4"…),
#    "bass": "C"..."B"|None, "bass_alter": int, "x": 원본 x|None, "y": 원본 y|None, "src": "pdf"|"omr"|"user"}
# 화면 글자는 display()로 그때그때 만든다(저장값과 화면 문구 분리).
# ---------------------------------------------------------------------------
import re

_STEP_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHARP_NAMES = [("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0), ("F", 1), ("G", 0), ("G", 1),
                ("A", 0), ("A", 1), ("B", 0)]
_FLAT_NAMES = [("C", 0), ("D", -1), ("D", 0), ("E", -1), ("E", 0), ("F", 0), ("G", -1), ("G", 0), ("A", -1),
               ("A", 0), ("B", -1), ("B", 0)]

# 꼬리 글자 → (MusicXML kind, 반주 구성음(근음에서 반음))
_KINDS = [
    (("", "maj", "M"), "major", (0, 4, 7)),
    (("m", "min", "-", "mi"), "minor", (0, 3, 7)),
    (("7", "dom7"), "dominant", (0, 4, 7, 10)),
    (("maj7", "M7", "Δ", "Δ7", "ma7", "j7"), "major-seventh", (0, 4, 7, 11)),
    (("m7", "min7", "-7", "mi7"), "minor-seventh", (0, 3, 7, 10)),
    (("dim", "°", "o", "dim5"), "diminished", (0, 3, 6)),
    (("dim7", "°7", "o7"), "diminished-seventh", (0, 3, 6, 9)),
    (("m7b5", "m7♭5", "ø", "ø7", "min7b5", "-7b5"), "half-diminished", (0, 3, 6, 10)),
    (("aug", "+", "#5", "+5"), "augmented", (0, 4, 8)),
    (("aug7", "+7", "7#5", "7+5", "7+"), "augmented-seventh", (0, 4, 8, 10)),
    (("6", "maj6", "M6"), "major-sixth", (0, 4, 7, 9)),
    (("m6", "min6", "-6"), "minor-sixth", (0, 3, 7, 9)),
    (("9", "7(9)"), "dominant-ninth", (0, 4, 7, 10, 14)),
    (("maj9", "M9", "Δ9"), "major-ninth", (0, 4, 7, 11, 14)),
    (("m9", "min9", "-9"), "minor-ninth", (0, 3, 7, 10, 14)),
    (("11", "7(11)"), "dominant-11th", (0, 4, 7, 10, 14, 17)),
    (("13", "7(13)"), "dominant-13th", (0, 4, 7, 10, 14, 21)),
    (("sus4", "sus"), "suspended-fourth", (0, 5, 7)),
    (("7sus4", "7sus", "sus7"), "suspended-fourth", (0, 5, 7, 10)),
    (("sus2", "2"), "suspended-second", (0, 2, 7)),
    (("add9", "add2", "(add9)", "add 9"), "major", (0, 4, 7, 14)),
    (("madd9", "m(add9)"), "minor", (0, 3, 7, 14)),
    (("mM7", "m(maj7)", "mmaj7", "-maj7", "m(M7)"), "major-minor", (0, 3, 7, 11)),
    (("5",), "power", (0, 7)),
    (("7b9", "7♭9", "7(b9)"), "dominant", (0, 4, 7, 10, 13)),
    (("7#9", "7♯9", "7(#9)"), "dominant", (0, 4, 7, 10, 15)),
]
_KIND_OF = {}
for tails, kind, tones in _KINDS:
    for t in tails:
        _KIND_OF[t] = (kind, tones)

_RE = re.compile(r"^\s*([A-Ga-g])([#♯b♭]?)(.*?)(?:/([A-Ga-g])([#♯b♭]?))?\s*$")


def _alt(ch):
    return {"#": 1, "♯": 1, "b": -1, "♭": -1}.get(ch or "", 0)


def parse(text):
    """'C#m7/G#' → 코드 dict(위치 없이). 코드로 읽을 수 없으면 None."""
    if not text:
        return None
    t = text.strip().replace("（", "(").replace("）", ")")
    mo = _RE.match(t)
    if not mo:
        return None
    root, acc, tail, bass, bacc = mo.groups()
    tail = tail.strip()
    if tail not in _KIND_OF:
        # 괄호 안 텐션(예: 7(b9))이나 대소문자 섞임을 조금 너그럽게
        alt_tail = tail.replace(" ", "")
        if alt_tail not in _KIND_OF:
            return None
        tail = alt_tail
    kind, _tones = _KIND_OF[tail]
    return {"root": root.upper(), "alter": _alt(acc), "kind": kind, "kind_text": tail,
            "bass": bass.upper() if bass else None, "bass_alter": _alt(bacc) if bass else 0}


def is_chord_text(text):
    return parse(text) is not None


def display(h):
    """화면에 그릴 글자(♯·♭ 기호 사용)."""
    if not h:
        return ""
    acc = {1: "♯", -1: "♭"}.get(h.get("alter", 0), "")
    s = h["root"] + acc + (h.get("kind_text") or "")
    if h.get("bass"):
        s += "/" + h["bass"] + {1: "♯", -1: "♭"}.get(h.get("bass_alter", 0), "")
    return s


def tones(h):
    """반주 구성음(근음에서 반음, 근음 MIDI 기준) — (근음 pc, [반음들], 베이스 pc|None)."""
    tail = h.get("kind_text") or ""
    _k, tns = _KIND_OF.get(tail, ("major", (0, 4, 7)))
    root_pc = (_STEP_PC[h["root"]] + h.get("alter", 0)) % 12
    bass_pc = ((_STEP_PC[h["bass"]] + h.get("bass_alter", 0)) % 12) if h.get("bass") else None
    return root_pc, list(tns), bass_pc


def _spell(pc, prefer_flats):
    return (_FLAT_NAMES if prefer_flats else _SHARP_NAMES)[pc % 12]


def transpose(h, semis, target_fifths):
    """반음 semis만큼 옮기고 목표 조에 맞는 철자로(내림표 조면 ♭, 올림표 조면 ♯)."""
    if not h or not semis:
        return h
    flats = target_fifths < 0
    pc = (_STEP_PC[h["root"]] + h.get("alter", 0) + semis) % 12
    h["root"], h["alter"] = _spell(pc, flats)
    if h.get("bass"):
        bpc = (_STEP_PC[h["bass"]] + h.get("bass_alter", 0) + semis) % 12
        h["bass"], h["bass_alter"] = _spell(bpc, flats)
    return h
