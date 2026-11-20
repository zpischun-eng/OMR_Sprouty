# tools/make_v110_test.py
# ---------------------------------------------------------------------------
# [v1.10 시험 악보] 다성부(한 보표 두 성부)·겹점·마디 중간 음자리표 변경·조표 바뀜을
# 한 곡에 담은 피아노 2보표 합성 악보를 만든다(정답 MusicXML → Verovio 조판 → PNG/PDF).
#   python tools/make_v110_test.py  → resource/synthetic/v110_poly/
# ---------------------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.make_synthetic import render_sample  # noqa: E402

DIV = 8   # 4분음표 = 8 (32분음표까지 정수)
TYPES = {"w": ("whole", 32), "h": ("half", 16), "q": ("quarter", 8), "e": ("eighth", 4),
         "s": ("16th", 2)}


def note(p, t, dots=0, voice=1, staff=1, stem=None, beam=None, chord=False, alter=None):
    """p='E5' 또는 'r'(쉼표). alter는 조표와 다를 때만(임시표 표시용) 쓴다."""
    tname, dur = TYPES[t]
    d = dur
    add = dur
    for _ in range(dots):
        add //= 2
        d += add
    x = ["<note>"]
    if chord:
        x.append("<chord/>")
    if p == "r":
        x.append("<rest/>")
    else:
        step, octv = p[0], p[-1]
        acc = p[1:-1]
        al = {"#": 1, "b": -1, "": 0}[acc]
        x.append(f"<pitch><step>{step}</step>" + (f"<alter>{al}</alter>" if al else "")
                 + f"<octave>{octv}</octave></pitch>")
    x.append(f"<duration>{d}</duration><voice>{voice}</voice><type>{tname}</type>")
    x += ["<dot/>"] * dots
    if alter:
        x.append(f"<accidental>{alter}</accidental>")
    if stem:
        x.append(f"<stem>{stem}</stem>")
    x.append(f"<staff>{staff}</staff>")
    if beam:
        x.append(f'<beam number="1">{beam}</beam>')
    x.append("</note>")
    return "".join(x)


def backup(q):
    return f"<backup><duration>{int(q * DIV)}</duration></backup>"


def measure(num, body, attrs=""):
    return f'<measure number="{num}">{attrs}{body}</measure>'


def build():
    m = []
    first_attr = ("<attributes><divisions>8</divisions><key><fifths>4</fifths></key>"
                  "<time><beats>4</beats><beat-type>4</beat-type></time><staves>2</staves>"
                  "<clef number=\"1\"><sign>G</sign><line>2</line></clef>"
                  "<clef number=\"2\"><sign>F</sign><line>4</line></clef></attributes>")
    U, D = "up", "down"
    # 1: 두 성부 + 성부별 빔
    m.append(measure(1, "".join([
        note("E5", "q", stem=U), note("F#5", "e", stem=U, beam="begin"),
        note("G#5", "e", stem=U, beam="end"), note("A5", "h", stem=U),
        backup(4),
        note("B4", "e", voice=2, stem=D, beam="begin"), note("B4", "e", voice=2, stem=D, beam="end"),
        note("C#5", "e", voice=2, stem=D, beam="begin"), note("B4", "e", voice=2, stem=D, beam="end"),
        note("A4", "h", voice=2, stem=D),
        backup(4),
        note("E2", "h", voice=5, staff=2), note("B2", "h", voice=5, staff=2),
    ]), first_attr))
    # 2: 겹점 2분 + 8분 / 아래 성부
    m.append(measure(2, "".join([
        note("B5", "h", dots=2, stem=U), note("A5", "e", stem=U),
        backup(4),
        note("G#4", "q", voice=2, stem=D), note("F#4", "q", voice=2, stem=D),
        note("E4", "h", voice=2, stem=D),
        backup(4),
        note("E3", "q", voice=5, staff=2), note("B3", "q", voice=5, staff=2),
        note("G#3", "h", voice=5, staff=2),
    ])))
    # 3: 단선율, 겹점 4분 / 왼손 높은음자리표로 바뀜(4마디에서 낮은음자리표로 복귀)
    m.append(measure(3, "".join([
        note("E5", "q", dots=2), note("D#5", "s"),
        note("C#5", "e", beam="begin"), note("B4", "e", beam="end"), note("E5", "q"),
        backup(4),
        note("E4", "h", voice=5, staff=2), note("G#4", "h", voice=5, staff=2),
    ]), '<attributes><clef number="2"><sign>G</sign><line>2</line></clef></attributes>'))
    m.append(measure(4, "".join([
        note("E5", "w", stem=U),
        backup(4),
        note("B4", "h", voice=2, stem=D), note("G#4", "h", voice=2, stem=D),
        backup(4),
        note("E2", "w", voice=5, staff=2),
    ]), '<attributes><clef number="2"><sign>F</sign><line>4</line></clef></attributes>'))
    # 5: 조표 바뀜(E장조 → F장조)
    key_f = "<attributes><key><fifths>-1</fifths></key></attributes>"
    m.append(measure(5, "".join([
        note("F5", "q", stem=U), note("G5", "e", stem=U, beam="begin"),
        note("A5", "e", stem=U, beam="end"), note("Bb5", "h", stem=U),
        backup(4),
        note("C5", "e", voice=2, stem=D, beam="begin"), note("A4", "e", voice=2, stem=D, beam="end"),
        note("Bb4", "e", voice=2, stem=D, beam="begin"), note("G4", "e", voice=2, stem=D, beam="end"),
        note("F4", "h", voice=2, stem=D),
        backup(4),
        note("F2", "h", voice=5, staff=2), note("C3", "h", voice=5, staff=2),
    ]), key_f))
    m.append(measure(6, "".join([
        note("A5", "h", dots=2, stem=U), note("G5", "e", stem=U),
        backup(4),
        note("F4", "q", voice=2, stem=D), note("E4", "q", voice=2, stem=D),
        note("F4", "h", voice=2, stem=D),
        backup(4),
        note("D3", "h", voice=5, staff=2), note("C3", "h", voice=5, staff=2),
    ])))
    m.append(measure(7, "".join([
        note("F5", "e", beam="begin"), note("E5", "e", beam="continue"),
        note("D5", "e", beam="continue"), note("C5", "e", beam="end"),
        note("Bb4", "q"), note("A4", "q"),
        backup(4),
        note("Bb2", "h", voice=5, staff=2), note("C3", "h", voice=5, staff=2),
    ])))
    m.append(measure(8, "".join([
        note("F5", "w"),
        backup(4),
        note("F2", "w", voice=5, staff=2),
    ]), "") .replace("</measure>",
                     '<barline location="right"><bar-style>light-heavy</bar-style></barline></measure>'))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<score-partwise version="3.1"><part-list><score-part id="P1">'
            '<part-name>Piano</part-name></score-part></part-list><part id="P1">'
            + "".join(m) + "</part></score-partwise>")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "resource", "synthetic", "v110_poly")
    os.makedirs(out, exist_ok=True)
    xml = build()
    with open(os.path.join(out, "score.musicxml"), "w", encoding="utf-8") as f:
        f.write(xml)
    render_sample(xml, out, font="Bravura", scale=60, page_w=2100, target_px=3600)
    print("ok", out)
