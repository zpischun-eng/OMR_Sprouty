# make_synthetic.py
# ---------------------------------------------------------------------------
# [업그레이드 0단계] 합성 샘플 생성기 — 정답(ground truth)이 딸린 단선율
# 악보를 자동 조판한다.
#
#   MusicXML 생성(=100% 정확한 정답) → Verovio 로 SVG 조판 → PNG/PDF
#
# 목적:
#   * 회귀 검사 표본을 4개 실측 샘플 → 임의 개수의 합성 샘플로 확장.
#   * 다중 페이지(1단계)·못갖춘마디·조표·쉼표·부점 등 원하는 난이도의
#     케이스를 마음대로 만들어 파이프라인 전 구간을 검증.
#   * Verovio 는 프로젝트 렌더러와 완전히 독립된 조판기이므로
#     "자기 것으로 자기를 검증"하는 순환을 피한다.
#
# 사용:
#   python tools/make_synthetic.py            # 기본 스위트 생성
#   → resource/synthetic/<이름>/ 에 score.pdf, page*.png, truth.json, score.musicxml
#
# 주의: 합성 판각(Leipzig/Bravura 폰트, 컴퓨터 조판)은 엔진이 튜닝된 실측
# 스캔과 잉크 특성이 다르다. 여기서 나오는 정확도는 '새 판각에 대한
# 일반화 성능'의 정직한 기준선이며, 100%가 아니어도 실패가 아니다 —
# 5.2절의 후속 과제(판각별 적응 임계값)의 진행도를 재는 자가 된다.
# ---------------------------------------------------------------------------
import io
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

_STEP_SEMI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHARP_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
_FLAT_ORDER = ["B", "E", "A", "D", "G", "C", "F"]
# 온음계 스텝 순서(멜로디 보행용)
_SCALE = ["C", "D", "E", "F", "G", "A", "B"]


def _key_alter(fifths):
    alt = {}
    if fifths > 0:
        for s in _SHARP_ORDER[:fifths]:
            alt[s] = 1
    elif fifths < 0:
        for s in _FLAT_ORDER[:-fifths]:
            alt[s] = -1
    return alt


def _midi(step, octave, alter):
    return 12 * (octave + 1) + _STEP_SEMI[step] + alter


# --------------------------------------------------------------- 멜로디 생성
# 리듬 어휘: (4분음표 길이, duration_base, dotted)
RHYTHMS_BASIC = [(1.0, 4, False), (2.0, 2, False), (0.5, 8, False)]
RHYTHMS_DOTTED = RHYTHMS_BASIC + [(1.5, 4, True), (0.75, 8, True), (0.25, 16, False)]
RHYTHMS_FULL = RHYTHMS_DOTTED + [(4.0, 1, False), (3.0, 2, True)]


def _fill_measure(cap, rng, vocab, rest_p=0.12):
    """마디 용량(cap, 4분음표 단위)을 어휘로 정확히 채운다."""
    out = []
    left = cap
    guard = 0
    while left > 1e-9 and guard < 200:
        guard += 1
        opts = [r for r in vocab if r[0] <= left + 1e-9]
        if not opts:
            break
        # 남은 공간을 정확히 채울 수 있는 선택을 우선(막다른 길 방지)
        exact = [r for r in opts if abs(r[0] - left) < 1e-9]
        if exact and (left <= 0.75 or rng.random() < 0.35):
            r = rng.choice(exact)
        else:
            r = rng.choice(opts)
        # 0.75/0.25 짝 강제: 점8분 뒤엔 16분(가창 악보 관용)
        out.append((r, rng.random() < rest_p and r[0] <= 1.0))
        left -= r[0]
        if abs(r[0] - 0.75) < 1e-9 and left >= 0.25 - 1e-9:
            out.append(((0.25, 16, False), False))
            left -= 0.25
    return out


def make_melody(fifths=0, beats=4, beat_type=4, n_measures=16,
                anacrusis=False, vocab=RHYTHMS_DOTTED, seed=0,
                lyrics=None):
    """정답 데이터 생성 → (measures 토큰 목록, meta).

    토큰: ["note", step, octave, alter, q, base, dotted] | ["rest", q, base, dotted]
    """
    rng = random.Random(seed)
    cap = beats * 4.0 / beat_type
    alt_map = _key_alter(fifths)
    # 보행 멜로디: C4~C6 대역, 도약 확률 낮게
    deg = _SCALE.index(rng.choice(["C", "E", "G"]))
    octv = 4
    measures = []
    for mi in range(n_measures):
        mcap = cap
        if anacrusis and mi == 0:
            mcap = min(1.0, cap / 2)
        elif anacrusis and mi == n_measures - 1:
            mcap = cap - min(1.0, cap / 2)
        toks = []
        for (q, base, dotted), is_rest in _fill_measure(mcap, rng, vocab):
            if is_rest:
                toks.append(["rest", q, base, dotted])
                continue
            step = _SCALE[deg]
            toks.append(["note", step, octv, alt_map.get(step, 0), q, base, dotted])
            move = rng.choice([-2, -1, -1, 0, 1, 1, 2, 3, -3])
            deg += move
            while deg < 0:
                deg += 7
                octv -= 1
            while deg > 6:
                deg -= 7
                octv += 1
            octv = max(4, min(5, octv))
        measures.append(toks)
    meta = {"fifths": fifths, "beats": beats, "beat_type": beat_type,
            "anacrusis": anacrusis, "n_measures": n_measures, "seed": seed,
            "lyrics": bool(lyrics)}
    return measures, meta


# --------------------------------------------------------------- MusicXML 화
_TYPE = {1: "whole", 2: "half", 4: "quarter", 8: "eighth", 16: "16th", 32: "32nd"}
DIV = 8         # divisions per quarter


def measures_to_musicxml(measures, meta, title="synthetic", lyrics=None):
    fifths, beats, beat_type = meta["fifths"], meta["beats"], meta["beat_type"]
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<score-partwise version="3.1">',
             f'<work><work-title>{title}</work-title></work>',
             '<part-list><score-part id="P1"><part-name></part-name>'
             '</score-part></part-list>', '<part id="P1">']
    syl = list(lyrics) if lyrics else []
    si = 0
    for mi, toks in enumerate(measures):
        parts.append(f'<measure number="{mi + 1}">')
        if mi == 0:
            parts.append(
                f'<attributes><divisions>{DIV}</divisions>'
                f'<key><fifths>{fifths}</fifths></key>'
                f'<time><beats>{beats}</beats><beat-type>{beat_type}</beat-type></time>'
                '<clef><sign>G</sign><line>2</line></clef></attributes>')
        # 8분 이하 연속 음표에 빔 부여(2개씩 짝)
        beam_run = []
        for k, t in enumerate(toks):
            if t[0] == "note" and t[5 if t[0] == "note" else 2] >= 8:
                beam_run.append(k)
            else:
                beam_run.append(None)
        for k, t in enumerate(toks):
            if t[0] == "rest":
                _, q, base, dotted = t
                dur = int(round(q * DIV))
                parts.append(f'<note><rest/><duration>{dur}</duration>'
                             f'<type>{_TYPE[base]}</type>'
                             + ('<dot/>' if dotted else '') + '</note>')
            else:
                _, step, octv, alter, q, base, dotted = t
                dur = int(round(q * DIV))
                alter_x = f'<alter>{alter}</alter>' if alter else ''
                beam_x = ''
                if base >= 8:
                    prev_b = k > 0 and toks[k - 1][0] == "note" and toks[k - 1][5] >= 8
                    next_b = k + 1 < len(toks) and toks[k + 1][0] == "note" \
                        and toks[k + 1][5] >= 8
                    if not prev_b and next_b:
                        beam_x = '<beam number="1">begin</beam>'
                    elif prev_b and next_b:
                        beam_x = '<beam number="1">continue</beam>'
                    elif prev_b:
                        beam_x = '<beam number="1">end</beam>'
                lyr_x = ''
                if syl:
                    lyr_x = (f'<lyric number="1"><syllabic>single</syllabic>'
                             f'<text>{syl[si % len(syl)]}</text></lyric>')
                    si += 1
                parts.append(
                    f'<note><pitch><step>{step}</step>{alter_x}'
                    f'<octave>{octv}</octave></pitch><duration>{dur}</duration>'
                    f'<type>{_TYPE[base]}</type>'
                    + ('<dot/>' if dotted else '') + beam_x + lyr_x + '</note>')
        parts.append('</measure>')
    parts += ['</part>', '</score-partwise>']
    return "\n".join(parts)


# --------------------------------------------------------------- 조판·저장
def render_sample(mxl, out_dir, font="Leipzig", scale=52, page_w=2100,
                  target_px=2000):
    """Verovio 조판 → 페이지별 PNG(+크롭) → 다중 페이지 PDF."""
    import verovio
    import cairosvg
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    tk = verovio.toolkit()
    tk.setOptions({"scale": scale, "pageWidth": page_w, "pageHeight": 2970,
                   "adjustPageHeight": False, "breaks": "auto", "font": font,
                   "header": "none", "footer": "none",
                   "pageMarginLeft": 60, "pageMarginRight": 60,
                   "pageMarginTop": 80, "pageMarginBottom": 80})
    if not tk.loadData(mxl):
        raise RuntimeError("verovio 로드 실패")
    n_pages = tk.getPageCount()
    pages = []
    for p in range(1, n_pages + 1):
        svg = tk.renderToSVG(p)
        png = cairosvg.svg2png(bytestring=svg.encode(), output_width=target_px,
                               background_color="white")
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        # 내용 크롭: 페이지의 빈 여백을 잘라 오선이 폭 대부분을 차지하게
        # (실제 스캔 악보의 화면 점유율과 비슷하게 맞춘다)
        ink = (255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) > 40
        rows = np.nonzero(ink.sum(axis=1) > 2)[0]
        cols = np.nonzero(ink.sum(axis=0) > 2)[0]
        if len(rows) and len(cols):
            m = 45
            y0, y1 = max(0, rows[0] - m), min(img.shape[0], rows[-1] + m)
            x0, x1 = max(0, cols[0] - m), min(img.shape[1], cols[-1] + m)
            img = img[y0:y1, x0:x1]
        pages.append(img)
        cv2.imwrite(os.path.join(out_dir, f"page{p}.png"), img)
    pil_pages = [Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)) for p in pages]
    pil_pages[0].save(os.path.join(out_dir, "score.pdf"), save_all=True,
                      append_images=pil_pages[1:], resolution=200)
    return n_pages


def build_sample(name, out_root="resource/synthetic", lyrics=None, **kw):
    render_kw = {k: kw.pop(k) for k in ("font", "scale", "page_w")
                 if k in kw}
    measures, meta = make_melody(**kw)
    mxl = measures_to_musicxml(measures, meta, title=name, lyrics=lyrics)
    out_dir = os.path.join(out_root, name)
    n_pages = render_sample(mxl, out_dir, **render_kw)
    meta["n_pages"] = n_pages
    meta.update(render_kw)
    with open(os.path.join(out_dir, "score.musicxml"), "w", encoding="utf-8") as f:
        f.write(mxl)
    with open(os.path.join(out_dir, "truth.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "measures": measures}, f, ensure_ascii=False,
                  indent=1)
    n_el = sum(len(m) for m in measures)
    print(f"  {name}: {meta['n_measures']}마디 {n_el}요소 {n_pages}페이지 "
          f"(조표 {meta['fifths']:+d}, {meta['beats']}/{meta['beat_type']})")
    return out_dir


DEFAULT_SUITE = [
    # 이름, 파라미터 — 난이도 축: 조표·부점·못갖춘마디·페이지 수·판각 폰트
    ("syn01_c44_basic", dict(fifths=0, beats=4, beat_type=4, n_measures=16,
                             vocab=RHYTHMS_BASIC, seed=11)),
    ("syn02_g34_dotted", dict(fifths=1, beats=3, beat_type=4, n_measures=20,
                              vocab=RHYTHMS_DOTTED, seed=22)),
    ("syn03_f44_anacrusis", dict(fifths=-1, beats=4, beat_type=4, n_measures=24,
                                 anacrusis=True, vocab=RHYTHMS_DOTTED, seed=33)),
    ("syn04_d24_sixteenth", dict(fifths=2, beats=2, beat_type=4, n_measures=20,
                                 vocab=RHYTHMS_DOTTED, seed=44)),
    ("syn05_c44_long_dense", dict(fifths=0, beats=4, beat_type=4,
                                  n_measures=64, vocab=RHYTHMS_FULL, seed=55,
                                  scale=80)),
    ("syn07_g44_multipage", dict(fifths=1, beats=4, beat_type=4,
                                 n_measures=168, vocab=RHYTHMS_DOTTED, seed=77,
                                 scale=80)),
    ("syn06_bb44_bravura", dict(fifths=-2, beats=4, beat_type=4, n_measures=18,
                                vocab=RHYTHMS_DOTTED, seed=66, font="Bravura")),
]


def main():
    out_root = sys.argv[1] if len(sys.argv) > 1 else "resource/synthetic"
    print(f"합성 샘플 생성 → {out_root}/")
    for name, kw in DEFAULT_SUITE:
        build_sample(name, out_root, **kw)
    print("완료")


if __name__ == "__main__":
    main()
