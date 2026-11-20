# omr_bridge/chord_ocr.py
# ---------------------------------------------------------------------------
# [v3.20] 그림 악보(JPG·스캔 PDF)의 기타 코드 읽기.
# PDF 글자 층이 없으면 코드 글자는 그림뿐이다. Audiveris 글자 인식을 켜도 시험 그림의 코드 12개 중 1개만
# 코드로 나와(나머지는 쪼개지거나 빠짐) 쓸 수 없었다 → 직접 찾는다:
#   ① 보표마다 맨 윗줄 5칸 위 ~ 0.3칸 위 띠를 자르고, 이미 아는 기호(머리·기둥·빔·쉼표·이음줄 자리)를 하얗게 지운다
#   ② 남은 잉크가 거의 없으면 건너뛴다(대부분의 띠 — 속도)
#   ③ Tesseract(이 PC에 있을 때) 두 가지 방식(블록·흩어진 글자)으로 읽어 단어 상자를 얻고
#   ④ 근음이 대문자이고 코드 문법에 맞으며 확신도 60 이상·글자 높이가 칸 크기에 맞는 것만 코드로.
# 결과는 PDF 글자와 같은 꼴({text, x0, x1, y}, 그 페이지 그림 좌표)로 돌려 page_builder.attach_pdf_chords가 붙인다.
# ---------------------------------------------------------------------------
import os
import subprocess
import tempfile

import cv2
import numpy as np


def _mask_known(gray, score, y0, y1, il):
    """띠 안의 알고 있는 기호를 하얗게."""
    g = gray.copy()
    H, W = g.shape[:2]

    def white(x0, ya, x1, yb):
        ya, yb = int(max(y0, ya)) - y0, int(min(y1, yb)) - y0
        x0, x1 = int(max(0, x0)), int(min(W, x1))
        if yb > ya and x1 > x0:
            g[ya:yb, x0:x1] = 255
    for m in score.measures:
        for e in m.elements:
            if e.src_bbox:
                bx, by, bw, bh = e.src_bbox
                white(bx - 2, by - 2, bx + bw + 2, by + bh + 2)
            st = getattr(e, "src_stem", None)
            if st:
                white(st[0] - 3, st[1] - 2, st[0] + 3, st[2] + 2)
            for bm in (getattr(e, "src_beams", None) or []):
                try:
                    bx1, by1, bx2, by2, bh = bm[:5]
                    white(min(bx1, bx2) - 2, min(by1, by2) - bh - 0.2 * il, max(bx1, bx2) + 2,
                          max(by1, by2) + bh + 0.2 * il)
                except Exception:
                    pass
    return g


def _ocr(img, exe, tdir, psm):
    fn = os.path.join(tempfile.gettempdir(), f"sprouty_chord_band_{os.getpid()}.png")
    cv2.imwrite(fn, img)
    try:
        r = subprocess.run([exe, fn, "-", "--psm", psm, "--tessdata-dir", tdir, "-l", "eng",
                            "-c", "tessedit_create_tsv=1"], capture_output=True, text=True, timeout=30,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        return []
    out = []
    for ln in (r.stdout or "").splitlines()[1:]:
        f = ln.split("\t")
        if len(f) < 12 or not f[11].strip():
            continue
        try:
            out.append((f[11].strip(), int(f[6]), int(f[7]), int(f[8]), int(f[9]), float(f[10])))
        except ValueError:
            continue
    return out


MIN_CONF = 75            # 진짜 코드 76~92(시험 그림), 거짓(가사·꾸밈음 조각)은 그 아래가 대부분
_FIX = (("♯", "#"), ("♭", "b"), ("|", "/"), ("l/", "/"), ("\\", "/"))


def _normalize(t):
    t = t.strip(".,;:'\"`")
    for a, b in _FIX:
        t = t.replace(a, b)
    return t


def find_chords(gray, score, staff_groups, il, tess, scale=1.0):
    """gray: 그 페이지 그림(인식 좌표 = gray 좌표 × scale). 반환: [{text, x0, x1, y, kind:'text'}] (인식 좌표)."""
    from score import chords as _ch
    if not tess:
        return []
    exe, tdir = tess
    H, W = gray.shape[:2]
    items = []
    for line, grp in (staff_groups or {}).items():
        # 그 줄 맨 위 보표 위만 — 아래 보표 위 띠에는 윗보표의 가사가 있다(교회: 한글 가사를 'Co'로 읽음)
        notes_x = [e.src_bbox[0] for m in score.measures if m.system == line for e in m.elements
                   if e.src_bbox and e.staff_no == 1]
        x_min = (min(notes_x) - 1.5 * ((grp[0][4] - grp[0][0]) / 4.0)) / scale if notes_x and grp else 0
        for ys in grp[:1]:
            top = ys[0] / scale
            sil = (ys[4] - ys[0]) / 4.0 / scale
            y0, y1 = int(max(0, top - 5.0 * sil)), int(max(0, top - 0.3 * sil))
            if y1 - y0 < 2 * sil:
                continue
            band = gray[y0:y1]
            band = _mask_known(band, _ScaledScore(score, scale), y0, y1, sil)
            ink = (band < 140).mean()
            if ink < 0.002:
                continue                                 # 글자가 없는 띠
            words = {}
            for psm in ("6", "11"):
                for t, x, y, w, h, cf in _ocr(band, exe, tdir, psm):
                    t = _normalize(t)
                    if cf < MIN_CONF or not t or not t[0].isupper() or not _ch.is_chord_text(t):
                        continue
                    if x + w < x_min:                    # 줄 머리(마디 번호·음자리표·조표)
                        continue
                    if not (0.8 * sil <= h <= 2.6 * sil):
                        continue
                    k = int((x + w / 2.0) // (1.5 * sil))
                    old = words.get(k)
                    if old is None or cf > old[5]:
                        words[k] = (t, x, y, w, h, cf)
            for t, x, y, w, h, cf in words.values():
                items.append({"text": t, "kind": "text", "x0": (x) * scale, "x1": (x + w) * scale,
                              "y": (y0 + y + h / 2.0) * scale, "h": h * scale, "src": "ocr"})
    return items


class _ScaledScore:
    """인식 좌표의 악보 기하를 그림 좌표로(배율이 1이 아닐 때)."""

    def __init__(self, score, scale):
        self._s, self._k = score, scale

    @property
    def measures(self):
        if self._k == 1.0:
            return self._s.measures
        k = self._k
        out = []
        for m in self._s.measures:
            els = []
            for e in m.elements:
                class _E:
                    pass
                x = _E()
                x.src_bbox = tuple(v / k for v in e.src_bbox) if e.src_bbox else None
                st = getattr(e, "src_stem", None)
                x.src_stem = tuple(v / k for v in st) if st else None
                x.src_beams = [tuple(v / k for v in bm[:5]) for bm in (getattr(e, "src_beams", None) or [])]
                els.append(x)

            class _M:
                pass
            mm = _M()
            mm.elements = els
            out.append(mm)
        return out
