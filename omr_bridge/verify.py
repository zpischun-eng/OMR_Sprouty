# omr_bridge/verify.py
# ---------------------------------------------------------------------------
# [v3.60] 일괄 검증 — 악보 폴더를 통째로 인식해 '원본 그림'과 '편집 악보 그림'을 겹쳐 비교한다.
#
# 비교 방법(허용 오차를 둔 겹침 비교):
#   · 그냥 XOR은 쓸 수 없다 — 맞게 읽은 음표도 글꼴·위치가 한두 픽셀 달라 절반이 '다름'으로 나온다.
#   · 그래서 서로 오선 간격의 약 1/3 안에 상대 잉크가 있으면 같은 것으로 본다.
#       miss  = 원본에만 있는 잉크(인식이 빠뜨렸거나 그리지 못한 것)
#       extra = 편집 악보에만 있는 잉크(잘못 읽었거나 위치가 다른 것)
#   · 가사·제목 같은 글자는 편집 악보에 원본과 같은 모양으로 그릴 수 없으므로 따로 센다(유형 'text').
# 차이 덩어리(blob)마다 유형(내부 키)·크기·마디를 적고, 조각 그림(원본 | 편집 악보 | 차이)을 만든다.
# 유형 키는 언어와 무관한 내부 값이고, 화면·보고서 문구는 TYPE_LABELS로 그때 바꾼다.
#
# 결과 폴더:
#   summary.csv   곡(파일)별 점수표          blobs.csv   차이 덩어리 전체 목록
#   index.html    나쁜 순서 보고서(내 PC에서 보기)   pages/   페이지별 차이 그림   crops/   조각 그림
#   items/        파일별 결과(JSON) — 다시 돌리면 끝난 파일은 건너뛴다
#   보내기용 zip  점수표 + 덩어리 목록 + 유형별로 고른 조각 그림(작게)
# ---------------------------------------------------------------------------
import csv
import hashlib
import json
import os
import time
import traceback
import zipfile
from pathlib import Path

import cv2
import numpy as np

EXTS = (".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
VERIFY_VERSION = "4"                    # 비교 방법이 바뀌면 올린다(예전 items 결과를 다시 계산)
GOOD = ("ok", "anacrusis", "time_change")

# 유형(내부 키) → 보고서 문구
TYPE_LABELS = {
    "dot": "점(부점·스타카토·도돌이 점)",
    "head": "음표 머리 크기",
    "vline": "세로선(기둥·세로줄)",
    "hline": "가로로 긴 것(빔·이음줄·셈여림 선·괄호)",
    "moved": "위치·모양이 다름(둘 다 있음)",
    "symbol": "기호 크기(쉼표·임시표·음자리표·셈여림 등)",
    "big": "큰 덩어리(여러 기호·마디 통째)",
    "text": "글자(가사·제목·빠르기말 등)",
}
TYPE_ORDER = ["big", "symbol", "head", "moved", "hline", "vline", "dot"]     # 글자는 덩어리로 세지 않고 text_pct


# ------------------------------------------------------------------ 파일 찾기
def find_scores(folder, out_dir=None):
    """폴더(하위 폴더 포함)의 악보 파일 목록(이름 순). 결과 폴더 안은 뺀다."""
    folder = Path(folder)
    out_dir = Path(out_dir).resolve() if out_dir else None
    found = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        rp = p.resolve()
        if out_dir and (out_dir == rp or out_dir in rp.parents):
            continue
        if any(part.startswith("_sprouty_verify") for part in p.parts):
            continue
        found.append(p)
    return found


def pick_sample(files, limit):
    """limit곡만 고르게 뽑는다(0·None이면 전부) — 앞에서부터가 아니라 전체에 걸쳐."""
    if not limit or limit >= len(files):
        return list(files)
    step = len(files) / float(limit)
    return [files[int(i * step)] for i in range(limit)]


def file_id(path):
    st = os.stat(path)
    h = hashlib.sha1(f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8"))
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ 한 파일 → 페이지별 (원본, 편집 악보)
class PageData:
    def __init__(self, page_no, orig_gray, ren_gray, staves, il, boxes, text_boxes=None):
        self.text_boxes = text_boxes or []   # [(x0, y0, x1, y1)] 코드 이름 자리(페이지 좌표) — 글자로 뺀다
        self.page_no = page_no
        self.orig = orig_gray           # 원본(회색)
        self.ren = ren_gray             # 편집 악보(회색, 원본과 같은 크기·좌표)
        self.staves = staves            # [(줄 번호, top, bottom)] — 페이지 좌표
        self.il = il                    # 오선 간격(px)
        self.boxes = boxes              # [(마디 번호, 줄 번호, x0, x1)]


def _gray(bgr):
    if bgr.ndim != 3:
        return bgr
    from omr_bridge.ink_prep import ink_gray, has_color
    # 색 잉크(색음표·빨간 악보)는 인식 때와 같은 기준으로 검게 — 안 그러면 밝은 색 머리가 '원본에 없음'이 된다
    return ink_gray(bgr) if has_color(bgr) else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _staves_from_groups(groups, sy, dy=0.0, h=None):
    out = []
    items = groups.items() if isinstance(groups, dict) else enumerate(groups)
    for line, staves in items:
        for lines in staves:
            top, bot = lines[0] * sy - dy, lines[-1] * sy - dy
            if h is None or (bot > 0 and top < h):
                out.append((line, top, bot))
    return out


def load_pages(path, progress=None, timeout_sec=900):
    """파일을 인식해 페이지별 비교 자료를 만든다. 반환 (pages, score, 메모들)."""
    from score import layout as sl, render as sr
    from omr_bridge.page_builder import build_page
    path = str(path)
    if path.lower().endswith(".pdf"):
        from omr_bridge.book import n_pdf_pages, BookJob, BookState, append_page
        if n_pdf_pages(path) > 1:
            job = BookJob(path)
            job.prepare()
            st = BookState()
            for n in range(1, job.n_pages + 1):
                if progress:
                    progress(f"{n}/{job.n_pages}쪽 인식")
                job.process_sheet(n, timeout_sec=timeout_sec)
                doc = job.page_doc(n)
                append_page(st, build_page(doc, sheet=n), n,
                            (doc.stacked_bgr.shape[1], doc.stacked_bgr.shape[0]), job.logical_parts(n))
            lay = sl.layout_score_world(st.score, st.by_sys, 1.0, 1.0, st.barlines or None,
                                        staff_groups=st.staff_groups, staff_left_by_line=st.staff_left,
                                        staff_clefs=st.staff_clefs, staff_parts=st.staff_parts)
            lay.header_items = st.header_items
            pages = []
            for i, p in enumerate(st.pages):
                w, h = st.page_sizes[i]
                pv = sl.page_view(lay, p.dy_orig)
                pv.no_measure_numbers = True
                img, lm = sr.render(pv, w, h, F=1, return_margin=True, lyric_y_off=p.dy_orig)
                ren = np.array(img.convert("L").crop((lm, 0, lm + w, h)))
                orig = _gray(job.page_image(i + 1))
                staves = _staves_from_groups(st.staff_groups, 1.0, p.dy_orig, h)
                il = _il_of(st.staff_groups, 1.0)
                boxes = [(mb.measure.number, mb.measure.system, mb.bbox[0], mb.bbox[2])
                         for mb in lay.measure_boxes]
                pages.append(PageData(i + 1, orig, ren, staves, il, boxes,
                                      _harmony_boxes(st.score, il, 1.0, 1.0, p.dy_orig, h, st.header_items)))
            return pages, st.score, list(st.warnings)
    from omr_bridge.document_adapter import run_document
    if progress:
        progress("인식")
    doc = run_document(path)
    pm = build_page(doc, sheet=1)
    pm.score.load_notes = pm.notes
    lay = sl.layout_score_world(pm.score, staves_by_system=pm.by_sys, sx=pm.sx, sy=pm.sy, barlines=pm.barlines,
                                staff_groups=pm.staff_groups, staff_left_by_line=pm.staff_left,
                                staff_clefs=pm.staff_clefs, staff_parts=pm.staff_parts)
    lay.header_items = pm.header_items
    lay.no_measure_numbers = True
    oh, ow = doc.stacked_bgr.shape[:2]
    img, lm = sr.render(lay, ow, oh, F=1, return_margin=True)
    ren = np.array(img.convert("L").crop((lm, 0, lm + ow, oh)))
    staves = _staves_from_groups(pm.staff_groups, pm.sy)
    boxes = [(mb.measure.number, mb.measure.system, mb.bbox[0], mb.bbox[2]) for mb in lay.measure_boxes]
    il1 = _il_of(pm.staff_groups, pm.sy)
    return [PageData(1, _gray(doc.stacked_bgr), ren, staves, il1, boxes,
                     _harmony_boxes(pm.score, il1, pm.sx, pm.sy, 0.0, oh, pm.header_items))], \
        pm.score, list(doc.warnings)


def brush_candidates(score, orig_bgr, by_sys, sx, sy, barlines, staff_groups, staff_left, staff_clefs,
                     staff_parts, header_items):
    """[v4.10] 붓 후보 자리: 지금 악보를 원본 크기로 그려 원본과 겹쳐 보고, '원본에만 있는 잉크' 덩어리의 상자를
    돌려준다(원본 그림 좌표 [(x0, y0, x1, y1, 유형)]). 글자(가사·코드 이름·제목)와 글꼴 차이는 일괄 검증과 같은
    방법으로 뺀다. 자리만 옮겨 그려진 것('moved')과 편집 악보 쪽이 더 많은 덩어리는 후보가 아니다.
    한 페이지 악보용(편집기에서 불러온 직후·붓을 켤 때 부른다)."""
    from score import layout as sl, render as sr
    lay = sl.layout_score_world(score, staves_by_system=by_sys, sx=sx, sy=sy, barlines=barlines,
                                staff_groups=staff_groups, staff_left_by_line=staff_left,
                                staff_clefs=staff_clefs, staff_parts=staff_parts)
    lay.header_items = header_items
    lay.no_measure_numbers = True
    oh, ow = orig_bgr.shape[:2]
    img, lm = sr.render(lay, ow, oh, F=1, return_margin=True)
    ren = np.array(img.convert("L").crop((lm, 0, lm + ow, oh)))
    staves = _staves_from_groups(staff_groups, sy)
    il = _il_of(staff_groups, sy)
    boxes = [(mb.measure.number, mb.measure.system, mb.bbox[0], mb.bbox[2]) for mb in lay.measure_boxes]
    pd = PageData(1, _gray(orig_bgr), ren, staves, il, boxes,
                  _harmony_boxes(score, il, sx, sy, 0.0, oh, header_items))
    out = []
    res = diff_page(pd)
    miss_mask = res["_masks"][2]
    for b in res["blobs"]:
        if b["type"] == "moved" or b["miss"] < 0.12 * il * il or b["miss"] < 2 * b["extra"]:
            continue
        if b.get("staff_dist", 0) > 6:
            continue
        # 그 덩어리의 '원본에만 있는 잉크'를 가로 토막들로(두 줄에 하나) — 붓이 상자 전체가 아니라 그 잉크만 칠하게.
        # 상자를 통째로 칠하면 옆 음표의 기둥·오선이 섞여 이음줄을 못 읽는다(성가 451번 8마디).
        x0, y0, x1, y1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
        sub = miss_mask[y0:y1, x0:x1]
        runs = []
        for r in range(0, sub.shape[0], 2):
            row = sub[r]
            if not row.any():
                continue
            d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
            for s_, e_ in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
                runs.append((int(x0 + s_), int(x0 + e_ - 1), int(y0 + r)))
        out.append((x0, y0, x1, y1, b["type"], runs))
    return out


def _harmony_boxes(score, il, sx, sy, dy, h, items=None):
    """코드 이름·악보 안 글자(Fine·빠르기말 등)가 그려지는 자리(페이지 좌표 상자) — 원본에서 읽은 위치 둘레."""
    out = []
    for it in items or []:
        if it.get("x0") is None or it.get("y") is None:
            continue
        hh = (it.get("h") or il) * sy
        y = it["y"] * sy - dy
        out.append((it["x0"] * sx - 0.6 * il, y - 0.9 * hh - 0.3 * il, it.get("x1", it["x0"]) * sx + 0.6 * il,
                    y + 0.9 * hh + 0.3 * il))
    for m in score.measures:
        for e in m.elements:
            hm = getattr(e, "harmony", None)
            if not hm or hm.get("x") is None or hm.get("y") is None:
                continue
            x, y = hm["x"] * sx, hm["y"] * sy - dy
            if -3 * il < y < h + 3 * il:
                out.append((x - 2.0 * il, y - 1.6 * il, x + 4.5 * il, y + 1.6 * il))
    return out


def _il_of(groups, sy):
    vals = []
    items = groups.values() if isinstance(groups, dict) else groups
    for staves in items:
        for lines in staves:
            if len(lines) >= 5:
                vals.append((lines[-1] - lines[0]) / 4.0 * sy)
    return float(np.median(vals)) if vals else 20.0


# ------------------------------------------------------------------ 한 페이지 비교
def _binarize(gray):
    """잉크(부울). 스캔·사진의 회색 바탕에도 견디게 Otsu 문턱(너무 밝거나 어두우면 150)."""
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = 150 if not (60 <= t <= 215) else t
    return gray < t


def _find_text(miss, staves, il):
    """원본에만 있는 잉크 중 '글자 줄'로 보이는 것(부울 마스크).
    오선 밖(위아래 0.6칸 여유)에 있고, 글자만 한 조각(높이 0.5~3칸)이 같은 높이에 5개 이상 늘어서 있으면 글자 줄."""
    h, w = miss.shape
    n, lab, st, cen = cv2.connectedComponentsWithStats(miss.astype(np.uint8), 8)
    cand = []
    for j in range(1, n):
        x, y, cw, ch, a = st[j]
        if not (0.5 * il <= ch <= 3.0 * il and cw <= 4.0 * il and a >= 0.08 * il * il):
            continue
        cy = y + ch / 2.0
        if any(top - 0.6 * il <= cy <= bot + 0.6 * il for _l, top, bot in staves):
            continue
        cand.append((cy, x, j, ch))
    text = np.zeros_like(miss)
    if not cand:
        return text
    cand.sort()
    rows, cur = [], [cand[0]]
    for c in cand[1:]:
        if c[0] - cur[-1][0] <= 0.9 * il:
            cur.append(c)
        else:
            rows.append(cur)
            cur = [c]
    rows.append(cur)
    keep = set()
    for r in rows:
        xs = [c[1] for c in r]
        if len(r) >= 5 and (max(xs) - min(xs)) >= 6 * il:
            keep.update(c[2] for c in r)
            # 그 줄 높이의 작은 조각(쉼표·마침표·받침 조각)도 같은 글자 줄
            y0, y1 = min(c[0] for c in r) - 1.2 * il, max(c[0] for c in r) + 1.2 * il
            for j in range(1, n):
                x, y, cw, ch, a = st[j]
                if y0 <= y + ch / 2.0 <= y1 and ch <= 3.0 * il and cw <= 4.0 * il:
                    keep.add(j)
    if keep:
        text = np.isin(lab, list(keep))
    return text


def _text_bands(ink, staves, il):
    """잉크에서 글자 줄 띠(부울 마스크). 오선 밖에서 글자만 한 조각이 같은 높이에 5개 이상, 6칸 넘게 늘어서 있고
    속이 덜 찬(음표 머리처럼 꽉 찬 타원이 아닌 — 덧줄 음표 줄을 글자로 오인하지 않게) 줄."""
    h, w = ink.shape
    n, lab, st, _c = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    cand = []
    for j in range(1, n):
        x, y, cw, ch, a = st[j]
        if not (0.45 * il <= ch <= 3.0 * il and 0.15 * il <= cw <= 4.0 * il and a >= 0.06 * il * il):
            continue
        cy = y + ch / 2.0
        if any(top - 0.6 * il <= cy <= bot + 0.6 * il for _l, top, bot in staves):
            continue
        cand.append((cy, x, x + cw, y, y + ch, a / float(cw * ch)))
    band = np.zeros((h, w), bool)
    if not cand:
        return band
    cand.sort()
    rows, cur = [], [cand[0]]
    for c in cand[1:]:
        if c[0] - cur[-1][0] <= 0.7 * il:
            cur.append(c)
        else:
            rows.append(cur)
            cur = [c]
    rows.append(cur)
    pad = int(0.5 * il)
    for r in rows:
        x0, x1 = min(c[1] for c in r), max(c[2] for c in r)
        fills = sorted(c[5] for c in r)
        if len(r) < 5 or x1 - x0 < 6 * il or fills[len(fills) // 2] >= 0.6:
            continue
        # 줄을 이루는 글자들의 x 범위만(띄엄띄엄 있는 코드 이름 사이의 음표까지 덮지 않게): 글자 주변만
        y0, y1 = max(0, int(min(c[3] for c in r) - pad)), min(h, int(max(c[4] for c in r) + pad))
        rs = sorted(r, key=lambda c: c[1])
        for c in rs:
            band[y0:y1, max(0, int(c[1] - 1.2 * il)):min(w, int(c[2] + 1.2 * il))] = True
        # [검증 4판] 글자 사이가 5칸 안이면 그 사이도 글자 줄(가사의 붙임표 '-'가 글자에서 떨어져 있어
        # 띠 밖에 남았다 — 전곡 검증의 'dot' 약 2,500개)
        for p_, c in zip(rs, rs[1:]):
            if 0 < c[1] - p_[2] < 5 * il:
                band[y0:y1, max(0, int(p_[2])):min(w, int(c[1]))] = True
    return band


def _cancel_shifted_stems(a, b, miss, extra, il):
    """기둥·세로줄이 옆으로 조금(0.9칸 안) 어긋난 것은 인식 오류가 아니라 그리기 위치 차이다 —
    원본·편집 악보 양쪽에 짝이 있는 세로선은 차이에서 뺀다(한쪽에만 있는 세로선 = 빠뜨린 기둥·세로줄은 남는다)."""
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(1.6 * il))))
    av = cv2.morphologyEx(a.astype(np.uint8), cv2.MORPH_OPEN, kv)
    bv = cv2.morphologyEx(b.astype(np.uint8), cv2.MORPH_OPEN, kv)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * int(0.9 * il) + 1, 1))
    a_ok = (av > 0) & (cv2.dilate(bv, kh) > 0)             # 원본 세로선 중 짝이 있는 것
    b_ok = (bv > 0) & (cv2.dilate(av, kh) > 0)
    kw = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    a_ok = cv2.dilate(a_ok.astype(np.uint8), kw) > 0
    b_ok = cv2.dilate(b_ok.astype(np.uint8), kw) > 0
    return miss & ~a_ok, extra & ~b_ok


def _outside_music(a, staves, il):
    """악보 밖(부울 마스크): 모든 오선에서 7칸 넘게 떨어진 줄(제목·작곡자), 오선 왼쪽 끝보다 1.5칸 넘게 왼쪽
    (파트 이름). 이곳의 원본 잉크는 글자로 본다."""
    h, w = a.shape
    out = np.ones((h, w), bool)
    left = w
    for _l, top, bot in staves:
        y0, y1 = max(0, int(top - 7 * il)), min(h, int(bot + 7 * il))
        out[y0:y1, :] = False
        r = int(min(h - 1, max(0, round(top))))
        row = a[max(0, r - 2):r + 3].any(axis=0)
        run, need = 0, int(5 * il)
        for x in range(w):
            run = run + 1 if row[x] else 0
            if run >= need:
                left = min(left, x - run + 1)
                break
    if left < w:
        out[:, :max(0, int(left - 1.5 * il))] = True
    return out


def _classify(bw, bh, miss_px, extra_px, il):
    """차이 덩어리의 유형(내부 키) — 크기·모양·양쪽 잉크 비율로."""
    w_, h_ = bw / il, bh / il
    both = min(miss_px, extra_px) / float(max(1, max(miss_px, extra_px)))
    if w_ >= 4.5 and h_ >= 3.0 or w_ * h_ >= 16:
        return "big"
    if w_ <= 0.9 and h_ <= 0.9:
        return "dot"
    if h_ >= 2.2 and w_ <= 0.8:
        return "vline"
    if w_ >= 2.5 and h_ <= 1.3 or (w_ >= 4.0 and h_ <= 2.5):
        return "hline"
    if both >= 0.35:
        return "moved"
    if w_ <= 2.0 and h_ <= 1.6:
        return "head"
    return "symbol"


def diff_page(pd):
    """한 페이지 비교. 반환 dict: 잉크 수·차이 수·덩어리 목록 + 그림용 마스크."""
    il = pd.il
    a = _binarize(pd.orig)
    b = pd.ren < 150
    k = max(3, int(round(il * 0.35))) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ad = cv2.dilate(a.astype(np.uint8), ker) > 0
    bd = cv2.dilate(b.astype(np.uint8), ker) > 0
    miss, extra = a & ~bd, b & ~ad
    miss, extra = _cancel_shifted_stems(a, b, miss, extra, il)
    # 가는 테두리 잔여(글꼴 굵기 차이)는 지운다
    k2 = max(2, int(round(il * 0.10)))
    op = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
    miss = cv2.morphologyEx(miss.astype(np.uint8), cv2.MORPH_OPEN, op) > 0
    extra = cv2.morphologyEx(extra.astype(np.uint8), cv2.MORPH_OPEN, op) > 0
    outside = _outside_music(a, pd.staves, il)
    # [검증 3판] 글자 줄(가사·코드 이름)을 차이가 아니라 원본·편집 악보 '전체 잉크'에서 찾아 그 띠 안의 차이를
    # 모두 글자로 본다. 2판은 차이 조각에서 찾아, 글꼴이 달라 조금씩 어긋난 글자 조각(5개 미만)이 남아
    # 성가 20곡의 '덩어리'의 대부분이 글자였다.
    band = _text_bands(a, pd.staves, il) | _text_bands(b, pd.staves, il)
    for x0, y0, x1, y1 in pd.text_boxes:          # 코드 이름(띄엄띄엄 있어 글자 줄로 안 잡힌다 — 글꼴만 다름)
        band[max(0, int(y0)):max(0, int(y1)), max(0, int(x0)):max(0, int(x1))] = True
    text_m = _find_text(miss, pd.staves, il) | (miss & (outside | band))
    text_r = _find_text(extra, pd.staves, il) | (extra & (outside | band))
    miss_m = miss & ~text_m
    extra = extra & ~text_r
    text = text_m | text_r
    union = (miss_m | extra).astype(np.uint8)
    km = max(3, int(round(il * 0.6))) | 1
    merged = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (km, km)))
    n, lab, st, _c = cv2.connectedComponentsWithStats(merged, 8)
    blobs = []
    lab_u = lab * union                                     # 실제 차이 픽셀에만 덩어리 번호
    miss_cnt = np.bincount(lab_u[miss_m].ravel(), minlength=n)
    extra_cnt = np.bincount(lab_u[extra].ravel(), minlength=n)
    for j in range(1, n):
        mp, ep = int(miss_cnt[j]), int(extra_cnt[j])
        if mp + ep < 0.10 * il * il:
            continue
        x, y, bw, bh, _a = st[j]
        bw2, bh2 = max(1, bw - km + 1), max(1, bh - km + 1)   # 합치느라 불린 만큼 되돌린 크기
        # [검증 4판] 꼬리·머리 모양의 글꼴 차이(양쪽에 다 있고 아주 작은 것)는 인식 오류가 아니다 —
        # 전곡 검증 덩어리의 1/3(약 6,400개)이 8분음표 꼬리 곡선 차이였다
        if mp > 0 and ep > 0 and bw2 <= 1.3 * il and bh2 <= 1.7 * il and mp + ep < 0.45 * il * il:
            continue
        cx, cy = x + bw / 2.0, y + bh / 2.0
        line, dist = None, 1e9
        for l_, top, bot in pd.staves:
            d = 0.0 if top <= cy <= bot else min(abs(cy - top), abs(cy - bot))
            if d < dist:
                line, dist = l_, d
        meas = None
        for num, sysno, x0, x1 in pd.boxes:
            if sysno == line and x0 <= cx <= x1:
                meas = num
                break
        blobs.append({"page": pd.page_no, "x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
                      "miss": mp, "extra": ep, "type": _classify(bw2, bh2, mp, ep, il),
                      "measure": meas, "staff_dist": round(dist / il, 2)})
    ink_o, ink_r = int(a.sum()), int(b.sum())
    tpx = int((text & a).sum())                             # 글자 비율은 원본 잉크 기준
    return {"ink_orig": ink_o, "ink_ren": ink_r, "miss": int(miss_m.sum()), "extra": int(extra.sum()),
            "text": tpx, "blobs": blobs,
            "_masks": (a, b, miss_m, extra, text)}


def diff_image(masks):
    """차이 그림(BGR): 같은 잉크 회색, 원본에만 빨강, 편집 악보에만 파랑, 글자 연한 주황."""
    a, b, miss, extra, text = masks
    vis = np.full(a.shape + (3,), 255, np.uint8)
    vis[a | b] = (190, 190, 190)
    vis[text] = (150, 200, 255)
    vis[miss] = (0, 0, 230)
    vis[extra] = (230, 60, 0)
    return vis


def crop_triptych(pd, masks, blob, max_w=300):
    """조각 그림: 원본 | 편집 악보 | 차이 (덩어리 둘레 2.5칸 여유)."""
    il = pd.il
    h, w = pd.orig.shape
    m = int(2.5 * il)
    x0, y0 = max(0, blob["x"] - m), max(0, blob["y"] - m)
    x1, y1 = min(w, blob["x"] + blob["w"] + m), min(h, blob["y"] + blob["h"] + m)
    o = cv2.cvtColor(pd.orig[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    r = cv2.cvtColor(pd.ren[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    d = diff_image(tuple(mk[y0:y1, x0:x1] for mk in masks))
    sc = min(1.0, max_w / float(max(1, x1 - x0)), 260.0 / float(max(1, y1 - y0)))
    if sc < 1.0:
        sz = (max(1, int((x1 - x0) * sc)), max(1, int((y1 - y0) * sc)))
        o, r, d = (cv2.resize(v, sz, interpolation=cv2.INTER_AREA) for v in (o, r, d))
    gap = np.full((o.shape[0], 6, 3), 255, np.uint8)
    gap[:, 2:4] = (120, 120, 120)
    return np.hstack([o, gap, r, gap, d])


def _imwrite(path, img, params=None):
    """한글 경로에서도 저장되게(cv2.imwrite는 Windows 한글 경로에서 실패한다)."""
    ext = os.path.splitext(str(path))[1]
    ok, buf = cv2.imencode(ext, img, params or [])
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
    return ok


# ------------------------------------------------------------------ 한 파일 검증
def verify_file(path, out_dir, rel_name, progress=None, crops_per_file=10, timeout_sec=900):
    """한 파일을 검증해 결과 dict를 돌려준다(그림은 out_dir에 저장). 실패해도 예외 대신 status='fail'."""
    from score import editing as ed
    out_dir = Path(out_dir)
    fid = file_id(path)
    t0 = time.time()
    item = {"id": fid, "file": rel_name, "status": "ok", "error": "", "verify_version": VERIFY_VERSION,
            "pages": 0, "measures": 0, "measures_ok": 0, "bad_measures": [],
            "ink_orig": 0, "ink_ren": 0, "miss": 0, "extra": 0, "text": 0, "blobs": [], "notes": []}
    try:
        pages, score, notes = load_pages(path, progress=progress, timeout_sec=timeout_sec)
        item["notes"] = [str(n) for n in notes][:8]
        v = ed.validate(score)
        item["measures"] = len(v)
        item["measures_ok"] = sum(1 for s in v.values() if s[0] in GOOD)
        item["bad_measures"] = sorted(k for k, s in v.items() if s[0] not in GOOD)
        item["pages"] = len(pages)
        (out_dir / "pages").mkdir(parents=True, exist_ok=True)
        (out_dir / "crops").mkdir(parents=True, exist_ok=True)
        all_blobs = []
        for pd in pages:
            res = diff_page(pd)
            for k in ("ink_orig", "ink_ren", "miss", "extra", "text"):
                item[k] += res[k]
            masks = res.pop("_masks")
            vis = diff_image(masks)
            sc = min(1.0, 1500.0 / vis.shape[1])
            if sc < 1.0:
                vis = cv2.resize(vis, (int(vis.shape[1] * sc), int(vis.shape[0] * sc)), interpolation=cv2.INTER_AREA)
            _imwrite(out_dir / "pages" / f"{fid}_p{pd.page_no}.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 72])
            # 조각 그림: 큰 것부터, 유형이 고르게(한 유형이 다 차지하지 않게)
            by_type = {}
            for b in sorted(res["blobs"], key=lambda b: -(b["miss"] + b["extra"])):
                by_type.setdefault(b["type"], []).append(b)
            quota = max(4, crops_per_file // max(1, len(pages)))
            taken, rnd = [], 0
            while len(taken) < quota and any(len(v_) > rnd for v_ in by_type.values()):
                for t in TYPE_ORDER:                       # 유형마다 큰 것부터 하나씩 돌아가며
                    if len(by_type.get(t, [])) > rnd and len(taken) < quota:
                        taken.append(by_type[t][rnd])
                rnd += 1
            for i, b in enumerate(taken):
                name = f"{fid}_p{pd.page_no}_{i:02d}_{b['type']}.png"
                if _imwrite(out_dir / "crops" / name, crop_triptych(pd, masks, b)):
                    b["crop"] = name
            all_blobs += res["blobs"]
        item["blobs"] = all_blobs
    except Exception as ex:                                   # 인식 실패·시간 초과·알 수 없는 형식
        item["status"] = "fail"
        item["error"] = f"{type(ex).__name__}: {ex}"[:6000]        # Audiveris 기록 끝부분까지(원인 분석용)
        item["trace"] = traceback.format_exc()[-1500:]
    item["seconds"] = round(time.time() - t0, 1)
    return item


def scores_of(item):
    """점수(%) — miss: 원본 잉크 중 편집 악보에 없는 비율, extra: 편집 악보 잉크 중 원본에 없는 비율(글자 제외)."""
    io = max(1, item["ink_orig"] - item["text"])
    ir = max(1, item["ink_ren"])
    miss, extra = 100.0 * item["miss"] / io, 100.0 * item["extra"] / ir
    ok = 100.0 * item["measures_ok"] / max(1, item["measures"])
    return round(miss, 2), round(extra, 2), round(ok, 1)


# ------------------------------------------------------------------ 폴더 일괄
def run_batch(folder, out_dir=None, limit=0, progress=None, cancelled=None, crops_per_file=10, app_version=""):
    """폴더를 일괄 검증한다. progress(i, n, 파일 이름, 단계 글) / cancelled() → True면 멈춤.
    끝난 파일은 items/에 저장해 두고 다시 돌릴 때 건너뛴다. 반환: 결과 요약 dict."""
    folder = Path(folder)
    out_dir = Path(out_dir) if out_dir else folder / "_sprouty_verify"
    (out_dir / "items").mkdir(parents=True, exist_ok=True)
    files = pick_sample(find_scores(folder, out_dir), limit)
    items = []
    for i, p in enumerate(files):
        if cancelled and cancelled():
            break
        rel = str(p.relative_to(folder)).replace("\\", "/")
        fid = file_id(p)
        ip = out_dir / "items" / f"{fid}.json"
        item = None
        if ip.exists():
            try:
                with open(ip, encoding="utf-8") as f:
                    old = json.load(f)
                if old.get("verify_version") == VERIFY_VERSION and old.get("app_version") == app_version \
                        and old.get("status") == "ok":
                    item = old
            except (OSError, ValueError):
                item = None
        if item is None:
            if progress:
                progress(i, len(files), rel, "")
            item = verify_file(p, out_dir, rel, crops_per_file=crops_per_file,
                               progress=(lambda s, _i=i, _r=rel: progress(_i, len(files), _r, s)) if progress else None)
            item["app_version"] = app_version
            with open(ip, "w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False)
        items.append(item)
    if progress:
        progress(len(items), len(files), "", "보고서 만드는 중")
    return write_reports(out_dir, items, folder, app_version, total=len(files))


def write_reports(out_dir, items, folder="", app_version="", total=None):
    out_dir = Path(out_dir)
    rows = []
    for it in items:
        miss, extra, ok = scores_of(it) if it["status"] == "ok" else (None, None, None)
        counts = {t: 0 for t in TYPE_ORDER}
        for b in it["blobs"]:
            counts[b["type"]] = counts.get(b["type"], 0) + 1
        rows.append({"file": it["file"], "status": it["status"], "pages": it["pages"], "measures": it["measures"],
                     "measures_ok_pct": ok, "miss_pct": miss, "extra_pct": extra,
                     "diff_score": None if miss is None else round(miss + extra, 2),
                     "text_pct": None if miss is None else round(100.0 * it["text"] / max(1, it["ink_orig"]), 1),
                     "blobs": len(it["blobs"]), **{f"n_{t}": counts[t] for t in TYPE_ORDER},
                     "bad_measures": " ".join(str(m) for m in it["bad_measures"][:40]),
                     "seconds": it.get("seconds"),
                     "error": (it.get("error", "") or "").split("\n")[0][:200], "id": it["id"]})
    rows.sort(key=lambda r: (r["status"] == "ok", -(r["diff_score"] or 0)))
    cols = list(rows[0].keys()) if rows else ["file"]
    with open(out_dir / "summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    bcols = ["file", "id", "page", "measure", "type", "x", "y", "w", "h", "miss", "extra", "staff_dist", "crop"]
    with open(out_dir / "blobs.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bcols, extrasaction="ignore")
        w.writeheader()
        for it in items:
            for b in it["blobs"]:
                w.writerow({**b, "file": it["file"], "id": it["id"]})
    ok_rows = [r for r in rows if r["status"] == "ok"]
    summary = {"folder": str(folder), "app_version": app_version, "verify_version": VERIFY_VERSION,
               "when": time.strftime("%Y-%m-%d %H:%M"), "files_total": total if total is not None else len(items),
               "files_done": len(items), "files_ok": len(ok_rows), "files_fail": len(rows) - len(ok_rows),
               "avg_miss_pct": round(float(np.mean([r["miss_pct"] for r in ok_rows])), 2) if ok_rows else None,
               "avg_extra_pct": round(float(np.mean([r["extra_pct"] for r in ok_rows])), 2) if ok_rows else None,
               "avg_measures_ok_pct": round(float(np.mean([r["measures_ok_pct"] for r in ok_rows])), 1)
               if ok_rows else None,
               "type_counts": {t: sum(r[f"n_{t}"] for r in rows) for t in TYPE_ORDER},
               "seconds_total": round(sum((it.get("seconds") or 0) for it in items), 1),
               "env": _env_info()}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    _write_html(out_dir, rows, items, summary)
    summary["zip"] = str(make_send_zip(out_dir, items, summary))
    summary["out_dir"] = str(out_dir)
    return summary


def _env_info():
    """실행 환경(실패 원인을 가릴 때 쓴다): OS·메모리·Audiveris 위치."""
    import platform
    info = {"platform": platform.platform(), "python": platform.python_version(), "cpu": os.cpu_count()}
    try:
        from omr_bridge import audiveris_runner as ar
        info["ram_total_free_mb"] = ar._total_ram_mb()
        info["audiveris"] = str(ar.find_audiveris_bin())
        info["java_tool_options"] = ar.jvm_env({}).get("JAVA_TOOL_OPTIONS")
    except Exception as ex:
        info["audiveris_error"] = str(ex)[:200]
    return info


def _write_html(out_dir, rows, items, summary):
    import html as _h
    by_id = {it["id"]: it for it in items}
    s = summary
    parts = ["<!doctype html><html lang='ko'><meta charset='utf-8'><title>콩나물지기 일괄 검증</title>",
             "<style>body{font-family:'Malgun Gothic',sans-serif;margin:20px;color:#222}"
             "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:3px 7px;"
             "text-align:right}td.l,th.l{text-align:left}tr.fail td{background:#fde}"
             "h2{margin-top:34px;border-top:2px solid #888;padding-top:10px;font-size:16px}"
             "img{max-width:100%;border:1px solid #ddd;margin:4px 0}.c img{max-height:170px;margin:3px}"
             ".k{color:#666;font-size:12px}</style>",
             f"<h1>콩나물지기 일괄 검증 — v{_h.escape(s['app_version'])}</h1>",
             f"<p>{_h.escape(s['when'])} · 파일 {s['files_done']}개(정상 {s['files_ok']}, 실패 {s['files_fail']}) · "
             f"원본에만 평균 {s['avg_miss_pct']}% · 편집 악보에만 평균 {s['avg_extra_pct']}% · "
             f"마디 검산 통과 평균 {s['avg_measures_ok_pct']}%</p>",
             "<p class='k'>빨강 = 원본에만 있음(빠뜨림) · 파랑 = 편집 악보에만 있음(잘못 읽음·위치 다름) · "
             "연한 주황 = 글자(점수에서 뺌) · 회색 = 같음. 표는 차이가 큰 순서.</p>",
             "<table><tr><th class='l'>파일</th><th>쪽</th><th>마디</th><th>검산 통과%</th><th>원본에만%</th>"
             "<th>편집에만%</th><th>글자%</th><th>차이 덩어리</th>" +
             "".join(f"<th title='{_h.escape(TYPE_LABELS[t])}'>{t}</th>" for t in TYPE_ORDER) + "<th>초</th></tr>"]
    for r in rows:
        if r["status"] != "ok":
            parts.append(f"<tr class='fail'><td class='l'>{_h.escape(r['file'])}</td>"
                         f"<td class='l' colspan='{8 + len(TYPE_ORDER)}'>실패: {_h.escape(r['error'])}</td></tr>")
            continue
        parts.append(f"<tr><td class='l'><a href='#f{r['id']}'>{_h.escape(r['file'])}</a></td><td>{r['pages']}</td>"
                     f"<td>{r['measures']}</td><td>{r['measures_ok_pct']}</td><td>{r['miss_pct']}</td>"
                     f"<td>{r['extra_pct']}</td><td>{r['text_pct']}</td><td>{r['blobs']}</td>" +
                     "".join(f"<td>{r['n_' + t]}</td>" for t in TYPE_ORDER) + f"<td>{r['seconds']}</td></tr>")
    parts.append("</table><p class='k'>유형: " + " · ".join(
        f"<b>{t}</b> {_h.escape(TYPE_LABELS[t])}" for t in TYPE_ORDER) + "</p>")
    for r in rows:
        if r["status"] != "ok":
            continue
        it = by_id[r["id"]]
        parts.append(f"<h2 id='f{r['id']}'>{_h.escape(r['file'])} — 원본에만 {r['miss_pct']}% · 편집에만 "
                     f"{r['extra_pct']}% · 검산 통과 {r['measures_ok_pct']}%"
                     + (f" · 오류 마디 {_h.escape(r['bad_measures'])}" if r["bad_measures"] else "") + "</h2>")
        crops = [b for b in it["blobs"] if b.get("crop")]
        if crops:
            parts.append("<div class='c'>" + "".join(
                f"<img loading='lazy' src='crops/{b['crop']}' title='{b['type']} · {b['page']}쪽 "
                f"{'' if b['measure'] is None else str(b['measure']) + '마디'}'>" for b in crops) + "</div>")
        for p in range(1, it["pages"] + 1):
            parts.append(f"<img loading='lazy' src='pages/{r['id']}_p{p}.jpg'>")
    with open(out_dir / "index.html", "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def make_send_zip(out_dir, items, summary, per_type=70, per_file=2, max_mb=14):
    """보내기용 zip: 점수표·덩어리 목록 + 유형별로 고른 조각 그림. 악보 전체 그림은 넣지 않는다."""
    out_dir = Path(out_dir)
    chosen = {}
    pool = []
    for it in items:
        mine = sorted((b for b in it["blobs"] if b.get("crop")), key=lambda b: -(b["miss"] + b["extra"]))
        for b in mine[:per_file]:
            chosen[b["crop"]] = (it, b)
        pool += [(it, b) for b in mine]
    for t in TYPE_ORDER:
        got = sum(1 for _it, b in chosen.values() if b["type"] == t)
        cand = [x for x in pool if x[1]["type"] == t and x[1]["crop"] not in chosen]
        cand.sort(key=lambda x: -(x[1]["miss"] + x[1]["extra"]))
        if len(cand) > per_type:                      # 큰 것 절반 + 고르게 뽑은 것 절반
            half = per_type // 2
            rest = cand[half:]
            step = len(rest) / float(per_type - half)
            cand = cand[:half] + [rest[int(i * step)] for i in range(per_type - half)]
        for it, b in cand[:max(0, per_type - got)]:
            chosen[b["crop"]] = (it, b)
    zp = out_dir / f"sprouty_verify_send_{time.strftime('%Y%m%d_%H%M')}.zip"
    budget = max_mb * 1024 * 1024
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("summary.csv", "blobs.csv", "summary.json"):
            if (out_dir / name).exists():
                z.write(out_dir / name, name)
        fails = [{"file": it["file"], "error": it.get("error"), "trace": it.get("trace")}
                 for it in items if it["status"] != "ok"]
        z.writestr("failures.json", json.dumps(fails, ensure_ascii=False, indent=1))
        index = []
        used = 0
        # [검증 4판] 유형을 돌아가며 담는다(예전엔 유형 이름 순이라 용량이 차면 symbol·vline 조각이 통째로 빠졌다)
        per = {}
        for crop, (it, b) in sorted(chosen.items(), key=lambda kv: -(kv[1][1]["miss"] + kv[1][1]["extra"])):
            per.setdefault(b["type"], []).append((crop, (it, b)))
        order, k = [], 0
        while any(len(v) > k for v in per.values()):
            order += [v[k] for t, v in sorted(per.items()) if len(v) > k]
            k += 1
        for crop, (it, b) in order:
            fp = out_dir / "crops" / crop
            if not fp.exists():
                continue
            sz = fp.stat().st_size
            if used + sz > budget:
                continue
            used += sz
            z.write(fp, f"crops/{b['type']}/{crop}")
            index.append({"crop": f"crops/{b['type']}/{crop}", "file": it["file"], "page": b["page"],
                          "measure": b["measure"], "type": b["type"], "miss": b["miss"], "extra": b["extra"]})
        z.writestr("crops_index.json", json.dumps(index, ensure_ascii=False, indent=1))
    return zp
