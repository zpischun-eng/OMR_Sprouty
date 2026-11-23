# omr_bridge/ink_check.py
# ---------------------------------------------------------------------------
# [v1.40 미인식 보완] 원본 그림에서 'Audiveris가 인식하지 못한 잉크'를 찾고, 빠진 음표 머리마다
# 넣을 음표를 제안한다.
#
# 방법(B): 원본 잉크 − Audiveris가 인식한 모든 기호의 원본 위 상자(.omr inter의 bounds — 머리·기둥·
# 빔·점·임시표·덧줄·쉼표·음자리표…) − 오선 − 가는 곡선. 남는 덩어리가 미인식 잉크다. 우리 편집 악보
# 그림과 겹치는 방식(A)은 글꼴·모양 차이가 잡음이 되어(빔 두께·가사 글꼴 등) 기각했다 — .omr 상자는
# 원본 좌표 그대로라 위치 오차가 없다. 묶음 상자(head-chord·beam-group 등)와 이음줄·쐐기 상자는 넓은
# 영역을 덮어 그 안의 빠진 기호를 가리므로 쓰지 않는다. 덧줄 상자도 쓰지 않는다 — 빠진 머리 한가운데를
# 지나가 머리를 위아래 두 조각으로 잘라 버린다("교회" 7마디 A3). 덧줄은 가로선 지우기로 처리한다.
#
# 제안: 빠진 머리마다 음높이(세로 위치 + 그 자리 음자리표·조표), 박 위치(같은 줄 다른 보표 포함,
# 마디를 꼭 채운 줄의 같은 자리), 길이(검정/빈 머리 + 그 성부의 빈 박), 성부(원본에서 찾은 기둥
# 방향)를 정한다. 같은 박에 이미 음이 있는 성부면 화음 구성음으로 제안한다. 자동으로 넣지 않는다.
# ---------------------------------------------------------------------------
import zipfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np

# 덮는 영역이 넓어 그 안의 빠진 기호를 가리는 묶음·곡선 inter는 '인식됨'으로 치지 않는다
_SKIP = {"head-chord", "rest-chord", "beam-group", "key", "time-pair", "slur", "wedge", "sheet", "ledger",
         "page", "system", "staff", "part", "measure", "stack", "bounds", "lyric-line", "sentence",
         "ending", "segment"}


def _sheet_root(omr_path, sheet=1):
    """[v2.00] .omr 안의 sheet#N/sheet#N.xml 루트(여러 페이지 책 지원)."""
    with zipfile.ZipFile(omr_path) as z:
        want = f"sheet#{sheet}/sheet#{sheet}.xml"
        names = z.namelist()
        name = want if want in names else next(
            (n for n in names if n.endswith(".xml") and n.startswith("sheet#")), None)
        return ET.fromstring(z.read(name))


def _trace_curve(mask, x_off, y_off):
    """[v2.30] 원본 잉크(부울 배열)에서 이음줄 곡선을 따낸다: 열마다 잉크의 가운데 y → 양 끝과 가장 많이 휜
    점을 지나는 2차 곡선을 3차 곡선(x1,y1,c1x,c1y,c2x,c2y,x2,y2)으로. 잉크가 모자라면 None."""
    cols = [(c, float(np.nonzero(mask[:, c])[0].mean())) for c in range(mask.shape[1]) if mask[:, c].any()]
    if len(cols) < 4:
        return None
    (xa, ya), (xb, yb) = cols[0], cols[-1]
    base = lambda x: ya + (yb - ya) * (x - xa) / max(1e-6, xb - xa)
    xm, ym = max(cols, key=lambda p: abs(p[1] - base(p[0])))
    # 꼭짓점(t=1/2)이 (xm, ym)인 2차 곡선의 제어점 Q = 2·P(1/2) − (P0+P2)/2
    qx, qy = 2 * xm - (xa + xb) / 2.0, 2 * ym - (ya + yb) / 2.0
    c1 = (xa + 2.0 / 3 * (qx - xa), ya + 2.0 / 3 * (qy - ya))
    c2 = (xb + 2.0 / 3 * (qx - xb), yb + 2.0 / 3 * (qy - yb))
    return (xa + x_off, ya + y_off, c1[0] + x_off, c1[1] + y_off, c2[0] + x_off, c2[1] + y_off,
            xb + x_off, yb + y_off)


_CURVE_CACHE = {}


def _slur_curves(omr_path, sheet=1):
    """_omr_boxes의 이음줄 상자와 같은 차례의 원본 곡선 목록(없으면 None)."""
    key = (omr_path, sheet)
    if key not in _CURVE_CACHE:
        root = _sheet_root(omr_path, sheet)
        out = []
        for el in root.iter():
            b = el.find("bounds")
            if b is None or el.tag != "slur":
                continue
            c = el.find("curve")
            try:
                out.append(tuple(float(c.get(k)) for k in ("x1", "y1", "ctrlx1", "ctrly1", "ctrlx2", "ctrly2",
                                                           "x2", "y2")) if c is not None else None)
            except (TypeError, ValueError):
                out.append(None)
        _CURVE_CACHE[key] = out
    return _CURVE_CACHE[key]


def _omr_boxes(omr_path, sheet=1):
    """(인식된 기호 상자들, 이음줄 상자들) — Audiveris 이미지 좌표."""
    root = _sheet_root(omr_path, sheet)
    boxes, slurs, heads = [], [], []
    for el in root.iter():
        b = el.find("bounds")
        if b is None:
            continue
        try:
            x, y, w, h = (float(b.get(k)) for k in ("x", "y", "w", "h"))
        except (TypeError, ValueError):
            continue
        if el.tag == "slur":
            slurs.append((x, y, w, h))
        elif el.tag not in _SKIP:
            boxes.append((x, y, w, h))
        if el.tag == "head":
            heads.append((x, y, w, h, el.get("shape") or ""))
    return boxes, slurs, heads


def _gray(img_bgr, scale):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    return gray


def _head_ink(gray, box):
    """머리 상자 안 잉크: (안쪽 타원 비율, 테두리 띠 비율)."""
    x0, y0, x1, y1 = (int(v) for v in box)
    sub = (gray[max(0, y0):y1, max(0, x0):x1] < 140).astype(np.uint8)
    if sub.size == 0:
        return 0.0, 0.0
    H_, W_ = sub.shape
    yy, xx = np.mgrid[0:H_, 0:W_]
    r = ((xx - W_ / 2) / (W_ / 2)) ** 2 + ((yy - H_ / 2) / (H_ / 2)) ** 2
    inner = sub[r < 0.35]
    ring = sub[(r >= 0.45) & (r < 1.0)]
    return (float(inner.mean()) if inner.size else 0.0), (float(ring.mean()) if ring.size else 0.0)


def _is_head_ink(gray, box, il):
    """[v2.01] 진짜 머리 모양인가 — 검정 머리는 안이 꽉 차고(0.9 이상), 빈 머리는 안이 비고(0.5 미만) 테두리가
    또렷하며(0.6 이상) 키가 보통 머리만 하다(0.85칸 이상). 이음줄 가운데 굵은 부분·악센트 조각은 안쪽이
    0.34~0.74라 걸러진다("교회" 28·32·37·54마디). 반환: (머리인가, 빈 머리인가)."""
    inner, ring = _head_ink(gray, box)
    if inner >= 0.9:
        return True, False
    if inner < 0.5 and ring >= 0.6 and (box[3] - box[1]) >= 0.85 * il:
        return True, True
    return False, False


def find_unrecognized(img_bgr, omr_path, staff_groups, score, scale=1.0):
    """반환: [{"kind": "head"|"dot"|"other", "box": (x0,y0,x1,y1), "line", "staff_no", "il",
    "hollow": bool}] — Audiveris 이미지 좌표."""
    if img_bgr is None or not staff_groups or not omr_path:
        return []
    gray = _gray(img_bgr, scale)
    H, W = gray.shape[:2]
    boxes, slurs, omr_heads = _omr_boxes(omr_path, getattr(score, "_omr_sheet", 1))
    ink = (gray < 140).astype(np.uint8)
    ils = [(g[4] - g[0]) / 4.0 for gs in staff_groups.values() for g in gs]
    il0 = float(np.median(ils)) if ils else 20.0
    cover = np.zeros_like(ink)
    pad = max(2, int(il0 * 0.15))
    for x, y, w, h in boxes:
        cover[max(0, int(y) - pad):int(y + h) + pad, max(0, int(x) - pad):int(x + w) + pad] = 1
    resid = ink & (1 - cover)
    k = max(3, int(il0 * 2))
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    resid &= (1 - lines)
    # 오선·덧줄을 지우면 그 줄 위에 걸친 머리가 위아래로 갈라진다 → 세로로 얇게 다시 잇는다
    resid = cv2.morphologyEx(resid, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(il0 * 0.35)))))
    resid = cv2.morphologyEx(resid, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(il0 * 0.3)),) * 2))
    x_lo = {}
    for m in score.measures:
        for e in m.elements:
            if e.src_bbox:
                x_lo[m.system] = min(x_lo.get(m.system, 1e9), e.src_bbox[0])
    out = []
    # (v2.32 검토: 인식된 머리 둘레를 훑어 빠진 빈 머리를 찾는 방법 — 스캔 악보에서는 화음 머리 사이 공간·
    #  빔 사이 등을 빈 머리로 잘못 봐 오탐이 많아 쓰지 않는다. scan_missing_chord_heads는 참고로 남김)
    # [v1.40] Audiveris가 머리로 '인식은 했지만' 음악(MusicXML)에는 넣지 않은 머리("교회" 9마디 오른손
    # F♯4: NOTEHEAD_BLACK 0.789 + 기둥까지 있는데 마디에서 빠짐). 편집 모델의 어느 음과도 위치가 맞지 않는
    # 머리 inter를 '빠진 머리'로 더한다(잉크 덩어리보다 확실한 근거).
    model_boxes = [e.src_bbox for m in score.measures for e in m.elements if e.src_bbox]
    phantoms = getattr(score, "_phantom_heads", None) or []
    for hx, hy, hw, hh, shape in omr_heads:
        if any(abs(bx - hx) <= 4 and abs(by - hy) <= 4 for bx, by, _bw, _bh in model_boxes):
            continue
        if any(abs(px - hx) <= 4 and abs(py - hy) <= 4 for px, py in phantoms):
            continue                                   # [v2.01] 유령 머리로 지운 것
        cy = hy + hh / 2.0
        for line, groups in staff_groups.items():
            for i, ys in enumerate(groups):
                il = (ys[4] - ys[0]) / 4.0
                if ys[0] - 2.5 * il <= cy <= ys[4] + 2.5 * il and hx >= x_lo.get(line, 1e9) - il0 * 1.5:
                    box_ = (int(hx), int(hy), int(hx + hw), int(hy + hh))
                    ok_, _hol = _is_head_ink(gray, box_, il)
                    if not ok_:
                        continue                       # 머리 모양 잉크가 아니다
                    out.append({"kind": "head", "box": box_,
                                "line": line, "staff_no": i + 1, "il": il, "from_omr": True,
                                "hollow": shape in ("NOTEHEAD_VOID", "WHOLE_NOTE", "BREVE",
                                                    "NOTEHEAD_VOID_SMALL")})
    for line, groups in staff_groups.items():
        if line not in x_lo:
            continue
        xl = int(max(0, x_lo[line] - il0 * 1.5))      # 줄 머리(괄호·음자리표·조표)는 제외
        for i, ys in enumerate(groups):
            il = (ys[4] - ys[0]) / 4.0
            if il <= 2:
                continue
            y0, y1 = max(0, int(ys[0] - 2.5 * il)), min(H, int(ys[4] + 2.5 * il))
            band = resid[y0:y1, xl:W]
            n, _lab, st, _c = cv2.connectedComponentsWithStats(band, 8)
            for j in range(1, n):
                bx, by, bw, bh, area = st[j]
                if area < 0.08 * il * il:
                    continue
                gx0, gy0 = bx + xl, by + y0
                gx1, gy1 = gx0 + bw, gy0 + bh
                cy = (gy0 + gy1) / 2.0
                kind = None
                if bw < 0.75 * il and bh < 0.75 * il:
                    # 점: 오선 안팎 1칸 안(아래 가사 줄의 '–'는 제외), 납작하지 않음, 이음줄 끝 조각 제외
                    if ys[0] - il <= cy <= ys[4] + il and bh >= 0.25 * il and bw <= 2 * bh and \
                            not any(sx <= (gx0 + gx1) / 2 <= sx + sw and sy <= cy <= sy + sh
                                    for sx, sy, sw, sh in slurs):
                        kind = "dot"
                elif 0.7 * il <= bw <= 2.9 * il and 0.6 * il <= bh <= 1.6 * il:
                    kind = "head"                         # (2.0칸 넘으면 덧줄이 양옆으로 붙은 머리)
                    # 머리는 꽉 찬 타원(상자의 55% 이상) — 기둥을 가로지르는 이음줄 조각(비스듬·얇음) 제외
                    if area < 0.55 * bw * bh and bw <= 2.0 * il:
                        kind = "other"
                    # 세로 획 '둘'이 덩어리를 위아래로 꿰뚫으면 ♯·♮ 같은 임시표 — 머리가 아니다. 획 하나는
                    # 머리 위아래로 기둥이 이어진 경우(2도로 붙은 두 음의 위·아래 기둥, "교회" 5마디)라 남긴다.
                    # (기둥 한쪽만이면 꿰뚫는 비율 ~0.6, 꿰뚫는 획은 ~0.8 이상)
                    else:
                        strong = [c for c in range(int(gx0), int(gx1))
                                  if (gray[max(0, int(gy0 - il)):int(gy1 + il), c] < 140).mean() > 0.72]
                        runs = sum(1 for i_, c in enumerate(strong) if i_ == 0 or c - strong[i_ - 1] > 2)
                        if runs >= 2:
                            kind = "other"
                else:
                    kind = "other"
                if kind is None:
                    continue
                if kind == "other" and any(sx - 0.3 * il <= (gx0 + gx1) / 2 <= sx + sw + 0.3 * il and
                                           sy - 0.3 * il <= cy <= sy + sh + 0.3 * il
                                           for sx, sy, sw, sh in slurs):
                    continue        # [v1.42] 인식된 이음줄의 굵은 부분 — 인식된 것이다(13마디 첫 이음줄 오표시)
                hollow = False
                if kind == "head":
                    ok_, hollow = _is_head_ink(gray, (gx0, gy0, gx1, gy1), il)
                    if not ok_:
                        kind = "other"                 # [v2.01] 머리 모양 잉크가 아니다(이음줄·악센트 조각)
                out.append({"kind": kind, "box": (int(gx0), int(gy0), int(gx1), int(gy1)),
                            "line": line, "staff_no": i + 1, "il": il, "hollow": bool(hollow)})
    return out


def _ring_all_around(gray, box, il, staff_ys):
    """빈 머리 엄격 판정: 오선 줄 행은 빼고, 테두리를 8방향으로 나눠 6방향 이상(왼쪽·오른쪽 포함)에 잉크가
    있고 안쪽이 비어야 한다(오선 사이 공간·기둥 한 줄로는 통과 못 함)."""
    x0, y0, x1, y1 = (int(v) for v in box)
    sub = (gray[y0:y1, x0:x1] < 140)
    if sub.size == 0:
        return False
    H_, W_ = sub.shape
    yy, xx = np.mgrid[0:H_, 0:W_]
    valid = np.ones_like(sub, bool)
    for ly in staff_ys:
        r = int(round(ly)) - y0
        valid[max(0, r - 2):max(0, r + 3), :] = False
    cxr, cyr = (W_ - 1) / 2.0, (H_ - 1) / 2.0
    r = ((xx - cxr) / (W_ / 2.0)) ** 2 + ((yy - cyr) / (H_ / 2.0)) ** 2
    inner = sub[(r < 0.3) & valid]
    if inner.size and inner.mean() > 0.35:
        return False
    ring = (r >= 0.45) & (r < 1.0) & valid
    ang = np.arctan2(yy - cyr, xx - cxr)
    sec = ((ang + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    known, inked = 0, set()
    for k in range(8):
        sel = ring & (sec == k)
        if sel.sum() < 4:
            continue
        known += 1
        if sub[sel].mean() > 0.35:
            inked.add(k)
    left = {3, 4} & inked
    right = {0, 7} & inked
    return known >= 5 and len(inked) >= 6 and bool(left) and bool(right)


def scan_missing_chord_heads(score, gray, staff_groups):
    """[v2.32 화음 짝 머리 찾기 — 실사용] 인식된 머리 바로 위·아래(같은 열, 2도면 옆으로 비킨 자리)에 원본의
    머리 모양 잉크가 있는데 인식된 음이 없으면 빠진 머리로 본다. ("교회" 9마디 오른손 G♯4 2분음표: F♯4 머리와
    2도로 겹쳐, 미인식 잉크 찾기에서는 F♯4의 인식 상자 안이라 가려졌다. Audiveris도 머리·위 기둥 모두 못 읽음.)
    판정은 _is_head_ink(빈 머리: 안 비고 테두리 또렷, 검정: 안 꽉 참)."""
    H, W = gray.shape[:2]
    out = []
    for m in score.measures:
        grp = staff_groups.get(m.system) or []
        heads = [e for e in m.elements if e.kind == "note" and e.src_bbox and not e.is_grace]
        for e in heads:
            if e.staff_no - 1 >= len(grp):
                continue
            ys = grp[e.staff_no - 1]
            il = (ys[4] - ys[0]) / 4.0
            x, y, w, h = e.src_bbox
            cx, cy = x + w / 2.0, y + h / 2.0
            for k in (-4, -3, -2, -1, 1, 2, 3, 4):
                ny = cy + k * il / 2.0
                dxs = (-w, w) if abs(k) == 1 else (0.0,)
                for dx in dxs:
                    nx = cx + dx
                    box = (int(nx - w / 2), int(ny - h / 2), int(nx + w / 2), int(ny + h / 2))
                    if box[0] < 0 or box[1] < 0 or box[2] >= W or box[3] >= H:
                        continue
                    if any(abs((o.src_bbox[0] + o.src_bbox[2] / 2) - nx) < 0.6 * w and
                           abs((o.src_bbox[1] + o.src_bbox[3] / 2) - ny) < 0.35 * il
                           for o in m.elements if o.src_bbox):
                        continue                      # 이미 인식된 음
                    if any(abs((b["box"][0] + b["box"][2]) / 2 - nx) < 0.6 * w and
                           abs((b["box"][1] + b["box"][3]) / 2 - ny) < 0.35 * il for b in out):
                        continue
                    if not _ring_all_around(gray, box, il, ys):
                        continue                      # 빈 머리만(검정 머리는 미인식 잉크 찾기가 이미 잘 찾는다)
                    out.append({"kind": "head", "box": box, "line": m.system, "staff_no": e.staff_no,
                                "il": il, "hollow": True, "from_scan": True})
    return out


def attach_to_score(score, blobs):
    """마디 검사용: 각 마디에 {보표: 빠진 머리 수}를 m.unexplained_heads 로 둔다."""
    res = {}
    by_line = {}
    for m in score.measures:
        if m.elements:
            by_line.setdefault(m.system, []).append(
                (min(e.src_bbox[0] if e.src_bbox else e.src_x for e in m.elements), m))
    for line, lst in by_line.items():
        lst.sort(key=lambda t: t[0])
        for i, (x0, m) in enumerate(lst):
            x1 = lst[i + 1][0] if i + 1 < len(lst) else 10 ** 9
            for b in blobs:
                if b["line"] != line:
                    continue
                bx = (b["box"][0] + b["box"][2]) / 2
                if x0 - b["il"] <= bx < x1 - b["il"]:
                    b["measure"] = m.number
                    if b["kind"] == "head":
                        key = (m.number, b["staff_no"])
                        res[key] = res.get(key, 0) + 1
    for m in score.measures:
        m.unexplained_heads = {sn: c for (num, sn), c in res.items() if num == m.number}
    score.unexplained_heads = res
    return res


# --------------------------------------------------------------------- 제안
_STEP = "CDEFGAB"
_DIA_TOP = {"G": 38, "F": 26, "C": 32}


def _stem_dir(gray, box, il):
    """원본에서 머리 옆 기둥 방향: 'up'(오른쪽 위로) | 'down'(왼쪽 아래로) | None."""
    x0, y0, x1, y1 = box
    H, W = gray.shape[:2]
    L = int(il * 2.2)

    def run(col, ya, yb):
        if col < 0 or col >= W:
            return 0.0
        seg = gray[max(0, ya):min(H, yb), max(0, col - 1):col + 2] < 140
        return float(seg.any(axis=1).mean()) if seg.size else 0.0
    up = max(run(c, y0 - L, y0 + 2) for c in range(x1 - 5, x1 + 3))
    dn = max(run(c, y1 - 2, y1 + L) for c in range(x0 - 3, x0 + 6))
    if max(up, dn) < 0.85:
        return None
    return "up" if up >= dn else "down"


_NAMES = {1: "온", 2: "2분", 4: "4분", 8: "8분", 16: "16분", 32: "32분"}


def _fit_duration(q, hollow, stem):
    """빈 박 q(4분음표 단위)에 맞는 음가 (base, dotted). 빈 머리면 2분·온, 검정이면 4분 이하."""
    cands = [(1, False, 4.0), (2, True, 3.0), (2, False, 2.0)] if hollow else \
        [(4, True, 1.5), (4, False, 1.0), (8, True, 0.75), (8, False, 0.5), (16, False, 0.25)]
    if hollow and stem is not None:
        cands = cands[1:]                                  # 기둥이 있으면 온음표가 아니다
    for base, dot, ql in cands:
        if ql <= q + 1e-6:
            return base, dot
    return cands[-1][0], cands[-1][1]


class _OffsetGray:
    """[v2.40] 페이지 그림(페이지 좌표)을 곡 좌표(y + dy)로 읽게 하는 얇은 감싸개(복사 없음)."""

    def __init__(self, g, dy):
        self.g, self.dy = g, int(round(dy))
        self.shape = (g.shape[0] + self.dy, g.shape[1])

    def __getitem__(self, key):
        ys, xs = key
        if isinstance(ys, slice):
            a = None if ys.start is None else max(0, ys.start - self.dy)
            b = None if ys.stop is None else max(0, ys.stop - self.dy)
            ys = slice(a, b, ys.step)
        else:
            ys = ys - self.dy
        return self.g[ys, xs]


def read_head_at(gray, x, y, staff_ys, il):
    """[v2.40 원본에서 기호 넣기] 원본의 (x, y) 근처 음표 머리를 읽는다(페이지 좌표).
    세로: 오선 칸(줄·사이)에 맞춰 붙이고, 가로: 그 높이 띠에서 잉크가 있는 열의 덩어리(클릭에 가장 가까운).
    빈 머리/검정 머리는 안쪽 채움으로. 반환 ((x0, y0, x1, y1), hollow) 또는 None."""
    H, W = gray.shape[:2]
    s = int(round((y - staff_ys[0]) / (il / 2.0)))
    cy = staff_ys[0] + s * il / 2.0
    hy = 0.42 * il
    ya, yb = int(cy - hy), int(cy + hy) + 1
    xa, xb = int(max(0, x - 1.4 * il)), int(min(W, x + 1.4 * il))
    if ya < 0 or yb > H or xb - xa < 4:
        return None
    band = gray[ya:yb, xa:xb] < 140
    rows = band.mean(axis=1) < 0.85                       # 오선·덧줄처럼 띠를 가로지르는 줄은 뺀다
    band = band[rows] if rows.any() else band
    colf = band.mean(axis=0)
    inked = colf > 0.25
    if not inked.any():
        return None
    # 클릭에 가장 가까운 잉크 열에서 좌우로 이어진 덩어리
    cols = np.nonzero(inked)[0]
    c0 = int(cols[np.argmin(np.abs(cols - (x - xa)))])
    l = c0
    while l - 1 >= 0 and inked[l - 1]:
        l -= 1
    r = c0
    while r + 1 < len(inked) and inked[r + 1]:
        r += 1
    w = r - l + 1
    if w < 0.6 * il:
        return None
    if w > 1.8 * il:                                      # 이웃 머리와 붙어 있으면 클릭 쪽 한 머리 폭만
        hw = 0.65 * il
        mid = min(max(x - xa, l + hw), r - hw)
        l, r = int(mid - hw), int(mid + hw)
    box = (xa + l, int(cy - 0.5 * il), xa + r + 1, int(cy + 0.5 * il))
    ix0, ix1 = int(box[0] + 0.3 * (box[2] - box[0])), int(box[2] - 0.3 * (box[2] - box[0]))
    iy0, iy1 = int(cy - 0.18 * il), int(cy + 0.18 * il) + 1
    inner = gray[iy0:iy1, ix0:ix1] < 140
    on_line = any(abs(cy - ly) < 0.15 * il for ly in staff_ys)
    if on_line and inner.shape[0] > 2:                    # 줄 위 머리는 가운데 줄 행을 빼고 본다
        keep = [i for i in range(inner.shape[0]) if abs(iy0 + i - cy) > 0.1 * il]
        inner = inner[keep] if keep else inner
    hollow = bool(inner.size) and inner.mean() < 0.5
    return box, hollow


def make_suggestions(score, blobs, img_bgr, staff_groups, scale=1.0, y_off=0.0):
    """빠진 머리마다 넣을 음표 하나를 제안한다. 반환: [{"measure": 번호, "element": Element,
    "chord_with": Element|None, "blob": blob, "text": 설명}]"""
    from score.model import Element, _alter_map
    from score.musicxml import measure_anchors, _slots, _place_voices, DIVISIONS
    from score import editing as ed
    gray = _gray(img_bgr, scale)
    if y_off:
        gray = _OffsetGray(gray, y_off)                   # [v2.40] 곡 좌표로 읽기(여러 페이지)
    mby = {m.number: m for m in score.measures}
    staff_clefs = getattr(score, "staff_clefs", None) or {}
    out = []
    for b in blobs:
        if b["kind"] != "head" or b.get("measure") not in mby:
            continue
        m = mby[b["measure"]]
        sn, il = b["staff_no"], b["il"]
        x0, y0, x1, y1 = b["box"]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        grp = (staff_groups or {}).get(m.system)
        if not grp or sn - 1 >= len(grp):
            continue
        # 음자리표: 줄 머리 → 이 줄에서 cx 앞에 있는 clef_change
        cl = staff_clefs.get(m.system) or []
        clef = cl[sn - 1] if sn - 1 < len(cl) else ("F" if sn >= 2 else "G")
        for mm in sorted(score.measures, key=lambda q: q.number):
            if mm.system != m.system or mm.number > m.number:
                continue
            for e in sorted(mm.elements, key=lambda e: e.src_x):
                if e.staff_no == sn and e.clef_change and (mm.number < m.number or e.src_x <= cx):
                    clef = e.clef_change
        top = grp[sn - 1][0]
        sp = (grp[sn - 1][4] - grp[sn - 1][0]) / 4.0
        steps = int(round((cy - top) / (sp / 2.0)))
        dia = _DIA_TOP.get(clef, 38) - steps
        step, octave = _STEP[dia % 7], dia // 7
        alter = _alter_map(score.key_at(m.number)).get(step, 0)
        cap = int(round(ed._capacity_of(score, m) * DIVISIONS))
        anchors = measure_anchors(m.elements, cap)
        near = [a for a in anchors if abs(a["x"] - x0) <= 1.3 * max(a["w"], x1 - x0)]
        if not near:
            continue
        onset = min(near, key=lambda a: abs(a["x"] - x0))["on"]
        # 1도·2도로 옆에 비켜 찍힌 머리: 바로 왼쪽(2.6칸 안)에 같은·이웃 높이 음이 있으면 그 음의 박
        # ("교회" 9마디 왼손: 8분음표 E3 오른쪽에 붙은 2분음표 E3)
        side = [e for e in m.elements if e.staff_no == sn and e.kind == "note" and e.src_bbox
                and 0 < x0 - e.src_bbox[0] <= 2.6 * il
                and abs((e.src_bbox[1] + e.src_bbox[3] / 2.0) - cy) <= 0.6 * il]
        if side:
            nb = max(side, key=lambda e: e.src_bbox[0])
            na = [a for a in anchors if abs(a["x"] - nb.src_x) <= 1.3 * max(a["w"], 20)]
            if na:
                onset = min(na, key=lambda a: abs(a["x"] - nb.src_x))["on"]
        voices = {}
        for e in m.elements:
            if e.staff_no == sn and not getattr(e, "suggested", False):
                voices.setdefault(e.voice_no, []).append(e)
        vv = {v: _slots(es) for v, es in voices.items()}
        _place_voices(vv, cap, anchors)
        stem = _stem_dir(gray, b["box"], il)
        # 이 보표의 '다음 박'(성부 번호와 무관 — Audiveris가 성부를 섞어 매겨도): 각 음의 x를 기준
        # 박 위치에 맞춰 본 시작점 중 onset 다음 것
        on_by_x = []
        for e in m.elements:
            if e.staff_no != sn or e.is_grace or getattr(e, "suggested", False):
                continue
            nn = [a for a in anchors if abs(a["x"] - e.src_x) <= 1.3 * max(a["w"], 20)]
            if nn:
                on_by_x.append(min(nn, key=lambda a: abs(a["x"] - e.src_x))["on"])
        next_on = min([o for o in on_by_x if o > onset] + [cap])

        def _mean(v):
            ds = [e.octave * 7 + _STEP.index(e.step) for s in vv[v] for e in s["els"]
                  if e.kind == "note" and e.step]
            return sum(ds) / len(ds) if ds else -99
        order = sorted(vv, key=lambda v: -_mean(v))      # 높은 성부부터
        pref = list(reversed(order)) if stem == "down" else order
        if stem in ("up", "down"):
            # [v2.40] 원본 기둥 방향이 같은 성부가 먼저(원본이 정한다) — 평균 음높이는 그다음.
            # ("교회" 9마디 오른손: 위 기둥 G♯4는 위 기둥 성부(B4…)로, 아래 기둥 성부(F♯4 점4분)가 아니라)
            def _match(v):
                ds = [getattr(e, "stem_dir_img", None) for s_ in vv[v] for e in s_["els"] if e.kind == "note"]
                ds = [d for d in ds if d]
                return 0 if ds and sum(1 for d in ds if d == stem) * 2 >= len(ds) else 1
            pref = sorted(pref, key=lambda v: (_match(v), pref.index(v)))
        choice = None
        for v in pref:                                   # ① 그 박이 비어 있는 성부 → 새 음
            sl = [s for s in vv[v] if not s["grace"]]
            if any(s["on"] <= onset < s["on"] + s["dur"] for s in sl):
                continue
            nxt = min([s["on"] for s in sl if s["on"] > onset] + [cap, next_on])
            if nxt - onset > 0:
                choice = ("new", v, nxt - onset)
                break
        if choice is None and not vv:
            choice = ("new", 1 if sn == 1 else 5, next_on - onset)
        if choice is None:                               # ② 같은 박에 음이 있으면 화음 구성음으로
            for v in pref:
                hit = [s for s in vv[v] if s["on"] == onset and not s["grace"]
                       and s["els"][0].kind == "note"]
                if hit:
                    rep_dir = getattr(hit[0]["els"][0], "stem_dir_img", None)
                    if stem and rep_dir and stem != rep_dir:
                        # [v2.01] 기둥이 그 음과 반대 방향이면 화음이 아니라 다른 성부다("교회" 18마디 오른손:
                        # 위로 기둥인 G♯4 4분음표 + 아래로 기둥인 F♯4·E4 8분음표)
                        newv = max(vv) + 1
                        q_ = 2 if b["hollow"] else 1
                        dur_ = min(q_ * DIVISIONS, cap - onset)
                        choice = ("new", newv, dur_)
                    else:
                        choice = ("chord", v, hit[0])
                    break
        if choice is None:
            continue
        if choice[0] == "new":
            _k, v, gap = choice
            base, dotted = _fit_duration(gap / DIVISIONS, b["hollow"], stem)
            el = Element(kind="note", duration_base=base, dotted=dotted, step=step, octave=octave,
                         alter=alter, src_bbox=(x0, y0, x1 - x0, y1 - y0), src_x=x0,
                         staff_no=sn, voice_no=v, synthetic=True)
            chord_with = None
        else:
            _k, v, s = choice
            rep = s["els"][0]
            el = Element(kind="note", duration_base=rep.duration_base, dotted=rep.dotted,
                         double_dot=rep.double_dot, step=step, octave=octave, alter=alter,
                         src_bbox=(x0, y0, x1 - x0, y1 - y0), src_x=rep.src_x, staff_no=sn,
                         voice_no=v, beam_group=rep.beam_group, synthetic=True,
                         in_triplet=rep.in_triplet, tup_actual=rep.tup_actual, tup_normal=rep.tup_normal)
            chord_with = rep
        el.suggested = True
        if stem in ("up", "down"):
            el.stem_dir_img = stem                       # [v2.40] 원본에서 읽은 기둥 방향 그대로
        acc = {1: "♯", -1: "♭"}.get(alter, "")
        text = (f"{m.number}마디 {onset / DIVISIONS + 1:g}박 · {step}{acc}{octave} "
                f"{'점' if el.dotted else ''}{_NAMES.get(el.duration_base, '')}음표"
                + (" (화음에 추가)" if chord_with is not None else ""))
        out.append({"measure": m.number, "element": el, "chord_with": chord_with, "blob": b,
                    "text": text})
    return out


def accept_suggestion(score, sug):
    """제안 하나를 악보에 넣는다. 넣은 Element를 돌려준다."""
    m = next(mm for mm in score.measures if mm.number == sug["measure"])
    el = sug["element"]
    el.suggested = False
    if sug.get("kind") == "grace":                  # [v1.43] 꾸밈음: 본음 앞에 넣고(박자 없음) 이음줄
        m.elements.append(el)
        main = sug.get("main")
        if sug.get("slur") and main is not None and main in m.elements:
            from score.model import Arc
            arc_ = Arc(start=el, stop=main, kind="slur", above=True)
            if sug.get("slur_curve"):
                arc_.src_curve = tuple(sug["slur_curve"])         # [v2.30] 원본 곡선
            score.arcs.append(arc_)
            el.slur_start, main.slur_stop = True, True
        sugs = getattr(score, "suggestions", None) or []
        if sug in sugs:
            sugs.remove(sug)
        return el
    cw = sug.get("chord_with")
    if cw is not None and cw in m.elements:
        if cw.chord_group == -1:
            cw.chord_group = max((e.chord_group for e in m.elements), default=-1) + 1
        el.chord_group = cw.chord_group
    m.elements.append(el)
    uh = getattr(m, "unexplained_heads", None) or {}
    if uh.get(el.staff_no):                           # 이 보표의 '빠진 머리'를 하나 지운다
        uh[el.staff_no] -= 1
        if uh[el.staff_no] <= 0:
            uh.pop(el.staff_no)
    from score import editing as ed
    ed.repair_voices(score, m)                        # 넣은 뒤 성부가 섞여 있으면 다시 나눠 본다
    sugs = getattr(score, "suggestions", None) or []
    if sug in sugs:
        sugs.remove(sug)
    return el


def dismiss_suggestion(score, sug):
    sugs = getattr(score, "suggestions", None) or []
    if sug in sugs:
        sugs.remove(sug)


def auto_apply_measure(score, number):
    """[v2.11] 편집으로 한 마디가 바뀐 뒤, 그 마디에 남은 제안 음표를 다시 넣어 본다(정상이 되면 둔다).
    예: 붙임줄을 잘못 읽은 쉼표를 지우면 그 때문에 막혀 있던 빠진 음이 들어간다."""
    keep = [q for q in (getattr(score, "suggestions", None) or []) if q["measure"] != number]
    mine = [q for q in (getattr(score, "suggestions", None) or []) if q["measure"] == number]
    if not mine:
        return []
    score.suggestions = mine
    done = auto_apply(score)
    score.suggestions = keep + [q for q in (score.suggestions or []) if q not in keep]
    return done


def auto_apply(score):
    """[v1.41 자동 보완 — 실사용 요청] 마디마다 그 마디의 제안을 '함께' 넣어 보고, 마디 검사가
    정상이 되면 그대로 두고(자동으로 넣음), 아니면 되돌려 제안으로 남긴다. 넣은 제안 목록을 돌려준다."""
    from score import editing as ed
    sugs = [q for q in (getattr(score, "suggestions", None) or []) if q.get("kind") != "grace"]
    by_m = {}
    for q in sugs:
        by_m.setdefault(q["measure"], []).append(q)
    applied = []
    good = ("ok", "anacrusis", "time_change")
    for num, qs in by_m.items():
        m = next((mm for mm in score.measures if mm.number == num), None)
        if m is None:
            continue
        # [v2.01] 이미 정상인 마디(화음 구성음·다른 성부 음이 빠진 경우)도 넣어 보고, 넣은 뒤에도 정상이면 둔다
        saved = {id(e): (e.voice_no, e.chord_group) for e in m.elements}
        saved_uh = dict(getattr(m, "unexplained_heads", None) or {})
        added = [accept_suggestion(score, q) for q in qs]
        if ed.validate(score).get(num, ("",))[0] in good:
            for el in added:
                el.auto_added = True
            applied.extend(qs)
            continue
        for el in added:                         # 되돌리기
            if el in m.elements:
                m.elements.remove(el)
            el.suggested = True
        score.suggestions = [q for q in (score.suggestions or []) if q not in qs]
        for e in m.elements:
            if id(e) in saved:
                e.voice_no, e.chord_group = saved[id(e)]
        m.unexplained_heads = saved_uh
        score.suggestions.extend(qs)
    return applied


def recover_slurs(score, omr_path, staff_groups, min_grade=0.5):
    """[v1.42 빠진 이음줄 되살리기 — 실사용] Audiveris가 이음줄을 '인식은 했지만' 음표에 잇지 않아 음악에서
    빠진 경우("교회" 14마디 왼손: 0.585·0.374 두 조각). 편집 악보의 어느 이음줄·붙임줄에도 쓰이지 않은
    slur inter를 이웃 조각끼리 합치고, 가장 높은 신뢰도가 min_grade 이상이며 양 끝이 같은 보표의 음표
    머리에 닿으면 그 두 음 사이에 이음줄을 넣는다. 넣은 개수를 돌려준다."""
    from score.model import Arc
    if not omr_path or not staff_groups:
        return 0
    root = _sheet_root(omr_path, getattr(score, "_omr_sheet", 1))
    pieces = []
    for sl in root.iter("slur"):
        b = sl.find("bounds")
        if b is None or sl.get("grade") is None:
            continue
        x, y, w, h = (float(b.get(k)) for k in ("x", "y", "w", "h"))
        cv_ = sl.find("curve")
        curve = None
        if cv_ is not None:
            try:
                curve = tuple(float(cv_.get(k)) for k in ("x1", "y1", "ctrlx1", "ctrly1", "ctrlx2", "ctrly2", "x2", "y2"))
            except (TypeError, ValueError):
                curve = None
        pieces.append({"x0": x, "x1": x + w, "y": y + h / 2.0, "h": h, "g": float(sl.get("grade")),
                       "above": (sl.get("shape") or "").endswith("ABOVE"), "curve": curve, "n": 1})
    # 줄·보표: 조각 가운데 높이가 가장 가까운 보표
    def staff_of(p):
        best = None
        for line, grp in staff_groups.items():
            for i, ys in enumerate(grp):
                il = (ys[4] - ys[0]) / 4.0
                d = 0 if ys[0] <= p["y"] <= ys[4] else min(abs(p["y"] - ys[0]), abs(p["y"] - ys[4]))
                if d <= 4 * il and (best is None or d < best[0]):
                    best = (d, line, i + 1, il)
        return best
    for p in pieces:
        st = staff_of(p)
        p["line"], p["sn"], p["il"] = (st[1], st[2], st[3]) if st else (None, None, None)
        if st and p["h"] > 3.0 * st[3]:
            p["line"] = None        # 3칸보다 높은 곡선은 보표를 건너는 이음줄 — 한 보표에 잇지 않는다("교회" 3마디)
    notes = {}
    for m in score.measures:
        for e in m.elements:
            if e.kind == "note" and e.src_bbox and not e.is_grace:
                notes.setdefault((m.system, e.staff_no), []).append(e)

    def cx(e):
        return e.src_bbox[0] + e.src_bbox[2] / 2.0
    used = []
    for a in score.arcs:
        used.append((cx(a.start), cx(a.stop), a.start.staff_no))
    # 이미 쓰인 조각(편집 악보의 이음줄·붙임줄이 그 가로 범위를 잇고 있음)은 먼저 뺀다 — 합친 뒤에 보면
    # 이어진 붙임줄 둘(B3→B3→B3)이 긴 곡선 하나처럼 보여 다시 넣게 된다("교회" 9마디 왼손)
    def _used(p):
        il_ = p["il"] or 20.0
        return any(us == p["sn"] and u0 <= p["x0"] + 1.5 * il_ and u1 >= p["x1"] - 1.5 * il_
                   and u1 - u0 <= (p["x1"] - p["x0"]) + 4 * il_ for u0, u1, us in used)
    # 이웃 조각 합치기(같은 보표, 가로 틈 1.5칸 이하)
    pieces = sorted((p for p in pieces if p["line"] is not None and not _used(p)),
                    key=lambda p: (p["line"], p["sn"], p["x0"]))
    groups = []
    for p in pieces:
        g = groups[-1] if groups else None
        if g and g["line"] == p["line"] and g["sn"] == p["sn"] and p["x0"] - g["x1"] <= 1.5 * p["il"] \
                and g["above"] == p["above"]:
            g["x1"] = max(g["x1"], p["x1"])
            g["g"] = max(g["g"], p["g"])
            g["n"] = g.get("n", 1) + 1
        else:
            groups.append(dict(p))
    added = 0
    for g in groups:
        if g["g"] < min_grade:
            continue
        il = g["il"]
        cands = notes.get((g["line"], g["sn"]), [])
        left = [e for e in cands if g["x0"] - 1.5 * il <= cx(e) <= g["x0"] + 1.2 * il]
        right = [e for e in cands if g["x1"] - 1.2 * il <= cx(e) <= g["x1"] + 1.5 * il]
        if not left or not right:
            continue
        a = min(left, key=lambda e: abs(cx(e) - g["x0"]))
        b = min(right, key=lambda e: abs(cx(e) - g["x1"]))
        if a is b or cx(b) <= cx(a):
            continue
        # 이미 이 구간을 잇는 이음줄·붙임줄이 있으면 '쓰인' 조각
        if any(us == g["sn"] and abs(u0 - cx(a)) <= 1.5 * il and abs(u1 - cx(b)) <= 1.5 * il
               for u0, u1, us in used):
            continue
        same = (a.step, a.octave) == (b.step, b.octave)
        kind = "tie" if same else "slur"
        arc_ = Arc(start=a, stop=b, kind=kind, above=g["above"])
        arc_.src_above = bool(g["above"])
        if g.get("n", 1) == 1 and g.get("curve"):
            arc_.src_curve = g["curve"]              # [v2.30] 한 조각이면 원본 곡선 그대로
        score.arcs.append(arc_)
        if kind == "tie":
            a.tie_start, b.tie_stop = True, True
        else:
            a.slur_start, b.slur_stop = True, True
        used.append((cx(a), cx(b), g["sn"]))
        added += 1
    return added


def assign_stem_dirs(score, img_bgr, staff_groups, scale=1.0):
    """[v1.43] 음표마다 원본 그림의 기둥 방향(e.stem_dir_img = 'up'|'down'|None)을 적어 둔다
    (마디 검사의 '같은 방향 기둥' 규칙에 쓴다)."""
    if img_bgr is None or not staff_groups:
        return
    gray = _gray(img_bgr, scale)
    for m in score.measures:
        grp = staff_groups.get(m.system) or []
        for e in m.elements:
            if e.kind != "note" or not e.src_bbox or e.duration_base == 1 or e.staff_no - 1 >= len(grp):
                continue
            ys = grp[e.staff_no - 1]
            il = (ys[4] - ys[0]) / 4.0
            x, y, w, h = e.src_bbox
            e.stem_dir_img = _stem_dir(gray, (int(x), int(y), int(x + w), int(y + h)), il)


def find_graces(score, img_bgr, staff_groups, omr_path=None, scale=1.0, _dbg=None):
    """[v1.43 앞꾸밈음 찾기] Audiveris는 꾸밈음(작은 음표)을 읽지 못한다("교회" 1·2·13·15마디). 음표마다
    바로 왼쪽(0.3~3.4칸)의 원본 그림에서, 오선·세로선을 지운 뒤 남는 '작은 검정 타원'(보통 머리보다 작음)
    중 인식된 음이 아니고 오른쪽 끝에서 기둥이 위로 올라가는 것을 꾸밈음으로 본다(♭은 기둥이 왼쪽이라
    제외). 반환: 제안 목록(kind="grace") — 클릭하면 본음 앞에 8분 꾸밈음(+이음줄)으로 들어간다."""
    from score.model import Element, _alter_map
    if img_bgr is None or not staff_groups:
        return []
    gray = _gray(img_bgr, scale)
    H, W = gray.shape[:2]
    slur_boxes, sym_boxes = [], []
    if omr_path:
        try:
            _b, slur_boxes, _h = _omr_boxes(omr_path, getattr(score, "_omr_sheet", 1))
            rt = _sheet_root(omr_path, getattr(score, "_omr_sheet", 1))
            for el in rt.iter():
                if el.tag in ("clef", "alter", "key-alter", "time-number", "time-whole", "barline",
                              "staff-barline", "brace", "bracket", "repeat-dot", "fermata", "dynamics"):
                    bb = el.find("bounds")
                    # 확실히 인식된 기호만(0.5 이상) — "교회" 13마디 꾸밈음은 Audiveris가 f(0.231)로 오인해 덮었다
                    if bb is not None and float(el.get("grade") or 0) >= 0.5:
                        sym_boxes.append(tuple(float(bb.get(k)) for k in ("x", "y", "w", "h")))
        except Exception:
            slur_boxes, sym_boxes = [], []
    model_boxes = {}
    for m in score.measures:
        for e in m.elements:
            if e.src_bbox:
                model_boxes.setdefault(m.system, []).append(e.src_bbox)
    staff_clefs = getattr(score, "staff_clefs", None) or {}
    out, seen = [], set()
    for m in score.measures:
        grp = staff_groups.get(m.system) or []
        has_grace = {id(e) for e in m.elements if e.is_grace}
        for main in sorted(m.elements, key=lambda e: e.src_x):
            if main.kind != "note" or main.is_grace or not main.src_bbox or main.staff_no - 1 >= len(grp):
                continue
            if has_grace:
                continue                             # 이미 꾸밈음이 있는 마디는 건드리지 않는다
            ys = grp[main.staff_no - 1]
            il = (ys[4] - ys[0]) / 4.0
            mx, my, mw, mh = main.src_bbox
            x0, x1 = int(max(0, mx - 3.4 * il)), int(max(0, mx - 0.3 * il))
            y0, y1 = int(max(0, my + mh / 2 - 3 * il)), int(min(H, my + mh / 2 + 3 * il))
            if x1 - x0 < il:
                continue
            reg = (gray[y0:y1, x0:x1] < 140).astype(np.uint8)
            k = max(3, int(il * 1.6))
            hl = cv2.morphologyEx(reg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
            vl = cv2.morphologyEx(reg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
            rest = reg & (1 - hl) & (1 - vl)
            rest = cv2.morphologyEx(rest, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(il * 0.3)))))
            rest = cv2.morphologyEx(rest, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(il * 0.3)),) * 2))
            n, _lab, st, _c = cv2.connectedComponentsWithStats(rest, 8)
            best = None
            for j in range(1, n):
                bx, by, bw, bh, area = st[j]
                # 작은 머리: 폭 0.55~1.1칸, 높이 0.4~0.85칸, 꽉 찬 타원
                if not (0.62 * il <= bw <= 1.1 * il and 0.5 * il <= bh <= 0.85 * il and area >= 0.5 * bw * bh):
                    _dbg.append((m.number, cur_reason := "size", int(x0 + bx), int(y0 + by))) if _dbg is not None else None
                    continue
                gx0, gy0 = x0 + bx, y0 + by
                gx1, gy1 = gx0 + bw, gy0 + bh
                cxg, cyg = (gx0 + gx1) / 2.0, (gy0 + gy1) / 2.0
                if any(kx - 0.2 * il <= cxg <= kx + kw + 0.2 * il and ky - 0.2 * il <= cyg <= ky + kh + 0.2 * il
                       for kx, ky, kw, kh in model_boxes.get(m.system, [])):
                    continue                         # 인식된 음표·쉼표
                # 머리가 main과 같은 보표에 있어야 한다: 가장 가까운 보표가 main의 보표(옆 보표 머리 제외)
                def _dist(yy):
                    return 0 if yy[0] <= cyg <= yy[4] else min(abs(cyg - yy[0]), abs(cyg - yy[4]))
                if min(range(len(grp)), key=lambda q: _dist(grp[q])) != main.staff_no - 1 or _dist(ys) > 2.0 * il:
                    _dbg.append((m.number, cur_reason := "staff", int(x0 + bx), int(y0 + by))) if _dbg is not None else None
                    continue

                def colr(c, ya, yb, thr=140):
                    if not (0 <= c < W) or yb <= ya:
                        return 0.0
                    col = gray[max(0, int(ya)):min(H, int(yb)), max(0, c - 1):c + 2] < thr
                    return float(col.any(axis=1).mean()) if col.size else 0.0
                # 인식된 음자리표·임시표·마디선·박자표 등 기호 안이면 그 기호의 일부다
                if any(kx - 2 <= cxg <= kx + kw + 2 and ky - 2 <= cyg <= ky + kh + 2
                       for kx, ky, kw, kh in sym_boxes):
                    _dbg.append((m.number, cur_reason := "symbol", int(x0 + bx), int(y0 + by))) if _dbg is not None else None
                    continue
                # ⓪ 머리는 홀로 떨어져 있다: 머리를 0.35칸 넓힌 테두리에 잉크가 거의 없어야 한다(오선 줄 자리와
                #    오른쪽 기둥 자리는 뺀다). 음자리표 꼬리·빔 끝·화음 머리·임시표·마디선·꼬리 끝은 다른 기호에
                #    붙어 있어 여기서 걸러진다.
                pad_ = max(2, int(0.35 * il))
                ra0, ra1 = max(0, int(gy0) - pad_), min(H, int(gy1) + pad_)
                rb0, rb1 = max(0, int(gx0) - pad_), min(W, int(gx1) + pad_)
                ring = (gray[ra0:ra1, rb0:rb1] < 140).copy()
                ring[int(gy0) - ra0:int(gy1) - ra0, int(gx0) - rb0:int(gx1) - rb0] = False
                for ly in ys:                                       # 오선 줄
                    r0_, r1_ = int(ly - 0.12 * il) - ra0 - 1, int(ly + 0.12 * il) - ra0 + 2
                    ring[max(0, r0_):max(0, r1_), :] = False
                sc0 = int(gx1) - 4 - rb0                            # 오른쪽 위 기둥 자리
                ring[:max(0, int(gy0) + 2 - ra0), max(0, sc0):sc0 + 9] = False
                if ring.size and ring.mean() > 0.12:
                    _dbg.append((m.number, cur_reason := "ring", int(x0 + bx), int(y0 + by))) if _dbg is not None else None
                    continue
                # ① 오른쪽 끝에서 위로 올라가는 짧은 기둥(1.1칸 이상 — 꾸밈음 기둥은 보통 기둥보다 짧다)
                L = 1.1 * il
                cols = range(int(gx1) - 4, int(gx1) + 5)
                # 스캔에서 가는 기둥은 끊기거나 흐리다 — 회색(밝기 200 미만)까지 센다("교회" 36마디 꾸밈음은
                # 1px 회색 기둥이라 진한 잉크만 세면 놓쳤다. 모양 검사가 여럿이라 오탐은 늘지 않음)
                if max(colr(c, gy0 - L, gy0 + 2, 200) for c in cols) < 0.5:
                    _dbg.append((m.number, cur_reason := "stem", int(x0 + bx), int(y0 + by))) if _dbg is not None else None
                    continue
                # ①-2 기둥이 2.8칸보다 길면 보통 음의 기둥(꼬리 끝)이나 마디선이다(꾸밈음 기둥은 짧다)
                bc = max(cols, key=lambda c: colr(c, gy0 - L, gy0 + 2))
                run, gap_, yy = 0, 0, int(gy0)
                while yy > 0 and gap_ <= 2:
                    yy -= 1
                    if (gray[yy, max(0, bc - 1):bc + 2] < 140).any():
                        run += 1 + gap_
                        gap_ = 0
                    else:
                        gap_ += 1
                if run > 2.8 * il:
                    _dbg.append((m.number, cur_reason := "stemlong", int(x0 + bx), int(y0 + by))) if _dbg is not None else None
                    continue
                # ①-3 머리 바로 위(기둥 자리 제외)는 비어 있어야 한다 — 꼬리 끝·음자리표 꼬리는 위·왼쪽에 몸통이 있다
                above = gray[max(0, int(gy0 - 1.2 * il)):max(0, int(gy0 - 0.15 * il)),
                             int(gx0):max(int(gx0) + 1, int(gx1) - 4)] < 140
                if above.size:
                    # 오선·덧줄(가로로 꽉 찬 줄)은 빼고 본다 — 덧줄 위 꾸밈음("교회" 51마디 왼손)
                    rows_ = above.mean(axis=1) < 0.7
                    above = above[rows_] if rows_.any() else above[:0]
                if above.size and above.mean() > 0.2:      # 사선(꾸밈음 빗금)은 ~0.12, 꼬리·음자리표 몸통은 0.4 이상
                    _dbg.append((m.number, cur_reason := "above", int(x0 + bx), int(y0 + by), round(float(above.mean()), 3))) if _dbg is not None else None
                    continue
                # ② 그 기둥이 머리 아래로도 이어지면 ♯의 세로획·마디선이다
                if max(colr(c, gy1 + 2, gy1 + 1.0 * il) for c in cols) > 0.6:
                    _dbg.append((m.number, cur_reason := "below", int(x0 + bx), int(y0 + by))) if _dbg is not None else None
                    continue
                # ③ 머리 왼쪽으로 잉크가 이어지면 빔 끝(기둥과 만나는 모서리)이다
                # (오선 줄 자리는 빼고 본다 — 줄 위에 걸친 꾸밈음은 오선이 왼쪽으로 이어진다)
                rows = [r for r in range(int(gy0), int(gy1))
                        if all(abs(r - ly) > max(2.0, 0.12 * il) for ly in ys)]
                if rows:
                    left = gray[rows, max(0, int(gx0 - 0.5 * il)):max(0, int(gx0) - 1)] < 140
                    if left.size and left.mean() > 0.35:   # 빔은 빽빽(0.6↑), 꾸밈음 빗금은 가는 사선
                        _dbg.append((m.number, cur_reason := "left", int(x0 + bx), int(y0 + by))) if _dbg is not None else None
                        continue
                # ④ 기둥 꼭대기에서 오른쪽으로 길게(1.2칸 넘게) 이어지는 굵은 가로 잉크는 빔 — 꾸밈음의
                #    꼬리는 짧다(실제 음의 기둥에 빔이 붙은 경우 제외)
                topy = int(gy0 - 2.6 * il)
                beam = gray[max(0, topy):int(gy0 - 0.8 * il), int(gx1) + 2:min(W, int(gx1 + 1.1 * il))] < 140
                def _maxrun(col):
                    best_, cur_ = 0, 0
                    for v_ in col:
                        cur_ = cur_ + 1 if v_ else 0
                        best_ = max(best_, cur_)
                    return best_
                # 빔 = 한 덩어리로 두꺼운 띠(0.35칸 이상). 이음줄·꼬리처럼 가는 선이 겹친 것은 빔이 아니다
                if beam.size and np.mean([_maxrun(beam[:, c]) >= 0.35 * il for c in range(beam.shape[1])]) > 0.6:
                    _dbg.append((m.number, cur_reason := "beamtop", int(x0 + bx), int(y0 + by))) if _dbg is not None else None
                    continue
                d = mx - gx1
                if best is None or d < best[0]:
                    best = (d, (int(gx0), int(gy0), int(gx1), int(gy1)))
            if best is None:
                continue
            gx0, gy0, gx1, gy1 = best[1]
            # 본음: 같은 보표에서 이 머리 바로 오른쪽(가장 가까운) 음 — 지금 본 음보다 더 가까운 음이 있으면 그 음
            right = [e for e in m.elements if e.staff_no == main.staff_no and e.kind == "note" and e.src_bbox
                     and not e.is_grace and e.src_bbox[0] > gx1 - 0.2 * il]
            if right and min(right, key=lambda e: e.src_bbox[0]) is not main:
                continue
            key = (m.number, gx0 // 4, gy0 // 4)
            if key in seen:
                continue
            seen.add(key)
            # 음높이: 그 자리 음자리표
            cl = staff_clefs.get(m.system) or []
            clef = cl[main.staff_no - 1] if main.staff_no - 1 < len(cl) else ("F" if main.staff_no >= 2 else "G")
            for mm in score.measures:
                if mm.system != m.system or mm.number > m.number:
                    continue
                for e in sorted(mm.elements, key=lambda e: e.src_x):
                    if e.staff_no == main.staff_no and e.clef_change and (mm.number < m.number or e.src_x <= gx0):
                        clef = e.clef_change
            steps = int(round(((gy0 + gy1) / 2.0 - ys[0]) / (il / 2.0)))
            dia = _DIA_TOP.get(clef, 38) - steps
            step, octave = _STEP[dia % 7], dia // 7
            alter = _alter_map(score.key_at(m.number)).get(step, 0)
            el = Element(kind="note", duration_base=8, dotted=False, step=step, octave=octave, alter=alter,
                         src_bbox=(gx0, gy0, gx1 - gx0, gy1 - gy0), src_x=gx0, staff_no=main.staff_no,
                         voice_no=main.voice_no, is_grace=True, synthetic=True)
            el.suggested = True
            # 꾸밈음 → 본음 이음줄: 그 구간을 지나는 이음줄 곡선이 원본(인식된 곡선)에 있으면 함께 넣는다
            slur = any(sx <= gx1 + il and sx + sw >= mx - il and sy - 2 * il <= gy0 <= sy + sh + 3 * il
                       for sx, sy, sw, sh in slur_boxes)
            # [v2.30] 원본에 그 이음줄 곡선이 있으면 곡선 그대로 쓰도록 찾아 둔다
            slur_curve = None
            for (sx, sy, sw, sh), cv in zip(slur_boxes, _slur_curves(omr_path, getattr(score, "_omr_sheet", 1))
                                            if omr_path else []):
                if cv and sx <= gx1 + il and sx + sw >= mx - il and sy - 2 * il <= gy0 <= sy + sh + 3 * il:
                    slur_curve, slur = cv, True
                    break
            if not slur:
                # 원본 그림에서: 꾸밈음 기둥 오른쪽 ~ 본음 사이 위쪽을 가로지르는 곡선(대부분의 열에 잉크)
                top_ = min(gy0, my) - 2.8 * il
                bot_ = min(gy0, my) - 0.2 * il
                ca, cb = int(gx1 + 0.6 * il), int(mx + mw * 0.5)
                if cb - ca >= 3:
                    arc = gray[max(0, int(top_)):max(0, int(bot_)), ca:cb] < 140
                    slur = bool(arc.size) and arc.any(axis=0).mean() > 0.7
                    if slur and slur_curve is None:
                        slur_curve = _trace_curve(arc, ca, max(0, int(top_)))      # [v2.30] 원본 잉크로 곡선
                    if not slur:
                        # 아래로 휜 이음줄(머리 아래 — "교회" 36마디)
                        lo_ = max(gy1, my + mh) + 0.2 * il
                        arc2 = gray[int(lo_):min(H, int(lo_ + 2.6 * il)), ca:cb] < 140
                        slur = bool(arc2.size) and arc2.any(axis=0).mean() > 0.7
                        slur_below = slur
                        if slur and slur_curve is None:
                            slur_curve = _trace_curve(arc2, ca, int(lo_))
            acc = {1: "♯", -1: "♭"}.get(alter, "")
            out.append({"measure": m.number, "element": el, "chord_with": None, "kind": "grace",
                        "main": main, "slur": slur, "slur_curve": slur_curve,
                        "blob": {"box": (gx0, gy0, gx1, gy1), "il": il, "kind": "grace", "line": m.system,
                                 "staff_no": main.staff_no, "measure": m.number},
                        "text": f"{m.number}마디 · 꾸밈음 {step}{acc}{octave} → "
                                f"{main.step}{ {1: '♯', -1: '♭'}.get(main.alter, '') }{main.octave} 앞"})
    return out


def drop_phantom_heads(score, omr_path, img_bgr, scale=1.0):
    """[v2.01 유령 머리] Audiveris가 빔·이음줄이 겹친 자리 등을 음표 머리로 잘못 읽은 것("교회" 19마디 왼손:
    빔+붙임줄 자리의 'B3' — 신뢰도 0.295, 페이지에서 가장 낮음)을 지운다. 신뢰도만으로는 못 가른다
    (4쪽의 촘촘한 화음은 0.16~0.40인데 진짜). 그래서 원본 잉크를 함께 본다:
      · 검정 머리: 신뢰도 < 0.4 이고 머리 안쪽 타원의 잉크가 80% 미만(진짜 머리는 0.84~1.0, 유령은 0.65~0.69)
      · 빈 머리: 신뢰도 < 0.4 이고 안쪽이 비고(15% 미만) 테두리 잉크도 약함(60% 미만)
    화음 구성음이면 바로 지우고, 혼자 있는 음이면 지워도 마디 검사가 나빠지지 않을 때만 지운다.
    반환: 지운 개수."""
    from score import editing as ed
    if not omr_path or img_bgr is None:
        return 0
    gray = _gray(img_bgr, scale)
    root = _sheet_root(omr_path, getattr(score, "_omr_sheet", 1))
    bad = []
    for h in root.iter("head"):
        g = float(h.get("grade") or 1.0)
        b = h.find("bounds")
        if g >= 0.4 or b is None:
            continue
        x, y, w, hh = (float(b.get(k)) for k in ("x", "y", "w", "h"))
        sub = (gray[int(y):int(y + hh), int(x):int(x + w)] < 140).astype(np.uint8)
        if sub.size == 0:
            continue
        H_, W_ = sub.shape
        yy, xx = np.mgrid[0:H_, 0:W_]
        r = ((xx - W_ / 2) / (W_ / 2)) ** 2 + ((yy - H_ / 2) / (H_ / 2)) ** 2
        inner, ring = float(sub[r < 0.35].mean()), float(sub[(r >= 0.45) & (r < 1.0)].mean())
        void = "VOID" in (h.get("shape") or "")
        if (void and inner < 0.15 and ring < 0.6) or (not void and inner < 0.8):
            bad.append((x, y))
    score._phantom_heads = list(bad)        # 'Audiveris가 찾고 뺀 머리'로 다시 제안되지 않게
    removed = 0
    good = ("ok", "anacrusis", "time_change")
    for bx, by in bad:
        for m in score.measures:
            hit = [e for e in m.elements if e.src_bbox and abs(e.src_bbox[0] - bx) <= 4 and abs(e.src_bbox[1] - by) <= 4]
            if not hit:
                continue
            e = hit[0]
            mates = [o for o in m.elements if o is not e and e.chord_group != -1 and o.chord_group == e.chord_group]
            if not mates:
                before = ed.validate(score).get(m.number, ("",))[0]
                m.elements.remove(e)
                after = ed.validate(score).get(m.number, ("",))[0]
                if before in good and after not in good:
                    m.elements.append(e)               # 나빠지면 되돌린다
                    break
            else:
                m.elements.remove(e)
                if len(mates) == 1:
                    mates[0].chord_group = -1
            score.arcs = [a for a in score.arcs if a.start is not e and a.stop is not e]
            removed += 1
            break
    return removed


def _fit_curve(mask, x_off, y_off, il):
    """열마다 잉크 가운데 y에 2차식을 맞추고(어긋난 조각은 버리고 다시 맞춤) 3차 곡선으로."""
    cols = [(c, float(np.nonzero(mask[:, c])[0].mean())) for c in range(mask.shape[1]) if mask[:, c].any()]
    if len(cols) < 6:
        return None
    xs = np.array([c for c, _ in cols], float)
    ys = np.array([y for _, y in cols], float)
    keep = np.ones(len(xs), bool)
    for _ in range(3):
        if keep.sum() < 6:
            return None
        co = np.polyfit(xs[keep], ys[keep], 2)
        res = np.abs(np.polyval(co, xs) - ys)
        keep = res <= max(0.5 * il, 3.0)
    co = np.polyfit(xs[keep], ys[keep], 2)
    x0, x1 = xs[keep].min(), xs[keep].max()
    f = lambda x: float(np.polyval(co, x))
    d0 = float(np.polyval(np.polyder(co), x0))
    qx, qy = (x0 + x1) / 2.0, f(x0) + d0 * (x1 - x0) / 2.0          # 2차 베지어 제어점(두 접선의 만남)
    p0, p2 = (x0, f(x0)), (x1, f(x1))
    c1 = (p0[0] + 2.0 / 3 * (qx - p0[0]), p0[1] + 2.0 / 3 * (qy - p0[1]))
    c2 = (p2[0] + 2.0 / 3 * (qx - p2[0]), p2[1] + 2.0 / 3 * (qy - p2[1]))
    return (p0[0] + x_off, p0[1] + y_off, c1[0] + x_off, c1[1] + y_off, c2[0] + x_off, c2[1] + y_off,
            p2[0] + x_off, p2[1] + y_off)


def _fit_poly(mask, x_off, y_off, il):
    """[v2.31] 원본 잉크를 따라가는 꺾은선(점 목록): 열마다 잉크 가운데 y에 4차식을 맞추고 어긋난 조각을 버린
    뒤, 남은 열 범위에서 4px마다 점을 찍는다(납작한 꼭대기·가파른 끝을 한 곡선보다 잘 따라간다)."""
    cols = [(c, float(np.nonzero(mask[:, c])[0].mean())) for c in range(mask.shape[1]) if mask[:, c].any()]
    if len(cols) < 8:
        return None
    xs = np.array([c for c, _ in cols], float)
    ys = np.array([y for _, y in cols], float)
    keep = np.ones(len(xs), bool)
    deg = 4 if len(xs) >= 20 else 2
    for _ in range(3):
        if keep.sum() < deg + 3:
            return None
        co = np.polyfit(xs[keep], ys[keep], deg)
        keep = np.abs(np.polyval(co, xs) - ys) <= max(0.45 * il, 3.0)
    co = np.polyfit(xs[keep], ys[keep], deg)
    x0, x1 = xs[keep].min(), xs[keep].max()
    pts = [(float(x) + x_off, float(np.polyval(co, x)) + y_off) for x in np.arange(x0, x1 + 1, 4.0)]
    return pts if len(pts) >= 3 else None


def trace_arc_between(img_bgr, a, b, staff_ys, dy=0.0, prefer_above=None, heads=None, limits=None):
    """[v2.31] 사용자가 넣은 이음줄의 원본 곡선 찾기: 두 음(a→b) 사이, 위·아래 띠에서 오선·기둥(곧은 선)을 지우고
    남는 '가로로 긴 얇은 덩어리'를 곡선으로 보고 따낸다. 둘 다 있으면 넓게 걸친 쪽(같으면 prefer_above 쪽).
    좌표: 그 페이지 그림 기준으로 찾고 곡 좌표(dy 더함)로 돌려준다. 못 찾으면 None."""
    gray = _gray(img_bgr, 1.0)
    H, W = gray.shape[:2]
    il = (staff_ys[4] - staff_ys[0]) / 4.0
    ax0, ay0, aw, ah = a.src_bbox
    bx0, by0, bw, bh = b.src_bbox
    ay, by = ay0 - dy + ah / 2.0, by0 - dy + bh / 2.0
    xa, xb = int(ax0), int(bx0 + bw)
    if xb - xa < il:
        return None
    lo_y, hi_y = min(ay, by, staff_ys[0]), max(ay, by, staff_ys[4])
    bands = {"above": (int(max(0, min(ay, by) - 6 * il)), int(min(ay, by) + 0.3 * il)),
             "below": (int(max(ay, by) - 0.3 * il), int(min(H, max(ay, by) + 6 * il)))}
    # [v2.52] 이웃 보표 영역은 찾지 않는다(오른손 아래 띠가 왼손 오선 위 이음줄까지 닿아 엉뚱한 곡선을 잡았다
    # — "교회" 9마디). limits = (위 한계, 아래 한계): 이웃 보표와의 가운데(페이지 좌표)
    if limits:
        lo_lim, hi_lim = limits
        bands = {k: (max(v[0], int(lo_lim - dy)) if lo_lim is not None else v[0],
                     min(v[1], int(hi_lim - dy)) if hi_lim is not None else v[1]) for k, v in bands.items()}
    # 곡선이 있어야 할 높이: 기둥 쪽이면 기둥 끝(머리에서 약 3.5칸), 머리 쪽이면 머리 바로 옆(약 1칸)
    def _expect_y(side):
        stem_side = (prefer_above is not None) and (side == ("above" if prefer_above else "below"))
        off = 3.5 * il if stem_side else 1.0 * il
        return (min(ay, by) - off) if side == "above" else (max(ay, by) + off)
    found = {}
    for side, (y0, y1) in bands.items():
        if y1 - y0 < il:
            continue
        reg = (gray[y0:y1, xa:xb] < 140).astype(np.uint8)
        # 오선은 '알고 있는 줄 자리'만 지운다 — 모양(가로로 긴 선)으로 지우면 납작한 이음줄 꼭대기까지
        # 지워졌다("교회" 7마디 오른손 위 이음줄)
        hl = np.zeros_like(reg)
        for ly in staff_ys:
            r0, r1 = int(ly - 0.12 * il) - y0, int(ly + 0.12 * il) + 1 - y0
            if r1 > 0 and r0 < reg.shape[0]:
                hl[max(0, r0):min(reg.shape[0], r1), :] = 1
        # 음표 머리(같은 보표의 모든 머리)는 지운다 — 머리 조각이 곡선 끝을 머리로 끌어내리지 않게
        for hx, hy, hw, hh in (heads or []):
            hy -= dy
            pad = 0.3 * il
            c0, c1_ = int(hx - pad) - xa, int(hx + hw + pad) - xa
            r0, r1 = int(hy - pad) - y0, int(hy + hh + pad) - y0
            if c1_ > 0 and r1 > 0 and c0 < reg.shape[1] and r0 < reg.shape[0]:
                reg[max(0, r0):min(reg.shape[0], r1), max(0, c0):min(reg.shape[1], c1_)] = 0
        rest = reg & (1 - hl)
        # 오선 줄을 지운 틈(줄 굵기만큼)을 세로로 다시 잇는다 — 오선을 따라 지나가는 이음줄이 끊기지 않게
        rest = cv2.morphologyEx(rest, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(5, int(il * 0.45)))))
        vl = cv2.morphologyEx(rest, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(il)))))
        rest = rest & (1 - vl)                   # 기둥(세로로 긴 선)
        rest = cv2.morphologyEx(rest, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        n, lab, st, _c = cv2.connectedComponentsWithStats(rest, 8)
        # 오선을 따라 지나가는 곡선은 오선을 지우면 여러 조각으로 끊긴다 — 얇은 곡선 조각을 모두 모아 함께 본다
        keep = np.zeros(n, dtype=bool)
        for j in range(1, n):
            x, y, w, h, area = st[j]
            if area < 6 or h > 3.2 * il:
                continue                         # (가파른 곡선 끝 조각은 좁다 — 폭으로 거르지 않고, 어긋난 조각은 맞춤에서 버린다)
            if area > 0.5 * w * h and h > 0.6 * il:
                continue                         # 머리처럼 꽉 찬 덩어리
            keep[j] = True
        if keep.any():
            # 조각을 '사슬'로 묶는다(가까운 조각끼리 — 오선에 끊긴 곡선은 다시 이어지고, 떨어진 다른 곡선은 따로)
            kmask = keep[lab].astype(np.uint8)
            link = cv2.dilate(kmask, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(0.9 * il)),
                                                                                 max(3, int(0.5 * il)))))
            nc, clab, cst, _cc = cv2.connectedComponentsWithStats(link, 8)
            span = float(xb - xa)
            best = None
            ey = _expect_y(side) - y0
            for c in range(1, nc):
                cm = (clab == c) & (kmask > 0)
                cols = np.nonzero(cm.any(axis=0))[0]
                if not len(cols):
                    continue
                cover = cm.any(axis=0).mean()
                extent = (cols.max() - cols.min()) / span
                if extent < 0.45 or cover < 0.3:
                    continue
                rows = np.nonzero(cm.any(axis=1))[0]
                dist = abs(rows.mean() - ey) / il
                key = (round(extent, 1), -dist)           # 두 음 사이를 넓게 덮는 것 먼저, 같으면 있어야 할 높이에 가까운 것
                if best is None or key > best[0]:
                    best = (key, cm, int(cols.max() - cols.min()))
            if best is not None:
                cv_ = _fit_poly(best[1], xa, y0, il)
                if cv_ is not None:
                    found[side] = (best[2], cv_)
    if not found:
        return None
    # 조판 규칙이 말하는 쪽(prefer_above)에서 찾았으면 그쪽 — 다른 성부의 이음줄을 잘못 잡지 않게
    if prefer_above is not None and ("above" if prefer_above else "below") in found:
        side = "above" if prefer_above else "below"
        return [(x, y + dy) for x, y in found[side][1]]
    if len(found) == 2:
        wa, wb = found["above"][0], found["below"][0]
        if abs(wa - wb) < 0.15 * (xb - xa) and prefer_above is not None:
            side = "above" if prefer_above else "below"
        else:
            side = "above" if wa >= wb else "below"
    else:
        side = next(iter(found))
    return [(x, y + dy) for x, y in found[side][1]]


def mark_unexplained_stems(score, gray, staff_groups):
    """[v2.32 검토 후 미사용 — 이웃 보표 기둥·가사 글자 획·묶음괄호 등도 '설명 안 되는 기둥'으로 잡혀
    교회 56마디 중 51마디에서 나와 쓰지 않는다. 기둥↔머리 relation이 없는 인식 기둥까지 걸러야 쓸 수 있음]
    [v2.32 설명 안 되는 기둥 — 실사용] 원본에 기둥(세로로 긴 선)이 있는데 인식된 어느 음의 기둥(src_stem)
    으로도 설명되지 않으면, 그 자리에 인식 못 한 음이 있다는 증거다. 마디마다
    measure.unexplained_stems = {보표: [x, ...]} 로 적는다(박자 검사에서 '쉼표 생략'으로 봐주지 않게).
    "교회" 9마디 오른손: G♯4 2분음표(머리·위 기둥 모두 Audiveris 미인식)의 위 기둥이 F♯4(아래 기둥)
    머리 바로 위에 남아 있다 — 같은 x지만 F♯4 기둥은 아래로만 뻗으므로 설명되지 않는다.
    오선 높이만큼 위아래 줄에 딱 맞는 세로선(마디선)은 뺀다."""
    H, W = gray.shape[:2]
    for m in score.measures:
        grp = staff_groups.get(m.system) or []
        res = {}
        for sn in sorted({e.staff_no for e in m.elements}):
            if sn - 1 >= len(grp):
                continue
            ys = grp[sn - 1]
            il = (ys[4] - ys[0]) / 4.0
            es = [e for e in m.elements if e.staff_no == sn and e.src_bbox]
            if not es:
                continue
            x0 = int(max(0, min(e.src_bbox[0] for e in es) - 1.5 * il))
            x1 = int(min(W, max(e.src_bbox[0] + e.src_bbox[2] for e in es) + 1.0 * il))
            y0 = int(max(0, ys[0] - 3.5 * il))
            y1 = int(min(H, ys[4] + 3.5 * il))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            reg = (gray[y0:y1, x0:x1] < 140).astype(np.uint8)
            vl = cv2.morphologyEx(reg, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(2.5 * il)))))
            n, _lab, st, _c = cv2.connectedComponentsWithStats(vl, 8)
            stems = [getattr(e, "src_stem", None) for e in m.elements if getattr(e, "src_stem", None)]
            found = []
            for j in range(1, n):
                x, y, w, h, area = st[j]
                if w > 0.5 * il:
                    continue
                top, bot, cx = y + y0, y + h + y0, x + x0 + w / 2.0
                if abs(top - ys[0]) < 0.3 * il and abs(bot - ys[4]) < 0.3 * il:
                    continue                                   # 마디선
                # 인식된 기둥으로 덮이지 않는 길이
                cover = 0.0
                for sx_, st_t, st_b in stems:
                    if abs(sx_ - cx) <= 6:
                        cover += max(0.0, min(bot, st_b) - max(top, st_t))
                if (bot - top) - cover >= 2.0 * il:
                    found.append(cx)
            if found:
                res[sn] = found
        m.unexplained_stems = res


def split_far_unisons(score, blobs):
    """[v2.54 화음 속 먼 같은 음 — 실사용] Audiveris가 빔·기둥을 못 읽은 음을 앞 화음에 붙이면, 화음 안에 같은
    음높이가 두 번(머리 폭 1.5배 넘게 떨어져) 들어간다("교회" 20·39마디 오른손 마지막 박: F♯4+B3 8분 화음과
    B3 8분이 빔으로 묶였는데 빔·뒤 기둥 미인식 → B3 셋째 음이 4분 화음에 붙음). 먼 쪽을 따로 떼고,
    두 음의 기둥 끝 사이에 인식 안 된 빔 모양 잉크가 있으면 둘 다 8분으로 바꿔 빔으로 묶는다(박 합은 그대로:
    4분 = 8분 + 8분). 증거로 쓴 잉크는 미인식 목록에서 뺀다. 반환: 안내 문구 목록."""
    notes = []
    used = set()
    for m in score.measures:
        groups = {}
        for e in m.elements:
            if e.kind == "note" and e.chord_group != -1 and e.src_bbox and not e.is_grace:
                groups.setdefault((e.staff_no, e.chord_group), []).append(e)
        next_cg = max([e.chord_group for e in m.elements] + [0]) + 1
        next_bg = max([e.beam_group for e in m.elements if e.beam_group is not None] + [0]) + 1
        for (sn, cg), mem in groups.items():
            if len(mem) < 3:
                continue
            xs = sorted(e.src_bbox[0] for e in mem)
            col = xs[len(xs) // 2]
            w = max(e.src_bbox[2] for e in mem)
            far = [e for e in mem if abs(e.src_bbox[0] - col) > 1.5 * w]
            for b in far:
                twin = [e for e in mem if e is not b and e not in far and (e.step, e.octave, e.alter) == (b.step, b.octave, b.alter)]
                if not twin:
                    continue
                rest = [e for e in mem if e is not b]
                for e_ in rest + [b]:
                    e_._split_touched = True     # (저장 파일 고치기에서 원본 기하 표시를 다시 맞추려고)
                b.chord_group = -1
                b.src_x = b.src_bbox[0]          # 화음 자리로 맞춰졌던 위치 → 자기 머리 자리(원본이 정한다)
                # 빔 증거: 화음 기둥 끝 높이 근처, 화음과 이 음 사이를 가로지르는 인식 안 된 잉크
                stem = next((e.src_stem for e in rest if getattr(e, "src_stem", None)), None)
                beam_blob = None
                if stem is not None:
                    sx_, top, bot = stem
                    up = abs(top - min(e.src_bbox[1] for e in rest)) > abs(bot - max(e.src_bbox[1] + e.src_bbox[3] for e in rest))
                    tip = top if up else bot
                    for bl in blobs:
                        if bl.get("kind") != "other" or id(bl) in used:
                            continue
                        x0, y0, x1, y1 = bl["box"]
                        il_ = bl.get("il", 20.0)
                        if x0 <= b.src_bbox[0] + w and x1 >= sx_ - 0.5 * il_ and y0 - 1.5 * il_ <= tip <= y1 + 1.5 * il_:
                            beam_blob = bl
                            break
                if beam_blob is not None and b.duration_base == 4 and all(e.duration_base == 4 for e in rest):
                    for e in rest + [b]:
                        e.duration_base = 8
                        e.beam_group = next_bg
                    next_bg += 1
                    used.add(id(beam_blob))
                    notes.append(f"{m.number}마디 · 화음에 잘못 붙은 {b.step}{b.octave}를 떼어 앞 화음과 8분음표 빔으로 묶었습니다"
                                 "(원본의 빔을 Audiveris가 못 읽음)")
                else:
                    notes.append(f"{m.number}마디 · 화음에 잘못 붙은 {b.step}{b.octave}를 따로 떼었습니다 — 길이를 확인하세요")
                next_cg += 1
    if used:
        blobs[:] = [bl for bl in blobs if id(bl) not in used]
    return notes


def apply_orphan_flags(score, omr_path, sheet=1):
    """[v2.56 꼬리는 읽었는데 길이에 안 들어간 음 — 실사용] Audiveris가 꼬리(FLAG_n)를 인식했고 그 꼬리가 어떤 음의
    원본 기둥 끝에 붙어 있는데, MusicXML에서는 그 음이 4분음표(꼬리가 있을 수 없는 길이)로 나온 경우 — 꼬리 수대로
    길이를 바로잡는다(FLAG_1 → 8분, FLAG_2 → 16분 …). 화음이면 구성음 모두. ("교회" 41마디 오른손: FLAG_1 0.917이
    A3·C4 화음 기둥 끝에 있는데 4분으로 나와 마디가 4.5박) 반환: 안내 문구 목록."""
    import re as _re
    if not omr_path:
        return []
    try:
        root = _sheet_root(omr_path, sheet)
    except Exception:
        return []
    flags = []
    for el in root.iter("flag"):
        shp = el.get("shape") or ""
        mo = _re.match(r"(?:SMALL_)?FLAG_(\d)", shp)
        b = el.find("bounds")
        if not mo or b is None or el.get("grade") is None or float(el.get("grade")) < 0.6:
            continue
        flags.append((int(mo.group(1)), float(b.get("x")), float(b.get("y")), float(b.get("w")), float(b.get("h"))))
    if not flags:
        return []
    notes = []
    for m in score.measures:
        done_cg = set()
        for e in m.elements:
            st = getattr(e, "src_stem", None)
            if e.kind != "note" or st is None or e.duration_base != 4 or e.is_grace:
                continue
            key = (e.staff_no, e.chord_group) if e.chord_group != -1 else ("e", id(e))
            if key in done_cg:
                continue
            sx_, top, bot = st
            for n, fx, fy, fw, fh in flags:
                # 꼬리는 기둥 끝(위 기둥이면 위 끝, 아래 기둥이면 아래 끝)에서 기둥 오른쪽으로 붙는다
                near_x = sx_ - 6 <= fx <= sx_ + 10
                near_y = abs(fy - top) <= 12 or abs((fy + fh) - bot) <= 12
                if near_x and near_y:
                    new = 4 * (2 ** n)
                    mem = [e] if e.chord_group == -1 else [x for x in m.elements
                                                          if x.chord_group == e.chord_group and x.staff_no == e.staff_no]
                    for x in mem:
                        x.duration_base = new
                    done_cg.add(key)
                    notes.append(f"{m.number}마디 · {'화음' if len(mem) > 1 else '음'} {e.step}{e.octave} — Audiveris가 읽은 "
                                 f"꼬리({n}개)를 길이에 반영했습니다(4분 → {new}분)")
                    break
    return notes


def find_hidden_voice_stems(score, gray, staff_groups, barlines=None):
    """[v2.70 보류 과제 P-1(C안) — 숨은 다른 성부 음] 인식된 음 머리의 '자기 기둥 반대쪽'으로, 머리 가장자리에서
    오선 2.5칸 넘게 뻗는 세로선이 있는데 인식된 어느 기둥으로도 설명되지 않으면 → 같은 자리에 기둥 방향이 반대인
    다른 성부 음이 있다(머리를 Audiveris가 못 읽음). ("교회" 9마디 오른손: F♯4(아래 기둥) 머리 위로 G♯4의 위 기둥)
    마디선 자리는 뺀다. 반환: 표시용 덩어리 목록(kind "hidden")."""
    H, W = gray.shape[:2]
    out = []
    bars = barlines or {}
    for m in score.measures:
        grp = staff_groups.get(m.system) or []
        stems = [e.src_stem for e in m.elements if getattr(e, "src_stem", None)]
        bl = list(bars.get(m.system) or [])
        for e in m.elements:
            st = getattr(e, "src_stem", None)
            dimg = getattr(e, "stem_dir_img", None)
            if e.kind != "note" or (st is None and dimg not in ("up", "down")) or not e.src_bbox or e.is_grace \
                    or e.staff_no - 1 >= len(grp) or e.duration_base == 1:
                continue
            ys = grp[e.staff_no - 1]
            il = (ys[4] - ys[0]) / 4.0
            bx, by, bw, bh = e.src_bbox
            # 자기 기둥 방향: 원본 기둥이 있으면 그것, 없으면(되살린 머리 등) 그림에서 읽은 방향
            up = (st[1] < by - 0.5 * il) if st is not None else (dimg == "up")
            for ex in (int(bx + 1), int(bx + bw - 2)):
                if any(abs(ex - b) <= 8 for b in bl):
                    continue
                c0, c1 = max(0, ex - 2), min(W, ex + 3)
                run, gap = 0, 0
                yy = int(by + bh + 1) if up else int(by - 1)
                step = 1 if up else -1
                while 0 <= yy < H and gap <= 2:
                    if (gray[yy, c0:c1] < 150).any():
                        run += 1 + gap
                        gap = 0
                    else:
                        gap += 1
                    yy += step
                if run < 2.5 * il:
                    continue
                r_lo, r_hi = ((by + bh), (by + bh + run)) if up else ((by - run), by)
                covered = any(abs(s[0] - ex) <= 6 and (min(s[2], r_hi) - max(s[1], r_lo)) >= 0.5 * run for s in stems)
                if covered:
                    continue
                # 바로 옆에 그 방향 기둥을 가진 음이 이미 있으면(원본 기둥 정보만 없는 음) 그 음의 기둥이다
                want = "down" if up else "up"
                if any(o is not e and o.kind == "note" and o.staff_no == e.staff_no and o.src_bbox
                       and abs((o.src_bbox[0] + o.src_bbox[2] / 2.0) - ex) <= 1.5 * o.src_bbox[2]
                       and abs((o.src_bbox[1] + o.src_bbox[3] / 2.0) - (by + bh / 2.0)) <= 1.5 * il
                       and (getattr(o, "stem_dir_img", None) == want
                            or (getattr(o, "src_stem", None) and ((o.src_stem[1] < o.src_bbox[1]) == (want == "up"))))
                       for o in m.elements):
                    continue
                if any(o["measure"] == m.number and o["staff_no"] == e.staff_no and abs(o["x"] - ex) < 0.8 * il
                       for o in out):
                    continue
                out.append({"kind": "hidden", "line": m.system, "staff_no": e.staff_no, "il": il, "measure": m.number,
                            "x": ex, "dir": "down" if up else "up",
                            "box": (int(ex - 1.2 * il), int(by + bh / 2 - 0.8 * il), int(ex + 1.2 * il),
                                    int(by + bh / 2 + 0.8 * il))})
    return out



def fix_durations_by_ink(score, gray, staff_groups):
    """[v3.62] 길이를 잘못 읽어 마디가 '넘침'인 곳을 원본 잉크로 확인해 고친다. 고친 뒤 그 마디가 정확히 맞을
    때만 바꾼다(일괄 검증 — 가톨릭 성가 20곡에서 나온 Audiveris 오인식 유형).
      ① 2분음표인데 머리 속이 꽉 찼다 → 4분음표 (덧줄이 머리를 가로지르는 낮은 C4 등을 빈 머리로 읽음, 54번 7마디)
      ② 겹점인데 머리 오른쪽에 점이 하나뿐 → 점 하나 (붙임줄 끝을 점으로 읽음, 28번 13마디)
      ③ 4분음표인데 기둥 끝 옆에 꼬리 잉크가 있다 → 8분음표 (꼬리가 코드 글자와 겹쳐 놓침, 308번 4마디)
    반환: 알림 문구 목록."""
    from score import editing as ed
    H, W = gray.shape[:2]
    fixed = []

    def il_of(m, sn):
        grp = staff_groups.get(m.system) or []
        if sn - 1 < len(grp):
            ys = grp[sn - 1]
            return (ys[4] - ys[0]) / 4.0
        return None

    def dots_right(e, il):
        x0, y0, w, h = e.src_bbox
        cy = y0 + h / 2.0
        xa, xb = int(x0 + w + 0.1 * il), int(min(W, x0 + w + 1.6 * il))
        ya, yb = int(max(0, cy - 0.75 * il)), int(min(H, cy + 0.45 * il))
        if xb - xa < 3 or yb - ya < 3:
            return None
        reg = (gray[ya:yb, xa:xb] < 140).astype(np.uint8)
        n, _l, st, _c = cv2.connectedComponentsWithStats(reg, 8)
        return sum(1 for j in range(1, n)
                   if 0.15 * il <= st[j][2] <= 0.6 * il and 0.15 * il <= st[j][3] <= 0.6 * il
                   and abs(st[j][2] - st[j][3]) <= 0.3 * il)

    def flag_ink(e, il):
        st_ = getattr(e, "src_stem", None)
        if not st_:
            return 0.0
        sx, sy0, sy1 = st_
        up = (e.src_bbox[1] + e.src_bbox[3] / 2.0) > (sy0 + sy1) / 2.0     # 머리가 아래 → 기둥 위
        ty = sy0 if up else sy1
        ya, yb = (int(ty), int(ty + 1.8 * il)) if up else (int(ty - 1.8 * il), int(ty))
        xa, xb = int(sx + 0.15 * il), int(sx + 1.0 * il)
        ya, yb, xa, xb = max(0, ya), min(H, yb), max(0, xa), min(W, xb)
        if yb - ya < 3 or xb - xa < 3:
            return 0.0
        return float((gray[ya:yb, xa:xb] < 140).mean())

    free = not getattr(score, "time_sig_known", True) and not getattr(score, "implied_capacity", None)
    for m in score.measures:
        if free:
            # [v3.71] 자유 리듬(박자표 없음)은 마디 길이로 검사할 수 없다 → 머리 속이 확실히 꽉 찬 2분음표만 4분으로
            for e in m.elements:
                if e.kind == "note" and not e.is_grace and e.src_bbox and e.duration_base == 2:
                    inner, _r = _head_ink(gray, (e.src_bbox[0], e.src_bbox[1], e.src_bbox[0] + e.src_bbox[2],
                                                 e.src_bbox[1] + e.src_bbox[3]))
                    if inner >= 0.92:
                        e.duration_base = 4
                        fixed.append(f"{m.number}마디 2분→4분(머리가 꽉 참)")
            continue
        st = ed.validate(score).get(m.number)
        if not st or st[0] != "over":
            continue
        excess = st[1] - st[2]
        notes = [e for e in m.elements if e.kind == "note" and not e.is_grace and e.src_bbox]
        cands = []
        for e in notes:
            il = il_of(m, e.staff_no)
            if not il:
                continue
            if e.duration_base == 2 and not e.in_triplet:
                inner, _ring = _head_ink(gray, (e.src_bbox[0], e.src_bbox[1], e.src_bbox[0] + e.src_bbox[2],
                                                e.src_bbox[1] + e.src_bbox[3]))
                if inner >= 0.85:
                    cands.append((e, "half_to_quarter"))
            if getattr(e, "double_dot", False):
                if dots_right(e, il) == 1:
                    cands.append((e, "double_to_single"))
            if e.duration_base == 4 and flag_ink(e, il) >= 0.22 and getattr(e, "beam_group", -1) == -1:
                cands.append((e, "quarter_to_eighth"))
        for e, kind in cands:
            group = [x for x in m.elements if x is e or (getattr(e, "chord_group", -1) != -1
                                                          and getattr(x, "chord_group", -2) == e.chord_group)]
            saved = [(x, x.duration_base, x.dotted, getattr(x, "double_dot", False)) for x in group]
            for x in group:
                if kind == "half_to_quarter":
                    x.duration_base = 4
                elif kind == "double_to_single":
                    x.double_dot = False
                else:
                    x.duration_base = 8
            st2 = ed.validate(score).get(m.number)
            if st2 and st2[0] in ("ok", "anacrusis", "time_change"):
                label = {"half_to_quarter": "2분→4분(머리가 꽉 참)", "double_to_single": "겹점→점(점이 하나)",
                         "quarter_to_eighth": "4분→8분(꼬리가 있음)"}[kind]
                fixed.append(f"{m.number}마디 {label}")
                break
            for x, b, d, dd in saved:
                x.duration_base, x.dotted, x.double_dot = b, d, dd
        _ = excess
    return [f"원본 잉크로 확인해 길이를 바로잡았습니다: {', '.join(fixed)}"] if fixed else []


def detect_time_signature(score, gray, staff_groups):
    """[v3.80] Audiveris가 박자표를 읽지 못했을 때 원본 그림에서 직접 찾는다(붓의 박자표 비교와 같은 방법).
    첫 줄 첫 보표에서 첫 음표 바로 왼쪽의 잉크 덩어리를 오선을 지우고 박자표 모양(C·₵·숫자 12가지)과 비교한다.
    실사용: 교과서 '아름다운 나라'의 12/8을 놓쳐 박자표 없이 그렸다. 반환 (beats, beat_type, symbol) 또는 None."""
    from omr_bridge import brush
    ms = sorted(score.measures, key=lambda m: m.number)
    if not ms:
        return None
    line = ms[0].system
    grp = staff_groups.get(line) or []
    if not grp:
        return None
    ys = grp[0]
    il = (ys[4] - ys[0]) / 4.0
    firsts = [e.src_bbox[0] for m in ms if m.system == line for e in m.elements if e.src_bbox and e.staff_no == 1]
    if not firsts:
        return None
    H, W = gray.shape[:2]
    xb = int(max(0, min(firsts) - 0.4 * il))
    xa = int(max(0, xb - 7.0 * il))
    ya, yb = int(max(0, ys[0] - 0.3 * il)), int(min(H, ys[4] + 0.3 * il))
    if xb - xa < 2 * il:
        return None
    ink = (gray[ya:yb, xa:xb] < 140).astype(np.uint8)
    hw = max(1, int(round(il * 0.12)))
    for ly in ys:                                          # 오선 지우기 + 세로로 잇기(붓과 같은 처리)
        r = int(round(ly - ya))
        ink[max(0, r - hw):r + hw + 1, :] = 0
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(il * 0.3)))))
    cols = ink.any(axis=0)
    # 오른쪽부터: 0.3칸 이상 빈 틈으로 나뉜 잉크 덩어리들
    runs, x = [], len(cols) - 1
    gap = max(2, int(0.3 * il))
    while x >= 0:
        if cols[x]:
            x1 = x
            miss = 0
            while x >= 0 and miss < gap:
                miss = 0 if cols[x] else miss + 1
                x -= 1
            runs.append((x + miss + 1, x1 + 1))
        else:
            x -= 1
    best = None
    for x0, x1 in runs[:3]:                                # 음표에 가까운 것부터 세 덩어리까지
        sub = ink[:, x0:x1] > 0
        yy, xx = np.nonzero(sub)
        if len(xx) < 10:
            continue
        crop = sub[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
        sc = brush.match_timesigs(crop, il)
        if not sc:
            continue
        k, v = max(sc.items(), key=lambda kv: kv[1])
        if v >= 0.78 and (best is None or v > best[1]):
            best = (k, v)
        if best:
            break
    if best is None:
        return None
    k = best[0]
    if k == "ts:common":
        return 4, 4, "common"
    if k == "ts:cut":
        return 2, 2, "cut"
    b, bt = brush.TIMESIGS[k]
    return b, bt, None
