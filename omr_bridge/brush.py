# omr_bridge/brush.py
# ---------------------------------------------------------------------------
# [v2.60 붓으로 기호 넣기] 사용자가 원본 위에 붓으로 칠한 영역 안의 잉크를 읽어 어떤 기호인지 판단한다.
# 두 번 클릭(점 하나)으로는 '어디까지가 그 기호인지' 알 수 없던 작은 이음줄·쐐기 등을, 사용자가 범위를
# 정해 주므로 확실히 읽을 수 있다. 결과는 가능성 순 후보 목록 — 사용자가 하나를 골라 마무리한다.
#
# 후보 키(내부 값 — 화면 문구와 분리): slur, tie, cresc, decresc, staccato, tenuto, dot
# 좌표: 페이지 그림 좌표(인식 좌표, 곡 좌표 - 페이지 dy).
# ---------------------------------------------------------------------------
import cv2
import numpy as np

KINDS = ("slur", "tie", "cresc", "decresc", "staccato", "tenuto", "dot")

# [v2.70] 악보 글꼴(SMuFL·Bravura) 모양과 비교해 고르는 기호들: 내부 키 → 글리프(위·아래 변형)
TEMPLATES = {
    "accent": ("\uE4A0", "\uE4A1"), "marcato": ("\uE4AC", "\uE4AD"),
    "staccatissimo": ("\uE4A6", "\uE4A7"), "fermata": ("\uE4C0", "\uE4C1"),
    "sharp": ("\uE262",), "flat": ("\uE260",), "natural": ("\uE261",),
    "dyn:ppp": ("\uE52A",), "dyn:pp": ("\uE52B",), "dyn:p": ("\uE520",), "dyn:mp": ("\uE52C",),
    "dyn:mf": ("\uE52D",), "dyn:f": ("\uE522",), "dyn:ff": ("\uE52F",), "dyn:fff": ("\uE530",),
    "dyn:fp": ("\uE534",), "dyn:sfz": ("\uE539",),
    # [v2.80] 꾸밈·숨표·이동·옥타브·페달
    "trill": ("\uE566",), "mordent": ("\uE56C",), "turn": ("\uE567",), "breath": ("\uE4CE",),
    "segno": ("\uE047",), "coda": ("\uE048",), "ped": ("\uE650",), "ped_up": ("\uE655",),
    # [v2.90] 트레몰로(기둥 빗금 1~3개)·카이주라(//)
    "tremolo1": ("\uE220",), "tremolo2": ("\uE221",), "tremolo3": ("\uE222",), "caesura": ("\uE4D1",),
}
# [v2.80] 옥타브 기호는 칠한 범위 왼쪽의 글자 부분으로 비교(오른쪽은 점선)
OTTAVA = {"8va": "\uE511", "8vb": "\uE51C", "15ma": "\uE514"}
# [v2.80] 글자로 쓰는 이동 기호
TEXTS = {"dc": "D.C.", "ds": "D.S.", "fine": "Fine", "dc_al_fine": "D.C. al Fine", "dc_al_coda": "D.C. al Coda",
         "ds_al_fine": "D.S. al Fine", "ds_al_coda": "D.S. al Coda", "to_coda": "To Coda",
         # [v2.90] 글자 셈여림(점점 세게·여리게)
         "cresc_text": "cresc.", "dim_text": "dim.", "decresc_text": "decresc."}
_TXCACHE = {}
_TXCOUNT = {}


def _text_template(text):
    if text in _TXCACHE:
        return _TXCACHE[text]
    from PIL import Image, ImageDraw
    from score.render import _font_header
    fnt = _font_header(60)
    im = Image.new("L", (60 * (len(text) + 2), 140), 0)
    ImageDraw.Draw(im).text((20, 100), text, font=fnt, fill=255, anchor="ls")
    a = np.array(im) > 110
    ys, xs = np.nonzero(a)
    out = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    _TXCACHE[text] = out
    return out


def match_texts(ink, il=20.0):
    """글자 이동 기호: 글자 그림을 잉크 상자 크기로 맞춰 겹침(가로세로 비가 크게 다르면 0).
    [v2.90] 글자는 높이가 있고(0.55칸 이상) 조각이 여럿이다 — 납작한 선(테누토 등)을 글자로 늘려 맞추지 않게."""
    out = {}
    if ink is None or ink.shape[1] < 8 or ink.shape[0] < 0.55 * il:
        return out
    n_, _l, st_, _c = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    n_ink = sum(1 for j in range(1, n_) if st_[j][4] >= 4)
    if n_ink < 2:
        return out
    ar = ink.shape[1] / float(max(1, ink.shape[0]))
    for k, t in TEXTS.items():
        tt = _text_template(t)
        tar = tt.shape[1] / float(tt.shape[0])
        if not (0.65 <= ar / tar <= 1.5):
            out[k] = 0.0
            continue
        tr = cv2.resize(tt.astype(np.uint8), (ink.shape[1], ink.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        out[k] = _iou_centered(ink, tr)
    return out
_TCACHE = {}


def _template(ch, size):
    """글리프 하나를 그 크기로 그려 잉크 부분만(부울 배열)."""
    key = (ch, int(size))
    if key in _TCACHE:
        return _TCACHE[key]
    from PIL import Image, ImageDraw
    from score import smufl
    fnt = smufl.get_font(int(size))
    im = Image.new("L", (int(size * 3), int(size * 3)), 0)
    ImageDraw.Draw(im).text((size, size * 1.6), ch, font=fnt, fill=255, anchor="ls")
    a = np.array(im) > 110
    ys, xs = np.nonzero(a)
    out = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(xs) else None
    _TCACHE[key] = out
    return out


def _iou_centered(a, b):
    """두 부울 그림을 가운데를 맞춰 겹친 IoU(1px 번짐 허용)."""
    H = max(a.shape[0], b.shape[0]) + 4
    W = max(a.shape[1], b.shape[1]) + 4
    A = np.zeros((H, W), np.uint8)
    B = np.zeros((H, W), np.uint8)
    ya, xa = (H - a.shape[0]) // 2, (W - a.shape[1]) // 2
    yb, xb = (H - b.shape[0]) // 2, (W - b.shape[1]) // 2
    A[ya:ya + a.shape[0], xa:xa + a.shape[1]] = a
    B[yb:yb + b.shape[0], xb:xb + b.shape[1]] = b
    k = np.ones((3, 3), np.uint8)
    Ad, Bd = cv2.dilate(A, k), cv2.dilate(B, k)
    inter = (A & Bd).sum() + (B & Ad).sum()
    return inter / float(max(1, A.sum() + B.sum()))


def match_templates(ink, il):
    """잉크(부울, 기호 하나를 감싸게 자른 것)와 글꼴 모양을 크기 여러 단계로 맞춰 {키: 점수}."""
    if ink is None or ink.sum() < 6:
        return {}
    out = {}
    for key, chars in TEMPLATES.items():
        best = 0.0
        for ch in chars:
            for f in (0.75, 0.85, 0.95, 1.05, 1.18, 1.32):
                t = _template(ch, 4 * il * f)
                if t is None:
                    continue
                # 크기 차이가 너무 크면 건너뜀
                if not (0.6 <= t.shape[0] / max(1, ink.shape[0]) <= 1.7 and 0.6 <= t.shape[1] / max(1, ink.shape[1]) <= 1.7):
                    continue
                best = max(best, _iou_centered(ink, t))
        out[key] = best
    return out


def stroke_mask(shape, strokes, radius):
    """붓 자국(점 목록들)을 그림 크기의 마스크로."""
    m = np.zeros(shape[:2], np.uint8)
    r = max(1, int(round(radius)))
    for pts in strokes:
        pts = [(int(round(x)), int(round(y))) for x, y in pts]
        if len(pts) == 1:
            cv2.circle(m, pts[0], r, 1, -1)
        for a, b in zip(pts, pts[1:]):
            cv2.line(m, a, b, 1, thickness=2 * r, lineType=cv2.LINE_8)
            cv2.circle(m, b, r, 1, -1)
        if pts:
            cv2.circle(m, pts[0], r, 1, -1)
    return m


def _fit_poly(mask, il):
    """열마다 잉크 가운데 → 4차식(어긋난 조각 제외) → 점 목록(그림 좌표)."""
    cols = [(c, float(np.nonzero(mask[:, c])[0].mean())) for c in range(mask.shape[1]) if mask[:, c].any()]
    if len(cols) < 6:
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
    return [(float(x), float(np.polyval(co, x))) for x in np.arange(x0, x1 + 1, 3.0)]


def analyze(gray, strokes, radius, staff_ys_all, heads, il):
    """칠한 영역의 잉크를 읽어 후보를 가능성 순으로 돌려준다.
    staff_ys_all: 그 줄 보표들의 오선 y 목록들(그림 좌표), heads: 머리 상자 (x, y, w, h) 목록(그림 좌표).
    반환 {"cands": [(key, score), ...], "geom": {...}} 또는 None(잉크 없음)."""
    H, W = gray.shape[:2]
    m = stroke_mask(gray.shape, strokes, radius)
    ys_, xs_ = np.nonzero(m)
    if not len(xs_):
        return None
    pad = int(il)
    X0, X1 = max(0, xs_.min() - pad), min(W, xs_.max() + pad + 1)
    Y0, Y1 = max(0, ys_.min() - pad), min(H, ys_.max() + pad + 1)
    ink = ((gray[Y0:Y1, X0:X1] < 150) & (m[Y0:Y1, X0:X1] > 0)).astype(np.uint8)
    raw = ink.copy()
    # 알고 있는 것은 뺀다: 오선 줄(자리를 아는 행), 음표 머리, 기둥(세로로 긴 선)
    for ys in staff_ys_all:
        for ly in ys:
            r0, r1 = int(ly - 0.12 * il) - Y0, int(ly + 0.12 * il) + 1 - Y0
            if r1 > 0 and r0 < ink.shape[0]:
                ink[max(0, r0):min(ink.shape[0], r1), :] = 0
    for hx, hy, hw, hh in heads:
        c0, c1 = int(hx - 0.1 * il) - X0, int(hx + hw + 0.1 * il) - X0
        r0, r1 = int(hy - 0.1 * il) - Y0, int(hy + hh + 0.1 * il) - Y0
        if c1 > 0 and r1 > 0 and c0 < ink.shape[1] and r0 < ink.shape[0]:
            ink[max(0, r0):r1, max(0, c0):c1] = 0
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(5, int(il * 0.45)))))
    vl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(1.2 * il)))))
    has_vertical = bool(vl.any())             # 세로 획(임시표·f의 기둥 등)이 있으면 이음줄·쐐기·테누토가 아니다
    ink = ink & (1 - vl)
    n, lab, st, _c = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros(n, bool)
    for j in range(1, n):
        if st[j][4] >= 4:
            keep[j] = True
    ink = keep[lab].astype(np.uint8)
    yy, xx = np.nonzero(ink)
    if len(xx) < 4:
        # 남은 잉크가 거의 없으면(점 하나 정도) 원래 잉크로 다시 본다
        ink = raw
        yy, xx = np.nonzero(ink)
        if len(xx) < 3:
            return None
    bx0, bx1, by0, by1 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
    w, h = (bx1 - bx0) / il, (by1 - by0) / il
    area = len(xx)
    fill = area / float(max(1, (bx1 - bx0) * (by1 - by0)))
    cx, cy = X0 + (bx0 + bx1) / 2.0, Y0 + (by0 + by1) / 2.0
    # 열마다 잉크 덩어리 수와 세로 퍼짐
    runs, spread = [], []
    for c in range(bx0, bx1):
        col = ink[by0:by1, c]
        if not col.any():
            continue
        d = np.diff(np.concatenate([[0], col.astype(int), [0]]))
        runs.append(int((d == 1).sum()))
        nz = np.nonzero(col)[0]
        spread.append(nz.max() - nz.min() + 1)
    runs_avg = float(np.mean(runs)) if runs else 1.0
    q = max(1, len(spread) // 5)
    sp_l = float(np.mean(spread[:q])) if spread else 0.0
    sp_r = float(np.mean(spread[-q:])) if spread else 0.0

    sc = {k: 0.02 for k in KINDS}
    geom = {"box": (X0 + bx0, Y0 + by0, X0 + bx1, Y0 + by1), "center": (cx, cy)}
    small = max(w, h) <= 0.9
    # 점 모양(스타카토·부점)
    if small and fill >= 0.4 and 0.4 <= (w / max(h, 1e-3)) <= 2.5:
        near = None
        for hx, hy, hw, hh in heads:
            hcx, hcy = hx + hw / 2.0, hy + hh / 2.0
            dxn, dyn = (cx - hcx) / il, (cy - hcy) / il
            if 0.5 <= dxn <= 2.2 and abs(dyn) <= 0.7:
                near = "dot"
                break
            if abs(dxn) <= 0.8 and 0.5 <= abs(dyn) <= 3.0:
                near = near or "staccato"
        sc["dot"] = 0.9 if near == "dot" else 0.35
        sc["staccato"] = 0.9 if near == "staccato" else 0.4
    # 테누토(짧은 가로선)
    if h <= 0.5 and 0.5 <= w <= 2.4 and fill >= 0.45 and not has_vertical:
        sc["tenuto"] = 0.85
    # 긴 선: 쐐기 또는 이음줄
    if w >= 1.2 and not has_vertical:
        two_lines = runs_avg >= 1.5 or max(sp_l, sp_r) >= 2.2 * max(1.0, min(sp_l, sp_r))
        if w >= 2.0 and two_lines and abs(sp_r - sp_l) >= 0.35 * il:
            k = "cresc" if sp_r > sp_l else "decresc"
            sc[k] = 0.9
            sc["decresc" if k == "cresc" else "cresc"] = 0.3
            # 쐐기 자리: 가운데 선 높이와 벌어진 쪽 높이
            geom["hairpin"] = (X0 + bx0, X0 + bx1, cy, float(max(by1 - by0, 0.6 * il)))
        poly = _fit_poly(ink[by0:by1, bx0:bx1], il)
        if poly is not None:
            pts = [(X0 + bx0 + px, Y0 + by0 + py) for px, py in poly]
            geom["poly"] = pts
            ys_p = [p[1] for p in pts]
            base = (ys_p[0] + ys_p[-1]) / 2.0
            sag = max(abs(y - base) for y in ys_p)
            if runs_avg <= 1.4 and (sag >= 0.15 * il or h >= 0.3):
                sc["slur"] = max(sc["slur"], 0.8 if not two_lines else 0.45)
                sc["tie"] = max(sc["tie"], 0.6 if not two_lines else 0.3)
        if "hairpin" not in geom:
            geom["hairpin"] = (X0 + bx0, X0 + bx1, cy, float(max(by1 - by0, 0.6 * il)))
    # [v2.70] 글꼴 모양 비교(악센트·마르카토·스타카티시모·늘임표·임시표·셈여림) — 기둥은 지우지 않은 잉크로
    tink = raw.copy()
    for ys in staff_ys_all:
        for ly in ys:
            r0, r1 = int(ly - 0.1 * il) - Y0, int(ly + 0.1 * il) + 1 - Y0
            if r1 > 0 and r0 < tink.shape[0]:
                tink[max(0, r0):min(tink.shape[0], r1), :] = 0
    for hx, hy, hw, hh in heads:
        c0, c1 = int(hx) - X0, int(hx + hw) - X0
        r0, r1 = int(hy) - Y0, int(hy + hh) - Y0
        if c1 > 0 and r1 > 0 and c0 < tink.shape[1] and r0 < tink.shape[0]:
            tink[max(0, r0):r1, max(0, c0):c1] = 0
    tink = cv2.morphologyEx(tink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(il * 0.3)))))
    ty, tx = np.nonzero(tink)
    if len(tx) >= 6:
        crop = tink[ty.min():ty.max() + 1, tx.min():tx.max() + 1] > 0
        geom["tbox"] = (X0 + tx.min(), Y0 + ty.min(), X0 + tx.max() + 1, Y0 + ty.max() + 1)
        tm = match_templates(crop, il)
        geom["iou"] = tm
        for k, v in tm.items():
            # 겹침 0.65 이상이면 글꼴 모양이 뚜렷이 맞음 → 규칙 판단(0.8~0.9)보다 앞에 둔다
            sc[k] = (0.92 + (v - 0.65) * 0.2) if v >= 0.65 else max(0.02, (v - 0.35) * 2.0)
        # [v2.80] 옥타브: 왼쪽 글자 부분(최대 3.2칸)만
        lw = int(min(crop.shape[1], 3.2 * il))
        left = crop[:, :lw]
        lys = np.nonzero(left.any(axis=1))[0]
        if len(lys):
            left = left[lys.min():lys.max() + 1]
            for k, ch in OTTAVA.items():
                best = 0.0
                for f in (0.75, 0.9, 1.05, 1.2, 1.35):
                    t = _template(ch, 4 * il * f)
                    if t is not None and 0.6 <= t.shape[0] / max(1, left.shape[0]) <= 1.7:
                        best = max(best, _iou_centered(left, t))
                long_ = crop.shape[1] >= 3.5 * il          # 점선까지 칠해야 범위가 생긴다
                sc[k] = ((0.92 + (best - 0.65) * 0.2) if best >= 0.65 else max(0.02, (best - 0.35) * 2.0)) * \
                    (1.0 if long_ else 0.6)
        # [v2.80] 글자 이동 기호(D.C. 등)
        for k, v in match_texts(crop, il).items():
            sc[k] = (0.9 + (v - 0.6) * 0.2) if v >= 0.6 else max(0.02, (v - 0.3) * 1.8)
    # [v2.90] 계단형 페달 선(가로선 + 양 끝 세로 갈고리, 중간 ^ 꼭지 = 다시 밟기)
    if w >= 3.0 and h <= 2.0:
        rows = ink[by0:by1, bx0:bx1].mean(axis=1)
        ly_ = int(np.argmax(rows))
        # 가로선 띠(±2px)가 폭의 85% 이상을 덮는다 — 꼭지 자리만 잠깐 비고, 글자는 글자 사이가 빈다
        band = ink[max(by0, by0 + ly_ - 2):by0 + ly_ + 3, bx0:bx1].any(axis=0)
        if rows.max() >= 0.6 and band.mean() >= 0.85:
            ly = by0 + ly_
            notches = []
            col_top = [(c, np.nonzero(ink[by0:by1, c])[0].min() + by0) for c in range(bx0, bx1) if ink[by0:by1, c].any()]
            run = []
            for c, t in col_top:
                if ly - t >= 0.45 * il:
                    run.append(c)
                elif run:
                    notches.append(X0 + (run[0] + run[-1]) / 2.0)
                    run = []
            if run:
                notches.append(X0 + (run[0] + run[-1]) / 2.0)
            # 양 끝 갈고리 근처의 '꼭지'는 빼고(끝 갈고리 자체가 위로 서 있을 수 있음)
            notches = [x for x in notches if X0 + bx0 + 0.8 * il < x < X0 + bx1 - 0.8 * il]
            sc["ped_line"] = 0.87
            geom["ped_line"] = (X0 + bx0, X0 + bx1, Y0 + ly, notches)
    # [v2.90] 글리산도: 두 음 사이 비스듬한 곧은 선(물결선도 대략 곧은 띠)
    if w >= 1.5 and runs_avg <= 1.4 and not has_vertical:
        cols_ = [(c, float(np.nonzero(ink[by0:by1, c])[0].mean())) for c in range(bx0, bx1) if ink[by0:by1, c].any()]
        if len(cols_) >= 6:
            cx_ = np.array([c for c, _ in cols_], float)
            cy_ = np.array([y for _, y in cols_], float)
            co = np.polyfit(cx_, cy_, 1)
            resid = float(np.mean(np.abs(np.polyval(co, cx_) - cy_))) / il
            # 휨: 양 끝을 잇는 직선에서 가장 먼 거리 — 이음줄은 휘고, 글리산도는 곧다(물결도 고르게 작다)
            chord_ = cy_[0] + (cy_[-1] - cy_[0]) * (cx_ - cx_[0]) / max(1e-6, cx_[-1] - cx_[0])
            bend = float(np.max(np.abs(cy_ - chord_))) / il
            if 0.15 <= abs(co[0]) <= 2.5 and resid <= 0.25 and bend <= 0.35:
                sc["gliss"] = 0.86
                sc["slur"] = min(sc["slur"], 0.5)
                sc["tie"] = min(sc["tie"], 0.4)
                gx0, gx1 = cx_.min(), cx_.max()
                geom["gliss"] = (X0 + gx0, Y0 + by0 + np.polyval(co, gx0), X0 + gx1, Y0 + by0 + np.polyval(co, gx1))
    # [v2.80] 아르페지오: 세로로 길고 좁은 물결선
    if h >= 1.8 and w <= 1.1 and len(xx) >= 10:
        rows_c = [np.mean(xx[yy == r]) for r in np.unique(yy)]
        wav = float(np.std(rows_c)) / il if len(rows_c) > 4 else 0.0
        if 0.06 <= wav <= 0.45:
            sc["arpeggio"] = 0.88
    # [v2.90] 분명한 테누토(납작하고 꽉 찬 짧은 가로선)는 납작한 꾸밈 모양(턴·모르덴트·트릴)과 부분적으로 겹칠 뿐
    if sc.get("tenuto", 0) >= 0.85:
        for k in ("turn", "mordent", "trill"):
            sc[k] = min(sc.get(k, 0.02), 0.8)
    # [v2.80] 글자 줄(따로 떨어진 비슷한 높이의 조각 3개 이상이 가로로) → 선 기호(이음줄·쐐기)는 물러서고
    # 글자 이동 기호를 앞으로, 단일 글꼴 기호는 조금 낮춘다
    if len(tx) >= 6:
        nc_, lab_, st_, _cc = cv2.connectedComponentsWithStats((tink > 0).astype(np.uint8), 8)
        parts = [st_[j] for j in range(1, nc_) if st_[j][4] >= 6]
        if len(parts) >= 3:
            hs_ = sorted(p_[3] for p_ in parts)
            med = hs_[len(hs_) // 2]
            similar = sum(1 for p_ in parts if 0.4 * med <= p_[3] <= 2.2 * med)
            span_w = (tx.max() - tx.min()) / il
            if similar >= 3 and span_w >= 2.0 and (ty.max() - ty.min()) / il <= 2.6:
                geom["texty"] = True                  # [v3.00] 글자 줄 — 후보에 '코드 이름…'을 앞쪽에
                for k in ("slur", "tie", "cresc", "decresc", "tenuto", "ped_line", "gliss"):
                    sc[k] = min(sc.get(k, 0.02), 0.25)
                for k in TEMPLATES:
                    sc[k] = sc.get(k, 0.02) * 0.75
                for k in TEXTS:
                    sc[k] = sc.get(k, 0.02) + 0.15
    if len(tx) >= 6 and not geom.get("texty"):
        th_ = (ty.max() - ty.min()) / il
        if 0.8 <= th_ <= 2.4 and (tx.max() - tx.min()) / il <= 6.0 and max(sc.values()) < 0.85:
            geom["texty"] = True                      # 짧은 글자(코드 'C', 'Am7' 등)일 수 있음
    keys = list(KINDS) + ["arpeggio", "ped_line", "gliss"] + list(TEMPLATES.keys()) + list(OTTAVA.keys()) + \
        list(TEXTS.keys())
    cands = sorted(keys, key=lambda k: -sc.get(k, 0.02))
    return {"cands": [(k, sc.get(k, 0.02)) for k in cands], "geom": geom}
