# omr_bridge/header_layout.py
# ---------------------------------------------------------------------------
# [v2.25 곡 정보 자리 — 실사용] 스캔 악보는 글자 인식 결과가 없어, '곡 정보'로 직접 적은 제목·작사작곡·
# 빠르기말이 정해진 기본 자리에 그려져 원본과 많이 달랐다(제목이 아래로, 작사·작곡·편곡이 한 줄로).
# 원본 그림의 첫 줄(시스템) 위쪽에서 글자 줄 덩어리를 찾아 자리·크기를 정한다:
#   · 제목: 페이지 가운데 근처에서 가장 큰 덩어리
#   · 작사/작곡: 오른쪽에 붙은 덩어리들(위에서 아래로 줄마다)
#   · 빠르기말: 왼쪽에서 오선 바로 위에 있는 덩어리
# 반환 상자는 (x0, y0, x1, y1) — 그 그림의 픽셀 좌표.
# ---------------------------------------------------------------------------
import cv2
import numpy as np


def find_header_blocks(img_bgr, staff_top, x_left, x_right, il):
    if img_bgr is None or staff_top is None or il <= 0:
        return {}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    y_end = int(staff_top - 2.0 * il)          # 첫 줄 오선 위 음표(덧줄 등)는 피한다
    if y_end <= il:
        return {}
    reg = (gray[:y_end, :] < 150).astype(np.uint8)
    # 글자를 줄 단위로 묶기: 가로로 넓게, 세로로 조금 이어 붙인다
    kx = max(3, int(il * 1.2))
    ky = max(1, int(il * 0.25))
    merged = cv2.dilate(reg, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
    n, _lab, st, _c = cv2.connectedComponentsWithStats(merged, 8)
    lines = []
    for j in range(1, n):
        x, y, w, h, area = st[j]
        # 묶을 때 넓힌 만큼 되돌린 실제 잉크 상자
        sub = reg[y:y + h, x:x + w]
        ys, xs = np.where(sub > 0)
        if len(xs) < 10:
            continue
        x0, x1 = x + xs.min(), x + xs.max() + 1
        y0, y1 = y + ys.min(), y + ys.max() + 1
        hh, ww = y1 - y0, x1 - x0
        if hh < 0.5 * il or hh > 7 * il or ww < 0.8 * il:
            continue
        lines.append((int(x0), int(y0), int(x1), int(y1)))
    if not lines:
        return {}
    # 같은 줄의 조각(글자 사이가 넓은 제목 "교   회", 낱말로 갈라진 작사·작곡 줄)을 한 줄로 합친다
    merged_any = True
    while merged_any:
        merged_any = False
        lines.sort()
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i], lines[j]
                ov = min(a[3], b[3]) - max(a[1], b[1])
                hmin, hmax = min(a[3] - a[1], b[3] - b[1]), max(a[3] - a[1], b[3] - b[1])
                gap = max(a[0], b[0]) - min(a[2], b[2])
                if ov >= 0.5 * hmin and gap <= 1.5 * hmax:
                    lines[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    del lines[j]
                    merged_any = True
                    break
            if merged_any:
                break
    span = max(1.0, x_right - x_left)
    cx_page = (x_left + x_right) / 2.0
    out = {}
    centered = [b for b in lines if abs((b[0] + b[2]) / 2.0 - cx_page) < 0.12 * span]
    if centered:
        out["title"] = max(centered, key=lambda b: (b[3] - b[1], -b[1]))
    rest = [b for b in lines if b is not out.get("title")]
    credit = sorted([b for b in rest if b[2] > x_left + 0.75 * span and b[0] > x_left + 0.45 * span],
                    key=lambda b: b[1])
    if credit:
        out["credit"] = credit
    # 빠르기말은 글자 줄 — 넉넉히 길다(오선 위 이음줄·음표 조각 제외)
    # 빠르기말: 왼쪽 글자 줄 중 오선에 가장 가까운 것. 첫 줄 오선 위의 음표·빔·이음줄 덩어리는 오선에서
    # 2칸 남짓까지 올라오고, 빠르기말 줄은 4칸 이상 떨어져 있다 → 오선 위 3칸보다 높이 끝나는 줄만 본다.
    # (높이로는 못 가른다: "교회" 빠르기말은 괄호·♩ 때문에 2.8칸으로 음표 덩어리와 비슷)
    left = [b for b in rest if b[0] < x_left + 0.4 * span and b not in credit
            and (b[2] - b[0]) >= 2.0 * il and b[3] <= staff_top - 3.0 * il]
    if left:
        out["tempo"] = max(left, key=lambda b: b[3])
    return out
