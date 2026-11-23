# omr_bridge/ink_prep.py
# ---------------------------------------------------------------------------
# [v3.80] 색으로 인쇄된 악보를 Audiveris에 넣기 전에 '잉크는 검게' 바꾼다.
# 그냥 회색으로 바꾸면 노랑·연두·주황 음표 머리가 거의 흰색이 되어 사라지고(붐웨커 색음표 악보: 머리의 절반을
# 놓침), 빨간 잉크 악보는 오선이 흐려져 "오선을 찾지 못함"으로 실패했다(성가 183번).
#   · 색이 있는 화소(가장 밝은 채널 − 가장 어두운 채널이 큰 화소)는 가장 어두운 채널 값을 쓴다
#     (노랑 (255,220,0) → 0 = 검정).
#   · 단, 넓게 칠한 색 면(추임새 말풍선·강조 상자·바탕색)은 그대로 둔다 — 검은 덩어리가 되면 기호로 오인된다.
#     '굵기'로 가른다: 반지름 (그림 폭/110)짜리 침식에서 살아남는 색 덩어리 = 넓은 면.
# 원본 그림(화면 표시용)은 건드리지 않고, 인식에 넣을 사본만 만든다.
# ---------------------------------------------------------------------------
import cv2
import numpy as np


def has_color(rgb, min_frac=0.0005):
    """색 화소가 의미 있게 있는가(회색 스캔의 미세한 색 번짐은 무시)."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return False
    small = rgb[::3, ::3, :3].astype(np.int16)
    sat = small.max(axis=2) - small.min(axis=2)
    return float((sat > 70).mean()) >= min_frac


def ink_gray(rgb):
    """RGB(또는 BGR — 채널 순서는 상관없다) 그림 → 인식용 회색 그림(uint8)."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return rgb if rgb.ndim == 2 else rgb[:, :, 0]
    c = rgb[:, :, :3]
    mx, mn = c.max(axis=2), c.min(axis=2)
    luma = (0.299 * c[:, :, 0] + 0.587 * c[:, :, 1] + 0.114 * c[:, :, 2]).astype(np.uint8)
    colored = ((mx.astype(np.int16) - mn) > 70).astype(np.uint8)
    if not colored.any():
        return luma
    r = max(6, int(round(c.shape[1] / 110.0)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    core = cv2.erode(colored, k)
    if core.any():
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (r + 3) + 1, 2 * (r + 3) + 1))
        big = cv2.dilate(core, k2) & colored            # 넓은 색 면(말풍선·상자)
    else:
        big = np.zeros_like(colored)
    thin = (colored == 1) & (big == 0)
    out = luma.copy()
    out[thin] = np.minimum(luma, mn)[thin]
    return out


def prepare_for_omr(image_path, work_dir):
    """색 있는 그림이면 인식용 사본 경로를, 아니면 원래 경로를 돌려준다."""
    from pathlib import Path
    try:
        buf = np.fromfile(str(image_path), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None or not has_color(img):
            return str(image_path)
        g = ink_gray(img)
        out = Path(work_dir) / f"ink_{Path(image_path).stem}.png"
        ok, enc = cv2.imencode(".png", g)
        if not ok:
            return str(image_path)
        with open(out, "wb") as f:
            f.write(enc.tobytes())
        return str(out)
    except Exception:
        return str(image_path)
