# tools/compare_render.py
# ---------------------------------------------------------------------------
# [검증 도구] 화면 없이 '원본 ↔ 편집 악보' 비교 이미지를 만든다(편집기의 불러오기 경로와 같은
# 순서: 인식 → Score → 유령 쉼표 정리 → 가사 → 배치 → 그리기).
#   python tools/compare_render.py 악보.pdf|png 출력.png [--crop x0,y0,x1,y1] [--F 2]
# crop은 원본 픽셀 좌표. 위(원본)·아래(편집 악보)로 붙인다.
# ---------------------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from omr_bridge.document_adapter import run_document  # noqa: E402
from score import model as score_model  # noqa: E402
from score import editing as ed  # noqa: E402
from score import layout as sl  # noqa: E402
from score import render as sr  # noqa: E402


def build(path):
    """편집기와 같은 처리(page_builder)로 한 페이지를 만든다."""
    from omr_bridge.page_builder import build_page
    doc = run_document(path)
    pm = build_page(doc, sheet=1)
    lay_args = dict(staves_by_system=pm.by_sys, sx=pm.sx, sy=pm.sy, barlines=pm.barlines,
                    staff_groups=pm.staff_groups, staff_left_by_line=pm.staff_left,
                    staff_clefs=pm.staff_clefs, staff_parts=pm.staff_parts)
    pm.score.load_notes = pm.notes
    return doc, pm.score, lay_args, pm.header_items


def render_score(doc, score, lay_args, header, F=2):
    lay = sl.layout_score_world(score, **lay_args)
    lay.header_items = header
    oh, ow = doc.stacked_bgr.shape[:2]
    img, lm = sr.render(lay, ow, oh, F=F, return_margin=True)
    return lay, img, lm


def compare(doc, img, lm, F, out, crop=None):
    orig = Image.fromarray(cv2.cvtColor(doc.stacked_bgr, cv2.COLOR_BGR2RGB))
    oh, ow = doc.stacked_bgr.shape[:2]
    ren = img.crop((lm, 0, lm + int(ow * F), int(oh * F))).resize((ow, oh), Image.LANCZOS)
    if crop:
        x0, y0, x1, y1 = crop
        orig, ren = orig.crop(crop), ren.crop(crop)
        ow, oh = x1 - x0, y1 - y0
    canvas = Image.new("RGB", (ow, oh * 2 + 10), (200, 200, 200))
    canvas.paste(orig, (0, 0))
    canvas.paste(ren, (0, oh + 10))
    canvas.save(out)
    return out


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    crop = None
    F = 2
    if "--crop" in sys.argv:
        crop = tuple(int(v) for v in sys.argv[sys.argv.index("--crop") + 1].split(","))
    if "--F" in sys.argv:
        F = int(sys.argv[sys.argv.index("--F") + 1])
    doc, score, la, hd = build(src)
    lay, img, lm = render_score(doc, score, la, hd, F)
    print(compare(doc, img, lm, F, out, crop))
    print("\n".join(doc.warnings))
