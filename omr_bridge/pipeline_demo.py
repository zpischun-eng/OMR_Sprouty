#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
omr_bridge 1단계 검증용 통합 실행 스크립트.

원본 이미지/PDF 한 장 -> Audiveris 실행 -> .omr+.mxl 결합 -> 기존
score/ 렌더러로 그리기 -> 원본과 겹쳐서 비교 이미지까지 한 번에 생성한다.

사용법:
    python3 omr_bridge/pipeline_demo.py <원본이미지경로> <출력디렉토리>

예:
    python3 omr_bridge/pipeline_demo.py resource/선린교가.jpeg /tmp/demo_out
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omr_bridge.audiveris_runner import run_single
from omr_bridge.merge import build_ctx
from score.model import build_score_model
import score.layout as sl
import score.render as sr
from PIL import Image
import numpy as np


def run_pipeline(source_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Audiveris 실행 중: {source_path}")
    res = run_single(source_path, str(out))
    if res.status != "ok":
        print(f"  -> 실패: {res.error} (로그: {res.log_path})")
        return None
    print(f"  -> 성공. mxl={res.mxl_path}, omr={res.omr_path}, "
          f"업스케일={res.upscale_factor}x")

    print("[2/4] 좌표+음악정보 결합 중...")
    im = Image.open(source_path) if not source_path.lower().endswith(".pdf") else None
    if im is not None:
        ow, oh = im.size
    else:
        # PDF는 원본 렌더 크기를 알기 위해 Audiveris 로그나 별도 방법이
        # 필요하다 — 1단계 MVP는 이미지 입력을 우선 검증 대상으로 한다.
        raise NotImplementedError("1단계 MVP는 이미지 입력만 지원(PDF는 후속 과제)")

    merge_result = build_ctx(res.omr_path, res.mxl_path, ow, oh, res.upscale_factor)
    print(f"  -> 요소 {len(merge_result.ctx['recognition']['elements'])}개, "
          f"경고 {len(merge_result.warnings)}개")
    for w in merge_result.warnings[:10]:
        print(f"     ! {w}")

    print("[3/4] 기존 score/ 렌더러로 그리기 중...")
    ctx = merge_result.ctx
    rec = ctx["recognition"]
    score = build_score_model(
        rec["key"], rec["time_signature"], rec["elements"], rec["pitches"],
        ctx["barlines"], ctx["barline_types"],
        ctx.get("slurs"), ctx.get("voltas"), ctx.get("hairpins"),
    )
    n_ok = sum(1 for m in score.measures if m.quarter_sum(score.capacity()) == score.capacity())
    print(f"  -> 마디 {len(score.measures)}개 중 박자 정합 {n_ok}개")

    staves = ctx["staves"]
    by_sys = [staves[i * 5:(i + 1) * 5] for i in range(len(staves) // 5)]
    sx = sy = 1.0 / res.upscale_factor
    layout_obj = sl.layout_score_world(score, by_sys, sx, sy, ctx["barlines"])
    rendered = sr.render(layout_obj, ow, oh, F=3)

    rendered_path = out / f"{Path(res.mxl_path).stem}_rendered.png"
    rendered.save(rendered_path)
    print(f"  -> 렌더 결과 저장: {rendered_path}")

    print("[4/4] 원본-렌더 오버레이 비교 이미지 생성 중...")
    orig_gray = np.array(im.convert("L"))
    rend_gray = np.array(rendered.convert("L").resize((ow, oh), Image.LANCZOS))
    overlay = np.ones((oh, ow, 3), dtype=np.uint8) * 255
    overlay[..., 0] = orig_gray
    overlay[..., 1] = rend_gray
    overlay[..., 2] = np.minimum(orig_gray, rend_gray)
    overlay_path = out / f"{Path(res.mxl_path).stem}_overlay.png"
    Image.fromarray(overlay).save(overlay_path)
    print(f"  -> 오버레이 저장: {overlay_path}")

    return {
        "mxl": res.mxl_path, "omr": res.omr_path,
        "rendered": str(rendered_path), "overlay": str(overlay_path),
        "n_measures": len(score.measures), "n_measures_ok": n_ok,
        "n_warnings": len(merge_result.warnings),
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    result = run_pipeline(sys.argv[1], sys.argv[2])
    print("\n=== 결과 요약 ===")
    print(result)
