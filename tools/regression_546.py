#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[4단계] 546개(545개 실사 샘플 + 선린교가) 자동 인식 단계 회귀 검증.

HYBRID_OMR_PLAN_v6.md 7장 설계:
- "100% 인식률" = 편집기를 통한 보정을 포함한 최종 목표(7.2절).
- 이 스크립트는 그 중 "1) 자동 인식 단계 채점"만 수행한다(참고 지표).
  "2) 편집기 완성 단계 채점"은 사람이 편집기로 보정하는 과정을 포함하므로
  전부 자동화하기 어렵고, 이번 라운드 범위 밖이다(7.3절에 이미 명시됨).

이미 이전 대화에서 Audiveris로 545개를 처리해둔 .mxl/.omr을 그대로
재사용한다(재실행 없음 — 수 시간짜리 배치를 다시 돌리지 않기 위함).
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omr_bridge.merge import build_ctx
from score.model import build_score_model
import score.editing as ed

MXL_DIR = Path("/home/claude/omr_output/musicxml")
SELINGYOGA_OMR = "/home/claude/audiveris_compare/선린교가.omr"
SELINGYOGA_MXL = "/home/claude/audiveris_compare/선린교가.mxl"
OUT_JSON = Path(__file__).resolve().parent.parent / "baseline_546.json"


def analyze_one(name: str, omr_path: str, mxl_path: str, upscale=1.0,
                w=2000, h=2800):
    t0 = time.time()
    result = {"file": name, "status": "ok"}
    try:
        merge_result = build_ctx(omr_path, mxl_path, w, h, upscale_factor=upscale)
        ctx = merge_result.ctx
        rec = ctx["recognition"]
        if not rec["elements"]:
            result["status"] = "parse_error"
            result["error"] = "요소 0개(빈 mxl 등)"
            return result
        score = build_score_model(
            rec["key"], rec["time_signature"], rec["elements"], rec["pitches"],
            ctx["barlines"], ctx["barline_types"],
            sys_capacity=ctx["measure_info"]["sys_capacity"],
        )
        status_map = ed.validate(score)
        n_total = len(status_map)
        n_ok = sum(1 for st, _, _ in status_map.values() if st == "ok")
        result.update({
            "n_measures": n_total,
            "n_ok": n_ok,
            "n_merge_warnings": len(merge_result.warnings),
            "elapsed_sec": round(time.time() - t0, 2),
        })
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    return result


def main():
    mxl_files = sorted(MXL_DIR.glob("*.mxl"))
    print(f"545개 세트: {len(mxl_files)}개 파일 발견")

    results = []
    for i, mxl in enumerate(mxl_files):
        omr = mxl.with_suffix(".omr")
        if not omr.exists():
            results.append({"file": mxl.stem, "status": "parse_error",
                           "error": ".omr 파일 없음"})
            continue
        r = analyze_one(mxl.stem, str(omr), str(mxl))
        results.append(r)
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(mxl_files)}")

    # 선린교가 추가(546번째)
    r = analyze_one("선린교가", SELINGYOGA_OMR, SELINGYOGA_MXL, upscale=4.0,
                    w=1136, h=944)
    results.append(r)

    n_total = len(results)
    n_parse_error = sum(1 for r in results if r["status"] in ("parse_error", "error"))
    n_perfect = sum(1 for r in results if r["status"] == "ok" and r.get("n_ok") == r.get("n_measures"))
    n_partial = n_total - n_parse_error - n_perfect

    print("\n=== 546개 자동 인식 단계 회귀 결과 ===")
    print(f"전체: {n_total}")
    print(f"완전 정합(모든 마디 OK): {n_perfect} ({n_perfect/n_total*100:.1f}%)")
    print(f"일부 마디 불일치: {n_partial} ({n_partial/n_total*100:.1f}%)")
    print(f"처리 실패(파싱 오류 등): {n_parse_error} ({n_parse_error/n_total*100:.1f}%)")

    OUT_JSON.write_text(json.dumps({
        "summary": {"total": n_total, "perfect": n_perfect,
                   "partial": n_partial, "parse_error": n_parse_error},
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {OUT_JSON}")

    print("\n=== 처리 실패 목록 ===")
    for r in results:
        if r["status"] in ("parse_error", "error"):
            print(f"  {r['file']}: {r.get('error')}")


if __name__ == "__main__":
    main()
