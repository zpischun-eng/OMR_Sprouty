# tools/regression.py
# ---------------------------------------------------------------------------
# 회귀 검사 하니스: resource/ 의 검증 샘플 4종을 새 엔진으로 인식해
# 조표·박자표·마디 검산 통과율·실행 시간을 표로 출력한다.
#
# 사용법:  python tools/regression.py
#
# 엔진 임계값·검증 로직을 수정할 때마다 반드시 실행해 4개 샘플 전부에서
# 퇴행이 없는지 확인한다(과거 3개 샘플에만 맞춰 조정하다 music.jpg 가
# 퇴행한 사고의 재발 방지 장치).
# ---------------------------------------------------------------------------
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omr import engine, pipeline  # noqa: E402
from score.model import build_from_context  # noqa: E402
from score import editing as ed  # noqa: E402

GOOD = ("OK", "time_change", "anacrusis")

# (이름, 경로, 기대 조표, 기대 박자표, 최소 통과 마디수/전체)
#
# 마디합 자동 교정(리듬 재해석)은 기본 비활성화되어 있다(omr/engine.py
# AUTOFIX_ENABLED). 자동교정이 인접 요소를 잘못 건드리는 부작용이 실전
# 검증에서 반복 관찰되어, 마디 불일치는 배지로만 표시하고 사용자가
# 에디터에서 직접 수정하도록 했다. 아래 기준값은 이 '있는 그대로' 검출
# 결과 기준이며, 종전(자동교정 활성) 기준보다 낮다 — 이는 퇴행이 아니라
# 원본 보존을 우선한 설계 변경의 결과다.
SAMPLES = [
    ("선린교가",   "resource/선린교가.jpeg",              0,   (4, 4), (30, 32)),
    ("guard",      "resource/score_4staves.jpg",           100, (4, 4), (18, 20)),
    ("butterfly",  "resource/score_4staves_butterfly.jpg", 100, (2, 4), (15, 16)),
    # music 은 개발용 테스트 악보(대부분의 마디가 의도적으로 8/4 를 채우지
    # 않음) — 마디 통과율보다 조표(♭4)·박자표(8/4) 정확성이 검사 목적.
    ("music",      "resource/music.jpg",                   400, (8, 4), (1, 16)),
]


def _find(path):
    if os.path.exists(path):
        return path
    if not os.path.isdir(os.path.dirname(path) or "."):
        return path
    # 한글 파일명이 압축 과정에서 손상된 경우 대비: 같은 확장자 파일 탐색
    base = os.path.dirname(path)
    ext = os.path.splitext(path)[1]
    for f in sorted(os.listdir(base)):
        if f.endswith(ext) and not os.path.exists(path):
            cand = os.path.join(base, f)
            if cand not in [s[1] for s in SAMPLES]:
                return cand
    return path


def main():
    total = [0, 0]
    failed = False
    print(f"{'샘플':<12} {'조표':>6} {'박자표':>8} {'마디(엔진)':>10} {'마디(UI)':>10} {'시간':>7}  판정")
    print("-" * 78)
    for name, path, exp_key, exp_ts, (min_good, n_meas) in SAMPLES:
        p = _find(path)
        if not os.path.exists(p):
            # 실제 곡 시험 악보는 저작권 때문에 저장소에 없다 — 각자 PC의 resource/에 두면 검사한다
            print(f"{name:<12} (시험 악보 없음 — 건너뜀: {path})")
            continue
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"{name:<12} 이미지 로드 실패: {p}")
            failed = True
            continue
        t0 = time.time()
        r = engine.run(img)
        dt = time.time() - t0
        good = sum(1 for m in r.measure_results if m[4] in GOOD)
        total[0] += good
        total[1] += len(r.measure_results)

        # UI 가 실제로 쓰는 경로(pipeline→model→editing.validate)로도 검사한다.
        # 엔진 자체 판정과 이 경로가 갈라지면(예: 박자 변화가 인식된 줄을
        # 편집기 검증이 모르는 경우) 사용자 화면에는 틀린 배지가 뜨는데
        # 이 스크립트의 엔진 전용 검사만으로는 절대 못 잡는다 — 실제로
        # 이런 드리프트가 여러 라운드 동안 발견되지 않은 사고가 있었다.
        _res, ctx = pipeline.run(img)
        score = build_from_context(ctx, ctx["barlines"], ctx["barline_types"])
        v = ed.validate(score)
        ui_good = sum(1 for st, _t, _c in v.values() if st == "ok")

        ok = (r.key == exp_key and tuple(r.time_signature) == exp_ts
              and good >= min_good)
        drift = abs(ui_good - good) > 0  # 반올림 경계 등 사소한 차이는 없어야 정상
        if not ok or drift:
            failed = True
        tag = "OK" if ok else f"FAIL(기대 key={exp_key} ts={exp_ts} ≥{min_good})"
        if drift:
            tag += f"  ⚠ UI/엔진 불일치(UI={ui_good} vs 엔진={good}) — 모델 계층 확인 필요"
        print(f"{name:<12} {r.key:>6} {str(r.time_signature):>8} "
              f"{good:>4}/{len(r.measure_results):<3} {ui_good:>4}/{len(v):<3} "
              f"{dt:>6.1f}s  {tag}")
    print("-" * 78)
    print(f"합계(엔진 기준): {total[0]}/{total[1]}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
