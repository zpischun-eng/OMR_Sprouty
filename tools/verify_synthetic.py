# verify_synthetic.py
# ---------------------------------------------------------------------------
# [업그레이드 0단계] 합성 샘플 검증 하니스.
#
# resource/synthetic/<이름>/ 의 score.pdf 를 문서 계층(다중 페이지 포함)으로
# 인식하고, truth.json 의 정답과 비교해 다음을 측정한다:
#   * 조표/박자표 정확 여부
#   * 마디 수 일치 여부
#   * 기호 정확도: (종류, MIDI, 길이) 토큰열의 정렬 일치율(SequenceMatcher)
#   * 음정 정확도: 정렬된 음표 쌍 중 MIDI 일치율
#
# 합성 판각은 엔진 튜닝 표본과 잉크 특성이 다르므로 여기 수치는 '새 판각
# 일반화 성능'의 기준선이다. 수치 자체보다 라운드 간 추이(개선/퇴행)를
# 보는 용도로 쓴다. --update-baseline 로 현재 수치를 기준선 파일에 기록하고,
# 이후 실행에서 기준선 대비 큰 하락이 있으면 실패 종료한다(회귀 감시).
#
# 사용:
#   python tools/verify_synthetic.py [샘플루트] [--update-baseline]
# ---------------------------------------------------------------------------
import difflib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_STEP_SEMI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
TOLERANCE = 0.10        # 기준선 대비 허용 하락폭(기호 정확도)


def _truth_tokens(truth):
    toks = []
    for m in truth["measures"]:
        for t in m:
            if t[0] == "rest":
                toks.append(("rest", None, round(t[1], 3)))
            else:
                _, step, octv, alter, q, _base, _d = t
                midi = 12 * (octv + 1) + _STEP_SEMI[step] + alter
                toks.append(("note", midi, round(q, 3)))
    return toks


def _score_tokens(score):
    from score.midi import element_midi
    toks = []
    for m in score.measures:
        for e in m.elements:
            if getattr(e, "is_grace", False):
                continue
            q = round(e.quarter_length(), 3)
            if e.kind == "rest":
                toks.append(("rest", None, q))
            else:
                toks.append(("note", element_midi(e), q))
    return toks


def evaluate(sample_dir):
    from omr import document
    from score import model as sm

    truth = json.load(open(os.path.join(sample_dir, "truth.json"),
                           encoding="utf-8"))
    pdf = os.path.join(sample_dir, "score.pdf")
    t0 = time.time()
    doc = document.run_document(pdf)
    ctx = doc.ctx
    score = sm.build_from_context(ctx, ctx["barlines"], ctx["barline_types"])
    dt = time.time() - t0

    meta = truth["meta"]
    key_map = {0: 0}
    fif = meta["fifths"]
    exp_key = (100 * -fif) if fif < 0 else (10 * fif)      # 모델 key 인코딩
    got_key = ctx["recognition"]["key"]
    got_ts = tuple(ctx["recognition"]["time_signature"])
    exp_ts = (meta["beats"], meta["beat_type"])

    gt = _truth_tokens(truth)
    rec = _score_tokens(score)
    smx = difflib.SequenceMatcher(a=gt, b=rec, autojunk=False)
    matched = sum(b.size for b in smx.get_matching_blocks())
    sym_acc = matched / max(1, len(gt))

    # 음정 정확도(리듬 무시하고 음표 MIDI 열만 비교)
    gtn = [t[1] for t in gt if t[0] == "note"]
    ren = [t[1] for t in rec if t[0] == "note"]
    smp = difflib.SequenceMatcher(a=gtn, b=ren, autojunk=False)
    pmatch = sum(b.size for b in smp.get_matching_blocks())
    pitch_acc = pmatch / max(1, len(gtn))

    return {
        "name": os.path.basename(sample_dir),
        "pages": f"{len(doc.pages)}/{meta.get('n_pages', 1)}",
        "key_ok": got_key == exp_key,
        "ts_ok": got_ts == exp_ts,
        "meas": f"{len(score.measures)}/{meta['n_measures']}",
        "meas_ok": len(score.measures) == meta["n_measures"],
        "sym_acc": sym_acc,
        "pitch_acc": pitch_acc,
        "n_gt": len(gt),
        "time": dt,
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    update = "--update-baseline" in sys.argv
    root = args[0] if args else "resource/synthetic"
    base_path = os.path.join(root, "baseline.json")
    baseline = {}
    if os.path.exists(base_path):
        baseline = json.load(open(base_path, encoding="utf-8"))

    dirs = sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))
    print(f"{'샘플':<22}{'페이지':>6}{'조표':>5}{'박자':>5}{'마디':>9}"
          f"{'기호':>8}{'음정':>8}{'시간':>7}")
    print("-" * 72)
    results = {}
    failed = []
    for d in dirs:
        sd = os.path.join(root, d)
        if not os.path.exists(os.path.join(sd, "truth.json")):
            continue
        try:
            r = evaluate(sd)
        except Exception as e:
            print(f"{d:<22}  실행 오류: {e}")
            failed.append(d)
            continue
        results[d] = r
        mark = lambda b: "O" if b else "X"
        print(f"{r['name']:<22}{r['pages']:>6}{mark(r['key_ok']):>5}"
              f"{mark(r['ts_ok']):>5}{r['meas']:>9}"
              f"{r['sym_acc']*100:>7.1f}%{r['pitch_acc']*100:>7.1f}%"
              f"{r['time']:>6.1f}s")
        # 기준선 대비 퇴행 감시
        if d in baseline and not update:
            if r["sym_acc"] < baseline[d]["sym_acc"] - TOLERANCE:
                failed.append(f"{d}: 기호 정확도 퇴행 "
                              f"{baseline[d]['sym_acc']:.2f}→{r['sym_acc']:.2f}")
    print("-" * 72)
    if results:
        avg_s = sum(r["sym_acc"] for r in results.values()) / len(results)
        avg_p = sum(r["pitch_acc"] for r in results.values()) / len(results)
        print(f"평균: 기호 {avg_s*100:.1f}% · 음정 {avg_p*100:.1f}% "
              f"({len(results)}개 샘플)")
    if update:
        json.dump({k: {"sym_acc": v["sym_acc"], "pitch_acc": v["pitch_acc"]}
                   for k, v in results.items()},
                  open(base_path, "w", encoding="utf-8"), indent=1)
        print(f"기준선 갱신: {base_path}")
    if failed:
        print("실패:", *failed, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
