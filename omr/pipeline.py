# pipeline.py
# ---------------------------------------------------------------------------
# [방향 2] 구조 우선 인식 파이프라인 — omr/engine.py 를 감싸는 어댑터.
#
# 기존 파이프라인은 '전체 페이지'에 휴리스틱을 적용해, 가사·코드 기호가
# 음악 기호와 섞여 검출되는 근본 문제가 있었다. 새 파이프라인은 엔진이
# 시스템(단) 단위 구조 분석 + Bravura 템플릿 매칭 + 음악 문법 검증으로
# 인식을 끝내고, 여기서는 그 결과를 기존 GUI/모델 계층이 기대하는
# ctx 계약으로 변환만 한다. (기존 구현은 pipeline_legacy.py 로 보존)
#
# ctx 계약(하위 호환 — editor_app_qt.py, score/model.py 가 사용):
#   ctx["image_clean"]    : 오선 제거된 정규화 이진 이미지(표시용·좌표 기준)
#   ctx["staves"]         : 오선 y좌표 평탄 리스트(줄당 5개)
#   ctx["recognition"]    : {key, time_signature, beats, pitches, elements}
#   ctx["barlines"]       : {줄: [마디선 x, ...]}          ← 엔진이 직접 제공
#   ctx["barline_types"]  : {줄: {x: 종류}}                ← 엔진이 직접 제공
#   ctx["measure_info"]   : {barlines, measure_results}    (박자 검산 결과 포함)
#   ctx["slurs"] / ctx["voltas"] / ctx["hairpins"]
#
# results(단계 목록)는 GUI 의 페이지 뷰용 (제목, 이미지) 리스트다.
# ---------------------------------------------------------------------------
import cv2

from . import modules
from . import engine


def _scaled_voltas(original_bgr, eng):
    """볼타(1절/2절) 브래킷 검출 — 원본 배율에서 검출 후 정규화 배율로 변환."""
    staves_raw = [y / eng.scale for y in eng.staves]
    try:
        voltas_raw = modules.detect_voltas(original_bgr, staves_raw)
    except Exception:
        return []
    out = []
    for v in voltas_raw or []:
        v = dict(v)
        for k in ("x1", "x2"):
            if k in v:
                v[k] = v[k] * eng.scale
        out.append(v)
    return out


def run(original_bgr, with_metadata=True):
    """원본 BGR 이미지 → (단계 결과 리스트, ctx).

    Returns
    -------
    tuple[list[tuple[str, np.ndarray]], dict]
        results: [("0. 원본", ...), ("1. ...", ...), ...] — GUI 페이지 뷰용.
        ctx    : 모델 계층이 쓰는 인식 결과 저장소(위 ctx 계약 참조).
    """
    eng = engine.run(original_bgr)

    # ── 이음줄/붙임줄, 하프핀: 기존 검출기를 엔진의 정규화 좌표계에서 재사용.
    #    (두 검출기는 '가늘고 긴 호/쐐기'만 찾으므로 clean 이미지에 그대로 적용.)
    # 이음줄/붙임줄: 엔진의 구조 정보(머리·기둥·마디선)를 활용한 잔여 성분
    # 분석으로 검출한다(레거시 검출기는 새 전처리와 튜닝이 맞지 않는다).
    slurs = list(eng.slurs)
    try:
        _img_hp, hairpins = modules.detect_and_remove_hairpins(
            eng.clean.copy(), eng.staves)
        # 레거시 검출기는 보표 위/아래를 폭넓게 훑는데, 가사(한글) 획의
        # 대각선 스트로크가 쐐기(<>) 모양으로 오검되는 사례가 있다(실측:
        # guard 샘플에서 가사 "그늘 되시니" 부근이 크레센도로 오검됨).
        # 각 줄의 가사 시작선(엔진의 lyric-cut) 아래에 찍힌 쐐기는 버린다.
        hairpins = [
            hp for hp in hairpins
            if 0 <= hp["line"] < len(eng.sys_results)
            and hp["y"] < engine._lyric_cut_y(eng.clean, eng.sys_results[hp["line"]].info)
        ]
    except Exception:
        hairpins = []
    voltas = _scaled_voltas(original_bgr, eng)

    beats = [el[5] for el in eng.elements]
    ctx = {
        "image_0": original_bgr,
        "image_clean": eng.clean,
        "staves": eng.staves,
        "recognition": {
            "key": eng.key,
            "time_signature": eng.time_signature,
        "time_symbol": eng.ts_symbol,
            "beats": beats,
            "pitches": eng.pitches,
            "elements": eng.elements,
        },
        "barlines": eng.barlines,
        "barline_types": eng.barline_types,
        "measure_info": {
            "barlines": eng.barlines,
            "measure_results": eng.measure_results,
            "sys_capacity": eng.sys_capacity,
        },
        "slurs": slurs,
        "voltas": voltas,
        "hairpins": hairpins,
        "engine_result": eng,
    }

    # [4단계] 곡 제목·작사/작곡 크레딧 — 선택 기능. 실패해도(Tesseract
    # 미설치 등) 나머지 인식 결과에 영향을 주면 안 되므로 안전하게 감싼다.
    # (※ 가사 OCR 은 인식률이 낮아[실측: 50% 미만, 일부 줄 완전 누락]
    #   기능 자체를 제거했다 — omr/lyrics.py 삭제됨. 제목/크레딧 인식은
    #   대상 텍스트가 짧고 큼직해 훨씬 안정적이라 유지한다.)
    if with_metadata:
        try:
            from . import metadata as metadata_mod
            ctx["title_credits"] = metadata_mod.extract_title_credits(ctx)
        except Exception:
            ctx["title_credits"] = {"title": None, "credits": None}

    overlay = engine.draw_overlay(eng)
    norm_vis = cv2.cvtColor(255 - eng.norm_bin, cv2.COLOR_GRAY2BGR)
    clean_vis = cv2.cvtColor(255 - eng.clean, cv2.COLOR_GRAY2BGR)
    results = [
        ("0. 원본 이미지", original_bgr),
        ("1. 이진화·정규화(오선 간격 10px)", norm_vis),
        ("2. 오선 제거(시스템 단위)", clean_vis),
        ("3. 인식 결과(음표·쉼표·마디선·마디 검산)", overlay),
    ]
    return results, ctx
