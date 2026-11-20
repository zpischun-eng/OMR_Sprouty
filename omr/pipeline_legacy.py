# pipeline.py
# ---------------------------------------------------------------------------
# 악보 인식 '단계'를 순서대로 실행하고, 각 단계의 결과 이미지를
# (제목, 이미지) 형태로 보관하는 파이프라인.
#
# GUI(app.py)는 이 파이프라인이 만든 단계 리스트를 그대로 받아
# 우측 화면에서 페이지를 넘겨가며 볼 수 있다.
#
# ── 설계 (확장 규칙) ────────────────────────────────────────────────
# 각 단계는 dict 로 정의한다.
#   {
#     "title":    페이지에 표시할 제목,
#     "fn":       ctx(dict)를 받아 결과를 반환하는 함수.
#                 반환값은 이미지, 또는 (이미지, 부가데이터) 튜플.
#     "save_as":  (선택) 결과 이미지를 ctx 에 이 이름으로 저장 → 다음 단계가 참조.
#     "extra_key":(선택) 튜플 반환 시 두 번째 값을 ctx 에 이 이름으로 저장.
#   }
#
# ctx(context) 는 단계 간에 공유되는 저장소다.
#   - 시작 시 ctx["image_0"] = 원본 BGR 이미지.
#   - 각 단계는 필요한 이전 결과를 ctx["image_1"] 처럼 이름으로 명시해 가져온다.
#   - 이 방식 덕분에 '시각화 전용 단계'(예: 히스토그램)를 중간에 끼워 넣어도
#     다음 단계의 입력 연결이 꼬이지 않는다(각 단계가 자기 입력을 직접 지정).
#
# 새 단계 추가 = STEPS 리스트에 dict 한 개 추가. GUI 는 수정할 필요 없다.
# ---------------------------------------------------------------------------

import cv2
from . import modules


def _run_recognition(ctx):
    """6~13단계 recognition 을 실행하고 (이미지, 부가정보 dict)로 정규화해 반환한다.

    recognition 은 (image, key, time_signature, beats, pitches, elements) 를 반환하므로,
    파이프라인의 (이미지, 부가데이터) 2-튜플 규약에 맞게 묶어 준다.
    """
    image, key, time_sig, beats, pitches, elements = modules.recognition(
        ctx["image_clean"], ctx["staves"], ctx["objects"])
    return image, {"key": key, "time_signature": time_sig, "beats": beats,
                   "pitches": pitches, "elements": elements}


def _run_voltas(ctx):
    """[2-B] 볼타(1절/2절 등 엔딩) 브래킷을 검출한다.

    반드시 '오선 제거' 직후(정규화 이전)에 실행한다. 이 시점의 ctx["staves"]는
    아직 원본 배율이라, detect_voltas()가 요구하는 좌표계와 일치한다.
    브래킷 자체는 remove_noise/remove_staves 단계에서 이미 파괴되므로,
    detect_voltas는 ctx["image_0"](가공 전 원본)을 독자적으로 다시 이진화해
    검출한다. 검출 결과는 원본 배율 좌표이므로, 정규화(다음 단계)가 쓸 배율과
    동일한 식으로 변환해 두어야 이후 단계(마디선 매칭 등)와 좌표가 맞는다.
    """
    staves_raw = ctx["staves"]
    voltas_raw = modules.detect_voltas(ctx["image_0"], staves_raw)

    # normalization()과 같은 방식으로 배율 계산(오선 한 칸 평균 간격 → 10px 기준)
    lines = len(staves_raw) // 5
    total = 0.0
    for line in range(lines):
        for i in range(4):
            total += abs(staves_raw[line * 5 + i] - staves_raw[line * 5 + i + 1])
    avg_distance = (total / (len(staves_raw) - lines)) if lines else 1.0
    scale = 10.0 / avg_distance if avg_distance else 1.0

    voltas = [{"line": v["line"], "x1": v["x1"] * scale, "x2": v["x2"] * scale,
              "number": v["number"]} for v in voltas_raw]

    # 시각화: 검출 위치를 원본 위에 표시(다음 단계 입력에는 영향 없음)
    vis = cv2.cvtColor(ctx["image_2"], cv2.COLOR_GRAY2BGR) if ctx["image_2"].ndim == 2 \
        else ctx["image_2"].copy()
    for v in voltas_raw:
        y = int(round(staves_raw[v["line"] * 5])) - 4
        cv2.rectangle(vis, (v["x1"], max(0, y - 14)), (v["x2"], y), (0, 140, 255), 2)
        cv2.putText(vis, str(v["number"]), (v["x1"], max(10, y - 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
    return vis, voltas


def _run_measure_validation(ctx):
    """[14단계] 마디선을 검출해 마디를 나누고, 각 마디 박자 합을 검산해 색칠한다."""
    barlines = modules.detect_barlines(ctx["image_clean"], ctx["staves"])
    rec = ctx["recognition"]
    vis = ctx["image_recognition"].copy()   # 인식 결과 이미지 위에 색칠
    vis, results = modules.measure_validation(
        vis, ctx["staves"], rec["elements"], barlines, rec["time_signature"])
    return vis, {"barlines": barlines, "measure_results": results}


STEPS = [
    {
        "title": "1. 보표 영역 추출 및 노이즈 제거",
        "fn": lambda ctx: modules.remove_noise(ctx["image_0"]),
        "save_as": "image_1",
    },
    {
        # 시각화 전용 단계: 오선 검출 원리를 보여주는 수평 히스토그램.
        # save_as 가 없으므로 이후 단계 입력에는 영향을 주지 않는다.
        "title": "2-A. 수평 히스토그램 (오선 검출 원리)",
        "fn": lambda ctx: modules.get_horizontal_histogram(ctx["image_1"]),
    },
    {
        "title": "2. 오선 제거",
        "fn": lambda ctx: modules.remove_staves(ctx["image_1"]),
        "save_as": "image_2",
        "extra_key": "staves",   # 검출된 오선 y좌표 리스트를 ctx["staves"]에 저장
    },
    {
        # 볼타(1절/2절) 브래킷은 remove_noise/remove_staves 단계에서 파괴되므로,
        # 그 전에(이 시점의 ctx["staves"]는 아직 원본 배율) 원본을 따로 다시
        # 이진화해 검출해 둔다. 검출 결과는 정규화 배율로 변환해 저장하므로
        # 이후 단계(마디선·모델)와 좌표계가 맞는다.
        "title": "2-B. 볼타(1절/2절 브래킷) 검출",
        "fn": _run_voltas,
        "extra_key": "voltas",   # [{"line":.., "x1":.., "x2":.., "number":..}, ...]
    },
    {
        "title": "3. 악보 이미지 정규화",
        "fn": lambda ctx: modules.normalization(ctx["image_2"], ctx["staves"]),
        "save_as": "image_3",
        "extra_key": "staves",   # 정규화된 오선 좌표로 갱신(다음 단계는 이 값을 사용)
    },
    {
        "title": "3-B. 잡티(작은 객체) 제거",
        "fn": lambda ctx: modules.remove_small_objects(ctx["image_3"], min_area=5),
        "save_as": "image_clean",
        "extra_key": "removed_noise",  # 제거한 잡티 개수를 ctx["removed_noise"]에 저장
    },
    {
        # 이음줄이 여러 음표(심지어 마디를 넘어)를 한 연결요소로 병합해 인식을
        # 무너뜨리는 것을 막기 위해, 호를 검출해 이미지에서 분리하고 끝점만
        # 기록한다. 기록된 호는 모델 단계에서 붙임줄/이음줄로 복원된다.
        #
        # ※ 반드시 '코드 제거'보다 먼저 실행한다. 높은 음의 이음줄은 첫 오선
        #    위로 올라가므로, 코드 제거를 먼저 하면 이음줄이 '오선 위에 떠 있는
        #    기호'로 오인돼 통째로 지워진다(실측 확인).
        "title": "3-C. 이음줄(슬러/타이) 검출·분리",
        "fn": lambda ctx: modules.detect_and_remove_slurs(ctx["image_clean"], ctx["staves"]),
        "save_as": "image_clean",       # 이후 단계는 호가 분리된 이미지를 사용
        "extra_key": "slurs",           # 검출된 호 리스트 (모델 단계에서 사용)
    },
    {
        # 크레센도(<)·디크레센도(>) 쐐기도 이음줄과 같은 이유로 미리 지운다.
        # 지우지 않으면 쐐기 선이 인접 음표·가사와 병합돼 객체 검출을 무너뜨릴
        # 수 있다. 좌표만 기록해 두고, 모델 단계에서 다이내믹스로 복원한다.
        "title": "3-C2. 크레센도·디크레센도(하프핀) 검출·분리",
        "fn": lambda ctx: modules.detect_and_remove_hairpins(ctx["image_clean"], ctx["staves"]),
        "save_as": "image_clean",
        "extra_key": "hairpins",   # [{"line":.., "x1":.., "x2":.., "y":.., "kind":"crescendo"|"decrescendo"}, ...]
    },
    {
        # 기타 코드(C, G7 …)가 closing 에서 음표와 병합돼 높은 음으로 오인식되는
        # 것을 막기 위해, 객체 검출 '이전에' 오선 위쪽의 분리된 기호를 지운다.
        "title": "3-D. 기타 코드(화음 기호) 제거",
        "fn": lambda ctx: modules.remove_chord_symbols(ctx["image_clean"], ctx["staves"]),
        "save_as": "image_clean",       # 이후 단계는 코드가 지워진 이미지를 사용
        "extra_key": "removed_chords",  # 제거된 코드 기호 bbox 리스트
    },
    {
        "title": "4. 객체 검출",
        "fn": lambda ctx: modules.object_detection(ctx["image_clean"], ctx["staves"]),
        "save_as": "image_objects",    # 박스가 그려진 시각화(다음 단계 입력으로는 쓰지 않음)
        "extra_key": "objects",        # 검출된 객체 리스트를 ctx["objects"]에 저장
    },
    {
        "title": "5. 객체 분석 (기둥·방향)",
        # 기둥 검출은 반드시 깨끗한 1채널(image_clean)에 대해 수행한다.
        "fn": lambda ctx: modules.object_analysis(ctx["image_clean"], ctx["objects"]),
        "save_as": "image_analysis",   # 기둥/개수를 표시한 시각화
        "extra_key": "objects",        # 특징이 추가된 objects로 갱신
    },
    {
        "title": "6~13. 인식 (조표·박자표·음표·음정·쉼표·온음표)",
        "fn": _run_recognition,        # (image, {key, time_signature, beats, pitches, elements}) 반환
        "save_as": "image_recognition",
        "extra_key": "recognition",    # ctx["recognition"] = {"key":.., "time_signature":.., ...}
    },
    {
        "title": "14. 마디 박자 검증 (초과=빨강 / 부족=파랑)",
        "fn": _run_measure_validation,
        "save_as": "image_measure",
        "extra_key": "measure_info",   # ctx["measure_info"] = {"barlines":.., "measure_results":..}
    },
    # 다음 단계(음표/쉼표 인식)는 여기에 추가한다. 입력은 ctx["image_clean"] + ctx["objects"]
    # (각 obj = [보표번호, (x,y,w,h,area), stems, direction]). 예)
    # {"title": "7. ...", ...},
]


def run(original_bgr):
    """원본 BGR 이미지를 받아 모든 단계를 순서대로 실행한다.

    Returns
    -------
    tuple[list[tuple[str, np.ndarray]], dict]
        - results: [("0. 원본", 원본이미지), ("1. ...", 결과이미지), ...]
          0번은 항상 원본, 이후 각 단계 결과가 이어진다(페이징용).
        - ctx: 단계 간 공유된 context. 부가 데이터(예: ctx["staves"])를 담는다.
    """
    ctx = {"image_0": original_bgr}
    results = [("0. 원본 이미지", original_bgr)]

    for step in STEPS:
        out = step["fn"](ctx)

        # 반환값 정규화: (이미지, 부가데이터) 튜플이면 분리, 아니면 이미지 단독
        if isinstance(out, tuple):
            image, extra = out[0], out[1]
            if step.get("extra_key") is not None:
                ctx[step["extra_key"]] = extra
        else:
            image = out

        if step.get("save_as") is not None:
            ctx[step["save_as"]] = image

        results.append((step["title"], image))

    return results, ctx
