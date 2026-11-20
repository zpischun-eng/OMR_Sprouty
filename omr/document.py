# document.py
# ---------------------------------------------------------------------------
# [업그레이드 1단계] 다중 페이지 문서 계층 — PDF/이미지 → 병합된 단일 ctx.
#
# 설계 원칙 (PLAN.md 7장):
#   * 인식 엔진은 손대지 않는다. 페이지마다 pipeline.run 을 독립 실행한 뒤,
#     결과를 "세로로 쌓인 하나의 가상 악보"로 병합한다.
#   * 좌표계 통일: 페이지마다 정규화 배율이 달라도, 각 페이지 원본을
#     '정규화 크기 × 공통 배율 S' 로 리샘플해 쌓으면 편집기의 전역
#     sx=sy=S 가정이 그대로 유지된다 → editor_app_qt / layout / render /
#     hit_test 의 좌표 코드를 전혀 바꾸지 않아도 된다.
#   * 병합 규칙: 단(line) 번호·beam_group 오프셋, y좌표 시프트, dict 키 시프트.
#   * "페이지 범위 = 곡 하나" — page_range 인자로 모음집 확장(향후)을 대비.
#
# 공개 API:
#   load_pages(path, dpi=...)        → [BGR ndarray, ...]
#   run_document(pages|path, ...)    → DocumentResult(stacked_bgr, ctx, pages)
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import pipeline

# 페이지 사이 시각 간격(정규화 좌표 px). 오선 간격 10px 기준 4칸 정도로,
# 시스템 간 간격보다 뚜렷이 넓어 페이지 경계가 눈에 보인다.
PAGE_GAP_NORM = 60

# PDF 렌더 목표 폭(px). 기존 샘플들(1200~2500px)과 같은 대역이 되도록.
PDF_TARGET_WIDTH = 2000
PDF_MAX_PAGES = 12          # 안전 상한(목표는 2~6페이지 단일 곡)


def is_pdf(path: str) -> bool:
    return str(path).lower().endswith(".pdf")


def load_pages(path: str, target_width: int = PDF_TARGET_WIDTH,
               page_range: tuple | None = None):
    """파일 → 페이지 BGR 이미지 리스트.

    이미지 파일이면 1페이지 리스트, PDF 면 각 페이지를 target_width 폭으로
    렌더링한다. page_range=(start, end)  (1부터, end 포함) 로 부분 범위를
    열 수 있다 — '여러 곡 모음집' 지원(향후)의 기반.
    """
    if not is_pdf(path):
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"이미지를 디코딩할 수 없습니다: {path}")
        return [img]

    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(path)
    try:
        n = len(doc)
        lo, hi = 1, n
        if page_range:
            lo = max(1, int(page_range[0]))
            hi = min(n, int(page_range[1]))
        if hi - lo + 1 > PDF_MAX_PAGES:
            hi = lo + PDF_MAX_PAGES - 1
        pages = []
        for i in range(lo - 1, hi):
            page = doc[i]
            w_pt = page.get_size()[0]           # 포인트(1/72inch) 단위 폭
            scale = target_width / max(1.0, w_pt)
            bitmap = page.render(scale=scale)
            pil = bitmap.to_pil().convert("RGB")
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            pages.append(bgr)
        return pages
    finally:
        doc.close()


@dataclass
class PageInfo:
    index: int                  # 0부터
    n_systems: int              # 이 페이지의 단 수
    dy_norm: float              # 정규화 좌표계에서 이 페이지의 y 오프셋
    dy_orig: int                # 쌓인 원본 캔버스에서 이 페이지의 y 오프셋
    h_orig: int                 # 쌓인 원본 캔버스에서 이 페이지의 높이
    key: int = 0
    time_signature: tuple = (0, 0)


@dataclass
class DocumentResult:
    stacked_bgr: np.ndarray     # 세로로 쌓인 가상 원본(편집기의 original_bgr)
    ctx: dict                   # 병합된 ctx (기존 계약과 동일 구조)
    pages: list = field(default_factory=list)   # [PageInfo]
    warnings: list = field(default_factory=list)


# beam_group 키를 페이지마다 유일하게 만드는 간격. 한 페이지의 그룹 수가
# 이 값을 넘을 일은 없다(-1 은 '단독'이므로 시프트하지 않는다).
_GROUP_STRIDE = 100000


def _shift_element(el, d_line, dy, d_group):
    """elements 10-튜플 (line,x,y,w,h,beat,stem_x,beam_group,acc,is_grace) 시프트."""
    el = list(el)
    el[0] = el[0] + d_line
    el[2] = el[2] + dy
    if len(el) > 7 and el[7] is not None and el[7] >= 0:
        el[7] = el[7] + d_group
    return tuple(el)


def _shift_dicts(items, d_line, dy):
    """slurs/voltas/hairpins 처럼 {"line":..,"y..":..} 딕셔너리 목록 시프트."""
    out = []
    for it in items or []:
        it = dict(it)
        if "line" in it:
            it["line"] = it["line"] + d_line
        for k in ("y", "y1", "y2"):
            if k in it and it[k] is not None:
                it[k] = it[k] + dy
        out.append(it)
    return out


def run_document(source, target_width: int = PDF_TARGET_WIDTH,
                 page_range: tuple | None = None,
                 progress=None, with_metadata: bool = True) -> DocumentResult:
    """파일 경로 또는 페이지 리스트 → 페이지별 인식 + 병합.

    progress : callable(i, n) — 페이지 인식 진행 콜백(선택, GUI 상태 표시용).
    with_metadata: 곡 제목/크레딧 OCR(4단계) 실행 여부. 대량 배치 검수처럼
        속도가 중요하고 제목이 필요 없는 경우 False 로 꺼서 페이지당
        1~2초를 아낄 수 있다(가사 OCR은 이미 삭제됨 — 이 플래그와 무관).
    """
    if isinstance(source, (str, bytes)):
        pages_bgr = load_pages(str(source), target_width, page_range)
    else:
        pages_bgr = list(source)
    if not pages_bgr:
        raise ValueError("페이지가 없습니다")

    per_page = []               # [(ctx, ow, oh, cw, ch)]
    for i, bgr in enumerate(pages_bgr):
        if progress:
            progress(i + 1, len(pages_bgr))
        _, ctx = pipeline.run(bgr, with_metadata=with_metadata)
        ch, cw = ctx["image_clean"].shape[:2]
        oh, ow = bgr.shape[:2]
        per_page.append((ctx, ow, oh, cw, ch))

    # ── 단일 페이지: 병합 없이 그대로 반환(기존 경로와 완전 동일 결과 보장)
    if len(per_page) == 1:
        ctx, ow, oh, cw, ch = per_page[0]
        n_sys = len(ctx["staves"]) // 5
        info = PageInfo(0, n_sys, 0.0, 0, oh,
                        ctx["recognition"]["key"],
                        ctx["recognition"]["time_signature"])
        return DocumentResult(pages_bgr[0], ctx, [info])

    # ── 공통 배율 S: 첫 페이지의 원본/정규화 비율. 모든 페이지의 원본을
    #    (cw_p*S, ch_p*S) 로 리샘플해 쌓으면 전역 sx=sy=S 가 성립한다.
    ctx0, ow0, oh0, cw0, ch0 = per_page[0]
    S = ow0 / float(cw0)

    max_cw = max(p[3] for p in per_page)
    total_ch = sum(p[4] for p in per_page) + PAGE_GAP_NORM * (len(per_page) - 1)
    W_orig = int(round(max_cw * S))
    gap_orig = int(round(PAGE_GAP_NORM * S))
    # 캔버스 높이는 '페이지별 반올림 높이의 정확한 합'으로 계산해야 한다.
    # round(total*S) 로 하면 페이지마다 round(ch*S) 한 조각들의 합과 1px
    # 어긋날 수 있다(3페이지 샘플에서 실측 발생 — broadcast 오류).
    H_orig = (sum(int(round(p[4] * S)) for p in per_page)
              + gap_orig * (len(per_page) - 1))

    stacked = np.full((H_orig, W_orig, 3), 255, np.uint8)
    clean_stacked = np.zeros((total_ch, max_cw), np.uint8)

    merged = {
        "image_0": None, "image_clean": clean_stacked, "staves": [],
        "recognition": {"key": 0, "time_signature": (0, 0), "time_symbol": None,
                        "beats": [], "pitches": [], "elements": []},
        "barlines": {}, "barline_types": {},
        "measure_info": {"barlines": {}, "measure_results": [], "sys_capacity": {}},
        "slurs": [], "voltas": [], "hairpins": [],
    }
    warnings = []
    infos = []
    dy = 0.0                    # 정규화 y 오프셋 누적
    y_orig = 0                  # 원본 캔버스 y 오프셋 누적
    d_line = 0                  # 단 번호 오프셋 누적

    for pi, (ctx, ow, oh, cw, ch) in enumerate(per_page):
        rec = ctx["recognition"]
        n_sys = len(ctx["staves"]) // 5
        h_orig = int(round(ch * S))
        w_orig = int(round(cw * S))

        # 원본을 공통 배율로 리샘플해 캔버스에 부착
        page_scaled = cv2.resize(pages_bgr[pi], (w_orig, h_orig),
                                 interpolation=cv2.INTER_AREA)
        stacked[y_orig:y_orig + h_orig, :w_orig] = page_scaled
        clean_stacked[int(dy):int(dy) + ch, :cw] = ctx["image_clean"]

        # 페이지 구분선(간격 중앙의 옅은 가로선)
        if pi > 0:
            ysep = y_orig - gap_orig // 2
            cv2.line(stacked, (0, ysep), (W_orig, ysep), (210, 210, 210), 2)

        # ── ctx 병합(시프트 규칙)
        merged["staves"].extend(y + dy for y in ctx["staves"])
        for el in rec["elements"]:
            merged["recognition"]["elements"].append(
                _shift_element(el, d_line, dy, d_line * _GROUP_STRIDE))
        merged["recognition"]["pitches"].extend(rec["pitches"])
        merged["recognition"]["beats"].extend(rec["beats"])
        for ln, xs in (ctx.get("barlines") or {}).items():
            merged["barlines"][ln + d_line] = list(xs)
        for ln, tmap in (ctx.get("barline_types") or {}).items():
            merged["barline_types"][ln + d_line] = dict(tmap)
        mi = ctx.get("measure_info", {})
        for ln, cap in (mi.get("sys_capacity") or {}).items():
            merged["measure_info"]["sys_capacity"][ln + d_line] = cap
        for tup in (mi.get("measure_results") or []):
            merged["measure_info"]["measure_results"].append(
                (tup[0] + d_line,) + tuple(tup[1:]))
        merged["slurs"].extend(_shift_dicts(ctx.get("slurs"), d_line, dy))
        merged["voltas"].extend(_shift_dicts(ctx.get("voltas"), d_line, dy))
        merged["hairpins"].extend(_shift_dicts(ctx.get("hairpins"), d_line, dy))
        if pi == 0:
            merged["title_credits"] = ctx.get("title_credits")

        # ── 조표·박자표: 1페이지 값을 곡 전체 값으로. 뒷페이지가 다르면 경고만.
        if pi == 0:
            merged["recognition"]["key"] = rec["key"]
            merged["recognition"]["time_signature"] = rec["time_signature"]
            merged["recognition"]["time_symbol"] = rec.get("time_symbol")
        else:
            if rec["key"] != merged["recognition"]["key"]:
                warnings.append(
                    f"{pi + 1}페이지 조표({rec['key']})가 1페이지"
                    f"({merged['recognition']['key']})와 다르게 인식됨 — 1페이지 기준 적용")
            ts0 = merged["recognition"]["time_signature"]
            if rec["time_signature"] != ts0 and rec["time_signature"] != (0, 0):
                warnings.append(
                    f"{pi + 1}페이지 박자표({rec['time_signature']})가 1페이지"
                    f"({ts0})와 다르게 인식됨 — 1페이지 기준 적용")

        infos.append(PageInfo(pi, n_sys, dy, y_orig, h_orig,
                              rec["key"], rec["time_signature"]))
        dy += ch + PAGE_GAP_NORM
        y_orig += h_orig + gap_orig
        d_line += n_sys

    merged["measure_info"]["barlines"] = merged["barlines"]

    # ── 못갖춘마디(여린내기)의 페이지 경계 보정: 엔진은 페이지 단위로
    #    첫/끝 마디를 짝지었지만, 병합 후에는 '곡 전체'의 첫/끝만 유효하다.
    #    (편집기 배지는 model 계층의 detect_anacrusis 가 곡 전체 기준으로
    #     다시 계산하므로 여기 measure_results 는 참고용 정합만 맞춘다.)
    mres = merged["measure_info"]["measure_results"]
    if mres:
        cap = None
        first_ln = min(t[0] for t in mres)
        caps = merged["measure_info"]["sys_capacity"]
        cap = caps.get(first_ln)
        fixed = []
        for k, t in enumerate(mres):
            st = t[4] if len(t) > 4 else None
            if st == "anacrusis" and not (k == 0 or k == len(mres) - 1):
                t = t[:4] + ("under",)     # 페이지 중간의 가짜 못갖춘마디 해제
            fixed.append(t)
        merged["measure_info"]["measure_results"] = fixed

    return DocumentResult(stacked, merged, infos, warnings)
