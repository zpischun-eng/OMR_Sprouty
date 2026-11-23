# omr_bridge/document_adapter.py
# ---------------------------------------------------------------------------
# [2단계] 기존 editor_app_qt.py가 쓰던 omr.document.run_document()와
# 완전히 동일한 인터페이스(반환 객체의 .stacked_bgr/.ctx/.pages/.warnings)를
# 제공하는 어댑터. 내부 구현만 omr_bridge(Audiveris 경유)로 바뀐다.
#
# 설계 근거: HYBRID_OMR_PLAN_v6.md 계획 — "editor_app_qt.py는 그대로
# 재사용, 인식 엔진만 교체". editor_app_qt.py 쪽 변경을 최소화하기 위해
# 기존 DocumentResult와 동일한 모양의 객체를 반환한다.
#
# 1단계 MVP 한계(계획서에 이미 기록된 대로, 후속 단계에서 확장):
# - 이미지 입력만 지원(PDF는 다음 단계)
# - 화음이 있어도 첫 음만 사용(Tier2에서 다성부 확장)
# - 다중 페이지 병합은 아직 없음(단일 페이지 문서 기준)
# ---------------------------------------------------------------------------
import numpy as np
import cv2
import tempfile
from dataclasses import dataclass, field
import os
from pathlib import Path

from omr_bridge.audiveris_runner import run_single, AudiverisResult, AudiverisNotFound
from omr_bridge.merge import build_ctx


@dataclass
class PageInfo:
    index: int
    n_systems: int
    dy_norm: float
    dy_orig: int
    h_orig: int
    key: int = 0
    time_signature: tuple = (0, 0)


@dataclass
class DocumentResult:
    stacked_bgr: np.ndarray
    ctx: dict
    pages: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class OmrBridgeError(Exception):
    """Audiveris 인식 실패 등 사용자에게 그대로 보여줄 오류."""


# PDF를 이미지로 만들 때의 크롭 오프셋·dpi를 이미지 경로별로 기억해 둔다 —
# 곡 정보(PDF 텍스트) 좌표를 같은 이미지 픽셀 좌표로 바꾸는 데 필요하다.
_PDF_RENDER_META = {}


def _render_pdf_page_to_image(pdf_path: str, page_index: int = 0,
                               dpi: int = 400) -> str:
    """PDF 한 페이지를 고해상도 PNG로 렌더링해 임시 파일 경로를 반환한다.

    [PDF 지원 추가] Audiveris에 PDF를 직접 넣을 수도 있지만, 그러면
    "Audiveris가 실제로 본 이미지"와 "화면 왼쪽 패널에 보여줄 원본
    이미지"가 서로 다른 렌더링(둘 다 같은 PDF에서 나왔어도 해상도·
    렌더링 엔진이 다르면 좌표계가 미세하게 어긋날 위험이 있음)이 될 수
    있다. 좌우 100% 일치를 설계 원칙(계획서 3.3절)으로 삼고 있으므로,
    PDF를 먼저 이미지로 직접 렌더링해서 그 파일 하나를 그대로 Audiveris
    입력과 화면 표시 양쪽에 동일하게 쓴다(이번 데모곡 제작 때 검증된
    것과 같은 방식).

    dpi=400은 실사용 검증에서 확인된 안전한 값이다(150dpi는 Audiveris
    가 아예 인식을 거부할 정도로 낮았던 사고가 있었음).
    """
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(pdf_path)
    if len(pdf) == 0:
        raise OmrBridgeError("PDF에 페이지가 없습니다.")
    page = pdf[page_index]
    scale = dpi / 72.0
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()

    # [처리 속도 개선] 렌더링 직후 그대로 쓰면 페이지 여백까지 포함된
    # 큰 이미지(A4 기준 400dpi ≈ 3300x4700)를 그대로 Audiveris에 넣게
    # 되어, 실제 내용(악보)이 페이지 일부에만 있어도 나머지 흰 여백
    # 때문에 인식 시간이 크게 늘어난다(실측: 여백 포함 34.5초 -> 실제
    # 내용만 크롭 21.8초, 약 40% 단축). 데모곡 제작 때 썼던 것과 같은
    # 방식으로 실제 잉크가 있는 영역만 남기고 자동으로 잘라낸다.
    import numpy as np
    arr = np.array(pil_image.convert("L"))
    rows = np.where(arr.min(axis=1) < 200)[0]
    cols = np.where(arr.min(axis=0) < 200)[0]
    crop_x0 = crop_y0 = 0
    if len(rows) and len(cols):
        pad = int(dpi * 0.1)  # 여유 여백(약 0.1인치)
        x0 = max(0, cols.min() - pad)
        y0 = max(0, rows.min() - pad)
        x1 = min(pil_image.width, cols.max() + pad)
        y1 = min(pil_image.height, rows.max() + pad)
        pil_image = pil_image.crop((x0, y0, x1, y1))
        crop_x0, crop_y0 = int(x0), int(y0)

    work_dir = tempfile.mkdtemp(prefix="omr_bridge_pdf_")
    out_path = str(Path(work_dir) / f"page{page_index + 1}.png")
    pil_image.save(out_path)
    _PDF_RENDER_META[out_path] = {"pdf": pdf_path, "page": page_index, "dpi": dpi,
                                  "x0": crop_x0, "y0": crop_y0}
    return out_path


def _imread_unicode(path: str):
    """[한글 경로 버그 수정 — 실사용 검증] cv2.imread는 Windows에서 경로에 한글 등 비영문
    문자가 있으면 파일을 못 읽고 None을 돌려준다("선린교가.jpeg" → '이미지를 읽을 수 없습니다').
    파일을 바이트로 읽은 뒤 cv2.imdecode로 해석하면 경로 문자와 무관하게 읽힌다."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- 인식 결과 캐시
# [로딩 속도 — 실사용 피드백] 불러오기 시간의 97%가 Audiveris(측정: 47.8초 중 46.4초)이고
# 우리 코드는 2초 남짓이다. 같은 이미지를 다시 열 때마다 Audiveris를 다시 돌릴 이유가
# 없으므로, 입력 이미지 내용(해시)을 열쇠로 결과(.omr/.mxl)를 저장해 두고 재사용한다.
# 프로그램과 함께 배포하는 데모곡 결과도 같은 형식으로 동봉해, 시작할 때 Audiveris를
# 아예 돌리지 않게 한다. CACHE_VERSION을 올리면(엔진·변환 방식 변경 시) 전부 무효가 된다.
CACHE_VERSION = "audiveris-5.11.0/r1"
_CACHE_MAX_ENTRIES = 60


def _user_cache_root() -> Path:
    from omr_bridge.paths import user_cache_root          # [v3.30] 운영체제별(맥: ~/Library/Caches)
    return user_cache_root() / "omr_cache"


def _bundled_cache_root() -> Path:
    return Path(__file__).resolve().parent.parent / "resource_dist" / "omr_cache"


def _cache_key(image_path: str) -> str:
    import hashlib
    from omr_bridge import ocr_data
    h = hashlib.sha1((CACHE_VERSION + ocr_data.cache_tag()).encode("utf-8"))   # [v3.20] 글자 인식 켜면 새로
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    # [v3.80] 색 있는 그림은 인식 전처리(색 잉크 → 검정)가 바뀌었으므로 예전 결과를 다시 쓰지 않는다
    # (흑백 그림의 캐시·함께 넣은 데모 캐시는 그대로 유효)
    try:
        import cv2
        import numpy as np
        from omr_bridge.ink_prep import has_color
        img = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
        if img is not None and has_color(img):
            h.update(b"+ink1")
    except Exception:
        pass
    return h.hexdigest()[:24]


def _cache_get(key: str):
    import json
    for root in (_user_cache_root(), _bundled_cache_root()):
        d = root / key
        omr, mxl, meta = d / "result.omr", d / "result.mxl", d / "meta.json"
        if omr.exists() and mxl.exists():
            try:
                uf = json.loads(meta.read_text(encoding="utf-8")).get("upscale_factor", 1.0)
            except Exception:
                uf = 1.0
            try:
                os.utime(d, None)               # 최근 사용 표시(오래된 것부터 정리)
            except OSError:
                pass
            return AudiverisResult(status="ok", mxl_path=str(mxl), omr_path=str(omr),
                                   upscale_factor=uf)
    return None


def _cache_put(key: str, res):
    import json, shutil
    try:
        root = _user_cache_root()
        d = root / key
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(res.omr_path, d / "result.omr")
        shutil.copyfile(res.mxl_path, d / "result.mxl")
        (d / "meta.json").write_text(json.dumps({"upscale_factor": res.upscale_factor,
                                                 "version": CACHE_VERSION}), encoding="utf-8")
        entries = sorted((x for x in root.iterdir() if x.is_dir()), key=lambda x: x.stat().st_mtime)
        for old in entries[:-_CACHE_MAX_ENTRIES]:
            shutil.rmtree(old, ignore_errors=True)
    except Exception:
        pass                                    # 캐시는 부가 기능 — 실패해도 인식 결과는 정상


def clear_cache():
    import shutil
    shutil.rmtree(_user_cache_root(), ignore_errors=True)


def attach_pdf_text(ctx, pdf_meta, pdf_page_warnings):
    """[v2.00 분리] PDF 텍스트 레이어에서 곡 정보·가사·나머지 글자를 뽑아 ctx에 붙이고, 글자와 겹치는
    셈여림을 거른다(한 페이지·여러 페이지 공통). pdf_meta = {pdf, page, dpi, x0, y0}."""
    # [곡 정보] PDF에서 온 입력이면 텍스트 레이어에서 제목/작사작곡/빠르기말을 뽑는다.
    ctx["header_items"] = []
    if pdf_meta and ctx["staves"]:
        try:
            from omr_bridge.pdf_header import extract_header_items
            ctx["header_items"] = extract_header_items(
                pdf_meta["pdf"], pdf_meta["page"], pdf_meta["dpi"],
                pdf_meta["x0"], pdf_meta["y0"], ctx["staves"][0],
                ctx["staves"][1] - ctx["staves"][0])
        except Exception:
            ctx["header_items"] = []   # 곡 정보는 부가 기능 — 실패해도 인식 결과는 정상 제공
    # [가사 — PDF 인식분만] 각 시스템(줄) 오선 아래 글자를 절별 토큰으로 뽑는다.
    # 음표에 붙이는 일은 Score가 만들어진 뒤 편집기에서 한다(model.attach_lyrics).
    # [v3.80] 박자표: 악보 글꼴로 찍힌 숫자가 글자 층에 있으면 그대로 읽는다(그림 인식보다 정확)
    ctx["pdf_time_sigs"] = []
    if pdf_meta and ctx["staves"]:
        try:
            from omr_bridge.pdf_header import extract_time_signatures
            ctx["pdf_time_sigs"] = extract_time_signatures(
                pdf_meta["pdf"], pdf_meta["page"], pdf_meta["dpi"], pdf_meta["x0"], pdf_meta["y0"])
        except Exception:
            ctx["pdf_time_sigs"] = []
    ctx["lyric_tokens"] = []
    if pdf_meta and ctx["staves"]:
        try:
            from omr_bridge.pdf_header import extract_lyric_tokens
            groups = ctx.get("staff_groups") or {}
            if not groups:
                st = ctx["staves"]
                groups = {i: [st[i * 5:(i + 1) * 5]] for i in range(len(st) // 5)}
            il_ = ctx["staves"][1] - ctx["staves"][0]
            # 보표마다 (줄, 보표번호, 위, 아래, 바로 아래 보표의 위, 비울 높이)
            flat = [(li, si + 1, g[0], g[4]) for li, grp in sorted(groups.items())
                    for si, g in enumerate(grp)]
            staves_ = []
            for k, (li, sn, top_, bot_) in enumerate(flat):
                nxt = flat[k + 1] if k + 1 < len(flat) else None
                same_sys = nxt is not None and nxt[0] == li
                staves_.append((li, sn, top_, bot_, nxt[2] if nxt else None,
                                (1.2 if same_sys else 2.5) * il_))
            ctx["lyric_tokens"] = extract_lyric_tokens(
                pdf_meta["pdf"], pdf_meta["page"], pdf_meta["dpi"],
                pdf_meta["x0"], pdf_meta["y0"], staves_, il_)
            ctx["_lyric_staves"] = staves_
        except Exception:
            ctx["lyric_tokens"] = []
    # [셈여림 — PDF 텍스트 겹침 필터] PDF에서 실제 '글자'(가사·제목·빠르기말)가 있는
    # 자리에서 나온 셈여림은 글자를 기호로 오인한 것이다. 신뢰도 필터를 통과했더라도
    # 글자 상자와 절반 이상 겹치면 버린다(이미지 입력에는 적용되지 않음).
    if pdf_meta and ctx.get("dynamics"):
        try:
            from omr_bridge.pdf_header import _pdf_chars
            chars = _pdf_chars(pdf_meta["pdf"], pdf_meta["page"], pdf_meta["dpi"],
                               pdf_meta["x0"], pdf_meta["y0"])
            uf = ctx.get("_upscale_factor") or 1.0
            kept, dropped = [], 0
            for dyn in ctx["dynamics"]:
                if dyn.get("w") is None:
                    kept.append(dyn)
                    continue
                dx0, dy0 = (dyn["x"] - dyn["w"] / 2) / uf, (dyn["y"] - dyn["h"] / 2) / uf
                dx1, dy1 = dx0 + dyn["w"] / uf, dy0 + dyn["h"] / uf
                area = max(1.0, (dx1 - dx0) * (dy1 - dy0))
                over = [c for c in chars if max(0, min(dx1, c[3]) - max(dx0, c[2])) *
                        max(0, min(dy1, c[1]) - max(dy0, c[0])) > 0.3 * area]
                # [예외 — 실사용 검증] 셈여림을 '글자'로 넣은 PDF도 있다("봄이 오면": p가
                # 텍스트). 겹친 글자가 그 셈여림 자체(p, mf …)면 진짜 셈여림이므로 남긴다.
                spelled = "".join(c[4] for c in sorted(over, key=lambda c: c[2])).strip().lower()
                hit = bool(over) and spelled != dyn["key"]
                if hit:
                    dropped += 1
                else:
                    kept.append(dyn)
            ctx["dynamics"] = kept
            if dropped:
                pdf_page_warnings.append(f"가사·글자와 겹치는 셈여림 {dropped}개는 글자 오인으로 보고 제외했습니다")
        except Exception:
            pass
    # [나머지 글자] Andante·Piano·절 번호 등 — 원본 자리에 그대로 표시(편집기에서 header_items와 함께 그림)
    ctx["text_items"] = []
    if pdf_meta and ctx["staves"]:
        try:
            from omr_bridge.pdf_header import extract_other_text
            il_ = ctx["staves"][1] - ctx["staves"][0]
            sl_ = ctx.get("staff_left") or {}
            zones = [(bot + 0.3 * il_, min(bot + 9 * il_, (nxt - mg_) if nxt is not None else 1e9),
                      (sl_.get(li_) or 0) - 0.5 * il_)
                     for (li_, _sn, _top, bot, nxt, mg_) in ctx.get("_lyric_staves", [])]
            uf = ctx.get("_upscale_factor") or 1.0
            boxes = [((d_["x"] - d_["w"] / 2) / uf, (d_["y"] - d_["h"] / 2) / uf,
                      (d_["x"] + d_["w"] / 2) / uf, (d_["y"] + d_["h"] / 2) / uf)
                     for d_ in ctx.get("dynamics", []) if d_.get("w") is not None]
            ctx["text_items"] = extract_other_text(
                pdf_meta["pdf"], pdf_meta["page"], pdf_meta["dpi"], pdf_meta["x0"], pdf_meta["y0"],
                ctx["staves"][0] - 2.5 * il_, zones, boxes, il_)
        except Exception:
            ctx["text_items"] = []

    return ctx


def run_document(source, target_width: int = None, page_range=None,
                 progress=None, with_metadata: bool = True) -> DocumentResult:
    """omr.document.run_document()와 동일한 시그니처/반환 형태.

    target_width, page_range, with_metadata는 현재 MVP에서 사용하지
    않는다(기존 시그니처와의 호환을 위해 인자만 받아둔다 — 2단계 이후
    필요해지면 채운다).
    """
    if isinstance(source, (list, tuple)):
        raise OmrBridgeError("이미지 배열 직접 입력은 아직 지원하지 않습니다(1단계 MVP)")

    path = str(source)
    pdf_page_warnings = []
    if path.lower().endswith(".pdf"):
        import pypdfium2 as pdfium
        n_pages = len(pdfium.PdfDocument(path))
        if n_pages > 1:
            # [PDF 지원 추가] 여러 페이지 자동 병합은 아직 범위 밖 —
            # 우선 첫 페이지만 처리하고, 몇 페이지가 더 있는지 사용자에게
            # 명확히 알린다(조용히 나머지를 버리지 않는다).
            pdf_page_warnings.append(
                f"이 PDF는 {n_pages}페이지입니다 — 현재는 첫 페이지만 "
                f"인식합니다(여러 페이지 자동 병합은 다음 단계 예정)."
            )
        path = _render_pdf_page_to_image(path, page_index=0)

    if progress:
        progress(1, 1)

    import tempfile
    work_dir = tempfile.mkdtemp(prefix="omr_bridge_")
    key = _cache_key(path)
    res = _cache_get(key)
    from_cache = res is not None
    if res is None:
        # [한글 경로 안전장치] 경로에 비영문 문자가 있으면 Audiveris(Java)에는 영문 이름의
        # 임시 사본을 넘긴다(Windows 언어 설정에 따라 Java가 한글 경로를 못 여는 경우 대비).
        aud_src = path
        try:
            path.encode("ascii")
        except UnicodeEncodeError:
            import shutil
            aud_src = os.path.join(work_dir, "input" + (Path(path).suffix.lower() or ".png"))
            shutil.copyfile(path, aud_src)
        try:
            res = run_single(aud_src, work_dir)
        except AudiverisNotFound as ex:
            raise OmrBridgeError(str(ex)) from None
        if res.status == "ok":
            _cache_put(key, res)
    if res.status != "ok":
        raise OmrBridgeError(
            f"Audiveris 인식에 실패했습니다: {res.error}\n"
            f"로그: {res.log_path}"
        )

    img_bgr = _imread_unicode(path)
    if img_bgr is None:
        raise OmrBridgeError(f"이미지를 읽을 수 없습니다: {path}")
    oh, ow = img_bgr.shape[:2]

    merge_result = build_ctx(res.omr_path, res.mxl_path, ow, oh, res.upscale_factor)
    ctx = merge_result.ctx
    ctx["_omr_path"] = res.omr_path       # [v1.40] 미인식 잉크 찾기에서 인식된 기호 상자를 읽는다

    # editor_app_qt.py가 ctx["image_clean"].shape[:2] 로 (cw,ch)를 읽어
    # sx,sy = ow/cw, oh/ch 를 계산한다. 우리는 sx=sy=1/upscale_factor 를
    # 의도했으므로, "clean" 이미지 크기 = Audiveris에 실제로 넣은(업스케일된)
    # 이미지 크기로 맞춰서 이 계산식이 그대로 성립하게 한다.
    aw = int(round(ow * res.upscale_factor))
    ah = int(round(oh * res.upscale_factor))
    ctx["image_clean"] = np.zeros((ah, aw), dtype=np.uint8)  # 내용은 안 씀, shape만 필요

    n_systems = len(ctx["staves"]) // 5
    page_info = PageInfo(
        index=0, n_systems=n_systems, dy_norm=0.0, dy_orig=0, h_orig=oh,
        key=ctx["recognition"]["key"], time_signature=ctx["recognition"]["time_signature"],
    )

    pdf_meta = _PDF_RENDER_META.get(path)
    attach_pdf_text(ctx, pdf_meta, pdf_page_warnings)

    warnings = list(merge_result.warnings)
    if warnings:
        warnings = [f"인식-좌표 결합 시 {len(warnings)}건의 불일치가 있었습니다 "
                    f"(일부 음표 위치가 부정확할 수 있음)"] + warnings[:5]
    warnings = pdf_page_warnings + warnings

    return DocumentResult(
        stacked_bgr=img_bgr, ctx=ctx, pages=[page_info], warnings=warnings,
    )
