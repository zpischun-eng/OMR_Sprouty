# omr_bridge/book.py
# ---------------------------------------------------------------------------
# [v2.00 여러 페이지] 여러 페이지 PDF를 Audiveris '책(book)'으로 한 장씩 이어서 인식하고, 페이지마다
# 한 페이지 처리(page_builder)를 그대로 거친 뒤 좌표·마디 번호·줄 번호를 옮겨 한 곡(Score)으로 잇는다.
#
#  · 인식: 페이지를 우리가 한 페이지 경로와 똑같이 그려(400dpi, 여백 자름) '여러 장 TIFF' 하나로 묶어
#    Audiveris에 책으로 넘긴다. 1장은 TIFF로, 2장부터는 저장된 book.omr에 '-sheets N'으로 이어서.
#    (PDF를 그대로 넘기면 Audiveris가 PDFBox로 다르게 그려 인식 결과가 달라졌다 — "교회" 1쪽 피아노
#    3마디가 MusicXML에서 빠짐. 같은 그림을 쓰면 한 페이지 경로와 결과가 같다.)
#    실행마다 그 장의 MusicXML만 내보내지므로 sheetN.mxl로 따로 보관한다.
#  · 캐시: %LOCALAPPDATA%/Sprouty/omr_cache/book_<PDF 해시>/ — 인식한 장은 다시 열 때 바로 나온다.
#  · 좌표: 페이지 그림은 Audiveris가 본 그림과 같은 크기(400dpi, 여백 자르지 않음)로 우리가 그린다.
#    한 곡 안에서는 페이지를 세로로 이어 붙인 좌표(페이지 k의 y + 앞 페이지 높이 합)를 쓴다.
#  · 파트: 시트 안의 파트 번호는 그 시트 기준이라, book.xml의 논리 파트(logical-id)로 바꿔 곡 전체에서
#    같은 파트가 같은 번호가 되게 한다(예: "교회" 1장 = 합창 1보표 + 피아노, 2장 = 합창 2보표 + 피아노).
# ---------------------------------------------------------------------------
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PDF_DPI = 400


def _cache_root() -> Path:
    from omr_bridge.paths import user_cache_root          # [v3.30] 운영체제별(맥: ~/Library/Caches)
    return user_cache_root() / "omr_cache"


def n_pdf_pages(pdf_path: str) -> int:
    import pypdfium2 as pdfium
    return len(pdfium.PdfDocument(pdf_path))


# [v2.53] 책 캐시 판(版). 예전엔 열쇠가 PDF 내용뿐이라, 처음 인식한 결과(그때 프로그램·그때 실행 상태
# 그대로)를 영원히 다시 썼다 — 한 페이지 캐시는 CACHE_VERSION이 바뀌면 무효가 되는데 책은 그렇지 않았다.
# 인식 방식(장 처리·자르기·내보내기)이 바뀌면 올린다.
BOOK_CACHE_VERSION = "book-r3"   # r3: [v3.80] 색 잉크 전처리


class BookJob:
    """여러 페이지 PDF 하나의 인식 작업(장 단위, 캐시)."""

    def __init__(self, pdf_path: str):
        self.src = pdf_path
        from omr_bridge import ocr_data
        h = hashlib.sha1((BOOK_CACHE_VERSION + ocr_data.cache_tag()).encode("utf-8"))   # [v3.20] 글자 인식 켜면 새로
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        self.key = h.hexdigest()[:24]
        self.dir = _cache_root() / f"book_{self.key}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pdf = self.dir / "book.pdf"             # 영문 경로의 사본(Audiveris가 다음 장을 이 파일에서 읽는다)
        if not self.pdf.exists():
            shutil.copyfile(pdf_path, self.pdf)
        self.n_pages = n_pdf_pages(str(self.pdf))
        self.omr = self.dir / "book.omr"
        self.tif = self.dir / "book.tif"

    def prepare(self, progress=None):
        """페이지 그림을 모두 그려 두고(pageN.png + 자른 위치) 여러 장 TIFF를 만든다(한 번만)."""
        meta = self._meta()
        if self.tif.exists() and len(meta.get("crops", [])) == self.n_pages:
            return
        from PIL import Image
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(self.pdf))
        crops, frames = [], []
        for i in range(self.n_pages):
            pil = doc[i].render(scale=PDF_DPI / 72.0).to_pil()
            from omr_bridge.ink_prep import ink_gray, has_color
            arr_ = np.array(pil.convert("RGB"))
            # [v3.80] 색 잉크(빨간 악보·색음표)는 검게 — 그냥 회색으로 바꾸면 밝은 색 머리가 사라진다
            pil = Image.fromarray(ink_gray(arr_)) if has_color(arr_) else pil.convert("L")
            arr = np.array(pil)
            rows = np.where(arr.min(axis=1) < 200)[0]
            cols = np.where(arr.min(axis=0) < 200)[0]
            x0 = y0 = 0
            if len(rows) and len(cols):                  # 한 페이지 경로와 같은 여백 자르기(0.1인치 남김)
                pad = int(PDF_DPI * 0.1)
                x0, y0 = max(0, int(cols.min()) - pad), max(0, int(rows.min()) - pad)
                x1, y1 = min(pil.width, int(cols.max()) + pad), min(pil.height, int(rows.max()) + pad)
                pil = pil.crop((x0, y0, x1, y1))
            pil = Image.fromarray(np.array(pil))        # pdfium 그림에 붙은 부가 정보를 떼어 낸다(TIFF 저장 오류 방지)
            pil.save(str(self.dir / f"page{i + 1}.png"))
            crops.append([x0, y0])
            frames.append(pil)
            if progress:
                progress(i + 1, self.n_pages)
        import cv2
        # 여러 장 TIFF(LZW). Pillow의 libtiff 저장은 큰 그림에서 오류가 나서 OpenCV로 쓴다.
        ok = cv2.imwritemulti(str(self.tif), [np.array(f) for f in frames],
                              [cv2.IMWRITE_TIFF_COMPRESSION, 5])
        if not ok:
            raise RuntimeError("여러 장 TIFF를 만들지 못했습니다")
        meta["crops"] = crops
        self._save_meta(meta)

    # ------------------------------------------------------------ 상태
    def _meta(self) -> dict:
        p = self.dir / "meta.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        return {"done": []}

    def _save_meta(self, meta):
        (self.dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def verify(self) -> list:
        """[v2.53 캐시 맞춤 검사] '끝난' 장마다 book.omr의 그 장과 sheetN.mxl이 같은 인식에서 나왔는지 본다.
        음표 위치·기둥·빔은 .omr에서, 음은 .mxl에서 가져와 합치므로 둘이 어긋나면(실행이 끊김, 1쪽이 다시
        처리돼 book.omr이 새로 만들어짐 등) 음이 빠지고 기둥·빔이 엉뚱해진다. 맞지 않는 장 번호 목록을 돌려준다.
        검사: ① .omr에 그 장이 있다 ② 마디 수(.omr stack = .mxl 첫 파트 measure) ③ 머리 수 ≥ 음 수."""
        import re
        import zipfile
        done = [n for n in self._meta().get("done", []) if (self.dir / f"sheet{n}.mxl").exists()]
        if not done:
            return []
        bad = []
        try:
            zo = zipfile.ZipFile(str(self.omr))
        except Exception:
            return done
        names = set(zo.namelist())
        for n in done:
            key = f"sheet#{n}/sheet#{n}.xml"
            if key not in names:
                bad.append(n)
                continue
            try:
                ox = zo.read(key).decode("utf-8", "replace")
                zm = zipfile.ZipFile(self.mxl(n))
                mx = zm.read([k for k in zm.namelist() if k.endswith(".xml") and "META" not in k][0]).decode(
                    "utf-8", "replace")
            except Exception:
                bad.append(n)
                continue
            n_stack = len(re.findall(r"<stack ", ox))
            first_part = mx.split("</part>")[0]
            n_meas = len(re.findall(r"<measure ", first_part))
            n_heads = len(re.findall(r"<head ", ox))
            n_notes = len(re.findall(r"<note[ >]", mx)) - len(re.findall(r"<rest", mx))
            if n_stack != n_meas or n_heads < 0.9 * n_notes:
                bad.append(n)
        return bad

    def reset_recognition(self):
        """[v2.53] 이 책의 인식 결과(book.omr·장별 .mxl·완료 기록)를 지운다 — 쪽 그림·자르기는 남긴다."""
        for p in [self.omr, self.dir / "book.mxl"] + list(self.dir.glob("sheet*.mxl")):
            try:
                p.unlink()
            except OSError:
                pass
        meta = self._meta()
        meta["done"] = []
        self._save_meta(meta) if hasattr(self, "_save_meta") else None

    def is_done(self, n: int) -> bool:
        return n in self._meta().get("done", []) and (self.dir / f"sheet{n}.mxl").exists()

    def mxl(self, n: int) -> str:
        return str(self.dir / f"sheet{n}.mxl")

    # ------------------------------------------------------------ 인식
    def process_sheet(self, n: int, timeout_sec: int = 600):
        """n번째 장을 인식한다(앞 장들이 끝나 있어야 한다). 실패하면 RuntimeError."""
        if self.is_done(n):
            return
        from omr_bridge.audiveris_runner import find_audiveris_bin
        env = os.environ.copy()
        env["LANG"] = env["LC_ALL"] = "C.utf8"
        self.prepare()
        stale = self.dir / "book.mxl"
        if stale.exists():
            stale.unlink()                               # 이번 실행의 내보내기만 쓰도록
        src = str(self.tif) if (n == 1 or not self.omr.exists()) else str(self.omr)
        from omr_bridge import ocr_data
        cmd = [find_audiveris_bin(), "-batch"] + ocr_data.apply(env, []) + \
            ["-sheets", str(n), "-save", "-export", "-output", str(self.dir), "--", src]   # [v3.20] 글자 인식
        log = self.dir / f"sheet{n}.log"
        from omr_bridge.audiveris_runner import jvm_env, log_tail
        jvm_env(env)                                     # [v3.61] 자바 메모리 한도를 넉넉히
        with open(log, "w", encoding="utf-8") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout_sec, env=env)
        out_mxl = self.dir / "book.mxl"
        if not out_mxl.exists() or not self._sheet_in_omr(n):
            raise RuntimeError(f"{n}페이지 인식에 실패했습니다(로그: {log})\n--- 기록 끝 ---\n"
                               + log_tail(log, 30, 2500))
        shutil.copyfile(out_mxl, self.mxl(n))
        meta = self._meta()
        meta.setdefault("done", [])
        if n not in meta["done"]:
            meta["done"].append(n)
        self._save_meta(meta)

    def _sheet_in_omr(self, n: int) -> bool:
        if not self.omr.exists():
            return False
        with zipfile.ZipFile(self.omr) as z:
            return f"sheet#{n}/sheet#{n}.xml" in z.namelist()

    # ------------------------------------------------------------ 페이지 그림·문서
    def picture_size(self, n: int):
        with zipfile.ZipFile(self.omr) as z:
            root = ET.fromstring(z.read(f"sheet#{n}/sheet#{n}.xml"))
        pic = root.find(".//picture")
        return int(pic.get("width")), int(pic.get("height"))

    def page_image(self, n: int):
        """n번째 페이지 그림(BGR) — Audiveris에 넘긴 TIFF의 그 장과 같은 그림."""
        import cv2
        from omr_bridge.document_adapter import _imread_unicode
        self.prepare()
        img = _imread_unicode(str(self.dir / f"page{n}.png"))
        return img

    def crop_of(self, n: int):
        c = self._meta().get("crops", [])
        return tuple(c[n - 1]) if n - 1 < len(c) else (0, 0)

    def page_doc(self, n: int):
        """n번째 페이지의 DocumentResult(한 페이지 경로와 같은 모양)."""
        from omr_bridge.merge import build_ctx
        from omr_bridge.document_adapter import DocumentResult, PageInfo, attach_pdf_text
        img = self.page_image(n)
        h, w = img.shape[:2]
        mr = build_ctx(str(self.omr), self.mxl(n), w, h, 1.0, sheet=n)
        ctx = mr.ctx
        ctx["_omr_path"] = str(self.omr)
        ctx["image_clean"] = np.zeros((h, w), dtype=np.uint8)
        warns = []
        if ctx.get("staves"):
            x0, y0 = self.crop_of(n)
            attach_pdf_text(ctx, {"pdf": str(self.pdf), "page": n - 1, "dpi": PDF_DPI, "x0": x0, "y0": y0}, warns)
        info = PageInfo(index=n - 1, n_systems=len(ctx.get("staves", [])) // 5, dy_norm=0.0,
                        dy_orig=0, h_orig=h, key=ctx["recognition"]["key"],
                        time_signature=ctx["recognition"]["time_signature"])
        w2 = list(mr.warnings)
        if w2:
            w2 = [f"{n}페이지: 인식-좌표 결합 시 {len(w2)}건의 불일치"] + w2[:3]
        return DocumentResult(stacked_bgr=img, ctx=ctx, pages=[info], warnings=warns + w2)

    def logical_parts(self, n: int):
        """n번째 장의 줄(시스템)마다 보표별 논리 파트 번호(0부터) 목록. 없으면 None."""
        try:
            with zipfile.ZipFile(self.omr) as z:
                root = ET.fromstring(z.read("book.xml"))
        except Exception:
            return None
        for sh in root.findall("sheet"):
            if int(sh.get("number", 0)) != n:
                continue
            out = []
            for sysel in sh.iter("system"):
                staves = []
                for part in sysel.findall("part"):
                    lid = int(part.get("logical-id", "1")) - 1
                    staves += [lid] * max(1, len(part.findall("staff-configuration")))
                out.append(staves)
            return out
        return None


# --------------------------------------------------------------------- 이어 붙이기
class BookState:
    """한 곡으로 이어 붙인 결과(편집기가 그대로 쓴다)."""

    def __init__(self):
        self.score = None
        self.notes, self.header_items = [], []
        self.title_credits = {}
        self.barlines, self.barline_types = {}, {}
        self.by_sys = []
        self.staff_groups, self.staff_left, self.staff_clefs, self.staff_parts = {}, {}, {}, {}
        self.part_names = []
        self.pages = []                 # PageInfo (dy_orig = 이어 붙인 좌표의 페이지 위쪽)
        self.page_sizes = []            # (w, h)
        self.warnings = []

    @property
    def total_h(self):
        return sum(h for _w, h in self.page_sizes)


def carry_time(state):
    """[v4.20] 지금까지 이은 악보 끝의 박자 (beats, beat_type) — 다음 페이지를 만들 때 build_page(carry_time=)로 넘긴다.
    아직 아무 페이지도 없거나 박자를 모르면 None."""
    sc = getattr(state, "score", None)
    if sc is None or not getattr(sc, "time_sig_known", True):
        return None
    t = (sc.beats, sc.beat_type)
    for m in sc.measures:
        if m.time_sig_change:
            t = tuple(m.time_sig_change)
    return t


def append_page(state: BookState, pm, page_no: int, page_size, sys_logical=None):
    """페이지 모델(pm)을 곡(state) 끝에 잇는다. 좌표는 앞 페이지 높이만큼 아래로, 줄·마디 번호는 이어서."""
    from omr_bridge.document_adapter import PageInfo
    w, h = page_size
    dy = state.total_h
    L = len(state.by_sys)
    M = len(state.score.measures) if state.score is not None else 0
    sc = pm.score
    sc._page_header_x = dict(getattr(sc, "header_x", None) or {})
    # ---- 파트: 시트 기준 번호 → 곡 전체 논리 번호
    names_local = list(getattr(sc, "part_names", None) or [])
    pmap = {}
    for line, parts in pm.staff_parts.items():
        logi = sys_logical[line] if sys_logical and line < len(sys_logical) else None
        if logi and len(logi) == len(parts):
            for lp, gp in zip(parts, logi):
                pmap.setdefault(lp, gp)
    if not pmap:
        pmap = {lp: lp for parts in pm.staff_parts.values() for lp in parts}
    for lp, gp in pmap.items():
        while len(state.part_names) <= gp:
            state.part_names.append("")
        if not state.part_names[gp] and lp < len(names_local):
            state.part_names[gp] = names_local[lp]
    # ---- 조표·박자표가 앞 페이지와 이어지는지
    first = min(sc.measures, key=lambda m: m.number) if sc.measures else None
    if state.score is not None and first is not None:
        prev_key = state.score.key_at(state.score.measures[-1].number)
        if first.key_change is None and sc.fifths != prev_key:
            first.key_change = sc.fifths
        prev_t = (state.score.beats, state.score.beat_type)
        for m in state.score.measures:
            if m.time_sig_change:
                prev_t = m.time_sig_change
        if first.time_sig_change is None and getattr(sc, "time_sig_known", True) \
                and not getattr(sc, "ts_header_inferred", False) \
                and (sc.beats, sc.beat_type) != prev_t:            # [v3.63] 박자표 없는 페이지는 앞 박자 그대로
            # [v4.20] ts_header_inferred: 페이지 머리에 박자표가 없어 마디 길이로 짐작한 것 — 앞 박자를 잇는다
            first.time_sig_change = (sc.beats, sc.beat_type)
    # ---- 좌표·번호 옮기기
    for m in sc.measures:
        m.number += M
        m.system += L
        m.page = page_no
        for e in m.elements:
            _shift_el(e, dy)
        if getattr(m, "unexplained_heads", None) is None:
            m.unexplained_heads = {}
    for hp in getattr(sc, "hairpins", []):
        if hp.y is not None:
            hp.y += dy
    for m_ in sc.measures:                               # [v2.54] 볼타 원본 높이
        if getattr(m_, "ending_ys", None):
            m_.ending_ys = [y_ + dy for y_ in m_.ending_ys]
    for a in getattr(sc, "arcs", []):                    # [v2.30] 원본 곡선
        if getattr(a, "src_poly", None):
            a.src_poly = [(px, py + dy) for px, py in a.src_poly]
        for k in ("src_curve", "src_curve_a", "src_curve_b"):
            cv = getattr(a, k, None)
            if cv:
                x1, y1, c1x, c1y, c2x, c2y, x2, y2 = cv
                setattr(a, k, (x1, y1 + dy, c1x, c1y + dy, c2x, c2y + dy, x2, y2 + dy))
    blobs = {}
    for q in getattr(sc, "suggestions", None) or []:
        q["measure"] += M
        _shift_el(q["element"], dy)
        blobs[id(q["blob"])] = q["blob"]
    for b in getattr(sc, "ink_blobs", None) or []:
        blobs[id(b)] = b
    import re as _re
    for q in getattr(sc, "suggestions", None) or []:
        q["text"] = _re.sub(r"^\d+마디", f"{q['measure']}마디", q["text"])   # 페이지 안 번호 → 곡 번호
    for b in blobs.values():                       # 제안과 미인식 목록이 같은 덩어리를 공유한다 — 한 번만 옮긴다
        x0, y0, x1, y1 = b["box"]
        b["box"] = (x0, y0 + dy, x1, y1 + dy)
        if b.get("line") is not None:
            b["line"] += L
        if b.get("measure") is not None:
            b["measure"] += M
    for it in pm.header_items:
        it["y"] = it["y"] + dy
    # ---- 곡에 잇기
    if state.score is None:
        state.score = sc
        state.score.arcs = list(sc.arcs)
        state.score.hairpins = list(getattr(sc, "hairpins", []))
        state.score.suggestions = list(getattr(sc, "suggestions", None) or [])
        state.score.ink_blobs = list(getattr(sc, "ink_blobs", None) or [])
        state.title_credits = pm.title_credits
    else:
        state.score.measures += sc.measures
        state.score.arcs += sc.arcs
        state.score.hairpins += getattr(sc, "hairpins", [])
        state.score.suggestions = list(getattr(state.score, "suggestions", None) or []) + \
            list(getattr(sc, "suggestions", None) or [])
        state.score.ink_blobs = list(getattr(state.score, "ink_blobs", None) or []) + \
            list(getattr(sc, "ink_blobs", None) or [])
    for line in sorted(pm.staff_groups):
        state.staff_groups[line + L] = [[v + dy for v in ys] for ys in pm.staff_groups[line]]
    for line, v in pm.staff_left.items():
        state.staff_left[line + L] = v
    hx = getattr(state.score, "header_x", None)
    if hx is None or state.score is sc:
        state.score.header_x = hx = {} if state.score is not sc else dict()
    for line, v in (getattr(sc, "_page_header_x", None) or {}).items():
        hx[line + L] = v
    for line, v in pm.staff_clefs.items():
        state.staff_clefs[line + L] = list(v)
    for line, v in pm.staff_parts.items():
        state.staff_parts[line + L] = [pmap.get(p, p) for p in v]
    for line, v in pm.barlines.items():
        state.barlines[line + L] = list(v)
    for line, v in pm.barline_types.items():
        state.barline_types[line + L] = dict(v)
    state.by_sys += [[v + dy for v in ys] for ys in pm.by_sys]
    state.header_items += pm.header_items
    # 안내 문구 속 마디 번호도 페이지 안 번호 → 곡 번호
    state.notes += [f"{page_no}페이지: " + _re.sub(r"(\d+)마디", lambda mo: f"{int(mo.group(1)) + M}마디", n)
                    for n in pm.notes]
    state.warnings += pm.warnings
    state.pages.append(PageInfo(index=page_no - 1, n_systems=len(pm.by_sys), dy_norm=0.0,
                                dy_orig=dy, h_orig=h))
    state.page_sizes.append((w, h))
    s = state.score
    renumber_endings(s)
    s.staff_parts, s.staff_clefs = state.staff_parts, state.staff_clefs
    s.part_names = state.part_names
    s.unexplained_heads = {}
    for m in s.measures:
        for sn, c in (getattr(m, "unexplained_heads", None) or {}).items():
            s.unexplained_heads[(m.number, sn)] = c
    return state


def _shift_el(e, dy):
    if e.src_bbox:
        x, y, w, h = e.src_bbox
        e.src_bbox = (x, y + dy, w, h)
    if getattr(e, "src_stem", None):                     # [v2.30] 원본 기둥·빔
        sx_, t_, b_ = e.src_stem
        e.src_stem = (sx_, t_ + dy, b_ + dy)
    if getattr(e, "src_beams", None):
        e.src_beams = [(a, b + dy, c, d_ + dy, h) for a, b, c, d_, h in e.src_beams]
    for ly in (getattr(e, "lyrics", None) or {}).values():
        if ly.get("y") is not None:
            ly["y"] += dy
    if getattr(e, "dynamic", None) and e.dynamic.get("y") is not None:
        e.dynamic = dict(e.dynamic, y=e.dynamic["y"] + dy)
    if getattr(e, "harmony", None) and e.harmony.get("y") is not None:     # [v3.00] 코드 원본 높이
        e.harmony = dict(e.harmony, y=e.harmony["y"] + dy)


def renumber_endings(score):
    """[v2.00 볼타 번호] Audiveris는 꺾쇠는 잘 읽지만 안의 작은 숫자는 자주 틀린다("교회" 40마디 '2.'를
    1로). 음악 규칙대로 고친다: 앞 볼타 뒤에 되돌이(:|)가 있었고 이번 볼타 번호가 앞 번호보다 크지 않으면
    앞 번호 + 1. 되돌이 없이 새 볼타가 나오면 새로 센다."""
    last, rep_since = None, False
    ms = sorted(score.measures, key=lambda m: m.number)
    i = 0
    while i < len(ms):
        m = ms[i]
        if m.ending:
            j = i
            while j + 1 < len(ms) and ms[j + 1].ending == m.ending:
                j += 1
            num = m.ending
            if last is not None and rep_since and num <= last:
                num = last + 1
                for k in range(i, j + 1):
                    ms[k].ending = num
            last, rep_since = num, False
            if ms[j].end_barline == "repeat_end":
                rep_since = True
            i = j + 1
            continue
        if m.end_barline == "repeat_end":
            rep_since = True
        i += 1
