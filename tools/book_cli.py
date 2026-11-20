# tools/book_cli.py — [v2.00] 여러 페이지 PDF 인식·조립 확인용(개발용)
#   python tools/book_cli.py 악보.pdf --sheets 1 2   # 그 장들을 인식(캐시)
#   python tools/book_cli.py 악보.pdf --build         # 인식된 장들을 한 곡으로 잇고 요약
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omr_bridge.book import BookJob, BookState, append_page  # noqa: E402
from omr_bridge.page_builder import build_page  # noqa: E402


def build(pdf, upto=None):
    job = BookJob(pdf)
    st = BookState()
    for n in range(1, (upto or job.n_pages) + 1):
        if not job.is_done(n):
            break
        doc = job.page_doc(n)
        pm = build_page(doc, sheet=n)
        append_page(st, pm, n, (doc.stacked_bgr.shape[1], doc.stacked_bgr.shape[0]), job.logical_parts(n))
    return job, st


if __name__ == "__main__":
    pdf = sys.argv[1]
    if "--sheets" in sys.argv:
        job = BookJob(pdf)
        i = sys.argv.index("--sheets")
        for a in sys.argv[i + 1:]:
            if a.startswith("--"):
                break
            import time
            t = time.time()
            job.process_sheet(int(a))
            print(f"sheet {a} 완료 {time.time() - t:.0f}s")
    if "--build" in sys.argv:
        from score import editing as ed
        job, st = build(pdf)
        bad = [k for k, x in ed.validate(st.score).items() if x[0] not in ("ok", "anacrusis", "time_change")]
        print(f"페이지 {len(st.pages)}/{job.n_pages} · 마디 {len(st.score.measures)} · 줄 {len(st.by_sys)} · "
              f"파트 {st.part_names} · 오류 마디 {bad}")
        for p in st.pages:
            ms = [m.number for m in st.score.measures if getattr(m, 'page', 1) == p.index + 1]
            print(f"  {p.index + 1}p dy={p.dy_orig} h={p.h_orig} 마디 {ms[0]}~{ms[-1]}")


def render_page(st, page_idx, F=1):
    """[개발 확인] 곡 전체 레이아웃에서 한 페이지만 그린다(편집기와 같은 방식)."""
    from score import layout as sl, render as sr
    lay = sl.layout_score_world(st.score, st.by_sys, 1.0, 1.0, st.barlines or None,
                                staff_groups=st.staff_groups, staff_left_by_line=st.staff_left,
                                staff_clefs=st.staff_clefs, staff_parts=st.staff_parts)
    lay.header_items = st.header_items
    p = st.pages[page_idx]
    w, h = st.page_sizes[page_idx]
    pv = sl.page_view(lay, p.dy_orig)
    img, lm = sr.render(pv, w, h, F=F, return_margin=True, lyric_y_off=p.dy_orig)
    return img, lm
