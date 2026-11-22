# omr_bridge/page_builder.py
# ---------------------------------------------------------------------------
# [v2.00] 한 페이지 인식 결과(DocumentResult) → 편집 모델 한 페이지.
#
# 예전에는 편집기(_apply_loaded)·검증 도구(compare_render)에 같은 순서의 코드가 따로 있었다.
# 여러 페이지를 다루려면 페이지마다 똑같이 처리해야 하므로 화면과 무관한 이 모듈로 모았다.
#   Score 만들기 → 유령 쉼표 정리 → 붙임줄 오인 쉼표 → 미인식 잉크(제안 음표) → 이음줄 되살리기
#   → 자동 보완 → 원본 기둥 방향 → 앞꾸밈음 → 가사 붙이기
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field

from score import model as score_model
from score import editing as ed


@dataclass
class PageModel:
    score: object
    notes: list = field(default_factory=list)          # 불러오기 안내('정보')
    header_items: list = field(default_factory=list)
    title_credits: dict = field(default_factory=dict)
    barlines: dict = field(default_factory=dict)
    barline_types: dict = field(default_factory=dict)
    by_sys: list = field(default_factory=list)         # 줄별 오선 5줄(첫 보표)
    staff_groups: dict = field(default_factory=dict)
    staff_left: dict = field(default_factory=dict)
    staff_clefs: dict = field(default_factory=dict)
    staff_parts: dict = field(default_factory=dict)
    sx: float = 1.0
    sy: float = 1.0
    warnings: list = field(default_factory=list)


def build_page(doc, sheet: int = 1) -> PageModel:
    """DocumentResult 하나(한 페이지)를 편집 모델로 만든다."""
    img = doc.stacked_bgr
    ctx = doc.ctx
    notes = []
    barlines = ctx.get("barlines") or {}
    barline_types = ctx.get("barline_types") or {}
    score = score_model.build_from_context(ctx, barlines, barline_types)
    score._omr_sheet = sheet
    score.header_x = dict(ctx.get("header_x") or {})     # [v2.30] 줄 머리 기호 원본 x
    il = ctx["staves"][1] - ctx["staves"][0] if len(ctx.get("staves") or []) > 1 else 20.0
    rec_ = ctx.get("recognition") or {}
    ts_added = ed.infer_missed_time_changes(score, rec_.get("time_printed") or {})
    # [v3.40] 원본에 찍힌 박자표를 마디에 기억 — 줄 첫머리에 다시 찍힌 박자표(값이 같아도)를 원본처럼 그리고,
    # C·₵ 기호면 숫자 대신 기호로
    syms_ = rec_.get("time_symbols") or {}
    for m_ in score.measures:
        pv = (rec_.get("time_printed") or {}).get(m_.number)
        if pv is not None:
            m_.ts_printed = (pv[0], pv[1], syms_.get(m_.number))
    if ts_added:
        notes.append("인식되지 않은 박자표를 마디 길이로 찾아 넣었습니다: " +
                     ", ".join(f"{n}마디 {b}/{bt}" for n, (b, bt) in ts_added))
    n_ghost = ed.remove_ghost_rests(score, il, ctx.get("staff_groups"))
    if n_ghost:
        notes.append(f"Audiveris가 잘못 만든 유령 쉼표 {n_ghost}개를 자동으로 지웠습니다")
    n_tie = ed.fix_tie_rests(score, il)
    if n_tie:
        notes.append(f"붙임줄을 쉼표로 잘못 읽은 곳 {n_tie}군데를 붙임줄로 바로잡았습니다")
    score.suggestions, score.ink_blobs = [], []
    try:
        from omr_bridge import ink_check as ik
        up = ctx.get("_upscale_factor", 1.0)
        omr = ctx.get("_omr_path")
        groups = ctx.get("staff_groups")
        n_ph = ik.drop_phantom_heads(score, omr, img, up)
        if n_ph:
            notes.append(f"빔·이음줄 자리를 음표로 잘못 읽은 '유령 음표' {n_ph}개를 지웠습니다")
        ik.assign_stem_dirs(score, img, groups, up)          # 제안 음표의 성부 판단에도 쓴다(먼저)
        blobs = ik.find_unrecognized(img, omr, groups, score, up)
        ik.attach_to_score(score, blobs)
        score.ink_blobs = blobs
        score.suggestions = ik.make_suggestions(score, blobs, img, groups, up)
        n_sl = ik.recover_slurs(score, omr, groups)
        if n_sl:
            notes.append(f"Audiveris가 찾았지만 음표에 잇지 않은 이음줄 {n_sl}개를 넣었습니다")
        done = ik.auto_apply(score)
        if done:
            notes.append(f"인식되지 않은 음표 {len(done)}개를 찾아 자동으로 넣었습니다(넣으면 마디가 정확히 "
                         "맞는 경우만): " + ", ".join(q["text"] for q in done))
        # [v2.54] 화음 속 먼 같은 음 떼기(+빔 증거면 8분 빔) — 자동 보완이 Audiveris가 빼 둔 머리를 앞 화음에
        # 붙이는 경우까지 보도록 자동 보완 뒤에
        notes += ik.split_far_unisons(score, blobs)
        notes += ik.apply_orphan_flags(score, omr, sheet)   # [v2.56] 읽은 꼬리를 길이에(4분 → 8분 등)
        # [v2.70 P-1] 숨은 다른 성부 음(머리 반대쪽으로 설명 안 되는 기둥) → 표시 + 그 마디를 '쉼표 생략'으로 봐주지 않음
        hid = ik.find_hidden_voice_stems(score, ik._gray(img, up), groups, ctx.get("barlines"))
        for h in hid:
            blobs.append(h)
            mm = next((m_ for m_ in score.measures if m_.number == h["measure"]), None)
            if mm is not None:
                uh = dict(getattr(mm, "unexplained_heads", None) or {})
                uh[h["staff_no"]] = uh.get(h["staff_no"], 0) + 1
                mm.unexplained_heads = uh
        if hid:
            notes.append("같은 자리에 기둥 방향이 반대인 다른 성부 음이 숨어 있는 것 같습니다(머리 미인식): "
                         + ", ".join(f"{h['measure']}마디" for h in hid)
                         + " — 원본의 주황 표시 자리에서 그 음 머리를 두 번 클릭해 넣으세요")
        ik.assign_stem_dirs(score, img, groups, up)          # 자동 보완으로 들어온 음까지 다시
        graces = ik.find_graces(score, img, groups, omr, up)
        # [v2.11] 꾸밈음은 6가지 모양 검사를 모두 통과한 것만 찾고, 시험 악보에서 오탐이 없어 바로 넣는다
        # (잘못 들어간 것은 선택 후 삭제). 꾸밈음은 박자를 차지하지 않아 마디 검사로는 확인할 수 없다.
        for q in graces:
            el = ik.accept_suggestion(score, q)
            el.auto_added = True
        if graces:
            notes.append(f"앞꾸밈음 {len(graces)}개를 찾아 넣었습니다(잘못 들어간 것은 선택 후 삭제): "
                         + ", ".join(q["text"] for q in graces))
        rest = [q for q in score.suggestions if q.get("kind") != "grace"]
        if rest:
            notes.append(f"인식되지 않은 음표 {len(rest)}개를 찾아 주황색 '제안 음표'로 표시했습니다 — "
                         "클릭한 뒤 Enter로 넣기, Delete로 무시: " + ", ".join(q["text"] for q in rest))
        other = [b for b in blobs if b["kind"] != "head" and b.get("measure") is not None]
        if other:
            notes.append(f"그 밖에 인식되지 않은 기호로 보이는 곳 {len(other)}군데를 원본에 주황 점선으로 "
                         "표시했습니다(점·임시표 등 — 필요하면 직접 보완)")
    except Exception:
        import traceback
        traceback.print_exc()                      # 부가 기능 — 실패해도 불러오기는 계속
    # [v2.30] 불러온 그대로의 음높이·길이·자리를 적어 둔다 — 바뀌지 않은 음만 원본 기하(곡선·빔·기둥)를 쓴다
    for m in score.measures:
        for e in m.elements:
            e._orig = (e.step, e.octave, e.duration_base, e.src_x)
    # [v3.00] PDF 글자 중 코드 이름(보표 바로 위, 코드 문법) → 음에 붙은 코드(e.harmony). 일반 글자에서는 뺀다
    text_items = list(ctx.get("text_items") or [])
    # [v3.20] 글자 층이 없는 그림 악보: 코드 글자를 직접 찾아 읽는다(이 PC에 Tesseract가 있을 때)
    if not ctx.get("lyric_tokens") and not any((it.get("text") or "").strip() for it in text_items):
        try:
            from omr_bridge import ocr_data, chord_ocr, ink_check as ik_c
            tess = ocr_data.find_tesseract()
            if tess:
                g_ = ik_c._gray(img, ctx.get("_upscale_factor", 1.0) or 1.0)   # 인식 좌표의 회색 그림
                text_items = text_items + chord_ocr.find_chords(g_, score, ctx.get("staff_groups") or {}, il, tess)
        except Exception:
            import traceback
            traceback.print_exc()
    try:
        text_items, n_ch = attach_pdf_chords(score, text_items, ctx.get("staff_groups") or {}, il)
        if n_ch:
            n_ocr = sum(1 for m_ in score.measures for e_ in m_.elements
                        if (getattr(e_, "harmony", None) or {}).get("src") == "ocr")
            if n_ocr:
                notes.append(f"그림에서 기타 코드 {n_ocr}개를 읽었습니다(글자 인식) — 틀린 코드는 음을 고르고 K로 고치세요")
            if n_ch - n_ocr:
                notes.append(f"PDF 글자에서 기타 코드 {n_ch - n_ocr}개를 읽었습니다")
    except Exception:
        import traceback
        traceback.print_exc()
    header_items = list(ctx.get("header_items") or []) + text_items
    if ctx.get("lyric_tokens"):
        score_model.attach_lyrics(score, ctx["lyric_tokens"], il)
    staves = ctx["staves"]
    ch, cw = ctx["image_clean"].shape[:2]
    oh, ow = img.shape[:2]
    return PageModel(
        score=score, notes=notes, header_items=header_items,
        title_credits=ctx.get("title_credits") or {},
        barlines=barlines, barline_types=barline_types,
        by_sys=[staves[i * 5:(i + 1) * 5] for i in range(len(staves) // 5)],
        staff_groups=dict(ctx.get("staff_groups") or {}),
        staff_left=dict(ctx.get("staff_left") or {}),
        staff_clefs=dict(ctx.get("staff_clefs") or {}),
        staff_parts=dict(ctx.get("staff_parts") or {}),
        sx=ow / cw, sy=oh / ch, warnings=list(doc.warnings or []))



def attach_pdf_chords(score, text_items, staff_groups, il):
    """[v3.00] PDF 글자(원본 자리) 중 코드 이름을 그 아래 보표의 가장 가까운 음에 붙인다.
    조건: 코드 문법(근음 대문자) + 어떤 보표 맨 윗줄보다 0.3~6칸 위. 반환: (남은 글자들, 붙인 수)."""
    from score import chords as _ch
    keep, n = [], 0
    for it in text_items:
        t = (it.get("text") or "").strip()
        if not t or not t[0].isupper() or not _ch.is_chord_text(t):
            keep.append(it)
            continue
        cx = (it.get("x0", 0) + it.get("x1", 0)) / 2.0
        cy = it.get("y", 0)
        best = None
        for line, grp in staff_groups.items():
            for k, ys in enumerate(grp):
                gap = ys[0] - cy
                if 0.3 * il <= gap <= 6.0 * il and (best is None or gap < best[0]):
                    best = (gap, line, k + 1)
        if best is None:
            keep.append(it)
            continue
        _g, line, sn = best
        cand = [e for m in score.measures if m.system == line for e in m.elements
                if e.staff_no == sn and e.src_bbox and not e.is_grace]
        if not cand:
            keep.append(it)
            continue
        e = min(cand, key=lambda q: abs((q.src_bbox[0] + q.src_bbox[2] / 2.0) - cx))
        h = _ch.parse(t)
        h.update({"x": cx, "y": cy, "src": it.get("src") or "pdf"})
        e.harmony = h
        n += 1
    return keep, n
