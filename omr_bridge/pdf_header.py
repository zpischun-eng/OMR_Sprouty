# pdf_header.py — PDF 텍스트 레이어에서 곡 제목·작사작곡·빠르기말을 뽑는다.
# ---------------------------------------------------------------------------
# [곡 정보 표시 — 실사용 피드백] 편집 악보에 제목/작사·작곡/빠르기말이 없다는
# 지적. Audiveris는 이 악보에서 <credit>/<work-title>을 하나도 내보내지 않았지만,
# 이 PDF는 텍스트가 실제 텍스트로 들어 있어(벡터 글꼴) 위치·크기까지 정확히
# 읽을 수 있다. 반환 좌표는 Audiveris에 넣은(여백 크롭된) 이미지의 픽셀 기준이라
# 편집 악보의 world 좌표와 그대로 일치한다.
# 이미지(스캔) 입력은 텍스트 레이어가 없으므로 이 경로로는 못 얻고, 편집기의
# "곡 정보" 입력창으로 직접 적는다.
# ---------------------------------------------------------------------------
import re


def _is_text_char(tx):
    """악보 글꼴로 그린 음표·기호(유니코드 사용자 정의 영역 U+E000~F8FF 등)나
    보이지 않는 제어문자는 글자가 아니므로 제외한다(가사·제목에 섞이지 않게)."""
    return bool(tx) and tx.isprintable() and not any(0xE000 <= ord(ch) <= 0xF8FF for ch in tx)


# [v1.10 빠르기 ♩ 표시] 빠르기 "♩=65"의 음표는 대개 악보 전용 글꼴 문자(SMuFL 사용자 정의 영역,
# 또는 Finale Maestro·Opus 계열의 'q'·'e'·'h')로 들어 있어 글자 추출에서 빠졌다. 음표 모양 문자만
# 따로 모아, 빠르기 줄의 '=' 바로 왼쪽에 있으면 표준 유니코드 음표(내부 키)로 되살린다.
_SMUFL_NOTES = {0xE1D2: "w", 0xE1D3: "h", 0xE1D4: "h", 0xE1D5: "q", 0xE1D6: "q", 0xE1D7: "e",
                0xE1D8: "e", 0xECA2: "w", 0xECA3: "h", 0xECA4: "h", 0xECA5: "q", 0xECA6: "q",
                0xECA7: "e", 0xECA8: "e", 0x2669: "q", 0x266A: "e", 0x1D15E: "h", 0x1D15F: "q",
                0x1D160: "e"}
_SMUFL_DOTS = {0xE1E7, 0xECB7}
_LEGACY_NOTES = {"q": "q", "Q": "q", "e": "e", "E": "e", "h": "h", "H": "h", "w": "w"}
# 내부 키 → 화면에 넣을 문자(그리기는 render.py가 Bravura 메트로놈 글리프로 바꿔 그린다)
TEMPO_NOTE_CHAR = {"w": "\U0001D15D", "h": "\U0001D15E", "q": "\u2669", "e": "\u266A"}


def _music_note_kind(ch, from_music_font):
    if not ch:
        return None
    o = ord(ch[0])
    if o in _SMUFL_NOTES:
        return _SMUFL_NOTES[o]
    if o in _SMUFL_DOTS or (from_music_font and ch == "."):
        return "dot"
    if 0xF020 <= o <= 0xF0FF:
        # 이름 없는 내장 기호 글꼴(심벌 인코딩 → U+F000+코드, 실측: "기도의 능력" ♩ = U+F047).
        # 빠르기 줄 '=' 바로 왼쪽에 있을 때만 쓰이므로, 아는 글자가 아니면 4분음표로 본다.
        letter = chr(o - 0xF000)
        if letter == ".":
            return "dot"
        return _LEGACY_NOTES.get(letter, "q")
    if from_music_font:
        return _LEGACY_NOTES.get(ch)
    return None


def _restore_tempo_notes(items, marks, interline_px):
    """빠르기 줄에서 '=' 바로 왼쪽에 있던 음표 문자를 되살린다(점음표 포함)."""
    for it in items:
        txt = it["text"]
        if "=" not in txt or any(ch in txt for ch in TEMPO_NOTE_CHAR.values()):
            continue
        eq = it.get("_eq_x")
        if eq is None:
            continue
        h = it["h"]
        near = [m for m in marks
                if abs((m[0] + m[1]) / 2 - it["y"]) < max(1.2 * h, 0.9 * interline_px)
                and eq - 3.0 * h < m[2] < eq + 0.2 * h]
        notes = sorted([m for m in near if m[4] != "dot"], key=lambda m: m[2])
        if not notes:
            continue
        n = notes[-1]
        dotted = any(m[4] == "dot" and n[3] - 0.1 * h <= m[2] < eq for m in near)
        sym = TEMPO_NOTE_CHAR.get(n[4], TEMPO_NOTE_CHAR["q"]) + ("." if dotted else "")
        if eq - n[3] > 0.3 * h:                 # 원본에 음표와 '=' 사이가 띄어져 있으면 그대로
            sym += " "
        i = txt.index("=")
        it["text"] = (txt[:i].rstrip() + " " if txt[:i].strip() else "") + sym + txt[i:]
        if not txt[:i].strip():
            it["x0"] = min(it["x0"], n[2])


def extract_header_items(pdf_path, page_index, dpi, crop_x0, crop_y0,
                         first_staff_top_px, interline_px):
    # 1) 글자 단위로 모은다(가짜 굵게·찌꺼기 객체 정리는 _pdf_chars가 처리)
    marks = []
    kept = _pdf_chars(pdf_path, page_index, dpi, crop_x0, crop_y0, music_marks=marks)

    # 2) 첫 오선 위 2.5칸보다 더 위에 있는 글자만 머리말로 본다(코드기호·가사 제외)
    limit = first_staff_top_px - 2.5 * interline_px
    head = [c for c in kept if c[1] < limit]
    lines = []
    for c in sorted(head, key=lambda c: (c[0] + c[1]) / 2):
        cy = (c[0] + c[1]) / 2
        for ln in lines:
            if abs(ln["cy"] - cy) < 0.7 * interline_px:
                ln["c"].append(c)
                break
        else:
            lines.append({"cy": cy, "c": [c]})

    items = []
    for ln in lines:
        cs = sorted(ln["c"], key=lambda c: c[2])
        h = max(c[1] - c[0] for c in cs)
        text, prev_r, eq_x = "", None, None
        for c in cs:
            if c[4] == "=" and eq_x is None:
                eq_x = c[2]
            if prev_r is not None and (c[5] or c[2] - prev_r > h * 0.8):   # PDF 공백 문자 우선
                text += " "
            text += c[4]
            prev_r = c[3]
        items.append({"text": text.strip(), "x0": cs[0][2], "x1": cs[-1][3],
                      "y": ln["cy"], "h": h, "_eq_x": eq_x,
                      "bold": sum(1 for c in cs if c[6]) * 2 > len(cs)})
    if not items:
        return []
    _restore_tempo_notes(items, [m for m in marks if m[1] < limit + 2 * interline_px], interline_px)
    for it in items:
        it.pop("_eq_x", None)

    # 3) 분류: 작사·작곡·역사(번역) 표기 = 크레딧(키워드로 먼저 가려냄),
    #    남은 줄 중 가장 큰 글자가 다른 줄보다 확실히(1.3배 이상) 크면 제목,
    #    나머지는 빠르기말. 교과서에서 잘라낸 악보처럼 제목이 아예 없는 PDF에서
    #    작곡자 줄을 제목으로 잡는 오류를 막는다.
    credit_re = re.compile(r"작사|작곡|편곡|작시|역사|번역|개사|Words|Music|Arr|Lyrics", re.I)
    rest = []
    for it in items:
        if credit_re.search(it["text"]):
            it["kind"] = "credit"
        else:
            rest.append(it)
    if rest:
        big = max(rest, key=lambda it: it["h"])
        others = [it["h"] for it in rest if it is not big]
        is_title = (not others) or big["h"] >= 1.3 * max(others)
        credits_h = [it["h"] for it in items if it.get("kind") == "credit"]
        if credits_h and big["h"] < 1.3 * max(credits_h):
            is_title = False
        for it in rest:
            it["kind"] = "title" if (it is big and is_title) else "tempo"
    return items


# 가사 줄에 섞여 들어올 수 있는 악보 지시어(가사 아님) — 소문자 비교
_NAV_WORDS = {"fine", "d.c", "d.s", "coda", "segno", "d.c. al fine", "d.s. al fine"}
_NON_LYRIC_WORDS = {"fine", "d.s.", "d.c.", "al", "coda", "segno", "d.s", "d.c", "to",
                    # 셈여림·빠르기 등(가사 자리에 글자로 들어 있는 경우) — 가사가 아님
                    "p", "pp", "ppp", "mp", "mf", "f", "ff", "fff", "sf", "sfz", "fp",
                    "cresc.", "dim.", "rit.", "rall.", "piano"}


_MUSIC_FONT_RE = re.compile(r"opus|maestro|petrucci|sonata|jazz|engraver|broadway|bravura|leland|"
                            r"emmentaler|feta|gootville|petaluma|finale|musiqwik|toccata|november|"
                            r"academico|bach|ekmelos|nwc\d|noteworthy|musicalsymbols|capella|encore|"
                            r"anastasia|tamburo|mscore|musescore|lilypond|sibelius|inkpen|reprise|helsinki|"
                            r"norfolk|scoring|golden\s*age|fughetta|vivace|musisync|seville", re.I)


def _pdf_chars(pdf_path, page_index, dpi, crop_x0, crop_y0, music_marks=None, music_digits=None):
    """PDF 글자 목록 [(top, bottom, left, right, char, 앞에_띄어쓰기, 굵게)] (크롭 이미지 픽셀 좌표).

    PDF마다 텍스트를 넣는 단위가 다르다(교가=글자 하나씩, 교과서 PDF=단어 단위 + 크기 0인
    찌꺼기 객체). 그래서 페이지 객체가 아니라 textpage의 '글자 단위' 상자를 읽는다.
    [악보 글꼴 제외 — 실사용 검증] Finale로 만든 PDF("봄이 오면")는 음표·음자리표까지
    악보 전용 글꼴(OpusStd 등)의 평범한 문자(œ, &, ? …)로 넣는다. 글꼴 이름으로 가려 뺀다.
    [보이지 않는 글자 제외] 흰색·투명으로 들어 있는 글자(화면엔 안 보임)는 뺀다.
    [굵게] 글꼴 이름에 Bold가 있거나, 같은 글자를 살짝 어긋나게 겹쳐 찍은 "가짜 굵게"면 굵게로 본다.
    """
    import ctypes
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[page_index]
    tp = page.get_textpage()
    scale = dpi / 72.0
    page_h = page.get_size()[1]
    font_cache = {}

    def _font_name(i):
        try:
            n = raw.FPDFText_GetFontInfo(tp.raw, i, None, 0, None)
            if n <= 0:
                return ""
            buf = ctypes.create_string_buffer(n)
            raw.FPDFText_GetFontInfo(tp.raw, i, buf, n, None)
            return buf.value.decode("latin1", "replace")
        except Exception:
            return ""

    def _visible(i):
        try:
            r_, g_, b_, a_ = (ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint())
            if raw.FPDFText_GetFillColor(tp.raw, i, r_, g_, b_, a_):
                if a_.value == 0 or (r_.value > 240 and g_.value > 240 and b_.value > 240):
                    return False
        except Exception:
            pass
        return True

    # [v3.61] 이름을 모르는 악보 글꼴도 모양으로 가린다. 악보 글꼴은 글자 칸(글꼴 높이)은 오선만큼 큰데
    # 실제 글자 그림은 작다(NoteWorthy 'NWC2STDA': 칸 38pt에 그림 6~15pt). 영숫자 글자의
    # (그림 높이 ÷ 칸 높이) 중앙값이 0.45 미만이고, 칸 높이가 이 페이지 글자들의 1.5배 이상이면 악보 글꼴.
    # (실사용: 가톨릭 성가 PDF — 음표 글자 a·f·k·j가 글자로 읽혀 코드 이름·제목이 4배 크기로 그려졌다)
    fname_of, stat = {}, {}
    n_chars = tp.count_chars()
    for i in range(n_chars):
        ch = tp.get_text_range(i, 1)
        if not ch or ch.isspace() or not _is_text_char(ch):
            continue
        fn = _font_name(i)
        fname_of[i] = fn
        if ch.isalnum():
            try:
                _l, b0, _r, t0 = tp.get_charbox(i, loose=False)
                _l, b1, _r, t1 = tp.get_charbox(i, loose=True)
            except Exception:
                continue
            if t1 - b1 > 0.5:
                stat.setdefault(fn, []).append(((t0 - b0) / (t1 - b1), t1 - b1))
    shape_music = set()
    all_h = sorted(h for v in stat.values() for _r, h in v)
    if all_h:
        med_h = all_h[len(all_h) // 2]
        for fn, v in stat.items():
            if len(v) < 4 or _MUSIC_FONT_RE.search(fn):
                continue
            ratios = sorted(r for r, _h in v)
            hs = sorted(h for _r, h in v)
            if ratios[len(ratios) // 2] < 0.45 and hs[len(hs) // 2] >= 1.5 * med_h:
                shape_music.add(fn)
        if len(shape_music) == len(stat) and len(stat) > 0:
            shape_music = set()                  # 전부 악보 글꼴일 리는 없다 — 판단 보류

    out = []
    pending_space = False
    for i in range(n_chars):
        ch = tp.get_text_range(i, 1)
        if ch and ch.isspace():
            pending_space = True                 # 다음 글자 앞에 띄어쓰기가 있었다는 표시
            continue
        if not ch:
            continue
        is_text = _is_text_char(ch)
        fname = (fname_of.get(i) or _font_name(i)) if (is_text or music_marks is not None) else ""
        in_music_font = bool(_MUSIC_FONT_RE.search(fname)) or fname in shape_music
        if music_marks is not None and (not is_text or in_music_font):
            kind = _music_note_kind(ch, in_music_font)
            if kind and _visible(i):
                l, b, r, t = tp.get_charbox(i)
                music_marks.append(((page_h - t) * scale - crop_y0, (page_h - b) * scale - crop_y0,
                                    l * scale - crop_x0, r * scale - crop_x0, kind))
        if music_digits is not None and in_music_font and ch.isdigit() and _visible(i):
            l, b, r, t = tp.get_charbox(i, loose=False)      # [v3.80] 악보 글꼴의 숫자 = 박자표 숫자
            music_digits.append(((page_h - t) * scale - crop_y0, (page_h - b) * scale - crop_y0,
                                 l * scale - crop_x0, r * scale - crop_x0, ch))
        if not is_text or in_music_font or not _visible(i):
            continue
        l, b, r, t = tp.get_charbox(i)
        if r - l < 0.05 and t - b < 0.05:       # 크기 0 찌꺼기(점(.)처럼 작은 글자는 남긴다)
            continue
        out.append([(page_h - t) * scale - crop_y0, (page_h - b) * scale - crop_y0,
                    l * scale - crop_x0, r * scale - crop_x0, ch, pending_space,
                    "bold" in fname.lower()])
        pending_space = False
    kept = []
    for c in sorted(out):
        hh = max(1.0, c[1] - c[0])
        dup = next((k for k in kept if k[4] == c[4] and abs(k[2] - c[2]) < 0.35 * hh
                    and abs(k[0] - c[0]) < 0.35 * hh), None)
        if dup is not None:
            dup[5] = dup[5] or c[5]              # 겹친 사본 쪽에 띄어쓰기 표시가 있었을 수도 있음
            dup[6] = True                        # 같은 글자 겹쳐 찍기 = 가짜 굵게
            continue
        kept.append(c)
    return [tuple(k) for k in kept]


def _is_hangul(ch):
    return "\uac00" <= ch <= "\ud7a3"


def extract_lyric_tokens(pdf_path, page_index, dpi, crop_x0, crop_y0, staves, interline_px):
    """[가사 — 실사용 요청: PDF에서 인식되는 경우만] 보표마다 그 바로 아래의 글자를
    줄(=절) 단위로 모아 음절/단어 토큰으로 돌려준다.
    staves: [(line_idx, staff_no, top_y, bottom_y, next_top_y, margin), ...]
      next_top_y는 바로 아래 보표(같은 시스템) 또는 다음 시스템 첫 보표의 맨 윗줄,
      margin은 그 위로 비워 둘 높이(코드기호·빠르기말 자리).
    반환: [{"text","x","y","h","line","staff","row"}, ...]
    [버그 수정] 예전엔 시스템의 '맨 아래 보표' 밑만 봐서, 노래+피아노 악보처럼 가사가
    노래 보표 바로 밑(피아노 위)에 있으면 하나도 못 찾았다("봄이 오면").
    한글은 한 글자=한 음절, 라틴 문자는 붙은 글자끼리 한 단어, '-'·'―'는 별도 토큰."""
    il = interline_px
    chars = _pdf_chars(pdf_path, page_index, dpi, crop_x0, crop_y0)
    tokens = []
    for (line_idx, staff_no, top, bottom, nxt_top, margin) in staves:
        lo = bottom + 0.3 * il
        hi = bottom + 9 * il
        if nxt_top is not None:
            hi = min(hi, nxt_top - margin)
        zone = [c for c in chars if lo < (c[0] + c[1]) / 2 < hi]
        # [v3.62] 마침표·쉼표 같은 작은 글자는 줄을 새로 만들지 않고 가장 가까운 줄에 붙인다.
        # (실사용: 성가 "1.불"의 '.'이 글자 아래쪽에 있어 혼자 한 줄(절)이 되고, 그 줄의 '.'이 첫 음의 가사로
        #  들어가 '불'이 밀려났다 — 절 줄 수도 하나씩 늘어났다)
        small = [c for c in zone if (c[1] - c[0]) < 0.35 * il]
        rows = []
        for c in sorted((c for c in zone if (c[1] - c[0]) >= 0.35 * il), key=lambda c: (c[0] + c[1]) / 2):
            cy = (c[0] + c[1]) / 2
            for rw in rows:
                if abs(rw["cy"] - cy) < 0.7 * il:
                    rw["c"].append(c)
                    rw["cy"] = sum((k[0] + k[1]) / 2 for k in rw["c"]) / len(rw["c"])
                    break
            else:
                rows.append({"cy": cy, "c": [c]})
        for c in small:
            if not rows:
                break
            cy = (c[0] + c[1]) / 2
            rw = min(rows, key=lambda r: abs(r["cy"] - cy))
            if abs(rw["cy"] - cy) < 1.0 * il:
                rw["c"].append(c)
        ri = 0
        for rw in sorted(rows, key=lambda r: r["cy"]):
            cs = sorted(rw["c"], key=lambda c: c[2])
            h = max(c[1] - c[0] for c in cs)
            words = []
            for c in cs:
                ch = c[4]
                prev = words[-1][-1] if words else None
                join = (prev is not None and not _is_hangul(ch) and not _is_hangul(prev[4])
                        and ch not in "-―" and prev[4] not in "-―" and c[2] - prev[3] < 0.3 * h)
                punct = prev is not None and ch in ",.!?;:'\")" and c[2] - prev[3] < 0.3 * h
                if join or punct:
                    words[-1].append(c)
                else:
                    words.append([c])
            row_tokens = []
            verse_left = None
            for w in words:
                text = "".join(c[4] for c in w)
                if re.fullmatch(r"\d+[.)]?", text):
                    verse_left = w[0][2]         # 절 번호(1. 2.)는 가사가 아님 — 바로 뒤 음절과 한 덩어리로 놓인다
                    continue
                if text.lower() in _NON_LYRIC_WORDS:
                    continue                     # 셈여림 글자 등
                x_ = (w[0][2] + w[-1][3]) / 2
                if verse_left is not None:
                    # [v3.62] "1.불"처럼 절 번호 뒤 첫 음절은 '절 번호+음절' 전체가 음표 아래 가운데에 온다
                    # (음절만의 가운데는 음표에서 1.6칸 넘게 오른쪽이라 붙지 못하고 빠졌다)
                    x_ = (verse_left + w[-1][3]) / 2
                    verse_left = None
                tok = {"text": "-" if text == "―" else text,
                       "x": x_, "y": rw["cy"], "h": h,
                       "line": line_idx, "staff": staff_no, "row": ri}
                if tok["text"] == "-" and len(w) == 1:        # [v3.71] 붙임표의 실제 높이·길이(원본 그대로 긋기)
                    tok["hdy"] = (w[0][0] + w[0][1]) / 2 - rw["cy"]   # 줄 가운데에서 얼마나 위·아래인지
                    tok["hw"] = w[0][3] - w[0][2]
                row_tokens.append(tok)
            if row_tokens:
                tokens.extend(row_tokens)
                ri += 1
    return tokens


def extract_other_text(pdf_path, page_index, dpi, crop_x0, crop_y0, header_limit_y,
                       lyric_zones, exclude_boxes, interline_px):
    """[나머지 글자 — 실사용 요청] 제목·가사·셈여림으로 쓰이지 않은 PDF 글자(빠르기말
    "Andante", 악기 이름 "Piano", 절 번호 "1." 등)를 원본 자리 그대로 표시하기 위해 뽑는다.
    - header_limit_y 위쪽: 제목·작사작곡 영역(header_items가 따로 그림) → 제외
    - 가사 영역(lyric_zones의 (위, 아래, 보표 시작 x)): 가사는 음표에 붙여 그리므로 제외하되,
      절 번호(숫자와 점)와 보표 시작선 왼쪽의 글자(악기 이름 "Piano")는 남긴다
    - exclude_boxes: 셈여림 기호로 그리는 자리(글자로 또 그리면 겹침) → 제외
    - 숫자만 있는 단어(마디 번호)는 편집기가 직접 그리므로 제외
    반환: [{"text","kind":"text","x0","x1","y","h"}, ...]"""
    il = interline_px
    chars = _pdf_chars(pdf_path, page_index, dpi, crop_x0, crop_y0)

    def _hit(c, b):
        ix = max(0.0, min(c[3], b[2]) - max(c[2], b[0]))
        iy = max(0.0, min(c[1], b[3]) - max(c[0], b[1]))
        return ix * iy > 0.3 * max(1.0, (c[3] - c[2]) * (c[1] - c[0]))

    # [v3.62] 가사 자리에 있어도 이동 기호 낱말(Fine·D.C.·D.S.·Coda…)은 가사가 아니다 — 가사에서도 빼고
    # 여기서도 빼면 아예 사라졌다(실사용: 성가 308번 가사 줄 높이의 "Fine")
    rescue = set()
    lines_ = []
    for i in sorted(range(len(chars)), key=lambda i: (chars[i][0] + chars[i][1]) / 2):
        c = chars[i]
        for ln in lines_:
            if c[0] < ln[1] and c[1] > ln[0]:
                ln[2].append(i)
                ln[0], ln[1] = min(ln[0], c[0]), max(ln[1], c[1])
                break
        else:
            lines_.append([c[0], c[1], [i]])
    for _t, _b, idx in lines_:
        word = []
        for i in sorted(idx, key=lambda i: chars[i][2]) + [None]:
            if i is not None and word:
                p_, c = chars[word[-1]], chars[i]
                if -0.4 * max(1.0, p_[1] - p_[0]) <= c[2] - p_[3] < 0.35 * max(1.0, p_[1] - p_[0]) and not _is_hangul(c[4]):
                    word.append(i)
                    continue
            if word and "".join(chars[k][4] for k in word).lower().strip(".") in _NAV_WORDS:
                rescue.update(word)
            word = [i] if (i is not None and not _is_hangul(chars[i][4])) else []
    keep = []
    for ci, c in enumerate(chars):
        cy = (c[0] + c[1]) / 2
        if c[1] < header_limit_y:
            continue
        in_lyric = any(lo < cy < hi and c[2] >= xl for lo, hi, xl in lyric_zones)
        if in_lyric and not re.fullmatch(r"[\d.)]", c[4]) and ci not in rescue:
            continue                          # 가사(음표에 붙여 그림) — 보표 시작선 왼쪽 글자(악기 이름)는 제외 안 함
        if any(_hit(c, b) for b in exclude_boxes):
            continue
        keep.append(c)
    # 줄 묶기: 세로 범위가 겹치면 같은 줄(가운데 높이로 비교하면 아래쪽에 붙는 점(.)이
    # 다른 줄로 갈라져 "1." 같은 절 번호의 점이 빠졌다)
    lines = []
    for c in sorted(keep, key=lambda c: (c[0] + c[1]) / 2):
        for ln in lines:
            if c[0] < ln["bot"] and c[1] > ln["top"]:
                ln["c"].append(c)
                ln["top"], ln["bot"] = min(ln["top"], c[0]), max(ln["bot"], c[1])
                break
        else:
            lines.append({"top": c[0], "bot": c[1], "c": [c]})
    items = []
    for ln in lines:
        cs = sorted(ln["c"], key=lambda c: c[2])
        h = max(c[1] - c[0] for c in cs)
        runs = [[cs[0]]]
        for c in cs[1:]:
            if c[2] - runs[-1][-1][3] > max(1.5 * h, 1.2 * il):
                runs.append([c])              # 같은 높이라도 멀리 떨어져 있으면 별개 글자 묶음
            else:
                runs[-1].append(c)
        for run in runs:
            text, prev_r = "", None
            for c in run:
                if prev_r is not None and (c[5] or c[2] - prev_r > h * 0.8):
                    text += " "
                text += c[4]
                prev_r = c[3]
            text = text.strip()
            # 절 번호: 가사 줄 안 + 보표 시작선보다 충분히 안쪽(마디 번호는 시작선 근처 위쪽)
            in_lyric_run = any(lo < (run[0][0] + run[0][1]) / 2 < hi and run[0][2] > xl + 2.5 * il
                               for lo, hi, xl in lyric_zones)
            if not text or (re.fullmatch(r"\d+", text) and not in_lyric_run):
                continue                      # 마디 번호(가사 줄 안의 숫자는 절 번호라 남김)
            if re.fullmatch(r"[.\-~=]+", text):
                continue                      # 점·기호만 남은 조각
            items.append({"text": text, "kind": "text", "x0": run[0][2], "x1": run[-1][3],
                          "y": (min(c[0] for c in run) + max(c[1] for c in run)) / 2,
                          "h": max(c[1] - c[0] for c in run),
                          "bold": sum(1 for c in run if c[6]) * 2 > len(run)})
    return items


def extract_time_signatures(pdf_path, page_index, dpi, crop_x0, crop_y0):
    """[v3.80] PDF 글자 층에서 박자표를 읽는다 — 악보 글꼴(NoteWorthy 등)로 찍은 숫자가 위·아래로 쌓인 것.
    그림을 인식하는 Audiveris는 이 숫자를 2분음표 화음으로 잘못 읽기도 한다(성가 324번: 줄마다 3/8·4/8이
    바뀌는데 모두 음표로 읽혀 마디가 넘쳤다). 글자 층은 원본 그 자체라 틀릴 일이 없다.
    반환: [{"beats", "beat_type", "x0", "x1", "y0", "y1"}] (인식 좌표)."""
    digs = []
    _pdf_chars(pdf_path, page_index, dpi, crop_x0, crop_y0, music_digits=digs)
    if len(digs) < 2:
        return []
    # 가로로 붙은 숫자를 한 수로(12 등)
    nums = []
    for d in sorted(digs, key=lambda d: (round((d[0] + d[1]) / 2 / max(1.0, (d[1] - d[0]) * 0.5)), d[2])):
        h = max(1.0, d[1] - d[0])
        if nums and abs((nums[-1]["y0"] + nums[-1]["y1"]) / 2 - (d[0] + d[1]) / 2) < 0.3 * h \
                and -0.2 * h <= d[2] - nums[-1]["x1"] < 0.5 * h:
            nums[-1]["text"] += d[4]
            nums[-1]["x1"] = d[3]
        else:
            nums.append({"text": d[4], "y0": d[0], "y1": d[1], "x0": d[2], "x1": d[3]})
    out, used = [], set()
    for i, a in enumerate(nums):
        if i in used:
            continue
        h = max(1.0, a["y1"] - a["y0"])
        best = None
        for j, b in enumerate(nums):
            if j == i or j in used:
                continue
            dx = abs((a["x0"] + a["x1"]) / 2 - (b["x0"] + b["x1"]) / 2)
            dy = b["y0"] - a["y0"]                       # b가 아래
            if dx < 0.8 * h and 0.7 * h <= dy <= 1.6 * h and (best is None or dx < best[0]):
                best = (dx, j)
        if best is None:
            continue
        b = nums[best[1]]
        try:
            beats, bt = int(a["text"]), int(b["text"])
        except ValueError:
            continue
        if bt not in (1, 2, 4, 8, 16) or not (1 <= beats <= 24):
            continue
        used.update((i, best[1]))
        out.append({"beats": beats, "beat_type": bt, "x0": min(a["x0"], b["x0"]), "x1": max(a["x1"], b["x1"]),
                    "y0": a["y0"], "y1": b["y1"]})
    return out
