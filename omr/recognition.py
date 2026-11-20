# recognition_modules.py
# ---------------------------------------------------------------------------
# 악보 '인식' 단계의 세부 알고리즘 모음.
# modules.recognition() 이 객체를 하나씩 돌면서 여기의 함수들을 호출해
# 조표·박자·음표 등을 판정한다.
#
# [6단계] 조표 인식 — recognize_key
#   - 블로그 https://hackids.tistory.com/119 를 구현.
# [7단계] 음표 머리 인식 — recognize_note / recognize_note_head
#   - 블로그 https://hackids.tistory.com/120 를 구현.
# [8단계] 음표 꼬리(깃발) 인식 — recognize_note_tail
#   - 블로그 https://hackids.tistory.com/121 를 구현.
# [9단계] 음표 점(부점) 인식 + 박자 분류 — recognize_note_dot
#   - 블로그 https://hackids.tistory.com/122 를 구현.
# [10단계] 음정(음높이) 인식 — recognize_pitch
#   - 블로그 https://hackids.tistory.com/123 를 구현.
# [11단계] 쉼표 인식 — recognize_rest / recognize_rest_dot
#   - 블로그 https://hackids.tistory.com/124 를 구현.
# [12단계] 온음표 인식 — recognize_whole_note
#   - 블로그 https://hackids.tistory.com/125 를 구현.
# [13단계] 박자표 숫자 인식 — recognize_time_signature (블로그 범위 밖, 자체 구현)
#   - 박자표는 조표(또는 음자리표) 뒤, 위 2칸/아래 2칸에 숫자로 기입된다.
#   - 숫자의 '구멍(닫힌 공간) 개수'와 보조 형태로 2·3·4·6·8을 구분한다.
# ---------------------------------------------------------------------------
import cv2
import numpy as np

from . import functions as fs


# 음정 인덱스(0~20) → 음이름. recognize_pitch 의 pitch_lines 는 아래(staff[4]+30)에서
# 위로 반칸씩 올라가므로, 높은음자리표 기준으로 다음과 같이 대응한다.
# (조표 미반영 = 오선상 위치 기준. 조표에 따른 반음 조정은 MusicXML 단계에서 처리)
#   i=6=E4(맨아래줄), i=8=G4, i=10=B4, i=12=D5, i=14=F5(맨위줄), i=4=C4(가온다)
PITCH_NAMES = [
    "F3", "G3", "A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4", "B4",
    "C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6", "D6", "E6",
]


# ===========================================================================
# SMuFL 템플릿 매칭 — 박자표 숫자 등 '표준 형태 기호'의 강건한 분류
# ---------------------------------------------------------------------------
# 프로젝트에 포함된 Bravura(SMuFL 표준 악보 폰트)로 기준 글리프를 렌더링해
# 템플릿으로 삼고, 인식 대상 영역과 겹침(F1) + 종횡비 유사도로 비교한다.
# 휴리스틱(구멍 개수·가로막대 위치)보다 폰트·해상도 변화에 훨씬 강하다.
# score.smufl(폰트 로더)이 없거나 폰트 파일이 없으면 None 을 반환해
# 호출부가 기존 휴리스틱으로 폴백하게 한다(프로그램이 깨지지 않음).
# ===========================================================================

_TEMPLATE_PX = 64          # 템플릿 렌더링 크기(px). em = 오선 4칸 기준
_template_cache = {}


def _load_smufl():
    """score.smufl 모듈을 지연 로드한다(폰트 없으면 None)."""
    try:
        from score import smufl
        return smufl if smufl.font_available() else None
    except Exception:
        return None


def _tight_crop(binary):
    """이진 영역에서 내용(255)이 있는 최소 사각형만 잘라 돌려준다(없으면 None)."""
    ys, xs = np.nonzero(binary)
    if len(ys) == 0:
        return None
    return binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _render_glyph_template(char):
    """SMuFL 글리프 문자 하나를 이진 템플릿(내용만 tight crop)으로 렌더링한다."""
    if char in _template_cache:
        return _template_cache[char]
    smufl = _load_smufl()
    tmpl = None
    if smufl is not None:
        try:
            from PIL import Image, ImageDraw
            font = smufl.get_font(_TEMPLATE_PX)
            canvas = Image.new("L", (_TEMPLATE_PX * 3, _TEMPLATE_PX * 3), 0)
            d = ImageDraw.Draw(canvas)
            d.text((_TEMPLATE_PX * 1.5, _TEMPLATE_PX * 1.5), char,
                   font=font, fill=255, anchor="mm")
            arr = np.array(canvas)
            arr = (arr > 96).astype(np.uint8) * 255
            tmpl = _tight_crop(arr)
        except Exception:
            tmpl = None
    _template_cache[char] = tmpl
    return tmpl


def _shape_score(sub, tmpl):
    """대상 영역(sub)과 템플릿의 형태 유사도(0~1). 겹침 F1 × 종횡비 유사도."""
    if sub is None or tmpl is None or sub.size == 0 or tmpl.size == 0:
        return 0.0
    th, tw = tmpl.shape
    resized = cv2.resize(sub, (tw, th), interpolation=cv2.INTER_NEAREST)
    a = resized > 0
    b = tmpl > 0
    inter = np.count_nonzero(a & b)
    denom = np.count_nonzero(a) + np.count_nonzero(b)
    f1 = (2.0 * inter / denom) if denom else 0.0
    ar_s = sub.shape[1] / sub.shape[0]
    ar_t = tw / th
    aspect = min(ar_s, ar_t) / max(ar_s, ar_t)
    # 종횡비 차이는 서체(좁은 세리프체 vs 두꺼운 악보체)에 따라 크게 달라지므로
    # 완전 비례가 아닌 제곱근으로 완화해 반영한다.
    return f1 * (aspect ** 0.5)


_COMMON_TIME_MIN = 0.55   # 오선 범위로 제한한 뒤의 임계값(제한 전보다 훨씬 안전하게 상향 가능)


def find_common_time_window(image, stats, staves):
    """넓은(음자리표 등과 융합된) 객체 안에서 온박자·컷타임(𝄴·𝄵) 기호를 찾는다.

    클레프와 박자표가 붙어(또는 오선 잔여 픽셀로 다리가 놓여) 하나의 거대한
    객체로 검출되는 경우가 실제로 있다(실측: 어느 정도 오래된 스타일의 조판에서
    클레프+공통 박자표가 한 덩어리로 잡힘). 이때는 온박자 기호만 잘라낼 수
    없으므로, 객체 폭 안에서 여러 위치를 슬라이딩하며 온박자·컷타임 템플릿과
    비교해 가장 잘 맞는 구간을 찾는다.

    핵심: 세로 범위를 '오선 첫째~다섯째 줄'로 딱 제한하는 것이 정확도를 크게
    좌우한다. 클레프는 장식적인 고리가 오선 위·아래로 튀어나오는데, 그 부분이
    섞이면 슬라이딩 윈도우가 클레프 고리 자체를 온박자 기호로 오인한다(실측:
    전체 높이로 비교하면 점수 0.31~0.37로 클레프 위치가 최고점을 차지하고
    "cut"으로 오분류; 오선 범위로만 제한하면 진짜 위치에서 0.73으로 급등하고
    "common"으로 정확히 분류됨). 온박자 기호는 정확히 오선 높이에 맞춰
    그려지므로 이 제한이 안전하다.

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지.
    stats : tuple
        탐색 대상 객체의 (x, y, w, h, area).
    staves : list[float]
        해당 객체가 속한 보표의 오선 5개 y좌표(오선 첫째~다섯째 줄).

    Returns
    -------
    tuple[str | None, float, tuple[int, int] | None]
        (글리프 이름 "common"|"cut" 또는 None, 점수, 찾은 구간(x0,x1) 또는 None).
    """
    x, y, w, h, area = stats
    ss = (staves[4] - staves[0]) / 4.0
    if ss <= 0:
        return None, 0.0, None
    r0 = max(0, int(round(staves[0] - y)))
    r1 = min(h, int(round(staves[4] - y)))
    if r1 <= r0:
        return None, 0.0, None
    region = image[y + r0:y + r1, x:x + w].copy()
    if region.size == 0:
        return None, 0.0, None

    # 오선이 완전히 제거되지 않고 일부 행이 거의 풀폭으로 남아 있으면(실측:
    # 클레프와 융합된 넓은 객체에서 흔함) 어떤 윈도우를 잘라도 그 잔여 행이
    # 섞여 매칭 점수가 낮아진다. 폭의 85% 이상이 채워진 행은 잔여 오선으로
    # 보고 미리 지운다(글리프 자체가 이렇게 넓게 채워진 행을 가질 일은 없다).
    rowfill = np.count_nonzero(region, axis=1)
    region[rowfill >= w * 0.85, :] = 0

    win_w = max(4, int(round(ss * 2.2)))
    if win_w >= w:
        name, score = match_glyph(region, [("common", "\uE08A"), ("cut", "\uE08B")])
        return (name, score, (0, w)) if score >= _COMMON_TIME_MIN else (None, score, None)

    step = max(2, int(round(ss * 0.4)))
    best_name, best_score, best_span = None, 0.0, None
    for x0 in range(0, w - win_w + 1, step):
        sub = region[:, x0:x0 + win_w]
        name, score = match_glyph(sub, [("common", "\uE08A"), ("cut", "\uE08B")])
        if score > best_score:
            best_name, best_score, best_span = name, score, (x0, x0 + win_w)
    if best_score >= _COMMON_TIME_MIN:
        return best_name, best_score, best_span
    return None, best_score, None


def match_glyph(region, glyph_names):
    """영역을 여러 SMuFL 글리프와 비교해 (최고 글리프 이름, 점수)를 반환.

    폰트가 없으면 (None, 0.0)을 반환한다 → 호출부는 휴리스틱으로 폴백.
    glyph_names 의 원소는 score.smufl.GLYPHS 의 키("restQuarter" 등) 또는
    ("이름", "글리프문자") 튜플이다.
    """
    smufl = _load_smufl()
    if smufl is None:
        return None, 0.0
    sub = _tight_crop((region > 0).astype(np.uint8) * 255)
    if sub is None:
        return None, 0.0
    best_name, best_score = None, 0.0
    for g in glyph_names:
        if isinstance(g, tuple):
            name, char = g
        else:
            name, char = g, smufl.GLYPHS.get(g)
        if not isinstance(char, str):
            continue
        s = _shape_score(sub, _render_glyph_template(char))
        if s > best_score:
            best_name, best_score = g if isinstance(g, str) else name, s
    return best_name, best_score


def recognize_key(image, staves, stats):
    """한 객체가 '박자표'인지 판정하고, 아니라면 '조표'(플랫/샾)로 해석한다.

    핵심 아이디어
    -------------
    박자표는 항상 오선 정중앙에, 일정한 크기(높이=오선 4칸, 너비=오선 1칸 정도)로
    나타난다. 따라서 박자표는 비교적 정확히 찾을 수 있고, '박자표를 만나기 전'의
    객체들은 모두 조표로 간주할 수 있다(음자리표는 recognition에서 미리 제외).

    - 박자표 조건을 만족하면 (True, 0) 을 반환 → 조표 탐색 종료 신호.
    - 아니면 조표로 보고 기둥(세로선) 위치로 플랫/샾을 구분:
        · 첫 기둥이 객체 앞쪽에 있으면(빨리 발견) → 플랫, key = 100 × 플랫수
        · 첫 기둥이 객체 뒤쪽에 있으면(늦게 발견) → 샾,  key = 10 × 샾수
          (샾은 세로선이 2개이므로 len(stems)/2 가 실제 샾 개수)

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지(기둥 검출용). 반드시 1채널을 넘긴다.
    staves : list[float]
        해당 객체가 속한 보표의 오선 5개 y좌표.
    stats : tuple
        객체의 (x, y, w, h, area).

    Returns
    -------
    tuple[bool, int]
        (박자표 여부, 이 객체가 기여하는 key 값).
        박자표면 (True, 0), 조표면 (False, key), 판정 불가면 (False, 0).
    """
    (x, y, w, h, area) = stats
    ts_conditions = (
        staves[0] + fs.weighted(5) >= y >= staves[0] - fs.weighted(5) and          # 상단: 첫 오선 부근
        staves[4] + fs.weighted(5) >= y + h >= staves[4] - fs.weighted(5) and       # 하단: 마지막 오선 부근
        staves[2] + fs.weighted(5) >= fs.get_center(y, h) >= staves[2] - fs.weighted(5) and  # 중단: 3번째 오선 부근
        fs.weighted(18) >= w >= fs.weighted(10) and                                 # 너비 조건
        fs.weighted(45) >= h >= fs.weighted(35)                                     # 높이 조건
    )
    if ts_conditions:
        return True, 0

    # 온박자(𝄴)·컷타임(𝄵) 기호는 숫자 박자표보다 세로로 길어(오선 거의 전체를
    # 채움) 위 ts_conditions(높이 상한 45)를 벗어나는 경우가 있고, 클레프와
    # 붙어 훨씬 넓은 객체로 검출되기도 한다(실측 확인). 그런 경우를 위해
    # 객체가 클레프 하나보다 뚜렷이 넓을 때(=붙어 있을 가능성) 슬라이딩
    # 윈도우로 온박자·컷타임 템플릿을 다시 찾아본다. 못 찾으면(name is None)
    # 아래의 일반 조표 판정 로직으로 그대로 이어간다.
    ss = (staves[4] - staves[0]) / 4.0
    if w > fs.weighted(30) and ss > 0:
        name, score, span = find_common_time_window(image, stats, staves)
        if name is not None:
            return True, 0

    # 조표(다장조를 제외한 모든 조)
    stems = fs.stem_detection(image, stats, 20)
    if len(stems) == 0:
        # 세로선이 전혀 없으면 조표(플랫/샾)로 볼 수 없다(방어 처리).
        return False, 0
    if stems[0][0] - x >= fs.weighted(3):   # 첫 기둥이 늦게(뒤에서) 발견 → 샾
        key = int(10 * len(stems) / 2)
    else:                                    # 첫 기둥이 일찍(앞에서) 발견 → 플랫
        key = 100 * len(stems)

    return False, key


_ACCIDENTAL_MIN = 0.45     # 임시표로 확정할 최소 매칭 점수


def split_leading_accidental(image, stats, ss):
    """객체 왼쪽 끝에 붙은 임시표(#,♭,♮)를 찾아 분리한다.

    마디 중간의 임시표는 조표와 달리 별도 객체로 안 잡히고, 바로 뒤 음표와
    닫힘(closing) 연산에서 합쳐져 하나의 객체가 되는 일이 흔하다(실측 확인:
    샤프가 빔그룹과 같은 객체로 잡힘). 그대로 두면 기둥 개수·머리 위치 판정이
    임시표의 획까지 포함해 흔들린다.

    분리 방법: 객체 폭 앞쪽(오선 2.5칸 이내)에서 '완전히 빈 열'(진짜 틈)을
    찾는다. 임시표는 음표(머리·기둥)와 붙어 있어도 실측상 그 사이에 1px
    이상의 빈 열이 남는다(둘이 서로 다른 획이므로). 그 틈 왼쪽을 잘라
    SMuFL 임시표 템플릿과 비교해 확인한다.

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지.
    stats : tuple
        (x, y, w, h, area) — 대상 객체.
    ss : float
        오선 한 칸 간격(px).

    Returns
    -------
    tuple[str | None, tuple]
        (임시표 종류 "sharp"|"flat"|"natural" 또는 None, 잘라낸 뒤의 새 stats).
        못 찾으면 (None, 원래 stats).
    """
    x, y, w, h, area = stats
    search_w = int(round(ss * 2.5))
    if w <= search_w + 2:
        return None, stats
    band = image[y:y + h, x:x + min(w, search_w + int(ss))]
    colsum = np.count_nonzero(band, axis=0)
    gaps = np.nonzero(colsum == 0)[0]
    gaps = gaps[gaps >= max(2, int(ss * 0.4))]         # 너무 이른 틈(잡티)은 제외
    if len(gaps) == 0:
        return None, stats

    cut = int(gaps[0])
    lead = image[y:y + h, x:x + cut]
    tight = _tight_crop(lead)
    if tight is None:
        return None, stats
    # 임시표 크기 범위(실측: 폭 0.6~1.6칸, 높이 1.3~3.2칸) 벗어나면 다른 것으로 본다.
    th, tw = tight.shape
    if not (ss * 0.5 <= tw <= ss * 1.8 and ss * 1.2 <= th <= ss * 3.4):
        return None, stats

    smufl = _load_smufl()
    if smufl is None:
        return None, stats
    name, score = match_glyph(tight, [
        ("sharp", "\uE262"), ("flat", "\uE260"), ("natural", "\uE261")])
    if name is None or score < _ACCIDENTAL_MIN:
        return None, stats

    new_x = x + cut
    new_w = w - cut
    return name, (new_x, y, new_w, h, area)


def detect_grace_slash(image, stem_x, note_y, stem_up, ss):
    """음표 기둥에 짧은 사선(꾸밈음/acciaccatura 표시)이 있는지 확인한다.

    표준 표기: 꾸밈음은 보통 음표보다 작게 그려지고, 기둥에 머리 쪽 1/3
    지점 부근을 가로지르는 짧은 대각선이 붙는다. 이 대각선은 기둥 자체보다
    훨씬 넓게 퍼져(기둥 폭의 3배 이상) 지나가므로, '기둥 근처 창에서 잉크가
    기둥 폭보다 훨씬 넓게 퍼지는 행이 있는가'로 판정한다.

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지.
    stem_x : int
        기둥의 x좌표(머리와 붙는 지점).
    note_y : int
        음표 머리의 y좌표.
    stem_up : bool
        기둥이 위로 향하는지(True) 아래로 향하는지(False).
    ss : float
        오선 한 칸 간격(px).

    Returns
    -------
    bool
        사선이 발견되면 True.
    """
    H, W = image.shape
    half = max(2, int(round(ss * 0.9)))
    if stem_up:
        y0, y1 = max(0, int(note_y - ss * 1.9)), max(0, int(note_y - ss * 0.3))
    else:
        y0, y1 = min(H, int(note_y + ss * 0.3)), min(H, int(note_y + ss * 1.9))
    x0, x1 = max(0, int(stem_x - half)), min(W, int(stem_x + half))
    if y1 <= y0 or x1 <= x0:
        return False
    band = image[y0:y1, x0:x1]
    rowspan = np.count_nonzero(band, axis=1)
    return bool(np.any(rowspan >= (x1 - x0) * 0.6))


def recognize_note(image, vis, staff, stats, stems, direction):
    """한 객체가 '음표'의 최소 조건을 만족하면 각 기둥마다 머리를 인식한다.

    음표 최소 조건: 기둥이 있고(len(stems)), 너비 weighted(10) 이상,
    높이 weighted(35) 이상, 픽셀 수(area) weighted(95) 이상.
    한 객체 안에 기둥이 여러 개일 수 있으므로(예: 8분음표 두 개가 빔으로 연결)
    기둥 개수만큼 머리를 탐색한다.

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지(머리 특징 분석용).
    vis : np.ndarray
        BGR 시각화 이미지(판정 결과를 글자로 표시).
    staff : list[float]
        해당 보표의 오선 5개 y좌표(다음 단계 음정 인식에 사용 예정).
    stats : tuple
        객체의 (x, y, w, h, area).
    stems : list
        객체의 기둥 리스트.
    direction : bool
        음표 방향(True=정방향, False=역방향).

    Returns
    -------
    tuple[list, list]
        (notes, pitches). notes = 음표 종류(박자) 코드 리스트
        (2/4/8/16/32 = 각 분음표, 음수 = 점음표). pitches = 음이름(다음 단계에서 채움).
    """
    (x, y, w, h, area) = stats
    notes = []
    pitches = []
    stem_xs = []
    group_ids = []

    # [마디선/도돌이표 배제] 마디선·도돌이표의 세로선도 '기둥'으로 검출되지만
    # 음표가 아니다. 마디선은 오선 첫 줄~다섯째 줄을 '완전히 덮는다'(위로도
    # 아래로도 오선 밖까지 이어짐). 음표 기둥은 머리에 붙어 있어 한쪽 끝이
    # 반드시 오선 안쪽에서 끝난다. 모든 기둥이 오선을 완전히 덮으면 배제한다.
    # (실측: 마디선·도돌이표 전부 True / 진짜 음표 전부 False 로 완전 분리)
    if stems and staff is not None and len(staff) >= 5:
        st_top, st_bot = staff[0], staff[4]
        if all(s[1] <= st_top + 1 and (s[1] + s[3]) >= st_bot - 1 for s in stems):
            return notes, pitches, stem_xs, group_ids

    note_condition = (
        len(stems) and
        w >= fs.weighted(10) and      # 너비 조건
        h >= fs.weighted(35) and      # 높이 조건
        area >= fs.weighted(95)       # 픽셀 개수 조건
    )
    if note_condition:
        # 빔 음표(기둥 2개 이상): 기둥 사이 빔 개수로 각 음표의 박자를 정한다.
        # (각 기둥 옆 한 열에서 꼬리를 세면 빔이 과다 계산되어 32분음표로 오분류됨)
        beam_counts = count_beams(image, stems, direction) if len(stems) >= 2 else None
        links = beam_links(image, stems, direction) if len(stems) >= 2 else []
        _BEAM2BASE = {1: 8, 2: 16, 3: 32}

        # 빔 연결이 끊긴 지점에서 하위 그룹을 나눈다(이음줄로만 붙은 음표 분리).
        group_seq = [0]
        for i in range(len(links)):
            group_seq.append(group_seq[-1] if links[i] else group_seq[-1] + 1)

        # 각 기둥이 '실제로 빔에 물려 있는가' = 좌 또는 우 이웃과 빔으로 연결.
        # 빔에 물리지 않은 기둥(이음줄로만 붙은 음표)은 단독 음표로 분류해야 한다.
        # (그렇지 않으면 count_beams 가 최소 1을 강제해 4분음표가 8분음표가 된다.)
        in_beam = []
        for i in range(len(stems)):
            left = links[i - 1] if i - 1 >= 0 and i - 1 < len(links) else False
            right = links[i] if i < len(links) else False
            in_beam.append(bool(left or right))

        for i in range(len(stems)):
            stem = stems[i]
            head_exist, head_fill, head_center = recognize_note_head(image, stem, direction)
            if head_exist:
                # 꼬리(깃발) 개수를 먼저 센다. 부점 탐색 구역은 정방향 단독 음표의
                # 꼬리와 겹치므로, 꼬리 수를 넘겨 임계값을 올려야 꼬리를 점으로
                # 오인하지 않는다(기존엔 0으로 고정 전달해 16분음표에 가짜 점이 붙었다).
                tail_cnt = recognize_note_tail(image, i, stem, direction)
                dot_exist = recognize_note_dot(image, stem, direction,
                                               tail_cnt, len(stems))

                if beam_counts is not None and in_beam[i]:
                    # 빔에 물린 음표: 빔 개수 → 박자(8/16/32분).
                    base = _BEAM2BASE.get(beam_counts[i], 8)
                    note = -base if dot_exist else base
                else:
                    # 단일 음표(또는 빔에 안 물린 음표): 머리 채움 + 꼬리 수 + 점
                    note = _classify_single_note(head_fill, tail_cnt, dot_exist)

                if note is not None:
                    pitch = recognize_pitch(image, staff, head_center)
                    notes.append(note)
                    pitches.append(pitch)
                    stem_xs.append(int(stem[0]))
                    # 빔에 물리지 않은 음표는 단독(-1). 물린 음표만 그룹키 부여.
                    if len(stems) >= 2 and in_beam[i]:
                        group_ids.append(group_seq[i])
                    else:
                        group_ids.append(-1)
                    fs.put_text(vis, note,
                                (stem[0] - fs.weighted(10),
                                 stem[1] + stem[3] + fs.weighted(30)))
                    if pitch is not None:
                        fs.put_text(vis, PITCH_NAMES[pitch],
                                    (stem[0] - fs.weighted(10),
                                     stem[1] + stem[3] + fs.weighted(48)))

    return notes, pitches, stem_xs, group_ids


def _classify_single_note(head_fill, tail_cnt, dot_exist):
    """단일 음표(기둥 1개)를 머리 채움·꼬리 수·점으로 분류해 박자 코드를 돌려준다."""
    table = (
        ((not head_fill and tail_cnt == 0 and not dot_exist), 2),
        ((not head_fill and tail_cnt == 0 and dot_exist), -2),
        ((head_fill and tail_cnt == 0 and not dot_exist), 4),
        ((head_fill and tail_cnt == 0 and dot_exist), -4),
        ((head_fill and tail_cnt == 1 and not dot_exist), 8),
        ((head_fill and tail_cnt == 1 and dot_exist), -8),
        ((head_fill and tail_cnt == 2 and not dot_exist), 16),
        ((head_fill and tail_cnt == 2 and dot_exist), -16),
        ((head_fill and tail_cnt == 3 and not dot_exist), 32),
        ((head_fill and tail_cnt == 3 and dot_exist), -32),
    )
    for cond, note in table:
        if cond:
            return note
    return None


def recognize_note_head(image, stem, direction):
    """기둥 위치를 기준으로 음표 머리 구역을 정하고 특징점을 뽑아 판정한다.

    - 정방향 음표: 머리는 기둥의 '왼쪽 아래'에 위치.
    - 역방향 음표: 머리는 기둥의 '오른쪽 위'에 위치.
    머리 구역에서 다음을 계산한다.
      · cnt      : 끊기지 않고 이어진 가로선의 개수
      · pixel_cnt: 구역 내 픽셀 수
      · mid_max  : 머리 세로 범위의 '가운데 1/3 행'에서 가장 긴 가로선
    판정:
      · head_exist = cnt≥3 and pixel_cnt≥50   (머리가 있는가)
      · head_fill  = mid_max ≥ weighted(8)    (채워진 머리인가: 4분음표 등)

    head_fill 원리 — 빈 머리(2분음표)는 타원 테두리만 있어 가운데 행에서는
    좌/우 테두리의 짧은 런(1~3px)만 검출되고, 채운 머리(4분음표 이하)는 가운데
    행이 통째로 채워져 머리 폭 전체(11px 이상)의 런이 나온다. 실측 분포가
    빈≤6 / 채움≥11 로 완전히 분리되어 weighted(8)을 경계로 삼는다.
    (기존의 cnt·cnt_max·pixel_cnt 조합은 테두리가 두꺼운 작은 빈 머리에서
    세 값이 동시에 임계값을 넘겨 4분음표로 오인하는 문제가 있었다.)
    또한 다음 단계(음정)를 위한 머리 중심 y좌표(head_center)를 함께 반환한다.

    Returns
    -------
    tuple[bool, bool, float]
        (head_exist, head_fill, head_center).
    """
    (x, y, w, h) = stem
    if direction:                             # 정방향 음표: 기둥 왼쪽 아래
        area_top = y + h - fs.weighted(7)
        area_bot = y + h + fs.weighted(7)
        area_left = x - fs.weighted(14)
        area_right = x
    else:                                     # 역방향 음표: 기둥 오른쪽 위
        area_top = y - fs.weighted(7)
        area_bot = y + fs.weighted(7)
        area_left = x + w
        area_right = x + w + fs.weighted(14)

    # 이미지 경계 클리핑(구역이 이미지 밖으로 나가는 경우 방어)
    H, W = image.shape
    area_top = max(0, area_top)
    area_bot = min(H, area_bot)
    area_left = max(0, area_left)
    area_right = min(W - 1, area_right)

    cnt = 0          # 끊기지 않고 이어진 가로선 개수
    head_center = 0
    pixel_cnt = fs.count_rect_pixels(
        image, (area_left, area_top, area_right - area_left, area_bot - area_top))

    runs = []        # 행별 최장 가로런 길이(가운데 행 판정용)
    for row in range(area_top, area_bot):
        col, pixels = fs.get_line(image, fs.HORIZONTAL, row, area_left, area_right, 5)
        pixels += 1
        runs.append(pixels)
        if pixels >= fs.weighted(5):
            cnt += 1
            head_center += row

    # 머리(픽셀이 있는 행)의 세로 범위에서 가운데 1/3 행들의 최장 런
    rows_on = [i for i, p in enumerate(runs) if p >= 2]
    fill_ratio = 0.0
    if rows_on:
        r0, r1 = rows_on[0], rows_on[-1]
        third = (r1 - r0 + 1) // 3
        mid = runs[r0 + third: r1 - third + 1]
        # 채운 머리는 가운데 '연속 여러 행'이 모두 길다. 반면 덧줄(ledger line)이
        # 빈 머리를 가로지르면 '한두 행'만 길어진다. 따라서 최댓값 하나가 아니라
        # '가운데 행 중 긴 런의 비율'로 판정해 덧줄 간섭을 걸러낸다.
        if mid:
            long_rows = sum(1 for p in mid if p >= fs.weighted(8))
            fill_ratio = long_rows / len(mid)

    head_exist = (cnt >= 3 and pixel_cnt >= 50)
    head_fill = (fill_ratio >= 0.6)                  # 가운데 행 과반이 길면 채움
    head_center = head_center / cnt if cnt else 0   # cnt=0 방어(블로그엔 없음)

    return head_exist, head_fill, head_center


def recognize_note_tail(image, index, stem, direction):
    """음표 꼬리(깃발)의 개수를 센다. 8분음표=1, 16분음표=2, 4분음표=0.

    기둥 옆의 한 열을 세로로 훑어 픽셀 덩어리(깃발)가 몇 번 나타나는지 센다.
    빔으로 연결된 음표는 두 번째 음표부터(index>0) 꼬리가 기둥 '왼쪽'에 생기므로,
    index 값에 따라 탐색 열의 좌/우를 바꾼다.

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지.
    index : int
        객체 안에서 이 기둥이 몇 번째인지(0=첫 번째). 빔 연결 시 탐색 방향 결정.
    stem : list[int]
        기둥의 [x, y, w, h].
    direction : bool
        음표 방향(True=정방향, False=역방향). 꼬리 세로 탐색 범위를 정한다.

    Returns
    -------
    int
        검출된 꼬리(깃발) 개수.
    """
    (x, y, w, h) = stem
    if direction:                              # 정방향: 기둥 위쪽에서 꼬리 탐색
        area_top = y
        area_bot = y + h - fs.weighted(15)
    else:                                      # 역방향: 기둥 아래쪽에서 꼬리 탐색
        area_top = y + fs.weighted(15)
        area_bot = y + h
    if index:                                  # 빔 연결의 2번째 이상 → 왼쪽 탐색
        area_col = x - fs.weighted(4)
    else:                                      # 첫 번째 → 오른쪽 탐색
        area_col = x + w + fs.weighted(4)

    cnt = fs.count_pixels_part(image, area_top, area_bot, area_col)
    return cnt


def count_beams(image, stems, direction):
    """빔으로 연결된 음표 그룹에서 각 기둥의 빔(수평선) 개수를 센다.

    원리 (기울기 추적 + 기둥 인접 양측 스캔):
      1. 빔은 기둥 끝(정방향=상단, 역방향=하단)에 붙으므로, 첫 기둥과 끝 기둥의
         끝점을 잇는 직선의 기울기가 곧 빔의 기울기다(실측 오차 ≈0.01).
      2. 각 기둥의 '바로 옆'(좌·우 각각 2~4px 떨어진 3개 열)을 세로로 스캔하되,
         스캔 밴드를 그 x좌표에서의 빔 기준선(기울기 직선) 위치에 맞춰 이동시킨다.
         밴드 높이는 빔 3단(두께+간격)을 덮는 weighted(18).
      3. 한쪽의 빔 수 = 3개 열의 중앙값(오선 제거 잔여 잡음 내성),
         기둥의 빔 수 = 좌·우 중 큰 값(부분빔은 한쪽에만 붙으므로).

    기존 방식('기둥 사이 중간 x' 한 열을 기둥 상단 고정 8px 밴드로 스캔)의 문제:
      · 부분빔(짧은 스텁)은 기둥 옆에만 존재해 중간 지점에서는 절대 검출 불가
        → 32분·16분이 한 단계씩 낮게 인식.
      · 기울어진 빔에서 고정 밴드가 2·3번 빔을 벗어남.
      · 단일 열 스캔이라 잡음 한 점에도 취약.

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지.
    stems : list
        기둥들의 [x, y, w, h] 리스트(2개 이상).
    direction : bool
        음표 방향(True=정방향/빔 위, False=역방향/빔 아래).

    Returns
    -------
    list[int]
        각 기둥의 빔 개수(1~3). 1=8분음표, 2=16분음표, 3=32분음표.
    """
    H, W = image.shape

    # 1) 기둥 끝(빔이 붙는 쪽) 좌표 → 빔 기준선(기울기 직선)
    def tip(s):
        sx, sy, sw, sh = s
        return (sx + sw / 2, sy if direction else sy + sh)

    (x0, y0), (x1, y1) = tip(stems[0]), tip(stems[-1])
    slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0

    def baseline(cx):
        return y0 + slope * (cx - x0)

    zone = fs.weighted(18)     # 빔 3단(두께+간격)을 덮는 세로 범위
    pad = fs.weighted(2)       # 기준선 추정 오차 여유

    def blob_lengths(at, ab, cx):
        """열 cx의 at~ab 범위에서 각 픽셀 덩어리의 세로 길이 리스트를 구한다."""
        at, ab = max(0, at), min(H, ab)
        col = image[at:ab, cx]
        lengths, run = [], 0
        for v in col:
            if v:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
        if run:
            lengths.append(run)
        return lengths

    unit = fs.weighted(6)      # 빔 1개의 세로 두께(실측 5~6px) 기준 단위

    counts = []
    for (sx, sy, sw, sh) in stems:
        side_counts = []
        for side_cols in ((sx - 4, sx - 3, sx - 2),
                          (sx + sw + 1, sx + sw + 2, sx + sw + 3)):
            vals = []
            for cx in side_cols:
                if not (0 <= cx < W):
                    continue
                base = baseline(cx)
                if direction:                      # 빔이 기준선 아래로 전개
                    at, ab = int(base - pad), int(base + zone)
                else:                              # 빔이 기준선 위로 전개
                    at, ab = int(base - zone), int(base + pad)
                # 덩어리 개수가 아니라 두께 기반으로 빔 수 추정:
                # 기울어진 빔은 기둥 부근에서 서로 붙어 한 덩어리로 병합되므로
                # (예: 빔 2개 = 11~12px 덩어리 하나), 덩어리 길이를 빔 1개
                # 두께로 나눠 반올림한 값을 합산한다.
                vals.append(sum(max(1, round(L / unit)) for L in blob_lengths(at, ab, cx)))
            if vals:
                vals.sort()
                side_counts.append(vals[len(vals) // 2])   # 중앙값
        c = max(side_counts) if side_counts else 1
        counts.append(min(3, max(1, c)))                    # 1~3 으로 제한
    return counts


def beam_links(image, stems, direction):
    """인접한 기둥 쌍이 실제 빔(수평선)으로 연결되어 있는지 판정한다.

    한 연결요소 안에 기둥이 여러 개라도, 이음줄(slur/tie) 곡선 때문에 빔이 없는
    음표까지 한 덩어리로 붙는 경우가 있다. 이때 빔이 실제로 이어진 구간만 하나의
    빔그룹으로 봐야 한다. 각 인접 기둥 쌍의 사이 중간 x에서 빔 기준선(기울기
    직선) 부근을 세로로 스캔해 빔 두께만큼 픽셀이 있으면 연결로 본다. 이음줄은
    머리 아래 곡선이라 이 상단(정방향) 빔 밴드에는 나타나지 않는다.

    Returns
    -------
    list[bool]
        길이 len(stems)-1. i번째 = 기둥 i와 i+1이 빔으로 연결되었는가.
    """
    H, W = image.shape
    if len(stems) < 2:
        return []

    def tip(s):
        sx, sy, sw, sh = s
        return (sx + sw / 2, sy if direction else sy + sh)

    (x0, y0), (x1, y1) = tip(stems[0]), tip(stems[-1])
    slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
    band_h = fs.weighted(7)
    pad = fs.weighted(3)

    links = []
    for i in range(len(stems) - 1):
        sx1 = stems[i][0] + stems[i][2] / 2
        sx2 = stems[i + 1][0] + stems[i + 1][2] / 2
        midx = int((sx1 + sx2) / 2)
        base = y0 + slope * (midx - x0)
        if direction:
            at, ab = int(base - pad), int(base + band_h)
        else:
            at, ab = int(base - band_h), int(base + pad)
        at, ab = max(0, at), min(H, ab)
        col = image[at:ab, max(0, min(W - 1, midx))]
        filled = int(np.count_nonzero(col))
        links.append(filled >= fs.weighted(3))
    return links


def recognize_note_dot(image, stem, direction, tail_cnt, stems_cnt):
    """음표 오른쪽의 점(부점) 유무를 판정한다.

    점 탐색 구역(기둥 오른쪽)의 픽셀 수를 세어 임계값과 비교한다.
    주의: 정방향 단독 음표는 꼬리가 아래로 내려와 점 탐색 구역과 겹치므로,
    꼬리 개수가 많을수록 임계값을 높여 오판을 막는다(빔 음표는 꼬리가 내려오지
    않으므로 기본 임계값 사용).

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지.
    stem : list[int]
        기둥의 [x, y, w, h].
    direction : bool
        음표 방향(True=정방향, False=역방향).
    tail_cnt : int
        이 음표의 꼬리 개수.
    stems_cnt : int
        객체 안의 기둥 개수(1이면 단독 음표, >1이면 빔 연결).

    Returns
    -------
    bool
        점(부점)이 있으면 True.
    """
    (x, y, w, h) = stem
    if direction:                              # 정방향 음표: 점은 머리 오른쪽 아래
        area_top = y + h - fs.weighted(10)
        area_bot = y + h + fs.weighted(5)
        area_left = x + w + fs.weighted(2)
        area_right = x + w + fs.weighted(12)
    else:                                      # 역방향 음표: 점은 머리 오른쪽 위
        area_top = y - fs.weighted(10)
        area_bot = y + fs.weighted(5)
        area_left = x + w + fs.weighted(14)
        area_right = x + w + fs.weighted(24)

    dot_rect = (area_left, area_top, area_right - area_left, area_bot - area_top)
    return _has_dot_blob(image, dot_rect)


_cc_cache = {"key": None, "labels": None, "stats": None}


def _components(image):
    """이미지 전체의 연결요소를 캐시해 재사용한다(음표마다 재계산하지 않도록)."""
    key = (id(image), image.shape, int(np.count_nonzero(image)))
    if _cc_cache["key"] != key:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(image)
        _cc_cache.update(key=key, labels=labels, stats=stats)
    return _cc_cache["labels"], _cc_cache["stats"]


def _has_dot_blob(image, rect):
    """구역 안에 '점(부점)처럼 생긴 독립 덩어리'가 있는지 판정한다.

    기존에는 구역의 픽셀 수만 셌다. 그러면 음표 꼬리(깃발)가 구역으로 흘러들 때
    오판이 생겨, 꼬리가 많을수록 임계값을 크게 올려야 했다(32분음표는 30px).
    그런데 진짜 부점은 10~16px밖에 안 되므로, 그 임계값에서는 점이 있어도
    놓쳤다(music.jpg 마디4의 점32분음표 누락 원인).

    부점의 본질은 '작고 독립된 덩어리'다. 따라서 구역에 걸친 연결요소를
    이미지 '전체' 기준으로 조회해, 그 요소의 크기가 점 크기(가로·세로 모두
    오선 4/5칸 이하)이고 면적이 충분하면 부점으로 본다. 음표 꼬리는 기둥·머리와
    한 덩어리(=큰 요소)라서 여기서 확실히 걸러진다. 구역만 잘라내 연결요소를
    구하면 잘린 꼬리 끝이 작은 덩어리로 보여 오판하므로, 반드시 전체 기준으로
    조회해야 한다.
    """
    x, y, w, h = (int(v) for v in rect)
    H, W = image.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return False
    labels, stats = _components(image)
    sub = labels[y0:y1, x0:x1]
    lim = fs.weighted(8)                     # 점의 최대 크기
    for li in np.unique(sub):
        if li == 0:
            continue
        bw, bh, area = stats[li][2], stats[li][3], stats[li][4]
        if bw <= lim and bh <= lim and area >= fs.weighted(5):
            return True
    return False


def recognize_pitch(image, staff, head_center):
    """음표 머리 중심 y좌표로 음정(음높이) 인덱스를 판정한다.

    마지막(가장 아래) 오선 staff[4] 를 기준으로 아래(+weighted(30))에서 위로
    반칸(weighted(5))씩 올라가며 21개의 가상 좌표선을 만든다. 이 중 중간 5개는
    실제 오선이고 나머지는 덧줄/칸에 해당한다. 머리 중심이 어느 선의 ±weighted(2)
    안에 들면 그 인덱스를 음정으로 반환한다(0=가장 낮음 … 20=가장 높음).

    Parameters
    ----------
    image : np.ndarray
        (블로그 시그니처 유지를 위한 인자. 현재 로직에서는 사용하지 않음)
    staff : list[float]
        해당 보표의 오선 5개 y좌표. staff[0]=맨 위, staff[4]=맨 아래.
    head_center : float
        음표 머리의 중심 y좌표.

    Returns
    -------
    int | None
        음정 인덱스(0~20). 어느 선에도 안 걸리면 None(덧줄 범위 밖).
    """
    pitch_lines = [staff[4] + fs.weighted(30) - fs.weighted(5) * i for i in range(21)]
    for i in range(len(pitch_lines)):
        line = pitch_lines[i]
        if line + fs.weighted(2) >= head_center >= line - fs.weighted(2):
            return i
    return None


def recognize_rest(image, vis, staff, stats):
    """쉼표 종류를 인식한다(음표가 아닌 객체 대상).

    쉼표는 뚜렷한 직선 특징이 없지만 위치가 항상 2~4번째 오선 사이로 고정되고,
    높이·너비·픽셀로 종류를 구분할 수 있다.
      - 높이 25~35: 왼쪽 픽셀 덩어리 cnt==3 & 너비 7~11 → 4분쉼표(4)
                    cnt==1 & 너비 11~14 → 16분쉼표(16)
      - 높이 16~22 & 너비 9~15 → 8분쉼표(8)
      - 높이 ≤8: 중심이 staff[1]~+5 → 온쉼표(1), +5~staff[2] → 2분쉼표(2)
    점쉼표면 값을 음수로 만든다.

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지(분석용).
    vis : np.ndarray
        BGR 시각화 이미지.
    staff : list[float]
        해당 보표의 오선 5개 y좌표.
    stats : tuple
        객체의 (x, y, w, h, area).

    Returns
    -------
    int
        쉼표 종류(1/2/4/8/16, 음수=점쉼표). 쉼표가 아니면 0.
        ※ 음표 박자와 코드가 겹치지만, 쉼표는 pitch=-1 로 저장해 구분한다.
    """
    (x, y, w, h, area) = stats
    rest = 0
    center = fs.get_center(y, h)
    rest_condition = staff[3] > center > staff[1]   # 2~4번 오선 사이
    if rest_condition:
        cnt = fs.count_pixels_part(image, y, y + h, x + fs.weighted(1))
        if fs.weighted(35) >= h >= fs.weighted(25):
            # [조정] 블로그 원본은 4분=cnt==3, 16분=너비 11~14 였으나, 이 샘플(창 캡처+정규화)
            # 에서 4분쉼표가 cnt=2로, 16분쉼표 너비가 15로도 나와 일부가 누락됐다.
            # → 4분쉼표 cnt 2~3 허용, 16분쉼표 너비 상한 15로 넓혀 보정.
            if 3 >= cnt >= 2 and fs.weighted(11) >= w >= fs.weighted(7):
                rest = 4      # 4분쉼표
            elif cnt == 1 and fs.weighted(15) >= w >= fs.weighted(11):
                rest = 16     # 16분쉼표
        elif fs.weighted(22) >= h >= fs.weighted(16):
            if fs.weighted(15) >= w >= fs.weighted(9):
                rest = 8
        elif fs.weighted(8) >= h:
            if staff[1] + fs.weighted(5) >= center >= staff[1]:
                rest = 1
            elif staff[2] >= center >= staff[1] + fs.weighted(5):
                rest = 2

        if rest and recognize_rest_dot(image, stats):
            rest *= -1   # 점쉼표

        if rest:
            label = "R" + str(abs(rest)) + ("." if rest < 0 else "")
            fs.put_text(vis, label, (x, y + h + fs.weighted(20)))

    return rest


def recognize_rest_dot(image, stats):
    """쉼표 오른쪽 구역의 픽셀 수로 점(부점) 유무를 판정한다.

    부점은 쉼표의 '세로 중앙' 높이, 오른쪽에 찍힌다. 기존에는 구역을 쉼표
    상단(y)을 기준으로 잡아, 세로로 긴 4분쉼표에서는 점이 구역 아래로 벗어나
    검출되지 않았다(마디16의 점4분쉼표 누락 원인). 쉼표 중심(y + h/2)을
    기준으로 잡고, 구역을 조금 넓혀 확실히 포함시킨다.
    """
    (x, y, w, h, area) = stats
    cy = y + h // 2                            # 쉼표 세로 중앙
    area_top = cy - fs.weighted(8)
    area_bot = cy + fs.weighted(8)
    area_left = x + w
    area_right = x + w + fs.weighted(12)
    dot_rect = (area_left, area_top, area_right - area_left, area_bot - area_top)
    pixels = fs.count_rect_pixels(image, dot_rect)
    return pixels >= fs.weighted(6)


def recognize_whole_note(image, vis, staff, stats):
    """온음표를 인식한다(조표·음표·쉼표가 모두 걸러진 뒤 남은 객체 대상).

    온음표는 기둥이 없는 가로로 긴 타원이므로, 너비 12~22 & 높이 9~12 조건으로 찾는다.
    오른쪽 구역 픽셀로 점 유무를 판정해 온음표(1)/점온음표(-1)를 구분하고, 음정도 함께 구한다.

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지(분석용).
    vis : np.ndarray
        BGR 시각화 이미지.
    staff : list[float]
        해당 보표의 오선 5개 y좌표.
    stats : tuple
        객체의 (x, y, w, h, area).

    Returns
    -------
    tuple[int, int|None]
        (whole_note, pitch). whole_note: 1=온음표, -1=점온음표, 0=온음표 아님.
        pitch: 음정 인덱스(온음표가 아니면 0).
    """
    whole_note = 0
    pitch = 0
    (x, y, w, h, area) = stats
    # 파이썬 체이닝: 22>=w>=12 그리고 12>=h>=9 (가로로 긴 타원)
    whole_note_condition = (fs.weighted(22) >= w >= fs.weighted(12) >= h >= fs.weighted(9))
    if whole_note_condition:
        dot_rect = (
            x + w,
            y - fs.weighted(10),
            fs.weighted(10),
            fs.weighted(20),
        )
        pixels = fs.count_rect_pixels(image, dot_rect)
        whole_note = -1 if pixels >= fs.weighted(10) else 1   # 점 있으면 점온음표
        pitch = recognize_pitch(image, staff, fs.get_center(y, h))

        # 시각화: 온음표(W)/점온음표(W.) + 음이름
        label = "W" + ("." if whole_note < 0 else "")
        fs.put_text(vis, label, (x, y + h + fs.weighted(20)))
        if pitch is not None:
            fs.put_text(vis, PITCH_NAMES[pitch], (x, y + h + fs.weighted(38)))

    return whole_note, pitch



def _count_holes(sub):
    """숫자 영역의 닫힌 구멍 수를 센다(4·6·8·9 구분용). 획의 오목부·잡음은 제외.

    진짜 숫자 구멍은 (1) 면적이 숫자 넓이의 상당 비율이고 (2) 폭·높이가 어느
    정도 있으며 (3) 지나치게 납작하지 않다(가로/세로 비가 극단적이지 않음).
    스캔 잡음이나 획이 살짝 겹쳐 생기는 작은 틈을 구멍으로 오인하지 않도록
    임계를 실측 기반으로 높였다.
    """
    if sub is None or sub.size == 0:
        return 0, []
    inv = (sub == 0).astype(np.uint8)
    # 배경(0) 테두리를 1px 둘러, 이미지 가장자리에 닿는 모든 배경이 서로 연결되게
    # 한다. (0,0) 한 점에서만 flood fill 하면, 전체 폭을 가로지르는 획 —
    # 예: '4'의 가로 막대 — 아래쪽 배경이 위쪽과 분리돼 구멍으로 오인된다.
    inv = np.pad(inv, 1, constant_values=1)
    ff = np.zeros((inv.shape[0] + 2, inv.shape[1] + 2), np.uint8)
    cv2.floodFill(inv, ff, (0, 0), 0)                    # 외부 배경 제거
    inv = inv[1:-1, 1:-1]                                # 패딩 제거(좌표 복원)
    n, _, st, cent = cv2.connectedComponentsWithStats(inv)
    h, w = sub.shape
    holes = []
    for i in range(1, n):
        a = int(st[i][4])
        hw, hh = int(st[i][2]), int(st[i][3])
        if a < max(6, w * h * 0.06):        # 면적: 숫자 넓이의 6% 이상
            continue
        if hw < w * 0.28 or hh < h * 0.14:  # 폭·높이 최소치
            continue
        if hw > w * 0.95 or hh > h * 0.75:  # 과대 구멍(잘못 닫힌 전체 배경 등) 배제
            continue
        holes.append((a, cent[i][1], hh))           # (면적, y중심, 세로높이)
    return len(holes), holes


def _classify_time_digit(sub):
    """박자표 한 숫자를 형태 특징으로 판정. 실제 박자표에 쓰이는 숫자(0~9) 대응.

    판정 우선순위(가장 안정적인 특징부터):
      1. 구멍 2개 이상 → 8. ('8'은 폰트에 따라 허리에 강한 가로획이 생겨
         '4' 규칙에 걸릴 수 있으므로, 구멍 개수를 가장 먼저 본다. 진짜 '4'는
         구멍이 최대 1개라 이 규칙에 걸리지 않는다.)
      2. '4': 중~하단(0.45~0.93h)에 강한 가로 막대가 있고, 그 막대 '아래'로
         세로획(기둥)이 이어진다. '2'도 바닥에 강한 가로 막대를 갖지만 막대
         아래에는 아무것도 없으므로 이 조건으로 구분된다. 폰트에 따라 4의
         막대 위치가 0.8h 아래까지 내려오는 경우(좁은 세리프체)도 포괄한다.
      3. 구멍 1개 → 구멍이 세로의 절반 이상을 차지하면 0,
         위쪽이면 9, 아래쪽이면 6.
      4. 구멍 0개 → 최상단에 강한 가로 막대(0~0.2h)가 있고 하단이 가벼우면 7,
         바닥이 두꺼우면 2, 그밖에 3.
    """
    if sub is None or sub.size == 0 or np.count_nonzero(sub) < 5:
        return None
    h, w = sub.shape
    rowsum = [int(np.count_nonzero(sub[r])) for r in range(h)]
    hbar_row = int(np.argmax(rowsum))
    hbar_strong = max(rowsum) >= w * 0.82
    n_hole, holes = _count_holes(sub)

    # 1) 구멍 2개 이상 → 8
    if n_hole >= 2:
        return 8

    # 2) '4' 판정: 중~하단의 강한 가로 막대 + 막대 아래로 이어지는 세로획
    if hbar_strong and 0.45 * h <= hbar_row <= 0.93 * h:
        # 막대는 여러 행 두께일 수 있으므로, 연속으로 굵은 행을 모두 지나
        # '막대가 끝난 다음'부터 아래를 검사한다. ('2'의 바닥 막대는 그 아래에
        # 아무것도 없고, '4'는 기둥이 바닥까지 이어진다.)
        bar_end = hbar_row
        while bar_end + 1 < h and rowsum[bar_end + 1] >= w * 0.55:
            bar_end += 1
        rows_below = h - (bar_end + 1)
        below = int(np.count_nonzero(sub[bar_end + 1:, :]))
        if rows_below >= 2 and below >= max(2, rows_below):   # 기둥 존재 → 4
            return 4
        if rows_below >= 2 and hbar_row <= 0.75 * h:          # 막대가 확실한 중단 → 4
            return 4

    # 3) 구멍 1개 → 0/9/6
    if n_hole == 1:
        area_max = max(holes, key=lambda t: t[0])
        hole_cy = area_max[1]
        # 구멍이 세로 대부분을 차지하면 0 (구멍 면적이 숫자 면적의 40% 이상)
        if area_max[0] >= 0.4 * np.count_nonzero(sub):
            return 0
        return 9 if hole_cy < h * 0.45 else 6

    # 4) 구멍 0개 → 7/2/3
    top_rows = max(1, int(h * 0.2))
    top_bar = max(rowsum[:top_rows]) >= w * 0.82
    bottom_mass = sum(rowsum[int(h * 0.7):])
    top_mass = sum(rowsum[:max(1, int(h * 0.33))])
    if top_bar and bottom_mass < top_mass * 0.5:
        return 7
    bottom = sum(rowsum[int(h * 0.66):])
    return 2 if bottom >= top_mass else 3


_TIMESIG_DIGIT_MIN = 0.42     # 템플릿 매칭으로 숫자를 확정할 최소 점수
_VALID_DENOMS = (1, 2, 4, 8, 16, 32)

# 각 숫자의 기대 구멍 수(폰트에 따라 두 값이 모두 가능한 경우 튜플로 나열).
_DIGIT_HOLES = {0: (1,), 1: (0,), 2: (0,), 3: (0,), 4: (0, 1),
                5: (0,), 6: (1,), 7: (0,), 8: (2,), 9: (1,)}
_HOLE_PENALTY = 0.55          # 기대 구멍 수와 1개 차이날 때마다 점수에 곱하는 감쇠


def _text_font_paths():
    """숫자 템플릿용 일반 텍스트 폰트 후보 경로(OS별). 존재하는 것만 사용."""
    import os
    win = os.environ.get("WINDIR", r"C:\Windows")
    return [
        # Windows
        os.path.join(win, "Fonts", "arial.ttf"),
        os.path.join(win, "Fonts", "times.ttf"),
        os.path.join(win, "Fonts", "arialbd.ttf"),
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    ]


_digit_library_cache = None


def _digit_template_library():
    """숫자 0~9의 다중 폰트 템플릿 라이브러리를 만든다(1회 생성 후 캐시).

    구성: Bravura(SMuFL timeSig 글리프) + Pillow 기본 폰트 + 사용 가능한
    시스템 텍스트 폰트. 악보마다 박자표 숫자의 서체가 제각각이므로
    (조판 프로그램의 두꺼운 악보체 vs 좁은 세리프체), 여러 서체 중
    '최고 일치'를 취하면 특정 서체에 대한 편향이 사라진다.

    반환: {숫자: [이진 템플릿(tight crop), ...]}. 폰트가 하나도 없으면 빈 dict.
    """
    global _digit_library_cache
    if _digit_library_cache is not None:
        return _digit_library_cache

    from PIL import Image, ImageDraw, ImageFont
    px = 80
    fonts = []
    smufl = _load_smufl()
    if smufl is not None:
        f = smufl.get_font(px)
        if f is not None:
            fonts.append(("smufl", f))
    try:
        fonts.append(("default", ImageFont.load_default(px)))
    except Exception:
        pass
    for path in _text_font_paths():
        try:
            fonts.append((path, ImageFont.truetype(path, px)))
        except Exception:
            continue

    lib = {d: [] for d in range(10)}
    for name, font in fonts:
        for d in range(10):
            ch = (smufl.GLYPHS["timeSig"][str(d)]
                  if name == "smufl" else str(d))
            try:
                canvas = Image.new("L", (px * 4, px * 4), 0)
                ImageDraw.Draw(canvas).text(
                    (px * 2, px * 2), ch, font=font, fill=255, anchor="mm")
                arr = (np.array(canvas) > 96).astype(np.uint8) * 255
                t = _tight_crop(arr)
                if t is not None:
                    lib[d].append(t)
            except Exception:
                continue
    if not any(lib.values()):
        lib = {}
    _digit_library_cache = lib
    return lib


def _digit_holes(sub):
    """앙상블 분류용 구멍 카운터(검증판: 상한 없음, 면적 하한만).

    _count_holes 와 달리 '0'의 세로로 큰 구멍(높이 75% 초과)도 정상으로 센다.
    tight crop 된 숫자 하나를 전제로 하므로, 분할 잔해를 걸러내기 위한
    상한 규칙이 필요 없다. 교차 폰트 검증(50케이스)에서 오류 2건.
    """
    if sub is None or sub.size == 0:
        return 0
    inv = (sub == 0).astype(np.uint8)
    inv = np.pad(inv, 1, constant_values=1)   # 외부 배경을 전부 연결
    ff = np.zeros((inv.shape[0] + 2, inv.shape[1] + 2), np.uint8)
    cv2.floodFill(inv, ff, (0, 0), 0)
    n, _, st, _ = cv2.connectedComponentsWithStats(inv)
    h, w = sub.shape
    cnt = 0
    for i in range(1, n):
        if int(st[i][4]) >= max(4, w * h * 0.04):
            cnt += 1
    return cnt


def _classify_digit_ensemble(sub):
    """숫자 하나(tight crop)를 '다중 폰트 템플릿 × 구멍 수 페널티'로 분류한다.

    - 템플릿 점수: 여러 서체의 같은 숫자 중 최고 겹침(F1 × 종횡비^0.5).
    - 구멍 페널티: 실측 구멍 수가 그 숫자의 기대 구멍 수와 다르면 차이 1당
      _HOLE_PENALTY 를 곱한다(8은 구멍 2, 6·9·0은 1, 나머지는 0 — 저해상도에서
      구멍이 뭉개질 수 있어 하드 필터가 아닌 소프트 페널티로 쓴다).

    반환: (숫자 int 또는 None, 점수). 템플릿 라이브러리가 없으면 (None, 0.0).
    """
    if sub is None or sub.size == 0:
        return None, 0.0
    lib = _digit_template_library()
    if not lib:
        return None, 0.0
    n_hole = _digit_holes(sub)
    # 구조 휴리스틱(가로 막대·구멍 위치 등)의 예측과 일치하는 숫자에는 소폭
    # 가산점을 준다. 템플릿 점수가 근소차일 때(예: 좁은 세리프 '2' vs '1')
    # 서로 다른 원리의 두 분류기가 합의하는 쪽을 선택하게 하는 장치다.
    hint = _classify_time_digit(sub)
    best_d, best_s = None, 0.0
    for d in range(10):
        weight = _HOLE_PENALTY ** min(abs(n_hole - e) for e in _DIGIT_HOLES[d])
        bonus = 0.08 if d == hint else 0.0
        for tmpl in lib.get(d, []):
            s = _shape_score(sub, tmpl) * weight + bonus
            if s > best_s:
                best_d, best_s = d, s
    if best_s < _TIMESIG_DIGIT_MIN:
        return None, best_s
    return best_d, best_s


def _classify_digit_smufl(sub):
    """숫자 하나(tight crop 전제 아님)를 SMuFL timeSig 숫자 템플릿과 대조한다.

    반환: (숫자 int 또는 None, 점수). 폰트가 없으면 (None, 0.0).
    """
    smufl = _load_smufl()
    if smufl is None or sub is None or sub.size == 0:
        return None, 0.0
    candidates = [(str(d), smufl.GLYPHS["timeSig"][str(d)]) for d in range(10)]
    name, score = match_glyph(sub, candidates)
    if name is None or score < _TIMESIG_DIGIT_MIN:
        return None, score
    return int(name), score


def _classify_field_scored(sub):
    """분자/분모 한 칸을 숫자(들)로 해석해 (값, 신뢰 점수)를 반환한다.

    - tight crop 상태의 이미지를 받는다(None 이면 (None, 0)).
    - 종횡비로 한 자리/두 자리(12/8 등)를 판단(두 자리면 세로 투영 골짜기 분할).
    - 각 자리는 SMuFL 템플릿 매칭으로 분류하고, 폰트가 없으면 형태 휴리스틱
      (_classify_time_digit)으로 폴백(점수는 낮은 고정값을 부여).
    """
    if sub is None or sub.size == 0:
        return None, 0.0
    h, w = sub.shape

    def one(digit_img):
        if digit_img is None or digit_img.size == 0:
            return None, 0.0
        # 1순위: 다중 폰트 템플릿 × 구멍 페널티 앙상블(서체 무관, 교차검증 47/50)
        d, score = _classify_digit_ensemble(digit_img)
        if d is not None:
            return d, score
        # 2순위(폰트가 전혀 없을 때): 구조 특징 휴리스틱
        d = _classify_time_digit(digit_img)
        return d, (0.3 if d is not None else 0.0)

    # 두 자리 판정: 박자표 숫자 한 자는 폭/높이 ≈ 0.7~1.0. 1.25를 넘으면 두 자리.
    if h > 0 and w / h > 1.25:
        colsum = np.count_nonzero(sub, axis=0)
        lo, hi = int(w * 0.3), int(w * 0.7)
        cut = lo + int(np.argmin(colsum[lo:hi])) if hi > lo else w // 2
        a, sa = one(_tight_crop(sub[:, :cut]))
        b, sb = one(_tight_crop(sub[:, cut:]))
        if a is not None and b is not None:
            return a * 10 + b, (sa + sb) / 2
        return (a, sa) if a is not None else (b, sb)
    return one(sub)


def recognize_time_signature(image, staff, stats):
    """[13단계] 박자표 객체에서 (분자, 분모)를 인식한다. 일반 박자표를 지원.

    처리 순서
    ---------
    1) common/cut time(𝄴·𝄵) 검사: 객체가 한 덩어리 글리프(분할 골 없음)이고
       C/¢ 템플릿과 잘 맞으면 (4,4)/(2,2)로 확정.
    2) 분자/분모 분할: 3번선(staff[2])을 1차 기준으로 하되, 그 근처(±오선 반칸)
       세로 투영의 '골짜기'(숫자 사이 빈틈)로 미세 보정한다. 분할선이 숫자의
       획(예: 4의 하단 가로획)을 자르는 문제를 막는다.
    3) 각 칸을 SMuFL 템플릿 매칭으로 분류(두 자리 분자 12/8 등 지원).
       폰트가 없으면 형태 휴리스틱으로 폴백.
    4) 검증: 분모는 2의 거듭제곱(1~32)만 유효. 아니면 유효 후보 중 재선택.
    """
    x, y, w, h, area = stats
    region = image[y:y + h, x:x + w]
    if region.size == 0:
        return None, None

    # 1) common/cut time — 한 덩어리 글리프. 먼저 객체 전체로 시도하고,
    #    (클레프 등과 융합돼 넓어진 객체라 실패하면) 슬라이딩 윈도우로 재시도.
    smufl = _load_smufl()
    if smufl is not None:
        name, score = match_glyph(region, [
            ("common", "\uE08A"), ("cut", "\uE08B")])
        if name is not None and score >= 0.62:
            return (4, 4) if name == "common" else (2, 2)
        if w > fs.weighted(30):
            name, score, span = find_common_time_window(image, stats, staff)
            if name is not None:
                return (4, 4) if name == "common" else (2, 2)

    # 2) 분자/분모 분할 — 3번선(geo) 주변에서 '숫자 사이 빈틈'(픽셀 최소 행)
    #    후보들을 찾고, 위/아래 숫자 높이가 가장 균형잡히는 분할을 선택한다.
    #    박자표의 두 숫자는 항상 같은 크기이므로, 균형은 폰트와 무관하게
    #    올바른 분할을 고른다. (고정 분할선이 '4'의 하단 가로획을 잘라
    #    오분류되던 문제의 해결책)
    geo = int(round(staff[2] - y))
    if not (0.3 * h <= geo <= 0.72 * h):
        geo = h // 2
    rowsum = np.count_nonzero(region, axis=1)
    band = fs.weighted(5)
    lo_b = max(2, geo - band)
    hi_b = min(h - 2, geo + band + 1)
    window = rowsum[lo_b:hi_b]
    if window.size:
        vmin = int(window.min())
        # 픽셀 수가 최소(+1)인 행들 = 숫자 사이 빈틈 후보. 항상 geo 자체도 포함.
        candidates = sorted({lo_b + int(i) for i in np.nonzero(window <= vmin + 1)[0]}
                            | {geo})
    else:
        candidates = [geo]

    def balance(split):
        n_sub = _tight_crop(region[:split])
        d_sub = _tight_crop(region[split:])
        if n_sub is None or d_sub is None:
            return (10 ** 9, 10 ** 9)
        # 1차: 위/아래 숫자 높이의 균형(두 숫자는 항상 같은 크기로 조판됨)
        # 2차(동률 시): 분할 행의 픽셀 수가 적은 쪽 = 숫자 획을 덜 자르는 분할
        return (abs(n_sub.shape[0] - d_sub.shape[0]), int(rowsum[split]))

    best_split = min(candidates, key=balance)
    numerator, _ = _classify_field_scored(_tight_crop(region[:best_split]))
    denominator, _ = _classify_field_scored(_tight_crop(region[best_split:]))

    # 4) 분모 검증: 2의 거듭제곱이 아니면 유효 후보 중 최고 점수로 재선택
    if denominator not in _VALID_DENOMS:
        den_sub = _tight_crop(region[best_split:])
        if den_sub is not None and smufl is not None:
            best, best_s = None, 0.0
            for d in _VALID_DENOMS:
                if d >= 10:
                    continue
                s = _shape_score(den_sub, _render_glyph_template(
                    smufl.GLYPHS["timeSig"][str(d)]))
                if s > best_s:
                    best, best_s = d, s
            if best is not None and best_s >= _TIMESIG_DIGIT_MIN:
                denominator = best
    return numerator, denominator
