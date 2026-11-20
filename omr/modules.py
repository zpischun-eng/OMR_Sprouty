# modules.py
# ---------------------------------------------------------------------------
# 악보 인식의 '단계(step)' 단위 처리를 담는 모듈.
# 각 단계는 하나의 함수로 구현하며, 입력/출력 모두 numpy 이미지를 사용해
# 파이프라인에서 연쇄적으로 호출할 수 있도록 한다.
#
# [1단계] 보표 영역 추출 및 그 외 노이즈 제거
#   - 블로그 https://hackids.tistory.com/114 의 remove_noise 를 구현.
# [2단계] 오선 제거
#   - 블로그 https://hackids.tistory.com/115 의 remove_staves 를 구현.
# [3단계] 악보 이미지 정규화
#   - 블로그 https://hackids.tistory.com/116 의 normalization 을 구현.
# [보강] 잡티(작은 객체) 제거 — remove_small_objects
#   - 오선 제거 과정에서 음표 근처에 남는 1~5px 파편을 정리한다.
# [4단계] 객체 검출
#   - 블로그 https://hackids.tistory.com/117 의 object_detection 을 구현.
# [5단계] 객체 분석 (음표 기둥·방향 특징 추출)
#   - 블로그 https://hackids.tistory.com/118 의 object_analysis 를 구현.
# [6단계] 인식 과정 - 조표
#   - 블로그 https://hackids.tistory.com/119 의 recognition / recognize_key 를 구현.
# ---------------------------------------------------------------------------
import cv2
import numpy as np

from . import functions as fs
from . import recognition as rs

# 한 행이 '오선'으로 인정되기 위한 흰색 픽셀 비율 임계값(이미지 너비 대비).
STAFF_LINE_RATIO = 0.5
# 오선의 안티에일리어싱된 가장자리를 인정하는 행 통계 하한(실측: 순수 음표
# 내용 행은 폭의 1~12%, 오선 경계 행은 20%+ — 그 사이에서 넉넉히 잡는다).
STAFF_EDGE_RATIO = 0.22
# 확장된 경계 밖으로 '한 걸음 더' 두는 안전 여유(행). 안티에일리어싱은 이
# 여유 안에서 끝나지만, 박자표 숫자·기둥처럼 키 큰 내용은 이 여유 밖까지
# 이어지므로 걸러진다(열 단위 오선 제거 안전장치).
STAFF_EDGE_MARGIN = 2


def remove_noise(image, crop_frame=True):
    """1단계: 보표 영역만 남기고 제목/가사 등 노이즈를 제거한다.

    처리 순서
    ---------
    1) 이진화(threshold)
    2) 보표 영역만 추출하기 위한 검은 마스크 생성
    3) connectedComponentsWithStats 로 레이블링(객체 검출)
    4) 이미지 너비의 50% 이상을 차지하는 객체(=오선이 그려진 보표)만
       마스크에 '꽉 찬 사각형'으로 그려 넣는다.
    5) 원본 이진 이미지와 마스크를 AND 연산하여 보표 영역만 남긴다.

    Parameters
    ----------
    image : np.ndarray
        BGR 원본 이미지.

    Returns
    -------
    np.ndarray
        보표 영역만 남은 1채널 이진 이미지.
    """
    image = fs.threshold(image)                 # 이미지 이진화
    mask = np.zeros(image.shape, np.uint8)      # 보표 영역만 추출하기 위한 마스크

    # 레이블링(연결 요소 분석)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    for i in range(1, cnt):                     # 0번은 배경이므로 제외
        x, y, w, h, area = stats[i]
        if w > image.shape[1] * 0.5:            # 너비가 전체의 50% 이상인 객체 = 보표
            cv2.rectangle(mask, (x, y, w, h), (255, 0, 0), -1)  # 안을 채운 사각형

    masked_image = cv2.bitwise_and(image, mask)  # 보표 영역만 추출

    # [보강] 샘플이 '창 캡처' 이미지라 창 테두리/타이틀바 경계선이 남는다.
    # 이 완전한 가로/세로 프레임선은 다음 단계에서 오선으로 오인되므로 제거한다.
    # (테두리가 없는 스캔/사진 악보에는 영향 없음)
    if crop_frame:
        masked_image = fs.remove_frame(masked_image)

    return masked_image


def get_horizontal_histogram(image):
    """(시각화용) 각 행의 흰색 픽셀 개수를 좌→우로 채운 수평 히스토그램 이미지.

    오선이 있는 행은 흰색 픽셀이 아주 길게 존재하므로, 이 히스토그램에서
    가로로 길게 뻗은 막대가 곧 오선의 위치를 나타낸다. (오선 제거의 원리)

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지(1단계 결과).

    Returns
    -------
    np.ndarray
        각 행의 흰 픽셀 개수만큼 좌측부터 채운 1채널 이미지.
    """
    height, width = image.shape
    histogram = np.zeros(image.shape, np.uint8)
    # 각 행의 흰색(255) 픽셀 개수 — 블로그의 이중 for문을 numpy로 벡터화(결과 동일)
    row_counts = np.count_nonzero(image == 255, axis=1)
    for row in range(height):
        cnt = int(row_counts[row])
        if cnt > 0:
            histogram[row, :cnt] = 255  # 좌측부터 개수만큼 흰색으로 채움
    return histogram


def remove_staves(image):
    """2단계: 수평 히스토그램으로 오선을 찾아 제거한다.

    처리 순서
    ---------
    1) 각 행의 흰색 픽셀 개수를 세어, 너비의 50% 이상이면 그 행을 '오선'으로 판단.
    2) 인접한 행들은 같은 오선(2픽셀 이상 두께)으로 묶어 [y좌표, 높이]로 저장.
    3) 각 오선에 대해 열 단위로 '오선 바로 위/아래에 픽셀이 있는지' 검사하고,
       위·아래 모두 비어 있을 때만(=음표 등과 겹치지 않을 때만) 오선을 지운다.
       → 음표/쉼표의 모양을 훼손하지 않고 오선만 깔끔히 제거한다.

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지(1단계 결과).

    Returns
    -------
    tuple[np.ndarray, list[int]]
        (오선이 제거된 이미지, 검출된 각 오선의 최상단 y좌표 리스트).
        오선 좌표 리스트는 이후 단계(보표 분리 등)에서 사용된다.

    Note
    ----
    블로그 원본은 입력 이미지를 in-place 로 수정하지만, 여기서는 단계별 비교를
    위해 1단계 결과(입력)를 보존하도록 image.copy() 로 복사본을 만들어 처리한다.
    """
    image = image.copy()                 # 1단계 결과 보존을 위해 복사본에서 작업
    height, width = image.shape
    staves = []                          # [오선의 y좌표, 오선 높이] 리스트

    # --- 1) 오선 검출: 각 행의 흰 픽셀 개수로 판단 ---
    row_counts = np.count_nonzero(image == 255, axis=1)  # 벡터화(블로그 로직과 동일 결과)
    for row in range(height):
        pixels = int(row_counts[row])
        if pixels >= width * STAFF_LINE_RATIO:           # 이미지 너비의 50% 이상이면 오선
            if len(staves) == 0 or abs(staves[-1][0] + staves[-1][1] - row) > 1:
                staves.append([row, 0])                  # 새 오선 [y좌표, 높이(0부터)]
            else:
                staves[-1][1] += 1                       # 같은 오선이면 높이 증가

    # --- 2) 오선 제거: 위/아래에 픽셀이 없을 때만 지움 ---
    #
    # [보강] 스캔 해상도·압축에 따라 오선이 완전한 검정 1px이 아니라 위아래로
    # 안티에일리어싱된 회색 경계를 가질 수 있다. 그 경계 행은 50% 임계값에는
    # 못 미치지만(그래서 '오선 행'으로 검출되지 않음) 여전히 상당한 잉크가
    # 남아 있어(실측: 핵심 행 638/1136 대비 바로 옆 행도 565, 331 — 순수 음표
    # 내용 행의 70~130과는 뚜렷이 구분됨), 위/아래 '빈 행' 검사에서 거의 항상
    # 걸려 오선이 통째로 안 지워지는 문제가 있었다.
    #
    # 두 단계로 나눠 처리한다(둘 다 필요 — 하나만 쓰면 다른 회귀가 생김):
    #  (a) '이 행이 안티에일리어싱 경계인가'는 행 전체 통계(STAFF_EDGE_RATIO
    #      이상 채워짐)로 판단한다 — 음표 기둥처럼 한두 열만 이어지는 내용은
    #      전체 폭의 극히 일부라 이 통계에 안 걸린다(실측: 진짜 경계는 폭의
    #      20%+ 채움, 개별 기둥 하나가 만드는 행은 1% 안팎).
    #  (b) 다만 그 확장된 범위 안에 박자표 숫자처럼 '오선 대역과 겹치면서도
    #      그보다 훨씬 키가 큰' 진짜 내용이 있으면 지우면 안 된다. 그래서 열
    #      단위 안전 검사는 확장된 경계에서 '한 걸음 더' 바깥(STAFF_EDGE_MARGIN)
    #      까지 완전히 비어 있는지 확인한다 — 안티에일리어싱은 그 여유 안에서
    #      끝나지만, 숫자·기둥처럼 키 큰 내용은 그 여유 밖까지 이어지므로 걸러진다.
    for staff in range(len(staves)):
        top_pixel = staves[staff][0]
        bot_pixel = staves[staff][0] + staves[staff][1]
        while (top_pixel - 1 >= 0
               and row_counts[top_pixel - 1] >= width * STAFF_EDGE_RATIO):
            top_pixel -= 1
        while (bot_pixel + 1 < height
               and row_counts[bot_pixel + 1] >= width * STAFF_EDGE_RATIO):
            bot_pixel += 1
        chk_top = top_pixel - STAFF_EDGE_MARGIN
        chk_bot = bot_pixel + STAFF_EDGE_MARGIN
        if chk_top < 0 or chk_bot >= height:
            continue
        for col in range(width):
            if image[chk_top][col] == 0 and image[chk_bot][col] == 0:
                for row in range(top_pixel, bot_pixel + 1):
                    image[row][col] = 0                  # 오선(+경계) 픽셀 제거

    return image, _filter_staff_groups([x[0] for x in staves])


def _filter_staff_groups(lines):
    """검출된 가로선들 중 '등간격 5선 그룹'에 속하는 것만 오선으로 인정한다.

    도돌이 괄호(1./2. 가로선), 가사 밑줄, 이음줄 등도 이미지 너비의 50%를
    넘으면 오선으로 오검출된다(예: score_4staves 의 y=180 괄호선).
    실제 오선은 반드시 '간격이 거의 같은 5개 묶음'으로 나타나므로,
    연속 5개의 인접 간격이 서로 1.5배 이내인 묶음만 남긴다(왼쪽부터 탐욕 선택).

    Parameters
    ----------
    lines : list[int]
        검출된 가로선들의 y좌표(오름차순).

    Returns
    -------
    list[int]
        등간격 5선 그룹에 속한 선들만 남긴 y좌표 리스트(길이 = 5의 배수).
    """
    result = []
    i = 0
    while i + 4 < len(lines):
        group = lines[i:i + 5]
        gaps = [group[k + 1] - group[k] for k in range(4)]
        if min(gaps) > 0 and max(gaps) <= min(gaps) * 1.5:
            result.extend(group)
            i += 5
        else:
            i += 1                    # 첫 선이 가짜 → 하나 건너뛰고 재시도
    return result


def normalization(image, staves, standard=10.0):
    """3단계: 오선 간격이 기준값(standard)이 되도록 이미지를 정규화(리사이즈)한다.

    악보마다 해상도·크기가 달라 오선 간격이 제각각인데, 이를 일정한 값으로
    통일해 두면 이후 음표 인식 단계에서 크기에 상관없이 동일한 기준을 쓸 수 있다.

    처리 순서
    ---------
    1) 오선 좌표로 '오선 한 칸(인접 오선 사이)'의 평균 간격을 구한다.
       - 보표 1개당 오선 5개 → 칸 간격은 4개. 전체 (오선수 - 보표수)개의 간격 평균.
    2) weight = standard / 평균간격  → 이 배율로 이미지를 리사이즈.
    3) 리사이즈로 흐려진 경계를 재이진화한다.
    4) 오선 좌표에도 같은 배율을 곱해 정규화된 좌표로 갱신한다.

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지(2단계: 오선 제거 결과).
    staves : list[int]
        오선의 y좌표 리스트(2단계 결과). 길이는 5의 배수(보표수 × 5)여야 한다.
    standard : float
        정규화 후 목표로 하는 오선 한 칸의 간격(px). 기본 10.

    Returns
    -------
    tuple[np.ndarray, list[float]]
        (정규화된 이미지, 배율이 적용된 오선 y좌표 리스트).
    """
    avg_distance = 0
    lines = int(len(staves) / 5)          # 보표(5선)의 개수
    for line in range(lines):
        for staff in range(4):            # 한 보표 안의 칸 간격 4개
            staff_above = staves[line * 5 + staff]
            staff_below = staves[line * 5 + staff + 1]
            avg_distance += abs(staff_above - staff_below)   # 간격 누적
    avg_distance /= len(staves) - lines   # 오선 한 칸의 평균 간격

    height, width = image.shape
    weight = standard / avg_distance      # 기준 간격을 만들기 위한 배율
    new_width = int(width * weight)
    new_height = int(height * weight)

    image = cv2.resize(image, (new_width, new_height))
    # 리사이즈로 생긴 중간 밝기 픽셀을 다시 0/255로 정리(재이진화)
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    staves = [x * weight for x in staves]  # 오선 좌표도 배율 적용

    return image, staves


def remove_small_objects(image, min_area=5):
    """(보강) 잡티(면적이 아주 작은 연결 요소)를 제거한다.

    오선 제거 단계에서 음표·마디선 근처의 오선 일부가 '위/아래에 픽셀이 있어'
    지워지지 못하고 1~5px 크기의 파편으로 남는 경우가 있다(특히 음표가 촘촘한 줄).
    이런 파편은 실제 음악 기호가 아니므로 면적 기준으로 제거한다.

    임계값(min_area)은 정규화 이후(오선 간격 10px 기준)를 전제로 한다.
    실제 부점·스타카토 점은 면적이 약 10px 이상이라 보존되고,
    파편(≤5px)만 제거되도록 기본값을 5로 둔다.

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지(정규화 결과).
    min_area : int
        이 값 이하 면적의 연결 요소를 잡티로 보고 제거한다.

    Returns
    -------
    tuple[np.ndarray, int]
        (잡티가 제거된 이미지, 제거한 잡티 개수).
    """
    image = image.copy()
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    removed = 0
    for i in range(1, cnt):                    # 0번은 배경
        if stats[i, cv2.CC_STAT_AREA] <= min_area:
            image[labels == i] = 0             # 해당 객체 픽셀을 배경으로
            removed += 1
    return image, removed


def remove_chord_symbols(image, staves):
    """(보강) 오선 '위쪽'에 떠 있는 기타 코드(화음 기호: C, G7, Am 등)를 제거한다.

    기타 코드 글자는 보표 첫 오선보다 위에, 어떤 음악 기호와도 연결되지 않은 채
    떠 있다. 이를 그대로 두면 4단계의 닫힘(closing) 연산이 코드 글자를 바로 아래
    음표·빔과 한 덩어리로 병합해, 코드 글자가 '높은 음표 머리'로 오인식된다
    (편집 화면에서 오선 위에 이상한 음표가 그려지는 원인). 따라서 객체 검출
    '이전에' 원본 이진 이미지에서 미리 지운다.

    판정 규칙 (오탐을 막기 위한 3중 조건)
    -------------------------------------
    각 연결 요소에 대해, 세로 중심이 가장 가까운 보표를 기준으로:
      1. 요소 전체가 그 보표의 첫 오선보다 위에 있다(bottom < 첫 오선 - 1).
         → 덧줄 높은 음표는 기둥·덧줄이 오선 쪽으로 이어져 이 조건에 안 걸린다.
      2. 윗보표가 있다면, 윗보표의 낮은 음 영역(마지막 오선 + 3옥타브 덧줄 여유)
         보다 아래에서 시작한다 → 윗보표의 낮은 덧줄 음표를 지우지 않는다.
      3. 크기가 코드 기호 범위(글자·프렛 다이어그램) 안이다 → 과대 삭제 방지.

    한계: 오선 '위에만' 떠 있는 온음표(기둥·덧줄 연결이 전혀 없는 극단적 고음)는
    코드로 오인될 수 있으나, 실제 악보에서 매우 드물고 현재 인식 범위(E6) 밖이다.

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지(정규화·잡티 제거된 결과).
    staves : list[float]
        정규화된 오선 y좌표 리스트(길이 = 보표수 × 5).

    Returns
    -------
    tuple[np.ndarray, list]
        (코드 기호가 제거된 이미지, 제거된 요소들의 (x, y, w, h) 리스트).
    """
    lines = len(staves) // 5
    if lines == 0:
        return image, []
    tops = [staves[l * 5] for l in range(lines)]
    bots = [staves[l * 5 + 4] for l in range(lines)]
    centers = [(t + b) / 2 for t, b in zip(tops, bots)]

    image = image.copy()
    cnt, labels, stats, _ = cv2.connectedComponentsWithStats(image)
    removed = []
    for i in range(1, cnt):                        # 0번은 배경
        x, y, w, h, area = stats[i]
        cy = y + h / 2.0
        line = min(range(lines), key=lambda l: abs(cy - centers[l]))

        above_staff = (y + h) < tops[line] - 1                 # 조건 1
        if line > 0:                                           # 조건 2
            below_prev = y > bots[line - 1] + fs.weighted(32)
        else:
            below_prev = True
        size_ok = (fs.weighted(3) <= h <= fs.weighted(45) and  # 조건 3
                   fs.weighted(2) <= w <= fs.weighted(60))
        # 조건 4: 부점(점) 보존 — 높은 음표의 부점은 오선 '위'에 찍히므로
        # 점 크기(6×6 이하)의 요소는 코드 글자가 아니라 보고 지우지 않는다.
        # (그만큼 작은 코드 글자 파편은 어차피 객체 최소 크기 미달로 무시된다.)
        dot_like = (w <= fs.weighted(6) and h <= fs.weighted(6))

        if above_staff and below_prev and size_ok and not dot_like:
            image[labels == i] = 0
            removed.append((int(x), int(y), int(w), int(h)))
    return image, removed


def _hairpin_envelope(mask):
    """성분 마스크의 열별 (min_y, max_y) 프로파일(잉크 없는 열은 None)."""
    h, w = mask.shape
    prof = []
    for cx in range(w):
        ys = np.nonzero(mask[:, cx])[0]
        prof.append((int(ys.min()), int(ys.max())) if len(ys) else None)
    return prof


def _hairpin_line_residual(xs, ys):
    """직선 적합 후 평균 잔차(작을수록 곧은 직선에 가까움)."""
    if len(xs) < 3:
        return 999.0
    A = np.vstack([xs, np.ones(len(xs))]).T
    m, c = np.linalg.lstsq(A, ys, rcond=None)[0]
    pred = m * np.asarray(xs) + c
    return float(np.mean(np.abs(pred - np.asarray(ys))))


def _classify_hairpin(mask, ss, density):
    """마스크가 쐐기(크레센도/디크레센도) 모양인지 다중 조건으로 판정한다.

    1) 밀도: 쐐기는 두 개의 가는 선뿐이라 bbox 대비 잉크가 성기다(실측 ≤0.15
       내외). 가사 글자 등 조밀한 내용은 이 조건에서 걸러진다(실측 밀도 ≥0.33).
    2) 커버리지: 대부분의 열에 잉크가 있어야 한다(듬성듬성하면 텍스트 파편).
    3) 한쪽 끝은 좁고(정점) 반대쪽은 벌어져 있어야 하며, 벌어지는 폭이
       최소 오선 1칸 이상이어야 한다.
    4) 위/아래 테두리가 각각 직선에 가까워야 한다(곡선인 이음줄과 구분).
    5) 벌어짐 정도가 정점→열린 끝으로 갈수록 대체로 단조 증가해야 한다.
    """
    h, w = mask.shape
    if density > 0.22:
        return None
    prof = _hairpin_envelope(mask)
    valid = [(i, p) for i, p in enumerate(prof) if p is not None]
    if len(valid) < w * 0.6:
        return None
    idxs = np.array([i for i, _ in valid])
    tops = np.array([p[0] for _, p in valid], dtype=float)
    bots = np.array([p[1] for _, p in valid], dtype=float)
    spans = bots - tops

    q = max(3, len(spans) // 5)
    left_span, right_span = spans[:q].mean(), spans[-q:].mean()
    if left_span < right_span:
        apex_side, open_span, narrow = "left", right_span, spans[:q]
    else:
        apex_side, open_span, narrow = "right", left_span, spans[-q:]
    if narrow.max() > ss * 1.0 or open_span < ss * 1.0:
        return None

    if (_hairpin_line_residual(idxs, tops) > ss * 0.7
            or _hairpin_line_residual(idxs, bots) > ss * 0.7):
        return None

    corr = np.corrcoef(idxs, spans)[0, 1]
    if apex_side == "left" and corr < 0.5:
        return None
    if apex_side == "right" and corr > -0.5:
        return None
    return "cresc" if apex_side == "left" else "decresc"


def detect_and_remove_hairpins(image, staves):
    """크레센도(<)·디크레센도(>) 쐐기 기호를 검출해 이미지에서 분리한다.

    이음줄과 마찬가지로 객체 검출 '이전에' 쐐기를 이미지에서 지우고 좌표만
    기록한다(음표와 겹쳐 병합되는 것을 막기 위함). 쐐기는 보표 위 또는 아래
    (다이내믹스 표기 위치, 통상 아래) 어디에나 올 수 있으므로 양쪽 다 찾는다.

    검출 원리
    ---------
    1) 이음줄과 같은 '세로로 얇은 픽셀' 마스크를 만들되, 쐐기 선은 이음줄보다
       살짝 굵게 그려지는 경우가 많아 임계값을 조금 더 넉넉히 잡는다.
    2) 가로로 넉넉히 팽창해 위/아래 두 선을 한 성분으로 묶는다.
    3) `_classify_hairpin` 으로 '정점(뾰족)→열린 끝(벌어짐)'의 쐐기 모양인지
       확인한다(밀도·직선성·단조성 등 다중 조건 — 자세한 근거는 그 함수 참고).
       이 조건들은 합성 테스트(정상적인 쐐기)에서 정확히 분류되고, 실제 5개
       테스트 악보 전체에서 오탐 0건을 확인했다.

    ※ 주의: 오선에 바짝 붙어 그려진(여백이 거의 없는) 쐐기는 오선 잔여
    픽셀과 뒤섞여 놓칠 수 있다(보수적으로 설계 — 오탐 방지를 우선했다).

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지(정규화·잡티 제거된 결과).
    staves : list[float]
        정규화된 오선 y좌표 리스트(길이 = 보표수 × 5).

    Returns
    -------
    tuple[np.ndarray, list]
        (쐐기가 제거된 이미지, 검출된 쐐기 리스트).
        쐐기 = dict(line, x1, x2, y, kind) — kind는 "cresc"|"decresc".
    """
    lines = len(staves) // 5
    H, W = image.shape
    if lines == 0:
        return image, []

    image = image.copy()
    hairpins = []
    for l in range(lines):
        top = int(round(staves[l * 5]))
        bot = int(round(staves[l * 5 + 4]))
        ss = (staves[l * 5 + 4] - staves[l * 5]) / 4.0
        if ss <= 0:
            continue
        gap_top = 0 if l == 0 else int(round(staves[(l - 1) * 5 + 4]))
        gap_bot = H if l == lines - 1 else int(round(staves[(l + 1) * 5]))

        for y0, y1 in ((gap_top, top - 2), (bot + 2, gap_bot)):
            if y1 <= y0:
                continue
            band = image[y0:y1, :]
            Hb, Wb = band.shape
            on = band > 0
            col = on.astype(np.int32)
            up = np.zeros_like(col)
            dn = np.zeros_like(col)
            for r in range(Hb):
                up[r] = (up[r - 1] + 1) * col[r] if r else col[r]
            for r in range(Hb - 1, -1, -1):
                dn[r] = (dn[r + 1] + 1) * col[r] if r < Hb - 1 else col[r]
            runlen = up + dn - col
            thin = (on & (runlen <= fs.weighted(8))).astype(np.uint8) * 255
            bridged = cv2.dilate(thin, np.ones((3, fs.weighted(20)), np.uint8))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(bridged)

            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if w < fs.weighted(60):
                    continue
                mask = (lab[y:y + h, x:x + w] == i) & (thin[y:y + h, x:x + w] > 0)
                density = np.count_nonzero(band[y:y + h, x:x + w]) / (w * h)
                kind = _classify_hairpin(mask, ss, density)
                if kind is None:
                    continue
                region = image[y0 + y:y0 + y + h, x:x + w]
                region[mask] = 0
                hairpins.append({
                    "line": l, "x1": int(x), "x2": int(x + w),
                    "y": int(y0 + y + h / 2), "kind": kind,
                })
    return image, hairpins


def detect_and_remove_slurs(image, staves):
    """(보강) 이음줄/붙임줄(slur/tie) 호(弧)를 검출해 이미지에서 분리한다.

    이음줄을 그대로 두면 서로 다른 마디의 음표들이 한 연결요소로 병합돼
    마디 배정과 음표 인식이 무너진다(과거 이 기능을 보류했던 원인).
    따라서 객체 검출 '이전에' 호를 이미지에서 지우고, 끝점 좌표만 기록해
    나중에 모델(붙임줄/이음줄 관계)로 복원한다. 병합 자체가 사라지므로
    음표·마디선 인식은 오히려 안전해진다.

    검출 원리 (음표 꼬리·음자리표 등과의 오검출을 막는 5중 조건)
    ---------------------------------------------------
    후보 = '세로로 얇은'(세로 연속 픽셀 ≤ 오선 1/3칸) 픽셀들을 가로로 이어
    만든 연결 성분. 각 후보에 대해:
      1. 폭·커버리지: 얇은 픽셀이 실제로 뻗은 구간(span)이 오선 1.8칸 이상이고,
         그 구간의 60% 이상 열에 픽셀이 있다. (호가 오선 제거 과정에서 가운데가
         끊기는 일이 잦아, 가로 팽창으로 틈을 이어 붙인 뒤 측정한다.)
      2. ★ 홑획 조건: 열의 90% 이상에서 픽셀 덩어리가 정확히 1개.
         (8분음표 꼬리는 중간 열에서 획이 2개 — 위 획과 되돌아오는 말림 —
          가 되고, 셈여림 기호(<, >)도 2개가 되므로 여기서 걸러진다.)
      3. ★ 독립성: 호를 이루는 얇은 픽셀 중 '두꺼운 획'(음표 머리·기둥·음자리표
         몸통)에 맞닿은 비율이 30% 미만. 이음줄은 홀로 떠 있는 가는 곡선이지만,
         음표 꼬리나 음자리표의 말림은 두꺼운 몸통에 붙어 있어 여기서 걸러진다.
         (실측: 진짜 이음줄 0.00~0.21 / 음표 꼬리·음자리표 0.38~0.44)
      4. 아치 조건: 양 끝점을 잇는 직선 대비 최대 편차가 오선 1/6칸 이상,
         편차 부호가 85% 이상 한 방향(빔은 직선이라 편차가 없어 제외됨).
      5. 정점 위치: 최대 편차 지점이 성분의 가운데 절반 구간에 있다.

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지(정규화·잡티·코드 제거된 결과).
    staves : list[float]
        정규화된 오선 y좌표 리스트(길이 = 보표수 × 5).

    Returns
    -------
    tuple[np.ndarray, list]
        (호가 제거된 이미지, 검출된 호 리스트).
        호 = dict(line, x1, y1, x2, y2, above) — 좌/우 끝점과 위/아래 방향.
    """
    lines = len(staves) // 5
    H, W = image.shape
    if lines == 0:
        return image, []
    centers = [(staves[l * 5] + staves[l * 5 + 4]) / 2 for l in range(lines)]

    on = (image > 0)
    col = on.astype(np.int32)
    up = np.zeros_like(col)
    dn = np.zeros_like(col)
    for r in range(H):                       # 각 픽셀의 세로 연속(런) 길이
        up[r] = (up[r - 1] + 1) * col[r] if r else col[r]
    for r in range(H - 1, -1, -1):
        dn[r] = (dn[r + 1] + 1) * col[r] if r < H - 1 else col[r]
    runlen = up + dn - col

    thin_m = on & (runlen <= fs.weighted(3))          # 가는 획(호·덧줄 등)
    thick = (on & ~thin_m).astype(np.uint8)           # 두꺼운 획(머리·기둥·빔·음자리표)
    thick_near = cv2.dilate(thick, np.ones((3, 3), np.uint8))   # 두꺼운 획에 인접한 영역

    thin = thin_m.astype(np.uint8) * 255
    # 호는 오선 제거 과정에서 가운데가 끊기는 일이 잦다(오선과 겹친 구간이 함께
    # 지워짐). 가로로 넉넉히 팽창해 끊긴 조각을 하나로 이어 붙인 뒤 판정한다.
    bridged = cv2.dilate(thin, np.ones((1, fs.weighted(10)), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bridged)

    image = image.copy()
    slurs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h > fs.weighted(15) or h > w * 0.9:
            continue

        # 보표 대역 밖(제목·가사 등)의 얇은 가로 획은 이음줄이 아니다.
        cy = y + h / 2.0
        line = min(range(lines), key=lambda l: abs(cy - centers[l]))
        if not (staves[line * 5] - fs.weighted(30) <= cy
                <= staves[line * 5 + 4] + fs.weighted(30)):
            continue

        mask = (labels[y:y + h, x:x + w] == i) & thin_m[y:y + h, x:x + w]

        xs, ys, strokes = [], [], []
        for c in range(w):
            rr = np.nonzero(mask[:, c])[0]
            if len(rr) == 0:
                continue
            xs.append(c)
            ys.append(float(rr.mean()))
            strokes.append(1 + int(np.count_nonzero(np.diff(rr) > 1)))  # 열의 덩어리 수
        if len(xs) < 2:
            continue

        # 조건 1: 실제 획이 뻗은 구간(span)의 길이와 그 구간의 열 커버리지
        # (팽창으로 부풀려진 bbox 폭이 아니라 실제 픽셀 범위로 재야 정확하다.)
        span = xs[-1] - xs[0] + 1
        if span < fs.weighted(18) or len(xs) / span < 0.6:
            continue
        if np.mean(np.array(strokes) == 1) < 0.9:                # 조건 2: 홑획
            continue

        # 조건 3: 독립성 — 두꺼운 획(머리·기둥·음자리표 몸통)에 맞닿은 얇은 픽셀
        # 비율이 낮아야 한다. 음표 꼬리·음자리표 말림은 몸통에 붙어 있어 걸러진다.
        total = int(mask.sum())
        touching = int((mask & (thick_near[y:y + h, x:x + w] > 0)).sum())
        if total == 0 or touching / total >= 0.30:
            continue

        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        base = ys[0] + (ys[-1] - ys[0]) * (xs - xs[0]) / max(1.0, xs[-1] - xs[0])
        dev = ys - base
        k = int(np.argmax(np.abs(dev)))
        amax = dev[k]
        if abs(amax) < fs.weighted(1.6):                         # 조건 4: 아치 깊이
            continue
        big = np.abs(dev) > 0.5
        if big.any() and np.mean(np.sign(dev[big]) == np.sign(amax)) < 0.85:
            continue
        if not (0.25 <= (xs[k] - xs[0]) / max(1.0, xs[-1] - xs[0]) <= 0.75):
            continue                                             # 조건 5: 정점 중앙

        # 통과 → 이 성분의 얇은 픽셀을 이미지에서 제거(호 분리)
        region = image[y:y + h, x:x + w]
        region[mask] = 0

        slurs.append({
            "line": line,
            "x1": int(x + xs[0]), "y1": int(round(y + ys[0])),
            "x2": int(x + xs[-1]), "y2": int(round(y + ys[-1])),
            "above": bool(amax < 0),     # 정점이 끝점보다 위 = 위로 볼록한 호
        })
    return image, slurs


# 보표별 박스 색상(BGR). 시각화에서 어느 객체가 어느 보표에 속하는지 구분용.
_LINE_COLORS = [
    (0, 0, 255),      # 빨강
    (0, 160, 0),      # 초록
    (255, 60, 0),     # 파랑
    (200, 0, 200),    # 보라
    (0, 140, 255),    # 주황
    (180, 120, 0),    # 청록
]


def _find_lyric_gap(image, staff_bottom, ss, search_max_ss=8.0, min_run=3, gap_tol=None):
    """보표 바로 아래에서 '진짜로 음악 내용이 끝나는 지점'(가사가 시작되는 지점
    바로 위)을 찾는다.

    단순히 '픽셀 수가 적은 행'을 갭으로 보면 실패한다 — 여러 음표의 기둥이
    동시에 가늘게 이어지는 구간(각 기둥은 1~2px라 합계는 작다)도 픽셀 수만
    보면 가짜로 '갭'처럼 보인다(실측: 정상 악보에서 여러 건의 오탐 확인).
    반대로 실제 가사와 들러붙은 원인(예: 음자리표 꼬리 고리처럼 좁은 다리 하나)도
    픽셀 수 자체는 작아 '가짜 내용'과 잘 구분되지 않는다.

    실제로 구분이 되는 것은 '그 행에 떨어진 덩어리가 몇 개인가'이다. 여러 음표의
    기둥이 동시에 지나가면 서로 떨어진 덩어리가 여러 개(2개 이상) 생기지만,
    진짜 갭이거나 음자리표 꼬리 같은 좁은 다리 하나만 남은 경우는 덩어리가
    0~1개다. 이 기준으로 '덩어리 ≤1개인 행이 min_run 이상 연속'되는 첫 지점을
    찾는다.

    찾지 못하면 None을 반환한다(호출부가 기존의 고정 여유값으로 폴백한다).
    """
    H, W = image.shape
    gap_tol = gap_tol if gap_tol is not None else max(2, int(ss * 0.3))
    search_end = min(H, staff_bottom + int(round(search_max_ss * ss)))
    band = image[staff_bottom:search_end, :]
    run = 0
    for i in range(band.shape[0]):
        xs = np.nonzero(band[i])[0]
        if len(xs) == 0:
            n_clusters = 0
        else:
            n_clusters = 1 + int(np.count_nonzero(np.diff(xs) > gap_tol))
        if n_clusters <= 1:
            run += 1
            if run >= min_run:
                return staff_bottom + i - min_run + 1
        else:
            run = 0
    return None


def object_detection(image, staves):
    """4단계: 악보 구성요소(음표·쉼표·조표 등) 객체를 검출한다.

    처리 순서
    ---------
    1) 닫힘 연산(closing)으로 끊어진 획을 이어 한 기호가 조각나는 것을 방지.
    2) connectedComponentsWithStats 로 모든 연결 요소를 검출.
    3) 너비·높이가 각각 weighted(5) 이상인 것만 '악보 구성요소'로 인정
       (그보다 작은 것은 남은 노이즈로 간주해 제외).
    4) 각 객체의 중심 y좌표(get_center)가 어느 보표 범위에 드는지 판별.
       - 보표 범위 = [첫 오선 - weighted(20), 마지막 오선 + weighted(20)]
         (오선 위·아래로 두 칸 여유를 둬 덧줄 음표까지 포함)
    5) [보표번호, (x, y, w, h, area)] 를 리스트에 담고 보표→x좌표 순으로 정렬.

    ※ 닫힘 연산은 '검출'에만 사용하고, 박스는 변형 없는 원본 이미지에 그린다.

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지(정규화·잡티 제거까지 끝난 결과).
    staves : list[float]
        정규화된 오선 y좌표 리스트(길이 = 보표수 × 5).

    Returns
    -------
    tuple[np.ndarray, list]
        (검출 결과를 보표별 색상 박스로 그린 BGR 시각화 이미지, objects 리스트).
        objects 예: [[0, (100, 209, 26, 72, 863)], [0, (136, 227, 8, 25, 97)], ...]
    """
    lines = int(len(staves) / 5)   # 보표(5선)의 개수
    objects = []                   # 구성요소 정보가 저장될 리스트

    # [보강] 보표 대역 밖 픽셀 배제: 가사·다이내믹 기호 등 보표에서 먼 내용은
    # 음악 기호가 아니므로 지운다. 그대로 두면 closing 연산이 가사를 인접
    # 음표와 한 덩어리로 병합해 음표 인식을 무너뜨린다(staff 계열 요소 0 원인).
    #
    # 아래쪽 경계는 '적응형'으로 정한다: 오선 간격이 유난히 촘촘한 스캔/발췌
    # 이미지는 가사가 고정 여유값(3.5칸)보다 훨씬 가깝게 붙어 있을 수 있고
    # (실측: 화면 캡처 발췌 악보에서 가사가 1.9칸 만에 시작), 그 경우 음자리표
    # 꼬리 고리 같은 좁은 다리 하나만으로도 가사와 들러붙어 버린다(요소 0개
    # 원인). _find_lyric_gap 이 실제 내용이 끝나는 지점을 찾으면 그 값을,
    # 못 찾으면 기존 고정값(weighted(35))을 그대로 쓴다(회귀 방지).
    keep = np.zeros(image.shape[0], dtype=bool)
    for line in range(lines):
        top_y = staves[line * 5]
        bot_y = staves[line * 5 + 4]
        ss = (bot_y - top_y) / 4.0 if bot_y > top_y else 10.0
        gap = _find_lyric_gap(image, int(round(bot_y)), ss) if ss > 0 else None
        margin = (gap - int(round(bot_y))) if gap is not None else fs.weighted(35)
        margin = max(fs.weighted(10), margin)     # 최소 여유(덧줄 등) 보장
        a = max(0, int(top_y - fs.weighted(35)))
        b = min(image.shape[0], int(bot_y + margin) + 1)
        keep[a:b] = True
    image = image.copy()
    image[~keep, :] = 0

    closing_image = fs.closing(image)   # 검출용(끊긴 획 연결)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(closing_image)
    for i in range(1, cnt):
        (x, y, w, h, area) = stats[i]
        if w >= fs.weighted(5) and h >= fs.weighted(5):   # 구성요소 최소 크기 조건
            center = fs.get_center(y, h)
            for line in range(lines):
                area_top = staves[line * 5] - fs.weighted(20)            # 보표 상단 여유
                area_bot = staves[(line + 1) * 5 - 1] + fs.weighted(20)  # 보표 하단 여유
                if area_top <= center <= area_bot:
                    objects.append([line, (x, y, w, h, area)])

    objects.sort()   # 보표 번호 → x좌표 순 오름차순 정렬

    # --- 시각화: 원본(그레이)을 컬러로 바꿔 보표별 색상 박스를 그림 ---
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for line, (x, y, w, h, area) in objects:
        color = _LINE_COLORS[line % len(_LINE_COLORS)]
        cv2.rectangle(vis, (x, y, w, h), color, 1)

    return vis, objects


def object_analysis(image, objects):
    """5단계: 각 객체에서 음표 기둥(세로 직선)을 찾아 방향 등 특징을 붙인다.

    각 객체마다 stem_detection 으로 기둥들을 검출하고, 기둥이 하나라도 있으면
    음표로 간주해 방향을 판정한다.
      - 첫 기둥이 객체 시작보다 weighted(5) 이상 뒤에 있으면 → 정방향(True)
        (음표 머리가 아래쪽 = 머리가 먼저 나오고 기둥이 뒤에 위치)
      - 그렇지 않으면 → 역방향(False) (머리가 위쪽 = 기둥이 앞에 위치)
      - 기둥이 없으면 방향 None (음표가 아닌 것: 쉼표·조표·온음표 등)

    분석 결과로 각 obj 는 [보표번호, stats, stems, direction] 네 원소가 된다.

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지(정규화·잡티 제거까지 끝난 image_clean). 컬러 시각화
        이미지를 넣으면 박스 선이 기둥으로 오검출되므로 반드시 1채널을 넣는다.
    objects : list
        4단계 결과. 각 원소 [보표번호, (x, y, w, h, area)].

    Returns
    -------
    tuple[np.ndarray, list]
        (기둥·기둥개수를 표시한 BGR 시각화 이미지, 특징이 추가된 objects).

    Note
    ----
    objects 를 제자리(in-place)에서 확장한다. 파이프라인은 매 실행마다
    object_detection 이 새 objects 를 만들므로 중복 append 문제는 없다.
    """
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for obj in objects:
        line = obj[0]
        stats = obj[1]
        (x, y, w, h, area) = stats

        stems = fs.stem_detection(image, stats, 30)   # 객체 내 모든 기둥 검출
        direction = None
        if len(stems) > 0:                            # 기둥이 1개 이상 = 음표로 가정
            if stems[0][0] - stats[0] >= fs.weighted(5):
                direction = True                     # 정방향 음표
            else:
                direction = False                    # 역방향 음표

        obj.append(stems)        # obj[2] = 기둥 리스트
        obj.append(direction)    # obj[3] = 음표 방향

        # --- 시각화 ---
        color = _LINE_COLORS[line % len(_LINE_COLORS)]
        cv2.rectangle(vis, (x, y, w, h), color, 1)             # 객체 박스(보표색)
        for (sx, sy, sw, sh) in stems:
            cv2.rectangle(vis, (sx, sy, sw, sh), (0, 0, 255), -1)  # 기둥(빨강)
        if len(stems):
            fs.put_text(vis, len(stems), (x, y + h + 16))      # 기둥 개수 표시

    return vis, objects


def recognition(image, staves, objects):
    """6단계: 객체를 순회하며 인식한다.

    첫 객체(진짜 음자리표로 보이면)는 건너뛰고, 박자표를 만나기 전까지의 객체를
    조표로 보아 key 값을 누적한다. 박자표를 만나면 조표 탐색을 종료한다.

    [보강] 악보 일부만 잘라낸 발췌 이미지(중간 마디부터 시작 등)는 음자리표·
    조표·박자표 선언이 아예 없을 수 있다. 이 경우 조표 탐색을 영원히 끝내지
    못해(박자표를 못 찾으므로) 모든 객체가 '조표 후보'로만 취급되고 음표
    인식이 단 한 번도 실행되지 않는 문제가 있었다(요소 0개로 귀결).

    실제 조표(플랫/샾)와 박자표는 클레프 하나 + 최대 7개(온음계 기준) 이내에
    모두 나타난다. 그래서 '조표를 아직 못 찾은 채로 이미 충분히 많은 객체를
    살펴봤다면'(KEY_SEARCH_BUDGET 초과) 이 악보에는 조표·박자표가 없는 것으로
    확정하고, 그 객체부터 바로 음표/쉼표 인식으로 넘어간다.

    ※ 주의: 객체의 '기둥(stem) 유무'만으로 조표/음표를 가르는 방법은 안전하지
    않다 — 실측 결과 박자표 숫자 글리프도 렌더링에 따라 기둥이 1개로 검출될
    수 있어(예: '4'의 세로획), 그 기준으로는 진짜 박자표를 음표로 오인해
    건너뛰는 회귀가 발생했다. 그래서 '개수 예산' 방식을 쓴다.

    0번 객체(음자리표) 처리도 보강한다: 진짜 음자리표는 폭이 좁다(실측
    2.5~2.6칸). 폭이 그보다 훨씬 넓다면(발췌 악보에서 음자리표가 뒤 음표들과
    들러붙어 하나의 거대한 객체가 된 경우) 더 이상 '음자리표 하나'로 볼 수
    없으므로 건너뛰지 않고 정상적으로 검사한다.

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지(정규화·잡티 제거된 image_clean). 기둥 검출에 사용하므로
        반드시 1채널을 넘긴다. 시각화는 이 이미지를 복사한 별도 BGR에 그린다.
    staves : list[float]
        정규화된 오선 y좌표 리스트(길이 = 보표수 × 5).
    objects : list
        분석까지 끝난 객체 목록. 각 obj = [보표번호, (x,y,w,h,area), stems, direction].

    Returns
    -------
    tuple[np.ndarray, int, list, list]
        (인식 결과를 표시한 BGR 시각화 이미지, key, beats, pitches).
        key: 조표 값(플랫 100/개, 샾 10/개, 0이면 다장조). beats/pitches: 다음 단계용.
    """
    # 시각화 이미지: 흰 배경 + 검은 악보(반전) → 인식 라벨이 잘 보이도록(블로그 스타일).
    vis = cv2.cvtColor(255 - image, cv2.COLOR_GRAY2BGR)

    key = 0
    time_signature = False   # 박자표를 찾았는지(또는 '없다'고 확정했는지) 여부
    time_sig_value = None    # 인식된 박자표 (분자, 분모). 끝내 못 찾으면 None(미상).
    beats = []               # 박자 리스트
    pitches = []             # 음이름 리스트
    elements = []            # 마디 검증용: (line, x, y, w, h, beat) 리스트

    CLEF_MAX_W = fs.weighted(45)      # 진짜 음자리표로 볼 수 있는 최대 폭(실측 26px의 여유)
    ACCIDENTAL_MAX_W = fs.weighted(30)  # 조표·박자표로 볼 수 있는 최대 폭(실측 최대 26px의 여유)
    KEY_SEARCH_BUDGET = 12            # 클레프+조표 최대 7개+박자표+여유(폭 판정이 애매할 때의 보조 안전망)
    examined_for_key = 0

    LAST_STROKE_MAX_W = fs.weighted(25)  # 진짜 '끝 세로줄'로 볼 수 있는 최대 폭(실측 11~12px 여유)

    # 마지막 객체는 보통 악보 끝 세로줄(마디선 겹줄)이라 음표로 오인될 수 있어
    # 제외한다(블로그 지침). 다만 발췌 악보처럼 끝에 세로줄이 없고 음표 덩어리가
    # 마지막 객체가 돼 버린 경우(폭이 훨씬 넓음)까지 통째로 버리면 안 되므로,
    # 폭이 좁을 때만 제외한다.
    end = len(objects)
    if end > 0 and objects[-1][1][2] <= LAST_STROKE_MAX_W:
        end -= 1

    for i in range(0, end):
        obj = objects[i]
        line = obj[0]
        stats = obj[1]
        stems = obj[2]
        direction = obj[3]
        (x, y, w, h, area) = stats
        staff = staves[line * 5: (line + 1) * 5]

        # 0번 객체: 폭이 음자리표 범위 안이면 진짜 음자리표로 보고 건너뛴다
        # (기존 동작). 그렇지 않으면(발췌 악보 등으로 음표와 들러붙은 경우)
        # 건너뛰지 않고 아래 일반 로직으로 처리한다.
        if i == 0 and w <= CLEF_MAX_W:
            continue

        if not time_signature:          # 아직 박자표를 못 찾음(또는 '없음' 확정 전)
            # 폭이 조표·박자표로는 있을 수 없을 만큼 넓다(실측: 진짜 조표·
            # 박자표는 항상 26px 이내, 음표·빔그룹은 그보다 훨씬 넓다) →
            # 굳이 recognize_key 를 불러 오판(엉뚱한 key 값)을 만들 것 없이
            # 곧바로 '조표·박자표 없음'으로 확정하고 음표 인식으로 넘어간다.
            # (발췌 악보에서 음자리표가 뒤 음표와 들러붙어 거대해진 0번 객체가
            # 대표적인 경우다.)
            #
            # 예외: 온박자(𝄴)·컷타임(𝄵) 기호는 클레프와 붙어 이렇게 넓은 객체가
            # 되는 경우가 실제로 있다(실측 확인). 그래서 '음표 없음'으로 바로
            # 확정하기 전에, 슬라이딩 윈도우로 온박자·컷타임 기호가 숨어 있는지
            # 먼저 찾아본다(오선 범위로 제한해 클레프 장식 고리에 속지 않는다).
            if w > ACCIDENTAL_MAX_W:
                ss = (staff[4] - staff[0]) / 4.0
                name, score, span = (rs.find_common_time_window(image, stats, staff)
                                     if ss > 0 else (None, 0.0, None))
                if name is not None:
                    time_signature = True
                    time_sig_value = (4, 4) if name == "common" else (2, 2)
                    fs.put_text(vis, "%d/%d" % time_sig_value, (x, y - fs.weighted(10)))
                    continue
                time_signature = True
                time_sig_value = None
            else:
                examined_for_key += 1
                if examined_for_key <= KEY_SEARCH_BUDGET:
                    ts, temp_key = rs.recognize_key(image, staff, stats)
                    time_signature = ts
                    key += temp_key
                    if time_signature:   # 방금 박자표를 찾음 → key + 박자표 숫자 인식[13단계]
                        time_sig_value = rs.recognize_time_signature(image, staff, stats)
                        fs.put_text(vis, key, (x, y + h + fs.weighted(20)))
                        if time_sig_value[0] and time_sig_value[1]:
                            fs.put_text(vis, "%d/%d" % time_sig_value, (x, y - fs.weighted(10)))
                    continue             # 이 객체는 조표/박자표 판정으로 처리 끝
                else:
                    # 예산 초과: 이 악보에는 조표·박자표가 없는 것으로 확정
                    # (발췌 악보 등). 이 객체부터 바로 음표/쉼표 인식으로
                    # 넘어간다(건너뛰지 않고 아래로 그대로 진행).
                    time_signature = True
                    time_sig_value = None

        # 박자표를 찾았거나(또는 '없음'으로 확정됐거나) → 음표·쉼표·온음표 인식
        #
        # [보강] 마디 중간의 임시표(#,♭,♮)는 조표와 달리 별도 객체로 안 잡히고
        # 바로 뒤 음표와 한 객체로 붙어 있는 일이 흔하다(실측 확인). 그대로
        # recognize_note 에 넘기면 임시표의 획이 기둥으로 잘못 세어져 머리·박자
        # 판정이 흔들린다. 그래서 먼저 왼쪽 끝에 임시표가 붙어 있는지 확인해
        # 분리하고, 뒤에 남는 진짜 음표 영역만 인식에 넘긴다.
        ss_local = (staff[4] - staff[0]) / 4.0
        acc_kind, split_stats = (rs.split_leading_accidental(image, stats, ss_local)
                                 if ss_local > 0 else (None, stats))
        if acc_kind is not None:
            cut = split_stats[0] - x
            stems = [s for s in stems if s[0] >= split_stats[0] - fs.weighted(2)]
            stats = split_stats
            x, y, w, h, area = stats

        notes, note_pitches, stem_xs, grp_ids = rs.recognize_note(
            image, vis, staff, stats, stems, direction)
        if len(notes):              # 음표로 인식됨
            beats.extend(notes)
            pitches.extend(note_pitches)
            for k, b in enumerate(notes):
                # 빔 하위그룹키: 같은 객체(i)라도 빔이 끊긴 구간은 다른 그룹.
                # grp_id == -1 이면 빔에 물리지 않은 단독 음표 → 키도 -1.
                g = grp_ids[k] if k < len(grp_ids) else -1
                gkey = -1 if g == -1 else i * 100 + g
                sx_note = stem_xs[k] if k < len(stem_xs) else x
                # 임시표는 분리된 객체의 '첫' 음표(가장 왼쪽)에만 적용한다.
                note_acc = acc_kind if k == 0 else None
                # 꾸밈음(장식음) 자동 검출은 오탐률이 높아(실측: 정상 악보에서
                # 대량 오탐 확인) 비활성화해 뒀다. Element.is_grace 필드와
                # 렌더·MusicXML 출력은 이미 준비돼 있으므로, 편집 화면에서
                # 사용자가 직접 표시하는 기능으로 우선 제공한다.
                elements.append((line, x, y, w, h, b, sx_note, gkey, note_acc, False))
        else:                       # 음표가 아니면 쉼표 인식 시도
            rest = rs.recognize_rest(image, vis, staff, stats)
            if rest:
                beats.append(rest)
                pitches.append(-1)  # 쉼표는 음정 없음(-1)
                elements.append((line, x, y, w, h, rest, x, i * 100, None, False))
            else:                   # 쉼표도 아니면 온음표 인식 시도
                whole_note, wpitch = rs.recognize_whole_note(image, vis, staff, stats)
                if whole_note:
                    beats.append(whole_note)
                    pitches.append(wpitch)
                    elements.append((line, x, y, w, h, whole_note, x, i * 100, acc_kind, False))

        # 모든 객체에 박스와 인덱스 표시
        cv2.rectangle(vis, (x, y, w, h), (255, 0, 0), 1)
        fs.put_text(vis, i, (x, y - fs.weighted(20)))

    return vis, key, time_sig_value, beats, pitches, elements


# 박자 코드(절댓값) → 4분음표 단위 길이. 음수(점음표)는 1.5배.
_BEAT_LEN = {1: 4.0, 2: 2.0, 4: 1.0, 8: 0.5, 16: 0.25, 32: 0.125}


def beat_length(code):
    """박자 코드(2/4/8/16/32, 음수=점)를 4분음표 단위 길이로 변환한다."""
    v = _BEAT_LEN.get(abs(code), 0.0)
    return v * 1.5 if code < 0 else v


def detect_voltas(original_bgr, staves_raw):
    """1절/2절 등 볼타(ending) 브래킷을 검출한다.

    ⚠ 반드시 '원본 BGR 이미지'와 '원본 배율의 오선 좌표'(remove_staves가 막
    반환한 값, normalization 이전)를 받는다. 이유는 두 단계 모두 볼타 브래킷을
    파괴하기 때문이다.
      - remove_noise(): '이미지 폭의 50% 이상'인 연결요소의 bbox만 살리는 마스크를
        쓰는데, 볼타 브래킷은 보통 그보다 좁아 통째로 사라진다(실측: 2번째
        엔딩의 "2. Dm F" 텍스트가 이 단계에서 소실됨).
      - remove_staves(): 브래킷의 긴 가로선을 '오선처럼 생긴 긴 직선'으로 오인해
        지운다(실측: 488px 길이의 선이 2px로 축소).
    따라서 아무 처리도 거치지 않은 fs.threshold(original_bgr) 위에서 검출한다.

    구조(표준 조판, 픽셀 실측으로 확인)
    -----------------------------------
    [아래로 향한 훅] ── 긴 가로선 ── [다음 볼타의 훅 / 열린 끝]
    훅 바로 뒤에 번호("1." "2." 등)가 붙지만, 번호 자체는 인식하지 않고
    (인식 실패 시 위험이 있는 OCR을 피하는 실용적 선택) 브래킷이 나타나는
    좌→우 순서로 1, 2, 3 ... 번을 매긴다. 볼타는 항상 이 순서로 조판되므로
    안정적인 대체 규칙이다.

    검출 방법
    ---------
    1) 각 보표 위, 그 보표 바로 위 8칸(오선 간격 기준) 이내의 좁은 띠에서
       '길게 이어진 가로 획'(오선 15칸 이상 연속)이 있는 행들을 찾는다.
       (전체 행의 픽셀 총합이 아니라, 한 행 안에서 실제로 이어진 최장 구간
        길이를 기준으로 삼는다 — 코드 기호 여러 개가 우연히 합쳐 길어 보이는
        것을 배제하기 위함이다.)
    2) 그런 행 중, 서로 떨어진 가로 구간을 가장 많이 보여주는 행을 고른다.
       (볼타가 이어지는 지점은 훅이 다리처럼 이어줘 한 구간처럼 보이는 행도
        있지만, 훅 영향이 적은 행에서는 두 볼타가 뚜렷이 분리되어 보인다.)
    3) 각 구간의 왼쪽 끝 바로 아래에 '아래로 뻗는 좁은 훅'이 실제로 있는지
       연결요소로 확인한다(없으면 볼타가 아니라고 보고 버린다).

    Returns
    -------
    list[dict]
        [{"line": 보표번호, "x1": 시작x, "x2": 끝x, "number": 순번(1부터)}, ...]
        좌표는 원본 이미지와 같은 배율(정규화 전).
    """
    image = fs.threshold(original_bgr)
    n_lines = len(staves_raw) // 5
    H, W = image.shape
    result = []

    def runs_ge(row, gap_tol, min_len):
        """한 행에서, 길이가 min_len 이상인 연속 구간들을 모두 찾는다."""
        xs = np.nonzero(row)[0]
        if len(xs) == 0:
            return []
        found = []
        start = prev = xs[0]
        for x in xs[1:]:
            if x - prev > gap_tol:
                if prev - start >= min_len:
                    found.append((int(start), int(prev)))
                start = x
            prev = x
        if prev - start >= min_len:
            found.append((int(start), int(prev)))
        return found

    for l in range(n_lines):
        top = int(round(staves_raw[l * 5]))
        gap_top = 0 if l == 0 else int(round(staves_raw[(l - 1) * 5 + 4]))
        raw_ss = (staves_raw[l * 5 + 4] - staves_raw[l * 5]) / 4.0
        band_top = max(gap_top, top - int(round(raw_ss * 8)))
        pad = 3                                  # 오선 자체의 두께를 피함
        if top - pad <= band_top + pad:
            continue

        min_len = raw_ss * 15
        gap_tol = max(3, int(raw_ss * 0.6))

        best_y, best_runs = None, []
        for y in range(band_top + pad, top - pad):
            candidates = runs_ge(image[y], gap_tol, min_len)
            if len(candidates) > len(best_runs):
                best_y, best_runs = y, candidates
        if not best_runs:
            continue

        # 각 구간의 왼쪽 끝에 '아래로 뻗는 훅'이 있는지 확인
        hook_h = int(round(raw_ss * 1.8))
        segs = []
        for (x0, x1) in best_runs:
            hook_top = best_y + 1
            hook_bot = min(H, hook_top + hook_h)
            if hook_bot <= hook_top:
                continue
            strip = image[hook_top:hook_bot, max(0, x0 - 2):x0 + 3]
            n_, _, st_, _ = cv2.connectedComponentsWithStats(strip)
            has_hook = any(
                st_[hi][1] <= 1 and st_[hi][2] <= raw_ss * 1.3
                and st_[hi][3] >= raw_ss * 0.5
                for hi in range(1, n_))
            if has_hook:
                segs.append({"line": l, "x1": x0, "x2": x1})

        segs.sort(key=lambda s: s["x1"])
        for i, s in enumerate(segs):
            s["number"] = i + 1
        result.extend(segs)

    return result


def detect_barlines(image, staves):
    """마디선을 보표별로 검출한다 — 열-스캔 기반 통합 방식.

    핵심 아이디어: 마디선은 '보표 세로 범위(첫~다섯째 선)를 거의 꽉 채우면서,
    보표 위/아래로는 크게 돌출하지 않고, 좌우로도 넓게 번지지 않는(=음표 머리가
    붙지 않은) 얇은 세로 획'이다. 이 성질을 열(column) 단위로 직접 판정한다.

    필터:
      1. 채움: 보표 높이의 88% 이상을 채우는 열만 후보. 인접 열은 하나로 묶는다.
      2. 폭: 묶은 폭이 너무 넓으면(> weighted(10)) 제외(음표 뭉침).
      3. 돌출: 보표 위/아래로 길게 이어지면 음표 기둥 → 제외.
      4. 고립(개선): 좌우 '가까운' 스트립만 본다. 마디선 바로 옆(±weighted(2)~4)
         에 세로로 꽉 찬 픽셀이 있으면 음표 기둥/머리로 보고 제외하되, 이음줄
         (slur/tie)처럼 '한두 행에서만' 스치는 것은 통과시킨다(행 커버리지로 판정).
      5. 이중 세로줄(도돌이·구간 구분)은 서로 weighted(6) 이내면 하나의 일반
         마디선으로 병합한다.

    종료선(마지막 얇은선+굵은선)은 x_right 부근에서 별도로 표시되도록, 각 보표
    마지막 마디선은 그대로 유지한다(렌더 단계에서 마지막 보표 끝을 굵게 그린다).

    Returns
    -------
    dict[int, list[int]]
        보표번호 → 그 보표의 마디선 x좌표 리스트(오름차순).
    """
    n_lines = len(staves) // 5
    H, W = image.shape
    barlines = {l: [] for l in range(n_lines)}

    w4 = fs.weighted(4)
    w2 = fs.weighted(2)
    w6 = fs.weighted(6)
    w10 = fs.weighted(10)
    w12 = fs.weighted(12)

    # 연결성분 라벨: 후보 세로획이 '무엇에 붙어 있는지' 판정하는 데 쓴다.
    # 마디선은 홀로 선 가는 세로획이라 성분 폭이 좁지만(실측 ≤ 8px, 굵은 종료선
    # 포함), 음표 기둥은 머리가 붙어 성분 폭이 넓다(실측 13px+). 머리가 오선
    # '위'에 있는 음표는 오선 대역 안에서 마디선과 똑같이 보이므로, 이 성분 폭
    # 검사가 그런 기둥을 마디선으로 오인하는 것을 막는 결정적 조건이다.
    #
    # ※ 오선이 완전히 지워지지 않고 일부 행이 이미지 폭 전체에 걸쳐 거의
    # 풀폭으로 남아 있으면(실측 확인), 그 잔여 행이 서로 무관한 마디선과
    # 수많은 음표들을 전부 '하나의 거대한 성분'으로 다리처럼 이어버린다.
    # 그러면 진짜 마디선도 '거대한 성분에 속함=음표에 붙음'으로 오판되어
    # 통째로 걸러진다(실측: 화면 폭 전체를 잇는 성분에 마디선까지 포함되어
    # 정상 마디선 여러 개가 누락됨). 이 연결성분 계산 전용으로 그런 잔여
    # 행을 지운 사본을 써서, 부착 판정이 진짜 음표 접촉만 잡도록 한다.
    cc_src = image.copy()
    rowfill_full = np.count_nonzero(cc_src, axis=1)
    cc_src[rowfill_full >= W * 0.85, :] = 0
    # 잔여는 이미지 전체 폭이 아니라 마디선이 오선과 겹치는 지점 '근처에만'
    # 국소적으로 남기도 한다(실측 확인: remove_staves가 마디선과 오선이 겹치는
    # 자리에서 유독 깔끔히 지우지 못함). 그래서 실제 오선 y좌표(각 보표 5개 줄)
    # 근처 행은 폭 비율과 무관하게 항상 지운다 — 이 행들은 원래 오선 자체이지
    # 음표 콘텐츠가 아니므로 지워도 안전하다.
    for yv in staves:
        yi = int(round(yv))
        cc_src[max(0, yi - 1):min(H, yi + 2), :] = 0
    _, cc_lab, cc_stats, _ = cv2.connectedComponentsWithStats(cc_src)

    for l in range(n_lines):
        top = int(round(staves[l * 5]))
        bot = int(round(staves[l * 5 + 4]))
        sh = bot - top
        if sh <= 0:
            continue
        band = image[max(0, top):bot + 1, :]

        # 1) 채움 후보 열 → 인접(딱 붙은 것)만 묶기. 이중선(틈 있는 굵은+얇은)은
        #    여기서 합치지 않고 아래 최종 병합 단계에서 하나로 통합한다.
        fill = np.count_nonzero(band, axis=0)
        cand = np.nonzero(fill >= sh * 0.88)[0]
        groups = []                                       # [시작 x, 폭]
        for cx in cand:
            if groups and cx - (groups[-1][0] + groups[-1][1]) <= 1:
                groups[-1][1] = int(cx) - groups[-1][0] + 1
            else:
                groups.append([int(cx), 1])

        kept = []
        for gx, gw in groups:
            if gw > w10:                                  # 2) 폭 필터
                continue
            xs = slice(gx, gx + gw)
            # 3) 돌출 필터(보표 밖으로 이어지는 기둥)
            up = image[max(0, top - w12):max(0, top - w4), xs]
            dn = image[min(H, bot + w4):min(H, bot + w12), xs]
            if (up.size and np.count_nonzero(up.any(axis=1)) >= up.shape[0] * 0.8) or \
               (dn.size and np.count_nonzero(dn.any(axis=1)) >= dn.shape[0] * 0.8):
                continue
            # 4) 고립 필터(마디선 '옆' 스트립의 행 커버리지) — 두 겹으로 본다.
            #    바로 붙은 스트립(0~w2)은 조금 붙어 있어도(마디선 근처의 짧은
            #    덧줄·꾸밈음 등) 넉넉히 봐주되(임계 0.4), 그보다 먼 스트립
            #    (w2~w4)은 조금만 채워져도(0.2) 음표로 본다 — 실측: 진짜
            #    마디선도 바로 옆엔 이따금 음표가 살짝 스치지만(0.25~0.35),
            #    그보다 멀리까지 닿는 일은 없다. 반대로 음표 기둥은 바로 옆
            #    스트립이 훨씬 크게 채워진다(0.4~0.85).
            near_lt = image[max(0, top):bot + 1, max(0, gx - w2):gx]
            near_rt = image[max(0, top):bot + 1, gx + gw:gx + gw + w2]
            far_lt = image[max(0, top):bot + 1, max(0, gx - w4):max(0, gx - w2)]
            far_rt = image[max(0, top):bot + 1, gx + gw + w2:gx + gw + w4]
            n_lt = np.count_nonzero(near_lt.any(axis=1)) / sh if near_lt.size else 0
            n_rt = np.count_nonzero(near_rt.any(axis=1)) / sh if near_rt.size else 0
            f_lt = np.count_nonzero(far_lt.any(axis=1)) / sh if far_lt.size else 0
            f_rt = np.count_nonzero(far_rt.any(axis=1)) / sh if far_rt.size else 0
            if n_lt >= 0.4 or n_rt >= 0.4 or f_lt >= 0.2 or f_rt >= 0.2:
                continue
            # 4-b) 부착 필터: 이 세로획이 속한 연결성분이 넓으면(음표 머리·빔이
            #      붙어 있으면) 마디선이 아니라 기둥이다. 오선 위/아래에 머리가
            #      있는 음표는 대역 안에서는 마디선과 구분되지 않으므로 필수.
            # 4-b) 부착 필터: 이 세로획이 속한 연결성분이 넓으면(음표 머리·빔이
            #      붙어 있으면) 마디선이 아니라 기둥이다. 오선 위/아래에 머리가
            #      있는 음표는 대역 안에서는 마디선과 구분되지 않으므로 필수.
            #      중앙 행 하나만 보면 그 음표머리가 중앙행을 벗어난 위치에
            #      있을 때(덧줄 음표 등) 놓친다 — 후보 열의 오선 범위 전체에서
            #      검사한다.
            attached = False
            for cxp in range(gx, gx + gw):
                col_labels = cc_lab[max(0, top):bot + 1, cxp]
                for li in np.unique(col_labels):
                    if li and int(cc_stats[li, cv2.CC_STAT_WIDTH]) > w10:
                        attached = True
                        break
                if attached:
                    break
            if attached:
                continue
            kept.append(gx)

        # 5) 이중 세로줄(도돌이 시작 굵은+얇은, 구간 구분) 병합 → 하나의 마디선
        merged = []
        for gx in sorted(kept):
            if merged and gx - merged[-1] <= w12:
                continue
            merged.append(gx)

        # 6) 굵은 세로선(도돌이 끝·종료선)도 마디선에 포함.
        #    단, 보표 왼쪽 헤더 영역(음자리표·조표·박자표)의 굵은 획은 제외한다.
        #    헤더는 첫 음표보다 왼쪽에 있으므로, 얇은 마디선이 하나라도 있으면
        #    그보다 왼쪽의 굵은 획은 헤더로 보고 건너뛴다(도돌이 시작은 4)에서
        #    이미 얇은선과 함께 잡히므로 손실이 없다).
        head_guard = min(kept) if kept else 0
        for gx, gw in groups:
            if gw < fs.weighted(4) or gw > w10:
                continue
            if gx < head_guard:                  # 헤더 영역의 굵은 획
                continue
            if any(abs(gx - m) <= w12 for m in merged):
                continue
            merged.append(gx)
        barlines[l] = sorted(merged)

    return barlines


def detect_barline_types(image, staves, barlines):
    """각 마디선의 '종류'를 판정한다: 일반 / 도돌이 시작 / 도돌이 끝 / 종료선.

    구조(사용자 관찰 및 픽셀 실측으로 확인):
      · 도돌이 시작 : [굵은 세로선][얇은 세로선][점 2개(콜론)]   (왼→오른)
      · 도돌이 끝   : [점 2개(콜론)][얇은 세로선][굵은 세로선]   (좌우 대칭)
      · 종료선      : [얇은 세로선][굵은 세로선]  (점 없음, 보통 곡 끝)
      · 일반        : 얇은 세로선 하나

    판정 방법:
      1. 보표 높이를 88% 이상 채우는 세로 구간을 모두 찾고, 폭이 weighted(4)
         이상이면 '굵은선', 그 미만이면 '얇은선'으로 분류한다.
      2. 굵은선 주변(±weighted(20))에서 얇은선과 '점 쌍'의 배치 순서를 본다.
         점 쌍 = 보표 가운데 두 칸(2번칸·3번칸)에 있는 작은 덩어리 2개.

    Returns
    -------
    dict[int, dict[int, str]]
        보표번호 → {마디선x: 종류}. 종류는
        "normal" | "repeat_start" | "repeat_end" | "final".
    """
    n_lines = len(staves) // 5
    H, W = image.shape
    result = {l: {} for l in range(n_lines)}

    for l in range(n_lines):
        top = int(round(staves[l * 5]))
        bot = int(round(staves[l * 5 + 4]))
        sh = bot - top
        if sh <= 0:
            continue
        band = image[max(0, top):bot + 1, :]
        fill = np.count_nonzero(band, axis=0)

        # 1) 세로선 구간(굵은/얇은) 수집
        segs = []            # (x, width)
        cx = 0
        while cx < W:
            if fill[cx] >= sh * 0.88:
                st = cx
                while cx < W and fill[cx] >= sh * 0.88:
                    cx += 1
                segs.append((st, cx - st))
            else:
                cx += 1
        thick = [(x, w) for x, w in segs if w >= fs.weighted(4)]

        def has_dots(x_from, x_to):
            """구간 안에 도돌이 점 2개(보표 가운데 두 칸)가 있는지."""
            x_from = max(0, x_from)
            x_to = min(W, x_to)
            if x_to <= x_from:
                return False
            sub = image[max(0, top):bot + 1, x_from:x_to]
            if sub.size == 0:
                return False
            n, _, st_, cent = cv2.connectedComponentsWithStats(sub)
            dots = 0
            for i in range(1, n):
                a = int(st_[i][4])
                bw, bh = int(st_[i][2]), int(st_[i][3])
                # 점: 작고 동그란 덩어리, 보표 세로 중앙부에 위치
                if (fs.weighted(2) <= bw <= fs.weighted(8) and
                        fs.weighted(2) <= bh <= fs.weighted(8) and
                        a >= fs.weighted(3)):
                    cy = cent[i][1]
                    if sh * 0.2 <= cy <= sh * 0.8:
                        dots += 1
            return dots >= 2

        near = fs.weighted(20)
        for bx in barlines.get(l, []):
            kind = "normal"
            # 이 마디선 주변의 굵은선 찾기
            tk = [(x, w) for x, w in thick if abs(x - bx) <= near or
                  (bx <= x <= bx + near)]
            if tk:
                tx, tw = tk[0]
                dots_right = has_dots(tx + tw + 1, tx + tw + 1 + near)
                dots_left = has_dots(tx - near, tx)
                if dots_right and not dots_left:
                    kind = "repeat_start"      # 굵은|얇은|:  → 시작
                elif dots_left and not dots_right:
                    kind = "repeat_end"        # :|얇은|굵은 → 끝
                elif not dots_left and not dots_right:
                    kind = "final"             # 얇은+굵은, 점 없음 → 종료선
            result[l][bx] = kind
    return result


def measure_validation(vis, staves, elements, barlines, time_sig):
    """마디선으로 마디를 나누고, 각 마디의 박자 합을 박자표 용량과 비교해 색칠한다.

    한 마디 용량(4분음표 단위) = 분자 × 4 / 분모 (예: 8/4 → 8).
    - 합 == 용량: 정상(색칠 없음)
    - 합 >  용량: 초과 → 빨강 배경
    - 합 <  용량: 부족 → 파랑 배경

    Returns
    -------
    tuple[np.ndarray, list]
        (마디 배경색이 칠해진 이미지, [(line, measure_no, total, capacity, status), ...]).
    """
    results = []
    if not time_sig or not time_sig[0] or not time_sig[1]:
        return vis, results
    capacity = time_sig[0] * 4.0 / time_sig[1]
    n_lines = len(staves) // 5

    for line in range(n_lines):
        bars = barlines.get(line, [])
        le = sorted([(e[1], e[5]) for e in elements if e[0] == line])
        if not le or not bars:
            continue
        left = le[0][0] - fs.weighted(10)          # 보표 첫 요소 앞
        bounds = [left] + bars                     # 마디 경계 x좌표들
        top = int(staves[line * 5]) - fs.weighted(15)
        bot = int(staves[line * 5 + 4]) + fs.weighted(15)

        for m in range(len(bounds) - 1):
            lo, hi = bounds[m], bounds[m + 1]
            in_measure = [b for x, b in le if lo <= x < hi]
            total = sum(beat_length(b) for b in in_measure)

            if abs(total - capacity) < 0.01:
                status, color = "OK", None
            elif total > capacity:
                status, color = "over", (0, 0, 255)      # 빨강(초과)
            else:
                status, color = "under", (255, 0, 0)     # 파랑(부족)

            if color is not None:                        # 반투명 배경색
                overlay = vis.copy()
                cv2.rectangle(overlay, (max(0, lo), max(0, top)), (hi, bot), color, -1)
                cv2.addWeighted(overlay, 0.22, vis, 0.78, 0, vis)
            results.append((line, m + 1, round(total, 3), capacity, status))

    return vis, results
