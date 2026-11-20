# functions.py
# ---------------------------------------------------------------------------
# 악보 인식에 사용되는 기본(저수준) 이미지 처리 함수 모음.
# 블로그 1단계(https://hackids.tistory.com/114)의 threshold 함수를 그대로 옮기고,
# 이후 단계에서 재사용할 수 있도록 약간의 유틸을 함께 정의한다.
# ---------------------------------------------------------------------------
import cv2
import numpy as np


def threshold(image):
    """이미지를 그레이스케일로 변환한 뒤 Otsu 이진화(반전)를 적용해 반환한다.

    OpenCV 계열 함수들은 '검은색 = 배경, 흰색 = 객체'로 인식하므로,
    THRESH_BINARY_INV 를 사용해 배경을 검은색, 오선/음표 등 객체를 흰색으로 만든다.

    Parameters
    ----------
    image : np.ndarray
        BGR 3채널 원본 이미지.

    Returns
    -------
    np.ndarray
        0/255 값을 갖는 1채널 이진 이미지.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(
        image, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    return image


def remove_frame(image, ratio=0.95):
    """이진 이미지에서 '완전히 꽉 찬 가로/세로 선'(프레임 테두리)을 제거한다.

    이 프로젝트의 샘플(score.jpg)은 프로그램 '창을 캡처'한 이미지라서
    창 상단/하단 테두리, 타이틀바 경계선, 좌우 테두리 같은 '전 폭/전 높이를
    가득 채운 직선'이 포함되어 있다. 이런 선은 뒤 단계의 오선 검출에서
    오선으로 오인되므로 미리 제거한다.

    실제 오선은 음자리표·조표 등이 앞부분을 차지해 전 폭을 100% 채우지 못하므로
    (본 샘플에서 최대 약 93%), ratio(기본 0.95) 임계값으로 안전하게 구분된다.

    블로그 원본 알고리즘에는 없는 '샘플 이미지 보정'용 보강 처리이며,
    스캔/사진 형태의 교가 이미지처럼 테두리가 없는 입력에는 아무 영향이 없다.

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지.
    ratio : float
        가로선은 이미지 너비, 세로선은 이미지 높이의 이 비율 이상을 채웠을 때
        '프레임'으로 간주한다.

    Returns
    -------
    np.ndarray
        프레임선이 제거된 이진 이미지.
    """
    image = image.copy()
    h, w = image.shape
    counts_r = np.count_nonzero(image == 255, axis=1)
    counts_c = np.count_nonzero(image == 255, axis=0)
    # 창 테두리는 이미지 '양쪽 끝'까지 이어지지만, 오선은 음자리표 여백 때문에
    # 가장자리에서 시작하지 않는다. 여백 없이 잘린 악보(오선이 너비의 95%를
    # 넘는 경우)에서 오선을 테두리로 오인하지 않도록, '이미지 양쪽 가장자리
    # 3px 안에 픽셀이 닿아 있는' 선만 프레임으로 판정한다.
    touch_lr = (image[:, :3].any(axis=1)) & (image[:, -3:].any(axis=1))
    touch_tb = (image[:3, :].any(axis=0)) & (image[-3:, :].any(axis=0))
    full_rows = (counts_r >= w * ratio) & touch_lr
    full_cols = (counts_c >= h * ratio) & touch_tb
    image[full_rows, :] = 0     # 꽉 찬 가로 프레임선 제거
    image[:, full_cols] = 0     # 꽉 찬 세로 프레임선 제거
    return image


# ===========================================================================
# 4단계(객체 검출)용 보조 함수 — 블로그 https://hackids.tistory.com/117
# ===========================================================================

# 정규화 시 사용한 오선 간격 기준값. normalization(standard=10)과 일치시켜야 한다.
STANDARD = 10


def weighted(value):
    """기준 오선 간격(STANDARD)에 맞춰 상숫값을 비례 조정한 정수를 돌려준다.

    정규화에서 오선 간격을 10이 아니라 20으로 잡으면 모든 구성요소가 2배 커진다.
    그런 경우에도 코드가 그대로 동작하도록, 코드 안의 상수(예: '오선 반 칸=5')를
    이 함수로 감싸 STANDARD 변화에 자동으로 따라가게 한다.
    STANDARD=10 이면 입력값을 그대로 반환한다. (예: weighted(5) → 5)
    """
    return int(value * (STANDARD / 10))


def closing(image):
    """닫힘 연산(팽창 후 침식)으로 끊어진 획을 잇고 객체 내부 구멍을 메운다.

    스캔·리사이즈 과정에서 한 기호가 미세하게 끊겨 여러 조각으로 검출되는 것을
    막기 위해 사용한다. 커널 크기는 오선 반 칸(weighted(5))으로 잡는다.
    ※ 구성요소 모양이 변형되므로, 검출(레이블링)에만 쓰고 박스는 원본에 그린다.
    """
    kernel = np.ones((weighted(5), weighted(5)), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image


def put_text(image, text, loc):
    """이미지의 loc 위치에 text 를 그린다(객체 크기 등 디버깅 표시용)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, str(text), loc, font, 0.6, (255, 0, 0), 2)


def get_center(y, h):
    """세로 범위 [y, y+h] 의 중앙 y좌표를 돌려준다(객체가 속한 보표 판별용)."""
    return (y + y + h) / 2


# ===========================================================================
# 5단계(객체 분석)용 — 직선 검출 / 음표 기둥 검출
#   블로그 https://hackids.tistory.com/118
# ===========================================================================

VERTICAL = True      # 세로 방향 탐색
HORIZONTAL = False   # 가로 방향 탐색


def get_line(image, axis, axis_value, start, end, length):
    """지정한 축을 따라 '끊기지 않는 직선'을 탐색해 (끝좌표, 길이)를 반환한다.

    axis=VERTICAL 이면 x=axis_value 로 고정하고 y를 start~end 로 이동하며
    세로 직선을 찾고, axis=HORIZONTAL 이면 y=axis_value 로 고정하고 x를 이동하며
    가로 직선을 찾는다. length(오선 칸 기준) 이상 끊기지 않고 이어지는 흰색 구간을
    직선으로 인정한다. 중간에 끊기면 그 지점 이후부터 다시 길이를 센다.

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지(객체=255).
    axis : bool
        VERTICAL(세로) 또는 HORIZONTAL(가로).
    axis_value : int
        고정할 축의 좌표값(세로면 x, 가로면 y).
    start, end : int
        탐색을 진행할 범위(세로면 y범위, 가로면 x범위).
    length : int
        직선으로 인정할 최소 길이(weighted 로 기준 오선 간격에 맞춰 보정됨).

    Returns
    -------
    tuple[int, int]
        (검출된 직선의 끝 좌표, 이어진 길이(픽셀 수)).
        조건을 만족하는 직선이 없으면 마지막 좌표와 0에 가까운 길이를 반환.
    """
    if axis:   # VERTICAL: (y, x) 에서 y가 움직임
        points = [(i, axis_value) for i in range(start, end)]
    else:      # HORIZONTAL: (y, x) 에서 x가 움직임
        points = [(axis_value, i) for i in range(start, end)]

    pixels = 0
    y = x = 0
    H, W = image.shape[:2]
    for i in range(len(points)):
        (y, x) = points[i]
        if not (0 <= y < H and 0 <= x < W):                 # 범위 밖 방어
            continue
        pixels += (image[y][x] == 255)                      # 흰색 픽셀 개수 누적
        # 다음 탐색 지점(이미지 경계 밖은 검은 픽셀로 간주 → 경계에서 선이 끊긴 것으로 처리)
        if axis:
            next_point = image[y + 1][x] if y + 1 < H else 0
        else:
            next_point = image[y][x + 1] if x + 1 < W else 0
        if next_point == 0 or i == len(points) - 1:         # 선이 끊기거나 끝에 도달
            if pixels >= weighted(length):
                break                                       # 원하는 길이의 직선 발견
            else:
                pixels = 0                                  # 끊겼으니 다시 셈
    return (y if axis else x), pixels


def stem_detection(image, stats, length):
    """객체 영역을 x좌표로 훑으며 음표 기둥(세로 직선)들을 검출한다.

    각 x좌표(col)에서 get_line 으로 세로 직선을 찾고, 인접한 열들은 같은 기둥으로
    묶어 [x, y, w, h] 로 저장한다(오선 검출과 같은 병합 구조).

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지.
    stats : tuple
        객체의 (x, y, w, h, area).
    length : int
        기둥으로 인정할 최소 세로 길이.

    Returns
    -------
    list[list[int]]
        검출된 기둥들의 [x, y, w, h] 리스트.
    """
    (x, y, w, h, area) = stats
    stems = []   # 기둥 정보 [x, y, w, h]
    for col in range(x, x + w):
        end, pixels = get_line(image, VERTICAL, col, y, y + h, length)
        if pixels:
            if len(stems) == 0 or abs(stems[-1][0] + stems[-1][2] - col) >= 1:
                stems.append([col, end - pixels + 1, 1, pixels])   # 새 기둥
            else:
                stems[-1][2] += 1                                   # 같은 기둥, 폭 확장
    return stems


def count_rect_pixels(image, rect):
    """사각형 구역(rect=(x, y, w, h)) 안의 흰색(255) 픽셀 개수를 센다.

    객체 stats 의 area 는 closing 이미지에서 얻은 값이라 부정확할 수 있으므로,
    음표 머리처럼 정밀한 판정이 필요한 곳에서는 원본 이진 이미지에서 직접 센다.
    (블로그의 이중 for 문을 numpy 로 벡터화 — 결과는 동일하고 경계도 안전하다.)

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지.
    rect : tuple[int, int, int, int]
        (x, y, w, h) 구역.

    Returns
    -------
    int
        구역 안 흰색 픽셀 수.
    """
    x, y, w, h = rect
    roi = image[max(0, y):y + h, max(0, x):x + w]   # 슬라이싱이 경계를 자동 클리핑
    return int(np.count_nonzero(roi == 255))


def count_pixels_part(image, area_top, area_bot, area_col):
    """한 열(area_col)을 area_top→area_bot 로 훑으며 '픽셀 덩어리'가 몇 번 나오는지 센다.

    음표 꼬리(깃발) 개수를 세는 데 쓴다. 흰 픽셀이 시작될 때마다 카운트를 1 올리고,
    검은 픽셀을 만나면 플래그를 내려 다음 덩어리를 다시 셀 준비를 한다.
    (예: 8분음표 꼬리 1개 → 1, 16분음표 꼬리 2개 → 2, 4분음표 → 0)

    Parameters
    ----------
    image : np.ndarray
        1채널 이진 이미지.
    area_top, area_bot : int
        탐색할 세로 범위.
    area_col : int
        탐색할 열(x좌표).

    Returns
    -------
    int
        해당 열에서 나타난 픽셀 덩어리의 개수.
    """
    H, W = image.shape
    area_col = max(0, min(area_col, W - 1))     # 열 경계 클리핑
    area_top = max(0, area_top)
    area_bot = min(H, area_bot)

    cnt = 0
    flag = False
    for row in range(area_top, area_bot):
        if not flag and image[row][area_col] == 255:
            flag = True
            cnt += 1
        elif flag and image[row][area_col] == 0:
            flag = False
    return cnt
