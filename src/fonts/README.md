# 음악 폰트(SMuFL) 설치

편집 악보를 전문 악보 기호로 깨끗하게 표시하려면 이 폴더에 SMuFL 표준
음악 폰트를 넣으세요. **폰트가 없어도 프로그램은 정상 동작**하며, 기존
벡터 방식으로 자동 폴백합니다.

## 권장: Bravura (무료, 상업 배포 가능 — SIL Open Font License)

1. https://github.com/steinbergmedia/bravura 접속
2. `redist/otf/Bravura.otf` 다운로드
   (또는 릴리스 zip을 받아 그 안의 `Bravura.otf`)
3. 이 폴더(`src/fonts/`)에 `Bravura.otf` 로 저장

프로그램을 다시 실행하면 자동으로 인식됩니다.

## 대안 폰트

같은 폴더에 아래 파일명 중 하나가 있으면 자동 인식합니다(우선순위 순):
`Bravura.otf` → `Bravura.ttf` → `Leland.otf` → `Petaluma.otf`

- Leland: MuseScore 4 기본 폰트 (OFL)
- Petaluma: 손글씨 느낌의 SMuFL 폰트 (OFL)

## 라이선스 주의

이 폰트들은 SIL Open Font License로 배포되어 프로그램에 포함해
재배포할 수 있습니다. 배포 패키지에 각 폰트의 `OFL.txt`(라이선스 전문)를
함께 넣어 두는 것을 권장합니다.

## NanumGothic-Regular.ttf
악보 제목·작사작곡·빠르기말 등 한글 텍스트를 편집 악보에 그리기 위한 글꼴(DejaVu Sans에는
한글 글리프가 없다). SIL Open Font License 1.1 — 무료 재배포 가능(NAVER).
