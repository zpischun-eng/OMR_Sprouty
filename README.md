# 콩나물지기 (OMR_Sprouty)

종이 악보(PDF·JPG·PNG 등)를 읽어 화면에서 원본과 겹쳐 보며 고치고, MusicXML·MIDI로 내보내는 악보 편집기입니다.
악보 인식에는 공개 프로그램 [Audiveris](https://github.com/Audiveris/audiveris)를 씁니다.

- 원본 그림 위에 인식 결과를 겹쳐 그려, 틀린 곳(빨간 표시)을 바로 찾아 고칠 수 있습니다.
- 붓 도구로 악상 기호·셈여림·페달 등을 그려 넣고, 기타 코드·가사를 넣을 수 있습니다.
- 조옮김, 연주 순서(도돌이표·D.C.·D.S.)를 따른 재생, MusicXML·MIDI 내보내기.

## 내려받기

[Releases(내려받기)](https://github.com/zpischun-eng/OMR_Sprouty/releases) 페이지에서 내 컴퓨터에 맞는 파일을 받습니다.
악보 인식 엔진(Audiveris 5.11.0)이 함께 들어 있어 따로 설치할 것이 없습니다.

| 컴퓨터 | 받을 파일 |
|---|---|
| Windows (설치형) | `Kongnamuljigi_vX.XX_Setup.exe` |
| Windows (폴더형, 설치 없이) | `OMRSprouty_vX_XX_Windows.zip` |
| 맥 (M1 이후, Apple 칩) | `OMRSprouty_vX_XX_Mac_arm64.zip` |
| 맥 (예전 인텔 칩) | `OMRSprouty_vX_XX_Mac_x86_64.zip` |

### Windows

- **설치형**: 받은 Setup 파일을 실행합니다. "Windows의 PC 보호" 창이 뜨면 **추가 정보 → 실행**을 누릅니다
  (서명 인증서가 없는 무료 프로그램이라 뜨는 안내입니다).
- **폴더형**: 압축을 푼 폴더 안의 `OMRSprouty_vX_XX.exe`를 실행합니다.
  같은 폴더의 `_internal`과 `audiveris` 폴더는 지우거나 옮기지 말고 그대로 두세요.

### 맥

1. 받은 zip을 두 번 눌러 풉니다. 폴더 안에 `OMRSprouty_vX_XX.app`, `audiveris` 폴더, `처음 한 번 실행.command`가 있습니다.
   폴더째로 원하는 곳(예: 응용 프로그램)에 옮겨도 되지만, 이 셋은 **같은 폴더에 함께** 있어야 합니다.
2. **처음 한 번만**: `처음 한 번 실행.command`를 **오른쪽 클릭 → 열기 → 열기**.
   인터넷에서 받은 앱에 붙는 '격리' 표시를 이 폴더에서만 지워, 콩나물지기와 악보 인식 엔진이 막히지 않게 합니다.
3. 이제 `OMRSprouty_vX_XX.app`을 두 번 눌러 엽니다.

> 2번이 막히면 터미널에서 `xattr -dr com.apple.quarantine ` 뒤에 폴더를 끌어다 놓고 Enter를 눌러도 같습니다.
> 맥 버전은 아직 실제 맥에서 충분히 써 보지 못했습니다. 이상한 점은 [Issues](https://github.com/zpischun-eng/OMR_Sprouty/issues)에 남겨 주세요.

## 직접 실행·빌드하기 (개발용)

```
pip install -r requirements.txt
python editor_app_qt.py
```

Audiveris는 저장소에 들어 있지 않습니다. [Audiveris 5.11.0](https://github.com/Audiveris/audiveris/releases/tag/5.11.0)을 받아
아래 중 한 곳에 두거나, 환경 변수 `AUDIVERIS_BIN`에 실행 파일 경로를 적습니다.

- 프로그램 폴더와 나란히: `상위폴더/audiveris/Audiveris.exe` (맥: `상위폴더/audiveris/Audiveris.app`)
- Windows 기본 설치 위치(`Program Files\Audiveris`), 맥 응용 프로그램 폴더(`/Applications/Audiveris.app`)

실행 파일 만들기:

```
pip install pyinstaller
pyinstaller OMREditor.spec
```

- Windows: `dist/OMRSprouty_vX_XX/` (exe + `_internal`) — 여기에 `audiveris` 폴더를 넣어 함께 배포합니다.
- 맥: `dist/OMRSprouty_vX_XX.app` — 옆에 `audiveris/Audiveris.app`을 둡니다.
- Windows 설치 프로그램: Inno Setup 6으로 `installer.iss`를 컴파일합니다
  (Audiveris 폴더 위치는 `/DAudiverisDir=경로`로 바꿀 수 있음).

### 자동 빌드 (GitHub Actions)

`main`에 올리면 [Actions](https://github.com/zpischun-eng/OMR_Sprouty/actions)가 Windows·맥(Apple 칩·인텔) 판을 만들고,
공식 Audiveris를 받아 넣은 뒤 실제로 악보 하나를 인식해 보는 점검까지 합니다. 결과는 각 실행의 **Artifacts**에서 받습니다.
`v3.30`처럼 `v`로 시작하는 태그를 올리면 같은 파일로 Releases 페이지가 만들어집니다.

```
git tag v3.30
git push origin v3.30
```

자세한 실행·배포 안내는 [RUNNING_GUIDE.md](RUNNING_GUIDE.md), 사용법은 [USER_MANUAL.md](USER_MANUAL.md),
개발 기록은 [docs/OMR_Sprouty_개발총괄.md](docs/OMR_Sprouty_개발총괄.md)에 있습니다.

## 구조

```
editor_app_qt.py   화면(PySide6) — 편집기 진입점
omr_bridge/        Audiveris 실행·결과 읽기·원본 대조(붓 기호, 코드 글자 인식, 캐시)
score/             악보 데이터 — 모델·편집·배치·그리기·MusicXML·MIDI·코드·가사·연주 순서
omr/               PDF·그림 읽기, 예전 자체 인식 엔진(보존용)
tools/             개발용 점검 도구(회귀 시험, MusicXML 검사, 자동 빌드 점검)
resource/synthetic 시험용 합성 악보(저작권 없음)
```

## 라이선스

콩나물지기는 [GNU AGPL v3](LICENSE)로 공개합니다. 함께 배포하는 Audiveris도 AGPL v3이며
([LICENSE_Audiveris_AGPLv3.txt](LICENSE_Audiveris_AGPLv3.txt)), 원본 소스는
https://github.com/Audiveris/audiveris 에 있습니다. 악보 글꼴 Bravura는 SIL Open Font License를 따릅니다.
