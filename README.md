# 콩나물지기 (OMR_Sprouty)

종이 악보(PDF·JPG·PNG 등)를 읽어 화면에서 원본과 겹쳐 보며 고치고, MusicXML·MIDI로 내보내는 악보 편집기입니다.
악보 인식에는 공개 프로그램 [Audiveris](https://github.com/Audiveris/audiveris)를 씁니다.

- 원본 그림 위에 인식 결과를 겹쳐 그려, 틀린 곳(빨간 표시)을 바로 찾아 고칠 수 있습니다.
- 붓 도구로 악상 기호·셈여림·페달 등을 그려 넣고, 기타 코드·가사를 넣을 수 있습니다.
- 조옮김, 연주 순서(도돌이표·D.C.·D.S.)를 따른 재생, MusicXML·MIDI 내보내기.

## 내려받기

**최신 버전 내려받기: https://github.com/zpischun-eng/OMR_Sprouty/releases/latest** — 이 주소는 늘 가장 새 버전을 보여 줍니다.
내 컴퓨터에 맞는 파일을 받습니다. 새 버전이 나오면 프로그램을 켤 때 알려 줍니다('정보' 창의 [새 버전 확인]으로도 확인).
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

서명 인증서가 없는 무료 프로그램이라, 맥이 처음 한 번은 여는 것을 막습니다. 아래 순서로 한 번만 풀어 주면 됩니다.
M1~M4 등 Apple 칩 맥은 **`_Mac_arm64.zip`**, 예전 인텔 맥은 `_Mac_x86_64.zip`을 받으세요.

1. 받은 zip을 두 번 눌러 풉니다. 폴더 안에 `OMRSprouty_vX_XX.app`, `audiveris` 폴더, `처음 한 번 실행.command`가 있습니다.
   폴더째로 원하는 곳(예: 응용 프로그램)에 옮겨도 되지만, 이 셋은 **같은 폴더에 함께** 있어야 합니다.
2. **처음 한 번만** `처음 한 번 실행.command`를 두 번 누릅니다. "열 수 없습니다"라는 창이 뜨면 **완료**를 누른 뒤,
   **시스템 설정 → 개인정보 보호 및 보안**을 열어 맨 아래 '보안' 칸의 **[그래도 열기]**를 누르고 암호를 넣습니다.
   (이 단추는 막힌 뒤 한 시간쯤만 보입니다. 안 보이면 `.command`를 다시 두 번 눌러 주세요.)
   터미널 창이 열렸다 "준비가 끝났습니다"가 나오면 된 것입니다 — 이 폴더의 '격리' 표시를 지워 앱과 악보 인식 엔진이 막히지 않게 합니다.
3. 이제 `OMRSprouty_vX_XX.app`을 두 번 눌러 엽니다.

> **더 간단한 방법(터미널)**: 터미널을 열고 `xattr -cr ` (끝에 빈칸)까지 친 다음, 푼 폴더를 터미널 창에 끌어다 놓고 Enter.
> 그 뒤 앱을 두 번 눌러 엽니다. 2번과 같은 일을 합니다.
>
> macOS 15(Sequoia)부터는 예전의 '오른쪽 클릭 → 열기'로는 열리지 않습니다 — 위 방법을 쓰세요.
> macOS 27(Golden Gate)·26·15의 Apple 칩 맥에서 앱 실행과 악보 인식이 되는 것은 자동 점검으로 확인했습니다
> (경고 창을 눌러 넘기는 과정 자체는 자동으로 확인할 수 없습니다). 이상한 점은 [Issues](https://github.com/zpischun-eng/OMR_Sprouty/issues)에 남겨 주세요.

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
공식 Audiveris를 받아 넣은 뒤 실제로 악보 하나를 인식해 보는 점검까지 합니다.

- **시험판**: 버전 번호(`APP_VERSION`)를 올려 올리면 [Releases 목록](https://github.com/zpischun-eng/OMR_Sprouty/releases)에
  시험판(pre-release)으로 올라갑니다. 시험판은 '최신 버전' 주소와 프로그램의 새 버전 알림에 나오지 않습니다.
- **정식판**: `STABLE_VERSION` 파일에 적은 버전만 정식(최신)으로 지정됩니다. 그 파일을 바꿔 올리면 됩니다.

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
