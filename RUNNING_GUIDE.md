# 실행 가이드 — Sprouty

이 문서는 프로젝트 코드를 **(A) 지금 바로 내 컴퓨터에서 실행해보는
방법**과 **(B) Windows용 배포 설치 프로그램(OMREditor_Setup.exe)을
직접 만드는 방법** 두 가지를 안내합니다.

> **[v3.30] 이제 직접 빌드하지 않아도 됩니다.** GitHub 저장소(https://github.com/zpischun-eng/OMR_Sprouty)에
> 코드를 올리면 자동 빌드가 Windows·맥 판을 만들고 Audiveris까지 넣어 줍니다.
> Actions 화면 → 해당 실행 → **Artifacts**에서 받거나, `v3.30` 같은 태그를 올려 Releases 페이지에서 받습니다.
> 맥 판의 첫 실행 방법은 README의 "맥" 절을 보세요. 아래 내용은 내 PC에서 직접 만들 때의 안내입니다.

---

## 0. 폴더 구조가 바뀌었습니다 (중요)

**Audiveris(악보 인식 엔진, 약 160MB)를 이제 프로젝트 코드와 분리했습니다.**
매 버전(Sprouty_01, 02, 03...)마다 큰 용량을 반복해서 받지 않도록,
Audiveris는 **한 번만 받아서 코드 폴더의 "형제 위치"에 두면** 이후
버전이 올라가도 계속 재사용됩니다.

**요구되는 폴더 구조** (반드시 이 모양을 지켜주세요):

```
OMREditor\                    ← 아무 이름이나 상관없음(부모 폴더)
├─ audiveris\                 ← 한 번만 받으면 계속 재사용
│  └─ Audiveris.exe
├─ Sprouty_01\                ← 이전 버전(있다면)
└─ Sprouty_02\                ← 이번에 받은 최신 코드
   ├─ editor_app_qt.py
   ├─ omr_bridge\
   └─ ...
```

즉 **`audiveris` 폴더와 `Sprouty_XX` 폴더가 같은 부모 폴더 아래
나란히** 있어야 합니다. 코드가 자동으로 "내 위치의 부모 폴더 아래
audiveris 폴더"를 찾아서 씁니다 — 별도 설정 필요 없습니다.

이미 이전 버전(`Sprouty_01`)을 받아서 그 안에 `audiveris` 폴더가
있으시다면, 그 `audiveris` 폴더를 **`Sprouty_01` 밖으로 꺼내
`Sprouty_01`과 같은 위치**로 옮기시면 됩니다(위 그림처럼). 그러면
새로 받는 `Sprouty_02`부터는 이 폴더를 그대로 재사용합니다.

혹시 `audiveris` 폴더가 아직 없으시면, 처음 한 번만 별도로 받으신
`audiveris.zip`을 풀어서 위 구조대로 놓아주세요.

---

## A. 지금 바로 실행해보기 (Windows)

### 1단계 — 파이썬 준비

```powershell
cd Sprouty_02
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2단계 — 바로 실행

**환경변수 설정 없이 바로 실행됩니다** (0번 항목의 폴더 구조만
맞으면 자동으로 Audiveris를 찾습니다).

```powershell
python editor_app_qt.py
```

프로그램이 뜨면 자동으로 예제 악보("작은 발걸음")가 불러와집니다.
처음 실행은 Audiveris가 워밍업하느라 수십 초~1분 정도 걸릴 수
있습니다. 이후 "악보 열기"로 원하는 악보 파일을 열어보세요.

> 사용법 자체는 `USER_MANUAL.md`를 참고하세요.

### (참고) 다른 위치에 Audiveris를 이미 설치하신 경우

`AUDIVERIS_BIN` 환경변수로 경로를 지정하면 그게 최우선으로 쓰입니다
(0번 항목의 자동 탐색보다 항상 우선).

```powershell
$env:AUDIVERIS_BIN = "C:\Program Files\Audiveris\Audiveris.exe"
```

### (참고) Windows가 아닌 환경에서 개발하실 경우

`audiveris/` 폴더 안의 실행 파일은 Windows 전용이라 Mac/Linux에서는
자동 인식되지 않습니다. https://github.com/Audiveris/audiveris/releases
에서 해당 OS용을 받아 설치한 뒤 `AUDIVERIS_BIN` 환경변수로 지정하세요.

---

## B. Windows 배포 설치 프로그램 만들기

다른 학교 선생님들께 배포할 준비가 되면, 아래 순서로 **더블클릭
한 번이면 설치되는 `OMREditor_Setup.exe`**를 만들 수 있습니다.
**반드시 Windows 컴퓨터에서** 진행해야 합니다. 이때도 0번 항목의
폴더 구조(`audiveris`와 `Sprouty_XX`가 형제 위치)를 그대로 따르면
됩니다 — `installer.iss`가 빌드 시점에 그 형제 폴더를 자동으로
가져와 설치 파일 하나로 합쳐줍니다(최종 사용자에게는 여전히
"설치 파일 하나"로 배포됩니다 — 이건 개발 단계에서만 분리해 둔
것입니다).

### 1단계 — 프로그램 본체 빌드

```powershell
cd Sprouty_02
pip install -r requirements.txt
pip install pyinstaller
pyinstaller OMREditor.spec
```

`dist\OMRSprouty_v3_21\` 폴더가 생깁니다(실행 파일 `OMRSprouty_v3_21.exe` — 이름의 버전은
`editor_app_qt.py`의 `APP_VERSION`에서 자동으로 붙습니다). `resource_dist/`(데모곡 1개만)만
포함되고, 개발용 `resource/` 폴더(회귀 테스트용 샘플)는 자동
제외됩니다.

### 2단계 — Inno Setup으로 설치 프로그램 만들기

1. https://jrsoftware.org/isinfo.php 에서 Inno Setup(무료)을 설치합니다.
2. `installer.iss`를 열고 "Build" → "Compile"을 누릅니다.
3. `Output\OMREditor_Setup.exe`가 생성됩니다 — **이 파일 하나가
   최종 배포물**입니다(설치되면 Audiveris도 그 안에 통째로 포함됨).

### 3단계 — 배포 전 마지막 확인

1. (v3.21 완료) `LICENSE_AGPLv3.txt`·`LICENSE_Audiveris_AGPLv3.txt`에 AGPLv3 전문이 들어 있습니다.
2. 소스 코드를 공개할 저장소(GitHub 등)를 만들었다면 `editor_app_qt.py`의 `AboutDialog.GITHUB_URL`에
   주소를 적고 다시 빌드하세요. 비워 두면 '정보' 창에 "배포한 사람에게 요청" 안내가 나옵니다.
3. 버전을 올릴 때는 `editor_app_qt.py`의 `APP_VERSION`과 `installer.iss`의 `MyAppVersion`을 함께 바꿉니다.
4. 다른 컴퓨터에서 설치 파일을 실제로 설치해 보고 정상 작동하는지 확인하는 것을 권장합니다.

### 설치 파일 없이 폴더째 보내는 경우

`dist\OMRSprouty_v3_21\` 폴더 **통째로**(`OMRSprouty_v3_21.exe` + `_internal\`) 보내고, 그 안에 `audiveris\` 폴더를 넣습니다.
실행 파일만으로는 실행되지 않습니다(실제 프로그램은 `_internal\` 안에 있음).

```
OMRSprouty_v3_21\
├─ OMRSprouty_v3_21.exe
├─ _internal\
└─ audiveris\Audiveris.exe
```

---

## 참고 — 개발자용 회귀 테스트

```bash
python tools/regression.py
```

4개 검증 샘플(선린교가·guard·butterfly·music)로 마디 인식 결과를
빠르게 확인합니다. 이 4개는 `resource/` 폴더에 이미 들어있습니다.

---

## 참고 — audiveris 폴더에 대해

- 출처: https://github.com/Audiveris/audiveris/releases (버전 5.11.0,
  Windows용 공식 배포 파일)
- 용량: 약 163MB(자체 JRE 포함)
- 라이선스: AGPLv3(수정 없이 그대로 사용)
- **이 버전부터 코드 압축 파일에는 포함하지 않습니다** — 최초 1회만
  별도로 받아 0번 항목의 구조대로 놓아두면, 이후 버전(Sprouty_03,
  04...)은 코드만 받아 교체하시면 됩니다.
