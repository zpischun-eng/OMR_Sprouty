; installer.iss — Windows 설치 프로그램 스크립트 (Inno Setup)
;
; [5단계] HYBRID_OMR_PLAN_v6.md 6.3절 확정 사항:
;   - 배포 산출물은 "설치 프로그램(exe) 하나"로 확정.
;   - PyInstaller로 만든 OMREditor 폴더 + Audiveris 설치 폴더를
;     같은 곳에 나란히 넣어 하나의 설치 프로그램으로 묶는다.
;   - 사용자는 OMREditor_Setup.exe 파일 하나만 받아서 실행하면 된다
;     (더블클릭 -> "다음, 다음, 완료" -> 시작메뉴/바탕화면 바로가기 생성).
;
; 빌드 순서(Windows에서 실행):
;   0) 폴더 구조 확인: 이 프로젝트 폴더(Sprouty_XX)와 audiveris\ 폴더가
;      같은 부모 폴더 아래 나란히 있어야 한다:
;        OMREditor\audiveris\Audiveris.exe
;        OMREditor\Sprouty_01\installer.iss  (이 파일)
;   1) pip install -r requirements.txt
;   2) pyinstaller OMREditor.spec           -> dist/OMRSprouty_v3_21/ 생성(버전마다 이름이 바뀜)
;   3) Inno Setup(무료, https://jrsoftware.org/isinfo.php)으로 이 파일을
;      컴파일 -> OMREditor_Setup.exe 생성
;      (Audiveris는 위 0)의 형제 폴더에서 자동으로 가져온다)
;
; 이 스크립트 자체는 개발 환경(Linux 샌드박스)에서 실제 컴파일까지는
; 검증하지 못했다(Inno Setup은 Windows 전용 도구) — 문법은 Inno Setup
; 공식 문서 형식을 따라 작성했으며, 실제 빌드 시 경로 등을 다시 확인해야
; 한다.

#define MyAppName "콩나물지기"
; [v3.21] 버전은 editor_app_qt.py의 APP_VERSION과 맞춘다
#define MyAppVersion "4.21"
#define MyAppPublisher "콩나물지기(OMR_Sprouty)"
; [v3.21] 실행 파일 이름 = OMRSprouty_v + 버전(점을 밑줄로) — OMREditor.spec과 같은 규칙
#define MyAppVerU StringChange(MyAppVersion, ".", "_")
; [v3.30] Audiveris 폴더 위치 — 직접 빌드할 땐 코드 폴더의 형제(..\audiveris), GitHub Actions는 /DAudiverisDir=로 넘긴다
#ifndef AudiverisDir
  #define AudiverisDir "..\audiveris"
#endif
#define MyAppBuild "OMRSprouty_v" + MyAppVerU
#define MyAppExeName MyAppBuild + ".exe"

[Setup]
AppId={{A1B2C3D4-E5F6-47A8-9B0C-1D2E3F4A5B6C}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\OMREditor
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; 비상업적 교육용 배포이므로 별도 서명 인증서 없이 진행 가능(사용자가
; Windows SmartScreen 경고를 볼 수 있음 — 6.3절에서 이미 감수하기로 함).
OutputBaseFilename=Kongnamuljigi_v{#MyAppVersion}_Setup
; [앱 아이콘] 콩나물지기(Sprouty) 로고 — 설치 프로그램 실행파일 자체의
; 아이콘과, 설치 마법사 좌측 상단 아이콘에 쓰인다.
SetupIconFile=src\icon\app_icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; 라이선스 동의 화면에 AGPLv3 전문을 보여준다(6.1절: 배포 파일에
; 라이선스 전문 포함 요건).
LicenseFile=LICENSE_AGPLv3.txt

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "바탕화면에 바로가기 만들기"; GroupDescription: "추가 아이콘:"

; [구조 변경] Audiveris(Windows용, 5.11.0)는 이제 프로젝트 폴더 "안"이
; 아니라 "형제 위치"(이 프로젝트 폴더의 부모 폴더 아래 audiveris\)에
; 있다 — 예: OMREditor\audiveris\, OMREditor\Sprouty_01\(이 프로젝트).
; 매 버전(Sprouty_01, 02, ...)마다 85MB짜리 Audiveris를 반복 압축해
; 전달하지 않기 위한 구조다. Inno Setup은 스크립트 파일 위치를 기준으로
; 상대경로를 해석하므로, "..\audiveris\*"로 한 단계 위를 가리킨다.

[Files]
; PyInstaller 결과물(폴더 전체)
Source: "dist\{#MyAppBuild}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Audiveris(프로젝트 폴더의 형제 위치에 있음, 빌드 시점에만 참조) —
; 설치 후에는 {app}\audiveris\ 에 위치해 omr_bridge.audiveris_runner의
; 자동 탐색 로직(exe_dir/audiveris/Audiveris.exe)과 정확히 일치한다.
Source: "{#AudiverisDir}\*"; DestDir: "{app}\audiveris"; Flags: ignoreversion recursesubdirs createallsubdirs
; 라이선스 파일들
Source: "LICENSE_AGPLv3.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE_Audiveris_AGPLv3.txt"; DestDir: "{app}\audiveris"; Flags: ignoreversion skipifsourcedoesntexist

[InstallDelete]
; [v3.21] 새 버전을 덮어 설치할 때 이전 버전 실행 파일이 남지 않게(이름에 버전이 들어가므로)
Type: files; Name: "{app}\OMRSprouty_v*.exe"
Type: files; Name: "{app}\OMREditor.exe"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; IconFilename: "{app}\{#MyAppExeName}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
