# -*- mode: python ; coding: utf-8 -*-
# [5단계] 빌드:  pyinstaller OMREditor.spec
#
# 계획서 6.3절 확정 사항 반영:
# - onefile이 아니라 onedir(폴더) 모드로 빌드한다 — Audiveris(JRE 포함)를
#   같은 설치 폴더 아래 "audiveris/" 서브폴더로 나란히 둘 것이므로, 우리
#   앱도 처음부터 폴더 형태로 나와야 Inno Setup(installer.iss)이 그대로
#   묶어 하나의 설치 프로그램으로 만들 수 있다. "사용자가 받는 파일은
#   하나"라는 목표는 이 spec이 아니라 installer.iss가 만드는 설치
#   프로그램(OMREditor_Setup.exe)에서 달성된다.
# - omr_bridge/, LICENSE_AGPLv3.txt를 datas에 포함.

# [v3.21] 실행 파일·폴더 이름에 버전을 붙인다: OMRSprouty_v3_21.exe, dist/OMRSprouty_v3_21/
# 버전은 editor_app_qt.py의 APP_VERSION에서 읽는다(버전을 올릴 때 이 파일은 고치지 않는다).
import re as _re
with open('editor_app_qt.py', encoding='utf-8') as _f:
    _ver = _re.search(r'^APP_VERSION\s*=\s*"([^"]+)"', _f.read(), _re.M).group(1)
APP_NAME = 'OMRSprouty_v' + _ver.replace('.', '_')
# [v3.30] 맥: OMRSprouty_v3_30.app 으로 묶는다(Windows·리눅스는 예전처럼 폴더 + 실행 파일)
import sys as _sys
IS_MAC = _sys.platform == 'darwin'
ICON = 'src/icon/app_icon.icns' if IS_MAC else 'src/icon/app_icon.ico'

a = Analysis(
    ['editor_app_qt.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('resource_dist', 'resource_dist'),
        ('src/fonts', 'src/fonts'),
        ('src/icon', 'src/icon'),
        ('omr_bridge', 'omr_bridge'),
        ('score', 'score'),
        ('omr', 'omr'),
        ('LICENSE_AGPLv3.txt', '.'),
    ],
    hiddenimports=['omr_bridge.audiveris_runner', 'omr_bridge.geometry',
                   'omr_bridge.mxl_reader', 'omr_bridge.merge',
                   'omr_bridge.document_adapter'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter',           # 구 GUI 제거로 더 이상 불필요
              'pkg_resources'],    # [v3.21] 쓰지 않는데 끌려 들어와, 오래된 setuptools가 있는 PC에선 시작 시 오류
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,          # onedir 모드 — binaries는 COLLECT에서
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=not IS_MAC,                 # 맥은 UPX 압축이 서명을 깨뜨린다
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON,                      # [앱 아이콘] 콩나물지기 로고(Windows .ico / 맥 .icns)
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=not IS_MAC,
    upx_exclude=[],
    name=APP_NAME,
)

if IS_MAC:
    app = BUNDLE(
        coll,
        name=APP_NAME + '.app',
        icon=ICON,
        bundle_identifier='io.github.zpischun-eng.omrsprouty',
        info_plist={
            'CFBundleName': '콩나물지기',
            'CFBundleDisplayName': '콩나물지기',
            'CFBundleShortVersionString': _ver,
            'CFBundleVersion': _ver,
            'NSHighResolutionCapable': True,          # 레티나 화면에서 선명하게
            'LSMinimumSystemVersion': '11.0',
            # 악보 파일을 앱 아이콘에 끌어다 놓기·'다음으로 열기'
            'CFBundleDocumentTypes': [{
                'CFBundleTypeName': '악보',
                'CFBundleTypeRole': 'Viewer',
                'LSItemContentTypes': ['com.adobe.pdf', 'public.jpeg', 'public.png', 'public.tiff',
                                       'com.microsoft.bmp'],
            }],
        },
    )

