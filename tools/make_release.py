# tools/make_release.py — [v2.02] 배포 zip 만들기
# 리눅스 `zip` 명령은 한글 파일명에 UTF-8 표시를 켜지 않아 Windows에서 깨져 보였다.
# 파이썬 zipfile은 ASCII가 아닌 이름에 UTF-8 표시(플래그 0x800)를 자동으로 켠다.
#   python tools/make_release.py 2.02 [출력폴더]
import os
import sys
import zipfile

SKIP_DIRS = {"__pycache__", ".git"}


def make(src_root, version, out_dir):
    name = f"OMR_Sprouty_v{version}"
    out = os.path.join(out_dir, name + ".zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d, dirs, files in os.walk(src_root):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            for f in files:
                if f.endswith(".pyc"):
                    continue
                full = os.path.join(d, f)
                rel = os.path.relpath(full, src_root)
                z.write(full, os.path.join(name, rel).replace(os.sep, "/"))
    return out


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(make(root, sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "."))
