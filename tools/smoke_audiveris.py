# tools/smoke_audiveris.py
# ---------------------------------------------------------------------------
# [v3.30] 자동 빌드(GitHub Actions)용 짧은 점검 — 함께 넣을 Audiveris가 이 컴퓨터(OS)에서
# 실제로 악보를 읽는지 본다. 캐시를 쓰지 않고 Audiveris를 직접 돌린다.
#   python tools/smoke_audiveris.py 악보그림.png [시간제한초]
#   (AUDIVERIS_BIN 환경변수로 쓸 실행 파일을 정할 수 있다)
# 성공: 종료 코드 0 + 만들어진 .mxl 경로 출력. 실패: 종료 코드 1 + 오류·로그 끝부분.
# ---------------------------------------------------------------------------
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omr_bridge import audiveris_runner as ar  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(__doc__ or "사용법: smoke_audiveris.py 그림 [초]")
        return 2
    src = sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    print("Audiveris:", ar.find_audiveris_bin())
    out = tempfile.mkdtemp(prefix="sprouty_smoke_")
    res = ar.run_single(src, out, timeout_sec=timeout)
    print("status:", res.status, "| mxl:", res.mxl_path, "| upscale:", res.upscale_factor)
    if res.status != "ok" or not res.mxl_path or not os.path.exists(res.mxl_path):
        print("error:", res.error)
        if res.log_path and os.path.exists(res.log_path):
            with open(res.log_path, encoding="utf-8", errors="replace") as f:
                print("---- log (끝 60줄) ----")
                print("".join(f.readlines()[-60:]))
        return 1
    with zipfile.ZipFile(res.mxl_path) as z:
        xml = [n for n in z.namelist() if n.endswith(".xml") and not n.startswith("META-INF")]
        body = z.read(xml[0]).decode("utf-8", "replace") if xml else ""
    n_notes = body.count("<note")
    n_meas = body.count("<measure")
    print(f"마디 {n_meas}개, 음표 {n_notes}개")
    return 0 if n_notes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
