# tools/batch_verify.py — [v3.60] 폴더 일괄 검증(명령줄)
#   python tools/batch_verify.py 악보폴더 [--limit 20] [--out 결과폴더]
# 설치판에서는: OMRSprouty_vX_XX.exe --verify 악보폴더 [--limit 20] [--out 결과폴더]
# 결과: 결과폴더(기본 악보폴더/_sprouty_verify)의 index.html·summary.csv·보내기용 zip
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv):
    from omr_bridge import verify
    if not argv or argv[0].startswith("--"):
        print(__doc__ or "사용법: batch_verify.py 폴더 [--limit N] [--out 폴더]")
        return 2
    folder = argv[0]
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 0
    out = argv[argv.index("--out") + 1] if "--out" in argv else None
    ver = ""
    try:
        import re
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "editor_app_qt.py"), encoding="utf-8") as f:
            ver = re.search(r'^APP_VERSION\s*=\s*"([^"]+)"', f.read(), re.M).group(1)
    except Exception:
        pass

    def prog(i, n, name, stage):
        try:
            print(f"[{i + 1 if name else i}/{n}] {name} {stage}", flush=True)
        except Exception:
            pass
    s = verify.run_batch(folder, out, limit=limit, progress=prog, app_version=ver)
    print(f"끝: 파일 {s['files_done']}개(실패 {s['files_fail']}) · 원본에만 평균 {s['avg_miss_pct']}% · "
          f"편집 악보에만 평균 {s['avg_extra_pct']}% · 검산 통과 평균 {s['avg_measures_ok_pct']}%")
    print("보고서:", os.path.join(s["out_dir"], "index.html"))
    print("보내기용 zip:", s["zip"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
