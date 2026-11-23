# omr_bridge/audiveris_runner.py
# ---------------------------------------------------------------------------
# [0단계/1단계] Audiveris를 서브프로세스로 실행해 .mxl(MusicXML)과
# .omr(내부 프로젝트 파일, 원본 좌표 포함)을 얻는다.
#
# 설계 근거: HYBRID_OMR_PLAN_v6.md 5장(Audiveris 연동 방식).
# - 서브프로세스 분리 실행(라이선스 리스크 최소화, 5.2절 근거와 동일)
# - 저해상도 이미지는 자동 업스케일(계획서 5장 "필요 작업" 항목)
# ---------------------------------------------------------------------------
import subprocess
import os
import sys
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

try:
    from PIL import Image
except ImportError:
    Image = None


class AudiverisNotFound(Exception):
    """[v1.21] Audiveris 실행 파일을 찾지 못함 — 사용자에게 그대로 보여줄 안내 문구를 담는다."""


def _audiveris_candidates():
    """Audiveris 실행 파일을 찾아볼 자리(우선순위 순).

    1. 환경변수 AUDIVERIS_BIN(명시적으로 지정했으면 최우선)
    2. 배포 설치 폴더: <설치폴더>/audiveris/Audiveris.exe (installer.iss가 만드는 구조)
    3. 개발 폴더: 프로그램 폴더의 '형제' audiveris/ (예: OMR_Sprouty/audiveris/ 와
       OMR_Sprouty/OMR_Sprouty_v1.21/) — 버전마다 다시 받지 않고 한 번만 둔다
    4. 프로그램 폴더 '안'의 audiveris/ (예전 방식)
    5. [v1.21] 한 단계 위 작업 폴더의 다른 폴더들에 있는 audiveris/
       (예: pythonWorkspaces/OMREditor/audiveris/ — 작업 폴더 이름을 바꿔 옮겼을 때,
       실사용: OMREditor → OMR_Sprouty로 옮긴 뒤 "지정된 파일을 찾을 수 없습니다")
    6. [v1.21] Audiveris 공식 설치 프로그램의 기본 위치(Program Files 등)
    """
    exe_dir = Path(sys.executable).resolve().parent
    project_root = Path(__file__).resolve().parent.parent
    parent = project_root.parent
    names = ("Audiveris.exe", "bin/Audiveris.exe", "bin/Audiveris", "Audiveris",
             "Audiveris.app/Contents/MacOS/Audiveris")          # [v3.30] 맥 앱 안의 실행 파일
    out = []

    def add_dir(d):
        for n in names:
            out.append(d / n)
    add_dir(exe_dir / "audiveris")
    # [v3.30] 맥 배포: OMRSprouty_vX.app/Contents/MacOS/(실행 파일) — 앱 '옆'의 audiveris 폴더와 앱 안 Resources
    if exe_dir.name == "MacOS" and exe_dir.parent.name == "Contents":
        add_dir(exe_dir.parents[2] / "audiveris")
        add_dir(exe_dir.parent / "Resources" / "audiveris")
    add_dir(parent / "audiveris")
    add_dir(project_root / "audiveris")
    try:
        for sib in sorted(parent.parent.iterdir()):
            if sib.is_dir() and sib != parent:
                add_dir(sib / "audiveris")
    except OSError:
        pass
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
                 os.environ.get("LOCALAPPDATA") and os.path.join(os.environ["LOCALAPPDATA"], "Programs")):
        if base:
            add_dir(Path(base) / "Audiveris")
    add_dir(Path("/opt/audiveris"))
    # [v3.30] 맥: Audiveris 공식 설치(응용 프로그램 폴더)
    add_dir(Path("/Applications"))
    add_dir(Path.home() / "Applications")
    return out


def find_audiveris_bin():
    """찾은 실행 파일 경로를 돌려준다. 못 찾으면 AudiverisNotFound(안내 문구 포함)."""
    env_val = os.environ.get("AUDIVERIS_BIN")
    if env_val:
        if Path(env_val).exists():
            return env_val
        raise AudiverisNotFound(
            "환경 변수 AUDIVERIS_BIN에 적힌 Audiveris 경로에 파일이 없습니다.\n\n"
            f"  {env_val}\n\n"
            "경로를 고치거나 환경 변수를 지우고 다시 실행해 주세요.")
    cands = _audiveris_candidates()
    for c in cands:
        if c.is_file():
            return str(c)
    project_root = Path(__file__).resolve().parent.parent
    exe_dir = Path(sys.executable).resolve().parent
    frozen = getattr(sys, "frozen", False)
    where = (exe_dir / "audiveris") if frozen else (project_root.parent / "audiveris")
    mac = sys.platform == "darwin"
    if mac and frozen and exe_dir.name == "MacOS":
        where = exe_dir.parents[2] / "audiveris"
    target = (where / "Audiveris.app") if mac else (where / "Audiveris.exe")
    raise AudiverisNotFound(
        "악보 인식 엔진(Audiveris)을 찾을 수 없습니다.\n\n"
        + ("Audiveris.app을 아래 위치(또는 응용 프로그램 폴더)에 두고 다시 열어 주세요.\n\n" if mac else
           "Audiveris 폴더(안에 Audiveris.exe가 있는 폴더)를 아래 위치에 두고 다시 열어 주세요.\n\n")
        + f"  {target}\n\n"
        + ("" if frozen else
           f"(프로그램 폴더 {project_root.name} 와 나란히 두면 다음 버전에서도 다시 옮길 필요가 없습니다.)\n\n")
        + "또는 환경 변수 AUDIVERIS_BIN에 Audiveris.exe의 전체 경로를 적어도 됩니다.")


# 하위 호환용 — 실제 실행은 매번 find_audiveris_bin()으로 다시 찾는다(프로그램을 켠 채로
# 폴더를 옮겨 놓아도 바로 인식되게).
try:
    AUDIVERIS_BIN = find_audiveris_bin()
except AudiverisNotFound:
    AUDIVERIS_BIN = None

# [실사용 검증 후 재설계] 예전엔 MIN_INTERLINE_PX(최소 오선 간격 임계값)로
# 업스케일 필요 여부를 "사전에" 추정했으나, 이 추정이 부정확해 이미
# 충분히 고해상도인 이미지까지 잘못 업스케일하는 사고가 있었다(실사용
# 중 발견). 지금은 run_single()이 "원본으로 먼저 시도 → 실패하면 그때
# 업스케일 재시도"하는 방식으로 바뀌어 이 임계값 자체가 필요 없다.
#
# [조사 기록 — 2단계, 선린교가 기준 실측] 업스케일 배율 자체는 4배가
# 지금까지 테스트한 것 중 최선이었다:
#   4배: 32마디 중 24개 정합(기준)
#   5배: 32마디 중 20개 정합(더 나쁨 — Audiveris 로그에 "신뢰할 만한 빔
#        두께를 못 찾음" 등 경고 발생, 너무 큰 이미지에서 내부 이진화·
#        형태 추정이 오히려 불안정해지는 것으로 추정)
#   6배: 처리 시간이 과도하게 길어져 실용성 없음(250초 내 미완료)
# Audiveris는 이미지 크기(org.audiveris.omr.step.LoadStep.maxPixelCount)와
# 스텝별 처리 시간(org.audiveris.omr.Main.sheetStepTimeOut)에 내부 상한이
# 있으며 -constant 옵션으로 조정 가능함을 확인했으나, 배율 자체를 올리는
# 것만으로는 부점(augmentation dot) 인식이 개선되지 않았다.
# 다음으로 시도해볼 만한 방향(미착수, 후속 과제로 기록):
#   - waifu2x 등 AI 기반 초해상도 전처리(Audiveris 공식 문서 권장 방법 —
#     단순 보간(Lanczos)이 아니라 AI가 세부를 "복원"하는 방식이라 부점처럼
#     작은 디테일 인식에 도움이 될 가능성)
#   - org.audiveris.omr.sig.relation.AugmentationRelation.xOutGapMax 등
#     부점-음표 연결 판정 임계값을 직접 조정(단, 기본값 불명 + 다른
#     마디에서 오탐이 늘어날 위험 있어 신중한 접근 필요)


@dataclass
class AudiverisResult:
    status: str                 # "ok" | "error"
    mxl_path: Optional[str] = None
    omr_path: Optional[str] = None
    upscale_factor: float = 1.0  # 원본 이미지 대비 Audiveris에 실제로 넣은 배율
    error: Optional[str] = None
    log_path: Optional[str] = None


def _retry_scaled(image_path: str, work_dir: Path):
    """[v3.61] 첫 시도가 실패했을 때 다시 넣을 그림과 배율 (경로, 배율).

    예전에는 무조건 4배로 키웠다. 그런데 PDF에서 400dpi로 그린 그림(약 1천만 화소)은 이미 충분히 큰데도
    4배(1억 7천만 화소)로 키우다가 메모리가 모자라 죽거나(MemoryError), 키운 그림을 Audiveris가 감당하지
    못했다(실사용: 가톨릭 성가 20곡 중 14곡 실패).
      · 큰 그림(600만 화소 이상) → 해상도 부족이 아니다. 메모리 부담을 줄이려 0.75배로 줄여 다시 시도.
      · 작은 그림 → 키우되 결과가 4천만 화소를 넘지 않게(최대 4배). 1.25배도 못 키우면 그대로 다시.
    """
    import math
    with Image.open(image_path) as im:
        w, h = im.size
        px = float(w * h)
        if px >= 6e6:
            factor = 0.75
        else:
            factor = min(4.0, math.sqrt(40e6 / max(1.0, px)))
            if factor < 1.25:
                return image_path, 1.0
            factor = float(int(factor * 4) / 4.0)            # 0.25 단위(좌표 환산 오차를 줄인다)
        nw, nh = int(round(w * factor)), int(round(h * factor))
        im2 = im.resize((nw, nh), Image.LANCZOS)
        out_path = work_dir / f"{'upscaled' if factor > 1 else 'scaled'}_{Path(image_path).stem}.png"
        im2.save(out_path)
    return str(out_path), nw / float(w)


def _total_ram_mb():
    """이 컴퓨터의 전체 메모리(MB). 알 수 없으면 None."""
    try:
        if sys.platform.startswith("win"):
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return int(ms.ullTotalPhys // (1024 * 1024)), int(ms.ullAvailPhys // (1024 * 1024))
            return None
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout
            return int(out.strip()) // (1024 * 1024), None
        tot = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 * 1024)
        av = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") // (1024 * 1024)
        return int(tot), int(av)
    except Exception:
        return None


def jvm_env(env: dict) -> dict:
    """[v3.61] Audiveris(자바)가 쓸 수 있는 메모리 한도를 넉넉히 준다. 자바의 기본 한도는 전체 메모리의 1/4이라
    메모리가 적은 PC(4~8GB)에서는 큰 악보 그림을 처리하다 멈출 수 있다 → 전체의 절반(1~6GB 사이).
    사용자가 JAVA_TOOL_OPTIONS를 직접 정해 두었으면 건드리지 않는다."""
    if env.get("JAVA_TOOL_OPTIONS"):
        return env
    ram = _total_ram_mb()
    if ram and ram[0]:
        mx = int(max(1024, min(6144, ram[0] * 0.5)))
        env["JAVA_TOOL_OPTIONS"] = f"-Xmx{mx}m"
    return env


def _log_note(log_path, text):
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n" + text + "\n")
    except OSError:
        pass


def log_tail(log_path, n_lines=40, max_chars=3000):
    """Audiveris 기록의 끝부분(실패 원인을 알리는 데 쓴다)."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return ""
    # [v3.71] 끝부분만으로는 "Error in export"밖에 안 보인다 — 앞쪽의 첫 경고·예외 줄들을 함께 낸다
    head = [ln for ln in lines[:-n_lines] if ("WARN" in ln or "ERROR" in ln or "Exception" in ln)
            and "Error in export" not in ln][:12]
    tail = "".join(lines[-n_lines:])[-max_chars:]
    return ("[앞쪽 경고]\n" + "".join(head)[:1500] + "[끝부분]\n" if head else "") + tail


def _run_audiveris_once(src: str, out_dir: Path, timeout_sec: int, tag: str = ""):
    """Audiveris를 한 번 실행하고 (성공여부, mxl경로, omr경로, log경로)를 반환.

    실제 Audiveris 실행 자체만 담당하는 내부 헬퍼 — 재시도 로직은
    run_single()이 담당한다.
    """
    env = os.environ.copy()
    env["LANG"] = "C.utf8"
    env["LC_ALL"] = "C.utf8"
    jvm_env(env)

    stem = Path(src).stem
    log_path = out_dir / f"{stem}{tag}.audiveris.log"

    from omr_bridge import ocr_data
    cmd = [find_audiveris_bin(), "-batch"] + ocr_data.apply(env, []) + \
        ["-export", "-output", str(out_dir), "--", src]           # [v3.20] 글자 인식(데이터 있을 때)
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            cp = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                timeout=timeout_sec, env=env)
        _log_note(log_path, f"[sprouty] exit code {cp.returncode} · JAVA_TOOL_OPTIONS={env.get('JAVA_TOOL_OPTIONS')} "
                            f"· RAM(total, free MB)={_total_ram_mb()}")
    except subprocess.TimeoutExpired:
        _log_note(log_path, f"[sprouty] timeout {timeout_sec}s · RAM(total, free MB)={_total_ram_mb()}")
        return False, None, None, str(log_path)
    except OSError as ex:                       # 파일은 있으나 실행할 수 없음(손상·권한 등)
        raise AudiverisNotFound(
            f"Audiveris를 실행하지 못했습니다.\n\n  {cmd[0]}\n\n오류: {ex}\n\n"
            "Audiveris 폴더가 온전한지(압축을 끝까지 풀었는지) 확인해 주세요.") from ex

    mxl_path = out_dir / f"{stem}.mxl"
    omr_path = out_dir / f"{stem}.omr"
    ok = mxl_path.exists() and mxl_path.stat().st_size > 0
    return ok, (str(mxl_path) if ok else None), \
        (str(omr_path) if omr_path.exists() else None), str(log_path)


def run_single(source_path: str, output_dir: str, timeout_sec: int = 180) -> AudiverisResult:
    """PDF 또는 이미지 파일 하나를 Audiveris로 처리한다.

    [실사용 검증 후 재설계] 처음에는 "이미지 높이 대비 비율로 오선 간격을
    추정해 미리 업스케일 여부를 판단"하는 방식이었으나, 이 추정이 매우
    거칠어서(이미지 안에 시스템이 몇 줄인지에 따라 실제 오선 간격이
    크게 달라지므로) **이미 충분히 고해상도인 이미지까지 불필요하게
    업스케일해버려 오히려 인식이 실패하는 사고**가 실사용 중 발견됐다
    (이전 조사에서도 "업스케일을 과하게 하면 오히려 정확도가 떨어진다"는
    것을 이미 확인한 바 있음 — INVESTIGATION_LOG_upscale.md 참고).

    이제는 **"원본 그대로 먼저 시도 → 실패하면 그때 업스케일해서 재시도"**
    방식으로 바꿨다. 사전 추측보다 훨씬 정확하다(실패 여부는 Audiveris
    자신이 가장 정확히 판단하므로).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_work"
    work_dir.mkdir(exist_ok=True)

    # 1차 시도: 원본 그대로(PDF는 항상, 이미지도 우선 원본으로)
    ok, mxl_path, omr_path, log_path = _run_audiveris_once(
        source_path, out_dir, timeout_sec)
    if ok:
        return AudiverisResult(
            status="ok", mxl_path=mxl_path, omr_path=omr_path,
            upscale_factor=1.0, log_path=log_path,
        )

    # 2차 시도: 실패했고 이미지 파일이면(PDF는 업스케일 개념이 없음)
    # 업스케일해서 재시도. 로그에 "too low interline"이 찍혔는지 여부와
    # 무관하게 시도한다 — 이미지 관련 실패는 대부분 해상도 문제이거나,
    # 설령 다른 이유라도 업스케일로 우연히 해결되는 경우가 실사용에서
    # 관찰되었기 때문(예: 리듬 해석 실패도 더 선명한 입력에서 개선될 수 있음).
    if not source_path.lower().endswith(".pdf"):
        tail1 = log_tail(log_path, 25, 1800)
        try:
            up_src, upscale = _retry_scaled(source_path, work_dir)
        except MemoryError:
            up_src, upscale = source_path, 1.0
        ok2, mxl_path2, omr_path2, log_path2 = _run_audiveris_once(
            up_src, out_dir, timeout_sec, tag="_retry")
        if ok2:
            return AudiverisResult(
                status="ok", mxl_path=mxl_path2, omr_path=omr_path2,
                upscale_factor=upscale, log_path=log_path2,
            )
        return AudiverisResult(
            status="error",
            error=f"export_failed_or_empty(원본·배율 {upscale:g} 재시도 둘 다 실패)\n"
                  f"--- 1차 기록 끝 ---\n{tail1}\n--- 2차 기록 끝 ---\n{log_tail(log_path2, 25, 1800)}",
            omr_path=omr_path2 or omr_path, upscale_factor=upscale,
            log_path=log_path2,
        )

    return AudiverisResult(
        status="error", error="export_failed_or_empty\n--- 기록 끝 ---\n" + log_tail(log_path, 30, 2500),
        omr_path=omr_path, upscale_factor=1.0, log_path=log_path,
    )
