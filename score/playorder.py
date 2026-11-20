# score/playorder.py
# ---------------------------------------------------------------------------
# [v2.80] 연주 순서: 도돌이표·볼타·D.C./D.S.·Fine·Coda를 따라 '연주하는 차례'의 마디 목록을 만든다.
#   · 도돌이 끝(:||)에서 한 번 도돌이 시작(||:, 없으면 곡 처음)으로 돌아간다. 두 번째에는 지나간다.
#   · 볼타: 지금 몇 번째로 지나는지와 같은 번호의 괄호만 연주한다.
#   · D.C.(처음으로)·D.S.(세뇨로): 한 번만 뛴다. 뛰고 난 뒤에는 도돌이를 다시 하지 않고, 볼타는 마지막 괄호만.
#     al Fine → Fine이 있는 마디에서 끝, al Coda → To Coda(없으면 첫 코다 기호) 마디 다음에 코다로.
# 마디의 이동 기호는 measure.nav = [{"kind": ...}] (kind: segno, coda, dc, ds, fine, dc_al_fine, dc_al_coda,
# ds_al_fine, ds_al_coda, to_coda).
# ---------------------------------------------------------------------------


def _kinds(m):
    return {nv["kind"] for nv in (getattr(m, "nav", None) or [])}


def performance_order(score):
    ms = sorted(score.measures, key=lambda m: m.number)
    n = len(ms)
    if n == 0:
        return []
    has_any = any(getattr(m, "start_repeat", False) or getattr(m, "end_barline", "") == "repeat_end"
                  or getattr(m, "ending", None) or _kinds(m) for m in ms)
    if not has_any:
        return list(ms)
    segno = next((i for i, m in enumerate(ms) if "segno" in _kinds(m)), 0)
    coda_signs = [i for i, m in enumerate(ms) if "coda" in _kinds(m)]
    to_coda = next((i for i, m in enumerate(ms) if "to_coda" in _kinds(m)), None)
    if to_coda is not None:
        coda_target = next((i for i in coda_signs if i > to_coda), None)
    else:
        to_coda = coda_signs[0] if len(coda_signs) >= 2 else None
        coda_target = coda_signs[1] if len(coda_signs) >= 2 else None

    def ending_group_max(i):
        lo = i
        while lo > 0 and getattr(ms[lo - 1], "ending", None):
            lo -= 1
        hi = i
        while hi + 1 < n and getattr(ms[hi + 1], "ending", None):
            hi += 1
        return max(getattr(ms[k], "ending", 0) or 0 for k in range(lo, hi + 1))

    out = []
    i, rep_start, pass_no = 0, 0, 1
    taken = set()
    jumped = None            # None | "end" | "fine" | "coda" | "done"
    steps = 0
    while i < n and steps < 20 * n:
        steps += 1
        m = ms[i]
        if getattr(m, "start_repeat", False) and rep_start != i and jumped is None:
            rep_start, pass_no = i, 1
        end_no = getattr(m, "ending", None)
        if end_no:
            if jumped is None and end_no != pass_no:
                i += 1
                continue
            if jumped is not None and end_no != ending_group_max(i):
                i += 1
                continue
        out.append(m)
        k = _kinds(m)
        if jumped == "fine" and "fine" in k:
            break
        if jumped == "coda" and i == to_coda and coda_target is not None:
            i, jumped = coda_target, "done"
            continue
        if getattr(m, "end_barline", "") == "repeat_end" and jumped is None and i not in taken:
            taken.add(i)
            pass_no += 1
            i = rep_start
            continue
        jump = next((x for x in ("dc_al_fine", "dc_al_coda", "ds_al_fine", "ds_al_coda", "dc", "ds") if x in k), None)
        if jump and jumped is None:
            jumped = "fine" if jump.endswith("al_fine") else ("coda" if jump.endswith("al_coda") else "end")
            i = 0 if jump.startswith("dc") else segno
            continue
        i += 1
    return out
