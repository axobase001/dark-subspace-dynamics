"""Independent 30s Heartbeat Watchdog + Cumulative 20 CU Budget Guard for BAYES-DYN-05.

Usage:
  python -u runtime/run_and_watch_v05.py --model_key qwen25_3b
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path(r"C:\Users\User\.gemini\antigravity\brain\7acb0e9b-b241-4215-bf97-54b7f1e0d278\scratch")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cli(cmd: list[str], timeout_sec: int = 90) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return -1, (e.stdout or ""), f"TIMEOUT {timeout_sec}s: {e.stderr or ''}"


def parse_usage(text: str) -> dict[str, Any]:
    bal = re.search(r"Current balance:\s*([0-9.]+)\s*compute units", text)
    rate = re.search(r"Usage rate:\s*([0-9.]+)/hr", text)
    act = re.search(r"Active assignments:\s*(\d+)", text)
    return {
        "balance_cu": float(bal.group(1)) if bal else None,
        "rate_per_hr": float(rate.group(1)) if rate else None,
        "active_assignments": int(act.group(1)) if act else None,
    }


def pack_src() -> Path:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = SCRATCH / "bayes_dyn_05_src.zip"
    include = [
        "configs/suite_config.json",
        "configs/data_manifest.json",
        "run_bayes_dyn_05_model.py",
        "run_bayes_dyn_05_7b_stream.py",
    ]
    for folder in ("src", "data"):
        for p in sorted((ROOT / folder).glob("*")):
            if p.is_file() and not p.name.endswith(".pyc") and "__pycache__" not in str(p):
                include.append(p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel in include:
            zf.write(ROOT / rel, rel)
    print(f"[WATCHDOG v05] Packed {len(include)} files -> {out} ({out.stat().st_size / 1024:.1f} KB)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", required=True)
    args = ap.parse_args()
    mkey = args.model_key

    cfg = json.loads((ROOT / "configs" / "suite_config.json").read_text(encoding="utf-8"))
    assert mkey in cfg["models"], f"Unknown model_key={mkey}"
    mcfg = cfg["models"][mkey]
    gpu_type = str(mcfg["gpu"])
    high_mem = bool(mcfg.get("high_mem", False))
    session = f"bayes-dyn-05-{mkey.replace('_', '-')}"

    rt_dir = ROOT / "runtime" / mkey
    rt_dir.mkdir(parents=True, exist_ok=True)
    evt_log = rt_dir / "watchdog_events.jsonl"
    if evt_log.exists():
        evt_log.unlink()
    receipt_lines = [f"=== BAYES-DYN-05 ({mkey}: {mcfg['label']}) REMOTE LIFECYCLE & STOP RECEIPT ==="]

    def _log(evt: dict[str, Any]) -> None:
        with evt_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"utc": utc_now(), **evt}) + "\n")

    rc_h, _, _ = run_cli(["colab", "stop", "--help"], timeout_sec=20)
    assert rc_h == 0, "colab stop CLI unavailable!"
    _, u_bef_out, _ = run_cli(["colab", "usage"], timeout_sec=25)
    _, st_bef_out, _ = run_cli(["colab", "status"], timeout_sec=25)
    u_bef = parse_usage(u_bef_out)
    receipt_lines.append(f"\n--- PRE-RUN ({utc_now()}) ---\n{u_bef_out.strip()}\n{st_bef_out.strip()}")

    # Global cumulative CU budget guard (user hard cap 20.0 CU, watchdog soft ceiling 18.0 CU!)
    base_cu = float(cfg["budget"]["baseline_cu_before_suite"])
    global_cap = float(cfg["budget"]["watchdog_global_cu_cap"])
    spent_so_far = (base_cu - u_bef["balance_cu"]) if u_bef["balance_cu"] is not None else 0.0
    per_model_cap = float(cfg["budget"]["per_model_cu_cap_a100"] if gpu_type == "A100" else cfg["budget"]["per_model_cu_cap_l4"])
    rem_global = global_cap - spent_so_far
    print(f"[WATCHDOG v05:{mkey}] Pre-run CU={u_bef['balance_cu']} | suite spent={spent_so_far:.2f}/{global_cap:.1f} CU | model cap={per_model_cap:.1f} CU | GPU={gpu_type} high_mem={high_mem}", flush=True)
    if rem_global < 1.0:
        raise RuntimeError(f"Global CU safety cap reached ({spent_so_far:.2f} >= {global_cap:.1f} CU)! Refusing to launch {mkey}.")

    src_zip = pack_src()
    key_txt = SCRATCH / f"target_model_key_{mkey}.txt"
    key_txt.write_text(mkey, encoding="utf-8")

    session_created = False
    t_sess = time.time()
    stop_reason = "UNKNOWN"
    final_hb_status = "UNKNOWN"
    sess_line = ""
    rate_hr = 1.94 if gpu_type == "L4" else 11.77

    try:
        # For 7B/8B models: use exact FP32 layer-streaming on L4 (1.94 CU/hr, >12 GB free VRAM!)
        # or full-GPU on A100 if explicitly requested via --prefer-a100
        if gpu_type == "A100":
            gpu_attempts = [("L4", False, "run_bayes_dyn_05_7b_stream.py"), ("A100", False, "run_bayes_dyn_05_model.py")]
        else:
            gpu_attempts = [(gpu_type, high_mem, "run_bayes_dyn_05_model.py")]
        rc_new, out_new, err_new = 1, "", ""
        acquired_gpu = gpu_type
        chosen_driver = "run_bayes_dyn_05_model.py"
        for g_try, hm_try, drv_try in gpu_attempts:
            new_cmd = ["colab", "new", "--gpu", g_try, "-s", session]
            if hm_try:
                new_cmd.append("--high-mem")
            print(f"[WATCHDOG v05:{mkey}] Requesting GPU={g_try} high_mem={hm_try} driver={drv_try}...", flush=True)
            rc_new, out_new, err_new = run_cli(new_cmd, timeout_sec=120)
            if out_new.strip():
                print(out_new.strip(), flush=True)
            if rc_new == 0:
                acquired_gpu = g_try
                high_mem = hm_try
                chosen_driver = drv_try
                break
            print(f"[WATCHDOG v05:{mkey}] GPU={g_try} high_mem={hm_try} unavailable; trying next...", flush=True)
        if rc_new != 0:
            raise RuntimeError(f"colab new failed across {gpu_attempts}: {err_new}")
        gpu_type = acquired_gpu
        drv_txt = SCRATCH / f"target_driver_{mkey}.txt"
        drv_txt.write_text(chosen_driver, encoding="utf-8")
        session_created = True
        t_sess = time.time()
        _, st_s_out, _ = run_cli(["colab", "status"], timeout_sec=25)
        _, u_s_out, _ = run_cli(["colab", "usage"], timeout_sec=25)
        u_s = parse_usage(u_s_out)
        if u_s["rate_per_hr"] and u_s["rate_per_hr"] > 0:
            rate_hr = u_s["rate_per_hr"]
        eff_cu_cap = min(per_model_cap, rem_global)
        wall_cap_m = 65.0 if chosen_driver == "run_bayes_dyn_05_7b_stream.py" else float(cfg["budget"]["per_model_wall_cap_min"])
        hard_deadline_min = min(wall_cap_m, (eff_cu_cap / rate_hr) * 60.0)
        for ln in st_s_out.splitlines():
            if f"[{session}]" in ln:
                sess_line = ln.strip()
        receipt_lines.append(f"\n--- SESSION CREATED ({utc_now()}) ---\n{sess_line}\nDriver={chosen_driver} | Rate={rate_hr}/hr | eff_cu_cap={eff_cu_cap:.2f} | Hard deadline={hard_deadline_min:.1f} min\n{u_s_out.strip()}")
        _log({"event": "session_created", "session_line": sess_line, "driver": chosen_driver, "rate_hr": rate_hr, "eff_cu_cap": round(eff_cu_cap, 2), "hard_deadline_min": round(hard_deadline_min, 2)})
        print(f"[WATCHDOG v05:{mkey}] Session ready | driver={chosen_driver} | rate={rate_hr}/hr | hard deadline={hard_deadline_min:.1f} min", flush=True)

        rc_u1, _, err_u1 = run_cli(["colab", "upload", "-s", session, str(src_zip), "/content/bayes_dyn_05_src.zip"], timeout_sec=120)
        rc_u2, _, err_u2 = run_cli(["colab", "upload", "-s", session, str(key_txt), "/content/target_model_key.txt"], timeout_sec=60)
        rc_u3, _, err_u3 = run_cli(["colab", "upload", "-s", session, str(drv_txt), "/content/target_driver.txt"], timeout_sec=60)
        if rc_u1 != 0 or rc_u2 != 0 or rc_u3 != 0:
            raise RuntimeError(f"colab upload failed: {err_u1} / {err_u2} / {err_u3}")
        print(f"[WATCHDOG v05:{mkey}] Uploaded source bundle. Launching detached GPU driver...", flush=True)

        out_ex, err_ex, rc_ex = "", "", 1
        for attempt in range(1, 4):
            rc_ex, out_ex, err_ex = run_cli(["colab", "exec", "-s", session, "--timeout", "90", "-f", str(ROOT / "runtime" / "remote_launch_v05.py")], timeout_sec=110)
            if out_ex.strip():
                print(out_ex.strip(), flush=True)
            if rc_ex == 0 and "DETACHED_LAUNCH_OK" in out_ex:
                break
            print(f"[WATCHDOG v05:{mkey}] Launch attempt {attempt}/3 failed (rc={rc_ex}); checking heartbeat...", flush=True)
            rc_chk, _, _ = run_cli(["colab", "download", "-s", session, f"/content/BAYES_DYN_05/heartbeat_{mkey}.json", str(SCRATCH / f"pre_hb_{mkey}.json")], timeout_sec=20)
            if rc_chk == 0:
                rc_ex, out_ex = 0, "DETACHED_LAUNCH_OK (confirmed via heartbeat)"
                break
            time.sleep(5)
        if rc_ex != 0 or "DETACHED_LAUNCH_OK" not in out_ex:
            raise RuntimeError(f"Detached launch failed after 3 attempts: {err_ex}")
        _log({"event": "detached_launch_ok", "out": out_ex.strip()})

        hb_loc = SCRATCH / f"live_hb_{mkey}.json"
        last_ep = 0.0
        last_hb_wall = time.time()
        last_sig = ("", -1)
        last_prog_wall = time.time()
        last_u_chk = time.time()
        last_keepalive = time.time()

        while True:
            time.sleep(30)
            now = time.time()
            wall_min = (now - t_sess) / 60.0

            # Keep Jupyter kernel active every 4 min via valid `-f keepalive.py` so Colab's ~20-min idle reaper NEVER kills long 7B/8B runs!
            if now - last_keepalive >= 240.0:
                last_keepalive = now
                rc_ka, out_ka, _ = run_cli(["colab", "exec", "-s", session, "--timeout", "25", "-f", str(ROOT / "runtime" / "keepalive.py")], timeout_sec=35)
                ka_ok = (rc_ka == 0 and "KEEPALIVE_OK" in out_ka)
                print(f"[KEEPALIVE {mkey} {wall_min:5.1f}m] rc={rc_ka} ok={ka_ok}", flush=True)
                _log({"event": "kernel_keepalive", "wall_min": round(wall_min, 2), "rc": rc_ka, "ok": ka_ok})

            rc_hb, _, _ = run_cli(["colab", "download", "-s", session, f"/content/BAYES_DYN_05/heartbeat_{mkey}.json", str(hb_loc)], timeout_sec=30)
            hb = None
            if rc_hb == 0 and hb_loc.exists():
                try:
                    hb = json.loads(hb_loc.read_text(encoding="utf-8"))
                except Exception:
                    hb = None

            if hb is not None:
                ep = float(hb.get("epoch", 0.0))
                sig = (str(hb.get("phase", "")), int(hb.get("step", -1)))
                st_str = str(hb.get("status", "RUNNING"))
                if ep > last_ep:
                    last_ep = ep
                    last_hb_wall = now
                if sig != last_sig:
                    last_sig = sig
                    last_prog_wall = now
                no_hb = now - last_hb_wall
                no_prog = now - last_prog_wall
                print(f"[HB {mkey} {wall_min:5.1f}m] phase={sig[0]} step={sig[1]}/{hb.get('total', '?')} status={st_str} | no_prog={no_prog:.0f}s | {hb.get('note', '')}", flush=True)
                _log({"event": "heartbeat", "wall_min": round(wall_min, 2), "phase": sig[0], "step": sig[1], "status": st_str, "no_prog_s": round(no_prog, 1)})

                if st_str in ("COMPLETED", "FAILED"):
                    final_hb_status = st_str
                    stop_reason = f"NORMAL_{st_str}"
                    # Wait briefly for wrapper script to finish zipping bundle_<mkey>.zip
                    for _ in range(12):
                        rc_done, _, _ = run_cli(["colab", "download", "-s", session, f"/content/BAYES_DYN_05/BUNDLE_DONE_{mkey}", str(SCRATCH / f"done_{mkey}")], timeout_sec=20)
                        if rc_done == 0:
                            break
                        time.sleep(5)
                    break

                # Allow up to 10 min for 7B/8B HF download / SVD / CPU P0 ref / prior fit; 6 min for compute phases
                stall_lim = 600.0 if sig[0] in ("LOAD_MODEL", "SPECTRAL_SVD", "P0_CPU_REF", "MOVE_SUFFIX_GPU", "MOVE_TO_GPU", "FIT_PRIOR", "P0_CHECKS") else 360.0
                if no_prog > stall_lim:
                    stop_reason = f"STALLED_{sig[0]}_{int(no_prog)}s"
                    final_hb_status = "STALLED"
                    break
            else:
                if (now - last_hb_wall) > 300.0:
                    stop_reason = "NO_HEARTBEAT_5MIN"
                    final_hb_status = "UNREACHABLE"
                    break

            if now - last_u_chk >= 180.0:
                last_u_chk = now
                _, u_now_out, _ = run_cli(["colab", "usage"], timeout_sec=25)
                u_now = parse_usage(u_now_out)
                if u_now["balance_cu"] is not None:
                    cum_used = base_cu - u_now["balance_cu"]
                    print(f"[WATCHDOG v05:{mkey}] CU balance={u_now['balance_cu']} (suite used={cum_used:.2f}/{global_cap:.1f} CU)", flush=True)
                    if cum_used >= global_cap:
                        stop_reason = f"GLOBAL_CU_CAP_{cum_used:.2f}"
                        final_hb_status = "BUDGET_STOP"
                        break

            if wall_min >= hard_deadline_min:
                stop_reason = f"WALL_DEADLINE_{wall_min:.1f}m"
                final_hb_status = "BUDGET_STOP"
                break

        # Download single complete bundle_<mkey>.zip (or fallback to log)
        print(f"[WATCHDOG v05:{mkey}] Downloading results bundle (reason={stop_reason})...", flush=True)
        b_loc = SCRATCH / f"bundle_{mkey}.zip"
        if b_loc.exists():
            b_loc.unlink()
        rc_b, _, err_b = run_cli(["colab", "download", "-s", session, f"/content/BAYES_DYN_05/bundle_{mkey}.zip", str(b_loc)], timeout_sec=300)
        if rc_b == 0 and b_loc.exists():
            print(f"[DOWNLOAD {mkey}] OK bundle_{mkey}.zip ({b_loc.stat().st_size / (1024*1024):.2f} MB) -> extracting...", flush=True)
            with zipfile.ZipFile(b_loc, "r") as zf:
                zf.extractall(ROOT)
            # Move root-extracted log & heartbeat cleanly into runtime/<mkey>/
            for fn in (f"run_{mkey}.log", f"heartbeat_{mkey}.json"):
                src_f = ROOT / fn
                if src_f.exists():
                    dst_f = rt_dir / fn
                    dst_f.write_bytes(src_f.read_bytes())
                    src_f.unlink()
        else:
            print(f"[DOWNLOAD {mkey}] bundle missing ({err_b}); trying raw log...", flush=True)
            run_cli(["colab", "download", "-s", session, f"/content/BAYES_DYN_05/run_{mkey}.log", str(rt_dir / f"run_{mkey}.log")], timeout_sec=60)

    except Exception as e:
        stop_reason = f"LOCAL_EXCEPTION_{type(e).__name__}"
        final_hb_status = "EXCEPTION"
        print(f"[WATCHDOG ERROR:{mkey}] {stop_reason}: {e}", flush=True)
        _log({"event": "local_exception", "error": str(e)})
    finally:
        stop_req_utc = utc_now()
        stop_confirmed = False
        if session_created:
            print(f"[WATCHDOG v05:{mkey}] Stopping ONLY '{session}' (reason={stop_reason})...", flush=True)
            rc_st, out_st, err_st = run_cli(["colab", "stop", "-s", session], timeout_sec=60)
            stop_conf_utc = utc_now()
            _, st_aft_out, _ = run_cli(["colab", "status"], timeout_sec=25)
            _, u_aft_out, _ = run_cli(["colab", "usage"], timeout_sec=25)
            u_aft = parse_usage(u_aft_out)
            stop_confirmed = (f"[{session}]" not in st_aft_out)
            receipt_lines.append(f"\n--- REMOTE STOP ({stop_req_utc} -> {stop_conf_utc}) ---\nReason: {stop_reason}\nrc={rc_st}\n{out_st.strip()}\n{err_st.strip()}")
            receipt_lines.append(f"\n--- POST-STOP STATUS ---\n{st_aft_out.strip()}")
            receipt_lines.append(f"\n--- POST-STOP USAGE ---\n{u_aft_out.strip()}")
        else:
            stop_conf_utc = utc_now()
            _, u_aft_out, _ = run_cli(["colab", "usage"], timeout_sec=25)
            u_aft = parse_usage(u_aft_out)
            stop_confirmed = True

        wall_sec = round(time.time() - t_sess, 2) if session_created else 0.0
        cu_delta = round(u_bef["balance_cu"] - u_aft["balance_cu"], 4) if (u_bef["balance_cu"] is not None and u_aft["balance_cu"] is not None) else None
        est_cu = round((wall_sec / 3600.0) * rate_hr, 4)
        ledger = {
            "experiment_id": "BAYES-DYN-05", "model_key": mkey, "label": mcfg["label"],
            "gpu_requested": gpu_type, "high_mem": high_mem, "session_name": session,
            "session_id_line": sess_line, "stop_reason": stop_reason,
            "final_heartbeat_status": final_hb_status,
            "stop_requested_utc": stop_req_utc, "stop_confirmed_utc": stop_conf_utc,
            "remote_stop_confirmed": stop_confirmed, "wall_clock_sec": wall_sec,
            "reported_rate_per_hr": rate_hr, "estimated_cu_from_wall": est_cu,
            "colab_cu_before": u_bef["balance_cu"], "colab_cu_after": u_aft["balance_cu"],
            "colab_cu_delta_immediate": cu_delta,
            "post_stop_active_assignments_account": u_aft["active_assignments"],
        }
        (rt_dir / "remote_stop_receipt.txt").write_text("\n".join(receipt_lines) + "\n", encoding="utf-8")
        (rt_dir / "resource_ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        print(f"[WATCHDOG v05:{mkey}] stop_confirmed={stop_confirmed} | wall={wall_sec:.0f}s (est {est_cu:.2f} CU) | CU={u_bef['balance_cu']}->{u_aft['balance_cu']}", flush=True)


if __name__ == "__main__":
    main()
