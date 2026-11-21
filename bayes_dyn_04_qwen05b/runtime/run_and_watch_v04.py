"""Independent External 30-Second Heartbeat Watchdog & Auto-Release Manager for BAYES-DYN-04 (§9.6, §10, §11).

Archived directly inside BAYES_DYN_04/runtime/ for full auditability:
  - Launches run_bayes_dyn_04.py detached via a 5-second colab exec call (zero fragile long-lived websocket!).
  - Polls /content/BAYES_DYN_04/heartbeat.json every 30s with strict 30s subprocess timeouts.
  - Enforces hard auto-release on:
      * > 180s (3 min) without heartbeat advance
      * > 480s (8 min) without (phase, step) advance
      * >= 1.80 CU consumed OR min(60 min, 1.8/rate_hr * 60 min) hard cutoff
      * Any local exception / KeyboardInterrupt via guaranteed try/finally `colab stop -s bayes-dyn-04`!
  - NEVER touches any session other than `bayes-dyn-04`.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from typing import Any

ROOT = pathlib.Path(r"C:\Users\User\.gemini\antigravity\scratch\BAYES_DYN_04")
SCRATCH = pathlib.Path(r"C:\Users\User\.gemini\antigravity\brain\7acb0e9b-b241-4215-bf97-54b7f1e0d278\scratch")
SESSION = "bayes-dyn-04"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cli(cmd: list[str], timeout_sec: int = 45) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return -9, (e.stdout or ""), f"TIMEOUT {timeout_sec}s: {e.stderr or ''}"


def parse_usage(stdout: str) -> dict[str, Any]:
    bal = float(m.group(1)) if (m := re.search(r"Current balance:\s*([0-9.]+)", stdout)) else None
    rate = float(m.group(1)) if (m := re.search(r"Usage rate:\s*([0-9.]+)/hr", stdout)) else None
    act = int(m.group(1)) if (m := re.search(r"Active assignments:\s*(\d+)", stdout)) else None
    return {"balance_cu": bal, "rate_per_hr": rate, "active_assignments": act}


def pack_src() -> pathlib.Path:
    zpath = SCRATCH / "bayes_dyn_04_src.zip"
    files = (
        [ROOT / "run_bayes_dyn_04.py", ROOT / "results" / "prior_model.npz"]
        + list((ROOT / "src").glob("*.py"))
        + list((ROOT / "configs").glob("*"))
        + list((ROOT / "data").glob("*.jsonl"))
        + list((ROOT / "audit").glob("*.py"))
    )
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            if p.is_file() and "__pycache__" not in str(p):
                zf.write(p, p.relative_to(ROOT).as_posix())
    print(f"[WATCHDOG v04] Packed {len(files)} files (incl. pre-fitted prior_model.npz) -> {zpath} ({zpath.stat().st_size / (1024**2):.2f} MB)", flush=True)
    return zpath


def download_all(session: str) -> list[str]:
    targets = [
        ("/content/BAYES_DYN_04/run.log", ROOT / "runtime" / "run_colab.log"),
        ("/content/BAYES_DYN_04/heartbeat.json", ROOT / "runtime" / "final_heartbeat.json"),
        ("/content/BAYES_DYN_04/atlas_index.jsonl", ROOT / "atlas_index.jsonl"),
        ("/content/BAYES_DYN_04/configs/model_environment.json", ROOT / "configs" / "model_environment.json"),
        ("/content/BAYES_DYN_04/configs/frozen_config.json", ROOT / "configs" / "frozen_config.json"),
        ("/content/BAYES_DYN_04/results/pip_freeze.txt", ROOT / "results" / "pip_freeze.txt"),
        ("/content/BAYES_DYN_04/results/p0_checks.json", ROOT / "results" / "p0_checks.json"),
        ("/content/BAYES_DYN_04/results/predictions.jsonl", ROOT / "results" / "predictions.jsonl"),
        ("/content/BAYES_DYN_04/results/metrics_summary.json", ROOT / "results" / "metrics_summary.json"),
        ("/content/BAYES_DYN_04/results/observations/query_receipts.jsonl", ROOT / "results" / "observations" / "query_receipts.jsonl"),
        ("/content/BAYES_DYN_04/results/observations/cal_obs.npz", ROOT / "results" / "observations" / "cal_obs.npz"),
        ("/content/BAYES_DYN_04/results/observations/confirm_id_obs.npz", ROOT / "results" / "observations" / "confirm_id_obs.npz"),
        ("/content/BAYES_DYN_04/results/observations/confirm_ood_obs.npz", ROOT / "results" / "observations" / "confirm_ood_obs.npz"),
        ("/content/BAYES_DYN_04/results/posteriors/cal_posteriors.npz", ROOT / "results" / "posteriors" / "cal_posteriors.npz"),
        ("/content/BAYES_DYN_04/results/posteriors/confirm_id_posteriors.npz", ROOT / "results" / "posteriors" / "confirm_id_posteriors.npz"),
        ("/content/BAYES_DYN_04/results/posteriors/confirm_ood_posteriors.npz", ROOT / "results" / "posteriors" / "confirm_ood_posteriors.npz"),
        ("/content/BAYES_DYN_04/results/interventions/dev_pilot_interventions.jsonl", ROOT / "results" / "interventions" / "dev_pilot_interventions.jsonl"),
        ("/content/BAYES_DYN_04/results/interventions/confirm_id_interventions.jsonl", ROOT / "results" / "interventions" / "confirm_id_interventions.jsonl"),
        ("/content/BAYES_DYN_04/results/interventions/confirm_ood_interventions.jsonl", ROOT / "results" / "interventions" / "confirm_ood_interventions.jsonl"),
        ("/content/BAYES_DYN_04/results/interventions/intervention_vectors.npz", ROOT / "results" / "interventions" / "intervention_vectors.npz"),
        ("/content/BAYES_DYN_04/results/calibration/cal32_temperature_fit.json", ROOT / "results" / "calibration" / "cal32_temperature_fit.json"),
        ("/content/BAYES_DYN_04/results/calibration/cal32_raw_errors.npz", ROOT / "results" / "calibration" / "cal32_raw_errors.npz"),
        ("/content/BAYES_DYN_04/results/trajectory_panel/trajectory_16_states.npz", ROOT / "results" / "trajectory_panel" / "trajectory_16_states.npz"),
        ("/content/BAYES_DYN_04/results/trajectory_panel/trajectory_16_summary.json", ROOT / "results" / "trajectory_panel" / "trajectory_16_summary.json"),
        ("/content/BAYES_DYN_04/results/trajectory_panel/trajectory_16_mechanisms.json", ROOT / "results" / "trajectory_panel" / "trajectory_16_mechanisms.json"),
    ]
    ok_list = []
    for rem, loc in targets:
        loc.parent.mkdir(parents=True, exist_ok=True)
        rc, _, _ = run_cli(["colab", "download", "-s", session, rem, str(loc)], timeout_sec=180)
        if rc == 0 and loc.exists():
            ok_list.append(loc.name)
            print(f"[DOWNLOAD] OK {loc.name} ({loc.stat().st_size:,d} B)", flush=True)
        else:
            print(f"[DOWNLOAD] missing/skip {loc.name}", flush=True)
    return ok_list


def main() -> None:
    rt_dir = ROOT / "runtime"
    rt_dir.mkdir(parents=True, exist_ok=True)
    evt_log = rt_dir / "watchdog_events.jsonl"
    if evt_log.exists():
        evt_log.unlink()
    receipt_lines = ["=== BAYES-DYN-04 REMOTE LIFECYCLE & STOP RECEIPT ==="]

    def _log(evt: dict[str, Any]) -> None:
        with evt_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"utc": utc_now(), **evt}) + "\n")

    rc_h, _, _ = run_cli(["colab", "stop", "--help"], timeout_sec=20)
    assert rc_h == 0, "colab stop CLI unavailable!"
    _, u_bef_out, _ = run_cli(["colab", "usage"], timeout_sec=25)
    _, st_bef_out, _ = run_cli(["colab", "status"], timeout_sec=25)
    u_bef = parse_usage(u_bef_out)
    receipt_lines.append(f"\n--- PRE-RUN ({utc_now()}) ---\n{u_bef_out.strip()}\n{st_bef_out.strip()}")
    print(f"[WATCHDOG v04] Pre-run CU={u_bef['balance_cu']}, active={u_bef['active_assignments']}", flush=True)

    src_zip = pack_src()
    session_created = False
    t_sess = time.time()
    stop_reason = "UNKNOWN"
    final_hb_status = "UNKNOWN"
    sess_line = ""
    rate_hr = 1.94

    try:
        rc_new, out_new, err_new = run_cli(["colab", "new", "--gpu", "L4", "-s", SESSION], timeout_sec=90)
        print(out_new.strip(), flush=True)
        if rc_new != 0:
            raise RuntimeError(f"colab new failed: {err_new}")
        session_created = True
        t_sess = time.time()
        _, st_s_out, _ = run_cli(["colab", "status"], timeout_sec=25)
        _, u_s_out, _ = run_cli(["colab", "usage"], timeout_sec=25)
        u_s = parse_usage(u_s_out)
        if u_s["rate_per_hr"] and u_s["rate_per_hr"] > 0:
            rate_hr = u_s["rate_per_hr"]
        hard_deadline_min = min(60.0, (1.8 / rate_hr) * 60.0)
        for ln in st_s_out.splitlines():
            if f"[{SESSION}]" in ln:
                sess_line = ln.strip()
        receipt_lines.append(f"\n--- SESSION CREATED ({utc_now()}) ---\n{sess_line}\nHard deadline={hard_deadline_min:.1f} min\n{u_s_out.strip()}")
        _log({"event": "session_created", "session_line": sess_line, "rate_hr": rate_hr, "hard_deadline_min": round(hard_deadline_min, 2)})

        rc_up, _, err_up = run_cli(["colab", "upload", "-s", SESSION, str(src_zip), "/content/bayes_dyn_04_src.zip"], timeout_sec=180)
        if rc_up != 0:
            raise RuntimeError(f"colab upload failed: {err_up}")
        print("[WATCHDOG v04] Bundle uploaded. Launching detached GPU driver...", flush=True)

        out_ex, err_ex, rc_ex = "", "", 1
        for attempt in range(1, 4):
            rc_ex, out_ex, err_ex = run_cli(["colab", "exec", "-s", SESSION, "--timeout", "90", "-f", str(rt_dir / "remote_launch_v04.py")], timeout_sec=110)
            if out_ex.strip():
                print(out_ex.strip(), flush=True)
            if rc_ex == 0 and "DETACHED_LAUNCH_OK" in out_ex:
                break
            print(f"[WATCHDOG v04] Launch attempt {attempt}/3 failed (rc={rc_ex}); checking remote heartbeat before retry...", flush=True)
            rc_chk, _, _ = run_cli(["colab", "download", "-s", SESSION, "/content/BAYES_DYN_04/heartbeat.json", str(SCRATCH / "pre_hb_chk.json")], timeout_sec=20)
            if rc_chk == 0:
                print("[WATCHDOG v04] Remote heartbeat.json already exists; detached process IS running!", flush=True)
                rc_ex = 0
                out_ex = "DETACHED_LAUNCH_OK (confirmed via heartbeat)"
                break
            time.sleep(5)
        if rc_ex != 0 or "DETACHED_LAUNCH_OK" not in out_ex:
            raise RuntimeError(f"Detached launch failed after 3 attempts: {err_ex}")
        _log({"event": "detached_launch_ok", "out": out_ex.strip()})

        hb_loc = SCRATCH / "live_hb_v04.json"
        last_ep = 0.0
        last_hb_wall = time.time()
        last_sig = ("", -1)
        last_prog_wall = time.time()
        last_u_chk = time.time()
        cu_now = u_bef["balance_cu"]

        while True:
            time.sleep(30)
            now = time.time()
            wall_min = (now - t_sess) / 60.0

            rc_hb, _, _ = run_cli(["colab", "download", "-s", SESSION, "/content/BAYES_DYN_04/heartbeat.json", str(hb_loc)], timeout_sec=30)
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
                no_hb, no_pr = now - last_hb_wall, now - last_prog_wall
                print(f"[HEARTBEAT {wall_min:5.1f}m] phase={sig[0]} step={sig[1]}/{hb.get('total')} status={st_str} | no_hb={no_hb:.0f}s no_prog={no_pr:.0f}s | {hb.get('note','')}", flush=True)
                _log({"event": "hb_ok", "wall_min": round(wall_min, 2), "phase": sig[0], "step": sig[1], "status": st_str, "no_hb": round(no_hb, 1), "no_prog": round(no_pr, 1), "cu": cu_now})
                if st_str in ("COMPLETED", "STOPPED_GATE", "FAILED"):
                    final_hb_status = st_str
                    stop_reason = f"NORMAL_{st_str}"
                    break
            else:
                no_hb, no_pr = now - last_hb_wall, now - last_prog_wall
                print(f"[HEARTBEAT {wall_min:5.1f}m] missed | no_hb={no_hb:.0f}s no_prog={no_pr:.0f}s", flush=True)
                _log({"event": "hb_missed", "wall_min": round(wall_min, 2), "no_hb": round(no_hb, 1)})

            if now - last_u_chk >= 120.0:
                last_u_chk = now
                _, u_mid_out, _ = run_cli(["colab", "usage"], timeout_sec=25)
                um = parse_usage(u_mid_out)
                if um["balance_cu"] is not None:
                    cu_now = um["balance_cu"]
                if u_bef["balance_cu"] is not None and cu_now is not None and (u_bef["balance_cu"] - cu_now) >= 1.80:
                    stop_reason = "AUTO_RELEASE_CU_1.80"
                    final_hb_status = "INCOMPLETE_CU_LIMIT"
                    break

            if wall_min >= hard_deadline_min:
                stop_reason = f"AUTO_RELEASE_DEADLINE_{hard_deadline_min:.1f}M"
                final_hb_status = "INCOMPLETE_DEADLINE"
                break
            if (now - last_hb_wall) > 180.0:
                stop_reason = "AUTO_RELEASE_NO_HEARTBEAT_180S"
                final_hb_status = "INCOMPLETE_NO_HEARTBEAT"
                break
            if (now - last_prog_wall) > 480.0:
                stop_reason = "AUTO_RELEASE_NO_PROGRESS_480S"
                final_hb_status = "INCOMPLETE_STALLED"
                break

        print(f"[WATCHDOG v04] Downloading all outputs (reason={stop_reason})...", flush=True)
        download_all(SESSION)

    except Exception as exc:
        stop_reason = f"LOCAL_EXCEPTION_{type(exc).__name__}: {exc}"
        final_hb_status = "FAILED_LOCAL"
        print(f"[WATCHDOG ERROR] {stop_reason}", flush=True)
        _log({"event": "local_exception", "err": stop_reason})
    finally:
        req_utc = utc_now()
        if session_created:
            print(f"[WATCHDOG v04] Stopping ONLY '{SESSION}' (reason={stop_reason})...", flush=True)
            rc_st, out_st, err_st = run_cli(["colab", "stop", "-s", SESSION], timeout_sec=60)
            done_utc = utc_now()
            _, st_a_out, _ = run_cli(["colab", "status"], timeout_sec=25)
            _, u_a_out, _ = run_cli(["colab", "usage"], timeout_sec=25)
            u_a = parse_usage(u_a_out)
            stop_ok = (rc_st == 0) and (f"[{SESSION}]" not in st_a_out)
            receipt_lines.append(f"\n--- REMOTE STOP ({req_utc} -> {done_utc}) ---\nReason: {stop_reason}\nrc={rc_st}\n{out_st.strip()}\n{err_st.strip()}\n\n--- POST-STOP STATUS ---\n{st_a_out.strip()}\n\n--- POST-STOP USAGE ---\n{u_a_out.strip()}\n")
            cu_d = round(u_bef["balance_cu"] - u_a["balance_cu"], 2) if (u_bef["balance_cu"] is not None and u_a["balance_cu"] is not None) else None
            ledger = {
                "experiment_id": "BAYES-DYN-04", "session_name": SESSION, "session_id_line": sess_line,
                "stop_reason": stop_reason, "final_heartbeat_status": final_hb_status,
                "stop_requested_utc": req_utc, "stop_confirmed_utc": done_utc, "remote_stop_confirmed": stop_ok,
                "wall_clock_sec": round(time.time() - t_sess, 2), "reported_rate_per_hr": rate_hr,
                "colab_cu_before": u_bef["balance_cu"], "colab_cu_after": u_a["balance_cu"], "colab_cu_delta": cu_d,
                "post_stop_active_assignments_account": u_a["active_assignments"],
            }
            (rt_dir / "resource_ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
            (rt_dir / "remote_stop_receipt.txt").write_text("\n".join(receipt_lines) + "\n", encoding="utf-8")
            print(f"[WATCHDOG v04] '{SESSION}' stop_confirmed={stop_ok} | CU: {u_bef['balance_cu']} -> {u_a['balance_cu']} (delta={cu_d})", flush=True)


if __name__ == "__main__":
    main()
