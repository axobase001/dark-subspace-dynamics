"""Detached GPU launcher for BAYES-DYN-04 (archived in runtime/ per §9.6 & §11)."""
import os, subprocess, sys, zipfile

base = "/content/BAYES_DYN_04"
os.makedirs(base, exist_ok=True)
with zipfile.ZipFile("/content/bayes_dyn_04_src.zip", "r") as zf:
    zf.extractall(base)

wrapper = os.path.join(base, "_detached_driver.py")
with open(wrapper, "w", encoding="utf-8") as f:
    f.write(
        "import json, subprocess, sys, time, traceback\n"
        "from datetime import datetime, timezone\n"
        "from pathlib import Path\n"
        "base = Path('/content/BAYES_DYN_04')\n"
        "hb = base / 'heartbeat.json'\n"
        "t0 = time.time()\n"
        "try:\n"
        "    rc = subprocess.call([sys.executable, '-u', 'run_bayes_dyn_04.py'], cwd=str(base))\n"
        "    if rc != 0:\n"
        "        hb.write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'epoch': time.time(), 'elapsed_sec': round(time.time()-t0, 2), 'phase': 'CRASHED', 'step': 0, 'total': 0, 'status': 'FAILED', 'note': f'Exit code {rc}'}), encoding='utf-8')\n"
        "except Exception as e:\n"
        "    traceback.print_exc()\n"
        "    hb.write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'epoch': time.time(), 'elapsed_sec': round(time.time()-t0, 2), 'phase': 'CRASHED', 'step': 0, 'total': 0, 'status': 'FAILED', 'note': repr(e)}), encoding='utf-8')\n"
    )

log_f = open(os.path.join(base, "run.log"), "w", encoding="utf-8")
proc = subprocess.Popen(
    [sys.executable, "-u", wrapper],
    cwd=base,
    stdout=log_f,
    stderr=subprocess.STDOUT,
    start_new_session=True,
    close_fds=True,
)
with open(os.path.join(base, "run.pid"), "w", encoding="utf-8") as pf:
    pf.write(str(proc.pid))
print(f"DETACHED_LAUNCH_OK pid={proc.pid}", flush=True)
