"""Remote detached launcher executed on Colab inside BAYES-DYN-05 session.
Unpacks /content/bayes_dyn_05_src.zip, starts the chosen driver (full-GPU or exact FP32 streamed)
under detached bash with nohup, and zips models/<mkey>/ into bundle_<mkey>.zip upon exit!
"""

import os
import subprocess
import zipfile

MODEL_KEY = open("/content/target_model_key.txt", encoding="utf-8").read().strip()
DRIVER = open("/content/target_driver.txt", encoding="utf-8").read().strip()

os.makedirs("/content/BAYES_DYN_05", exist_ok=True)
with zipfile.ZipFile("/content/bayes_dyn_05_src.zip", "r") as zf:
    zf.extractall("/content/BAYES_DYN_05")

wrapper = f"""#!/bin/bash
cd /content/BAYES_DYN_05
python -u {DRIVER} --model_key {MODEL_KEY} > run_{MODEL_KEY}.log 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  python -c "import json, time; open('heartbeat_{MODEL_KEY}.json','w').write(json.dumps({{'status':'FAILED','phase':'CRASHED','step':0,'total':0,'epoch':time.time(),'note':'Exit code $RC'}}))"
fi
python -c "
import pathlib, zipfile
root = pathlib.Path('/content/BAYES_DYN_05')
mdir = root / 'models' / '{MODEL_KEY}'
out = root / 'bundle_{MODEL_KEY}.zip'
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
    for fn in ('run_{MODEL_KEY}.log', 'heartbeat_{MODEL_KEY}.json'):
        p = root / fn
        if p.exists(): zf.write(p, fn)
    if mdir.exists():
        for p in sorted(mdir.rglob('*')):
            if p.is_file():
                zf.write(p, p.relative_to(root).as_posix())
print('BUNDLE_READY', out.stat().st_size)
" >> run_{MODEL_KEY}.log 2>&1
touch /content/BAYES_DYN_05/BUNDLE_DONE_{MODEL_KEY}
exit $RC
"""
sh_path = f"/content/BAYES_DYN_05/run_detached_{MODEL_KEY}.sh"
with open(sh_path, "w", encoding="utf-8") as f:
    f.write(wrapper)
os.chmod(sh_path, 0o755)

proc = subprocess.Popen(
    ["nohup", "bash", sh_path],
    stdout=open(f"/content/BAYES_DYN_05/nohup_{MODEL_KEY}.out", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print(f"DETACHED_LAUNCH_OK model={MODEL_KEY} driver={DRIVER} pid={proc.pid}")
