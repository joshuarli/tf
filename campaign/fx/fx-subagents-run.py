import json
import os
import subprocess
import sys
import time
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo / 'campaign'))
import heavy

scratch = repo / '.work/tmp/fx-subagents-repo'
home = repo / '.work/tmp/fx-subagents-home-2'
output = repo / '.work/tmp/fx-subagents-output-2.txt'
prompt = ('Start exactly one subagent named fixer. Ask fixer to edit range_utils.py so clamp raises '
          'ValueError when low exceeds high, and to report its change. Wait for fixer to finish, '
          'then summarize the result. Work only in this repository; do not use the network.')
env = dict(os.environ, FX_HOME=str(home), PATH='/opt/homebrew/bin:' + os.environ['PATH'],
           FX_INKLING_MAX_OUTPUT='2048')
baseline = heavy.wired_bytes()
started = time.monotonic()
peak = 0
with output.open('w') as out:
    proc = subprocess.Popen([str(repo / 'campaign/fx/fx-local-inkling'), 'ask', '--effort', 'minimal',
                             '--full-access', '--no-color', prompt], cwd=scratch, env=env,
                            stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    while proc.poll() is None:
        members = heavy.tree(proc.pid)
        peak = max(peak, heavy.charge([heavy.usage(pid) for pid in members],
                                     max(0, heavy.wired_bytes() - baseline)))
        time.sleep(1)
print(json.dumps({'exit': proc.returncode, 'wall_seconds': time.monotonic() - started,
                  'peak_charged_bytes': peak, 'output': str(output)}), flush=True)
sys.exit(proc.returncode)
