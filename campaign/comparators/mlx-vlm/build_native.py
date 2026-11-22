import shlex
import subprocess
from pathlib import Path

base = Path('campaign/comparators/mlx-vlm')
command = ['zig', 'build', 'tf-inkling2-loop-run', '-Dcpu=apple_m1', '--prefix', str(base / 'build'), '--cache-dir', str(base / '.zig-cache'), '--verbose']
result = subprocess.run(command, capture_output=True, text=True, check=True)
line = next(line.removeprefix('info(verbose): ') for line in result.stderr.splitlines() if '-Mroot=zig/src/families/inkling2_loop/run.zig' in line)
args = shlex.split(line)
args = [arg.replace('-Mroot=zig/src/families/inkling2_loop/run.zig', '-Mroot=campaign/comparators/mlx-vlm/native.zig') for arg in args if arg != '--listen=-']
args.append('-femit-bin=campaign/comparators/mlx-vlm/build/bin/native-compare')
subprocess.run(args, check=True)
