"""tracesum.py SERVER.log...: each request's prefill phases from an instrumented server's `inkling trace:` lines."""
import re
import sys

PAIR = re.compile(r"(\w+)=([\d.]+)")
for path in sys.argv[1:]:
    print(f"## {path.rsplit('/', 1)[-1]}")
    request = None
    for line in open(path, errors="replace"):
        if line.startswith("inkling session:"):
            request = {"chunks": [], "snapshots": [], "admit": None, "rollback": None}
        elif line.startswith("inkling trace:") and request is not None:
            kind = line.split()[2]
            fields = {k: float(v) for k, v in PAIR.findall(line)}
            if kind == "chunk":
                request["chunks"].append(fields)
            elif kind == "snapshot":
                request["snapshots"].append(fields)
            else:
                request[kind] = fields.get("wall_ms")
        elif line.startswith("inkling prefill:") and request is not None:
            total = float(line.split()[2]) * 1e3
            chunks = request["chunks"]
            wall = sum(c["wall_ms"] for c in chunks)
            gpu = sum(c["gpu_ms"] for c in chunks)
            tokens = int(sum(c["rows"] for c in chunks))
            parts = " ".join(f"{int(c['rows'])}:{c['wall_ms']:.0f}/{c['gpu_ms']:.0f}" for c in chunks)
            snaps = sum(s["wall_ms"] for s in request["snapshots"])
            print(f"{tokens:6d} tokens prefill {total:8.1f} ms = chunks wall {wall:7.1f} (gpu {gpu:7.1f}) + snapshots {snaps:5.1f} "
                  f"+ rollback {request['rollback'] or 0:4.1f} + other {total - wall - snaps - (request['rollback'] or 0):5.1f}; "
                  f"admit {request['admit'] or 0:.1f}; rows:wall/gpu {parts}")
            request = None
