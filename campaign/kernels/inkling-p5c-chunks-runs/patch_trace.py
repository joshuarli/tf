"""patch_trace.py HOST_ZIG: add per-phase timing prints to an Inkling host's admit (scratch builds only)."""
import sys

path = sys.argv[1]
s = open(path).read()


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:90]
    s = s.replace(old, new)


sub('        std.debug.print("inkling session: slot {d}\\n", .{choice.index});\n        const prefill_start = h.now();\n',
    '        std.debug.print("inkling session: slot {d}\\n", .{choice.index});\n        const prefill_start = h.now();\n'
    '        std.debug.print("inkling trace: admit wall_ms={d:.3}\\n", .{ms(@intCast(prefill_start - job.began))});\n')
sub('            const filled = inkling.prefillWant(',
    '            const t_chunk = h.now();\n            const filled = inkling.prefillWant(')
sub('            h.prefill.taps = null;\n',
    '            h.prefill.taps = null;\n'
    '            std.debug.print("inkling trace: chunk at={d} rows={d} wall_ms={d:.3} gpu_ms={d:.3}\\n", .{ at, take, ms(@intCast(h.now() - t_chunk)), h.prefill.gpu_seconds * 1e3 });\n')
sub('        if (slot.rollback) |*rollback| {\n            rollback.copyRows(',
    '        const t_roll = h.now();\n'
    '        if (slot.rollback) |*rollback| {\n            rollback.copyRows(')
sub('        const end = h.now();\n        job.prefill_seconds =',
    '        std.debug.print("inkling trace: rollback wall_ms={d:.3}\\n", .{ms(@intCast(h.now() - t_roll))});\n'
    '        const end = h.now();\n        job.prefill_seconds =')
if "slot.snapshots[saved].?.copyRows(" in s:
    sub('                slot.snapshots[saved].?.copyRows(',
        '                const t_snap = h.now();\n                defer std.debug.print("inkling trace: snapshot at={d} wall_ms={d:.3}\\n", .{ at, ms(@intCast(h.now() - t_snap)) });\n'
        '                slot.snapshots[saved].?.copyRows(')
open(path, "w").write(s)
print("patched", path)
