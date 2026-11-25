"""schedules.py LEN...: the host's chunk schedule for fresh prompts of LEN tokens, main's and the lane's, as 1,024-row
chunks a scratch allows, fork window 256 and interval 128 (`inkling2_loop_host.zig` on each side, prefix reuse on)."""
import sys

CHUNK, WINDOW, INTERVAL = 1024, 256, 128


def first_point(n, at):
    first = (max(n - WINDOW, 0) + INTERVAL - 1) // INTERVAL * INTERVAL
    return max(INTERVAL, first, (at + INTERVAL - 1) // INTERVAL * INTERVAL)


def main_schedule(n, at=0):
    nxt, out = first_point(n, at), []
    while at < n:
        if nxt == at and nxt < n:
            nxt += INTERVAL
        remaining = n - 1 - at if at < n - 1 else n - at
        take = min(CHUNK, remaining, nxt - at if nxt > at else remaining)
        out.append(take)
        at += take
    return out


def lane_schedule(n, at=0):
    out = []
    while at < n:
        take = min(CHUNK, (n - 1 if at + 1 < n else n) - at)
        out.append(take)
        at += take
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        n = int(arg)
        print(f"{n}: main {','.join(map(str, main_schedule(n)))}  lane {','.join(map(str, lane_schedule(n)))}")
