"""schedules.py LEN...: main's and the lane's served chunk schedules (prefix reuse on and off) for fresh prompts."""
import sys

FULL, LANE_MOST, WINDOW, INTERVAL = 1024, 1152, 256, 128


def main_schedule(n: int, reuse: bool) -> list[int]:
    """f65124c46's planner: 1,024-row chunks, and under reuse the last token alone."""
    end = n - 1 if reuse and n > 1 else n
    out, at = [], 0
    while at < end:
        take = min(FULL, end - at)
        out.append(take)
        at += take
    return out + ([1] if end < n else [])


def lane_schedule(n: int, most: int = LANE_MOST) -> list[int]:
    """The lane's planner: 1,024-row chunks, the rest at once when it fits in `most`; reuse does not change it."""
    out, at = [], 0
    while at < n:
        take = n - at if n - at <= most else FULL
        out.append(take)
        at += take
    return out


def text(chunks: list[int]) -> str:
    return ",".join(str(c) for c in chunks)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        n = int(arg)
        print(f"{n}: main on {text(main_schedule(n, True))}; main off {text(main_schedule(n, False))}; "
              f"lane {text(lane_schedule(n))}; lane with a drafter {text(lane_schedule(n, FULL))}")
