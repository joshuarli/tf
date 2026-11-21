"""Rolling window statistics."""


def rolling_max(values, window):
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    for i in range(len(values)):
        out.append(max(values[i:i + window]))
    return out


def rolling_mean(values, window):
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    for i in range(len(values)):
        chunk = values[i:i + window]
        out.append(sum(chunk) / window)
    return out
