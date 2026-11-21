def dedupe(items):
    seen = []
    out = []
    for item in items:
        if item in seen:
            continue
        out.append(item)
    return out
