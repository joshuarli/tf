def clamp_percent(value):
    if value is None:
        return 0
    if value < 0:
        return 0
    if value > 100:
        return 100
    return value


def summarize(scores):
    total = 0
    for score in scores:
        total = total + clamp_percent(score)
    if len(scores) == 0:
        return {"count": 0, "total": 0, "mean": 0}
    return {"count": len(scores), "total": total, "mean": total / len(scores)}


def grade_bucket(mean):
    if mean >= 90:
        return "A"
    if mean >= 80:
        return "B"
    if mean >= 70:
        return "C"
    return "F"
