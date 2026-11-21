"""Retry policy for the ingest queue."""

QUEUES = {
    "ingest": {"workers": 4, "retry_limit": 3, "backoff_s": 30, "dead_letter": True},
    "backfill": {"workers": 1, "retry_limit": 8, "backoff_s": 300, "dead_letter": True},
    "webhook": {"workers": 2, "retry_limit": 0, "backoff_s": 0, "dead_letter": False},
}


class RetryPolicy:
    def __init__(self, queue):
        settings = QUEUES[queue]
        self.workers = settings["workers"]
        self.limit = settings["retry_limit"]
        self.backoff_s = settings["backoff_s"]
        self.dead_letter = settings["dead_letter"]

    def should_retry(self, attempts):
        return attempts < self.limit

    def delay_for(self, attempts):
        return self.backoff_s * (2 ** max(0, attempts - 1))

    def give_up(self, attempts, queue):
        return not self.should_retry(attempts) and QUEUES[queue]["dead_letter"]


def schedule(queue, attempts):
    policy = RetryPolicy(queue)
    if not policy.should_retry(attempts):
        return "dead-letter" if policy.dead_letter else "drop"
    return f"{queue}: retry in {policy.delay_for(attempts)}s"
