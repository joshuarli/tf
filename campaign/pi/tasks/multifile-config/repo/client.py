import settings


class Client:
    def __init__(self, timeout_s=None, retries=None):
        self.timeout_s = timeout_s if timeout_s is not None else settings.DEFAULT_TIMEOUT_S
        self.retries = retries if retries is not None else settings.DEFAULT_RETRIES

    def describe(self):
        return f"timeout={self.timeout_s} retries={self.retries}"
