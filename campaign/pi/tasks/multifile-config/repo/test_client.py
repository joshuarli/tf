import client

c = client.Client()
assert c.describe() == "timeout=10 retries=2"
assert client.Client(timeout_s=3).describe() == "timeout=3 retries=2"
print("ok")
