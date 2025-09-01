#!/usr/bin/env python3
import os, sys, subprocess

# Ensure the redis client is installed
try:
    import redis  # type: ignore
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "redis"])
    import redis  # type: ignore

# Read the URL from argv or REDIS_URL
url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("REDIS_URL")
if not url:
    print("Usage:\n  REDIS_URL=rediss://... python test_redis.py\n  or\n  python test_redis.py rediss://...")
    sys.exit(1)

r = redis.from_url(url, decode_responses=True)  # rediss:// works automatically
print("PING ->", r.ping())
r.set("hello", "world", ex=60)
print("GET hello ->", r.get("hello"))
