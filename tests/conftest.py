import os

# must be set BEFORE app.main is imported (it builds the app at import time)
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("ALLOWED_ORIGINS", "https://me.vercel.app")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "100")
