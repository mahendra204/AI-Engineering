import time
from pathlib import Path
from .config import POLL_INTERVAL

def run_poll_loop(callback, stop_after=None):
    """Run a simple polling loop calling callback() every POLL_INTERVAL seconds.
    If stop_after is set (seconds) the loop will end after that duration (useful for tests).
    """
    start = time.time()
    while True:
        callback()
        if stop_after is not None and (time.time() - start) > stop_after:
            break
        time.sleep(POLL_INTERVAL)
