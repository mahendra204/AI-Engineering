import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RESUME_PATH = Path(os.getenv("RESUME_PATH", BASE_DIR / "resume.txt"))
SAMPLE_JOBS_DIR = Path(os.getenv("SAMPLE_JOBS_DIR", BASE_DIR / "sample_jobs"))
APPLICATIONS_FILE = Path(os.getenv("APPLICATIONS_FILE", BASE_DIR / "applications.json"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))  # seconds
