import json
from pathlib import Path
from pprint import pprint

from config import RESUME_PATH, SAMPLE_JOBS_DIR, APPLICATIONS_FILE, SIMILARITY_THRESHOLD
from resume import load_resume_text
from fetcher import fetch_jobs_from_folder
from matcher import score_similarity
from applier import apply_to_job

APPLIED_CACHE = set()

def load_applied(applications_file: Path):
    if applications_file.exists():
        try:
            data = json.loads(applications_file.read_text(encoding='utf-8'))
            for e in data:
                APPLIED_CACHE.add(e.get('job_id'))
        except Exception:
            pass

def process_once():
    resume_text = load_resume_text(Path(RESUME_PATH))
    jobs = fetch_jobs_from_folder(Path(SAMPLE_JOBS_DIR))
    for job in jobs:
        jid = job.get('id')
        if jid in APPLIED_CACHE:
            continue
        job_text = job.get('description', '') + '\n' + job.get('title', '')
        score = score_similarity(resume_text, job_text)
        print(f"Checked job {jid}: score={score:.3f}")
        if score >= SIMILARITY_THRESHOLD:
            apply_to_job(job, score, Path(RESUME_PATH), Path(APPLICATIONS_FILE))
            APPLIED_CACHE.add(jid)

def main():
    load_applied(Path(APPLICATIONS_FILE))
    print("Starting auto-job-apply (prototype). Press Ctrl+C to stop.")
    try:
        # simple loop
        while True:
            process_once()
            import time
            time.sleep(60)
    except KeyboardInterrupt:
        print("Stopping")

if __name__ == '__main__':
    main()
