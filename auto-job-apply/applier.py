import json
from pathlib import Path
from datetime import datetime
from typing import Dict

def record_application(applications_file: Path, job: Dict, score: float, resume_path: Path):
    applications_file = Path(applications_file)
    applications = []
    if applications_file.exists():
        try:
            applications = json.loads(applications_file.read_text(encoding='utf-8'))
        except Exception:
            applications = []
    entry = {
        'job_id': job.get('id'),
        'title': job.get('title'),
        'company': job.get('company'),
        'score': score,
        'resume': str(resume_path),
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    applications.append(entry)
    applications_file.write_text(json.dumps(applications, indent=2), encoding='utf-8')

def apply_to_job(job: Dict, score: float, resume_path: Path, applications_file: Path):
    # Placeholder: real automation would use Selenium/Playwright to fill forms.
    print(f"[APPLY] Would apply to job {job.get('id')} ({job.get('title')}) with score={score:.2f}")
    record_application(applications_file, job, score, resume_path)
