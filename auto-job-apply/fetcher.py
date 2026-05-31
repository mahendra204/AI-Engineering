import json
from pathlib import Path
from typing import List, Dict

def fetch_jobs_from_folder(folder: Path) -> List[Dict]:
    folder = Path(folder)
    jobs = []
    if not folder.exists():
        return jobs
    for p in sorted(folder.glob('*.json')):
        try:
            with p.open('r', encoding='utf-8') as f:
                data = json.load(f)
                # ensure minimal fields
                if 'id' not in data:
                    data['id'] = p.stem
                jobs.append(data)
        except Exception:
            continue
    return jobs

def fetch_jobs_linkedin(*, max_results=10):
    raise NotImplementedError('LinkedIn scraper not implemented. Use Selenium/Playwright and handle login.')

def fetch_jobs_naukri(*, max_results=10):
    raise NotImplementedError('Naukri scraper not implemented. Use Selenium/Playwright and handle login.')
