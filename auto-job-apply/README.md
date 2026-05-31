# Auto Job Apply (Prototype)

Prototype framework to monitor job postings and auto-apply when resume similarity >= threshold.

Quick start:

1. Create a resume text file at `auto-job-apply/resume.txt`.
2. Install deps: `pip install -r auto-job-apply/requirements.txt`.
3. Run: `python auto-job-apply/autojobapply.py`.

This prototype uses local sample job postings in `auto-job-apply/sample_jobs/`.
You can use a PDF resume: place it at `auto-job-apply/resume.pdf` or set the `RESUME_PATH` env var.
Install PDF support with `pip install PyPDF2`.

LinkedIn automation notes:
- Fully automated applying (auto-submit) requires browser automation (Playwright/Selenium), handling login, and the user's explicit consent; it may violate LinkedIn's Terms of Service — use with caution.
- Safer options: monitor and auto-open pre-filled application pages for manual review, or notify you when matches are found.
- If you want full LinkedIn integration, I can scaffold a Playwright-based module, but you'll need to run `playwright install` and supply login credentials securely at runtime.

Extend `fetcher.py` to add real scrapers or implement `linkedin.py` using Playwright.
