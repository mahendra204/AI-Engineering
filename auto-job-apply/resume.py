from pathlib import Path

def load_resume_text(path: Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resume not found: {path}")
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    # For convenience, support minimal PDF extraction if PyPDF2 is installed
    try:
        if path.suffix.lower() == ".pdf":
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
    except Exception:
        pass
    raise ValueError("Unsupported resume format; provide a .txt or install PyPDF2 for PDF support")
