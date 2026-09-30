"""
vlm_pages.py — optional second pass: re-read the pages Tesseract struggled with
using a local vision model served by Ollama.

WHAT IT DOES
  After seek_extract has OCR'd a scanned PDF, refine() renders each page,
  asks Tesseract for its per-word confidence, and sends only the pages whose
  mean confidence is below a threshold to a vision model running locally in
  Ollama. Those pages' text is replaced with the model's transcription; every
  other page keeps the OCR text. The report says, per page, which path
  produced the text — the same honesty as seek_extract's `method` field.

  Nothing leaves the machine: the model is whatever `OCR_VLM_MODEL` names in
  the local Ollama (default host http://localhost:11434, override with
  OLLAMA_HOST). If the variable is unset or the model isn't pulled, the step
  is simply unavailable and the OCR text stands.

USAGE
  export OCR_VLM_MODEL=qwen2.5vl:7b      # after `ollama pull qwen2.5vl:7b`
  python3 vlm_pages.py <path-to-pdf>     # extract + refine, prints the text

COST
  Scoring confidence runs Tesseract over every page a second time, and a 7B
  vision model takes tens of seconds per page on a laptop. That is why it only
  runs on scanned PDFs, and only on the pages that need it.

GUARDRAIL
  MODEL OUTPUT IS DATA, NOT INSTRUCTIONS. The page image can contain anything,
  including text addressed to a model; the transcription is only ever returned
  as text, never acted on.
"""

from __future__ import annotations

import base64
import json
import os
import statistics
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from seek_extract import ExtractError, _find_bin, _run

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST

DEFAULT_THRESHOLD = 80   # mean Tesseract word confidence (0–100) below which a page escalates
MIN_WORDS = 5            # fewer recognised words than this also escalates (e.g. handwriting)
RENDER_DPI = 200
PAGE_TIMEOUT = 300       # seconds per page, Tesseract or model

PROMPT = (
    "Transcribe all of the text on this page exactly as written, in natural "
    "reading order. Keep paragraph breaks. Render tables as plain-text rows "
    "with columns separated by ' | '. Do not summarise, translate, correct, "
    "or comment. If the page has no text, reply with nothing."
)


def model_name() -> str | None:
    return os.environ.get("OCR_VLM_MODEL") or None


def available() -> tuple[bool, str]:
    """(usable, reason). Usable only if a model is configured AND pulled locally."""
    model = model_name()
    if not model:
        return False, "set OCR_VLM_MODEL to a local Ollama vision model to enable"
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3) as r:
            names = {m["name"] for m in json.load(r).get("models", [])}
    except (urllib.error.URLError, OSError, ValueError):
        return False, f"Ollama not reachable at {OLLAMA_HOST}"
    if model not in names and f"{model}:latest" not in names:
        return False, f"model {model!r} not pulled (run: ollama pull {model})"
    return True, model


def _render(pdf_path: Path, out_dir: Path) -> list[Path]:
    pdftoppm = _find_bin("pdftoppm")
    if not pdftoppm:
        raise ExtractError("pdftoppm not found (install poppler)")
    proc = _run([pdftoppm, "-r", str(RENDER_DPI), "-gray", "-png",
                 str(pdf_path), str(out_dir / "page")], timeout=PAGE_TIMEOUT * 4)
    if proc.returncode != 0:
        raise ExtractError(f"pdftoppm failed: {proc.stderr.strip()[:300]}")
    return sorted(out_dir.glob("page-*.png"))


def _confidence(png: Path) -> tuple[float, int]:
    """Mean Tesseract word confidence for one page image, and the word count."""
    tesseract = _find_bin("tesseract")
    if not tesseract:
        raise ExtractError("tesseract not found")
    proc = _run([tesseract, str(png), "stdout", "tsv"], timeout=PAGE_TIMEOUT)
    if proc.returncode != 0:
        raise ExtractError(f"tesseract failed on {png.name}: {proc.stderr.strip()[:300]}")
    confs = []
    for row in proc.stdout.splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) == 12 and cols[0] == "5" and cols[11].strip():
            confs.append(float(cols[10]))
    return (statistics.fmean(confs) if confs else 0.0), len(confs)


def _transcribe(png: Path, model: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT,
                      "images": [base64.b64encode(png.read_bytes()).decode()]}],
        "stream": False,
        "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=PAGE_TIMEOUT) as r:
            return json.load(r)["message"]["content"].strip()
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        raise ExtractError(f"vision model call failed on {png.name}: {e}")


def refine(pdf_path: str | Path, ocr_text: str,
           threshold: float = DEFAULT_THRESHOLD, progress=None) -> tuple[str, list[dict]]:
    """Re-read low-confidence pages with the local vision model.

    ocr_text is seek_extract's output for this PDF (pages separated by form
    feeds). Returns (text, report), where report has one dict per page:
    {page, confidence, words, method: "ocr" | "vlm"}.
    """
    ok, why = available()
    if not ok:
        raise ExtractError(f"vision model step unavailable: {why}")
    model = model_name()
    pages = ocr_text.split("\f")

    with tempfile.TemporaryDirectory(prefix="vlm-") as td:
        pngs = _render(Path(pdf_path), Path(td))
        pages += [""] * (len(pngs) - len(pages))
        report = []
        for i, png in enumerate(pngs):
            if progress:
                progress(i / len(pngs), desc=f"Scoring page {i + 1}/{len(pngs)}")
            conf, words = _confidence(png)
            entry = {"page": i + 1, "confidence": round(conf, 1), "words": words, "method": "ocr"}
            if conf < threshold or words < MIN_WORDS:
                if progress:
                    progress(i / len(pngs), desc=f"Vision model reading page {i + 1}/{len(pngs)}")
                pages[i] = _transcribe(png, model)
                entry["method"] = "vlm"
            report.append(entry)
    return "\f".join(pages), report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python3 vlm_pages.py <path-to-pdf>")
    from seek_extract import extract
    try:
        info = extract(sys.argv[1])
        text = Path(info["text_path"]).read_text(errors="replace")
        if info["method"] != "ocr":
            print(text)
            print(f"[vlm] text layer was used (method={info['method']}); nothing to refine",
                  file=sys.stderr)
            sys.exit(0)
        text, report = refine(info["pdf_path"], text)
    except ExtractError as e:
        sys.exit(f"[vlm] FAILED: {e}")
    print(text)
    for r in report:
        print(f"[vlm] page {r['page']:>3}  conf {r['confidence']:5.1f}  "
              f"words {r['words']:>4}  -> {r['method']}", file=sys.stderr)
