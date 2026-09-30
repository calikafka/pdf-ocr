"""
PDF → text, reading the text layer when there is one and OCR'ing only when there isn't.

Adapted from Seek's extraction path (seek_extract.py): run `pdftotext -layout`
first; if that yields near-empty text relative to the page count (a scanned
PDF), fall back to OCR — here via ocrmypdf, which also returns a searchable PDF.
The result says which path produced the text.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import gradio as gr

MAX_MB = 50
MAX_PAGES = 60
PDFTOTEXT_TIMEOUT = 120   # seconds
OCR_TIMEOUT = 600         # seconds

# Chars of extracted text per page below which we assume a scanned PDF.
MIN_CHARS_PER_PAGE = 100

LANG_NAMES = {
    "eng": "English", "fra": "French", "deu": "German", "spa": "Spanish",
    "ita": "Italian", "por": "Portuguese", "nld": "Dutch", "lat": "Latin",
}


class ExtractError(RuntimeError):
    pass


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ExtractError(f"{cmd[0]} timed out after {timeout} s")


def installed_langs() -> list[str]:
    try:
        out = _run(["tesseract", "--list-langs"], timeout=20).stdout
    except (ExtractError, FileNotFoundError):
        return ["eng"]
    langs = [l.strip() for l in out.splitlines()[1:] if l.strip()]
    return [l for l in langs if l not in ("osd", "snum")] or ["eng"]


def page_count(pdf: Path) -> int:
    proc = _run(["pdfinfo", str(pdf)], timeout=PDFTOTEXT_TIMEOUT)
    if proc.returncode != 0:
        raise ExtractError("could not read the PDF (is it damaged or encrypted?)")
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split()[1])
    raise ExtractError("could not determine the page count")


def pdftotext(pdf: Path, txt: Path) -> str:
    proc = _run(["pdftotext", "-layout", str(pdf), str(txt)], timeout=PDFTOTEXT_TIMEOUT)
    if proc.returncode != 0:
        raise ExtractError(f"pdftotext failed: {proc.stderr.strip()[:300]}")
    return txt.read_text(errors="replace")


def ocr(pdf: Path, out_pdf: Path, sidecar: Path, langs: str) -> str:
    proc = _run(
        ["ocrmypdf", "--force-ocr", "--rotate-pages", "--deskew",
         "--language", langs, "--jobs", "2",
         "--sidecar", str(sidecar), str(pdf), str(out_pdf)],
        timeout=OCR_TIMEOUT,
    )
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-3:]
        raise ExtractError("ocrmypdf failed: " + " / ".join(tail)[:400])
    # ocrmypdf separates pages in the sidecar with form feeds.
    return sidecar.read_text(errors="replace").replace("\f", "\n\n")


def extract(file_path: str | None, langs: list[str], force_ocr: bool):
    if not file_path:
        raise gr.Error("Upload a PDF first.")
    src = Path(file_path)
    if src.stat().st_size > MAX_MB * 1024 * 1024:
        raise gr.Error(f"File is over the {MAX_MB} MB limit.")
    with src.open("rb") as f:
        if f.read(5) != b"%PDF-":
            raise gr.Error("That file isn't a PDF.")

    work = Path(tempfile.mkdtemp(prefix="ocr-"))
    pdf = work / "input.pdf"
    shutil.copyfile(src, pdf)
    stem = src.stem or "document"

    try:
        pages = page_count(pdf)
        if pages > MAX_PAGES:
            raise gr.Error(f"{pages} pages is over the {MAX_PAGES}-page limit.")

        text = pdftotext(pdf, work / "layer.txt")
        chars_per_page = len(text.strip()) / max(pages, 1)
        searchable = None

        if force_ocr or chars_per_page < MIN_CHARS_PER_PAGE:
            reason = ("OCR forced" if force_ocr else
                      f"text layer too thin ({chars_per_page:.0f} chars/page)")
            searchable = work / f"{stem}-searchable.pdf"
            text = ocr(pdf, searchable, work / "sidecar.txt", "+".join(langs or ["eng"]))
            status = f"**Method: OCR** ({reason}; languages: {'+'.join(langs or ['eng'])}) · {pages} pages"
        else:
            status = (f"**Method: text layer** (pdftotext, {chars_per_page:.0f} chars/page, "
                      f"no OCR needed) · {pages} pages")

        txt_out = work / f"{stem}.txt"
        txt_out.write_text(text)
        return status, text, str(txt_out), (str(searchable) if searchable else None)
    except ExtractError as e:
        raise gr.Error(str(e))


LANGS = installed_langs()

with gr.Blocks(title="PDF OCR") as demo:
    gr.Markdown(
        "# PDF → text\n"
        "Reads the PDF's own text layer when it has one (fast and exact). "
        "If the PDF is a scan, runs OCR with [OCRmyPDF](https://ocrmypdf.readthedocs.io) "
        "and Tesseract, and also gives you back a searchable PDF.\n\n"
        f"Limits: {MAX_MB} MB, {MAX_PAGES} pages. "
        "Uploaded files are processed on Hugging Face's servers and are not kept after the request."
    )
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="PDF", file_types=[".pdf"], type="filepath")
            langs_in = gr.Dropdown(
                choices=[(LANG_NAMES.get(l, l), l) for l in LANGS],
                value=["eng"] if "eng" in LANGS else LANGS[:1],
                multiselect=True, label="OCR language(s)",
            )
            force_in = gr.Checkbox(label="Force OCR (ignore the existing text layer)")
            go = gr.Button("Extract text", variant="primary")
        with gr.Column(scale=2):
            status_out = gr.Markdown()
            text_out = gr.Textbox(label="Text", lines=20, max_lines=40, buttons=["copy"])
            with gr.Row():
                txt_file_out = gr.File(label="Text file")
                pdf_file_out = gr.File(label="Searchable PDF (OCR only)")

    go.click(extract, [file_in, langs_in, force_in],
             [status_out, text_out, txt_file_out, pdf_file_out])

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch()
