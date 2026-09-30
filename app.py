"""
Gradio front end for seek_extract (text layer first, OCR for scans) plus the
optional local vision-model pass in vlm_pages.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# On a Hugging Face Space, keep the cache in scratch space, not the app dir.
ON_SPACE = bool(os.environ.get("SPACE_ID"))
if ON_SPACE:
    os.environ.setdefault("SEEK_EXTRACT_CACHE", str(Path(tempfile.gettempdir()) / "seek-extract"))

import gradio as gr

import vlm_pages
from seek_extract import CACHE_DIR, ExtractError, _page_count, extract

MAX_PAGES = 60  # app-level cap so one upload can't tie up a shared Space

VLM_OK, VLM_WHY = vlm_pages.available()


def run(file_path: str | None, use_vlm: bool, threshold: float,
        progress=gr.Progress()):
    if not file_path:
        raise gr.Error("Upload a PDF first.")
    pages = _page_count(Path(file_path))
    if pages > MAX_PAGES:
        raise gr.Error(f"{pages} pages is over the {MAX_PAGES}-page limit.")

    progress(0, desc="Extracting")
    try:
        info = extract(file_path)
        text = Path(info["text_path"]).read_text(errors="replace")
        status = (f"**Method: {info['method']}** · {info['page_count']} pages · "
                  f"sha256 `{info['sha256'][:12]}`")

        if use_vlm and VLM_OK and info["method"] == "ocr":
            text, report = vlm_pages.refine(info["pdf_path"], text, threshold, progress)
            redone = [r for r in report if r["method"] == "vlm"]
            if redone:
                status += (f"\n\n**Vision model ({vlm_pages.model_name()})** re-read "
                           f"{len(redone)} of {len(report)} pages: " +
                           ", ".join(f"p{r['page']} (conf {r['confidence']:.0f})" for r in redone))
            else:
                status += (f"\n\nAll pages scored ≥ {threshold:.0f} Tesseract confidence; "
                           "no vision-model pass needed.")
        elif use_vlm and VLM_OK and info["method"] != "ocr":
            status += "\n\nText layer was used, so there was nothing for the vision model to do."
    except ExtractError as e:
        raise gr.Error(str(e))
    finally:
        # On the shared Space, don't keep people's documents around.
        if ON_SPACE and "info" in locals():
            for k in ("pdf_path", "text_path"):
                Path(info[k]).unlink(missing_ok=True)

    text = text.replace("\f", "\n\n")
    out = Path(tempfile.mkdtemp()) / f"{Path(file_path).stem or 'document'}.txt"
    out.write_text(text)
    return status, text, str(out)


if VLM_OK:
    vlm_note = f"Local vision model available: **{vlm_pages.model_name()}** (via Ollama)."
elif ON_SPACE:
    vlm_note = ("The vision-model step runs only on your own machine. Duplicate or clone this "
                "Space, pull a vision model in [Ollama](https://ollama.com), and set "
                "`OCR_VLM_MODEL` (see the README).")
else:
    vlm_note = f"Vision-model step off: {VLM_WHY}."

with gr.Blocks(title="PDF OCR") as demo:
    gr.Markdown(
        "# PDF → text\n"
        "1. Reads the PDF's own text layer when it has one (`pdftotext`, fast and exact).\n"
        "2. If the PDF is a scan, runs OCR (`ocrmypdf` / Tesseract).\n"
        "3. Optionally, pages Tesseract was unsure about are re-read by a **local** vision model.\n\n"
        f"{vlm_note}\n\n"
        f"Limit: {MAX_PAGES} pages."
        + (" Uploads are processed on Hugging Face's servers and deleted after each request."
           if ON_SPACE else "")
    )
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="PDF", file_types=[".pdf"], type="filepath")
            vlm_in = gr.Checkbox(label="Re-read hard pages with the local vision model",
                                 value=VLM_OK, visible=VLM_OK)
            thr_in = gr.Slider(40, 95, value=vlm_pages.DEFAULT_THRESHOLD, step=5,
                               label="Escalate pages below this Tesseract confidence",
                               visible=VLM_OK)
            go = gr.Button("Extract text", variant="primary")
        with gr.Column(scale=2):
            status_out = gr.Markdown()
            text_out = gr.Textbox(label="Text", lines=20, max_lines=40, buttons=["copy"])
            file_out = gr.File(label="Text file")

    go.click(run, [file_in, vlm_in, thr_in], [status_out, text_out, file_out])

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch()
