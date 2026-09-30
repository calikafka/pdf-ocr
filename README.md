---
title: PDF OCR
emoji: 📄
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: "6.29.0"
app_file: app.py
pinned: false
license: mit
short_description: PDF to text; OCR for scans, local VLM for hard pages
---

# PDF OCR

PDF → text, mostly with ordinary local tools, with a local vision model only for
the pages those tools struggle with. Nothing is sent to a cloud API.

1. **Text layer first.** `pdftotext -layout` reads the PDF's embedded text.
   For born-digital PDFs this is exact and takes about a second.
2. **OCR for scans.** If that yields under ~100 characters per page, the PDF is
   treated as a scan and OCR'd with `ocrmypdf` (or plain `tesseract` page by page
   if ocrmypdf isn't installed).
3. **Vision model for hard pages (optional).** Tesseract scores its confidence on
   every page. Pages below a threshold (default 80/100), or with almost no
   recognised words, are re-read by a vision model running locally in
   [Ollama](https://ollama.com). The output says which pages it handled.

## Files

| File | What it is |
|---|---|
| `seek_extract.py` | Steps 1–2. The extraction module from Seek, a research agent that reads many scanned papers, with only its Seek-specific parts removed. Caches each PDF and its text by SHA-256 and logs every extraction to `cache/sources/ledger.jsonl`. Works on its own from the command line. |
| `vlm_pages.py` | Step 3. Off unless `OCR_VLM_MODEL` is set and that model is pulled in Ollama. |
| `app.py` | A small Gradio web UI over both. |

## Run it on your own machine

**1. System tools** (poppler for `pdftotext`, plus OCR):

```bash
# macOS
brew install poppler tesseract ocrmypdf
# Debian / Ubuntu
sudo apt install poppler-utils tesseract-ocr ocrmypdf
```

**2. Python packages** (Python 3.10+):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**3. Optional: a local vision model for hard pages.** Install
[Ollama](https://ollama.com/download), then:

```bash
ollama pull glm-ocr               # 2.2 GB
export OCR_VLM_MODEL=glm-ocr
```

[GLM-OCR](https://ollama.com/library/glm-ocr) is a small (0.9B) model made only
for document OCR. Any other Ollama vision model also works (set
`OCR_VLM_PROMPT` if it needs a particular prompt), but avoid *thinking* models:
the plain `qwen3-vl:8b` tag was about as accurate on our test page and ~18x
slower, because it reasons at length before transcribing.

**4. Run it:**

```bash
python app.py                                  # web UI at http://127.0.0.1:7860
python seek_extract.py paper.pdf               # CLI: prints the cached .txt path
python seek_extract.py https://example.org/paper.pdf
python vlm_pages.py scanned.pdf                # CLI: extract + vision pass, prints text
```

## Settings

| Variable | Default | Purpose |
|---|---|---|
| `OCR_VLM_MODEL` | unset (step off) | Ollama model for hard pages |
| `OCR_VLM_PROMPT` | per model | Override the prompt sent with each page image |
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama is running |
| `SEEK_EXTRACT_CACHE` | `./cache/sources` | Cache and ledger location |
| `OCR_EXTRA_BIN` | unset | Extra directories to search for `pdftotext`/`tesseract`/`ocrmypdf` |

## Notes

- Scoring page confidence means Tesseract reads each scanned page a second time,
  and the vision model takes tens of seconds per page on a laptop. That's why
  step 3 only ever runs on scans, and only on the pages that need it.
- Tesseract is good on clean printed scans; the vision model helps most with
  tables, multi-column layouts, poor scans and handwriting.
- Text extracted from a PDF, and anything the vision model returns, is treated
  as data, never as instructions.
- The hosted Hugging Face Space runs steps 1–2 only, and deletes uploads after
  each request.
