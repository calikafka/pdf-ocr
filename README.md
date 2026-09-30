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
short_description: PDF to text, with OCR only when the PDF is a scan
---

# PDF OCR

Upload a PDF, get its text.

1. **Text layer first.** `pdftotext -layout` reads the PDF's embedded text. For born-digital PDFs this is exact and takes a second.
2. **OCR only when needed.** If that yields under ~100 characters per page (a scanned PDF), or you tick *Force OCR*, the file goes through [OCRmyPDF](https://ocrmypdf.readthedocs.io) with Tesseract (auto-rotate, deskew). You get the OCR text plus a searchable PDF.

The status line says which path produced the text.

Adapted from the extraction step of Seek, a research agent that reads a lot of scanned papers.

## Run it yourself

```bash
# Debian/Ubuntu
sudo apt install ocrmypdf poppler-utils tesseract-ocr
# macOS
brew install ocrmypdf poppler tesseract
pip install -r requirements.txt
python app.py
```

Or duplicate this Space. Add languages by adding `tesseract-ocr-<code>` lines to `packages.txt`.

Limits: 50 MB, 60 pages. Tesseract works well on clean printed scans and less well on handwriting, dense tables and complex multi-column layouts.
