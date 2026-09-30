"""PDF rendering and local myDATA QR-code decoding.

This module turns invoice PDFs into images for the vision model and reads the
myDATA QR code (which carries the verification URL) entirely locally — no LLM
tokens are spent on it.
"""

import base64
import logging
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ~144 DPI — enough for normal invoices while keeping the LLM cost reasonable.
RENDER_SCALE = 2.0

# Render scales tried while decoding the myDATA QR code locally. Higher scales
# cost nothing (no LLM involved) and help tiny, dense QR codes.
QR_RENDER_SCALES = (3.0, 4.0, 5.0)


def pdf_to_data_urls(pdf_path: Path) -> list[str]:
    """Render every PDF page to a PNG data URL for the vision model."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Cannot open PDF: {exc}") from exc

    if doc.needs_pass:
        raise RuntimeError(
            "Encrypted/password-protected PDFs are not supported."
        )

    if doc.page_count == 0:
        raise RuntimeError("PDF has no pages.")

    logger.debug(
        "Rendering %d page(s) at scale %.1f",
        doc.page_count,
        RENDER_SCALE,
    )

    images = []
    matrix = fitz.Matrix(RENDER_SCALE, RENDER_SCALE)

    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        image_bytes = pix.tobytes("png")
        encoded = base64.b64encode(image_bytes).decode("ascii")

        images.append(f"data:image/png;base64,{encoded}")

    doc.close()

    return images


def extract_qr_url(pdf_path: Path) -> str | None:
    """Decode the myDATA QR code from the PDF locally.

    Greek invoices carry a QR code containing the verification URL. Reading it
    with a barcode decoder is free and more accurate than asking the model to
    read it from a picture. Returns the decoded URL, or ``None`` if no QR code
    is found.
    """
    # Imported lazily so the CLI still runs without the QR decoder installed.
    import numpy as np
    import zxingcpp

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return None

    try:
        if doc.needs_pass or doc.page_count == 0:
            return None

        for scale in QR_RENDER_SCALES:
            matrix = fitz.Matrix(scale, scale)

            logger.debug("Trying QR decode at scale %.1f", scale)

            for page in doc:
                pix = page.get_pixmap(
                    matrix=matrix,
                    colorspace=fitz.csGRAY,
                    alpha=False,
                )

                img = np.frombuffer(
                    pix.samples,
                    dtype=np.uint8,
                ).reshape(pix.height, pix.width)

                texts = [
                    result.text
                    for result in zxingcpp.read_barcodes(img)
                    if result.format == zxingcpp.BarcodeFormat.QRCode
                    and result.text
                ]

                if texts:
                    # Prefer an actual URL, fall back to the raw payload.
                    url = next(
                        (
                            text
                            for text in texts
                            if text.startswith(("http://", "https://"))
                        ),
                        texts[0],
                    )

                    logger.info("Decoded QR code at scale %.1f", scale)

                    return url
    finally:
        doc.close()

    return None
