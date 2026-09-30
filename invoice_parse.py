#!/usr/bin/env python3

"""Extract structured invoice data from a PDF and cross-check it with AADE.

Pipeline
--------
1. Render the PDF pages to images and send them to a vision LLM (Qwen via
   OpenRouter) which extracts the accounting fields as JSON.
2. Decode the myDATA QR code locally (free, no LLM cost).
3. Fetch the authoritative record from the QR destination and cross-check it
   against the extracted fields (see tax_office.py).

stdout carries only the resulting JSON, so the CLI composes well with pipes.
"""

import argparse
import logging
import sys
from pathlib import Path

from llm import extract_invoice_fields
from models import InvoiceDocument
from pdf_utils import extract_qr_url, pdf_to_data_urls
from tax_office import validate_against_tax_office

logger = logging.getLogger(__name__)


def extract_invoice(pdf_path: Path) -> InvoiceDocument:
    """Extract an invoice from ``pdf_path`` and cross-check it with AADE."""
    logger.info("Parsing invoice: %s", pdf_path)

    # Decode the myDATA QR locally: free and more accurate than the model.
    qr_url = extract_qr_url(pdf_path)

    if qr_url:
        logger.info("QR URL: %s", qr_url)
    else:
        logger.warning("No QR code found in the PDF.")

    image_urls = pdf_to_data_urls(pdf_path)

    invoice = extract_invoice_fields(image_urls)

    invoice.qr_url = qr_url

    invoice.validation_with_tax_office = validate_against_tax_office(
        invoice,
        qr_url,
    )

    return invoice


def _configure_logging(verbose: bool) -> None:
    """Send logs to stderr; --verbose enables DEBUG, otherwise WARNING."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    logger.addHandler(handler)
    logger.propagate = False


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="invoice-parse",
        description=(
            "Extract structured invoice data from a PDF using Qwen "
            "via OpenRouter."
        ),
    )

    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="PDF invoice to parse",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information to stderr.",
    )

    args = parser.parse_args()

    _configure_logging(verbose=args.verbose)

    if not args.file.exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(2)

    if args.file.suffix.lower() != ".pdf":
        print("Currently only PDF files are supported.", file=sys.stderr)
        sys.exit(2)

    try:
        invoice = extract_invoice(args.file)

        # stdout carries only the JSON, so the CLI composes well with pipes.
        print(
            invoice.model_dump_json(
                indent=2,
                exclude_none=False,
            )
        )

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
