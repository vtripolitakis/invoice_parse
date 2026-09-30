"""OpenRouter / Qwen client for extracting invoices from page images.

This module builds the multimodal request, enforces the response JSON schema,
and parses the model's reply into an :class:`InvoiceDocument`.
"""

import json
import logging
import os
import sys

import requests

from .models import InvoiceDocument

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3.5-35b-a3b"

PROMPT = """
Extract the accounting document from the attached page images.

The document may be Greek or English and may be:
- an invoice
- a credit note
- a receipt
- another accounting document

Rules:

1. Extract only information actually visible in the document.
2. Never guess or invent missing values.
3. If a value is missing or unreadable, return null.
4. Supplier means the issuer/seller of the document.
5. Customer means the recipient/buyer.
6. VAT/tax numbers must remain strings.
7. Invoice/document numbers must remain strings.
8. Convert dates to YYYY-MM-DD only when the date is unambiguous.
   On Greek documents the date is day-first: DDMMYY or DDMMYYYY,
   optionally separated by / or - (09/10/2026 means 9 October 2026).
9. Monetary amounts must be JSON numbers, without currency symbols.
10. vat_rate must be a percentage number, e.g. 24, not 0.24.
11. Extract all visible line items.
12. Do not include subtotal/total rows as line items.
13. Extract myDATA MARK only if explicitly printed on the document.
    It is a 15-digit number (e.g. 400015192863370), never a word or label.
14. Do not calculate missing values.
15. Add a warning for anything important that is ambiguous or unreadable.

The images are pages of the same document, in page order.
"""


def build_content(image_urls: list[str]) -> list[dict]:
    """Build the multimodal message content for the vision model."""
    content: list[dict] = [
        {
            "type": "text",
            "text": PROMPT,
        }
    ]

    for image_url in image_urls:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
        )

    return content


def build_schema() -> dict:
    """Return the JSON schema to enforce, minus locally-populated fields.

    ``qr_url`` and ``validation_with_tax_office`` are filled in locally (not by
    the model), so they are removed to avoid wasting tokens or letting the
    model hallucinate them.
    """
    schema = InvoiceDocument.model_json_schema()

    for local_field in ("qr_url", "validation_with_tax_office"):
        schema.get("properties", {}).pop(local_field, None)

        required = schema.get("required", [])
        if local_field in required:
            required.remove(local_field)

    return schema


def extract_message_content(data: dict) -> str:
    """Pull the assistant message text out of an OpenRouter response."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Unexpected OpenRouter response: {json.dumps(data)}"
        ) from exc

    # Normally a string; handle array-style responses defensively.
    if isinstance(content, list):
        parts = [
            part["text"]
            for part in content
            if isinstance(part, dict) and "text" in part
        ]
        content = "".join(parts)

    return content


def log_cost(usage: dict | None) -> None:
    """Print the OpenRouter cost to stderr (always, even without --verbose)."""
    if not usage:
        return

    cost = usage.get("cost")

    if cost is not None:
        print(
            f"OpenRouter cost: ${cost:.6f}",
            file=sys.stderr,
        )


def extract_invoice_fields(image_urls: list[str]) -> InvoiceDocument:
    """Send the page images to the LLM and return the parsed document.

    Requires the ``OPENROUTER_API_KEY`` environment variable.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set."
        )

    payload = {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": build_content(image_urls),
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "invoice_document",
                "strict": True,
                "schema": build_schema(),
            },
        },

        # Don't silently route to a provider that ignores structured-output
        # parameters.
        "provider": {
            "require_parameters": True,
        },
    }

    logger.info(
        "Calling OpenRouter model %s with %d page image(s)",
        MODEL,
        len(image_urls),
    )

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "invoice-parse-poc",
        },
        json=payload,
        timeout=180,
    )

    if not response.ok:
        raise RuntimeError(
            f"OpenRouter HTTP {response.status_code}: "
            f"{response.text[:2000]}"
        )

    logger.debug("OpenRouter responded with HTTP %d", response.status_code)

    data = response.json()

    invoice = InvoiceDocument.model_validate_json(
        extract_message_content(data)
    )

    log_cost(data.get("usage"))

    return invoice
