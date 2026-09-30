"""OpenRouter / Qwen client for extracting invoices from page images.

This module builds the multimodal request, enforces the response JSON schema,
and parses the model's reply into an :class:`InvoiceDocument`.
"""

import copy
import json
import logging
import os
import sys
from typing import Any

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


def _make_strict_schema(schema: dict) -> dict:
    """Normalise a Pydantic JSON schema for strict structured outputs.

    Strict mode (OpenAI and others) requires that every object lists all of
    its properties in ``required``, rejects ``title``/``default`` and
    ``description``, and ``anyOf`` of ``{"type": "null"}``. It also mishandles
    unreferenced ``$defs``. We therefore inline every ``$ref`` into a
    self-contained schema and collapse nullable unions. This mutates a copy
    in place.
    """
    schema = copy.deepcopy(schema)

    # Inline $defs/$ref. This also drops unreferenced definitions such as
    # TaxOfficeValidation, which is excluded from the request schema.
    defs = schema.pop("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, list):
            return [inline(item) for item in node]

        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]

                if name in defs:
                    return inline(copy.deepcopy(defs[name]))

            return {key: inline(value) for key, value in node.items()}

        return node

    schema = inline(schema)

    def fix(node: object) -> None:
        if isinstance(node, list):
            for item in node:
                fix(item)
            return

        if not isinstance(node, dict):
            return

        # Collapse simple `anyOf` of {"type": X} / {"type": "null"} into a
        # union type, e.g. {"type": ["string", "null"]}.
        any_of = node.get("anyOf")
        if any_of:
            types = []
            nullable = False

            for option in any_of:
                if isinstance(option, dict) and set(option) == {"type"}:
                    if option["type"] == "null":
                        nullable = True
                    else:
                        types.append(option["type"])

            if types:
                node["type"] = types + (["null"] if nullable else [])
                node.pop("anyOf", None)

        # Every property must be required at each object level. Recurse into
        # the property schemas (never into the property-name mapping itself,
        # or a field literally named "description" would be stripped).
        if isinstance(node.get("properties"), dict):
            node["required"] = list(node["properties"].keys())

            for child in node["properties"].values():
                fix(child)

        # Recurse into any other schema-bearing children (items, ...).
        for key, value in node.items():
            if key != "properties":
                fix(value)

        # Unsupported annotation keys for OpenAI strict mode.
        for key in ("title", "default", "description"):
            node.pop(key, None)

    fix(schema)

    return schema


def build_schema() -> dict:
    """Return the JSON schema to enforce, minus locally-populated fields.

    ``qr_url``, ``validation_with_tax_office``, ``model`` and ``cost_usd`` are
    filled in locally (not by the model), so they are removed to avoid wasting
    tokens or letting the model hallucinate them. The result is normalised to
    the strict structured-outputs JSON Schema subset.
    """
    schema = InvoiceDocument.model_json_schema()

    for local_field in (
        "qr_url",
        "validation_with_tax_office",
        "model",
        "cost_usd",
    ):
        schema.get("properties", {}).pop(local_field, None)

        required = schema.get("required", [])
        if local_field in required:
            required.remove(local_field)

    return _make_strict_schema(schema)


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


def log_usage(model: str, usage: dict | None) -> None:
    """Print the model and cost to stderr (always, even without --verbose)."""
    cost = usage.get("cost") if usage else None

    if cost is not None:
        print(
            f"Model: {model} | Cost: ${cost:.6f}",
            file=sys.stderr,
        )
    else:
        print(f"Model: {model}", file=sys.stderr)


def extract_invoice_fields(
    image_urls: list[str],
    model: str | None = None,
) -> InvoiceDocument:
    """Send the page images to the LLM and return the parsed document.

    Requires the ``OPENROUTER_API_KEY`` environment variable. ``model``
    overrides the default :data:`MODEL`.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set."
        )

    model = model or MODEL

    payload = {
        "model": model,
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
        model,
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

    usage = data.get("usage")
    cost = usage.get("cost") if usage else None

    invoice.model = model
    invoice.cost_usd = cost

    log_usage(model, usage)

    return invoice
