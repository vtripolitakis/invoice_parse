# invoice-parse

## Evangelos Tripolitakis
## vtripolitakis@__NOSPAM__gmail.com

Extract structured data from Greek (and English) accounting documents and
cross-check them against the Greek tax office (AADE).

The tool renders a PDF invoice to images, sends them to a vision LLM (Qwen via
[OpenRouter](https://openrouter.ai)), and decodes the **myDATA QR code**
locally — the QR is read with a barcode decoder (free), never by the model.
The decoded URL is then used to fetch the authoritative document record from
AADE and compare it with the extracted fields.

## Features

- Extracts suppliers, customers, line items, totals, VAT numbers, the myDATA
  MARK and more, as JSON.
- Decodes the myDATA QR code locally (no LLM cost, no hallucinated URLs).
- Fetches the authoritative record from every common QR destination:
  - `mydatapi.aade.gr` — official AADE verification page
  - `e-invoicing.gr` — Entersoft viewer
  - `einvoice.s1ecos.gr` — SoftOne viewer
  - `*.epsilonnet.gr` — Epsilon Net viewer (raw myDATA XML)
- Cross-checks the extracted data against the tax office and reports
  `verified` / `mismatch` / `unavailable`.
- Strict JSON output on stdout — safe to pipe into other tools.

## Installation

```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

Dependencies are declared in [`requirements.txt`](requirements.txt):

- `PyMuPDF` — PDF rendering
- `pydantic` — validation and JSON schema
- `requests` — HTTP (LLM and tax office)
- `zxing-cpp`, `numpy` — local QR decoding

## Configuration

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

## Usage

```bash
./invoice_parse.py --file invoice.pdf
```

For detailed progress logs (stderr):

```bash
./invoice_parse.py --file invoice.pdf --verbose
```

stdout contains only the JSON document, so it composes well:

```bash
./invoice_parse.py --file invoice.pdf | jq .validation_with_tax_office.status
```

## Output

The result is a single JSON object, for example:

```json
{
  "document_type": "invoice",
  "invoice_number": "5793",
  "issue_date": "2026-09-10",
  "supplier": { "name": "...", "vat_number": "...", "address": "..." },
  "customer": { "name": "...", "vat_number": "...", "address": "..." },
  "currency": "EUR",
  "mydata_mark": "400015210789192",
  "line_items": [
    {
      "code": "1",
      "description": "...",
      "quantity": 1.0,
      "unit": "Τεμάχια",
      "unit_price": 3.54,
      "net_amount": 3.54,
      "vat_rate": 13.0,
      "vat_amount": 0.46,
      "gross_amount": null
    }
  ],
  "totals": { "net_amount": 12.9, "vat_amount": 3.09, "gross_amount": 15.99 },
  "qr_url": "https://...",
  "validation_with_tax_office": {
    "status": "verified",
    "tax_office": {
      "mark": "400015210789192",
      "series": "ΤΔΑ",
      "aa": "5793",
      "document_type": "Τιμολόγιο Πώλησης",
      "issue_date": "10/09/2026",
      "total_amount": 15.99,
      "net_amount": 12.9,
      "vat_amount": 3.09,
      "issuer_vat": "148605355",
      "customer_vat": "801233366"
    },
    "checks": {
      "mark": true,
      "issue_date": true,
      "total_amount": true,
      "net_amount": true,
      "vat_amount": true,
      "issuer_vat": true,
      "customer_vat": true,
      "invoice_number": true
    },
    "warnings": []
  },
  "warnings": []
}
```

`validation_with_tax_office.status` is:

- `verified` — every comparable field matches the tax office
- `mismatch` — at least one field differs
- `unavailable` — no QR code, or the tax-office record could not be fetched

## Project layout

| File               | Purpose                                                        |
| ------------------ | -------------------------------------------------------------- |
| `invoice_parse.py` | CLI entry point and orchestration (pipeline)                   |
| `llm.py`           | OpenRouter/Qwen client: prompt, request, response parsing      |
| `pdf_utils.py`     | PDF → images and local myDATA QR decoding                      |
| `tax_office.py`    | Tax-office record fetching, parsing, and cross-check           |
| `models.py`        | Pydantic schemas for the invoice and the validation result     |

## Notes

- Only PDF input is supported.
- Encrypted / password-protected PDFs are rejected.
- All monetary values are JSON numbers; dates are `YYYY-MM-DD`; VAT and
  document numbers remain strings.
