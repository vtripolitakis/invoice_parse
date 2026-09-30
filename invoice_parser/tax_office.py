"""Cross-check extracted invoices against the Greek tax office (AADE).

The myDATA QR code on a Greek invoice points to one of several viewer pages.
This module fetches the authoritative document record from each destination
and compares it with the LLM-extracted invoice:

- ``mydatapi.aade.gr``   official AADE verification HTML.
- ``e-invoicing.gr``     Entersoft viewer (embeds the AADE URL).
- ``einvoice.s1ecos.gr`` SoftOne viewer (embeds the AADE URL).
- ``*.epsilonnet.gr``    Epsilon Net viewer (serves raw myDATA XML).

Public API
----------
``fetch_tax_office_record(qr_url)``
    Return the :class:`TaxOfficeRecord` the tax office holds for a document.

``validate_against_tax_office(invoice, qr_url)``
    Compare an extracted invoice with the tax-office record and report
    whether they match.
"""

import html
import logging
import re
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Callable

import requests

from .models import InvoiceDocument, TaxOfficeRecord, TaxOfficeValidation

logger = logging.getLogger(__name__)

# Browser-like headers. Some viewer pages are slow or reject non-browser user
# agents, so we impersonate a regular browser.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "el,en;q=0.9",
}


# ---------------------------------------------------------------------------
# Value normalisation helpers
# ---------------------------------------------------------------------------

def _clean(value: str | None) -> str | None:
    """Strip whitespace and treat empty strings as missing."""
    if value is None:
        return None

    value = value.strip()

    return value or None


def _parse_amount(value: str | None) -> float | None:
    """Parse a European-formatted amount ("15,99" or "1.234,56")."""
    if value is None:
        return None

    value = value.strip().replace(" ", "").replace("\xa0", "")

    if not value:
        return None

    if "," in value:
        if "." in value:
            # The last separator is the decimal one.
            if value.rfind(",") > value.rfind("."):
                value = value.replace(".", "").replace(",", ".")
            else:
                value = value.replace(",", "")
        else:
            value = value.replace(",", ".")

    try:
        return float(value)
    except ValueError:
        return None


def _parse_date(value: str | None) -> datetime | None:
    """Parse an ISO (YYYY-MM-DD) or Greek day-first date."""
    if value is None:
        return None

    value = value.strip()

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        pass

    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    return None


def _amounts_match(a: float | None, b: float | None) -> bool:
    """Compare two monetary amounts within a one-cent tolerance."""
    if a is None or b is None:
        return False

    return abs(a - b) < 0.01


# ---------------------------------------------------------------------------
# AADE verification page (mydatapi.aade.gr)
# ---------------------------------------------------------------------------

# Maps TaxOfficeRecord fields to the <input id="..."> on the AADE page.
AADE_FIELD_IDS = {
    "mark": "tmark",
    "series": "snumber",
    "aa": "saa",
    "document_type": "dtype",
    "issue_date": "tdate",
    "total_amount": "tamount",
    "net_amount": "namount",
    "vat_amount": "vat",
    "issuer_vat": "vatnumber",
    "customer_vat": "crvatnumber",
}


def parse_aade_html(html_text: str) -> TaxOfficeRecord:
    """Parse the AADE verification HTML into a :class:`TaxOfficeRecord`.

    Every field lives in an ``<input ... value="...">`` tag. The page has a
    malformed, unclosed comment around the MARK, so we must NOT strip HTML
    comments first -- parsing inputs directly still finds them all.
    """
    values_by_id: dict[str, str] = {}

    for tag in re.finditer(r"<input\b[^>]*>", html_text, re.IGNORECASE):
        tag_text = tag.group(0)

        id_match = re.search(
            r'\bid\s*=\s*["\']?([^"\'\s>]+)',
            tag_text,
            re.IGNORECASE,
        )

        value_match = re.search(
            r'\bvalue\s*=\s*"([^"]*)"',
            tag_text,
            re.IGNORECASE,
        )

        if id_match and value_match:
            values_by_id[id_match.group(1)] = html.unescape(
                value_match.group(1)
            )

    def value_for(field: str) -> str | None:
        return values_by_id.get(AADE_FIELD_IDS[field])

    def amount_for(field: str) -> float | None:
        return _parse_amount(values_by_id.get(AADE_FIELD_IDS[field]))

    return TaxOfficeRecord(
        mark=_clean(value_for("mark")),
        series=_clean(value_for("series")),
        aa=_clean(value_for("aa")),
        document_type=_clean(value_for("document_type")),
        issue_date=_clean(value_for("issue_date")),
        total_amount=amount_for("total_amount"),
        net_amount=amount_for("net_amount"),
        vat_amount=amount_for("vat_amount"),
        issuer_vat=_clean(value_for("issuer_vat")),
        customer_vat=_clean(value_for("customer_vat")),
    )


def _fetch_aade_record(qr_url: str) -> TaxOfficeRecord:
    """Fetch and parse the AADE verification page."""
    response = requests.get(
        qr_url,
        headers={"User-Agent": "invoice-parse-poc/1.0"},
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            f"Tax office HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    # The page is UTF-8 but the header omits the charset.
    response.encoding = "utf-8"

    return parse_aade_html(response.text)


# ---------------------------------------------------------------------------
# Third-party viewer pages (Entersoft, SoftOne)
# ---------------------------------------------------------------------------

def _fetch_viewer_record(qr_url: str) -> TaxOfficeRecord:
    """Fetch a third-party viewer page and extract its tax-office data.

    e-invoicing.gr (Entersoft) and einvoice.s1ecos.gr (SoftOne) embed the
    official myDATA verification URL, so we extract that and delegate to the
    AADE parser. These pages are slow, hence the generous timeout and browser
    user agent.
    """
    response = requests.get(
        qr_url,
        headers=BROWSER_HEADERS,
        timeout=90,
    )

    if not response.ok:
        raise RuntimeError(
            f"Viewer page HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    response.encoding = "utf-8"

    html_text = response.text

    aade_match = re.search(
        r'href="(https://mydatapi\.aade\.gr/myDATA/TimologioQR/QRInfo[^"]*)"',
        html_text,
        re.IGNORECASE,
    )

    if aade_match:
        aade_url = html.unescape(aade_match.group(1))

        logger.info("Found embedded AADE URL, delegating: %s", aade_url)

        try:
            return _fetch_aade_record(aade_url)
        except Exception:
            logger.warning(
                "Could not fetch embedded AADE URL, using page data only."
            )
            # Fall through to the fields exposed by the viewer page itself.

    # Fallback: the viewer page exposes only MARK and transmission date.
    mark = None
    mark_match = re.search(
        r"M\.AR\.K:\s*([0-9]+)",
        html_text,
        re.IGNORECASE,
    )

    if mark_match:
        mark = mark_match.group(1)
    else:
        # Generic 15-digit MARK (SoftOne prints it as a bare number).
        fallback = re.search(r"\b(\d{15})\b", html_text)

        if fallback:
            mark = fallback.group(1)

    issue_date = None
    date_match = re.search(
        r"Ημ/νία διαβίβασης:\s*([^<\n]+)",
        html_text,
    )

    if date_match:
        issue_date = _clean(html.unescape(date_match.group(1)))

    return TaxOfficeRecord(mark=mark, issue_date=issue_date)


# ---------------------------------------------------------------------------
# Epsilon Net (serves raw myDATA XML)
# ---------------------------------------------------------------------------

MYDATA_NS = "{http://www.aade.gr/myDATA/invoice/v1.0}"

# myDATA invoice type code -> human-readable Greek label.
MYDATA_INVOICE_TYPES = {
    "1.1": "Τιμολόγιο Πώλησης",
    "1.2": "Τιμολόγιο Παροχής Υπηρεσιών",
    "1.3": "Πιστωτικό Τιμολόγιο",
    "1.4": "Στοιχείο Αυτοπαράδοσης",
    "1.5": "Συμπληρωματικό Στοιχείο",
    "2.1": "Τιμολόγιο Ενδοκοινοτικής Παράδοσης",
    "2.2": "Τιμολόγιο Παροχής Υπηρεσιών (Ενδοκοινοτικό)",
    "2.3": "Λοιπά Παραστατικά Εξαγωγών",
    "2.4": "Στοιχείο Πώλησης σε Τρίτη Χώρα",
    "3.1": "Τίτλος Κτήσης",
    "3.2": "Πιστωτικός Τίτλος Κτήσης",
    "5.1": "Τιμολόγιο Ενδοκοινοτικής Παράδοσης",
    "5.2": "Τιμολόγιο Παροχής Υπηρεσιών (Ενδοκοινοτική Λήψη)",
    "7.1": "Συμβόλαιο - Έσοδο",
    "8.1": "Ενοίκια - Έσοδο",
    "8.2": "Ειδικό Στοιχείο - Απόδειξη Είσπραξης",
    "8.3": "Ειδικό Στοιχείο - Απόδειξη Επιστροφής",
    "11.1": "Απόδειξη Λιανικής",
    "11.2": "Απόδειξη Παροχής Υπηρεσιών",
    "11.3": "Απλοποιημένο Τιμολόγιο",
    "11.4": "Πιστωτικό Στοιχείο Λιανικής",
    "13.1": "Τιμολόγιο Αγοράς",
    "13.2": "Πιστωτικό Τιμολόγιο Αγοράς",
    "14.1": "Τιμολόγιο Ενδοκοινοτικής Απόκτησης",
}


def _xml_text(elem: ET.Element, path: str) -> str | None:
    """Return the text of the element at ``path`` (slash-separated)."""
    node: ET.Element | None = elem

    for part in path.split("/"):
        if node is None:
            return None

        node = node.find(f"{MYDATA_NS}{part}")

    if node is None:
        return None

    return _clean(node.text)


def parse_epsilon_xml(xml_text: str) -> TaxOfficeRecord:
    """Parse the myDATA XML served by Epsilon Net."""
    root = ET.fromstring(xml_text)

    invoice = root.find(f"{MYDATA_NS}invoice")

    if invoice is None:
        invoice = root

    invoice_type = _xml_text(invoice, "invoiceHeader/invoiceType")
    document_type = MYDATA_INVOICE_TYPES.get(invoice_type or "", invoice_type)

    return TaxOfficeRecord(
        mark=_xml_text(invoice, "mark"),
        series=_xml_text(invoice, "invoiceHeader/series"),
        aa=_xml_text(invoice, "invoiceHeader/aa"),
        document_type=document_type,
        issue_date=_xml_text(invoice, "invoiceHeader/issueDate"),
        total_amount=_parse_amount(
            _xml_text(invoice, "invoiceSummary/totalGrossValue")
        ),
        net_amount=_parse_amount(
            _xml_text(invoice, "invoiceSummary/totalNetValue")
        ),
        vat_amount=_parse_amount(
            _xml_text(invoice, "invoiceSummary/totalVatAmount")
        ),
        issuer_vat=_xml_text(invoice, "issuer/vatNumber"),
        customer_vat=_xml_text(invoice, "counterpart/vatNumber"),
    )


def _fetch_epsilon_record(qr_url: str) -> TaxOfficeRecord:
    """Fetch the raw myDATA XML from an Epsilon Net viewer URL.

    Epsilon Net serves a Blazor viewer but also exposes the raw myDATA XML at
    a predictable endpoint, which is richer and easier to parse than the
    rendered page. Two URL shapes are supported:

    - ``/DocViewer/{dashed-guid}``
    - ``/fd/{hex-guid}:{version}``  (redirects to /DocViewer/{dashed-guid})
    """
    if "/DocViewer/" in qr_url:
        base_url, _, doc_id = qr_url.partition("/DocViewer/")
        doc_id = doc_id.split("?")[0].rstrip("/")
    elif "/fd/" in qr_url:
        base_url, _, rest = qr_url.partition("/fd/")
        hex_guid = rest.split(":")[0].split("?")[0].rstrip("/")
        doc_id = str(uuid.UUID(hex_guid))
    else:
        raise RuntimeError(f"Unrecognized Epsilon Net URL: {qr_url}")

    endpoint = f"{base_url}/filedocument/GetInvoiceDocDetailed/{doc_id}"

    logger.info("Fetching Epsilon Net XML: %s", endpoint)

    response = requests.get(
        endpoint,
        headers=BROWSER_HEADERS,
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            f"Epsilon Net HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    response.encoding = "utf-8"

    return parse_epsilon_xml(response.text)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def fetch_tax_office_record(qr_url: str) -> TaxOfficeRecord:
    """Fetch the tax-office record, dispatching on the QR destination."""
    if "mydatapi.aade.gr" in qr_url:
        logger.info("Fetching tax-office record from AADE")
        return _fetch_aade_record(qr_url)

    if "epsilonnet.gr" in qr_url:
        return _fetch_epsilon_record(qr_url)

    logger.info("Using third-party viewer handler: %s", qr_url)
    return _fetch_viewer_record(qr_url)


# ---------------------------------------------------------------------------
# Cross-check
# ---------------------------------------------------------------------------

def validate_against_tax_office(
    invoice: InvoiceDocument,
    qr_url: str | None,
) -> TaxOfficeValidation:
    """Compare the extracted invoice against the tax office.

    Returns a :class:`TaxOfficeValidation` with status ``verified``,
    ``mismatch`` or ``unavailable``, the per-field comparison results, and any
    warnings.
    """
    if not qr_url:
        return TaxOfficeValidation(
            status="unavailable",
            tax_office=TaxOfficeRecord(),
            checks={},
            warnings=["No QR code found on the document."],
        )

    try:
        record = fetch_tax_office_record(qr_url)
    except Exception as exc:
        return TaxOfficeValidation(
            status="unavailable",
            tax_office=TaxOfficeRecord(),
            checks={},
            warnings=[f"Could not fetch tax-office data: {exc}"],
        )

    checks: dict[str, bool] = {}
    warnings: list[str] = []

    def compare(
        name: str,
        invoice_value: str | float | None,
        tax_value: str | float | None,
        matches: Callable[..., bool],
    ) -> None:
        if tax_value is None or tax_value == "":
            warnings.append(f"Tax office has no value for '{name}'.")
            return

        if invoice_value is None or invoice_value == "":
            warnings.append(
                f"Invoice value for '{name}' is missing; cannot verify."
            )
            return

        checks[name] = bool(matches(invoice_value, tax_value))

    compare(
        "mark",
        invoice.mydata_mark,
        record.mark,
        lambda a, b: str(a).strip() == str(b).strip(),
    )

    compare(
        "issue_date",
        invoice.issue_date,
        record.issue_date,
        lambda a, b: _parse_date(a) is not None
        and _parse_date(a) == _parse_date(b),
    )

    compare(
        "total_amount",
        invoice.totals.gross_amount,
        record.total_amount,
        _amounts_match,
    )

    compare(
        "net_amount",
        invoice.totals.net_amount,
        record.net_amount,
        _amounts_match,
    )

    compare(
        "vat_amount",
        invoice.totals.vat_amount,
        record.vat_amount,
        _amounts_match,
    )

    compare(
        "issuer_vat",
        invoice.supplier.vat_number,
        record.issuer_vat,
        lambda a, b: str(a).strip() == str(b).strip(),
    )

    compare(
        "customer_vat",
        invoice.customer.vat_number,
        record.customer_vat,
        lambda a, b: str(a).strip() == str(b).strip(),
    )

    # Invoice number vs series + Α/Α.
    if record.series or record.aa:
        if not invoice.invoice_number:
            warnings.append(
                "Invoice number is missing; cannot verify series/Α.Α.."
            )
        else:
            number = str(invoice.invoice_number).strip()

            candidates = set()

            if record.aa:
                candidates.add(str(record.aa).strip())

            if record.series:
                candidates.add(str(record.series).strip())

            if record.series and record.aa:
                series = str(record.series).strip()
                aa = str(record.aa).strip()

                candidates.update(
                    {
                        f"{series}-{aa}",
                        f"{series} {aa}",
                        f"{series}{aa}",
                    }
                )

            checks["invoice_number"] = number in candidates

    if not checks:
        status = "unavailable"
    elif all(checks.values()):
        status = "verified"
    else:
        status = "mismatch"

    logger.info("Tax-office validation status: %s", status)

    return TaxOfficeValidation(
        status=status,
        tax_office=record,
        checks=checks,
        warnings=warnings,
    )
