"""Pydantic schemas for extracted invoices and the tax-office cross-check.

All leaf values are nullable: fields the document or the tax office does not
provide are ``null`` rather than guessed. Extra JSON keys are rejected
(:class:`StrictModel`), so a malformed model response fails loudly instead of
silently carrying unexpected data.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """A Pydantic model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class Party(StrictModel):
    """A business entity (supplier/issuer or customer/recipient)."""

    name: str | None
    vat_number: str | None
    address: str | None


class LineItem(StrictModel):
    # All fields default to None because per-line values (especially
    # gross_amount) are often not printed on the invoice, and the model must
    # be allowed to omit them rather than invent a value.
    code: str | None = None
    description: str | None = None
    quantity: float | None = None
    unit: str | None = None
    unit_price: float | None = None
    net_amount: float | None = None
    vat_rate: float | None = None
    vat_amount: float | None = None
    gross_amount: float | None = None


class Totals(StrictModel):
    """Document-level monetary totals."""

    net_amount: float | None
    vat_amount: float | None
    gross_amount: float | None


class TaxOfficeRecord(StrictModel):
    """Fields the myDATA QR verification endpoint has on record."""

    mark: str | None = None
    series: str | None = None
    aa: str | None = None
    document_type: str | None = None
    issue_date: str | None = None
    total_amount: float | None = None
    net_amount: float | None = None
    vat_amount: float | None = None
    issuer_vat: str | None = None
    customer_vat: str | None = None


class TaxOfficeValidation(StrictModel):
    """Result of cross-checking the invoice against the tax office."""

    status: Literal["verified", "mismatch", "unavailable"]
    tax_office: TaxOfficeRecord
    checks: dict[str, bool]
    warnings: list[str]


class InvoiceDocument(StrictModel):
    """A fully parsed accounting document."""

    document_type: Literal[
        "invoice",
        "credit_note",
        "receipt",
        "other",
    ]

    invoice_number: str | None
    issue_date: str | None

    supplier: Party
    customer: Party

    currency: str | None
    mydata_mark: str | None
    payment_method: str | None

    # Decoded locally from the myDATA QR code, not by the model.
    qr_url: str | None = None

    # Cross-check against the myDATA QR endpoint, done locally.
    validation_with_tax_office: TaxOfficeValidation | None = None

    # Extraction metadata, populated locally.
    model: str | None = None
    cost_usd: float | None = None

    line_items: list[LineItem]
    totals: Totals

    warnings: list[str]
