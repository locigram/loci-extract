"""Pydantic v2 models for every tax document type supported by loci-extract.

The top-level response is an ``Extraction`` with a ``documents`` list. Each
``Document`` carries a ``document_type`` discriminator, a ``data`` payload
(one of the per-type models below), a ``tax_year``, and a ``DocumentMetadata``
block for flags like ``is_corrected``, ``is_void``, ``is_summary_sheet``.

Shapes are defined by EXTRACT_SPEC.md. We deliberately keep the per-doc
models as plain pydantic models (not a discriminated Union) so the LLM can
return ``data`` as an unstructured dict and we validate it against the right
model post-hoc based on ``document_type``. This keeps the prompt simpler.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


class PartyInfo(BaseModel):
    """An employer/payer/recipient/filer block. Superset of fields used
    across document types. Fields not present in a given document simply
    remain unset."""

    name: str
    ein: str | None = None
    tin: str | None = None
    address: str | None = None
    phone: str | None = None
    state_id: str | None = None
    ssn_last4: str | None = None
    tin_last4: str | None = None


class DocumentMetadata(BaseModel):
    is_corrected: bool = False
    is_void: bool = False
    is_summary_sheet: bool = False
    payer_tin_type: Literal["EIN", "SSN"] | None = None
    notes: list[str] = Field(default_factory=list)


class StateWithholding(BaseModel):
    """Generic state-level wages/withheld row used by most W-2/1099 variants."""

    state_abbr: str
    state_id: str | None = None
    box16_state_wages: float | None = None
    box17_state_withheld: float | None = None
    # 1099s use different box numbers — leave optional rather than force
    # per-type variants:
    box5_state_income: float | None = None
    box6_state_withheld: float | None = None
    box7_state_income: float | None = None
    box14_state_income: float | None = None
    box15_state_withheld: float | None = None
    box16_state_income: float | None = None


class LocalWithholding(BaseModel):
    locality_name: str
    box18_local_wages: float | None = None
    box19_local_withheld: float | None = None


# ---------------------------------------------------------------------------
# W-2
# ---------------------------------------------------------------------------


class W2Federal(BaseModel):
    box1_wages: float = 0.0
    box2_federal_withheld: float = 0.0
    box3_ss_wages: float = 0.0
    box4_ss_withheld: float = 0.0
    box5_medicare_wages: float = 0.0
    box6_medicare_withheld: float = 0.0
    box7_ss_tips: float | None = None
    box8_allocated_tips: float | None = None
    box10_dependent_care: float | None = None
    box11_nonqualified_plans: float | None = None


class Box12Item(BaseModel):
    code: str
    amount: float = 0.0
    description: str = ""


class Box13(BaseModel):
    statutory_employee: bool = False
    retirement_plan: bool = False
    third_party_sick_pay: bool = False


class Box14Item(BaseModel):
    label: str
    amount: float = 0.0


class W2StateRow(BaseModel):
    state_abbr: str
    state_id: str | None = None
    box16_state_wages: float = 0.0
    box17_state_withheld: float = 0.0


class W2(BaseModel):
    employer: PartyInfo
    employee: PartyInfo
    federal: W2Federal = Field(default_factory=W2Federal)
    box12: list[Box12Item] = Field(default_factory=list)
    box13: Box13 = Field(default_factory=Box13)
    box14_other: list[Box14Item] = Field(default_factory=list)
    state: list[W2StateRow] = Field(default_factory=list)
    local: list[LocalWithholding] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 1099-NEC
# ---------------------------------------------------------------------------


class Form1099NEC(BaseModel):
    payer: PartyInfo
    recipient: PartyInfo
    account_number: str | None = None
    box1_nonemployee_compensation: float = 0.0
    box2_direct_sales: bool = False
    box4_federal_withheld: float = 0.0
    state: list[StateWithholding] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 1099-MISC
# ---------------------------------------------------------------------------


class Form1099MISC(BaseModel):
    payer: PartyInfo
    recipient: PartyInfo
    account_number: str | None = None
    box1_rents: float = 0.0
    box2_royalties: float = 0.0
    box3_other_income: float = 0.0
    box4_federal_withheld: float = 0.0
    box5_fishing_boat_proceeds: float = 0.0
    box6_medical_health_payments: float = 0.0
    box7_direct_sales: bool = False
    box8_substitute_payments: float = 0.0
    box9_crop_insurance: float = 0.0
    box10_gross_proceeds_attorney: float = 0.0
    box11_fish_purchased: float = 0.0
    box12_section_409a_deferrals: float = 0.0
    box13_fatca: bool = False
    box14_excess_golden_parachute: float = 0.0
    box15_nonqualified_deferred: float = 0.0
    state: list[StateWithholding] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 1099-INT
# ---------------------------------------------------------------------------


class Form1099INT(BaseModel):
    payer: PartyInfo
    recipient: PartyInfo
    account_number: str | None = None
    box1_interest_income: float = 0.0
    box2_early_withdrawal_penalty: float = 0.0
    box3_us_savings_bond_interest: float = 0.0
    box4_federal_withheld: float = 0.0
    box5_investment_expenses: float = 0.0
    box6_foreign_tax_paid: float = 0.0
    box7_foreign_country: str | None = None
    box8_tax_exempt_interest: float = 0.0
    box9_specified_private_activity_bond_interest: float = 0.0
    box10_market_discount: float = 0.0
    box11_bond_premium: float = 0.0
    box12_bond_premium_treasury: float = 0.0
    box13_bond_premium_tax_exempt: float = 0.0
    box14_tax_exempt_bond_cusip: str | None = None
    box15_fatca: bool = False
    state: list[StateWithholding] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 1099-DIV
# ---------------------------------------------------------------------------


class Form1099DIV(BaseModel):
    payer: PartyInfo
    recipient: PartyInfo
    account_number: str | None = None
    box1a_total_ordinary_dividends: float = 0.0
    box1b_qualified_dividends: float = 0.0
    box2a_total_capital_gain: float = 0.0
    box2b_unrecap_sec1250_gain: float = 0.0
    box2c_section1202_gain: float = 0.0
    box2d_collectibles_gain: float = 0.0
    box2e_section897_ordinary_dividends: float = 0.0
    box2f_section897_capital_gain: float = 0.0
    box3_nondividend_distributions: float = 0.0
    box4_federal_withheld: float = 0.0
    box5_section199a_dividends: float = 0.0
    box6_investment_expenses: float = 0.0
    box7_foreign_tax_paid: float = 0.0
    box8_foreign_country: str | None = None
    box9_cash_liquidation: float = 0.0
    box10_noncash_liquidation: float = 0.0
    box11_fatca: bool = False
    box12_exempt_interest_dividends: float = 0.0
    box13_specified_private_activity: float = 0.0
    state: list[StateWithholding] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 1099-B
# ---------------------------------------------------------------------------


class BTransaction(BaseModel):
    description: str
    cusip: str | None = None
    date_acquired: str | None = None
    date_sold: str | None = None
    box1a_quantity: float | None = None
    box1b_date_acquired: str | None = None
    box1c_date_sold: str | None = None
    box1d_proceeds: float = 0.0
    box1e_cost_basis: float = 0.0
    box1f_accrued_market_discount: float = 0.0
    box1g_wash_sale_loss_disallowed: float = 0.0
    box2_term: Literal["SHORT", "LONG", "ORDINARY"] | None = None
    box3_basis_reported_to_irs: bool = False
    box4_federal_withheld: float = 0.0
    box5_noncovered_security: bool = False
    box6_reported_gross_or_net: Literal["GROSS", "NET"] | None = None
    box7_loss_not_allowed: bool = False
    box8_proceeds_from_collectibles: bool = False
    box9_unrealized_profit_open_contracts: float = 0.0
    box10_unrealized_profit_closed_contracts: float = 0.0
    box11_aggregate_profit_loss: float = 0.0
    box12_basis_reported: bool = False
    box13_fatca: bool = False
    state: list[StateWithholding] = Field(default_factory=list)


class BSummary(BaseModel):
    total_proceeds: float = 0.0
    total_cost_basis: float = 0.0
    total_federal_withheld: float = 0.0


class Form1099B(BaseModel):
    payer: PartyInfo
    recipient: PartyInfo
    account_number: str | None = None
    transactions: list[BTransaction] = Field(default_factory=list)
    summary: BSummary = Field(default_factory=BSummary)


# ---------------------------------------------------------------------------
# 1099-R
# ---------------------------------------------------------------------------


class Form1099R(BaseModel):
    payer: PartyInfo
    recipient: PartyInfo
    account_number: str | None = None
    box1_gross_distribution: float = 0.0
    box2a_taxable_amount: float = 0.0
    box2b_taxable_amount_not_determined: bool = False
    box2b_total_distribution: bool = False
    box3_capital_gain: float = 0.0
    box4_federal_withheld: float = 0.0
    box5_employee_contributions: float = 0.0
    box6_net_unrealized_appreciation: float = 0.0
    box7_distribution_code: str | None = None
    box7_ira_sep_simple: bool = False
    box8_other: float = 0.0
    box9a_your_percentage_of_total: float | None = None
    box9b_total_employee_contributions: float | None = None
    box10_amount_allocable_to_ira: float = 0.0
    box11_1st_year_of_desig_roth: int | None = None
    box12_fatca: bool = False
    box13_date_of_payment: str | None = None
    box14_state_withheld: float = 0.0
    box15_state_id: str | None = None
    box16_state_distribution: float = 0.0
    box17_local_withheld: float = 0.0
    box18_locality_name: str | None = None
    box19_local_distribution: float = 0.0


# ---------------------------------------------------------------------------
# 1099-G
# ---------------------------------------------------------------------------


class Form1099G(BaseModel):
    payer: PartyInfo
    recipient: PartyInfo
    account_number: str | None = None
    box1_unemployment_compensation: float = 0.0
    box2_state_or_local_income_tax_refund: float = 0.0
    box3_box2_applies_to_tax_year: int | None = None
    box4_federal_withheld: float = 0.0
    box5_rtaa_payments: float = 0.0
    box6_taxable_grants: float = 0.0
    box7_agriculture_payments: float = 0.0
    box8_market_gain: bool = False
    box9_market_gain_amount: float = 0.0
    box10a_state_abbr: str | None = None
    box10b_state_id: str | None = None
    box11_state_income_tax_withheld: float = 0.0


# ---------------------------------------------------------------------------
# 1099-SA / K / S / C / A
# ---------------------------------------------------------------------------


class Form1099SA(BaseModel):
    payer: PartyInfo
    recipient: PartyInfo
    account_number: str | None = None
    box1_gross_distribution: float = 0.0
    box2_earnings_on_excess_contributions: float = 0.0
    box3_distribution_code: str | None = None
    box4_fmv_on_date_of_death: float | None = None
    box5_account_type: Literal["HSA", "Archer MSA", "MA MSA"] | None = None


class Form1099K(BaseModel):
    payer: PartyInfo
    recipient: PartyInfo
    account_number: str | None = None
    box1a_gross_payment_card_transactions: float = 0.0
    box1b_card_not_present_transactions: float = 0.0
    box2_merchant_category_code: str | None = None
    box3_number_of_transactions: int = 0
    box4_federal_withheld: float = 0.0
    box5_january: float = 0.0
    box5_february: float = 0.0
    box5_march: float = 0.0
    box5_april: float = 0.0
    box5_may: float = 0.0
    box5_june: float = 0.0
    box5_july: float = 0.0
    box5_august: float = 0.0
    box5_september: float = 0.0
    box5_october: float = 0.0
    box5_november: float = 0.0
    box5_december: float = 0.0
    state: list[StateWithholding] = Field(default_factory=list)


class Form1099S(BaseModel):
    filer: PartyInfo
    transferor: PartyInfo
    account_number: str | None = None
    box1_date_of_closing: str | None = None
    box2_gross_proceeds: float = 0.0
    box3_address_of_property: str | None = None
    box4_transferor_received_property: bool = False
    box5_buyer_part_of_real_estate_tax: float = 0.0


class Form1099C(BaseModel):
    creditor: PartyInfo
    debtor: PartyInfo
    account_number: str | None = None
    box1_date_of_identifiable_event: str | None = None
    box2_amount_of_debt_discharged: float = 0.0
    box3_interest_included_in_box2: float = 0.0
    box4_debt_description: str | None = None
    box5_personally_liable: bool = False
    box6_identifiable_event_code: str | None = None
    box7_fmv_of_property: float | None = None


class Form1099A(BaseModel):
    lender: PartyInfo
    borrower: PartyInfo
    account_number: str | None = None
    box1_date_of_lender_acquisition: str | None = None
    box2_balance_of_principal_outstanding: float = 0.0
    box4_fmv_of_property: float = 0.0
    box5_personally_liable: bool = False
    box6_description_of_property: str | None = None


# ---------------------------------------------------------------------------
# 1098 family
# ---------------------------------------------------------------------------


class Form1098(BaseModel):
    recipient: PartyInfo
    payer: PartyInfo
    account_number: str | None = None
    box1_mortgage_interest: float = 0.0
    box2_outstanding_mortgage_principal: float = 0.0
    box3_mortgage_origination_date: str | None = None
    box4_refund_of_overpaid_interest: float = 0.0
    box5_mortgage_insurance_premiums: float = 0.0
    box6_points_paid_on_purchase: float = 0.0
    box7_property_address: str | None = None
    box8_number_of_properties: int | None = None
    box9_other: str | None = None
    box10_property_tax: float | None = None
    box11_acquisition_date: str | None = None


class Form1098T(BaseModel):
    filer: PartyInfo
    student: PartyInfo
    account_number: str | None = None
    box1_payments_received: float = 0.0
    box2_reserved: float | None = None
    box3_change_in_reporting_method: bool = False
    box4_adjustments_prior_year: float = 0.0
    box5_scholarships_grants: float = 0.0
    box6_adjustments_scholarships_prior_year: float = 0.0
    box7_includes_amounts_for_next_period: bool = False
    box8_at_least_half_time_student: bool = False
    box9_graduate_student: bool = False
    box10_insurance_contract_reimbursement: float = 0.0


class Form1098E(BaseModel):
    recipient: PartyInfo
    borrower: PartyInfo
    account_number: str | None = None
    box1_student_loan_interest: float = 0.0
    box2_origination_fees_included: bool = False


# ---------------------------------------------------------------------------
# SSA-1099 / RRB-1099
# ---------------------------------------------------------------------------


class FormSSA1099(BaseModel):
    payer: PartyInfo
    beneficiary: PartyInfo
    claim_number: str | None = None
    box3_benefits_paid: float = 0.0
    box4_benefits_repaid: float = 0.0
    box5_net_benefits: float = 0.0
    box6_voluntary_federal_withheld: float = 0.0
    box7_address_of_payee: str | None = None
    description: str | None = None


class FormRRB1099(BaseModel):
    payer: PartyInfo
    recipient: PartyInfo
    box1_railroad_retirement_tier1: float = 0.0
    box2_non_contributory_tier1: float = 0.0
    box3_contributory_tier1: float = 0.0
    box4_federal_withheld: float = 0.0
    box5_supplemental_annuity: float = 0.0
    box6_workers_comp_offset: float = 0.0


# ---------------------------------------------------------------------------
# K-1 (1065 / 1120-S / 1041)
# ---------------------------------------------------------------------------


class CodedAmount(BaseModel):
    """Sub-line code + amount (+ optional description) used across K-1 boxes
    that report 'other' arrays — K-1 1065 box 11/13/15/17/20, K-1 1120-S box
    10/12/13/15/16/17, K-1 1041 box 9/11/12/13/14."""

    code: str
    amount: float = 0.0
    description: str = ""


class K11065International(BaseModel):
    # K-1 1065 Box 16 (international transactions) is typically a collection of codes.
    entries: list[CodedAmount] = Field(default_factory=list)


class K1Partnership1065(BaseModel):
    partnership: PartyInfo
    partner: PartyInfo
    partner_type: Literal["GENERAL", "LIMITED", "LLC_MEMBER", "OTHER"] | None = None
    ownership_percentage: float | None = None
    box1_ordinary_business_income: float = 0.0
    box2_net_rental_real_estate: float = 0.0
    box3_other_net_rental_income: float = 0.0
    box4_guaranteed_payments_services: float = 0.0
    box5_guaranteed_payments_capital: float = 0.0
    box6_guaranteed_payments_total: float = 0.0
    box7_interest_income: float = 0.0
    box8_ordinary_dividends: float = 0.0
    box8a_qualified_dividends: float = 0.0
    box9a_net_short_term_capital_gain: float = 0.0
    box9b_net_long_term_capital_gain: float = 0.0
    box9c_unrecaptured_1250_gain: float = 0.0
    box10_net_section_1231_gain: float = 0.0
    box11_other_income: list[CodedAmount] = Field(default_factory=list)
    box12_section_179_deduction: float = 0.0
    box13_other_deductions: list[CodedAmount] = Field(default_factory=list)
    box14_self_employment_earnings: float = 0.0
    box15_credits: list[CodedAmount] = Field(default_factory=list)
    box16_international: K11065International | None = None
    box17_amti: list[CodedAmount] = Field(default_factory=list)
    box18_tax_exempt_income: float = 0.0
    box19_distributions: list[CodedAmount] = Field(default_factory=list)
    box20_other_information: list[CodedAmount] = Field(default_factory=list)
    box21_foreign_taxes: float = 0.0


class K1SCorp1120S(BaseModel):
    corporation: PartyInfo
    shareholder: PartyInfo
    ownership_percentage: float | None = None
    stock_basis_beginning: float | None = None
    box1_ordinary_business_income: float = 0.0
    box2_net_rental_real_estate: float = 0.0
    box3_other_net_rental: float = 0.0
    box4_interest_income: float = 0.0
    box5a_ordinary_dividends: float = 0.0
    box5b_qualified_dividends: float = 0.0
    box6_royalties: float = 0.0
    box7_net_short_term_capital_gain: float = 0.0
    box8a_net_long_term_capital_gain: float = 0.0
    box8b_collectibles_gain: float = 0.0
    box8c_unrecaptured_1250_gain: float = 0.0
    box9_net_section_1231_gain: float = 0.0
    box10_other_income: list[CodedAmount] = Field(default_factory=list)
    box11_section_179_deduction: float = 0.0
    box12_other_deductions: list[CodedAmount] = Field(default_factory=list)
    box13_credits: list[CodedAmount] = Field(default_factory=list)
    box14_foreign_transactions: dict[str, Any] | None = None
    box15_amti: list[CodedAmount] = Field(default_factory=list)
    box16_items_affecting_basis: list[CodedAmount] = Field(default_factory=list)
    box17_other_information: list[CodedAmount] = Field(default_factory=list)


class K1EstateTrust1041(BaseModel):
    estate_or_trust: PartyInfo
    beneficiary: PartyInfo
    box1_interest_income: float = 0.0
    box2a_ordinary_dividends: float = 0.0
    box2b_qualified_dividends: float = 0.0
    box3_net_short_term_capital_gain: float = 0.0
    box4a_net_long_term_capital_gain: float = 0.0
    box4b_unrecaptured_1250_gain: float = 0.0
    box4c_section1202_gain: float = 0.0
    box5_other_portfolio_income: float = 0.0
    box6_ordinary_business_income: float = 0.0
    box7_net_rental_real_estate: float = 0.0
    box8_other_rental_income: float = 0.0
    box9_directly_apportioned_deductions: list[CodedAmount] = Field(default_factory=list)
    box10_estate_tax_deduction: float = 0.0
    box11_final_year_deductions: list[CodedAmount] = Field(default_factory=list)
    box12_amti: list[CodedAmount] = Field(default_factory=list)
    box13_credits: list[CodedAmount] = Field(default_factory=list)
    box14_other_information: list[CodedAmount] = Field(default_factory=list)
    box14h_foreign_tax_paid: float = 0.0
    box14i_gross_foreign_income: float = 0.0


# ---------------------------------------------------------------------------
# Top-level wrapper
# ---------------------------------------------------------------------------


DocumentTypeName = Literal[
    "W2",
    "1099-NEC",
    "1099-MISC",
    "1099-INT",
    "1099-DIV",
    "1099-B",
    "1099-R",
    "1099-G",
    "1099-SA",
    "1099-K",
    "1099-S",
    "1099-C",
    "1099-A",
    "1098",
    "1098-T",
    "1098-E",
    "SSA-1099",
    "RRB-1099",
    "K-1 1065",
    "K-1 1120-S",
    "K-1 1041",
]


# Map document_type → the pydantic model validating the ``data`` dict.
DATA_MODEL_BY_TYPE: dict[str, type[BaseModel]] = {
    "W2": W2,
    "1099-NEC": Form1099NEC,
    "1099-MISC": Form1099MISC,
    "1099-INT": Form1099INT,
    "1099-DIV": Form1099DIV,
    "1099-B": Form1099B,
    "1099-R": Form1099R,
    "1099-G": Form1099G,
    "1099-SA": Form1099SA,
    "1099-K": Form1099K,
    "1099-S": Form1099S,
    "1099-C": Form1099C,
    "1099-A": Form1099A,
    "1098": Form1098,
    "1098-T": Form1098T,
    "1098-E": Form1098E,
    "SSA-1099": FormSSA1099,
    "RRB-1099": FormRRB1099,
    "K-1 1065": K1Partnership1065,
    "K-1 1120-S": K1SCorp1120S,
    "K-1 1041": K1EstateTrust1041,
}


class Document(BaseModel):
    document_type: DocumentTypeName
    tax_year: int = 2025
    data: dict[str, Any]
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)

    def validated_data(self) -> BaseModel:
        """Validate ``data`` against the model keyed by ``document_type``."""
        model = DATA_MODEL_BY_TYPE.get(self.document_type)
        if model is None:
            raise ValueError(f"No data model registered for document_type={self.document_type!r}")
        return model.model_validate(self.data)


class Extraction(BaseModel):
    documents: list[Document] = Field(default_factory=list)

    def validate_all(self) -> list[BaseModel]:
        """Validate every document's ``data`` field against its type-specific model.

        Returns the list of validated per-type models. Raises ``pydantic.ValidationError``
        on the first failure.
        """
        return [doc.validated_data() for doc in self.documents]


__all__ = [
    # shared
    "PartyInfo",
    "DocumentMetadata",
    "StateWithholding",
    "LocalWithholding",
    # per-type
    "W2",
    "W2Federal",
    "Box12Item",
    "Box13",
    "Box14Item",
    "W2StateRow",
    "Form1099NEC",
    "Form1099MISC",
    "Form1099INT",
    "Form1099DIV",
    "Form1099B",
    "BTransaction",
    "BSummary",
    "Form1099R",
    "Form1099G",
    "Form1099SA",
    "Form1099K",
    "Form1099S",
    "Form1099C",
    "Form1099A",
    "Form1098",
    "Form1098T",
    "Form1098E",
    "FormSSA1099",
    "FormRRB1099",
    "K1Partnership1065",
    "K1SCorp1120S",
    "K1EstateTrust1041",
    "CodedAmount",
    # wrappers
    "Document",
    "Extraction",
    "DocumentTypeName",
    "DATA_MODEL_BY_TYPE",
]
