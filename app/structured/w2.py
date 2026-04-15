from __future__ import annotations

import re

from app.normalization import extract_last4, mask_identifier, parse_amount
from app.review import build_review_metadata
from app.schemas import ExtractionPayload, StructuredDocument
from app.structured.common import first_source_pages, get_form_lines, get_form_text, search_patterns, snippet_around_match


BOX_PATTERNS = {
    "1_wages_tips_other_comp": [r"\b1\s+wages,? tips,? other compensation\s+([\$\d,().-]+)"],
    "2_federal_income_tax_withheld": [r"\b2\s+federal income tax withheld\s+([\$\d,().-]+)"],
    "3_social_security_wages": [r"\b3\s+social security wages\s+([\$\d,().-]+)"],
    "4_social_security_tax_withheld": [r"\b4\s+social security tax withheld\s+([\$\d,().-]+)"],
    "5_medicare_wages_and_tips": [r"\b5\s+medicare wages and tips\s+([\$\d,().-]+)"],
    "6_medicare_tax_withheld": [r"\b6\s+medicare tax withheld\s+([\$\d,().-]+)"],
}


def _extract_tax_year(text: str) -> int | None:
    match = re.search(r"form\s+w-2.*?(20\d{2})", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(20\d{2})\b", text)
    return int(match.group(1)) if match else None


def _extract_labeled_value(text: str, patterns: list[str]) -> str | None:
    match = search_patterns(text, patterns)
    if not match:
        return None
    return " ".join(group for group in match.groups() if group is not None).strip() if match.groups() else match.group(0).strip()


_W2_FORM_SIGNALS = [
    "wage and tax statement",
    "form w-2",
    "employer identification number",
    "employee's social security",
    "wages, tips, other compensation",
    "federal income tax withheld",
    "social security wages",
    "medicare wages",
]


def build_w2_document(raw_payload: ExtractionPayload, *, mask_pii: bool = True) -> StructuredDocument:
    # Use only form data pages, skip instruction/copy pages
    text = get_form_text(raw_payload, form_signals=_W2_FORM_SIGNALS)
    lines = get_form_lines(raw_payload, form_signals=_W2_FORM_SIGNALS)

    employee_name_patterns = [
        r"employee(?:'s)?\s+name(?:,\s*address,\s*and\s*zip\s*code)?\s*[:#-]?\s*([^\n]+)",
        r"employee name\s*[:#-]?\s*([^\n]+)",
    ]
    employer_name_patterns = [
        r"employer(?:'s)?\s+name\s*[:#-]?\s*([^\n]+)",
        r"employer name\s*[:#-]?\s*([^\n]+)",
    ]
    employee_name = _extract_labeled_value(text, employee_name_patterns)
    employer_name = _extract_labeled_value(text, employer_name_patterns)
    employer_ein = _extract_labeled_value(text, [r"employer identification number\s*[:#-]?\s*([\d-]+)"])
    employee_ssn_source = _extract_labeled_value(
        text,
        [
            r"employee(?:'s)?\s+social security number\s*[:#-]?\s*([^\n]+)",
            r"ssn\s*[:#-]?\s*([^\n]+)",
        ],
    )
    boxes = {name: parse_amount(_extract_labeled_value(text, patterns)) for name, patterns in BOX_PATTERNS.items()}
    boxes["7_social_security_tips"] = None
    boxes["8_allocated_tips"] = None
    boxes["10_dependent_care_benefits"] = None
    boxes["11_nonqualified_plans"] = None
    boxes["12_codes"] = []
    boxes["13_flags"] = {
        "statutory_employee": any("statutory employee" in line.lower() for line in lines),
        "retirement_plan": any("retirement plan" in line.lower() for line in lines),
        "third_party_sick_pay": any(
            "third-party sick pay" in line.lower() or "third party sick pay" in line.lower() for line in lines
        ),
    }
    boxes["14_other"] = []

    validation_errors: list[str] = []
    if boxes["1_wages_tips_other_comp"] is None and re.search(r"wages,? tips,? other compensation", text, flags=re.IGNORECASE):
        validation_errors.append("unable_to_parse_1_wages_tips_other_comp")
    if boxes["2_federal_income_tax_withheld"] is None and re.search(r"federal income tax withheld", text, flags=re.IGNORECASE):
        validation_errors.append("unable_to_parse_2_federal_income_tax_withheld")

    evidence = {
        "source_pages": first_source_pages(raw_payload),
        "employee_name": snippet_around_match(text, employee_name_patterns),
        "employee_ssn": snippet_around_match(
            text,
            [
                r"employee(?:'s)?\s+social security number\s*[:#-]?\s*([^\n]+)",
                r"ssn\s*[:#-]?\s*([^\n]+)",
            ],
        ),
        "employer_name": snippet_around_match(text, employer_name_patterns),
        "box_1": snippet_around_match(text, BOX_PATTERNS["1_wages_tips_other_comp"]),
        "box_2": snippet_around_match(text, BOX_PATTERNS["2_federal_income_tax_withheld"]),
    }

    ssn_last4 = extract_last4(employee_ssn_source)
    fields = {
        "tax_year": _extract_tax_year(text),
        "employee": {
            "full_name": employee_name,
            "ssn_last4": ssn_last4,
            "ssn_masked": mask_identifier(employee_ssn_source) if mask_pii else employee_ssn_source,
            "address": None,
        },
        "employer": {
            "name": employer_name,
            "ein": mask_identifier(employer_ein) if mask_pii else employer_ein,
            "address": None,
        },
        "boxes": boxes,
        "state_local_entries": [],
        "evidence": evidence,
    }
    review = build_review_metadata(
        required_fields={
            "tax_year": fields["tax_year"],
            "employee.full_name": fields["employee"]["full_name"],
            "employer.name": fields["employer"]["name"],
            "boxes.1_wages_tips_other_comp": fields["boxes"]["1_wages_tips_other_comp"],
        },
        validation_errors=validation_errors,
        raw_extra=raw_payload.extra,
        document_type="w2",
    )
    return StructuredDocument(document_type="w2", fields=fields, review=review)
