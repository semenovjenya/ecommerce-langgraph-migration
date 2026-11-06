"""Validate model output before any candidate or result enters a checkpoint."""
import re

from commerce_agent.sales.contracts import SalesDiagnosticResult, EvidenceFact
from commerce_agent.sales.policy import MAX_RESULT_BYTES, assert_bounded_safe
from commerce_agent.sales.validation import validate_sales_diagnostic_result
from commerce_agent.sales.claims import native_claim_proofs
from commerce_agent.sales.semantics import check_proofs, validate_finding_semantics


class SalesNarrativePolicyError(ValueError):
    """Static local diagnostic role, never the offending provider text."""
    def __init__(self, message, role):
        super().__init__(message)
        self.role = role


def _validate_narrative_numbers(result, prepared):
    # Exact standalone native propositions bind metric/unit/period/SKU and
    # provenance. A matching scalar is NOT proof. Known SKU identifiers are not
    # quantities, but must be explicitly labelled as identifiers.
    skus = {str(fact["sku"]) for fact in prepared["facts"] if fact.get("sku")}
    skus.update(prepared["request"]["targets"])
    fields = [("summary", result.summary)]
    fields.extend(("recommendation", value) for value in result.recommendations)
    fields.extend((role, getattr(finding, role)) for finding in result.findings for role in
                  ("factor", "observation", "impact", "cause", "action"))
    # Gaps and delegated questions are durable provider-owned prose too. They
    # must not be an alternative route for invented metrics into root state.
    fields.extend((f"gap_{role}", getattr(gap, role)) for gap in result.data_gaps
                  for role in ("metric", "reason", "impact"))
    fields.extend((f"request_{role}", getattr(request, role)) for request in result.cross_agent_requests
                  for role in ("reason", "question"))
    fields.extend(("request_check", check) for request in result.cross_agent_requests
                  for check in request.required_checks)
    approved = native_claim_proofs(prepared)
    approved_checks = check_proofs(prepared)
    for finding in result.findings:
        for line in finding.observation.splitlines():
            if line in approved and not approved[line].intersection(finding.evidence):
                raise ValueError("Sales native proposition evidence is missing")
    unbound = []
    for role, text in fields:
        if role in {"action", "recommendation", "request_question", "request_check"} and text in approved_checks:
            text = ""  # neutral request, not a factual quantity/direction claim
        # Whole lines only: "not <true claim>" or "profit: <revenue claim>"
        # must not launder a proposition by removing an embedded substring.
        lines = text.splitlines()
        if any(line in approved for line in lines):
            if role not in {"summary", "observation"}:
                raise ValueError("Sales native proposition is not a causal/action/impact proof")
            if any(line.strip() and line not in approved for line in lines):
                raise ValueError("Sales native proposition cannot mix with provider context")
            text = ""
        unbound.append((role, text))
        for sku in sorted(skus, key=len, reverse=True):
            text = re.sub(r"(?:\bSKU|\bартикул\w*)\s*[:№#]?\s*" + re.escape(sku) + r"(?![\w.-])",
                          "артикул", text, flags=re.IGNORECASE)
        if re.search(r"\d|\b(?:вдвое|втрое|удво\w*|утро\w*|половин\w*|двукрат\w*|трехкрат\w*|"
                     r"double\w*|triple\w*|half|twice|thrice)\b", text.lower()):
            raise SalesNarrativePolicyError("Sales quantitative prose needs a bound native proposition", role)
        if re.search(r"\b(?:вырос\w*|растет|росл\w*|снизил\w*|снизив\w*|упал\w*|пада\w*|"
                     r"увеличил\w*|уменьшил\w*|выше|ниже|равна|grow\w*|grew|increas\w*|decreas\w*|"
                     r"higher|lower|equal\w*|fell|fall\w*)\b", text.lower()):
            raise ValueError("Sales directional prose needs a bound native comparison")
        # No causal identification exists in this read-only Sales slice. A
        # hypothesis is permitted only as a complete explicitly unconfirmed
        # line, not a marker elsewhere legitimising a proven causal assertion.
        for line in text.lower().splitlines():
            if re.search(r"вызван\w*|обусловлен\w*|привел\w*|привёл\w*|повлиял\w*|"
                         r"\b(?:caus\w*|due to|driven by|resulted in)\b", line) and not re.fullmatch(
                r"гипотеза \(не подтверждена\): [^.!?;\n]+\.?", line):
                raise ValueError("Sales causal assertion has no verified causal evidence")
    foreign = r"реклам|маркетинг|цен[аыу]|остатк|поставк|марж|прибыл|advertis|marketing|pricing|price|inventory|supply|margin|profit"
    dimensions = r"рекламе|маркетинге|ценах|остатках|поставках|марже|прибыли"
    absence = (rf"\s*(?:(?:нет|недостаточно|не хватает) (?:проверенных )?данных о(?:б)? (?:{dimensions})|"
               rf"данные о(?:б)? (?:{dimensions}) (?:недоступны|отсутствуют))\s*\.?\s*")
    for gap in result.data_gaps:
        if re.search(foreign, f"{gap.metric} {gap.reason} {gap.impact}".lower()):
            if (not re.fullmatch(absence, gap.reason.lower())
                or gap.impact not in {"Анализ ограничен доступными данными.",
                                     "Нужна дополнительная проверка.", "Невозможно подтвердить причину изменения."}):
                raise ValueError("Sales foreign-domain statement requires a separate verified check")
    # In this Sales-only slice, foreign-domain dimensions may only be proposed
    # checks, not free-form asserted causes or metrics. Clause-local allowlist
    # prevents an unrelated hypothesis marker from laundering a proven cause.
    for role, text in unbound:
        for clause in re.split(r"[.!?;\n]", text.lower()):
            if not re.search(foreign, clause):
                continue
            if role == "gap_metric" and re.fullmatch(
                r"\s*(?:реклама|маркетинг|цена|цены|остатки|поставки|маржа|прибыль)\s*", clause):
                continue  # missing dimension label, not a factual assertion
            if role.startswith("gap_") and re.fullmatch(
                rf"\s*(?:(?:нет|недостаточно|не хватает) (?:проверенных )?данных о(?:б)? (?:{dimensions})|"
                rf"данные о(?:б)? (?:{dimensions}) (?:недоступны|отсутствуют))\s*", clause):
                continue  # absence of evidence, never a causal finding
            if re.fullmatch(
                r"\s*(?:рекомендуется\s+)?(?:проверить|проверка|нужно проверить|требуется проверить)\s+"
                r"(?:рекламу|маркетинг|цену|цены|остатки|поставки|маржу|прибыль)\s*", clause):
                continue
            raise ValueError("Sales foreign-domain statement requires a separate verified check")
    for finding in result.findings:
        # Requests for another domain belong in cross_agent_requests, not an
        # asserted Sales finding. M4 will supply verified cross-domain evidence.
        if re.search(foreign, " ".join((finding.factor, finding.observation, finding.impact, finding.cause)).lower()):
            raise ValueError("Sales finding has no verified foreign-domain evidence")


def validate_sales_publication(candidate, prepared):
    raw = candidate.model_dump(mode="json") if isinstance(candidate, SalesDiagnosticResult) else candidate
    assert_bounded_safe(raw, max_bytes=MAX_RESULT_BYTES)
    result = SalesDiagnosticResult.model_validate(raw)
    if validate_sales_diagnostic_result(result, prepared["request"]["user_query"]):
        raise ValueError("Sales legacy guardrail failed")
    for key, table in prepared["tables"].items():
        actual = getattr(result, key)
        if actual is None or actual.model_dump(mode="json") != table:
            raise ValueError("Sales authoritative table changed")
    if [fact.model_dump(mode="json") for fact in result.facts] != [
        EvidenceFact.model_validate(fact).model_dump(mode="json") for fact in prepared["facts"]]:
        raise ValueError("Sales authoritative facts changed")
    refs = {item["evidence_ref"] for item in prepared["evidence"]}
    if any(ref not in refs for finding in result.findings for ref in finding.evidence):
        raise ValueError("Sales finding has unknown provenance")
    if any(fact.evidence_ref not in refs for fact in result.facts):
        raise ValueError("Sales fact has unknown provenance")
    _validate_narrative_numbers(result, prepared)
    validate_finding_semantics(result, prepared)
    # Legacy nested schemas are mutable; never share provider-owned instances.
    return SalesDiagnosticResult.model_validate(result.model_dump(mode="json"))
