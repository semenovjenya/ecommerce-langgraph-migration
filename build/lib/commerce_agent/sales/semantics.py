"""Evidence-bound presentation and proposed checks, never business calculations."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from commerce_agent.sales.claims import native_claim_catalog, native_claim_proofs, render_sales_claim, claim_scope
from commerce_agent.sales.analytics import analytical_claims


CheckKind = Literal["source_completeness", "period_comparability", "funnel",
                    "marketing", "price", "supply", "finance"]


class SalesRequestedCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: CheckKind
    claim_index: int | None = Field(default=None, ge=0, le=2000, strict=True)


_CHECK_TEXT = {
    "source_completeness": "Проверить полноту источника",
    "period_comparability": "Проверить сопоставимость периодов",
    "funnel": "Запросить отдельную проверку воронки",
    "marketing": "Проверить рекламу (показы, клики, расходы, ДРР и заказы)",
    "price": "Проверить цены (цену продавца, скидки и промо)",
    "supply": "Проверить остатки и поставки (наличие, дефицит и поступления)",
    "finance": "Проверить маржу и прибыль (себестоимость, комиссии и логистику)",
}

NEUTRAL_SUMMARIES = {"", "Продажи проверены по данным источника.",
                     "Продажи проверены по источнику.", "Данные проверены.", "Проверены доступные данные."}
_FOREIGN_GAPS = {"marketing": ("Реклама", "Нет проверенных данных о рекламе."),
                 "price": ("Цены", "Нет проверенных данных о ценах."),
                 "supply": ("Остатки", "Нет проверенных данных об остатках."),
                 "finance": ("Маржа", "Нет проверенных данных о марже.")}


def render_domain_gap(domain: str) -> dict:
    metric, reason = _FOREIGN_GAPS[domain]
    return {"metric": metric, "reason": reason, "impact": "Нужна дополнительная проверка."}


def render_delegated_check(check: SalesRequestedCheck, prepared: dict) -> dict:
    if check.kind not in _FOREIGN_GAPS:
        raise ValueError("Sales delegated check needs a foreign domain")
    question = render_requested_check(check, prepared)
    if check.claim_index is None:
        raise ValueError("Sales delegated check needs a native SKU and period")
    sku, period = claim_scope(native_claim_catalog(prepared)[check.claim_index], prepared)
    if not sku or sku == "Артикул не сопоставлен":
        raise ValueError("Sales delegated check needs a native SKU and period")
    return {"target_agent": check.kind, "reason": "Нужна отдельная проверка домена.",
            "question": question, "period": period, "vendor_codes": [sku],
            "nm_ids": [], "required_checks": [question]}


def render_requested_check(check: SalesRequestedCheck, prepared: dict) -> str:
    text = _CHECK_TEXT[check.kind]
    if check.claim_index is not None:
        catalog = native_claim_catalog(prepared)
        if not 0 <= check.claim_index < len(catalog):
            raise ValueError("Sales check has no native scope")
        sku, period = claim_scope(catalog[check.claim_index], prepared)
        text += f" для {'SKU ' + sku if sku else 'портфеля'} за период {period}"
    return text + "."


def check_proofs(prepared: dict) -> dict[str, tuple[str, str] | None]:
    proofs = {text + ".": None for text in _CHECK_TEXT.values()}
    for claim in native_claim_catalog(prepared):
        sku, period = claim_scope(claim, prepared)
        for kind in _CHECK_TEXT:
            text = f"{_CHECK_TEXT[kind]} для {'SKU ' + sku if sku else 'портфеля'} за период {period}."
            proofs[text] = (sku, period)
    return proofs


def observation_claims(observation: str, prepared: dict):
    proofs = {}
    for claim in native_claim_catalog(prepared):
        proofs.setdefault(render_sales_claim(claim, prepared), []).append(claim)
    lines = [line for line in observation.splitlines() if line.strip()]
    if not lines or any(line not in proofs for line in lines):
        return []
    return [claim for line in lines for claim in proofs[line]]


def finding_presentation(observation: str, prepared: dict) -> tuple[str, str, str]:
    """Reuse legacy fact mapping: confirmed facts are secondary, not refuted.

    See prompt_pipeline._fact_as_finding. Neither an amount nor comparison
    proves primary attribution. Confidence is capped at native fact confidence.
    """
    claims = observation_claims(observation, prepared)
    if not claims:
        return "Наблюдение", "secondary_driver", "low"
    analytic = analytical_claims(prepared)
    native = [claim for claim in claims if claim.kind != "analytic"]
    families = {"выручки" if "revenue" in prepared["facts"][claim.fact_index]["metric"] else "продаж"
                for claim in native}
    labels = []
    for family, amount_label in (("выручки", "Выручка"), ("продаж", "Продажи")):
        if family not in families:
            continue
        dynamic = any(claim.kind == "comparison" and
                      ("выручки" if "revenue" in prepared["facts"][claim.fact_index]["metric"] else "продаж") == family
                      for claim in native)
        labels.append(f"Динамика {family}" if dynamic else amount_label)
    factor = "; ".join(labels)
    labels.extend(analytic[claim.fact_index]["factor"] for claim in claims if claim.kind == "analytic")
    factor = "; ".join(dict.fromkeys(labels))
    levels = [prepared["facts"][index].get("confidence", "medium") for claim in native
              for index in (claim.fact_index, claim.previous_fact_index) if index is not None]
    levels.extend(analytic[claim.fact_index]["confidence"] for claim in claims if claim.kind == "analytic")
    confidence = min(levels, key={"low": 0, "medium": 1, "high": 2}.get)
    return factor, "secondary_driver", confidence


def signal_catalog(prepared: dict) -> list[dict]:
    """Whole sourced statements about returned data, not inferred business causes."""
    signals = []
    for scope in prepared.get("overview_signals", []):
        refs = scope["evidence_refs"]
        for field, name in (("sales_raw_rows", "строки продаж"), ("conversion_watchlist", "строки воронки")):
            presence = "есть" if scope[field] else "не возвращены"
            statement = f"В полученной выборке {presence} {name}."
            if refs:
                signals.append({"statement": statement, "evidence_refs": refs})
    return signals


def validate_finding_semantics(result, prepared: dict) -> None:
    proofs = check_proofs(prepared)
    signals = {}
    for signal in signal_catalog(prepared):
        signals.setdefault(signal["statement"], set()).update(signal["evidence_refs"])
    summary_lines = [line for line in result.summary.splitlines() if line.strip()]
    native_proofs = native_claim_proofs(prepared)
    sourced_summary = bool(summary_lines) and all(line in native_proofs for line in summary_lines)
    if result.summary not in NEUTRAL_SUMMARIES and not sourced_summary:
        # A whole explicitly unconfirmed hypothesis is not a factual assertion.
        import re
        if not re.fullmatch(r"Гипотеза \(не подтверждена\): [^.!?;\n]+\.?", result.summary):
            raise ValueError("Sales summary has no sourced proposition")
    gap_metrics = {"Реклама", "Маркетинг", "Цены", "Остатки", "Поставки", "Маржа", "Прибыль"}
    if any(gap.metric not in gap_metrics for gap in result.data_gaps):
        raise ValueError("Sales gap dimension is not a closed label")
    for request in result.cross_agent_requests:
        scope = proofs.get(request.question)
        if request.question not in proofs or request.reason != "Нужна отдельная проверка домена.":
            raise ValueError("Sales delegated request is not a typed check")
        candidates = [] if scope is None else [
            SalesRequestedCheck(kind=request.target_agent.value, claim_index=index)
            for index, claim in enumerate(native_claim_catalog(prepared))
            if claim_scope(claim, prepared) == scope and scope[0] is not None]
        if not any(render_delegated_check(check, prepared) == request.model_dump(mode="json") for check in candidates):
            raise ValueError("Sales delegated request scope has no native proof")
    if any(text not in proofs for text in result.recommendations):
        raise ValueError("Sales recommendation is not a typed requested check")
    for finding in result.findings:
        expected = finding_presentation(finding.observation, prepared)
        if not observation_claims(finding.observation, prepared):
            if finding.observation not in signals or not signals[finding.observation].intersection(finding.evidence):
                raise ValueError("Sales qualitative observation has no sourced signal")
        if (finding.factor, finding.driver_class.value, finding.confidence) != expected:
            raise ValueError("Sales finding label/classification has no matching native proof")
        if finding.cause != "Не устанавливается этой проверкой.":
            raise ValueError("Sales finding has no verified causal classification")
        if finding.impact != "Нужна дополнительная проверка.":
            raise ValueError("Sales finding impact has no verified business proof")
        if finding.action not in proofs:
            raise ValueError("Sales action is not a typed requested check")
        scope = proofs[finding.action]
        if scope is not None:
            observed = {claim_scope(claim, prepared) for claim in observation_claims(finding.observation, prepared)}
            observed.update((prepared["facts"][claim.previous_fact_index]["sku"],
                             prepared["facts"][claim.previous_fact_index]["period"])
                            for claim in observation_claims(finding.observation, prepared)
                            if claim.kind != "analytic" and claim.previous_fact_index is not None)
            if scope not in observed:
                raise ValueError("Sales finding check scope differs from its observation")
    validate_analytical_coverage(result, prepared)


def validate_analytical_coverage(result, prepared):
    """Missing analysis is a typed failure, never a successful empty report.

    Only activate the portfolio quality gate when typed business comparisons
    exist. Historical requests and native fixtures without those capabilities
    keep their explicit limitations instead of inventing unavailable windows.
    """
    analytic = analytical_claims(prepared)
    portfolio = {c["text"] for c in analytic if c["sku"] is None}
    if not portfolio:
        return
    observed = {line for finding in result.findings for line in finding.observation.splitlines()}
    if not portfolio <= observed or not portfolio <= set(result.summary.splitlines()):
        raise ValueError("Sales analysis omits available portfolio dynamics")
    for factor in ("Динамика выручки", "Динамика продаж", "Воронка"):
        native = {c["text"] for c in analytic if c["sku"] is not None and c["factor"] == factor}
        if native and not native.intersection(observed):
            raise ValueError("Sales analysis omits available SKU or funnel evidence")
    if any(c["sku"] is not None for c in analytic) and not result.cross_agent_requests:
        raise ValueError("Sales analysis omits scoped domain checks")
