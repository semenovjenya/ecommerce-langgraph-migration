"""Price output guard: native matrices, closed observations and scoped checks."""
from copy import deepcopy
import json
from commerce_agent.price.analysis import verify_price_catalog
from commerce_agent.price.contracts import AgentDiagnosticResult
from commerce_agent.price.policy import assert_bounded_safe
from commerce_agent.sales.contracts import CrossAgentRequest, DataGap, DiagnosticFinding, DriverClass

CHECKS = {
    "sales": "Проверить спрос, заказы и выкупы; цена не установлена как причина динамики.",
    "marketing": "Проверить рекламу и видимость; ценовое наблюдение не доказывает рекламную причину.",
    "supply": "Проверить доступность и остатки; ценовое наблюдение не доказывает дефицит.",
    "finance": "Проверить версию COGS и финансовые ограничения; сценарная прибыль не является фактической.",
}
CAUSE = "Причинность и эффект переоценки не установлены; raw кандидаты не заменяют строгие аналоги."
IMPACT = "Учитывать независимые даты и ограничения; сценарий не является текущим разрешением изменить цену."
ACTION = "Выполнить смежные read-only проверки до решения; цены и внешние системы не изменены."


def _select(indices, prepared):
    if not isinstance(indices, list) or not indices or len(indices) != len(set(indices)) or any(
        type(index) is not int or not 0 <= index < len(prepared["claims"]) for index in indices):
        raise ValueError("Price invalid observation selection")
    return [prepared["claims"][index] for index in indices]


def delegated_check(domain, claim):
    if domain not in CHECKS or not claim["sku"] or not claim["check_period"]:
        raise ValueError("Price delegation requires observed exact SKU and dated check scope")
    return CrossAgentRequest(target_agent=domain, reason="Смежная проверка, не подтверждённая причина.",
        question=f"Артикул {claim['sku']}, период проверки {claim['check_period']}. {CHECKS[domain]} "
            "Для latest источника это проверка на дату среза, не утверждение окна исходной метрики.",
        period=claim["check_period"], vendor_codes=[claim["sku"]], nm_ids=[claim["nm_id"]],
        required_checks=[CHECKS[domain]])


def _finding(claims):
    return DiagnosticFinding(factor="Проверенные ценовые наблюдения", observation="\n".join(claim["text"] for claim in claims),
        cause=CAUSE, impact=IMPACT, action=ACTION, confidence="low", driver_class=DriverClass.NOT_A_DRIVER,
        evidence=list(dict.fromkeys(ref for claim in claims for ref in claim["evidence_refs"])))


def native_gaps(prepared):
    return [DataGap(metric="Границы ценовых данных", reason=warning,
        impact="Ограничение сохранено; свежесть, причинность и допустимость переоценки не выдумываются.") for warning in prepared["warnings"]]


def mandatory_checks(prepared):
    # Snapshot dates only define narrow verification requests, not a fictitious
    # 7-day window for scenario inputs. Daily uses the actual native period.
    checks = []
    for indices in (prepared["analytical_groups"]["scenarios"][:1], prepared["analytical_groups"]["daily"][:1],
                    prepared["analytical_groups"]["layers"][:1]):
        for index in indices:
            claim = prepared["claims"][index]
            if not claim["sku"] or not claim["check_period"]:
                continue
            domain = "finance" if claim["operation"] == "decision" else "sales"
            check = delegated_check(domain, claim)
            if check not in checks:
                checks.append(check)
    return checks


def assemble_result(narrative, prepared):
    verify_price_catalog(prepared)
    raw = narrative.model_dump(mode="json") if hasattr(narrative, "model_dump") else narrative
    selected = []
    for indices in [*prepared["analytical_groups"].values(), *(group["claim_indices"] for group in raw["findings"])]:
        if indices:
            selected.append(_select(indices, prepared))
    selected.append(_select(prepared["required_claim_indices"], prepared))
    requests = mandatory_checks(prepared)
    for check in raw["cross_agent_requests"]:
        claim = _select([check["claim_index"]], prepared)[0]
        candidate = delegated_check(check["domain"], claim)
        if candidate not in requests:
            requests.append(candidate)
        selected.append([claim])  # Every requested check is also an observed finding.
    findings, observed = [], set()
    for group in selected:
        new = [claim for claim in group if claim["text"] not in observed]
        if new:
            findings.append(_finding(new))
            observed.update(claim["text"] for claim in new)
    summary_indices = list(dict.fromkeys(prepared["summary_claim_indices"] + raw["summary_claim_indices"]))
    summary = "\n".join(claim["text"] for claim in _select(summary_indices, prepared))
    return AgentDiagnosticResult(agent_name="Price Agent", summary=summary,
        facts=deepcopy(prepared["facts"]), tables=deepcopy(prepared["tables"]), findings=findings,
        recommendations=[ACTION], data_gaps=native_gaps(prepared), cross_agent_requests=requests)


def validate_price_publication(candidate, prepared):
    verify_price_catalog(prepared)
    if isinstance(candidate, dict) and set(candidate) - set(AgentDiagnosticResult.model_fields):
        raise ValueError("Price foreign publication fields")
    def exact(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    incoming = candidate.model_dump(mode="json") if hasattr(candidate, "model_dump") else candidate
    assert_bounded_safe(incoming, max_bytes=4_000_000)
    # Historical sparse DTOs may omit defaults, not change a supplied native
    # scalar. Check before Pydantic can normalize False into numeric zero.
    for field in ("facts", "tables"):
        supplied = incoming.get(field) if isinstance(incoming, dict) else None
        expected = prepared[field]
        if not isinstance(supplied, list) or len(supplied) != len(expected):
            raise ValueError("Price native facts/matrix length changed")
        for before, after in zip(supplied, expected):
            if not isinstance(before, dict) or any(key not in after or exact(value) != exact(after[key]) for key, value in before.items()):
                raise ValueError("Price native scalar changed before normalization")
    result = AgentDiagnosticResult.model_validate(candidate)
    raw = result.model_dump(mode="json")
    assert_bounded_safe(raw, max_bytes=4_000_000)
    if (result.agent_name != "Price Agent" or exact(raw["facts"]) != exact(prepared["facts"]) or exact(raw["tables"]) != exact(prepared["tables"])
        or result.recommendations != [ACTION] or result.data_gaps != native_gaps(prepared) or not result.findings):
        raise ValueError("Price changed native facts/matrices/limitations")
    proof = {claim["text"]: claim for claim in prepared["claims"]}
    lines = result.summary.split("\n")
    if not result.summary or len(lines) != len(set(lines)) or any(line not in proof for line in lines):
        raise ValueError("Price unproved or duplicate summary")
    if any(prepared["claims"][index]["text"] not in lines for index in prepared["summary_claim_indices"]):
        raise ValueError("Price omitted required summary topics")
    observed = set()
    for finding in result.findings:
        lines = finding.observation.split("\n")
        if not lines or any(line not in proof for line in lines) or finding != _finding([proof[line] for line in lines]):
            raise ValueError("Price unproved observation/cause/impact/evidence")
        if len(lines) != len(set(lines)) or observed.intersection(lines):
            raise ValueError("Price duplicate observation")
        observed.update(lines)
    if any(prepared["claims"][index]["text"] not in observed for index in prepared["required_claim_indices"]):
        raise ValueError("Price omitted required layer/scenario/quality observation")
    if not set(prepared["required_operations"]) <= {proof[line]["operation"] for line in observed}:
        raise ValueError("Price omitted requested capability")
    allowed = [delegated_check(domain, proof[line]) for line in observed if proof[line]["sku"] and proof[line]["check_period"] for domain in CHECKS]
    if len(result.cross_agent_requests) > 8 or any(check not in allowed for check in result.cross_agent_requests):
        raise ValueError("Price unscoped or unproved delegation")
    if any(check in result.cross_agent_requests[:index] for index, check in enumerate(result.cross_agent_requests)):
        raise ValueError("Price duplicate delegation")
    if any(check not in result.cross_agent_requests for check in mandatory_checks(prepared)):
        raise ValueError("Price omitted mandatory adjacent verification")
    return result
