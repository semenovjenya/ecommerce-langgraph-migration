"""Finance publication binds every assertion to the native observation catalog."""
from copy import deepcopy
import json
from commerce_agent.finance.contracts import AgentDiagnosticResult
from commerce_agent.finance.policy import assert_bounded_safe
from commerce_agent.finance.evidence import analysis_policy_version
from commerce_agent.sales.contracts import CrossAgentRequest, DataGap, DiagnosticFinding, DriverClass

CHECKS = {"sales": "Проверить продажи, выкупы и возвраты в обоих указанных окнах.",
    "marketing": "Проверить рекламные расходы, ДРР и кампании; реклама не установлена как причина маржи.",
    "price": "Проверить фактическую цену, скидки и ценовые ограничения; ценовая причина не подтверждена.",
    "supply": "Проверить остатки, хранение и доступность; финансовые расходы не доказывают overstock."}
FACTOR = "Проверенные финансовые наблюдения"
CAUSE = "Нативная финансовая механика; первопричина изменения не установлена без смежных проверок."
IMPACT = "Маржа маркетплейса не равна чистой прибыли или ликвидности; версия COGS требует отдельной проверки."
ACTION = "Проверить смежные данные до решения; внешние действия не выполнены."


def delegated_check(domain, claim):
    if domain not in CHECKS or not claim["sku"] or claim["period"] == "unknown":
        raise ValueError("Finance cross-check needs an observed SKU and period")
    return CrossAgentRequest(target_agent=domain, reason="Смежная проверка, не подтверждённая причина.",
        question=f"Артикул {claim['sku']}, период {claim['period']}. {CHECKS[domain]}",
        vendor_codes=[claim["sku"]], nm_ids=[claim["nm_id"]] if claim["nm_id"] is not None else [],
        period=claim["period"], required_checks=[CHECKS[domain]])


def native_gaps(prepared):
    return [DataGap(metric="Границы Finance данных", reason=warning,
        impact="Ограничение сохранено; неподтверждённая прибыльность/причина не заявляется.") for warning in prepared["warnings"]]


def _select(indices, prepared):
    if not indices or len(indices) != len(set(indices)) or any(type(index) is not int or not 0 <= index < len(prepared["claims"]) for index in indices):
        raise ValueError("Finance invalid observation selection")
    return [prepared["claims"][index] for index in indices]


def _finding(claims):
    return DiagnosticFinding(factor=FACTOR, observation="\n".join(claim["text"] for claim in claims),
        cause=CAUSE, impact=IMPACT, action=ACTION, confidence="medium", driver_class=DriverClass.NOT_A_DRIVER,
        evidence=list(dict.fromkeys(ref for claim in claims for ref in claim["evidence_refs"])))


def assemble_result(narrative, prepared):
    version = analysis_policy_version(prepared)
    raw = narrative.model_dump(mode="json") if hasattr(narrative, "model_dump") else narrative
    findings = []
    observed = set()
    groups = []
    if version == 2:
        # Keep analytical topics visible even when the model chooses a generic
        # first row. Each native observation is published at most once.
        groups.extend(prepared["analytical_groups"].values())
    groups.extend(group["claim_indices"] for group in raw["findings"])
    for indices in groups:
        if not indices:
            continue
        claims = _select(indices, prepared)
        if version == 2:
            claims = [claim for claim in claims if claim["text"] not in observed]
        if claims:
            findings.append(_finding(claims))
            observed.update(claim["text"] for claim in claims)
    observed = {line for finding in findings for line in finding.observation.split("\n")}
    missing = [index for index in prepared["required_claim_indices"] if prepared["claims"][index]["text"] not in observed]
    if missing:
        findings.append(_finding(_select(missing, prepared)))
        if version == 2:
            observed.update(prepared["claims"][index]["text"] for index in missing)
    summary = list(dict.fromkeys(prepared.get("summary_claim_indices", prepared["required_claim_indices"][:3]) + raw["summary_claim_indices"]))
    requests = [delegated_check(check["domain"], _select([check["claim_index"]], prepared)[0]) for check in raw["cross_agent_requests"]]
    # Every proposed neighbor check must be visible in the published analysis.
    for check in raw["cross_agent_requests"]:
        claim = prepared["claims"][check["claim_index"]]
        if claim["text"] not in observed:
            findings.append(_finding([claim]))
            observed.add(claim["text"])
    # Material native priorities must not disappear because the model chose an
    # easier first row. Requests remain proposals; M4 will execute them.
    for index in prepared["priority_claim_indices"][:2]:
        claim = prepared["claims"][index]
        domain = "marketing" if claim["operation"] == "sku" else "sales"
        check = delegated_check(domain, claim)
        if check not in requests:
            requests.append(check)
    if version == 2:
        unique_requests = []
        for check in requests:
            if check not in unique_requests:
                unique_requests.append(check)
        requests = unique_requests
    return AgentDiagnosticResult(agent_name="Finance Agent", summary="\n".join(claim["text"] for claim in _select(summary, prepared)),
        facts=deepcopy(prepared["facts"]), tables=deepcopy(prepared["tables"]), findings=findings,
        recommendations=[ACTION], data_gaps=native_gaps(prepared), cross_agent_requests=requests)


def validate_finance_publication(candidate, prepared):
    version = analysis_policy_version(prepared)
    assert_bounded_safe(prepared, max_bytes=4_000_000)
    if isinstance(candidate, dict) and set(candidate) - set(AgentDiagnosticResult.model_fields):
        raise ValueError("Finance unknown publication fields")
    def exact(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    incoming = candidate.model_dump(mode="json") if hasattr(candidate, "model_dump") else candidate
    assert_bounded_safe(incoming, max_bytes=4_000_000)
    # Allow omitted historical DTO defaults but never normalize supplied
    # native scalars (notably False -> 0) before checking their provenance.
    for field in ("facts", "tables"):
        supplied = incoming.get(field) if isinstance(incoming, dict) else None
        expected = prepared[field]
        if not isinstance(supplied, list) or len(supplied) != len(expected):
            raise ValueError("Finance native facts/matrix length changed")
        for before, after in zip(supplied, expected):
            if not isinstance(before, dict) or any(key not in after or exact(value) != exact(after[key]) for key, value in before.items()):
                raise ValueError("Finance native facts/matrix scalar changed before normalization")
    result = AgentDiagnosticResult.model_validate(candidate)
    raw = result.model_dump(mode="json")
    assert_bounded_safe(raw, max_bytes=4_000_000)
    if (result.agent_name != "Finance Agent" or exact(raw["facts"]) != exact(prepared["facts"]) or exact(raw["tables"]) != exact(prepared["tables"])
            or result.recommendations != [ACTION] or result.data_gaps != native_gaps(prepared) or not result.findings):
        raise ValueError("Finance changed native facts/matrix/limitations")
    proof = {claim["text"]: claim for claim in prepared["claims"]}
    if not result.summary or any(line not in proof for line in result.summary.split("\n")):
        raise ValueError("Finance unproved summary")
    if any(prepared["claims"][index]["text"] not in result.summary.split("\n")
           for index in prepared.get("summary_claim_indices", prepared["required_claim_indices"][:3])):
        raise ValueError("Finance omitted mandatory summary context")
    observed = set()
    for finding in result.findings:
        lines = finding.observation.split("\n")
        if any(line not in proof for line in lines) or finding != _finding([proof[line] for line in lines]):
            raise ValueError("Finance unproved observation/cause/impact/evidence")
        if version == 2 and (len(lines) != len(set(lines)) or observed.intersection(lines)):
            raise ValueError("Finance duplicate published observation")
        observed.update(lines)
    if not set(prepared["required_operations"]) <= {proof[line]["operation"] for line in observed}:
        raise ValueError("Finance omitted requested native capability")
    if any(prepared["claims"][index]["text"] not in observed for index in prepared["required_claim_indices"]):
        raise ValueError("Finance omitted portfolio/priority observation")
    allowed = [delegated_check(domain, proof[line]) for line in observed
               if proof[line]["sku"] and proof[line]["period"] != "unknown" for domain in CHECKS]
    if len(result.cross_agent_requests) > 8 or any(check not in allowed for check in result.cross_agent_requests):
        raise ValueError("Finance unscoped/unverified delegation")
    if version == 2:
        for index, check in enumerate(result.cross_agent_requests):
            if check in result.cross_agent_requests[:index]:
                raise ValueError("Finance duplicate delegated check")
    for index in prepared["priority_claim_indices"][:2]:
        claim = prepared["claims"][index]
        if delegated_check("marketing" if claim["operation"] == "sku" else "sales", claim) not in result.cross_agent_requests:
            raise ValueError("Finance omitted priority delegation")
    return result
