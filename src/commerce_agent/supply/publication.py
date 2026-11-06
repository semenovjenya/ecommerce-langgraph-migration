"""Evidence-bound Supply publication. Model selects native claims, never numbers."""
from copy import deepcopy
import json
from commerce_agent.sales.contracts import CrossAgentRequest, DataGap, DiagnosticFinding, DriverClass
from commerce_agent.supply.contracts import AgentDiagnosticResult
from commerce_agent.supply.policy import assert_bounded_safe

CHECKS = {
    "sales": "Проверить продажи, заказы и скорость спроса в том же периоде.",
    "marketing": "Проверить рекламный трафик и спрос; влияние запасов не подтверждено.",
    "price": "Проверить цену и скидки; ценовая причина не установлена.",
    "finance": "Проверить себестоимость, маржу и расходы хранения; запас не доказывает прибыльность.",
}
CAUSE = "Влияние запасов на другие домены не установлено этой проверкой."
IMPACT = "Нативный supply-сигнал; финансовые и коммерческие последствия требуют отдельной проверки."
ACTION = "Проверить смежные данные до решения о поставке; никакое действие не выполнено."


def delegated_check(domain, claim):
    if domain not in CHECKS or not claim["sku"] or claim["period"] == "unknown":
        raise ValueError("Supply check requires native SKU and period")
    return CrossAgentRequest(target_agent=domain, reason="Требуется смежная проверка, не установленная причина.",
        question=f"Артикул {claim['sku']}, период/снимок {claim['period']}. {CHECKS[domain]}",
        period=claim["period"], vendor_codes=[claim["sku"]],
        nm_ids=[claim["nm_id"]] if claim["nm_id"] is not None else [], required_checks=[CHECKS[domain]])


def native_gaps(prepared):
    return [DataGap(metric="Границы Supply данных", reason=warning,
        impact="Ограничение сохранено; непроверенный расчёт или причина не публикуются.") for warning in prepared["warnings"]]


def required_finance_indices(prepared):
    indices = prepared.get("required_finance_check_indices", [])
    if (not isinstance(indices, list) or len(indices) > 8
        or any(type(i) is not int or not 0 <= i < len(prepared["claims"]) for i in indices)
        or len(indices) != len(set(indices))):
        raise ValueError("Supply invalid required Finance indices")
    if any(prepared["claims"][i]["operation"] != "value" or not prepared["claims"][i]["sku"]
        or not prepared["claims"][i]["text"].endswith("valuation_status=missing_cogs.") for i in indices):
        raise ValueError("Supply required Finance check has no missing COGS")
    return indices


def assemble_result(narrative, prepared):
    from commerce_agent.supply.alternatives import validate_alternative_prepared
    validate_alternative_prepared(prepared)
    raw = narrative.model_dump(mode="json") if hasattr(narrative, "model_dump") else narrative
    def selected(indices):
        if not indices or len(indices) != len(set(indices)) or any(type(i) is not int or not 0 <= i < len(prepared["claims"]) for i in indices):
            raise ValueError("Supply invalid selection")
        return [prepared["claims"][i] for i in indices]
    findings = []
    for group in raw["findings"]:
        claims = selected(group["claim_indices"])
        findings.append(DiagnosticFinding(factor="Проверенные наблюдения запасов",
            observation="\n".join(c["text"] for c in claims), impact=IMPACT, cause=CAUSE, action=ACTION,
            evidence=list(dict.fromkeys(ref for c in claims for ref in c["evidence_refs"])),
            confidence="low" if any(c.get("confidence") == "low" for c in claims) else "medium", driver_class=DriverClass.NOT_A_DRIVER))
    observed = {line for finding in findings for line in finding.observation.split("\n")}
    mandatory = list(dict.fromkeys(prepared["required_claim_indices"] + prepared.get("required_context_claim_indices", [])))
    for index in mandatory:
        claim = selected([index])[0]
        if claim["text"] not in observed:
            findings.append(DiagnosticFinding(factor="Проверенные наблюдения запасов", observation=claim["text"],
                impact=IMPACT, cause=CAUSE, action=ACTION, evidence=claim["evidence_refs"],
                confidence=claim.get("confidence", "medium"), driver_class=DriverClass.NOT_A_DRIVER))
    requests = [delegated_check("finance", selected([i])[0]) for i in required_finance_indices(prepared)]
    for check in raw["cross_agent_requests"]:
        item = delegated_check(check["domain"], selected([check["claim_index"]])[0])
        if item not in requests and len(requests) < 8: requests.append(item)
    summary_indices = list(dict.fromkeys(prepared.get("required_context_claim_indices", []) + raw["summary_claim_indices"]))
    return AgentDiagnosticResult(agent_name="Supply Agent", summary="\n".join(c["text"] for c in selected(summary_indices)),
        facts=deepcopy(prepared["facts"]), tables=deepcopy(prepared["tables"]), findings=findings,
        recommendations=[ACTION], data_gaps=native_gaps(prepared), cross_agent_requests=requests)


def validate_supply_publication(candidate, prepared):
    from commerce_agent.supply.alternatives import validate_alternative_prepared
    validate_alternative_prepared(prepared)
    assert_bounded_safe(prepared, max_bytes=524288)
    def exact(value):
        # Python structural equality conflates True with 1.0 and False with
        # 0. Native facts/tables must preserve JSON scalar types as well.
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    incoming = candidate.model_dump(mode="json") if hasattr(candidate, "model_dump") else candidate
    assert_bounded_safe(incoming, max_bytes=1048576)
    if (not isinstance(incoming, dict) or exact(incoming.get("facts")) != exact(prepared["facts"])
        or exact(incoming.get("tables")) != exact(prepared["tables"])):
        raise ValueError("Supply changed native facts/matrix before normalization")
    result = AgentDiagnosticResult.model_validate(candidate)
    raw = result.model_dump(mode="json")
    assert_bounded_safe(raw, max_bytes=1048576)
    if (result.agent_name != "Supply Agent" or exact(raw["facts"]) != exact(prepared["facts"]) or exact(raw["tables"]) != exact(prepared["tables"])
        or result.recommendations != [ACTION] or result.data_gaps != native_gaps(prepared) or not result.findings):
        raise ValueError("Supply changed native facts/matrix/limitations")
    proof = {c["text"]: c for c in prepared["claims"]}
    if not result.summary or any(line not in proof for line in result.summary.split("\n")):
        raise ValueError("Supply unproved summary")
    if any(prepared["claims"][i]["text"] not in result.summary.split("\n") for i in prepared.get("required_context_claim_indices", [])):
        raise ValueError("Supply omitted portfolio/time-axis context")
    observed = set()
    for finding in result.findings:
        lines = finding.observation.split("\n")
        if (not lines or any(line not in proof for line in lines) or finding.factor != "Проверенные наблюдения запасов"
            or finding.impact != IMPACT or finding.cause != CAUSE or finding.action != ACTION
            or finding.confidence != ("low" if any(proof[line].get("confidence") == "low" for line in lines) else "medium")
            or finding.driver_class != DriverClass.NOT_A_DRIVER):
            raise ValueError("Supply unproved finding")
        if set(finding.evidence) != {ref for line in lines for ref in proof[line]["evidence_refs"]}:
            raise ValueError("Supply mismatched evidence")
        observed.update(lines)
    if not set(prepared["required_operations"]) <= {proof[line]["operation"] for line in observed}:
        raise ValueError("Supply omitted requested analysis")
    if any(prepared["claims"][i]["text"] not in observed for i in prepared["required_claim_indices"] + prepared.get("required_context_claim_indices", [])):
        raise ValueError("Supply omitted native-priority observations")
    allowed = [delegated_check(domain, claim) for claim in prepared["claims"] if claim["sku"] for domain in CHECKS]
    scopes = {(proof[line]["sku"], proof[line]["period"]) for line in observed}
    if not result.cross_agent_requests or len(result.cross_agent_requests) > 8 or any(check not in allowed or (check.vendor_codes[0], check.period) not in scopes for check in result.cross_agent_requests):
        raise ValueError("Supply delegation lacks scope proof")
    if any(delegated_check("finance", prepared["claims"][i]) not in result.cross_agent_requests
        for i in required_finance_indices(prepared)):
        raise ValueError("Supply missing COGS omitted Finance check")
    return result
