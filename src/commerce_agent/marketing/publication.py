"""Closed evidence-bound publication: model cannot manufacture numbers/causes."""
from copy import deepcopy
from commerce_agent.marketing.contracts import MarketingDiagnosticResult
from commerce_agent.sales.contracts import CrossAgentRequest, DataGap, DiagnosticFinding, DriverClass
from commerce_agent.marketing.policy import assert_bounded_safe

CHECKS = {
    "sales": "Проверить реализованные продажи и воронку по тому же артикулу и периоду; не приписывать продажи рекламе без атрибуции.",
    "finance": "Проверить маржинальность и допустимую рекламную нагрузку по артикулу; ДРР сам по себе не доказывает прибыльность.",
    "price": "Проверить фактическую динамику цены по артикулу и периоду; влияние цены пока не подтверждено.",
    "supply": "Проверить наличие и доступность товара по артикулу и периоду; влияние остатков пока не подтверждено.",
}
CAUSE = "Причина изменения не установлена; рекламная статистика и выручка SKU не доказывают причинную атрибуцию."
IMPACT = "Наблюдение требует проверки эффективности и качества атрибуции; не является решением об изменении рекламы."
ACTION = "Проверить атрибуцию и экономику до изменения ставок или бюджетов."


def delegated_check(domain, claim):
    if domain not in CHECKS or claim["kind"] != "sku" or not claim["sku"]:
        raise ValueError("Marketing delegation requires observed SKU and period")
    return CrossAgentRequest(target_agent=domain, reason="Нужна проверка соседнего домена; причина не подтверждена.",
        question=f"Артикул {claim['sku']}, период {claim['period']}. {CHECKS[domain]}",
        period=claim["period"], vendor_codes=[claim["sku"]],
        nm_ids=[claim["nm_id"]] if claim["nm_id"] is not None else [], required_checks=[CHECKS[domain]])


def native_gaps(prepared):
    return [DataGap(metric="Качество и границы данных", reason=warning,
        impact="Ограничение сохраняется; отсутствие/оценка не заменяются точным фактом.")
        for warning in prepared["warnings"]]


def assemble_result(narrative, prepared):
    raw = narrative.model_dump(mode="json") if hasattr(narrative, "model_dump") else narrative
    catalog = prepared["claims"]
    def selected(indices):
        if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(catalog) for i in indices):
            raise ValueError("Marketing selection is invalid")
        return [catalog[i] for i in indices]
    findings = []
    for group in raw["findings"]:
        claims = selected(group["claim_indices"])
        findings.append(DiagnosticFinding(factor="Проверенные наблюдения рекламы",
            observation="\n".join(c["text"] for c in claims), impact=IMPACT, cause=CAUSE, action=ACTION,
            evidence=list(dict.fromkeys(ref for c in claims for ref in c["evidence_refs"])),
            confidence="medium", driver_class=DriverClass.NOT_A_DRIVER))
    # Policy-owned coverage cannot be delegated to the LLM's preference for
    # convenient rows. Reuse native observations, never synthesize new scalars.
    observed = {line for finding in findings for line in finding.observation.split("\n")}
    for item in prepared.get("sku_priorities", []):
        missing = [i for i in item["claim_indices"] if catalog[i]["text"] not in observed]
        if missing:
            claims = selected(missing)
            findings.append(DiagnosticFinding(factor="Проверенные наблюдения рекламы",
                observation="\n".join(c["text"] for c in claims), impact=IMPACT, cause=CAUSE, action=ACTION,
                evidence=list(dict.fromkeys(ref for c in claims for ref in c["evidence_refs"])),
                confidence="medium", driver_class=DriverClass.NOT_A_DRIVER))
    checks = []
    for item in prepared.get("sku_priorities", []):
        for domain in item["domains"]:
            checks.append(delegated_check(domain, selected([item["claim_indices"][0]])[0]))
    for request in raw["cross_agent_requests"]:
        claim = selected([request["claim_index"]])[0]
        check = delegated_check(request["domain"], claim)
        if check not in checks and len(checks) < 8:
            checks.append(check)
    return MarketingDiagnosticResult(agent_name="Marketing Agent",
        summary="\n".join(c["text"] for c in selected(raw["summary_claim_indices"])),
        facts=deepcopy(prepared["facts"]), tables=deepcopy(prepared["tables"]),
        marketing_report_snapshot=deepcopy(prepared["tables"][0]), findings=findings,
        recommendations=[ACTION], data_gaps=native_gaps(prepared), cross_agent_requests=checks)


def validate_marketing_publication(candidate, prepared):
    assert_bounded_safe(prepared, max_bytes=262144)
    result = MarketingDiagnosticResult.model_validate(candidate)
    raw = result.model_dump(mode="json")
    assert_bounded_safe(raw, max_bytes=524288)
    if (result.agent_name != "Marketing Agent" or raw["facts"] != prepared["facts"]
        or raw["tables"] != prepared["tables"] or raw["marketing_report_snapshot"] != prepared["tables"][0]
        or result.recommendations != [ACTION]
        or result.data_gaps != native_gaps(prepared) or not result.findings):
        raise ValueError("Marketing model changed authoritative data")
    proof = {claim["text"]: claim for claim in prepared["claims"]}
    if not result.summary or any(line not in proof for line in result.summary.split("\n")):
        raise ValueError("Marketing summary has an unproved assertion")
    observed = set()
    for finding in result.findings:
        lines = finding.observation.split("\n")
        if (not lines or any(line not in proof for line in lines)
            or finding.factor != "Проверенные наблюдения рекламы" or finding.impact != IMPACT
            or finding.cause != CAUSE or finding.action != ACTION or finding.confidence != "medium"
            or finding.driver_class != DriverClass.NOT_A_DRIVER):
            raise ValueError("Marketing finding lacks native proof")
        refs = set(ref for line in lines for ref in proof[line]["evidence_refs"])
        if refs != set(finding.evidence):
            raise ValueError("Marketing evidence references mismatch")
        observed.update(lines)
    required = {c["text"] for c in prepared["claims"] if c["kind"] == "portfolio"}
    if not required <= observed or not any(proof[line]["kind"] == "sku" for line in observed) or not any(
        proof[line]["kind"] == "campaign" for line in observed):
        raise ValueError("Marketing omitted available portfolio/SKU/campaign analysis")
    allowed = [delegated_check(domain, claim) for domain in CHECKS for claim in prepared["claims"] if claim["kind"] == "sku"]
    if not result.cross_agent_requests or any(check not in allowed for check in result.cross_agent_requests):
        raise ValueError("Marketing delegation lacks scope proof")
    # A delegated SKU must actually appear in the selected analytical findings.
    scopes = {(proof[line]["sku"], proof[line]["period"]) for line in observed if proof[line]["kind"] == "sku"}
    if any((check.vendor_codes[0], check.period) not in scopes for check in result.cross_agent_requests):
        raise ValueError("Marketing delegates an unobserved scope")
    for item in prepared.get("sku_priorities", []):
        if any(prepared["claims"][i]["text"] not in observed for i in item["claim_indices"]):
            raise ValueError("Marketing omitted priority observations")
        claim = prepared["claims"][item["claim_indices"][0]]
        if any(delegated_check(domain, claim) not in result.cross_agent_requests for domain in item["domains"]):
            raise ValueError("Marketing omitted priority checks")
    if len(result.cross_agent_requests) > 8:
        raise ValueError("Marketing check budget exceeded")
    return result
