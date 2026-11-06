"""Versioned pure SKU/list coverage, independent from portfolio deep dives.

Only exact native receipts are accepted. No filtering, aggregation or business
calculation is performed here. Unmarked historical checkpoints keep their old
reader and are never silently enrolled in the stricter publication policy.
"""
from commerce_agent.sales.targeted import validate_native_scopes, native_scope_claims


def scoped_scopes(prepared):
    version = prepared.get("scoped_sales_policy_version")
    request = prepared["request"]
    has_scoped_facts = request["scope"] in {"single_sku", "sku_list"} and any(
        fact.get("target_type") == "sku" for fact in prepared["facts"])
    if version is None:
        if "scoped_sales_scopes" in prepared or has_scoped_facts:
            raise ValueError("Sales scoped source cannot lose its policy version")
        return []
    scopes = prepared.get("scoped_sales_scopes")
    if (type(version) is not int or version not in (1, 2, 3)
        or request["scope"] not in {"single_sku", "sku_list"}
        or not request["targets"] or not isinstance(scopes, list)):
        raise ValueError("Sales unsupported pure scoped source policy")
    validated = validate_native_scopes(prepared, scopes)
    # The main scoped tables, unlike additional portfolio targets, are public
    # authoritative tables. Bind every returned row to its own native read.
    from commerce_agent.sales.targeted import canonical
    if not validated:
        raise ValueError("Sales pure scoped source requires an available native scope")
    for key, table in validated[0]["tables"].items():
        if any(scope["tables"][key]["columns"] != table["columns"] for scope in validated):
            raise ValueError("Sales scoped source table layouts differ")
        expected = {"title": table["title"],
            "columns": (["Запрошенный артикул", *table["columns"]] if request["scope"] == "sku_list" else table["columns"]),
            "rows": [([scope["target"], *row] if request["scope"] == "sku_list" else row)
                     for scope in validated for row in scope["tables"][key]["rows"]]}
        if canonical(prepared["tables"].get(key)) != canonical(expected):
            raise ValueError("Sales scoped table differs from exact native scopes")
    return validated


def scoped_claims(prepared):
    return native_scope_claims(prepared, scoped_scopes(prepared),
        scope_label="строка сравнения SKU" if prepared.get("scoped_sales_policy_version") in (2, 3) else None)


def required_scoped_claims(prepared):
    claims = scoped_claims(prepared)
    return [next(claim for claim in claims
                 if claim["requested_target"] == scope["target"] and claim["factor"] == factor)
            for scope in scoped_scopes(prepared)
            for factor in ("Динамика выручки", "Динамика продаж")]


def assemble_required_scoped(raw, prepared):
    """Policy2/3 guarantee native coverage; publication stays strict.

    Model priorities remain visible. Only missing service-owned observations
    and neutral scoped read-only checks are added. Policy1 replay is unchanged.
    No scalar, delta, rate, causal claim or aggregate is generated here.
    """
    if prepared.get("scoped_sales_policy_version") not in (2, 3):
        return raw
    from copy import deepcopy
    from commerce_agent.sales.claims import native_claim_catalog, render_sales_claim
    from commerce_agent.sales.semantics import (finding_presentation, SalesRequestedCheck,
        render_requested_check, render_delegated_check, check_proofs)
    result = deepcopy(raw)
    if prepared["scoped_sales_policy_version"] == 3:
        from commerce_agent.sales.targeted import canonical
        seen,requests=set(),[]
        for request in result["cross_agent_requests"]:
            key=canonical(request)
            if key not in seen:
                requests.append(request);seen.add(key)
        result["cross_agent_requests"]=requests
    required = required_scoped_claims(prepared)
    catalog = native_claim_catalog(prepared)
    indices = {render_sales_claim(claim, prepared): index for index, claim in enumerate(catalog)}
    lines = result["summary"].splitlines()
    # A neutral fallback is not a native quantitative proposition. Remove only
    # that exact known fallback before adding verified lines, never free prose.
    if lines == ["Проверены доступные данные."]:
        lines = []
    result["summary"] = "\n".join(dict.fromkeys([*lines, *(claim["text"] for claim in required)]))
    observed = {line for finding in result["findings"] for line in finding["observation"].splitlines()}
    proof = check_proofs(prepared)
    proposed = {proof.get(request["question"]) for request in result["cross_agent_requests"]}
    for scope in scoped_scopes(prepared):
        claims = [claim for claim in required if claim["requested_target"] == scope["target"]]
        missing = [claim for claim in claims if claim["text"] not in observed]
        if missing:
            text = "\n".join(claim["text"] for claim in missing)
            factor, driver, confidence = finding_presentation(text, prepared)
            result["findings"].append({"factor": factor, "driver_class": driver, "confidence": confidence,
                "observation": text, "evidence": list(dict.fromkeys(ref for c in missing for ref in c["evidence_refs"])),
                "cause": "Не устанавливается этой проверкой.", "impact": "Нужна дополнительная проверка.",
                "action": render_requested_check(SalesRequestedCheck(kind="source_completeness", claim_index=indices[missing[0]["text"]]), prepared)})
            observed.update(c["text"] for c in missing)
        for claim in claims:
            key = claim["sku"], claim["period"]
            if key not in proposed:
                result["cross_agent_requests"].append(render_delegated_check(
                    SalesRequestedCheck(kind="marketing", claim_index=indices[claim["text"]]), prepared))
                proposed.add(key)
    if len(result["cross_agent_requests"]) > 40:
        raise ValueError("Sales scoped checks exceed finite publication budget")
    return result
