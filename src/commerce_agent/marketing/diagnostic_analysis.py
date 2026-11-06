"""Closed native diagnostic catalog, publication and readable domain report."""
from copy import deepcopy
from datetime import date, timedelta

from commerce_agent.marketing.contracts import MarketingDiagnosticResult
from commerce_agent.marketing.evidence import _table
from commerce_agent.marketing.policy import assert_bounded_safe
from commerce_agent.marketing.publication import ACTION, CAUSE, IMPACT, delegated_check, native_gaps
from commerce_agent.sales.contracts import DiagnosticFinding, DriverClass, EvidenceFact

BASE_FIELDS = ("request", "today", "sku", "nm_id", "requested_period", "source_windows",
               "evidence", "warnings", "campaign_row_windows", "campaign_totals_window", "native_data")


def build_diagnostic_catalog(native):
    p = {key: deepcopy(native[key]) for key in BASE_FIELDS}
    # Old masked checkpoints remain readable. Missing provenance does not
    # silently authorize a new source for development debug export.
    p["diagnostic_source_mode"] = native.get("diagnostic_source_mode", "unknown")
    p["capability"] = "sku_diagnostic"
    p["diagnostic_policy_version"] = 1
    p["claims"], p["facts"] = [], []
    data = p["native_data"]
    sku = data["sku"]["rows"][0]
    current, previous = p["requested_period"], f"{data['sku']['previous_date_from']}/{data['sku']['previous_date_to']}"
    by_source = {e["source"]: e["evidence_ref"] for e in p["evidence"] if e["claim"] == "sku"}

    def add(family, metric, value, unit, period, label, *, source, target_type, target_id,
            vendor=None, source_date=None, selectable=True, confirmed=True, paired_source=None):
        refs = [e["evidence_ref"] for e in p["evidence"] if e["claim"] == family and e["source"] == source]
        if not refs:
            return None  # Missing source does not create a manufactured fact.
        sources = [source]
        if paired_source:
            paired = [e["evidence_ref"] for e in p["evidence"] if e["claim"] == family and e["source"] == paired_source]
            if paired:
                refs.extend(paired)
                sources.append(paired_source)
            else:
                confirmed = False
        text = f"{label}: {'нет данных' if value is None else value} {unit}; период/срез {period or 'неизвестен'}."
        fact = EvidenceFact(metric=metric, value=value, unit=unit, period=period or "неизвестен",
            sku=vendor, target_type=target_type, target_id=str(target_id), source_table=" + ".join(sources),
            source_snapshot_at=source_date, evidence_ref=refs[0], source_agent="marketing",
            metric_definition=label, status="confirmed" if confirmed and value is not None and period else "unconfirmed",
            confidence="medium" if confirmed and value is not None and period else "low")
        p["facts"].append(fact.model_dump(mode="json"))
        if not selectable or not confirmed or value is None or not period:
            return None
        claim = {"kind": "sku" if family == "sku" else family, "metric": metric, "text": text,
            "period": period, "sku": vendor, "nm_id": p["nm_id"] if vendor else None,
            "evidence_refs": refs}
        p["claims"].append(claim)
        return len(p["claims"]) - 1

    required = []
    for metric, label, unit, sales_metric in (
        ("spend_rub", "Расход рекламы SKU", "RUB", False),
        ("impressions", "Показы в статистике рекламы SKU", "impressions", False),
        ("clicks", "Клики в статистике рекламы SKU", "clicks", False),
        ("ad_orders", "Заказы в статистике рекламы SKU, не все продажи", "orders", False),
        ("sales_units", "Все реализованные продажи SKU, не рекламная атрибуция", "units", True),
        ("sales_revenue_rub", "Вся реализованная выручка SKU, не рекламная атрибуция", "RUB", True),
        ("estimated_drr_pct", "Оценочный ДРР ко всей выручке SKU, не ROI/прибыль", "%", False),
        ("ctr_pct", "Native CTR SKU", "%", False),
        ("cpc_rub", "Native CPC SKU, не средний campaign CPC", "RUB/click", False),
        ("campaigns", "Кампании по текущим связям SKU, не доказанное историческое участие", "campaigns", False),
    ):
        source = "sales.wb_supplier_sales" if sales_metric else "marketing.wb_marketing_campaign_stats_daily"
        # DRR requires both observed source families; no missing sales source
        # can support an exact ratio claim even if native data coalesces zero.
        eligible = metric != "estimated_drr_pct" or "sales.wb_supplier_sales" in by_source
        pair = []
        for prefix, period in (("current_", current), ("previous_", previous)):
            index = add("sku", prefix+metric, sku[prefix+metric], unit, period,
                f"{label}, SKU {p['sku']} ({sku['attribution_quality']}; не доказательство причинного эффекта)",
                source=source, target_type="sku", target_id=p["sku"], vendor=p["sku"], selectable=eligible, confirmed=eligible,
                paired_source="sales.wb_supplier_sales" if metric == "estimated_drr_pct" else None)
            if index is not None:
                pair.append(index)
        # Group paired observations into one catalog item without recomputing
        # deltas. All underlying facts remain in the published native matrix.
        if pair:
            text = " ".join(p["claims"][i]["text"] for i in pair)
            first = pair[0]
            p["claims"][first]["text"] = text
            for i in reversed(pair[1:]):
                p["claims"].pop(i)
            required.append(first)
    # Preserve every native precomputed delta with its combined two-window grain.
    for metric, unit in (("spend_delta_rub","RUB"), ("spend_delta_pct","%"),
        ("impressions_delta","impressions"), ("impressions_delta_pct","%"),
        ("clicks_delta","clicks"), ("clicks_delta_pct","%"),
        ("ad_orders_delta","orders"), ("ad_orders_delta_pct","%")):
        add("sku", metric, sku[metric], unit, f"{current} vs {previous}",
            f"Готовое изменение по данным сервиса {metric}, SKU {p['sku']}",
            source="marketing.wb_marketing_campaign_stats_daily", target_type="sku", target_id=p["sku"], vendor=p["sku"], selectable=False)

    campaigns = data["campaigns"]
    for row_index, row in enumerate(campaigns["campaign_rows"]):
        window = p["campaign_row_windows"][row_index]
        if window["advert_id"] != row["advert_id"]:
            raise ValueError("Marketing diagnostic campaign row/window pairing mismatch")
        period = window["period"]
        for metric, unit in (("sum_spend","RUB"), ("views","impressions"), ("clicks","clicks"),
            ("orders","orders"), ("ctr","%"), ("cpc","RUB/click"), ("cr","%")):
            index = add("campaigns", metric, row[metric], unit, period,
                f"Кампания {row['advert_id']} — keyword-кандидат, historical SKU-связь не доказана; native {metric}",
                source="marketing.wb_marketing_campaign_stats_daily", target_type="campaign", target_id=row["advert_id"], source_date=row["report_date"])
            if row_index == 0 and metric in {"sum_spend","orders","cpc"} and index is not None:
                required.append(index)
    for row_index, row in enumerate(data["queries"]["top_query_rows"]):
        for metric, unit in (("views","frequency_proxy"), ("clicks","frequency_proxy"), ("orders","orders"),
                             ("spend","RUB"), ("cpc","RUB/click"), ("avg_pos","position")):
            index = add("queries", metric, row[metric], unit, p["source_windows"]["queries"],
                f"Поисковая строка {row_index+1}, SKU {p['sku']}, {metric}; частотный proxy, не paid-query экономика",
                source="marketing.wb_search_query_orders_text_daily", target_type="search_query", target_id=str(row_index+1),
                vendor=p["sku"], source_date=row["report_date"], selectable=metric not in {"spend","cpc"}, confirmed=metric not in {"spend","cpc"})
            if row_index == 0 and index is not None and metric in {"orders","avg_pos"}:
                required.append(index)
    for row_index, row in enumerate(data["visibility"]["visibility_rows"]):
        for metric, unit in (("avg_position_current","position"), ("avg_position_dynamics","native_change"),
            ("visibility_current","%"), ("visibility_dynamics","native_change"),
            ("open_card_current","opens"), ("open_card_dynamics","native_change"),
            ("add_to_cart_current","carts"), ("add_to_cart_dynamics","native_change"),
            ("orders_current","orders"), ("orders_dynamics","native_change"),
            ("cart_to_order_current","%"), ("cart_to_order_dynamics","native_change")):
            index = add("visibility", metric, row[metric], unit, row["snapshot_date"],
                f"SKU {p['sku']}, отдельный search snapshot/window-end, native {metric}; не рекламная воронка за requested period",
                source="marketing.wb_search_query_details", target_type="sku", target_id=p["sku"], vendor=p["sku"], source_date=row["snapshot_date"])
            if row_index == 0 and metric in {"avg_position_current","visibility_current","open_card_current","add_to_cart_current","cart_to_order_current"} and index is not None:
                required.append(index)
    overview = data["visibility"]["overview"]
    if overview:
        for metric, unit in (("total_products","products"), ("supplier_rating_current","rating"),
            ("position_average_current","position"), ("position_average_dynamics","native_change"),
            ("visibility_current","%"), ("visibility_dynamics","native_change"),
            ("open_card_current","opens"), ("open_card_dynamics","native_change")):
            index = add("visibility", "portfolio_"+metric, overview[metric], unit, overview["snapshot_date"],
                f"Весь портфель, не SKU {p['sku']}; native {metric}", source="marketing.wb_search_query_report_overview",
                target_type="portfolio", target_id="portfolio", source_date=overview["snapshot_date"])
            if index is not None and metric in {"position_average_current","visibility_current"}:
                required.append(index)
    p["tables"] = [
        _table(f"Exact SKU: {current} vs {previous}; native attribution, не рекламная выручка", data["sku"]["rows"]),
        _table("Keyword campaign candidates: native rows, независимые source dates", campaigns["campaign_rows"]),
        _table("Keyword campaign totals: собственное окно, не exact-SKU total", [campaigns["totals"]]),
        _table("Campaign concentration: дата источника неизвестна, не write decision", [campaigns["budget_concentration"]]),
        _table("Поисковые строки: частотный proxy, не paid query spend/CPC", data["queries"]["top_query_rows"]),
        _table("SKU search visibility: отдельный snapshot/window-end", data["visibility"]["visibility_rows"]),
        _table("Portfolio visibility overview: не показатели SKU", [overview] if overview else []),
    ]
    p["required_claim_indices"] = required
    p["required_groups"] = [required[i:i+6] for i in range(0,len(required),6)]
    p["delegation_claim_index"] = next(i for i,c in enumerate(p["claims"]) if c["kind"] == "sku" and c["period"] == current)
    assert_bounded_safe(p, max_bytes=262144)
    return p


def assemble_diagnostic_result(narrative, prepared):
    p = build_diagnostic_catalog(prepared)
    raw = narrative.model_dump(mode="json") if hasattr(narrative,"model_dump") else narrative
    def selected(indices):
        if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(p["claims"]) for i in indices):
            raise ValueError("Marketing diagnostic invalid selection")
        return [p["claims"][i] for i in indices]
    findings = []
    for group in raw["findings"]:
        claims = selected(group["claim_indices"])
        findings.append(DiagnosticFinding(factor="Проверенные наблюдения рекламы", observation="\n".join(c["text"] for c in claims),
            impact=IMPACT, cause=CAUSE, action=ACTION, evidence=list(dict.fromkeys(r for c in claims for r in c["evidence_refs"])),
            confidence="medium", driver_class=DriverClass.NOT_A_DRIVER))
    # Mandatory native scope coverage is policy-owned, just as the existing
    # portfolio triage appends required SKU observations. The LLM selects
    # emphasis/additional findings, never whether known evidence is hidden.
    observed = {line for finding in findings for line in finding.observation.split("\n")}
    for group in p["required_groups"]:
        missing = [i for i in group if p["claims"][i]["text"] not in observed]
        if missing:
            claims = selected(missing)
            findings.append(DiagnosticFinding(factor="Проверенные наблюдения рекламы",
                observation="\n".join(c["text"] for c in claims), impact=IMPACT, cause=CAUSE, action=ACTION,
                evidence=list(dict.fromkeys(r for c in claims for r in c["evidence_refs"])),
                confidence="medium", driver_class=DriverClass.NOT_A_DRIVER))
            observed.update(c["text"] for c in claims)
    primary = p["claims"][p["delegation_claim_index"]]
    checks = [delegated_check(domain,primary) for domain in ("sales","finance")]
    for item in raw["cross_agent_requests"]:
        check = delegated_check(item["domain"], selected([item["claim_index"]])[0])
        if check not in checks:
            checks.append(check)
    return MarketingDiagnosticResult(agent_name="Marketing Agent", summary="\n".join(c["text"] for c in selected(raw["summary_claim_indices"])),
        facts=p["facts"], tables=p["tables"], findings=findings, recommendations=[ACTION], data_gaps=native_gaps(p), cross_agent_requests=checks)


def validate_diagnostic_publication(candidate, prepared):
    p = build_diagnostic_catalog(prepared)
    if any(prepared.get(key) != p[key] for key in ("capability","diagnostic_policy_version","claims","facts","tables",
        "required_claim_indices","required_groups","delegation_claim_index")):
        raise ValueError("Marketing diagnostic catalog altered")
    result = MarketingDiagnosticResult.model_validate(candidate)
    raw = result.model_dump(mode="json")
    assert_bounded_safe(raw, max_bytes=524288)
    if (result.agent_name != "Marketing Agent" or result.marketing_report_snapshot is not None
        or raw["facts"] != p["facts"] or raw["tables"] != p["tables"] or result.recommendations != [ACTION]
        or result.data_gaps != native_gaps(p) or not result.findings or len(result.findings) > 12):
        raise ValueError("Marketing diagnostic changed native publication")
    proof = {c["text"]:c for c in p["claims"]}
    summary = result.summary.split("\n")
    if not result.summary or any(line not in proof for line in summary):
        raise ValueError("Marketing diagnostic summary unproved")
    observed = set()
    for f in result.findings:
        lines = f.observation.split("\n")
        if (any(line not in proof for line in lines) or f.factor != "Проверенные наблюдения рекламы"
            or f.impact != IMPACT or f.cause != CAUSE or f.action != ACTION or f.confidence != "medium"
            or f.driver_class != DriverClass.NOT_A_DRIVER
            or set(f.evidence) != {r for line in lines for r in proof[line]["evidence_refs"]}):
            raise ValueError("Marketing diagnostic finding unproved")
        observed.update(lines)
    if not {p["claims"][i]["text"] for i in p["required_claim_indices"]} <= observed:
        raise ValueError("Marketing diagnostic omitted required scopes/metrics")
    if not set(summary) <= observed:
        raise ValueError("Marketing diagnostic summary not observed")
    allowed = [delegated_check(d,c) for d in ("sales","finance","price","supply") for c in p["claims"]
               if c["kind"] == "sku" and c["text"] in observed and c["period"] == p["requested_period"]]
    required = [delegated_check(d,p["claims"][p["delegation_claim_index"]]) for d in ("sales","finance")]
    if (any(c not in allowed for c in result.cross_agent_requests) or any(c not in result.cross_agent_requests for c in required)
        or len(result.cross_agent_requests) > 4 or len(result.cross_agent_requests) != len(set(c.target_agent for c in result.cross_agent_requests))):
        raise ValueError("Marketing diagnostic delegation scope mismatch")
    return result


def render_diagnostic_report(result, prepared, *, format_version=2):
    # Version1 is retained only for exact replay of historical evidence;
    # current publication uses continuous GFM table blocks.
    if type(format_version) is not int or format_version not in {1, 2}:
        raise ValueError("unsupported Marketing diagnostic report format")
    result = validate_diagnostic_publication(result, prepared)
    parts = [f"# Marketing — диагностика SKU {prepared['sku']}", f"Период: {prepared['requested_period']}.",
        "Реализованные продажи не рекламная атрибуция; ДРР не ROI/прибыль. Кампании — keyword-кандидаты; search snapshot отдельно от периода.",
        "## Главное", result.summary, "## Проверенные наблюдения"]
    for finding in result.findings:
        parts.extend([finding.observation, finding.cause])
    parts.append("## Проверки соседних агентов (ещё не выполнены)")
    parts.extend(f"- {c.target_agent.value}: {c.question}" for c in result.cross_agent_requests)
    parts.append("## Ограничения")
    parts.extend(f"- {gap.reason}" for gap in result.data_gaps)
    def cell(value):
        return ("нет данных" if value is None else str(value)).replace("|","\\|").replace("\n"," ")
    for table in result.tables:
        lines = ["| " + " | ".join(map(cell,table.columns)) + " |",
            "| " + " | ".join("---" for _ in table.columns) + " |"]
        lines.extend("| " + " | ".join(map(cell,row)) + " |" for row in table.rows)
        parts.append(f"### {table.title}")
        parts.extend(lines if format_version == 1 else ["\n".join(lines)])
    return "\n\n".join(parts)
