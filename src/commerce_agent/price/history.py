"""Exact-window Price projection, reconstructed from preserved native evidence."""
from copy import deepcopy
from datetime import date
from hashlib import sha256
import json

from commerce_agent.adapters.contracts import DatabaseTime, ToolResult
from commerce_agent.adapters.price_history import PriceHistory
from commerce_agent.price.evidence import _hydrate
from commerce_agent.price.policy import PriceRequest, assert_bounded_safe
from commerce_agent.sales.contracts import EvidenceFact, ReportTable


SOURCES = {
    "seller": "pricing.wb_supplier_prices",
    "customer": "analytics_ai.mart_price_context_daily",
    "competitors": "analytics_ai.wb_price_ai_competitors_raw",
}
TITLES = {"seller": "История цен продавца", "customer": "История цен покупателя и рыночной сводки",
    "competitors": "Исторические кандидаты конкурентов"}
FACTS = {"seller": ("price", "discounted_price", "club_discounted_price"),
    "customer": ("source_price_site_display", "source_price_site_wallet", "source_price_site_effective"),
    "competitors": ()}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


async def collect_history(client, context, request):
    clock = await _hydrate(client, context, await client.database_time(context), DatabaseTime)
    result = await _hydrate(client, context, await client.price_history(context,
        vendor_code=request.targets[0], date_from=str(request.date_from), date_to=str(request.date_to)), PriceHistory)
    native = {"request": request.model_dump(mode="json"), "today": str(clock.data.db_current_date),
        "history_version": 1, "history_envelope": result.model_dump(mode="json")}
    return {**native, **build_catalog(native)}


def build_catalog(prepared):
    request = PriceRequest.model_validate(prepared["request"])
    if request.operation != "history" or type(prepared.get("history_version")) is not int or prepared["history_version"] != 1:
        raise ValueError("Price history capability mismatch")
    result = ToolResult[PriceHistory].model_validate(prepared["history_envelope"])
    data = result.data
    today = date.fromisoformat(prepared["today"])
    if (data.vendor_code != request.targets[0] or data.requested_period.date_from != request.date_from
            or data.requested_period.date_to != request.date_to or request.date_to > today
            or result.source != SOURCES["seller"] or result.period is not None
            or result.freshness.as_of is not None):
        raise ValueError("Price history source scope mismatch")
    present_sources = {source for name, source in SOURCES.items() if getattr(data, name)}
    if {item.source for item in result.evidence} != present_sources or any(
            item.period is not None or item.grain != "product_filter" for item in result.evidence):
        raise ValueError("Price history provenance mismatch")
    if any(row.source_snapshot_date and row.source_snapshot_date > row.run_date for row in data.competitors):
        raise ValueError("Price competitor source date is future")
    required_limits = {"missing_days_are_unknown_not_zero", "no_latest_snapshot_or_forward_fill",
        "seller_customer_wallet_prices_are_distinct_layers", "competitor_candidates_are_not_verified_strict_analogs",
        "competitor_run_date_is_not_source_price_snapshot_date", "daily_market_summary_is_not_individual_competitor_history",
        "price_observations_do_not_establish_causality", "identity_mapping_is_current_not_historical_identity_proof"}
    if not required_limits <= set(data.limitations) or not set(data.limitations) <= set(result.warnings):
        raise ValueError("Price history limitations missing")
    period = f"{request.date_from}/{request.date_to}"
    evidence, facts, tables, claims, coverage = [], [], [], [], {}
    warnings = list(data.limitations)
    if data.identity_status != "resolved":
        warnings.append("needs_data:price_identity_" + data.identity_status)
    for name, source in SOURCES.items():
        rows = [row.model_dump(mode="json") for row in getattr(data, name)]
        ref = "price-history-v1:" + name + ":" + sha256(canonical(rows).encode()).hexdigest()
        if rows:
            evidence.append({"evidence_ref": ref, "domain": "price", "source": source,
                "claim": TITLES[name] + "; наблюдения, не доказанная причина", "period": period, "freshness": "unknown"})
        day_key = "run_date" if name == "competitors" else "snapshot_date"
        columns = list(rows[0]) if rows else []
        tables.append(ReportTable(title=TITLES[name], columns=columns,
            rows=[[row[column] for column in columns] for row in rows]).model_dump(mode="json"))
        coverage[name] = {}
        for side, window in (("previous", data.previous_period), ("current", data.requested_period)):
            selected = [row for row in rows if str(window.date_from) <= row[day_key] <= str(window.date_to)]
            days = sorted({row[day_key] for row in selected})
            expected = (window.date_to - window.date_from).days + 1
            status = "window_absent" if not days else "available" if len(days) == expected else "partial_coverage"
            coverage[name][side] = {"status": status, "observed_days": days, "row_count": len(selected)}
            metrics = FACTS[name] if name != "competitors" else ("competitor_price_regular", "competitor_price_promo")
            price_days = sorted({row[day_key] for row in selected if any(row[metric] is not None for metric in metrics)})
            coverage[name][side]["price_observed_days"] = price_days
            if len(price_days) < len(days):
                warnings.append(f"needs_data:{name}:{side}:missing_price_values")
            if status != "available":
                warnings.append(f"needs_data:{name}:{side}:{status}")
            text = (f"{TITLES[name]}: {window.date_from}/{window.date_to}; "
                f"покрытие={status}; дней={len(days)}/{expected}; строк={len(selected)}. "
                "Пропуски не означают нулевую цену или отсутствие конкурентов; причинность не установлена.")
            if name == "competitors":
                text += " Дата запуска и дата исходной цены независимы; кандидаты не являются проверенными аналогами."
            claims.append({"operation": name, "text": text, "sku": data.vendor_code, "nm_id": data.nm_id,
                "period": f"{window.date_from}/{window.date_to}", "check_period": None,
                "evidence_refs": [ref] if rows else []})
            for row in selected:
                for metric in FACTS[name]:
                    facts.append(EvidenceFact(metric=f"price_history.{name}.{metric}", value=row[metric],
                        unit="native_price", sku=data.vendor_code, target_type="sku", target_id=data.vendor_code,
                        period=f"{window.date_from}/{window.date_to}",
                        source_table=source, source_snapshot_at=row[day_key], evidence_ref=ref,
                        source_agent="Price Agent", metric_definition=f"{metric}; exact source day; no causal attribution",
                        status="confirmed" if row[metric] is not None else "unconfirmed",
                        confidence="low").model_dump(mode="json"))
    comparison_indices = []
    descriptions = {
        "higher": "Все наблюдавшиеся цены текущего периода выше всех цен предыдущего",
        "lower": "Все наблюдавшиеся цены текущего периода ниже всех цен предыдущего",
        "unchanged": "Одна и та же цена наблюдалась во все дни обоих периодов",
        "mixed": "Ряды пересекаются или неоднородны; однозначное направление не установлено",
        "insufficient_data": "Недостаточно полных сопоставимых ежедневных цен",
    }
    for comparison in data.comparisons:
        source = SOURCES[comparison.layer]
        ref = next((item["evidence_ref"] for item in evidence if item["source"] == source), None)
        comparison_indices.append(len(claims))
        claims.append({"operation": comparison.layer, "sku": data.vendor_code, "nm_id": data.nm_id,
            "period": period, "check_period": None, "evidence_refs": [ref] if ref else [],
            "text": f"{TITLES[comparison.layer]}: {comparison.metric}; competitor_nm_id={comparison.competitor_nm_id}; "
                f"{descriptions[comparison.direction]}. Предыдущие наблюдавшиеся значения: {comparison.previous_values}; "
                f"текущие: {comparison.current_values}. Это описание цен, не причинный эффект и не средняя цена."})
    required = list(range(6)) + [index for index in comparison_indices if claims[index]["operation"] != "competitors"]
    output = {"facts": facts, "tables": tables, "claims": claims, "evidence": evidence,
        "warnings": list(dict.fromkeys(warnings)), "coverage": coverage,
        "required_claim_indices": required, "summary_claim_indices": required,
        "required_operations": list(SOURCES), "analytical_groups": {
            "layers": comparison_indices, "quality": [], "scenarios": [], "raw_market": [4, 5], "daily": [0, 1, 2, 3]},
        "analysis_policy_version": 2}
    assert_bounded_safe({**prepared, **output}, max_bytes=524288)
    return output


def verify_history(prepared):
    expected = build_catalog(prepared)
    if any(canonical(prepared.get(key)) != canonical(value) for key, value in expected.items()):
        raise ValueError("Price history catalog differs from source")
    return expected
