"""Native Price catalog: no calculations, inferred source dates or free prose.

Rebuilt at publication, so a candidate cannot modify facts/tables or turn raw
competitors and dated scenarios into trusted current price instructions.
"""
from copy import deepcopy
from datetime import date
from hashlib import sha256
import json

from commerce_agent.adapters.contracts import ToolResult
from commerce_agent.price.evidence import MODELS, validate_envelope, native_read_warnings, MAX_PREPARED_BYTES
from commerce_agent.price.policy import PriceRequest, assert_bounded_safe
from commerce_agent.sales.contracts import EvidenceFact, ReportTable

TITLES = {
    "context": "Канонические слои цен — независимые даты и качество",
    "decision": "Готовые сценарии переоценки — не выполненные действия",
    "competitor_overview": "Широкий рынок — не подтверждённые строгие аналоги",
    "competitor_details": "Исходные кандидаты конкурентов — ограниченная выборка",
    "daily": "Цена и продажи по дням — связь, не доказанная причинность",
}
CONTEXT_KEYS = (
    "seller_controlled_price", "customer_price_display", "customer_price_wallet",
    "customer_price_effective", "customer_price_basis", "seller_club_discounted_price",
    "strict_competitor_count", "strict_competitor_price_median", "broad_market_price_median",
    "customer_vs_strict_median_gap_percent", "competitor_quality", "freshness_status", "price_status",
)
SCENARIO_KEYS = (
    "source_price_promo", "recommended_price", "recommended_price_delta_pct",
    "expected_orders_7d", "expected_net_profit_7d", "expected_margin_pct",
    "cost_price_rub", "decision_status", "decision_confidence", "decision_reason_code",
    "decision_freshness", "decision_lag_days",
)
DAILY_KEYS = (
    "seller_effective_price_rub", "observed_transaction_seller_price_avg_rub",
    "observed_buyer_price_avg_rub", "price_basis", "non_cancelled_order_units",
    "realized_sales_units", "realized_for_pay_rub", "business_day_complete",
    "sales_day_complete", "marketing_day_complete",
)
LABELS = {
    "seller_controlled_price": "Цена, управляемая продавцом, ₽",
    "customer_price_display": "Цена на сайте, ₽", "customer_price_wallet": "Цена кошелька, ₽",
    "customer_price_effective": "Нативная эффективная цена покупателя, ₽",
    "customer_price_basis": "Основание эффективной цены", "seller_club_discounted_price": "Клубная цена, ₽",
    "strict_competitor_count": "Строгих аналогов", "strict_competitor_price_median": "Медиана строгих аналогов, ₽",
    "broad_market_price_median": "Медиана широкого рынка, ₽",
    "customer_vs_strict_median_gap_percent": "Нативное отклонение от строгой медианы, %",
    "competitor_quality": "Качество аналогов", "freshness_status": "Нативный статус свежести",
    "price_status": "Нативный ценовой статус", "source_price_promo": "Исходная promo цена сценария, ₽",
    "recommended_price": "Сценарная цена, ₽", "recommended_price_delta_pct": "Сценарное изменение цены, %",
    "expected_orders_7d": "Ожидаемые заказы 7d, не факт", "expected_net_profit_7d": "Ожидаемая прибыль 7d, не факт, ₽",
    "expected_margin_pct": "Ожидаемая маржа, не факт, %", "cost_price_rub": "COGS в сценарии, ₽",
    "decision_status": "Статус сценария", "decision_confidence": "Уверенность сценария",
    "decision_reason_code": "Код основания сценария", "decision_freshness": "Относительная свежесть сценария",
    "decision_lag_days": "Относительный лаг, не wall-clock", "seller_effective_price_rub": "Цена продавца по дню, ₽",
    "observed_transaction_seller_price_avg_rub": "Наблюдаемая средняя цена транзакций продавца, ₽",
    "observed_buyer_price_avg_rub": "Наблюдаемая средняя цена покупателя, ₽", "price_basis": "Основание дневной цены",
    "non_cancelled_order_units": "Заказы без отмен, шт", "realized_sales_units": "Реализованные продажи, шт",
    "realized_for_pay_rub": "К оплате, ₽", "business_day_complete": "Полнота бизнес-дня",
    "sales_day_complete": "Полнота дня продаж", "marketing_day_complete": "Полнота дня рекламы",
}
DATE_KEYS = ("seller_snapshot_date", "analysis_run_date", "source_snapshot_date",
             "context_snapshot_date", "context_as_of_date")


def shown(value):
    if value is None:
        return "нет данных"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if type(value) is bool:
        return "true" if value else "false"
    return str(value)


def _table_cell(value):
    # Legacy ReportTable has no boolean/nested cell type. Preserve them visibly
    # instead of bool->1/0 coercion or silently dropping nested source refs.
    return shown(value) if isinstance(value, (bool, list, dict)) else value


def _unit(key, value):
    if type(value) not in {int, float}:
        return None
    if "percent" in key or "pct" in key or "conversion" in key or key == "buyout_percent":
        return "%"
    if "price" in key and not any(word in key for word in ("ratio", "outlier", "status", "age", "id")) or "rub" in key or "profit" in key or "revenue" in key:
        return "RUB"
    if "days" in key:
        return "дни"
    if "ratio" in key or "confidence" in key or "elasticity" in key or "coef" in key or "share" in key or "rating" in key:
        return "коэффициент"
    if "count" in key or "units" in key or "qty" in key or "orders" in key or "buyouts" in key or "rows" in key or "feedback" in key:
        return "шт"
    return "ед. (native)"


def _source_date(operation, key, row):
    if operation == "context":
        if key.startswith("seller_"):
            return row.get("seller_snapshot_date")
        if key.startswith("customer_") or key.startswith("wallet_") or key.startswith("platform_") or key.startswith("report_to_customer_"):
            # Coalesced display/wallet sources have no exact layer date.
            return None
        # Strict metrics come directly from the analysis row. Broad median is
        # COALESCE across analysis/context, and quality/freshness/price status
        # are derived/composite. The view does not expose an exact source date
        # for those outputs; do not borrow source_snapshot_date for them.
        return row.get("analysis_run_date") if key.startswith("strict_competitor_") else None
    return row.get("report_date") if operation == "daily" else row.get("snapshot_date") if operation == "decision" else row.get("run_date")


def build_price_catalog(collection):
    """Project validated native reads; preserve omitted fields vs native nulls."""
    assert_bounded_safe(collection, max_bytes=MAX_PREPARED_BYTES)
    if type(collection.get("version")) is not int or collection["version"] != 1:
        raise ValueError("unsupported Price collection version")
    request = PriceRequest.model_validate(collection["request"])
    today = date.fromisoformat(collection["today"])
    _bind_request_reads(request, collection["reads"])
    claims, facts, tables, groups = [], [], [], {name: [] for name in ("layers", "quality", "scenarios", "raw_market", "daily")}
    required, expected_evidence = [], []
    for read_index, read in enumerate(collection["reads"]):
        operation = read["operation"]
        envelope = ToolResult[MODELS[operation]].model_validate(read["envelope"])
        validate_envelope(operation, envelope, read["arguments"], today, identity_namespace=read["identity_namespace"])
        if len(read["evidence_refs"]) != len(envelope.evidence):
            raise ValueError("Price missing native read references")
        digest = sha256(json.dumps(read["envelope"], ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
        expected_warnings = native_read_warnings(operation, envelope, today)
        if operation == "context" and read["scope"] is not None and not envelope.data.rows:
            expected_warnings.append(f"needs_data:price_target_missing:{read['scope']}")
        if read["warnings"] != expected_warnings:
            raise ValueError("Price source limitations differ from native read")
        for item, ref in zip(envelope.evidence, read["evidence_refs"]):
            if not ref.startswith("price:") or not ref.endswith(f":{operation}:{item.id}:{digest}"):
                raise ValueError("Price native evidence digest mismatch")
            expected_evidence.append({"domain": "price", "source": item.source, "claim": operation, "evidence_ref": ref,
                "period": f"{item.period.date_from}/{item.period.date_to}" if item.period else None, "freshness": "unknown"})
        source_refs = dict(zip((item.source for item in envelope.evidence), read["evidence_refs"]))
        source = envelope.source
        ref = source_refs.get(source)
        rows = read["envelope"]["data"]["rows"]
        columns = list(dict.fromkeys(key for row in rows for key in row))
        table = ReportTable(title=f"{TITLES[operation]} [filter={read['scope'] or 'portfolio'}]",
            columns=columns, rows=[[_table_cell(row[key]) if key in row else "не передано" for key in columns] for row in rows])
        tables.append(table.model_dump(mode="json"))
        local_claims = []
        if not rows and operation != "context":
            # A valid native empty scenario/raw/daily response is a gap, not
            # a reason to discard already validated canonical price evidence.
            # No synthetic source evidence, financial zero or date is invented.
            index = len(claims)
            claims.append({"operation": operation, "text": f"{TITLES[operation]} [filter={read['scope'] or 'portfolio'}]: "
                "источник вернул 0 строк; данные недоступны, результат не подтверждён.",
                "sku": None, "nm_id": None, "period": "unknown", "check_period": None,
                "evidence_refs": [], "read_index": read_index, "row_index": None})
            local_claims.append(index)
        for row_index, row in enumerate(rows):
            if not ref:
                raise ValueError("Price native row lacks primary provenance")
            sku = row.get("vendor_code")
            nm_id = row.get("nm_id", row.get("source_nm_id"))
            period = (f"{envelope.period.date_from}/{envelope.period.date_to}" if envelope.period else "latest; metric window unconfirmed")
            # The full native row is already preserved in the table and read.
            # Facts expose the analytical price/scenario/day metrics, not a
            # second several-megabyte expansion of URLs, refs and raw titles.
            keys = CONTEXT_KEYS if operation == "context" else SCENARIO_KEYS if operation == "decision" else DAILY_KEYS if operation == "daily" else (
                "competitor_count", "competitor_price_median", "price_status", "source_vs_median_gap_pct") if operation == "competitor_overview" else ()
            for key, value in row.items():
                if key not in keys or isinstance(value, (list, dict)):
                    continue
                snapshot = _source_date(operation, key, row)
                freshness = "future" if snapshot and date.fromisoformat(snapshot) > today else "unknown"
                facts.append(EvidenceFact(metric=key, value=value, unit=_unit(key, value), period=period,
                    period_start=str(envelope.period.date_from) if envelope.period else None,
                    period_end=str(envelope.period.date_to) if envelope.period else None,
                    sku=sku, target_type="sku", target_id=str(nm_id), source_table=source,
                    source_snapshot_at=snapshot, retrieved_at=read["envelope"]["freshness"]["observed_at"],
                    metric_definition=f"Native Price {operation}; row={row_index}; scope={read['scope'] or 'portfolio'}; "
                        "not causal, not execution authority; generated provenance is not source-completeness proof",
                    freshness_status=freshness, evidence_ref=ref, source_agent="Price Agent",
                    status="unconfirmed" if operation in {"decision", "competitor_overview", "competitor_details"} else "confirmed",
                    confidence="low").model_dump(mode="json"))
            if operation == "competitor_details":
                continue  # Full raw matrix retained; row-level candidates are not trusted claims.
            text = f"{TITLES[operation]} [filter={read['scope'] or 'portfolio'}]: SKU {sku or 'не указан'}, nm_id={nm_id}; "
            text += "; ".join(f"{LABELS.get(key, key)} — {shown(row.get(key))}" for key in keys)
            if operation == "context":
                text += "; " + "; ".join(f"{key}={shown(row.get(key))}" for key in DATE_KEYS)
                text += "; точная дата слоя покупателя не подтверждена; native статус не удостоверяет свежесть сегодня"
                check_date = row.get("seller_snapshot_date")
            else:
                check_date = row.get("report_date") if operation == "daily" else row.get("snapshot_date") if operation == "decision" else row.get("run_date")
                text += f"; дата источника={shown(check_date)}"
                if operation == "decision":
                    text += f"; price_ai_run_date={shown(row.get('price_ai_run_date'))}; price_context_as_of_date={shown(row.get('price_context_as_of_date'))}; сценарий, не фактический эффект"
                if operation == "daily":
                    text += "; причинность не установлена; средняя транзакций не равна точному дневному snapshot"
                if operation == "competitor_overview":
                    text += "; raw кандидаты не подтверждают strict positioning"
            index = len(claims)
            claims.append({"operation": operation, "text": text + ".", "sku": sku, "nm_id": nm_id,
                "period": period, "check_period": (period if operation == "daily" else f"{check_date}/{check_date}")
                    if check_date and date.fromisoformat(check_date) <= today else None,
                "evidence_refs": [ref], "read_index": read_index, "row_index": row_index})
            local_claims.append(index)
        if operation == "context":
            groups["layers"].extend(local_claims[:2])
            uncertain = [index for index in local_claims if rows[claims[index]["row_index"]].get("competitor_quality") != "trusted"
                or rows[claims[index]["row_index"]].get("freshness_status") != "fresh"]
            groups["quality"].extend(uncertain[:2])
            # Only native gaps; ranking is bounded to observed rows and not a
            # causal claim or price recommendation. Freshness remains explicit.
            gapped = [index for index in local_claims if type(rows[claims[index]["row_index"]].get("customer_vs_strict_median_gap_percent")) in {int, float}]
            if gapped:
                groups["layers"].extend([min(gapped, key=lambda i: rows[claims[i]["row_index"]]["customer_vs_strict_median_gap_percent"]),
                    max(gapped, key=lambda i: rows[claims[i]["row_index"]]["customer_vs_strict_median_gap_percent"])])
        elif operation == "decision":
            groups["scenarios"].extend(local_claims[:2])
        elif operation == "daily":
            groups["daily"].extend(local_claims[:1] + local_claims[-1:])
        elif operation == "competitor_overview":
            groups["raw_market"].extend(local_claims[:1])
        if rows and not local_claims:  # Raw details: a native count, never a market conclusion.
            index = len(claims)
            claims.append({"operation": operation, "text": f"{TITLES[operation]}: native row_count={envelope.data.row_count}; "
                f"run_date={shown(envelope.data.run_date)}; выборка ограничена limit={read['arguments']['limit']}; не strict proof.",
                "sku": None, "nm_id": None, "period": "latest; metric window unconfirmed", "check_period": None,
                "evidence_refs": [ref], "read_index": read_index, "row_index": None})
            groups["raw_market"].append(index)
        if not rows and operation == "competitor_details":
            groups["raw_market"].extend(local_claims)
        if local_claims:
            required.extend(local_claims[:1])  # Every requested read/scope remains visible.
    groups = {key: list(dict.fromkeys(indices)) for key, indices in groups.items()}
    required = list(dict.fromkeys(required + [index for indices in groups.values() for index in indices]))
    needed = {"context"} | ({"decision", "competitor_overview", "competitor_details"} if request.operation == "report" and request.scope == "portfolio" else
        {"competitor_details"} if request.operation in {"report", "competitors"} else {"daily"} if request.operation == "daily" else set())
    observed = {claim["operation"] for claim in claims}
    if not needed <= observed:
        raise ValueError("Price requested capability lacks native observations")
    if not any(read["operation"] == "context" and read["envelope"]["data"]["rows"] for read in collection["reads"]):
        raise ValueError("Price canonical context unavailable")
    if request.targets:
        for target in request.targets:
            if not any(read["scope"] == target and read["operation"] == "context" and read["envelope"]["data"]["rows"] for read in collection["reads"]):
                raise ValueError("Price target has no canonical data")
    if collection["evidence"] != expected_evidence:
        raise ValueError("Price evidence projection differs from native reads")
    native_warnings = list(dict.fromkeys(warning for read in collection["reads"] for warning in read["warnings"]))
    # The first collection contains only native warnings; analyzed prepared
    # state also contains the derived limitation list verified below.
    if "analysis_policy_version" not in collection and collection["warnings"] != native_warnings:
        raise ValueError("Price collection omitted native limitations")
    warnings = list(dict.fromkeys(native_warnings + ["scope:latest_sources_are_not_historical_metric_windows",
        "quality:price_catalog_observations_do_not_authorize_external_actions"]))
    if any(row.get("source_snapshot_date") is None for read in collection["reads"] if read["operation"] == "context" for row in read["envelope"]["data"]["rows"]):
        warnings.append("needs_data:canonical_source_snapshot_date_missing")
    result = {"claims": claims, "facts": facts, "tables": tables, "analytical_groups": groups,
        "required_claim_indices": required, "summary_claim_indices": required,
        "required_operations": sorted(needed), "warnings": warnings}
    assert_bounded_safe(result, max_bytes=MAX_PREPARED_BYTES)
    return result


def _bind_request_reads(request, reads):
    """Checkpointed native evidence must still belong to this exact request."""
    expected = [("context", None)] if request.scope == "portfolio" else []
    if request.scope == "portfolio":
        if request.operation == "report":
            expected.append(("decision", None))
        if request.operation in {"report", "competitors"}:
            expected.extend([("competitor_overview", None), ("competitor_details", None)])
    for target in request.targets:
        expected.append(("context", target))
        if request.operation == "report": expected.append(("decision", target))
        if request.operation in {"report", "competitors"}: expected.append(("competitor_details", target))
        if request.operation == "daily": expected.append(("daily", target))
    if [(read["operation"], read["scope"]) for read in reads] != expected:
        raise ValueError("Price native reads differ from requested capability/scope")
    resolved = {}
    for read in reads:
        operation, scope, args = read["operation"], read["scope"], read["arguments"]
        if scope is None:
            expected_args = {"limit": 100, "include_evidence": True} if operation == "context" else {"limit": 100}
            if operation in {"decision", "competitor_overview"}: expected_args["product_keyword"] = None
            if args != expected_args or read["identity_namespace"] is not None:
                raise ValueError("Price portfolio read has a foreign filter or limit")
        elif operation == "context":
            kinds = [key for key in ("nm_id", "vendor_code") if key in args]
            if len(kinds) != 1:
                raise ValueError("Price exact read needs one identity namespace")
            kind = kinds[0]
            if request.target_namespace not in {"auto", kind}:
                raise ValueError("Price read namespace differs from requested namespace")
            token = str(int(scope)) if kind == "nm_id" and scope.isascii() and scope.isdigit() else scope
            if args != {kind: token, "limit": 20, "include_evidence": True} or read["identity_namespace"] is not None:
                raise ValueError("Price exact read differs from requested target")
            resolved[scope] = (kind, token)
        else:
            kind, token = resolved[scope]
            if operation == "decision":
                expected_args = {"product_keyword": token, "limit": 100}
            elif operation == "daily":
                expected_args = {"product_keyword": token, "days": request.days}
            else:
                expected_args = {kind: token, "limit": 100}
            if args != expected_args or read["identity_namespace"] != (kind if operation in {"decision", "daily"} else None):
                raise ValueError("Price downstream read differs from resolved requested identity")


def prepare_price_analysis(collection):
    result = deepcopy(collection)
    result.update(build_price_catalog(collection))
    result["analysis_policy_version"] = 1
    result["capability"] = "native evidence-bound Price analysis; no tools or writes in analyze"
    assert_bounded_safe(result, max_bytes=MAX_PREPARED_BYTES)
    return result


def verify_price_catalog(prepared):
    if prepared.get("history_version") is not None:
        from commerce_agent.price.history import verify_history
        return verify_history(prepared)
    if type(prepared.get("analysis_policy_version")) is not int or prepared["analysis_policy_version"] != 1:
        raise ValueError("unsupported Price analysis policy version")
    # Warnings projected into analysis are not allowed to feed back into the
    # native source input, otherwise omissions could self-validate on replay.
    native = deepcopy(prepared)
    native["warnings"] = list(dict.fromkeys(warning for read in native["reads"] for warning in read["warnings"]))
    expected = build_price_catalog(native)
    if any(prepared.get(key) != value for key, value in expected.items()):
        raise ValueError("Price catalog differs from native evidence")
    return expected
