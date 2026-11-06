"""Finance native readers to verified facts/matrices; no financial formulas."""
from datetime import date, timedelta
from hashlib import sha256
import json

from commerce_agent.adapters.contracts import (
    DatabaseTime, FinanceOverview, FinanceMarketplaceReport, FinanceSkuReport,
    FinanceProfitChangeReport, ToolArtifactReference, ToolResult,
)
from commerce_agent.sales.contracts import EvidenceFact, ReportTable
from commerce_agent.finance.policy import assert_bounded_safe

REPORT = "finance.wb_report_detail_daily"
COST = "finance.wb_product_coast"
MAX_PREPARED_BYTES = 4_000_000
LABELS = {"revenue": "Выручка до возвратов", "returns": "Возвраты, ₽", "net_cogs": "Себестоимость нетто, ₽",
    "margin_rub": "Маржа калькулятора, ₽", "margin_percent": "Маржа к нетто-выручке, %",
    "total_marketplace_expenses": "Расходы маркетплейса, ₽", "total_to_pay": "К оплате, ₽",
    "commission_and_acquiring": "Комиссия и эквайринг, ₽", "marketing": "Реклама, ₽", "logistics": "Логистика, ₽",
    "storage_fee": "Хранение, ₽", "cross_docking": "Кросс-докинг, ₽",
    "missing_cost_row_count": "Строк без себестоимости", "missing_cost_sku_count": "SKU без себестоимости",
    "current_total_sku_count": "SKU текущего периода", "previous_total_sku_count": "SKU прошлого периода",
    "returned_sku_count": "Возвращено SKU", "total_sku_count": "SKU в источнике",
    "payout_sum": "Выплаты, ₽", "retail_amount_sum": "Сумма продаж, ₽",
    "payout_margin_before_cogs_pct": "Маржа выплат ДО себестоимости, %", "fee_burden_pct": "Нагрузка сборов, %"}


def comparison_input_warnings(operation, data):
    if operation in {"current_cogs", "previous_cogs"} and data.totals.returned_sku_count < data.totals.total_sku_count:
        return ["scope:comparison_inputs_top500_not_full_portfolio"]
    return []


async def _hydrate(client, context, response, model):
    if isinstance(response, ToolArtifactReference):
        response = await client.hydrate(context, response, result_type=model)
    return ToolResult[model].model_validate(response.model_dump(mode="json"))


def _validate(envelope, args, today):
    data = envelope.data
    if any(item.source not in {REPORT, COST} for item in envelope.evidence):
        raise ValueError("Finance foreign evidence source")
    if isinstance(data, FinanceOverview):
        if data.focus_keyword != args["product_keyword"]:
            raise ValueError("Finance overview filter mismatch")
        for family in (data.unit_economics_watchlist, data.margin_watchlist, data.fee_pressure_rows):
            if len(family) > args["limit"]:
                raise ValueError("Finance overview row limit mismatch")
        dates = [row.snapshot_date for row in data.unit_economics_watchlist] + [row.report_date for row in data.margin_watchlist]
        if any(day is not None and day > today for day in dates):
            raise ValueError("Finance future source date")
        return
    if isinstance(data, (FinanceSkuReport, FinanceProfitChangeReport)):
        if len(data.rows) > args["limit"] or data.totals.returned_sku_count != len(data.rows) or data.product_keyword != args["product_keyword"]:
            raise ValueError("Finance row/filter/limit mismatch")
    if isinstance(data, FinanceProfitChangeReport):
        for key, attr in (("current_date_from", "current_period_start"), ("current_date_to", "current_period_end"),
                          ("previous_date_from", "previous_period_start"), ("previous_date_to", "previous_period_end")):
            if str(getattr(data, attr)) != args[key]:
                raise ValueError("Finance comparison period mismatch")
        period = None
    else:
        if str(data.date_from) != args["date_from"] or str(data.date_to) != args["date_to"]:
            raise ValueError("Finance requested report period mismatch")
        if isinstance(data, FinanceSkuReport) and data.sort_by != args["sort_by"]:
            raise ValueError("Finance SKU sorting mismatch")
        period = {"date_from": args["date_from"], "date_to": args["date_to"]}
    if envelope.freshness.as_of is not None:
        raise ValueError("Finance report source freshness unconfirmed")
    for value in [envelope.period, *(item.period for item in envelope.evidence)]:
        if (value.model_dump(mode="json") if value else None) != period:
            raise ValueError("Finance period provenance mismatch")


def analysis_policy_version(prepared):
    version = prepared.get("analysis_policy_version", 1)
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("Finance unsupported analysis policy version")
    scoped = prepared.get("scoped_totals_policy_version", 0)
    if (type(scoped) is not int or scoped not in {0, 1}
            or scoped == 1 and (version != 2 or not prepared.get("request", {}).get("targets"))):
        raise ValueError("Finance unsupported scoped totals policy version")
    missing_side = prepared.get("missing_side_provenance_policy_version", 0)
    if (type(missing_side) is not int or missing_side not in {0, 1}
        or missing_side == 1 and version != 2):
        raise ValueError("Finance unsupported missing-side provenance policy version")
    return version


async def collect_finance_evidence(client, context, request, *, policy_version=2):
    analysis_policy_version({"analysis_policy_version": policy_version})
    scoped_totals = policy_version == 2 and bool(request.targets)
    clock = await _hydrate(client, context, await client.database_time(context), DatabaseTime)
    today = clock.data.db_current_date
    end = request.date_to or today - timedelta(days=1)
    start = request.date_from or end - timedelta(days=request.days - 1)
    previous_end = request.previous_date_to or start - timedelta(days=1)
    previous_start = request.previous_date_from or previous_end - (end - start)
    if end > today or previous_end >= start:
        raise ValueError("Finance future/overlapping requested period")
    current_period, previous_period = f"{start}/{end}", f"{previous_start}/{previous_end}"
    reads = []

    async def read(operation, method, model, arguments, scope):
        envelope = await _hydrate(client, context, await getattr(client, method)(context, **arguments), model)
        _validate(envelope, arguments, today)
        if scope is not None:
            # Native Finance readers use substring keyword search, not a list
            # operator. Never publish other matched SKUs as an exact target.
            candidates = (envelope.data.rows if hasattr(envelope.data, "rows") else
                          [*envelope.data.unit_economics_watchlist, *envelope.data.margin_watchlist, *envelope.data.fee_pressure_rows])
            if any(scope not in {str(row.nm_id), str(row.vendor_code)} for row in candidates):
                raise ValueError("Finance exact target not resolved by native keyword reader")
        reads.append((operation, envelope, scope))

    targets = [None, *request.targets] if request.scope == "portfolio" else list(request.targets)
    # Portfolio targets are additional deep dives, not a filter on the matrix.
    for target in targets:
        await read("overview", "finance_overview", FinanceOverview, {"product_keyword": target, "limit": 100}, target)
    if request.operation != "overview":
        if request.operation in {"marketplace", "report"} and request.scope == "portfolio":
            await read("marketplace", "finance_marketplace_report", FinanceMarketplaceReport,
                       {"date_from": str(start), "date_to": str(end)}, None)
        if request.operation in {"sku", "margin_change", "report"} or request.targets:
            historical_targets = list(request.targets) if request.operation == "marketplace" else targets
            for target in historical_targets:
                args = {"date_from": str(start), "date_to": str(end), "product_keyword": target,
                        "limit": 100, "sort_by": "net_revenue"}
                await read("sku", "finance_sku_report", FinanceSkuReport, args, target)
                if request.operation in {"margin_change", "report"}:
                    # Native change internally uses 500 per window. Collect the
                    # same input scope for cost completeness, not sum to totals.
                    comparison_args = {**args, "limit": 500, "sort_by": "margin_rub"}
                    await read("current_cogs", "finance_sku_report", FinanceSkuReport, comparison_args, target)
                    await read("previous_cogs", "finance_sku_report", FinanceSkuReport,
                               {**comparison_args, "date_from": str(previous_start), "date_to": str(previous_end)}, target)
                    await read("margin_change", "finance_sku_profit_change", FinanceProfitChangeReport,
                        {"current_date_from": str(start), "current_date_to": str(end),
                         "previous_date_from": str(previous_start), "previous_date_to": str(previous_end),
                         "product_keyword": target, "limit": 100}, target)

    facts, tables, claims, evidence, warnings = [], [], [], [], []
    required, current_candidates, change_candidates, expense_indices = [], [], [], []
    missing_side_policy_used = False
    for operation, envelope, scope in reads:
        data = envelope.data
        warnings.extend(envelope.warnings)
        digest = sha256(json.dumps(data.model_dump(mode="json"), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
        by_source = {}
        for item in envelope.evidence:
            if item.grain != ("product_filter" if scope else "portfolio"):
                raise ValueError("Finance evidence grain mismatch")
            ref = f"finance:{context.request_id}:{operation}:{item.id}:{digest}"
            by_source[item.source] = ref
            evidence.append({"evidence_ref": ref, "domain": "finance", "source": item.source,
                "claim": operation, "period": f"{item.period.date_from}/{item.period.date_to}" if item.period else None,
                "freshness": "unknown"})
        if not by_source.get(REPORT):
            warnings.append(f"needs_data:{operation}_report_provenance_missing")

        def add_claim(values, period, *, sku=None, nm_id=None, note="", selected_keys=None):
            ref = by_source.get(REPORT)
            if not ref:
                return None
            keys = selected_keys or list(values)
            subject = "SKU " + sku if sku else f"Фильтр запроса {scope}" if scoped_totals and scope else "Портфель"
            text = f"Finance {operation} [filter={scope or 'portfolio'}]: {subject}, nm_id={nm_id}, {period}: "
            text += "; ".join(f"{LABELS.get(key, key)} — {'нет данных' if values[key] is None else values[key]}" for key in keys)
            if note:
                text += "; " + note
            index = len(claims)
            claims.append({"operation": operation, "text": text + ".", "sku": sku, "nm_id": nm_id,
                           "period": period, "evidence_refs": list(by_source.values())})
            return index

        def add_facts(values, period, *, sku=None, title=None, source_snapshot_at=None):
            for key, value in values.items():
                if key in {"nm_id", "Артикул", "vendor_code", "brand_name", "subject_name", "snapshot_date", "report_date"}:
                    continue
                cost_metric = key in {"unit_cost_rub", "Себестоимость", "sales_cogs", "return_cogs", "net_cogs", "Себестоимость возвратов"}
                source = COST if cost_metric else REPORT
                ref = by_source.get(source) or by_source.get(REPORT)
                if not ref:
                    continue
                boundaries = period.split("/") if "/" in period else [period, period] if period != "unknown" else [None, None]
                unit = "%" if "pct" in key or "percent" in key or "%" in key or key == "Доля себестоимости в выручке до возвратов" else "шт" if "qty" in key or "quantity" in key or "шт" in key or "count" in key else "RUB" if type(value) in {int, float} else None
                facts.append(EvidenceFact(metric=key, value=value, unit=unit, period=period,
                    period_start=boundaries[0], period_end=boundaries[1], sku=sku,
                    # Totals/counts of an exact keyword read belong to that
                    # filter, not the portfolio. The requested identifier may
                    # be an nm_id, so do not invent a vendor_code for it.
                    target_type="sku" if sku else "product_filter" if scoped_totals and scope else "portfolio",
                    target_id=sku or (scope if scoped_totals else None),
                    source_table=source if source in by_source else REPORT,
                    source_snapshot_at=source_snapshot_at,
                    retrieved_at=envelope.freshness.observed_at, source_agent="finance", evidence_ref=ref,
                    status="confirmed" if value is not None and source in by_source and period != "unknown" else "unconfirmed",
                    metric_definition=f"Native {operation} reader ({title or key}); contribution/payout margin is not net profit; cost version unconfirmed.").model_dump(mode="json"))

        def matrix(title, rows, columns):
            tables.append(ReportTable(title=title, columns=list(columns),
                rows=[[row.get(key) for key in columns] for row in rows]).model_dump(mode="json"))

        if isinstance(data, FinanceOverview):
            warnings.append("provenance:overview_aggregate_window_unconfirmed_snapshot_date_is_not_metric_period")
            for name, family in (("Unit economics — последний срез", data.unit_economics_watchlist),
                                 ("Маржа выплат ДО COGS — последний срез", data.margin_watchlist),
                                 ("Сборы — дата источника не подтверждена", data.fee_pressure_rows)):
                rows = [row.model_dump(mode="json") for row in family]
                columns = list(type(family[0]).model_fields) if family else ["vendor_code", "nm_id"]
                matrix(f"Finance overview: {name}, filter={scope or 'portfolio'}", rows, columns)
                for row in rows:
                    snapshot_date = row.get("snapshot_date") or row.get("report_date")
                    # M2 exposes family-specific source dates, not aggregate
                    # window boundaries. A latest date is not a one-day report.
                    period = "unknown"
                    values = {key: value for key, value in row.items() if key not in {"nm_id", "vendor_code", "brand_name", "subject_name", "snapshot_date", "report_date"}}
                    add_facts(values, period, sku=row["vendor_code"], source_snapshot_at=snapshot_date)
                    index = add_claim(values, period, sku=row["vendor_code"], nm_id=row["nm_id"],
                        note=f"Latest source snapshot={snapshot_date or 'unknown'}; окно агрегата не подтверждено; это не выбранный исторический отчёт; маржа выплат не включает COGS")
                    if request.operation == "overview" and index is not None and row is rows[0]:
                        required.append(index)
        elif isinstance(data, FinanceMarketplaceReport):
            values = data.totals.model_dump(mode="json")
            matrix(f"Маркетплейс: {current_period}", [metric.model_dump(mode="json") for metric in data.metrics], ("label", "value", "percent"))
            add_facts(values, current_period)
            index = add_claim(values, current_period, selected_keys=["revenue", "returns", "total_marketplace_expenses", "margin_rub", "margin_percent", "missing_cost_row_count"],
                note="Маржа калькулятора не равна чистой прибыли; историческая версия себестоимости не подтверждена")
            if index is not None:
                required.append(index)
            if policy_version == 2:
                expense_keys = [key for key in ("commission_and_acquiring", "marketing", "logistics", "storage_fee", "cross_docking") if key in values]
                known = [key for key in expense_keys if type(values[key]) in {int, float}]
                largest = max(known, key=lambda key: values[key]) if known else None
                # Ranking existing amounts is not a new financial formula or
                # an explanation of the change in margin. No ratio is inferred.
                ranking_label = "Наибольшая из перечисленных статей МП" if len(known) == len(expense_keys) else "Наибольшая из статей МП с известной суммой"
                note = (f"{ranking_label}: {LABELS[largest]}" if largest else "Сравнение статей расходов недоступно")
                index = add_claim(values, current_period, selected_keys=[*expense_keys, "net_cogs"],
                    note=note + "; COGS показана отдельно от расходов МП; причины динамики не установлены")
                if index is not None:
                    required.append(index)
                    expense_indices.append(index)
        elif isinstance(data, FinanceSkuReport):
            warnings.extend(comparison_input_warnings(operation, data))
            period = f"{data.date_from}/{data.date_to}"
            rows = [row.model_dump(mode="json") for row in data.rows]
            matrix(f"SKU — {operation}, {period}, filter={scope or 'portfolio'}", rows, list(dict.fromkeys(["nm_id", *data.columns])))
            add_facts(data.totals.model_dump(mode="json"), period)
            index = add_claim(data.totals.model_dump(mode="json"), period,
                note="Матрица возвращённых финансовых SKU; не сумма портфельной маржи")
            if index is not None and operation == "sku":
                required.append(index)
            for row in rows:
                sku, nm_id = row["Артикул"], row["nm_id"]
                add_facts(row, period, sku=sku)
                index = add_claim(row, period, sku=sku, nm_id=nm_id,
                    selected_keys=["Выручка, руб (после возвратов)", "ИТОГО Расходы МП на артикул", "Себестоимость", "Маржа, руб", "Рентабельность, %", "МАРКЕТИНГ", "ЛОГИСТИКА"],
                    note="Нативные расчётные значения; без COGS маржа неполна; historical COGS version не подтверждена")
                if index is not None and operation == "sku" and sku:
                    current_candidates.append((index, row, "sku"))
        else:
            rows = [row.model_dump(mode="json") for row in data.rows]
            missing_sides = set()
            if policy_version == 2:
                if data.totals.current_total_sku_count == 0:
                    missing_sides.add("current")
                if data.totals.previous_total_sku_count == 0:
                    missing_sides.add("previous")
                missing_side_policy_used = missing_side_policy_used or bool(missing_sides)
            matrix(f"Изменение маржи: {current_period} относительно {previous_period}", rows, list(dict.fromkeys(["nm_id", *data.columns])))
            # Counts refer to a two-period union. Do not attach them to only one
            # window. Each current/previous/delta native column gets its own scope.
            index = add_claim(data.totals.model_dump(mode="json"), f"{current_period}; previous={previous_period}",
                note="missing_cost_count — native max, не union; вход каждой стороны ограничен top 500")
            if index is not None:
                required.append(index)
            for row in rows:
                sku, nm_id = row["Артикул"], row["nm_id"]
                for key, value in row.items():
                    if key in {"nm_id", "Артикул"}:
                        continue
                    period = previous_period if "пред." in key else current_period
                    if key.startswith("Δ") or key == "Роль":
                        # Generic legacy period is a display string; exact paired
                        # bounds are carried in metric definition below, not a false
                        # single-period source date.
                        period = current_period
                    before = len(facts)
                    add_facts({key: value}, period, sku=sku)
                    if len(facts) > before and missing_sides and (
                        "current" in missing_sides and "тек. период" in key
                        or "previous" in missing_sides and "пред. период" in key
                        or key.startswith("Δ") or key == "Роль"
                    ):
                        # Native COALESCE output is retained exactly. It is
                        # not an observed zero (nor comparable growth) when a
                        # source window has no financial rows. Old prepared
                        # candidates are read verbatim, never reclassified.
                        facts[-1]["status"] = "unconfirmed"
                        facts[-1]["metric_definition"] += " Missing source period=" + ",".join(sorted(missing_sides)) + "; native zero is not source availability proof."
                    if len(facts) > before and (key.startswith("Δ") or key == "Роль"):
                        facts[-1]["period"] = f"{current_period}; previous={previous_period}"
                        facts[-1]["period_start"] = None
                        facts[-1]["period_end"] = None
                        facts[-1]["metric_definition"] += f" Current={current_period}; previous={previous_period}; missing side margin native zero is not source availability proof."
                index = add_claim(row, f"{current_period}; previous={previous_period}", sku=sku, nm_id=nm_id,
                    selected_keys=["Маржа тек. период, руб", "Маржа пред. период, руб", "Δ маржи, руб", "Δ маржи, %", "Роль"],
                    note="Нативная динамика маржи, не подтверждённая первопричина; отсутствующая сторона может быть native zero")
                if index is not None and sku:
                    change_candidates.append((index, row, "change"))

    # Prioritization selects native measurements, never recomputes finance KPIs.
    # Portfolio + targeted reads can contain the same measured SKU. Rank
    # distinct identities before taking top-N so a deep dive cannot suppress
    # the next portfolio growth/decline. All scoped facts/tables remain intact.
    if policy_version == 2:
        def distinct_candidates(candidates):
            seen, selected = set(), []
            for item in candidates:
                claim = claims[item[0]]
                identity = (claim["sku"], claim["nm_id"], claim["period"])
                if identity not in seen:
                    seen.add(identity)
                    selected.append(item)
            return selected
        current_candidates = distinct_candidates(current_candidates)
        change_candidates = distinct_candidates(change_candidates)
    priorities = []
    missing = [item for item in current_candidates if item[2] == "sku" and item[1]["Себестоимость"] is None]
    declines = sorted([item for item in change_candidates if item[1]["Δ маржи, руб"] < 0], key=lambda item: item[1]["Δ маржи, руб"])
    growth = sorted([item for item in change_candidates if item[1]["Δ маржи, руб"] > 0], key=lambda item: item[1]["Δ маржи, руб"], reverse=True)
    spend = sorted([item for item in current_candidates if item[2] == "sku" and item[1]["МАРКЕТИНГ"] is not None and item[1]["МАРКЕТИНГ"] > 0], key=lambda item: item[1]["МАРКЕТИНГ"], reverse=True)
    priority_identities = set()
    for item in [*missing[:1], *declines[:2], *spend[:1], *current_candidates[:1], *change_candidates[:1]]:
        identity = (claims[item[0]]["sku"], claims[item[0]]["nm_id"])
        if (item[0] not in priorities and len(priorities) < 4
                and (policy_version == 1 or identity not in priority_identities)):
            priorities.append(item[0])
            priority_identities.add(identity)
    if request.date_from is None and request.operation != "overview":
        warnings.append("scope:default_completed_days_from_database_clock_not_overview_source_date")
    if end == today:
        warnings.append("scope:current_day_may_be_incomplete")
    warnings.append("semantics:marketplace_contribution_margin_not_net_profit_or_liquidity")
    if scoped_totals:
        warnings.append("scope:native_keyword_filter_exact_returned_rows_not_portfolio_coverage")
    if request.operation in {"marketplace", "report"} and request.scope == "portfolio":
        warnings.append("needs_data:portfolio_net_revenue_not_exposed_as_native_total")
    prepared = {"request": request.model_dump(mode="json"), "today": str(today), "period": current_period,
        "previous_period": previous_period, "facts": facts, "tables": tables, "claims": claims,
        "evidence": evidence, "warnings": list(dict.fromkeys(warnings)),
        "required_claim_indices": list(dict.fromkeys(required + priorities)), "priority_claim_indices": priorities,
        "required_operations": [request.operation] if request.operation != "report" else ["marketplace", "sku", "margin_change"] if request.scope == "portfolio" else ["sku", "margin_change"]}
    if policy_version == 2:
        decline_indices = [item[0] for item in declines[:2]]
        growth_indices = [item[0] for item in growth[:2]]
        prepared.update(analysis_policy_version=2,
            analytical_groups={"expenses": expense_indices, "declines": decline_indices, "growth": growth_indices},
            required_claim_indices=list(dict.fromkeys(required + priorities + growth_indices + decline_indices)),
            summary_claim_indices=list(dict.fromkeys(required[:3] + decline_indices + growth_indices)))
    if scoped_totals:
        # Explicit additive reader marker; already persisted unmarked claims,
        # facts and summaries retain their recorded publication semantics.
        prepared["scoped_totals_policy_version"] = 1
    if missing_side_policy_used:
        prepared["missing_side_provenance_policy_version"] = 1
    if not claims:
        raise ValueError("Finance has no verified observations")
    assert_bounded_safe(prepared, max_bytes=MAX_PREPARED_BYTES)
    return prepared
