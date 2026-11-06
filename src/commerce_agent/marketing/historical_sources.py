"""Opt-in, typed context from exact historical sources; old journals stay unchanged."""
from copy import deepcopy
from commerce_agent.adapters.contracts import ToolResult
from commerce_agent.adapters.historical_sources import HistoricalSources
from commerce_agent.marketing.evidence import _hydrate


def receipt_byte_limit(prepared):
    if "historical_sources_version" in prepared or "historical_sources" in prepared:
        validate(prepared)
        return 524288
    return 262144


async def collect(client, context, prepared):
    start, end = prepared["period"].split("/")
    value = await _hydrate(client, context, await client.historical_sources(context,
        vendor_code=prepared["request"]["targets"][0], date_from=start, date_to=end), HistoricalSources)
    prepared["historical_sources_version"] = value.data.version
    prepared["historical_sources"] = value.model_dump(mode="json")
    if value.data.version == 2:
        from commerce_agent.marketing.coverage import historical_manifest
        prepared["coverage_manifest_version"] = 1
        prepared["coverage_manifest"] = historical_manifest(value)
    validate(prepared)
    return prepared


def validate(prepared):
    if "historical_sources_version" not in prepared:
        if any(key in prepared for key in ("historical_sources", "coverage_manifest_version", "coverage_manifest")):
            raise ValueError("historical sources require version")
        return None
    if type(prepared["historical_sources_version"]) is not int or prepared["historical_sources_version"] not in (1, 2):
        raise ValueError("unsupported historical sources version")
    value = ToolResult[HistoricalSources].model_validate(prepared["historical_sources"])
    data = value.data
    if prepared["historical_sources_version"] != data.version:
        raise ValueError("historical source marker mismatch")
    if "coverage_manifest_version" in prepared or "coverage_manifest" in prepared:
        from commerce_agent.adapters.data_coverage import CoverageManifest
        from commerce_agent.marketing.coverage import historical_manifest
        if (type(prepared.get("coverage_manifest_version")) is not int or prepared["coverage_manifest_version"] != 1
                or data.version != 2 or "coverage_manifest" not in prepared):
            raise ValueError("unsupported coverage manifest marker")
        manifest = CoverageManifest.model_validate(prepared["coverage_manifest"]).model_dump(mode="json")
        if manifest != historical_manifest(value):
            raise ValueError("coverage manifest source or scope mismatch")
    ids = {c.get("nm_id") for c in prepared["claims"] if c.get("sku") == data.vendor_code and c.get("nm_id")}
    if (prepared["request"]["targets"] != [data.vendor_code]
        or f"{data.requested_period.date_from}/{data.requested_period.date_to}" != prepared["period"]
        or f"{data.previous_period.date_from}/{data.previous_period.date_to}" != prepared["previous_period"]
        or data.identity_status == "resolved" and ids != {data.nm_id}
        or value.period is not None or value.source != data.sources[0]
        or {e.source for e in value.evidence} != {source for source, rows in zip(data.sources,
            (data.funnel,data.search_details,data.search_texts,data.serp)) if rows}
        or any(e.period is not None or e.grain != "product_filter" for e in value.evidence)):
        raise ValueError("historical sources scope or provenance mismatch")
    return data


def model_context(prepared):
    data = validate(prepared)
    return {} if data is None else {"historical_sources": deepcopy(prepared["historical_sources"]),
        **({"coverage_manifest": deepcopy(prepared["coverage_manifest"])} if "coverage_manifest_version" in prepared else {}),
        "historical_sources_role": "Exact historical observations; use each row's measurement window, never sum overlapping search snapshots. Replay dates are simulated."}


def render(prepared):
    data = validate(prepared)
    if data is None:
        return ""
    parts = ["Исторические источники за обе недели", "Источник воронки: "+data.sources[0]]
    parts += ["ДЕМОИСТОРИЯ: строки с replay_batch перенесены из original_date; это не реальные наблюдения на сентябрьскую дату."] if any(r.replay_batch for r in data.funnel+data.search_details+data.search_texts+data.serp) else []
    parts.append("Период | Дней из ожидаемых | Открытия | Корзины | Заказы воронки | Сумма заказов")
    parts.append("--- | --- | --- | --- | --- | ---")
    for s in data.funnel_periods:
        parts.append(f"{s.period.date_from} — {s.period.date_to} | {s.observed_days}/{s.expected_days} | {s.open_count} | {s.cart_count} | {s.order_count} | {s.order_sum}")
    if data.version == 2:
        from commerce_agent.sales.historical_funnel import render_delivery
        parts.append(render_delivery(data))
    for label, period in (("Предыдущая неделя",data.previous_period),("Текущая неделя",data.requested_period)):
        exact = [r for r in data.search_details if r.current_period_start==period.date_from and r.current_period_end==period.date_to]
        parts.append(f"{label}: точных агрегатов поисковой видимости — {len(exact)}. Отсутствие агрегата означает неизвестное значение.")
    parts.append(f"Доступно поисковых срезов: {len(data.search_details)}; строк поисковых текстов: {len(data.search_texts)}; наблюдений выдачи: {len(data.serp)}. Окна пересекаются; их суммы не являются дневным спросом.")
    parts.append("Ограничения: "+"; ".join(data.limitations))
    parts.append("Происхождение связанных источников: строки / перенесённые строки / исходные даты")
    for origin in data.related_provenance:
        parts.append(f"{origin.source}: {origin.rows} / {origin.replayed_rows} / {origin.original_date_from} — {origin.original_date_to}")
    return "\n".join(parts)
