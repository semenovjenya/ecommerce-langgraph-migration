"""Exact-period funnel evidence. Totals come from the typed business response."""
from copy import deepcopy
from hashlib import sha256
import json
from commerce_agent.adapters.historical_sources import HistoricalSources, FUNNEL, DETAILS, TEXTS, SERP, DELIVERY_FIELDS, DELIVERY_LIMITATIONS
from commerce_agent.sales.contracts import EvidenceFact
from commerce_agent.sales.evidence import _hydrate


def native(prepared):
    raw = prepared.get("historical_funnel")
    if raw is None:
        if prepared.get("historical_funnel_version") is not None or any(f["metric"].startswith("historical_funnel.") for f in prepared["facts"]):
            raise ValueError("historical funnel lost source")
        return None
    if type(prepared.get("historical_funnel_version")) is not int or prepared["historical_funnel_version"] not in (1, 2):
        raise ValueError("historical funnel version missing")
    data = HistoricalSources.model_validate(raw["data"])
    if data.version != prepared["historical_funnel_version"]:
        raise ValueError("historical funnel marker mismatch")
    request = prepared["request"]
    ids = {r.get("nm_id") for scope in prepared.get("scoped_sales_scopes", [])
        for r in scope["native_change"]["rows"] if r.get("nm_id") is not None}
    if (request["targets"] != [data.vendor_code] or str(data.requested_period.date_from) != request["date_from"]
        or str(data.requested_period.date_to) != request["date_to"] or data.search_details or data.search_texts or data.serp
        or data.identity_status == "resolved" and ids != {data.nm_id}
        or raw["evidence"]["source"] != FUNNEL or raw["evidence"]["domain"] != "sales"
        or raw["evidence"] not in prepared["evidence"]):
        raise ValueError("historical funnel scope/provenance changed")
    signature = sha256(json.dumps(raw["data"],sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:16]
    if not raw["evidence"]["evidence_ref"].endswith(":"+signature):
        raise ValueError("historical funnel receipt changed")
    if [f for f in prepared["facts"] if f["metric"].startswith("historical_funnel.")] != facts(data,raw["evidence"]):
        raise ValueError("historical funnel facts changed")
    return data


def facts(data, evidence):
    return [EvidenceFact(metric="historical_funnel."+metric, value=getattr(s,metric),
        unit="RUB" if metric=="order_sum" else "units", sku=data.vendor_code,
        period=f"{s.period.date_from}/{s.period.date_to}",source_table=FUNNEL,
        source_agent="sales",evidence_ref=evidence["evidence_ref"],confidence="low").model_dump(
            mode="json",exclude_none=True,exclude_defaults=True)
        for s in data.funnel_periods if s.complete
        for metric in ("open_count","cart_count","order_count","order_sum")]


async def collect(client,context,request,prepared):
    if request.scope != "single_sku" or not request.date_from:
        raise ValueError("historical funnel requires exact single SKU request")
    value = await _hydrate(client,context,await client.historical_sources(context,vendor_code=request.targets[0],
        date_from=str(request.date_from),date_to=str(request.date_to)),HistoricalSources)
    ids = {r.get("nm_id") for scope in prepared.get("scoped_sales_scopes", [])
        for r in scope["native_change"]["rows"] if r.get("nm_id") is not None}
    expected = {source for source, rows in zip(value.data.sources,
        (value.data.funnel,value.data.search_details,value.data.search_texts,value.data.serp)) if rows}
    if (value.source != FUNNEL or value.period is not None or {e.source for e in value.evidence} != expected
        or any(e.period is not None or e.grain != "product_filter" for e in value.evidence)
        or value.data.identity_status == "resolved" and ids != {value.data.nm_id}):
        raise ValueError("historical funnel native identity/provenance mismatch")
    data = value.data.model_dump(mode="json")
    # Sales retains only its funnel source. The complete search context is a
    # separate Marketing receipt, never relabelled as a Sales observation.
    data.update(search_details=[],search_texts=[],serp=[])
    data["limitations"] = ["funnel_orders_are_not_realized_sales_or_ad_attribution",
        "missing_observations_are_unknown_not_zero","observational_data_does_not_establish_causality"]
    if data["version"] == 2:
        data["limitations"].extend(DELIVERY_LIMITATIONS)
    if any(r["replay_batch"] for r in data["funnel"]):
        data["limitations"].append("DEMO_REPLAY:rows_were_copied_from_original_date_not_observed_on_target_date")
    if not data["lineage_available"]:
        data["limitations"].append("lineage_table_unavailable_origin_not_verified")
    parsed = HistoricalSources.model_validate(data)
    signature = sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:16]
    ref = dict(evidence_ref=f"sales:{context.request_id}:historical-funnel:{signature}",domain="sales",
        claim="Exact historical funnel; non-causal",source=FUNNEL,
        period=f"{request.date_from}/{request.date_to}",freshness="unknown")
    prepared.update(historical_funnel_version=data["version"],historical_funnel=dict(data=data,evidence=ref))
    prepared["evidence"].append(ref)
    prepared["facts"].extend(facts(parsed,ref))
    prepared["warnings"].extend(parsed.limitations)
    native(prepared)
    return prepared


def claims(prepared):
    data=native(prepared)
    if data is None:return []
    ref=prepared["historical_funnel"]["evidence"]["evidence_ref"]
    replay=any(r.replay_batch for r in data.funnel)
    result=[]
    for s in data.funnel_periods:
        text=(f"Воронка SKU {data.vendor_code} за {s.period.date_from}/{s.period.date_to}: "
            f"дней {s.observed_days}/{s.expected_days}; открытия {s.open_count}; корзины {s.cart_count}; "
            f"заказы воронки {s.order_count}; сумма заказов {s.order_sum}. "
            "Заказы воронки не равны реализованным продажам и не являются рекламной атрибуцией.")
        if replay:text+=" Демоданные перенесены из исходных дат; реальное изменение в сентябре не доказано."
        result.append(dict(text=text,sku=data.vendor_code,period=f"{s.period.date_from}/{s.period.date_to}",
            evidence_refs=[ref],factor="Воронка",confidence="low"))
    return result


def render(prepared):
    data=native(prepared)
    if data is None:return ""
    lines=[c["text"] for c in claims(prepared)]
    lines += ["Дата | Открытия | Корзины | Заказы | Исходная дата | Перенос", "--- | --- | --- | --- | --- | ---"]
    lines += [f"{r.day} | {r.open_count} | {r.cart_count} | {r.order_count} | {r.original_date} | {r.replay_batch}" for r in data.funnel]
    if data.version == 2:
        lines.append(render_delivery(data))
    return "\n".join(lines)


def render_delivery(data):
    """Display the native series, including NULL/zero; never invent weekly averages."""
    def duration(row, prefix):
        values = [getattr(row, prefix + suffix) for suffix in ("days", "hours", "mins")]
        return "/".join("NULL" if v is None else str(v) for v in values)
    lines = ["Доставка и локализация по исходным дневным срезам (дни/часы/минуты).",
        "Нулевой срок неизвестен, не означает мгновенную доставку. Без весов недельное среднее не рассчитывается. Это не срок поставки на склад или обещание доставки в регион.",
        "Дата | Срок | Локализация % | Прошлое окно | Прошлый срок | Прошлая локализация % | Динамика срока | Динамика локализации | Исходная дата | Перенос | Происхождение",
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---"]
    lines += [f"{r.day} | {duration(r, 'time_to_ready_')} | {r.localization_percent} | {r.past_period_start} — {r.past_period_end} | {duration(r, 'past_time_to_ready_')} | {r.past_localization_percent} | {duration(r, 'comparison_time_to_ready_')} | {r.comparison_localization_percent_dynamic} | {r.original_date} | {r.replay_batch} | {r.observation_origin}" for r in data.funnel]
    return "\n".join(lines)


def validate_delivery_receipt(raw, *, vendor_code, date_from, date_to, evidence):
    """Revalidate the bounded native Sales extension at the Marketing boundary."""
    if not isinstance(raw, dict) or set(raw) != {"data", "evidence"}:
        raise ValueError("delivery receipt shape mismatch")
    data = HistoricalSources.model_validate(raw["data"])
    ref = raw["evidence"]
    signature = sha256(json.dumps(raw["data"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    if (data.version != 2 or data.vendor_code != vendor_code
        or str(data.requested_period.date_from) != date_from or str(data.requested_period.date_to) != date_to
        or data.search_details or data.search_texts or data.serp
        or ref not in evidence or ref["source"] != FUNNEL or ref["domain"] != "sales"
        or ref["period"] != date_from + "/" + date_to
        or not ref["evidence_ref"].endswith(":" + signature)):
        raise ValueError("delivery receipt scope or provenance mismatch")
    return data
