"""Native snapshot context for latest diagnostics; no CPC/DRR/delta formulas."""
from datetime import timedelta
from hashlib import sha256
import re

from commerce_agent.adapters.contracts import MarketingSnapshot,ToolResult
from commerce_agent.marketing.composite import canonical
from commerce_agent.sales.contracts import EvidenceFact

STATS="marketing.wb_marketing_campaign_stats_daily"
SALES="sales.wb_supplier_sales"
SOURCES={STATS,SALES,"marketing.wb_marketing_campaign_nms","sales.wb_sales_funnel_products_daily"}
METRICS=(
    ("spend_rub","Расходы рекламы","RUB","spend_delta_rub","RUB"),
    ("clicks","Клики","clicks","clicks_delta","clicks"),
    ("ad_orders","Заказы в рекламной статистике","orders","ad_orders_delta","orders"),
    ("promoted_sku_revenue_rub","Вся выручка продвигаемых SKU, не ad attribution","RUB","promoted_sku_revenue_delta_rub","RUB"),
    ("drr_pct","ДРР ко всей выручке продвигаемых SKU, не ROI","%","drr_delta_pp","pp"),
    ("cpc_rub","CPC из native snapshot, не средний CPC campaign rows","RUB/click",None,None),
)
PCT={"spend_rub":"spend_delta_pct","clicks":"clicks_delta_pct","ad_orders":"ad_orders_delta_pct",
     "promoted_sku_revenue_rub":"promoted_sku_revenue_delta_pct"}


def prepare_snapshot(envelope,request,today,request_id,mode):
    if mode not in ("inline","hydrated"):raise ValueError("Marketing snapshot context source mode unknown")
    value=ToolResult[MarketingSnapshot].model_validate(envelope)
    if canonical(value.model_dump(mode="json",exclude_unset=True))!=canonical(envelope):raise ValueError("Marketing snapshot native types changed")
    data=value.data
    expected_period=data.rows[-1].current_period if data.rows else None
    expected_sources=SOURCES if data.report_date else set()
    if (data.focus_keyword!=request.product_keyword or set(data.source_tables)!=SOURCES or value.source!=STATS or value.period!=expected_period
        or value.freshness.as_of!=data.report_date or value.freshness.status!=("available" if data.report_date else "missing")
        or {e.source for e in value.evidence}!=expected_sources
        or any(e.period!=expected_period or e.grain!=("product_filter" if request.product_keyword else "portfolio") for e in value.evidence)):
        raise ValueError("Marketing snapshot context scope/provenance changed")
    if any(re.fullmatch(r"marketing\.report_snapshot:[a-f0-9]{64}",e.id) is None for e in value.evidence):
        raise ValueError("Marketing snapshot context evidence namespace changed")
    if data.rows:
        if data.report_date is None or len(data.rows)!=3:raise ValueError("Marketing snapshot context incomplete window family")
        for days,row in zip((1,7,30),data.rows,strict=True):
            current,previous=row.current_period,row.previous_period
            if (current.date_to!=data.report_date or current.date_from!=data.report_date-timedelta(days=days-1)
                or previous.date_to!=current.date_from-timedelta(days=1) or previous.date_from!=current.date_from-timedelta(days=days)):
                raise ValueError("Marketing snapshot context native windows changed")
    digest=sha256(canonical(envelope["data"])).hexdigest()[:16]
    evidence=[{"domain":"marketing","claim":"native filtered snapshot context, not campaign average or causal attribution",
        "source":e.source,"period":None,"freshness":"unknown","evidence_ref":f"marketing:{request_id}:{e.id}:{digest}"} for e in value.evidence]
    warnings=[*value.warnings,"semantics:snapshot_cpc_and_campaign_average_have_independent_native_definitions",
        "semantics:drr_not_profit_or_causal_ad_attribution","scope:snapshot_context_latest_day_week_month_not_requested_diagnostic_window"]
    if not data.rows:warnings.append("needs_data:composite_snapshot_context_missing")
    if data.report_date and data.report_date>today:warnings.append("needs_data:composite_snapshot_context_future")
    return value.model_dump(mode="json",exclude_unset=True),evidence,warnings


def append_snapshot_catalog(prepared,claim,table):
    value=ToolResult[MarketingSnapshot].model_validate(prepared["snapshot_context"]);data=value.data
    refs={e["source"]:e["evidence_ref"] for e in prepared["evidence"] if ":marketing.report_snapshot:" in e["evidence_ref"]}
    if set(refs)!=(SOURCES if data.report_date else set()):raise ValueError("Marketing snapshot context evidence missing")
    rows=value.model_dump(mode="json",exclude_unset=True)["data"]["rows"]
    scope="фильтр "+str(data.focus_keyword) if data.focus_keyword else "портфель"
    for number,row in enumerate(data.rows):
        current=f"{row.current_period.date_from}/{row.current_period.date_to}"
        previous=f"{row.previous_period.date_from}/{row.previous_period.date_to}"
        group=[]
        for metric,label,unit,delta,delta_unit in METRICS:
            sources={SALES} if metric=="promoted_sku_revenue_rub" else {STATS,SALES} if metric=="drr_pct" else {STATS}
            a,b=getattr(row,"current_"+metric),getattr(row,"previous_"+metric)
            change=getattr(row,delta) if delta else None
            valid="needs_data:composite_snapshot_context_future" not in prepared["warnings"]
            values=[("current_"+metric,a,current,unit),("previous_"+metric,b,previous,unit),(delta,change,current+" vs "+previous,delta_unit)]
            if metric in PCT:values.append((PCT[metric],getattr(row,PCT[metric]),current+" vs "+previous,"%"))
            for key,value,period,units in values:
                if key is None:continue
                source=SALES if metric=="promoted_sku_revenue_rub" else STATS
                prepared["facts"].append(EvidenceFact(metric="snapshot."+key,value=value,unit=units,period=period,
                    target_type="product_filter" if data.focus_keyword else "portfolio",target_id=data.focus_keyword,
                    source_table=source,evidence_ref=refs[source],source_agent="marketing",source_snapshot_at=str(data.report_date),
                    metric_definition=label+"; native snapshot scope, not exact SKU/campaign causality",
                    status="confirmed" if value is not None and valid else "unconfirmed",confidence="medium" if value is not None and valid else "low").model_dump(mode="json"))
            text=f"{label}, {scope}: {a if a is not None else 'нет данных'} {unit} за {current} против {b if b is not None else 'нет данных'} {unit} за {previous}."
            if delta:text+=f" Native изменение: {change if change is not None else 'нет данных'} {delta_unit}; не пересчитано моделью."
            if metric in PCT:text+=f" Native относительное изменение: {getattr(row,PCT[metric]) if getattr(row,PCT[metric]) is not None else 'нет данных'} %."
            index=claim("snapshot",text)
            prepared["claims"][index]["evidence_refs"]=[refs[s] for s in sorted(sources)]
            group.append(index)
        prepared["required_groups"].append(group)
    if not data.rows:prepared["required_groups"].append([claim("snapshot","Native snapshot context отсутствует; динамика расходов/CPC/ДРР не установлена, не равна нулю.")])
    prepared["tables"].append(table("Marketing snapshot context — native day/week/month windows",rows,list(rows[0]) if rows else ["period"]))
    prepared["tables"].append(data.report_table.model_dump(mode="json"))
