"""Trusted read boundary for the first inline scoped debug source profile.

Validate BEFORE completing a tool span, not after its data has been exported.
No SQL/calculation/action implementation lives here. Static strings below
witness existing service metadata; they are never executed as formulas.
"""
from copy import deepcopy
from datetime import date, datetime
import re
from types import SimpleNamespace

from langchain_core.runnables import RunnableLambda

from commerce_agent.adapters.business_tools import BusinessToolClient
from commerce_agent.adapters.contracts import (DatabaseTime,EmptyToolRequest,MarketingSnapshotRequest,
    MarketingSnapshot,MarketingSkuComparisonRequest,MarketingSkuComparison,MarketingDrrRequest,
    MarketingDrrReport,MarketingCompositeRequest,MarketingActionMatrix,ToolArtifactReference,ToolResult)
from commerce_agent.marketing.scoped import canonical,validate_target,NATIVE_WARNINGS,STATS,SALES
from commerce_agent.marketing.portfolio_campaigns import validate as validate_portfolio,DRR_WARNINGS,ACTION_WARNINGS,read_plan
from commerce_agent.marketing.proposal_privacy import project_native_proposal,project_native_nm_ids,OMITTED,FAMILIES
from commerce_agent.marketing.scoped_debug_export import assert_scoped_debug_request

SNAPSHOT_LABELS = ("Последний день в данных к предыдущему", "Последние 7 дней к предыдущим 7", "Последние 30 дней к предыдущим 30")
SNAPSHOT_COLUMNS = ["Сравнение","Периоды","Расход тек./пред.","Δ расхода","Клики тек./пред.","CPC тек./пред.","Заказы рекламы тек./пред.","Выручка SKU тек./пред.","ДРР тек./пред.","Δ ДРР п.п."]
SNAPSHOT_REQUIRED = ["period","current_period","previous_period","current_spend_rub","previous_spend_rub","spend_delta_rub","spend_delta_pct","current_clicks","previous_clicks","current_cpc_rub","previous_cpc_rub","current_ad_orders","previous_ad_orders","current_promoted_sku_revenue_rub","previous_promoted_sku_revenue_rub","current_drr_pct","previous_drr_pct","drr_delta_pp"]
SNAPSHOT_CALCULATIONS = {
    "cpc_rub":"marketing_spend_rub / clicks", "drr_delta_pp":"current_drr_pct - previous_drr_pct",
    "drr_pct":"marketing_spend_rub / promoted_sku_revenue_rub * 100",
    "month_to_month_default":"rolling 30 days vs previous 30 days",
    "promoted_sku_revenue":"sales.wb_supplier_sales.for_pay for SKU linked to campaigns with spend in the same period side",
    "spend_delta_pct":"(current_spend_rub - previous_spend_rub) / previous_spend_rub * 100",
    "spend_delta_rub":"current_spend_rub - previous_spend_rub"}
SNAPSHOT_NOTE = "ДРР считается как расходы рекламы / выручка продвигаемых SKU * 100. Выручка продвигаемых SKU означает все фактические продажи SKU, которые связаны с кампаниями со spend > 0 в этой стороне периода."
SKU_NOTE = "WB campaign stats are campaign-level. For campaigns linked to multiple SKU, impressions, clicks, spend and ad orders are equally allocated across linked SKU and marked as estimated_allocated_from_campaign."
DRR_NOTE = "Это оценочный ДРР по всем кампаниям со spend > 0 за период. Выручка продвигаемых артикулов означает все фактические продажи SKU, которые были привязаны к рекламным кампаниям, а не строго атрибутированные рекламные продажи."
CLOCK_WARNINGS = {"semantics:database_clock_not_business_data_freshness","scope:database_session_timezone"}
STATES = {"ready","partial","invalid_freshness","missing","unpopulated","stale","legacy_fallback"}
OPERATIONS = {
    "shared/database-time":(EmptyToolRequest,DatabaseTime),
    "marketing/report-snapshot":(MarketingSnapshotRequest,MarketingSnapshot),
    "marketing/sku-period-comparison":(MarketingSkuComparisonRequest,MarketingSkuComparison),
    "marketing/drr-report":(MarketingDrrRequest,MarketingDrrReport),
    "marketing/action-matrix":(MarketingCompositeRequest,MarketingActionMatrix)}


def scoped_debug_read_plan(request):
    assert_scoped_debug_request(request)
    start,end=request.date_from,request.date_to
    return [{"operation":"shared/database-time","arguments":{}},
        {"operation":"marketing/report-snapshot","arguments":{"product_keyword":None}},
        *[{"operation":"marketing/sku-period-comparison","arguments":{"date_from":str(start),"date_to":str(end),
            "sku_list":target,"product_keyword":None,"period_days":request.days,"limit":100}} for target in request.targets],
        *read_plan(start,end)]


def assert_debug_clock(envelope):
    value=ToolResult[DatabaseTime].model_validate(envelope)
    if (canonical(value.model_dump(mode="json",exclude_unset=True))!=canonical(envelope)
        or value.source!="business_database.clock" or value.evidence or value.period is not None
        or set(value.warnings)!=CLOCK_WARNINGS or len(value.warnings)!=2
        or value.freshness.as_of!=value.data.db_current_date or value.freshness.status!="available"
        or datetime.fromisoformat(value.data.db_now.replace("Z","+00:00")).date()!=value.data.db_current_date):
        raise ValueError("Marketing scoped debug clock not accepted")
    return value.data.db_current_date


def _warnings(values, allowed):
    if len(values)!=len(set(values)) or not set(values)<=allowed:
        raise ValueError("Marketing scoped debug warning source not accepted")


def _vendor(value):
    if not isinstance(value,str) or re.fullmatch(r"[1-9][0-9]{4}",value) is None:
        raise ValueError("Marketing scoped debug identifier profile not accepted")


def validate_debug_source(operation,envelope,arguments,request,today):
    """Source-owned schema/lineage/text eligibility, not semantic name detection."""
    if operation=="shared/database-time":
        return assert_debug_clock(envelope)
    if today is None:
        raise ValueError("Marketing scoped debug clock missing")
    if operation=="marketing/report-snapshot":
        from commerce_agent.marketing.composite_snapshot import prepare_snapshot
        prepare_snapshot(envelope,SimpleNamespace(product_keyword=None),today,"debug-proof", "inline")
        data=envelope["data"]
        _warnings(envelope["warnings"],{"provenance_only:source_completeness_unverified"})
        if (not data["report_date"] or data["report_date"]>str(today) or len(data["rows"])!=3
            or data["required_fields"]!=SNAPSHOT_REQUIRED or data["calculations"]!=SNAPSHOT_CALCULATIONS
            or data.get("interpretation_note")!=SNAPSHOT_NOTE
            or [r["period"] for r in data["rows"]]!=list(SNAPSHOT_LABELS)):
            raise ValueError("Marketing scoped debug snapshot profile not accepted")
        table=data["report_table"]
        if table["title"]!="Marketing-сводка" or table["columns"]!=SNAPSHOT_COLUMNS or len(table["rows"])!=3:
            raise ValueError("Marketing scoped debug snapshot table not accepted")
        for row,label in zip(table["rows"],SNAPSHOT_LABELS,strict=True):
            if len(row)!=10 or row[0]!=label or any(not isinstance(c,str) or (c!="нет данных" and re.fullmatch(r"[0-9.,+±/%₽\s()–—\-кп\.]+",c) is None) for c in row[1:]):
                raise ValueError("Marketing scoped debug snapshot text not accepted")
        expected="| "+" | ".join(table["columns"])+" |\n|"+"|".join("---" for _ in table["columns"])+"|\n"+"\n".join("| "+" | ".join(row)+" |" for row in table["rows"])
        if data["markdown_table"]!=expected:
            raise ValueError("Marketing scoped debug snapshot mirror not accepted")
    elif operation=="marketing/sku-period-comparison":
        validate_target({"target":arguments["sku_list"],"mode":"inline","envelope":envelope},arguments["sku_list"],request.date_from,request.date_to)
        data=envelope["data"]
        _warnings(envelope["warnings"],NATIVE_WARNINGS|{"needs_data:sku_comparison_rows_missing","needs_data:realized_sales_coverage_unconfirmed","scope:row_limit_may_hide_additional_skus"})
        if data["allocation_note"]!=SKU_NOTE:
            raise ValueError("Marketing scoped debug allocation metadata not accepted")
        for row in data["rows"]:_vendor(row["vendor_code"])
    elif operation in {"marketing/drr-report","marketing/action-matrix"}:
        expected=next(v for v in read_plan(request.date_from,request.date_to) if v["operation"]==operation)
        validate_portfolio({"operation":operation,"arguments":arguments,"mode":"inline","envelope":envelope},expected,today)
        data=envelope["data"]
        if operation=="marketing/drr-report":
            _warnings(envelope["warnings"],DRR_WARNINGS|{"needs_data:campaign_spend_coverage_unconfirmed","needs_data:realized_sales_coverage_unconfirmed","scope:row_limit_may_hide_additional_campaigns_or_skus"})
            if data["interpretation_note"]!=DRR_NOTE or any(project_native_nm_ids(r["nm_ids"])==OMITTED for r in data["campaign_rows"]):
                raise ValueError("Marketing scoped debug DRR metadata not accepted")
            for row in data["promoted_sku_rows"]:_vendor(row["vendor_code"])
        else:
            _warnings(envelope["warnings"],ACTION_WARNINGS|{"needs_data:invalid_mart_freshness_blocks_bid_actions","needs_data:campaign_reference_date_missing","needs_data:invalid_legacy_freshness_blocks_bid_actions","needs_data:action_proposals_missing","scope:action_limit_may_hide_additional_proposals","quality:legacy_demand_fallback_not_ready_demand_mart"})
            if (data["marketing_source_status"] not in STATES or data["demand_source_status"] not in STATES
                or any(v not in STATES|{"used","not_used"} for v in data["supporting_sources"].values())
                or data["source_freshness"]["interpretation"] not in {
                    "All action sources are fresh enough for current bid actions.",
                    "Only fresh rows are used for current bid actions; stale, future-dated, or undated rows are blocked."}):
                raise ValueError("Marketing scoped debug action metadata not accepted")
            for family in FAMILIES:
                for row in data[family]:
                    projected=project_native_proposal(row,family=family)
                    private={"entity_name","norm_query"}|({"entity_id"} if family=="query_actions" else set())
                    if any(k not in private and (v==OMITTED or isinstance(v,list) and OMITTED in v) for k,v in projected.items()):
                        raise ValueError("Marketing scoped debug proposal text not accepted")
    else:
        raise ValueError("Marketing scoped debug operation not accepted")
    return today


class ScopedDebugReadBoundary:
    """Only the approved ordered read plan; HTTP auth remains in business client."""
    def __init__(self,client,request):
        if not isinstance(client,BusinessToolClient):
            raise ValueError("Marketing scoped debug requires typed business client")
        self.client,self.request=client,request
        self.plan=scoped_debug_read_plan(request)
        self.index=0;self.today=None;self.clock_record=None

    async def _read(self,operation,context,arguments):
        expected={"operation":operation,"arguments":arguments}
        if self.index>=len(self.plan) or canonical(self.plan[self.index])!=canonical(expected):
            raise ValueError("Marketing scoped debug read plan changed")
        self.index+=1
        request_type,result_type=OPERATIONS[operation]
        async def invoke(params):
            value=await self.client._operation_untraced(context,params,request_type,result_type,"/internal/v1/agent-tools/"+operation)
            if isinstance(value,ToolArtifactReference):
                raise ValueError("Marketing scoped debug artifact source not accepted")
            envelope=value.model_dump(mode="json",exclude_unset=True)
            self.today=validate_debug_source(operation,envelope,params,self.request,self.today)
            if operation=="shared/database-time":self.clock_record={"mode":"inline","envelope":deepcopy(envelope)}
            return value
        name="tool."+operation.replace("/",".").replace("-","_")
        return await RunnableLambda(invoke,name=name).ainvoke(arguments)

    async def database_time(self,context):return await self._read("shared/database-time",context,{})
    async def marketing_report_snapshot(self,context,*,product_keyword=None):return await self._read("marketing/report-snapshot",context,{"product_keyword":product_keyword})
    async def marketing_sku_period_comparison(self,context,**arguments):return await self._read("marketing/sku-period-comparison",context,arguments)
    async def marketing_drr_report(self,context,**arguments):return await self._read("marketing/drr-report",context,arguments)
    async def marketing_action_matrix(self,context,**arguments):return await self._read("marketing/action-matrix",context,arguments)
    async def hydrate(self,*args,**kwargs):raise ValueError("Marketing scoped debug hydration not accepted")
