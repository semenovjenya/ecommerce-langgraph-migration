"""Native portfolio campaign matrices and read-only proposals, never execution."""
from datetime import date
from hashlib import sha256
import re

from commerce_agent.adapters.contracts import (MarketingDrrReport,MarketingDrrRequest,
    MarketingActionMatrix,MarketingCompositeRequest,ToolResult,ToolArtifactReference)
from commerce_agent.adapters.business_tools import ToolAdapterError
from commerce_agent.marketing.composite import MARTS
from commerce_agent.marketing.composite_analysis import _table
from commerce_agent.marketing.scoped import canonical,STATS,SALES,ERRORS
from commerce_agent.sales.contracts import EvidenceFact

MODELS={"marketing/drr-report":(MarketingDrrRequest,MarketingDrrReport),
    "marketing/action-matrix":(MarketingCompositeRequest,MarketingActionMatrix)}
DRR_WARNINGS={"freshness:source_date_unconfirmed","semantics:estimated_drr_not_ad_attributed_roi",
    "provenance_only:latest_campaign_links_not_historical_attribution","scope:totals_and_returned_top_rows_differ"}
ACTION_WARNINGS={"quality:read_only_proposals_not_manager_decision_or_approval",
    "freshness:relative_campaign_reference_not_wall_clock","scope:independent_source_windows_no_shared_period",
    "scope:summary_counts_before_output_limit","scope:keyword_matches_campaign_query_or_sku_by_source",
    "provenance_only:declared_sources_not_complete_coverage","freshness:visibility_review_proposals_not_freshness_gated"}


def read_plan(start,end):
    return [{"operation":"marketing/drr-report","arguments":{"date_from":str(start),"date_to":str(end),"limit":50}},
        {"operation":"marketing/action-matrix","arguments":{"product_keyword":None,"days":(end-start).days+1,"limit":20}}]


async def collect(client,context,start,end):
    records=[]
    for entry in read_plan(start,end):
        op,args=entry["operation"],entry["arguments"]
        if op=="marketing/action-matrix" and args["days"]>30:
            records.append({**entry,"mode":"unsupported","reason":"native_action_window_exceeds_30_days"});continue
        try:
            result=await (client.marketing_drr_report(context,**args) if op=="marketing/drr-report" else client.marketing_action_matrix(context,**args))
            mode="hydrated" if isinstance(result,ToolArtifactReference) else "inline"
            if isinstance(result,ToolArtifactReference):result=await client.hydrate(context,result,result_type=MODELS[op][1])
            # Sparse proposals must not acquire absent optional null fields.
            records.append({**entry,"mode":mode,"envelope":result.model_dump(mode="json",exclude_unset=True)})
        except ToolAdapterError as exc:
            if exc.code not in ERRORS:raise ValueError("unknown portfolio adapter receipt") from None
            records.append({**entry,"mode":"failed","error_code":exc.code})
    return records


def validate(record,expected,today):
    op=expected["operation"]
    if record.get("operation")!=op or canonical(record.get("arguments"))!=canonical(expected["arguments"]):
        raise ValueError("portfolio native operation/request changed")
    if record.get("mode")=="failed":
        if set(record)!={"operation","arguments","mode","error_code"} or record["error_code"] not in ERRORS:
            raise ValueError("portfolio source failure receipt changed")
        return None
    if record.get("mode")=="unsupported":
        if canonical(record)!=canonical({**expected,"mode":"unsupported","reason":"native_action_window_exceeds_30_days"}) or op!="marketing/action-matrix" or expected["arguments"]["days"]<=30:
            raise ValueError("portfolio unsupported source receipt changed")
        return None
    if set(record)!={"operation","arguments","mode","envelope"} or record["mode"] not in {"inline","hydrated"}:
        raise ValueError("portfolio source mode changed")
    request_type,result_type=MODELS[op]
    args=request_type.model_validate(record["arguments"]).model_dump(mode="json")
    result=ToolResult[result_type].model_validate(record["envelope"])
    if canonical(args)!=canonical(record["arguments"]) or canonical(result.model_dump(mode="json",exclude_unset=True))!=canonical(record["envelope"]):
        raise ValueError("portfolio native JSON types/defaults changed")
    data=result.data
    if op=="marketing/drr-report":
        if (str(data.date_from)!=args["date_from"] or str(data.date_to)!=args["date_to"]
            or len(data.campaign_rows)>50 or len(data.promoted_sku_rows)>50
            or data.spend_source!=STATS+".sum_spend" or data.sales_source!=SALES+".for_pay where is_realization = true"
            or result.source!=STATS or result.freshness.as_of is not None or result.freshness.status!="missing"
            or result.period is None or result.period.date_from!=data.date_from or result.period.date_to!=data.date_to
            or not DRR_WARNINGS<=set(result.warnings)):
            raise ValueError("portfolio DRR native source/window changed")
        sources=set()
        if data.campaign_rows or data.totals and data.totals.campaign_count:sources.add(STATS)
        if data.promoted_sku_rows or data.totals and data.totals.store_sales:sources.add(SALES)
        ids=[r.advert_id for r in data.campaign_rows]
        sku_ids=[(r.vendor_code,r.nm_id) for r in data.promoted_sku_rows]
        if (any(i is None or i<=0 for i in ids) or len(ids)!=len(set(ids))
            or any(not v or n is None or n<=0 for v,n in sku_ids) or len(sku_ids)!=len(set(sku_ids))):
            raise ValueError("portfolio DRR native identities changed")
        if STATS not in sources and "needs_data:campaign_spend_coverage_unconfirmed" not in result.warnings:
            raise ValueError("portfolio missing campaign gap omitted")
        if SALES not in sources and "needs_data:realized_sales_coverage_unconfirmed" not in result.warnings:
            raise ValueError("portfolio missing realized sales gap omitted")
        if (len(data.campaign_rows)==50 or len(data.promoted_sku_rows)==50) and "scope:row_limit_may_hide_additional_campaigns_or_skus" not in result.warnings:
            raise ValueError("portfolio returned row limit omitted")
    else:
        f=data.source_freshness
        if (data.focus_keyword is not None or data.period_days!=args["days"] or result.period is not None
            or result.source!=MARTS["campaign_mart"] or not ACTION_WARNINGS<=set(result.warnings)
            or any(len(getattr(data,k))>20 for k in ("campaign_actions","query_actions","sku_actions"))):
            raise ValueError("portfolio native action scope changed")
        states={k:getattr(f.sources,k).status for k in MARTS}
        if f.stale_sources!=[k for k,v in states.items() if v=="stale"] or f.invalid_sources!=[k for k,v in states.items() if v!="fresh"]:
            raise ValueError("portfolio native relative freshness diagnostics changed")
        required_warnings=set(ACTION_WARNINGS)
        if f.invalid_sources:required_warnings.add("needs_data:invalid_mart_freshness_blocks_bid_actions")
        if f.reference_date is None:required_warnings.add("needs_data:campaign_reference_date_missing")
        if any(x and x.status!="fresh" for x in (f.legacy_campaign,f.legacy_query)):
            required_warnings.add("needs_data:invalid_legacy_freshness_blocks_bid_actions")
        if not (data.campaign_actions or data.query_actions or data.sku_actions):required_warnings.add("needs_data:action_proposals_missing")
        if len(data.campaign_actions)==20 or len(data.query_actions)==20:required_warnings.add("scope:action_limit_may_hide_additional_proposals")
        if data.supporting_sources.legacy_demand_fallback=="used":required_warnings.add("quality:legacy_demand_fallback_not_ready_demand_mart")
        if not required_warnings<=set(result.warnings):raise ValueError("portfolio native action limitations omitted")
        observed=[getattr(f.sources,k) for k in MARTS]+[x for x in (f.legacy_campaign,f.legacy_query) if x]
        dates=[x.report_date for x in observed if x.report_date]+([data.report_date] if data.report_date else [])
        as_of=min(dates) if dates else None
        if result.freshness.as_of!=as_of or result.freshness.status!=("available" if as_of else "missing"):
            raise ValueError("portfolio action native source date changed")
        sources={s for k,s in MARTS.items() if getattr(f.sources,k).status=="fresh" and getattr(data.supporting_sources,k)=="ready"}
        if data.campaign_actions and f.legacy_campaign and f.legacy_campaign.status=="fresh":sources.add(STATS)
        if data.query_actions and f.legacy_query and f.legacy_query.status=="fresh":sources.add("marketing.wb_search_query_orders_text_daily")
        for rows,kind in ((data.campaign_actions,"кампания"),(data.query_actions,"поисковый запрос"),(data.sku_actions,"артикул")):
            seen=set()
            for row in rows:
                key=(row.entity_id,row.advert_id,row.norm_query,row.query_band,row.nm_id,row.vendor_code)
                if row.entity_type!=kind or key in seen:raise ValueError("portfolio native proposal family/identity changed")
                seen.add(key)
                if kind=="кампания" and (row.advert_id is None or row.advert_id<=0 or str(row.entity_id)!=str(row.advert_id)):
                    raise ValueError("portfolio proposal campaign identity changed")
    if {e.source for e in result.evidence}!=sources or len(result.evidence)!=len(sources):
        raise ValueError("portfolio native evidence source coverage changed")
    prefix=op.replace("/",".").replace("-","_")+":"
    if any(e.grain!="portfolio" or e.period!=result.period or re.fullmatch(re.escape(prefix)+r"[a-f0-9]{64}",e.id) is None for e in result.evidence):
        raise ValueError("portfolio native evidence namespace/period changed")
    return result


TOTAL_UNITS={"campaign_count":"campaigns","promoted_nm_count":"SKU","marketing_spend":"RUB","views":"impressions",
    "clicks":"clicks","ad_orders":"orders","store_revenue":"RUB","store_sales":"units","promoted_sku_revenue":"RUB",
    "promoted_sku_sales":"units","promoted_revenue_share_pct":"%","promoted_sales_share_pct":"%",
    "estimated_drr_promoted_sku_pct":"%","estimated_drr_store_pct":"%"}


def append_catalog(p,records,start,end,today,claim,warnings):
    if len(records)!=2:raise ValueError("portfolio campaign read coverage changed")
    period=f"{start}/{end}"
    def add(text,refs,required=True,kind="portfolio_campaign"):
        index=claim(text,"portfolio",refs)
        p["claims"][index]["kind"]=kind
        # These entries never delegate an SKU. A proposal has no shared period;
        # don't persist misleading requested-window or null identity fields.
        for key in ("sku","nm_id"):
            p["claims"][index].pop(key,None)
        if kind=="portfolio_proposal":p["claims"][index].pop("period",None)
        if not required:p["required_claim_indices"].pop()
        return index
    def fact(key,value,unit,source,refs,target_type="portfolio",target_id=None):
        if source not in refs:return
        native_fact=EvidenceFact(metric="portfolio_campaign."+key,value=value,unit=unit,period=period,
            target_type=target_type,target_id=target_id,source_table=source,evidence_ref=refs[source],source_agent="marketing",
            metric_definition="Native DRR report; full realized revenue is not ad attribution; latest links not historical proof",
            status="confirmed" if value is not None else "unconfirmed",confidence="medium" if value is not None else "low").model_dump(mode="json")
        # Only omit redundant nullable metadata in internal policy3 catalog.
        # value=None remains explicit; the public legacy DTO restores defaults.
        p["facts"].append({k:v for k,v in native_fact.items() if v is not None or k=="value"})
    for record,expected in zip(records,read_plan(start,end),strict=True):
        value=validate(record,expected,today);op=expected["operation"]
        if value is None:
            reason=record.get("error_code",record.get("reason"))
            warnings.append("needs_data:portfolio_source_unavailable:"+op+":"+reason)
            index=add("Портфельный источник "+op+" недоступен: "+reason+". Это не нулевые показатели и не разрешение на изменение рекламы.",[])
            p["required_groups"].append([index]);p["tables"].append(_table(op+": source unavailable",[],["Данные"]));continue
        data=value.model_dump(mode="json",exclude_unset=True)["data"];warnings.extend(value.warnings)
        digest=sha256(canonical(data)).hexdigest()[:16]
        refs={e.source:f"marketing:{p['request_id']}:{e.id}:{digest}" for e in value.evidence}
        p["evidence"].extend({"domain":"marketing","claim":"portfolio_campaign_native:"+op,"source":e.source,
            "evidence_ref":refs[e.source],"period":period if op=="marketing/drr-report" else None,"freshness":"unknown"} for e in value.evidence)
        if op=="marketing/drr-report":
            totals=data["totals"]
            if totals:
                for key,unit in TOTAL_UNITS.items():
                    source=SALES if any(t in key for t in ("revenue","sales")) else STATS
                    fact("totals."+key,totals[key],unit,source,refs)
                index=add("Native DRR portfolio totals за "+period+": "+"; ".join(f"{k}={v} {TOTAL_UNITS[k]}" for k,v in totals.items())+". Store/promoted выручка — все реализации, не ad-attributed/ROI.",list(refs.values()))
                p["required_groups"].append([index]);p["tables"].append(_table("Портфель: native DRR totals; не ROI",[totals],list(totals)))
            rows=data["campaign_rows"];indices=[]
            top=sorted(range(len(rows)),key=lambda i:rows[i]["spend_rub"] if rows[i]["spend_rub"] is not None else float("-inf"),reverse=True)
            required=set(top[:2]+[i for i in top if rows[i]["ad_orders"]==0][:2])
            for i,row in enumerate(rows):
                for key,unit in (("spend_rub","RUB"),("ad_orders","orders")):
                    fact("campaign."+key,row[key],unit,STATS,refs,"campaign",str(row["advert_id"]))
                index=add(f"Native campaign {row['advert_id']} за {period}: spend={row['spend_rub']} RUB; clicks={row['clicks']}; ad_orders={row['ad_orders']}; views={row['views']}; CPC={row['cpc_rub']} RUB/click. Current nm links={row['nm_ids']}; не historical SKU attribution, причина/прибыль не установлены.",
                    [refs[STATS]] if STATS in refs else [],required=i in required)
                if i in required:indices.append(index)
            p["required_groups"].extend(indices[i:i+6] for i in range(0,len(indices),6))
            p["tables"].append(_table("Портфель: все returned native campaign rows; latest links не historical",rows,list(rows[0]) if rows else ["advert_id"]))
            sku=data["promoted_sku_rows"]
            p["tables"].append(_table("Портфель: все returned realized promoted-SKU rows, не ad attribution",sku,list(sku[0]) if sku else ["nm_id","vendor_code"]))
            if not rows:warnings.append("needs_data:portfolio_campaign_rows_missing")
        else:
            warnings.append("quality:portfolio_proposals_not_execution_or_manager_approval")
            mandatory=[]
            # One opaque anchor binds the full native matrix record, not the
            # proposal's metric/source/period. All observed source evidence
            # remains in p.evidence and the original envelope; don't duplicate
            # seven long refs in every catalog entry or invent row provenance.
            anchor_refs=[refs[value.source]] if value.source in refs else []
            for family in ("campaign_actions","query_actions","sku_actions"):
                rows=data[family];seen_modes=set()
                for row in rows:
                    required=row["action_mode"] not in seen_modes;seen_modes.add(row["action_mode"])
                    index=add(f"Read-only portfolio proposal {family}, {row['entity_id']}: mode={row['action_mode']}; blockers={canonical(row['blockers']).decode()}. Окно/row provenance неизвестны; не численный факт или approval/исполнение.",anchor_refs,required=required,kind="portfolio_proposal")
                    if required:mandatory.append(index)
                columns=list(dict.fromkeys(k for row in rows for k in row)) or ["entity_type","entity_id","action_mode"]
                p["tables"].append(_table("Read-only portfolio "+family+"; не approved actions",rows,columns))
            if not mandatory:
                mandatory.append(add("Native action proposals отсутствуют; это missing data, не approval на изменение рекламы.",anchor_refs,kind="portfolio_proposal"))
            p["required_groups"].extend(mandatory[i:i+6] for i in range(0,len(mandatory),6))
            for key in ("summary","supporting_sources","source_freshness"):
                p["tables"].append(_table("Native action "+key+"; relative reference, не сегодня",[data[key]],list(data[key])))
            f=value.data.source_freshness
            dates=[value.data.report_date,f.reference_date,
                *(getattr(f.sources,k).report_date for k in MARTS),
                *(x.report_date for x in (f.legacy_campaign,f.legacy_query) if x)]
            if any(d and d>today for d in dates):warnings.append("needs_data:portfolio_action_future_source_date")
