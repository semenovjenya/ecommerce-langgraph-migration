"""Exact per-target read-only reports; no list keyword, totals or new formulas."""
from copy import deepcopy
from datetime import date,timedelta
from hashlib import sha256
import json
import re
from typing import Literal

from pydantic import Field,model_validator
from commerce_agent.adapters.contracts import DatabaseTime,MarketingSnapshot,MarketingSkuComparison,ToolResult,ToolArtifactReference
from commerce_agent.adapters.business_tools import ToolAdapterError
from commerce_agent.marketing.policy import MarketingRequest,assert_bounded_safe
from commerce_agent.marketing.evidence import _hydrate,_table
from commerce_agent.marketing.composite_snapshot import prepare_snapshot
from commerce_agent.marketing.contracts import MarketingDiagnosticResult
from commerce_agent.marketing.publication import ACTION,CAUSE,IMPACT,delegated_check,native_gaps
from commerce_agent.sales.contracts import EvidenceFact,DiagnosticFinding,DriverClass

STATS="marketing.wb_marketing_campaign_stats_daily"
SALES="sales.wb_supplier_sales"
SOURCES={STATS,SALES}
DECLARED_SOURCES=SOURCES|{"marketing.wb_marketing_campaign_nms","sales.wb_sales_funnel_products_daily"}
NATIVE_WARNINGS={"freshness:source_date_unconfirmed","semantics:campaign_metrics_equally_allocated_not_exact_sku",
    "semantics:realized_sales_not_ad_attributed_revenue","scope:comparison_periods_in_native_payload",
    "scope:returned_skus_only","provenance_only:latest_campaign_links_not_historical_attribution"}
ERRORS={"unavailable","denied","invalid_request","invalid_response","result_too_large","artifact_unavailable"}
BASE=("request","today","request_id","anchor_snapshot","anchor_source_mode","scoped_read_results")


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(",",":")).encode()


class MarketingScopedRequest(MarketingRequest):
    capability: Literal["scoped_report"]="scoped_report"
    scope: Literal["portfolio","single_sku","sku_list"]

    @model_validator(mode="after")
    def bounded_targets(self):
        if not self.targets:raise ValueError("Marketing scoped report requires targets")
        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",t) for t in self.targets):
            raise ValueError("Marketing scoped report requires literal targets")
        if self.date_from and (self.date_to-self.date_from).days>=31:
            raise ValueError("Marketing scoped report window exceeds budget")
        return self


async def collect_scoped_report(client,context,request,*,portfolio_policy_version=3,allow_debug=False):
    if type(portfolio_policy_version) is not int or portfolio_policy_version not in (2,3):
        raise ValueError("unsupported portfolio scoped policy")
    from commerce_agent.debug_privacy import development_debug_enabled
    if type(allow_debug) is not bool:raise ValueError("Marketing scoped debug flag invalid")
    debug=development_debug_enabled()
    if debug:
        if not allow_debug or portfolio_policy_version!=3:raise ValueError("Marketing scoped report debug not accepted")
        from commerce_agent.marketing.scoped_debug_source import ScopedDebugReadBoundary
        client=ScopedDebugReadBoundary(client,request)
    clock=await _hydrate(client,context,await client.database_time(context),DatabaseTime)
    snapshot=await client.marketing_report_snapshot(context)
    anchor_mode="hydrated" if isinstance(snapshot,ToolArtifactReference) else "inline"
    snapshot=await _hydrate(client,context,snapshot,MarketingSnapshot)
    prepared={"request":request.model_dump(mode="json"),"today":str(clock.data.db_current_date),"request_id":context.request_id,
        "anchor_snapshot":snapshot.model_dump(mode="json",exclude_unset=True),"anchor_source_mode":anchor_mode,"scoped_read_results":[]}
    if debug:prepared["scoped_debug_clock"]=deepcopy(client.clock_record)
    anchor=snapshot.data.report_date
    if anchor is None or anchor>clock.data.db_current_date:raise ValueError("Marketing scoped source anchor unavailable")
    end=request.date_to or anchor;start=request.date_from or end-timedelta(days=request.days-1)
    if end>anchor:raise ValueError("Marketing requested window exceeds complete source day")
    for target in request.targets:
        try:
            value=await client.marketing_sku_period_comparison(context,date_from=str(start),date_to=str(end),
                sku_list=target,product_keyword=None,period_days=(end-start).days+1,limit=100)
            mode="hydrated" if isinstance(value,ToolArtifactReference) else "inline"
            value=await _hydrate(client,context,value,MarketingSkuComparison)
            prepared["scoped_read_results"].append({"target":target,"mode":mode,"envelope":value.model_dump(mode="json",exclude_unset=True)})
        except ToolAdapterError as exc:
            if exc.code not in ERRORS:raise ValueError("unknown scoped adapter error") from None
            prepared["scoped_read_results"].append({"target":target,"mode":"failed","error_code":exc.code})
    if request.scope=="portfolio" and portfolio_policy_version==3:
        from commerce_agent.marketing.portfolio_campaigns import collect
        prepared["portfolio_campaign_reads"]=await collect(client,context,start,end)
    result=build_scoped_catalog(prepared)
    if debug:
        from commerce_agent.marketing.scoped_debug_export import assert_scoped_debug_prepared
        assert_scoped_debug_prepared(result,require_clock=True)
    return result


def base_projection(prepared):
    fields=BASE+("portfolio_campaign_reads",) if "portfolio_campaign_reads" in prepared else BASE
    if "marketing_price_feedback_version" in prepared:
        if type(prepared["marketing_price_feedback_version"]) is not int or prepared["marketing_price_feedback_version"] != 1:
            raise ValueError("Marketing Price capability version invalid")
        fields += ("marketing_price_feedback_version",)
    if "scoped_debug_clock" in prepared:fields+=("scoped_debug_clock",)
    if "scoped_report_presentation_version" in prepared:
        if type(prepared["scoped_report_presentation_version"]) is not int or prepared["scoped_report_presentation_version"]!=1:
            raise ValueError("Marketing scoped presentation version unsupported")
        fields+=("scoped_report_presentation_version",)
    return {key:deepcopy(prepared[key]) for key in fields}


def validate_target(record,target,start,end):
    if record.get("target")!=target:raise ValueError("Marketing scoped request target changed")
    if record.get("mode")=="failed":
        if set(record)!={"target","mode","error_code"} or record["error_code"] not in ERRORS:
            raise ValueError("Marketing scoped failure receipt changed")
        return None
    if set(record)!={"target","mode","envelope"} or record["mode"] not in {"inline","hydrated"}:
        raise ValueError("Marketing scoped native mode unknown")
    envelope=record["envelope"];value=ToolResult[MarketingSkuComparison].model_validate(envelope)
    if canonical(value.model_dump(mode="json",exclude_unset=True))!=canonical(envelope):raise ValueError("Marketing scoped native types changed")
    data=value.data;days=(end-start).days+1
    if (data.focus_sku_list!=[target] or data.focus_keyword is not None or data.period_days!=days
        or data.row_count!=len(data.rows) or len(data.rows)>100 or value.source!=STATS
        or not {e.source for e in value.evidence}<=SOURCES or data.metric_grain!="SKU period comparison from campaign-day stats"
        or len(data.source_tables)!=4 or set(data.source_tables)!=DECLARED_SOURCES
        or value.freshness.as_of is not None or value.freshness.status!="missing" or not NATIVE_WARNINGS<=set(value.warnings)):
        raise ValueError("Marketing scoped native scope/grain changed")
    if any(e.grain!="product_filter" or re.fullmatch(r"marketing\.sku_period_comparison:[a-f0-9]{64}",e.id) is None for e in value.evidence):
        raise ValueError("Marketing scoped evidence scope changed")
    previous_start,previous_end=start-timedelta(days=days),start-timedelta(days=1)
    for period in [value.period,*(e.period for e in value.evidence)]:
        # M2 native provenance has no observation date/window for this family.
        # Request and row periods cannot be promoted to source coverage.
        if period is not None:raise ValueError("Marketing scoped provenance window changed")
    if not data.rows:
        if value.evidence or any(d is not None for d in (data.date_from,data.date_to,data.previous_date_from,data.previous_date_to)):
            raise ValueError("Marketing empty scoped read has manufactured coverage")
        if not {"needs_data:sku_comparison_rows_missing","needs_data:realized_sales_coverage_unconfirmed"}<=set(value.warnings):
            raise ValueError("Marketing empty scoped read omitted native gaps")
        return value
    if (data.date_from,data.date_to,data.previous_date_from,data.previous_date_to)!=(start,end,previous_start,previous_end):
        raise ValueError("Marketing scoped native period changed")
    identities=set()
    for row in data.rows:
        if (not row.vendor_code or row.nm_id is None or row.nm_id<=0 or row.vendor_code!=target and str(row.nm_id)!=target
            or (row.current_date_from,row.current_date_to,row.previous_date_from,row.previous_date_to)!=(start,end,previous_start,previous_end)
            or (row.vendor_code,row.nm_id) in identities):raise ValueError("Marketing scoped native identity/window changed")
        identities.add((row.vendor_code,row.nm_id))
    if STATS not in {e.source for e in value.evidence}:raise ValueError("Marketing scoped advertising evidence missing")
    return value


METRICS=(
    ("spend_rub","Расход рекламы","RUB","spend_delta_rub","spend_delta_pct"),
    ("impressions","Показы рекламы","impressions","impressions_delta","impressions_delta_pct"),
    ("clicks","Клики рекламы","clicks","clicks_delta","clicks_delta_pct"),
    ("ad_orders","Заказы рекламной статистики, не все продажи","orders","ad_orders_delta","ad_orders_delta_pct"),
    ("sales_units","Все реализованные продажи, не ad attribution","units",None,None),
    ("sales_revenue_rub","Вся реализованная выручка, не ad attribution","RUB",None,None),
    ("estimated_drr_pct","Оценочный ДРР ко всей выручке, не ROI/прибыль","%",None,None),
    ("ctr_pct","Native CTR SKU","%",None,None),
    ("cpc_rub","Native CPC SKU, не campaign-average CPC","RUB/click",None,None),
    ("campaigns","Кампании по текущим связям, не historical attribution","campaigns",None,None),
)


def build_scoped_catalog(native):
    assert_bounded_safe(native,max_bytes=262144)
    p=base_projection(native);request=MarketingScopedRequest.model_validate(p["request"])
    if canonical(request.model_dump(mode="json"))!=canonical(p["request"]):raise ValueError("Marketing scoped request types changed")
    today=date.fromisoformat(p["today"])
    from types import SimpleNamespace
    snapshot,evidence,warnings=prepare_snapshot(p["anchor_snapshot"],SimpleNamespace(product_keyword=None),today,p["request_id"],p["anchor_source_mode"])
    anchor=date.fromisoformat(snapshot["data"]["report_date"]) if snapshot["data"]["report_date"] else None
    if anchor is None or anchor>today:raise ValueError("Marketing scoped source anchor unavailable")
    end=request.date_to or anchor;start=request.date_from or end-timedelta(days=request.days-1)
    if end>anchor:raise ValueError("Marketing scoped requested window exceeds complete source day")
    previous_start,previous_end=start-timedelta(days=(end-start).days+1),start-timedelta(days=1)
    current,previous=f"{start}/{end}",f"{previous_start}/{previous_end}"
    if len(p["scoped_read_results"])!=len(request.targets):raise ValueError("Marketing scoped target coverage changed")
    portfolio=request.scope=="portfolio"
    if "portfolio_campaign_reads" in p and not portfolio:raise ValueError("campaign context requires portfolio scope")
    policy=3 if "portfolio_campaign_reads" in p else 2 if portfolio else 1
    p.update(capability="scoped_report",scoped_marketing_policy_version=policy,period=current,previous_period=previous,
        report_date=str(anchor),claims=[],facts=[],tables=[],evidence=evidence,required_groups=[],required_claim_indices=[],checks=[])
    warnings.extend(["scope:scoped_reads_independent_no_combined_total",
        "scope:portfolio_native_latest_windows_independent_from_requested_sku_window" if portfolio else "scope:portfolio_snapshot_reference_not_requested_sku_total",
        "semantics:sku_allocation_not_causal_attribution","freshness:source_coverage_not_current_date"])
    scopes=set()
    def claim(text,target,refs,sku=None,nm_id=None):
        p["claims"].append({"kind":"scoped_target","text":text,"requested_target":target,"sku":sku,"nm_id":nm_id,"period":current,"evidence_refs":refs})
        p["required_claim_indices"].append(len(p["claims"])-1)
        return len(p["claims"])-1
    if portfolio:
        # Reuse the already-validated global snapshot. It remains an explicit
        # portfolio scope; never sum requested-SKU rows to invent a total.
        from commerce_agent.marketing.composite_snapshot import append_snapshot_catalog
        from commerce_agent.marketing.composite_analysis import _table as native_snapshot_table
        snapshot_catalog={"snapshot_context":snapshot,"evidence":p["evidence"],"warnings":warnings,
            "claims":p["claims"],"facts":p["facts"],"tables":p["tables"],"required_groups":p["required_groups"]}
        def portfolio_claim(kind,text):
            index=claim(text,"portfolio",[])
            p["claims"][index]["kind"]="scoped_portfolio"
            return index
        append_snapshot_catalog(snapshot_catalog,portfolio_claim,
            lambda title,rows,columns:native_snapshot_table("Портфель: "+title,rows,columns))
    for target,record in zip(request.targets,p["scoped_read_results"],strict=True):
        value=validate_target(record,target,start,end);group=[]
        if value is None or not value.data.rows:
            reason="native rows отсутствуют" if value is not None else "read unavailable: "+record["error_code"]
            warning="needs_data:scoped_target_unavailable:"+target;warnings.append(warning)
            group.append(claim(f"Запрошенный SKU/nm_id {target}: {reason}; это не нулевые расходы/продажи и не доказательство отсутствия рекламы.",target,[]))
            p["tables"].append(_table(f"Запрошенный target {target}: нет подтверждённых строк, окно {current}",[]))
            if value is not None:warnings.extend(value.warnings)
        else:
            warnings.extend(value.warnings)
            digest=sha256(canonical(record["envelope"]["data"])).hexdigest()[:16]
            refs={e.source:f"marketing:{p['request_id']}:{e.id}:{digest}" for e in value.evidence}
            p["evidence"].extend({"domain":"marketing","claim":"scoped_sku:"+target,"source":e.source,
                "evidence_ref":refs[e.source],"period":None,"freshness":"unknown"} for e in value.evidence)
            rows=[{key:v for key,v in row.model_dump(mode="json").items() if key!="subject_name"} for row in value.data.rows]
            p["tables"].append(_table(f"Exact target {target}: {current} vs {previous}; native allocation, не causal attribution",rows))
            if len(rows)==100:warnings.append("coverage:scoped_row_limit_reached:"+target)
            for row in value.data.rows:
                row_claims=[]
                for metric,label,unit,delta,pct in METRICS:
                    sources={SALES} if metric.startswith("sales_") else {STATS,SALES} if metric=="estimated_drr_pct" else {STATS}
                    if not sources<=set(refs):
                        warnings.append("needs_data:scoped_metric_source_unconfirmed:"+target+":"+metric);continue
                    primary=SALES if metric.startswith("sales_") else STATS
                    a,b=getattr(row,"current_"+metric),getattr(row,"previous_"+metric)
                    amounts=[("current_"+metric,a,current,unit),("previous_"+metric,b,previous,unit)]
                    if delta:amounts.append((delta,getattr(row,delta),current+" vs "+previous,unit))
                    if pct:amounts.append((pct,getattr(row,pct),current+" vs "+previous,"%"))
                    for key,v,window,units in amounts:
                        p["facts"].append(EvidenceFact(metric="scoped."+key,value=v,unit=units,period=window,sku=row.vendor_code,
                            target_type="sku",target_id=row.vendor_code,source_table=" + ".join(sorted(sources)),evidence_ref=refs[primary],
                            source_agent="marketing",metric_definition=f"{label}; target {target}; {row.attribution_quality}; не causal attribution",
                            status="confirmed" if v is not None else "unconfirmed",confidence="medium" if v is not None else "low").model_dump(mode="json"))
                    missing=lambda v:"нет данных" if v is None else str(v)
                    text=f"{label}, SKU {row.vendor_code}, nm_id {row.nm_id} (запрос {target}, {row.attribution_quality}): {missing(a)} {unit} за {current} против {missing(b)} {unit} за {previous}."
                    if delta:text+=f" Native изменение {missing(getattr(row,delta))} {unit}, {missing(getattr(row,pct))} %; не пересчитано моделью."
                    row_claims.append(claim(text,target,[refs[s] for s in sorted(sources)],row.vendor_code,row.nm_id))
                group.extend(row_claims)
                identity=(row.vendor_code,row.nm_id,current)
                if row_claims and identity not in scopes:
                    scopes.add(identity)
                    if len(p["checks"])<=6:
                        c={**p["claims"][row_claims[0]],"kind":"sku"}
                        p["checks"].extend(delegated_check(domain,c).model_dump(mode="json") for domain in ("sales","finance"))
                    else:warnings.append("needs_data:scoped_neighbor_check_budget:"+target)
        if not group:group=[claim(f"Запрошенный target {target}: строки есть, но источники метрик не подтверждены; численный вывод недоступен.",target,[])]
        p["required_groups"].extend(group[i:i+6] for i in range(0,len(group),6))
    if policy==3:
        from commerce_agent.marketing.portfolio_campaigns import append_catalog
        append_catalog(p,p["portfolio_campaign_reads"],start,end,today,claim,warnings)
    p["marketing_report_snapshot"]=deepcopy(snapshot["data"]["report_table"])
    p["warnings"]=sorted(set(warnings))
    if "historical_sources" in native or "historical_sources_version" in native:
        for key in ("historical_sources", "historical_sources_version"):
            p[key]=deepcopy(native[key])
        for key in ("coverage_manifest", "coverage_manifest_version"):
            if key in native:
                p[key]=deepcopy(native[key])
        from commerce_agent.marketing.historical_sources import validate as validate_history
        validate_history(p)  # Bind to regenerated native claims and exact periods.
    assert_bounded_safe(p,max_bytes=262144)
    return p


def public_facts(prepared):
    if prepared["scoped_marketing_policy_version"]==3:
        return [EvidenceFact.model_validate(f).model_dump(mode="json") for f in prepared["facts"]]
    return prepared["facts"]


def assemble_scoped_result(narrative,prepared):
    p=build_scoped_catalog(prepared)
    raw=narrative.model_dump(mode="json") if hasattr(narrative,"model_dump") else narrative
    from commerce_agent.marketing.scoped_model import narrative_schema
    raw=narrative_schema(p).model_validate(raw,strict=True).model_dump(mode="json")
    summary=list(dict.fromkeys([*raw["summary_claim_indices"],*p["required_claim_indices"]]))
    groups=deepcopy(raw["findings"])
    seen=set()
    for group in groups:
        indices=group["claim_indices"]
        if len(set(indices))!=len(indices) or set(indices)&seen:raise ValueError("Marketing scoped duplicate selections")
        seen.update(indices)
    # Coverage is a deterministic publication policy, not a preference of the
    # LLM. Preserve its priority selections, then restore every missing native
    # observation/group (including absent targets); no new quantities are made.
    for required in p["required_groups"]:
        missing=[i for i in required if i not in seen]
        if missing:groups.append({"claim_indices":missing});seen.update(missing)
    findings=[]
    for group in groups:
        claims=[p["claims"][i] for i in group["claim_indices"]]
        findings.append(DiagnosticFinding(factor="Проверенные scoped наблюдения рекламы",observation="\n".join(c["text"] for c in claims),
            impact=IMPACT,cause=CAUSE,action=ACTION,evidence=list(dict.fromkeys(ref for c in claims for ref in c["evidence_refs"])),
            confidence="medium",driver_class=DriverClass.NOT_A_DRIVER))
    return MarketingDiagnosticResult(agent_name="Marketing Agent",summary="\n".join(p["claims"][i]["text"] for i in summary),
        facts=deepcopy(public_facts(p)),tables=deepcopy(p["tables"]),marketing_report_snapshot=deepcopy(p["marketing_report_snapshot"]),
        findings=findings,recommendations=[ACTION],data_gaps=native_gaps(p),cross_agent_requests=deepcopy(p["checks"]))


def validate_scoped_publication(candidate,prepared):
    p=build_scoped_catalog(prepared)
    if canonical(p)!=canonical(prepared):raise ValueError("Marketing scoped prepared data changed")
    raw=candidate.model_dump(mode="json") if hasattr(candidate,"model_dump") else candidate
    assert_bounded_safe(raw,max_bytes=524288)
    for key in ("facts","tables","marketing_report_snapshot","cross_agent_requests"):
        expected=p["checks"] if key=="cross_agent_requests" else public_facts(p) if key=="facts" else p[key]
        if canonical(raw[key])!=canonical(expected):raise ValueError("Marketing scoped authoritative data changed")
    result=MarketingDiagnosticResult.model_validate(raw)
    if result.agent_name!="Marketing Agent" or result.recommendations!=[ACTION] or result.data_gaps!=native_gaps(p):
        raise ValueError("Marketing scoped policy fields changed")
    proof={c["text"]:c for c in p["claims"]};observed=set()
    for finding in result.findings:
        lines=finding.observation.split("\n")
        if (len(set(lines))!=len(lines) or set(lines)&observed or any(line not in proof for line in lines) or finding.factor!="Проверенные scoped наблюдения рекламы"
            or finding.impact!=IMPACT or finding.cause!=CAUSE or finding.action!=ACTION or finding.confidence!="medium"
            or finding.driver_class!=DriverClass.NOT_A_DRIVER
            or set(finding.evidence)!={ref for line in lines for ref in proof[line]["evidence_refs"]}):
            raise ValueError("Marketing scoped finding is not native-bound")
        observed.update(lines)
    summary=result.summary.split("\n")
    required={p["claims"][i]["text"] for i in p["required_claim_indices"]}
    if len(set(summary))!=len(summary) or any(line not in proof for line in summary) or not required<=observed or not required<=set(summary):
        raise ValueError("Marketing scoped target observations omitted")
    if canonical(result.model_dump(mode="json"))!=canonical(raw):raise ValueError("Marketing scoped publication normalization changed")
    return result


def render_scoped_report(result,prepared):
    result=validate_scoped_publication(result,prepared)
    if prepared.get("scoped_report_presentation_version")==1:
        from commerce_agent.marketing.brief import render_brief_scoped_report
        return render_brief_scoped_report(result,prepared)
    portfolio=prepared["request"]["scope"]=="portfolio"
    parts=["# Marketing — портфель и выбранные SKU" if portfolio else "# Marketing — отчёт по запрошенным SKU",f"Окно {prepared['period']} против {prepared['previous_period']}; последний общий день источников {prepared['report_date']}.",
        "Каждый target прочитан отдельно. Срезы не складываются; missing не ноль. Native allocation не доказывает causal lift или прибыль.",
        "## Главное",result.summary,"## Что проверить"]
    parts.extend("- "+c.target_agent.value+": "+c.question for c in result.cross_agent_requests)
    parts.extend(["## Ограничения",*['- '+g.reason for g in result.data_gaps],"## Проверенные native строки"])
    def cell(v):return ("нет данных" if v is None else str(v)).replace("\\","\\\\").replace("|","\\|").replace("\n"," ")
    for table in [*result.tables,result.marketing_report_snapshot]:
        parts.extend(["### "+table.title,"| "+" | ".join(cell(v) for v in table.columns)+" |\n| "+" | ".join("---" for _ in table.columns)+" |\n"+
            "\n".join("| "+" | ".join(cell(v) for v in row)+" |" for row in table.rows)])
    parts.append("Портфельные latest day/week/month показатели — отдельный scope. Они не являются суммой выбранных SKU; их окна не подменяются запрошенным периодом SKU."
        if portfolio else "Портфельный latest snapshot выше — справочный контекст, не итог выбранных SKU и не requested-period evidence.")
    return "\n\n".join(parts)
