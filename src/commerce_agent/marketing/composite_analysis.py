"""Native composite catalog/report: facts are not proposal text or paid-query math."""
from copy import deepcopy
from datetime import date
from hashlib import sha256
import html
import json
import re

from commerce_agent.marketing.composite import canonical,MarketingCompositeDiagnosticRequest,prepare_native
from commerce_agent.marketing.contracts import MarketingDiagnosticResult
from commerce_agent.marketing.policy import assert_bounded_safe
from commerce_agent.marketing.publication import CAUSE,IMPACT,ACTION,delegated_check,native_gaps
from commerce_agent.sales.contracts import EvidenceFact,ReportTable,DiagnosticFinding,DriverClass,CrossAgentRequest

BASE=("request","capability","composite_policy_version","today","source_mode","native_envelope","source_windows","evidence","warnings")
CAMP="marketing.wb_marketing_campaign_stats_daily"
QUERY="marketing.wb_search_query_orders_text_daily"
DETAIL="marketing.wb_search_query_details"
OVERVIEW="marketing.wb_search_query_report_overview"


def base_projection(prepared):
    version=prepared["composite_policy_version"]
    if type(version) is not int or version not in (1,2,3,4):raise ValueError("Marketing composite policy version unsupported")
    fields=BASE+("snapshot_context","snapshot_context_mode") if version>=3 else BASE
    return {key:prepared[key] for key in fields}


def _cell(value):
    if isinstance(value,(list,dict,bool)):return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)
    return value


def _table(title,rows,columns):
    return ReportTable(title=title,columns=columns,rows=[[_cell(row.get(k)) for k in columns] for row in rows]).model_dump(mode="json")


def derive_composite_catalog(base):
    """Ephemeral deterministic cache, never a checkpoint representation.

    Its separate memory bound does not change persisted prepared (256KiB),
    provider input (256KiB), or public result (512KiB) limits.
    """
    request=MarketingCompositeDiagnosticRequest.model_validate(base["request"])
    if canonical(request.model_dump(mode="json"))!=canonical(base["request"]):raise ValueError("Marketing composite request types changed")
    evidence=base["evidence"]
    # Provenance identity is opaque and non-authorizing; derive only the same
    # local projection, not actor scope or service credentials from it.
    if not evidence:raise ValueError("Marketing composite source evidence unavailable")
    ref=evidence[0]["evidence_ref"].split(":")
    if re.fullmatch(r"marketing:[^:\r\n]{1,128}:marketing\.deep_diagnostic:[0-9a-f]{64}:[0-9a-f]{16}",evidence[0]["evidence_ref"]) is None:
        raise ValueError("Marketing composite provenance format changed")
    expected=prepare_native(base["native_envelope"],request,date.fromisoformat(base["today"]),ref[1],base["source_mode"],policy_version=base["composite_policy_version"],
        snapshot_envelope=base.get("snapshot_context"),snapshot_mode=base.get("snapshot_context_mode"))
    if canonical(base_projection(base))!=canonical(expected):raise ValueError("Marketing composite native projection changed")
    p=deepcopy(expected);p["claims"]=[];p["facts"]=[];p["tables"]=[];p["required_groups"]=[];p["eligible_check_indices"]=[]
    if p["composite_policy_version"]>=4:p.update(required_query_check_indices=[],query_check_overflow=False)
    data=p["native_envelope"]["data"];w=p["source_windows"]
    refs={e["source"]:e["evidence_ref"] for e in evidence if ":marketing.deep_diagnostic:" in e["evidence_ref"]}
    def claim(kind,text,source=None,sku=None,nm_id=None,period=None):
        if source is not None and source not in refs:
            text=f"Native {kind}: source coverage {source} не подтверждена; значения в native таблице не доказательство наблюдаемого нуля или эффекта."
        index=len(p["claims"])
        p["claims"].append({"kind":kind,"text":text,"evidence_refs":[refs[source]] if source in refs else list(refs.values()),
            "sku":sku,"nm_id":nm_id,"period":period})
        return index
    def fact(family,field,value,unit,source,period,*,sku=None,nm_id=None,target_id=None,target_type=None,confirmed=True,snapshot=None,definition=None):
        if source not in refs:return
        # No proposal strings are parsed back into native metric facts.
        valid=confirmed and value is not None and period is not None and "needs_data:composite_future_source_date" not in p["warnings"]
        p["facts"].append(EvidenceFact(metric=family+"."+field,value=value,unit=unit,period=period or "неизвестен",
            sku=sku,target_type=target_type or ("sku" if sku else "campaign" if family.startswith("campaign") else "source"),
            target_id=str(target_id or nm_id or source),source_table=source,evidence_ref=refs[source],source_agent="marketing",
            source_snapshot_at=snapshot,metric_definition=definition,
            status="confirmed" if valid else "unconfirmed",confidence="medium" if valid else "low").model_dump(mode="json"))
    scope=claim("scope",f"Latest native diagnostic: фильтр {request.product_keyword or 'портфель'}, days={request.days}, limit={request.limit}. "
        "Окна источников независимы; keyword не доказывает exact SKU/историческую связь, предложения не являются решениями.")
    p["required_groups"].append([scope])
    c=data["campaign_efficiency"]
    values=["total_spend","total_views","total_clicks","total_orders","campaign_count"]
    for key,unit in zip(values,("RUB","views","clicks","orders","campaigns")):
        fact("campaign_totals",key,c["totals"][key],unit,CAMP,w["campaign_totals"])
    text="; ".join(f"{key}={c['totals'][key]}" for key in values)
    index=claim("campaign",f"Native campaign totals: {text}; окно {w['campaign_totals'] or 'неизвестно'}. Keyword candidates, не атрибуция SKU.",CAMP)
    p["required_groups"].append([index])
    p["tables"].append(_table("Campaign totals — независимое окно",[c["totals"]],["report_date",*values]))
    for row,window in zip(c["campaign_rows"],w["campaign_rows"]):
        for key,unit in (("sum_spend","RUB"),("views","views"),("clicks","clicks"),("orders","orders"),("ctr","native_pct"),("cpc","native_scale"),("cr","native_pct")):
            fact("campaign",key,row[key],unit,CAMP,window["period"],target_id=row["advert_id"])
    p["tables"].append(_table("Campaign keyword candidates",c["campaign_rows"],list(c["campaign_rows"][0]) if c["campaign_rows"] else ["advert_id","report_date","sum_spend"]))
    q=data["query_performance"];s=data["search_demand"]
    for family,native,rows_key,period,fields in (
        ("query",q,"top_query_rows",w["queries"],(("views","search_proxy"),("clicks","search_proxy"),("orders","orders"),("spend","unavailable_paid_proxy"),("cpc","unavailable_paid_proxy"),("avg_pos","position"))),
        ("demand",s,"search_text_rows",w["demand_current"],(("frequency_sum","search_frequency"),("avg_position","position"),("orders_count","orders"),("prior_orders_count","orders"),("orders_delta","orders")))):
        indices=[];lineages=set()
        observed_rows=native[rows_key] if p["composite_policy_version"]<4 or QUERY in refs else []
        for row in observed_rows:
            query_key="norm_query" if family=="query" else "query_text"
            lineage={"family":family,"sku":row["vendor_code"],"nm_id":row["nm_id"],query_key:row[query_key],"report_date":row["report_date"]}
            if p["composite_policy_version"]>=4:
                signature=canonical(lineage)
                if signature in lineages:raise ValueError("Marketing composite duplicate query identity")
                lineages.add(signature)
            policy2=p["composite_policy_version"]>=2
            query_definition=("Native per-query row, not SKU total: "+canonical(lineage).decode()) if policy2 else None
            query_target="query:"+sha256(canonical(lineage)).hexdigest() if policy2 else None
            for key,unit in fields:
                fact_period=(w["demand_previous"] if key=="prior_orders_count" else
                    f"{period} vs {w['demand_previous']}" if key=="orders_delta" and period and w["demand_previous"] else period)
                fact(family,key,row[key],unit,QUERY,fact_period,sku=row["vendor_code"],nm_id=row["nm_id"],
                    target_id=query_target,target_type="query" if policy2 else None,definition=query_definition,
                    confirmed=key not in {"spend","cpc"} and (key!="orders_delta" or row["prior_orders_count"] is not None
                        and (p["composite_policy_version"]<4 or row["orders_count"] is not None))
                        and (not policy2 or row[query_key] is not None))
            # Null prior is never stated as observed zero or proved growth.
            numeric="; ".join(f"{key}={row[key]}" for key,_ in fields)
            dimension=f", {query_key}={json.dumps(row[query_key],ensure_ascii=False)} (per-query, не SKU total)" if policy2 else ""
            text=(f"Native {family} row, SKU {row['vendor_code']}, nm_id {row['nm_id']}{dimension}: {numeric}; "
                f"current {period or 'неизвестно'}, previous {w['demand_previous'] if family=='demand' else 'не применимо'}. "
                "Search proxy не paid efficiency; missing prior не доказанный ноль/рост.")
            if p["composite_policy_version"]>=4:
                from commerce_agent.marketing.composite_quality import observation_suffix
                text+=observation_suffix(family,native,row)
            indices.append(claim(family,text,QUERY,row["vendor_code"],row["nm_id"],period))
        if not indices:
            indices=[claim(family,f"Native {family}: возвращённые строки отсутствуют. Это missing data, не нулевой спрос/paid spend.",
                QUERY if p["composite_policy_version"]>=4 and QUERY not in refs else None)]
        p["required_groups"].append(indices[:1])
        if p["composite_policy_version"]>=4:
            from commerce_agent.marketing.composite_quality import add_family_coverage
            add_family_coverage(p,family,native,observed_rows,indices if observed_rows else [],source_covered=QUERY in refs)
        p["tables"].append(_table("Search "+family+" — native, не paid performance",native[rows_key],list(native[rows_key][0]) if native[rows_key] else ["nm_id","vendor_code","report_date"]))
    v=data["visibility"];visibility_indices=[]
    for row in v["visibility_rows"]:
        day=row["snapshot_date"];policy2=p["composite_policy_version"]>=2
        period=None if policy2 else f"{day}/{day}" if day else None
        for key,unit in (("orders_current","orders"),("open_card_current","opens"),("visibility_current","native_scale"),("avg_position_current","position"),("add_to_cart_current","native_scale"),("cart_to_order_current","native_scale")):
            fact("visibility",key,row[key],unit,DETAIL,period,sku=row["vendor_code"],nm_id=row["nm_id"],
                snapshot=day if policy2 else None,
                definition="Native current-window metric; source_snapshot_at is window-end marker, not observation timestamp; aggregation window unknown." if policy2 else None)
        index=claim("sku",f"Native visibility SKU {row['vendor_code']}, nm_id {row['nm_id']}, snapshot {day or 'неизвестен'}: "
            f"orders_current={row['orders_current']}; open_card_current={row['open_card_current']}; visibility_current={row['visibility_current']}; "
            f"avg_position_current={row['avg_position_current']}. Snapshot/window-end, не общий рекламный период или эффект."
            +(" Начало окна неизвестно; значения не являются однодневными метриками." if policy2 else ""),DETAIL,row["vendor_code"],row["nm_id"],period)
        if policy2:p["claims"][index]["window_end_marker"]=day
        visibility_indices.append(index)
        if (row["vendor_code"] and row["nm_id"] and day and DETAIL in refs
            and (not policy2 or day<=p["today"])
            and (p["composite_policy_version"]<4 or type(row["nm_id"]) is int and row["nm_id"]>0)):
            p["eligible_check_indices"].append(index)
    if not visibility_indices:visibility_indices=[claim("visibility","Native visibility: SKU rows отсутствуют; портфельный overview не заменяет SKU evidence.")]
    p["required_groups"].append(visibility_indices[:1])
    p["tables"].append(_table("Visibility SKU snapshot",v["visibility_rows"],list(v["visibility_rows"][0]) if v["visibility_rows"] else ["nm_id","vendor_code","snapshot_date"]))
    if v["overview"]:p["tables"].append(_table("Visibility overview — портфель, не SKU",[v["overview"]],list(v["overview"])))
    m=data["action_matrix"];proposal_indices=[]
    for family in ("campaign_actions","query_actions","sku_actions"):
        rows=m[family]
        for row in rows:
            proposal_indices.append(claim("proposal",f"Read-only native proposal {row['entity_type']} {row['entity_id']}: mode={row['action_mode']}; "
                f"marketing={row['marketing_source_status']}; demand={row['demand_source_status']}; blockers={json.dumps(row['blockers'],ensure_ascii=False)}. "
                "Не Manager Decision/approval; не доказательство метрик из reason/evidence и не команда исполнения."))
        columns=list(dict.fromkeys(k for row in rows for k in row)) or ["entity_type","entity_id","action_mode","blockers"]
        p["tables"].append(_table("Read-only "+family,rows,columns))
    if not proposal_indices:proposal_indices=[claim("proposal","Native action proposals отсутствуют; это не разрешение на изменение рекламы.")]
    # Bounded representative of each returned native action family. The full
    # proposal matrices remain visible regardless of model choice.
    firsts=[];offset=0
    for family in ("campaign_actions","query_actions","sku_actions"):
        if m[family]:firsts.append(proposal_indices[offset])
        offset+=len(m[family])
    p["required_groups"].append(firsts or proposal_indices[:1])
    p["tables"].append(_table("Native action summary — counts before output limit",[m["summary"]],list(m["summary"])))
    p["tables"].append(_table("Native diagnostic summary — returned family counts",[data["summary"]],list(data["summary"])))
    p["tables"].append(_table("Native diagnostic gaps — not all capability gaps",data["data_gaps"],["metric","reason","impact"]))
    p["tables"].append(_table("Native supporting sources and fallbacks",[m["supporting_sources"]],list(m["supporting_sources"])))
    diagnostics={k:value for k,value in m["source_freshness"].items() if k!="sources"}
    p["tables"].append(_table("Native freshness diagnostics including legacy fallback",[diagnostics],list(diagnostics)))
    freshness=[{"source":name,**row} for name,row in m["source_freshness"]["sources"].items()]
    p["tables"].append(_table("Source freshness relative to campaign reference, не сегодня",freshness,list(freshness[0])))
    if p["composite_policy_version"]>=3:
        from commerce_agent.marketing.composite_snapshot import append_snapshot_catalog
        append_snapshot_catalog(p,claim,_table)
    assert_bounded_safe(p,max_bytes=2*1024*1024)
    return p


def build_composite_catalog(base):
    """Historical full-cache prepared format; keep its original size gate."""
    p=derive_composite_catalog(base)
    assert_bounded_safe(p,max_bytes=262144)
    return p


def composite_delegated_check(domain,claim,prepared):
    if prepared["composite_policy_version"]>=4 and claim["kind"] in {"query","demand"}:
        from commerce_agent.marketing.composite_quality import query_check
        return query_check(domain,claim)
    if prepared["composite_policy_version"]==1:return delegated_check(domain,claim)
    from commerce_agent.marketing.publication import CHECKS
    if domain not in CHECKS or claim["kind"]!="sku" or not claim["sku"] or not claim.get("window_end_marker"):
        raise ValueError("Marketing composite delegation lacks native SKU/window-end marker")
    scope="Восстановить начало и длительность native visibility окна; маркер конца не означает однодневный период."
    return CrossAgentRequest(target_agent=domain,period=None,vendor_codes=[claim["sku"]],nm_ids=[claim["nm_id"]],
        reason="Точный период visibility неизвестен; перед сопоставлением нужны границы окна и проверка соседнего домена.",
        question=f"Артикул {claim['sku']}, nm_id {claim['nm_id']}, маркер конца окна {claim['window_end_marker']}. {scope} После восстановления границ: {CHECKS[domain]}",
        required_checks=[scope,CHECKS[domain]])


def assemble_composite_result(selection,p):
    from commerce_agent.marketing.composite_storage import load_composite_catalog
    p=load_composite_catalog(p)
    raw=selection.model_dump(mode="json") if hasattr(selection,"model_dump") else selection
    if p["composite_policy_version"]>=4:
        from commerce_agent.marketing.composite_quality import complete_selection
        raw=complete_selection(raw,p)
    catalog=p["claims"]
    def selected(indices):
        if not indices or len(set(indices))!=len(indices) or any(type(i) is not int or not 0<=i<len(catalog) for i in indices):raise ValueError("Marketing composite selection invalid")
        return [catalog[i] for i in indices]
    findings=[]
    for group in raw["findings"]:
        chosen=selected(group["claim_indices"])
        findings.append(DiagnosticFinding(factor="Native observations and read-only proposals",observation="\n".join(c["text"] for c in chosen),
            cause=CAUSE,impact=IMPACT,action=ACTION,evidence=list(dict.fromkeys(r for c in chosen for r in c["evidence_refs"])),confidence="low",driver_class=DriverClass.NOT_A_DRIVER))
    from commerce_agent.marketing.composite_quality import required_checks,data_gaps
    checks=required_checks(p) if p["composite_policy_version"]>=4 else []
    for item in raw["cross_agent_requests"]:
        if item["claim_index"] not in p["eligible_check_indices"]:raise ValueError("Marketing composite delegation lacks observed SKU/day")
        check=composite_delegated_check(item["domain"],selected([item["claim_index"]])[0],p)
        if p["composite_policy_version"]<4:checks.append(check)
        elif check not in checks:
            if len(checks)>=8:raise ValueError("Marketing composite check budget exceeded")
            checks.append(check)
    if p["composite_policy_version"]>=4 and not checks and p["eligible_check_indices"]:
        checks.append(composite_delegated_check("sales",p["claims"][p["eligible_check_indices"][0]],p))
    return MarketingDiagnosticResult(agent_name="Marketing Agent",summary="\n".join(c["text"] for c in selected(raw["summary_claim_indices"])),
        facts=deepcopy(p["facts"]),tables=deepcopy(p["tables"]),findings=findings,recommendations=[ACTION],data_gaps=data_gaps(p),cross_agent_requests=checks)


def validate_composite_publication(candidate,prepared):
    from commerce_agent.marketing.composite_storage import load_composite_catalog
    prepared=load_composite_catalog(prepared)
    raw=candidate.model_dump(mode="json") if hasattr(candidate,"model_dump") else candidate
    assert_bounded_safe(raw,max_bytes=524288)
    if canonical(raw.get("facts"))!=canonical(prepared["facts"]) or canonical(raw.get("tables"))!=canonical(prepared["tables"]):
        raise ValueError("Marketing composite raw facts/tables changed")
    result=MarketingDiagnosticResult.model_validate(raw);body=result.model_dump(mode="json")
    from commerce_agent.marketing.composite_quality import required_checks,data_gaps
    if (canonical(body["facts"])!=canonical(prepared["facts"]) or canonical(body["tables"])!=canonical(prepared["tables"])
        or result.agent_name!="Marketing Agent" or result.marketing_report_snapshot is not None or result.recommendations!=[ACTION]
        or result.data_gaps!=data_gaps(prepared)):raise ValueError("Marketing composite authoritative result changed")
    proof={c["text"]:c for c in prepared["claims"]};observed=set()
    for f in result.findings:
        lines=f.observation.split("\n")
        if prepared["composite_policy_version"]>=4 and (len(lines)!=len(set(lines)) or observed&set(lines)):
            raise ValueError("Marketing composite duplicate observations")
        if (any(line not in proof for line in lines) or f.factor!="Native observations and read-only proposals"
            or f.cause!=CAUSE or f.action!=ACTION or f.impact!=IMPACT or f.driver_class!=DriverClass.NOT_A_DRIVER or f.confidence!="low"
            or f.evidence!=list(dict.fromkeys(r for line in lines for r in proof[line]["evidence_refs"]))):raise ValueError("Marketing composite finding is not closed native proof")
        observed.update(lines)
    required={prepared["claims"][i]["text"] for group in prepared["required_groups"] for i in group}
    summary=set(result.summary.split("\n"))
    if not required<=observed or not required<=summary or not summary<=observed:raise ValueError("Marketing composite required family coverage missing")
    eligible=[composite_delegated_check(domain,prepared["claims"][i],prepared) for i in prepared["eligible_check_indices"] for domain in ("sales","finance","price","supply")]
    if prepared["composite_policy_version"]>=4:
        mandatory=required_checks(prepared);eligible.extend(mandatory)
        if (any(c not in result.cross_agent_requests for c in mandatory) or len(result.cross_agent_requests)>8
            or len({canonical(c.model_dump(mode="json")) for c in result.cross_agent_requests})!=len(result.cross_agent_requests)
            or len(result.summary.split("\n"))!=len(summary)):
            raise ValueError("Marketing composite mandatory checks/summary changed")
    if any(check not in eligible for check in result.cross_agent_requests):raise ValueError("Marketing composite scoped check changed")
    if prepared["eligible_check_indices"] and not result.cross_agent_requests:raise ValueError("Marketing composite needs observed scoped check")
    return result


def render_composite_report(result,prepared):
    result=validate_composite_publication(result,prepared)
    def cell(v):
        text=html.escape("нет данных" if v is None else str(v),quote=False).replace("\\","\\\\").replace("\r"," ").replace("\n"," ")
        for char in "|`*_[]!":text=text.replace(char,"\\"+char)
        return text
    lines=["# Marketing — расширенная native диагностика",""]
    if prepared["composite_policy_version"]>=2:
        lines.extend([f"Дата запуска по БД: {cell(prepared['today'])}. Даты источников ниже независимы.",""])
    lines.extend(["## Главное","",*(cell(line) for line in result.summary.splitlines()),"",
        "Предложения ниже read-only: ничего не исполнено. Freshness относительно campaign reference не означает свежесть сегодня.","",
        "## Что проверить","",*(f"- {c.target_agent.value}: {cell(c.question)}" for c in result.cross_agent_requests),"","## Ограничения","",
        *("- "+cell(g.reason) for g in result.data_gaps),"","## Native таблицы",""])
    for table in result.tables:
        lines.extend(["### "+cell(table.title),"","| "+" | ".join(cell(c) for c in table.columns)+" |","| "+" | ".join("---" for _ in table.columns)+" |"])
        lines.extend("| "+" | ".join(cell(c) for c in row)+" |" for row in table.rows);lines.append("")
    return "\n".join(lines)
