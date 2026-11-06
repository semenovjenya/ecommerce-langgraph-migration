"""Policy4 native subset coverage; no thresholds, formulas or action execution."""
from copy import deepcopy
from datetime import date

from commerce_agent.marketing.composite import canonical
from commerce_agent.marketing.publication import native_gaps,CHECKS
from commerce_agent.sales.contracts import CrossAgentRequest,DataGap

SUBSETS={"query":("query_waste_rows","query_scale_rows","query_position_risk_rows"),
    "demand":("growing_query_rows","high_frequency_low_order_rows","order_driving_query_rows")}
OVERFLOW="needs_data:composite_targeted_checks_budget_exceeded"


def labels(family,native,row):
    return [name for name in SUBSETS[family] if any(canonical(v)==canonical(row) for v in native[name])]


def observation_suffix(family,native,row):
    names=labels(family,native,row)
    prefix=" Native subsets: "+(", ".join(names) if names else "нет отмеченных subsets")+"."
    if family=="query":
        return prefix+" views/clicks — частотный proxy, не paid показы/клики; paid spend/CPC недоступны. " \
            "Native spend=0/CPC=null — placeholders, не наблюдаемый ноль расходов. query_scale_rows — не разрешение на масштабирование."
    if row["prior_orders_count"] is None:
        return prefix+" Предыдущее значение отсутствует: native orders_delta сохранена, но это не подтверждённый рост или спад. " \
            "growing_query_rows — классификация источника, не доказательство роста при missing prior."
    if row["orders_count"] is None:
        return prefix+" Текущее значение отсутствует: native orders_delta сохранена, но это не подтверждённый рост или спад. " \
            "growing_query_rows — классификация источника, не доказательство роста при missing current."
    return prefix+" Предыдущее значение доступно; orders_delta — готовая дельта источника. Причина изменения не установлена."


def _target(claim):
    return claim["sku"],claim["nm_id"],claim["period"]


def add_family_coverage(p,family,native,rows,indices,*,source_covered):
    material=[index for row,index in zip(rows,indices,strict=True) if labels(family,native,row)
        or family=="demand" and (row["prior_orders_count"] is None or row["orders_count"] is None)]
    required={i for group in p["required_groups"] for i in group}
    missing=[i for i in material if i not in required]
    p["required_groups"].extend(missing[i:i+6] for i in range(0,len(missing),6))
    if not source_covered:return
    candidates=list(dict.fromkeys([*indices[:1],*material]))
    existing={_target(p["claims"][i]) for i in p["required_query_check_indices"]}
    for index in candidates:
        claim=p["claims"][index];target=_target(claim)
        if (not claim["sku"] or type(claim["nm_id"]) is not int or claim["nm_id"]<=0 or not claim["period"]
            or date.fromisoformat(claim["period"].split("/")[1])>date.fromisoformat(p["today"]) or target in existing):continue
        existing.add(target)
        if len(p["required_query_check_indices"])<4:p["required_query_check_indices"].append(index)
        else:p["query_check_overflow"]=True


def query_check(domain,claim):
    if domain not in CHECKS or claim["kind"] not in {"query","demand"} or not all(_target(claim)):
        raise ValueError("Marketing populated check lacks observed SKU/window")
    caveat="Поисковая статистика не является paid атрибуцией; причина изменения не установлена."
    return CrossAgentRequest(target_agent=domain,period=claim["period"],vendor_codes=[claim["sku"]],nm_ids=[claim["nm_id"]],
        reason="Нужна проверка экономики/продаж наблюдаемого поискового SKU; не рекламный write.",
        question=f"Артикул {claim['sku']}, nm_id {claim['nm_id']}, период {claim['period']}. {caveat} {CHECKS[domain]}",
        required_checks=[caveat,CHECKS[domain]])


def required_checks(p):
    return [query_check(domain,p["claims"][i]) for i in p.get("required_query_check_indices",[]) for domain in ("sales","finance")]


def data_gaps(p):
    result=native_gaps(p)
    if p.get("query_check_overflow"):
        result.append(DataGap(metric="Покрытие адресных проверок",reason=OVERFLOW,
            impact="Не все наблюдаемые SKU вошли в лимит восьми Sales/Finance запросов; дополнительные проверки остаются открытыми."))
    return result


def complete_selection(raw,p):
    """Model priorities cannot remove required native observations/checks."""
    raw=deepcopy(raw);seen=set()
    for group in raw["findings"]:
        indices=group["claim_indices"]
        if len(indices)!=len(set(indices)) or seen&set(indices):raise ValueError("Marketing composite duplicate observations")
        seen.update(indices)
    for group in p["required_groups"]:
        missing=[i for i in group if i not in seen]
        if missing:raw["findings"].append({"claim_indices":missing});seen.update(missing)
    summary=raw["summary_claim_indices"]
    if len(summary)!=len(set(summary)):raise ValueError("Marketing composite duplicate summary")
    for group in p["required_groups"]:
        summary.extend(i for i in group if i not in summary)
    return raw
