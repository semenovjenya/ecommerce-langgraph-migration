"""Exact diagnostic debug contract; never applied to authoritative state.

Arbitrary campaign/search/brand/subject labels can contain personal prose.
Hide those labels, not their native metrics or source dates. This is not a
semantic name detector for new schemas or arbitrary free text.
"""
from copy import deepcopy
from datetime import date
import re

CONTRACT = "marketing_diagnostic_prepared_v1"
_QUESTION = re.compile(
    r"Диагностика рекламы SKU ([1-9][0-9]{4}) за (\d{4}-\d{2}-\d{2}) [—–-] (\d{4}-\d{2}-\d{2})",
    re.IGNORECASE)
_LABEL = re.compile(r"\b(?:campaign[_ -]*name|norm[_ -]*query|query[_ -]*text|brand[_ -]*name|subject[_ -]*name)\s*[\"']?\s*[:=]", re.IGNORECASE)
_DESCRIPTOR = re.compile(r"\b(?:brand|top[_ -]*subject[_ -]*name|subject[_ -]*name)\s*[\"']?\s*[:=]", re.IGNORECASE)
FIELDS = {"export_contract", "request", "today", "requested_period", "source_windows",
          "warnings", "required_groups", "claim_catalog", "eligible_check_indices"}


def safe_diagnostic_question(value):
    if not isinstance(value,str) or len(value)>160 or "\n" in value or "\r" in value:
        return None
    match = _QUESTION.fullmatch(value)
    if not match:
        return None
    try:
        start,end = date.fromisoformat(match[2]),date.fromisoformat(match[3])
        if not 0 <= (end-start).days < 30:
            return None
    except ValueError:
        return None
    return value


def private_marketing_label(key, *, include_descriptors=True):
    compact = re.sub(r"[^\w]", "", key).replace("_", "").lower()
    return (compact in {"campaignname", "normquery", "querytext", "searchquery", "searchtext", "requestkey"}
            or include_descriptors and compact in {"brand", "brandname", "subjectname", "topsubjectname"})


def private_marketing_prose(value, *, include_descriptors=True):
    return bool(_LABEL.search(value) or include_descriptors and _DESCRIPTOR.search(value))


def assert_diagnostic_debug_request(value):
    from commerce_agent.marketing.diagnostics import MarketingDiagnosticRequest
    request = MarketingDiagnosticRequest.model_validate(value)
    match = _QUESTION.fullmatch(request.user_query)
    if (safe_diagnostic_question(request.user_query) is None or match[1] != request.sku
        or match[2] != str(request.date_from) or match[3] != str(request.date_to)):
        raise ValueError("Marketing diagnostic debug request outside approved grammar")


def prepared_diagnostic_model_input(prepared):
    result = {key:deepcopy(prepared[key]) for key in
        ("request", "today", "requested_period", "source_windows", "warnings", "required_groups")}
    result["export_contract"] = CONTRACT
    result["claim_catalog"] = [{"index":i,"kind":claim["kind"],"text":claim["text"]}
        for i,claim in enumerate(prepared["claims"])]
    result["eligible_check_indices"] = [i for i,claim in enumerate(prepared["claims"])
        if claim["kind"]=="sku" and claim["period"]==prepared["requested_period"]]
    return result


def assert_diagnostic_debug_prepared(prepared):
    assert_diagnostic_debug_request(prepared["request"])
    if prepared.get("diagnostic_source_mode") != "inline":
        raise ValueError("Marketing diagnostic source debug not accepted")
    if not is_diagnostic_prepared(prepared_diagnostic_model_input(prepared)):
        raise ValueError("Marketing diagnostic prepared debug contract invalid")
    # Populated frequency-query/hydrated source acceptance is independent of
    # this inline SKU/campaign/visibility capture. Never infer approval from
    # the global environment switch.
    if any(prepared["native_data"]["queries"].get(key) for key in
        ("top_query_rows","query_waste_rows","query_scale_rows","query_position_risk_rows")):
        raise ValueError("Marketing populated diagnostic query debug not accepted")


def is_diagnostic_prepared(value):
    if not isinstance(value,dict) or set(value)!=FIELDS or value["export_contract"]!=CONTRACT:
        return False
    request = value["request"]
    keys = {"capability","sku","date_from","date_to"}
    if (not isinstance(request,dict) or not keys<=set(request)<=keys|{"user_query"}
        or request["capability"]!="sku_diagnostic" or not isinstance(request["sku"],str)
        or re.fullmatch(r"[1-9][0-9]{4}",request["sku"]) is None
        or "user_query" in request and not isinstance(request["user_query"],str)):
        return False
    def iso(item):
        try: return isinstance(item,str) and date.fromisoformat(item).isoformat()==item
        except ValueError: return False
    def window(item):
        return (isinstance(item,str) and len(item.split("/"))==2
            and all(iso(part) for part in item.split("/")) and item.split("/")[0]<=item.split("/")[1])
    if (not all(iso(item) for item in (request["date_from"],request["date_to"],value["today"]))
        or not 0<=(date.fromisoformat(request["date_to"])-date.fromisoformat(request["date_from"])).days<30
        or request["date_to"]>value["today"]
        or value["requested_period"]!=request["date_from"]+"/"+request["date_to"]):
        return False
    windows = value["source_windows"]
    if (not isinstance(windows,dict) or set(windows)!={"sku","campaigns","queries","bids","visibility"}
        or windows["sku"]!=value["requested_period"] or windows["bids"] is not None or windows["visibility"] is not None
        or any(item is not None and (not window(item) or item.split("/")[1]>value["today"]) for item in windows.values())):
        return False
    claims = value["claim_catalog"]
    if (not isinstance(claims,list) or not claims or len(claims)>1000
        or not all(isinstance(c,dict) and set(c)=={"index","kind","text"} and type(c["index"]) is int
            and c["index"]==i and isinstance(c["kind"],str) and c["kind"] in {"sku","campaigns","queries","visibility"}
            and isinstance(c["text"],str) for i,c in enumerate(claims))
        or not isinstance(value["warnings"],list) or not all(isinstance(w,str) for w in value["warnings"])):
        return False
    def indices(items):
        return (isinstance(items,list) and all(type(i) is int and 0<=i<len(claims) for i in items)
            and len(items)==len(set(items)))
    groups = value["required_groups"]
    return (isinstance(groups,list) and bool(groups) and all(indices(group) and 1<=len(group)<=6 for group in groups)
        and len([i for group in groups for i in group])==len({i for group in groups for i in group})
        and indices(value["eligible_check_indices"]) and bool(value["eligible_check_indices"])
        and all(claims[i]["kind"]=="sku" for i in value["eligible_check_indices"]))
