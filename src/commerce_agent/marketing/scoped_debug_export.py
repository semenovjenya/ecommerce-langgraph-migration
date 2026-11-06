"""Closed scoped debug contract; source enrollment additionally needs a trusted flag."""
from copy import deepcopy
from datetime import date, timedelta
import re

CONTRACT = "marketing_scoped_prepared_v1"
_QUESTION = re.compile(r"Отч[её]т рекламы по портфелю и SKU ([1-9][0-9]{4}(?:, [1-9][0-9]{4}){0,19}) за (\d{4}-\d{2}-\d{2}) [—–-] (\d{4}-\d{2}-\d{2})", re.IGNORECASE)
FIELDS = {"export_contract", "scoped_marketing_policy_version", "request", "today", "period",
          "previous_period", "report_date", "warnings", "required_groups", "claim_catalog"}


def safe_scoped_question(value):
    if not isinstance(value, str) or len(value) > 256 or "\n" in value or "\r" in value:
        return None
    match = _QUESTION.fullmatch(value)
    if not match or len(set(match[1].split(", "))) != len(match[1].split(", ")):
        return None
    try:
        if not 0 <= (date.fromisoformat(match[3]) - date.fromisoformat(match[2])).days < 30:
            return None
    except ValueError:
        return None
    return value


def assert_scoped_debug_request(value):
    from commerce_agent.marketing.scoped import MarketingScopedRequest
    request = MarketingScopedRequest.model_validate(value)
    question = safe_scoped_question(request.user_query)
    match = _QUESTION.fullmatch(question) if question else None
    if (match is None or request.scope != "portfolio" or tuple(request.targets) != tuple(match[1].split(", "))
        or str(request.date_from) != match[2] or str(request.date_to) != match[3]
        or request.days != (request.date_to - request.date_from).days + 1):
        raise ValueError("Marketing scoped debug request outside approved grammar")


def prepared_scoped_model_input(prepared, *, include_export_contract=False):
    """One compact builder shared by provider and export/reconstruction tests.

    Non-debug output is byte-equivalent to the existing provider contract.
    The versioned debug variant adds only the contract/policy markers. It is
    used only by the separately enrolled development source profile.
    """
    if type(include_export_contract) is not bool:
        raise ValueError("invalid scoped export contract flag")
    payload = {key:deepcopy(prepared[key]) for key in
        ("request", "today", "period", "previous_period", "report_date", "warnings", "required_groups")}
    payload["claim_catalog"] = [{"index":i, "text":c["text"], "target":c["requested_target"]}
        for i, c in enumerate(prepared["claims"])]
    if include_export_contract:
        payload.update(export_contract=CONTRACT, scoped_marketing_policy_version=prepared["scoped_marketing_policy_version"])
        if not is_scoped_model_input(payload):
            raise ValueError("Marketing scoped compact contract invalid")
    return payload


def is_scoped_model_input(value):
    if (not isinstance(value, dict) or set(value) != FIELDS or value["export_contract"] != CONTRACT
        or type(value["scoped_marketing_policy_version"]) is not int or value["scoped_marketing_policy_version"] != 3):
        return False
    request = value["request"]
    keys = {"capability", "scope", "targets", "days", "date_from", "date_to"}
    if (not isinstance(request, dict) or not keys <= set(request) <= keys | {"user_query"}
        or request["capability"] != "scoped_report" or request["scope"] != "portfolio"
        or not isinstance(request["targets"], list) or not 1 <= len(request["targets"]) <= 20
        or not all(isinstance(t, str) and re.fullmatch(r"[1-9][0-9]{4}", t) for t in request["targets"])
        or len(set(request["targets"])) != len(request["targets"])
        or type(request["days"]) is not int or not 1 <= request["days"] <= 30):
        return False
    try:
        dates = [value["today"], value["report_date"], request["date_from"], request["date_to"]]
        if not all(isinstance(d, str) and date.fromisoformat(d).isoformat() == d for d in dates):
            return False
        today, anchor, start, end = (date.fromisoformat(d) for d in dates)
        days = request["days"]
        if not start <= end <= anchor <= today or (end-start).days+1 != days:
            return False
        if value["period"] != f"{start}/{end}" or value["previous_period"] != f"{start-timedelta(days=days)}/{start-timedelta(days=1)}":
            return False
    except (ValueError, TypeError, OverflowError):
        return False
    if "user_query" in request:
        try:
            assert_scoped_debug_request(request)
        except (ValueError, TypeError):
            return False
    claims = value["claim_catalog"]
    if (not isinstance(claims, list) or not 1 <= len(claims) <= 1000
        or not all(isinstance(c, dict) and set(c) == {"index", "text", "target"}
            and type(c["index"]) is int and c["index"] == i and isinstance(c["text"], str) and len(c["text"]) <= 8192
            and isinstance(c["target"], str) and c["target"] in {"portfolio", *request["targets"]} for i, c in enumerate(claims))
        or not isinstance(value["warnings"], list) or len(value["warnings"]) > 300
        or not all(isinstance(w, str) and len(w) <= 512 for w in value["warnings"])):
        return False
    groups = value["required_groups"]
    return (isinstance(groups, list) and bool(groups)
        and all(isinstance(g, list) and 1 <= len(g) <= 6 and all(type(i) is int and 0 <= i < len(claims) for i in g) for g in groups)
        and len([i for g in groups for i in g]) == len({i for g in groups for i in g}))


def assert_scoped_debug_prepared(prepared,*,require_clock=False):
    """Preflight only; neither this helper nor a global flag enrolls a source."""
    from commerce_agent.marketing.scoped import build_scoped_catalog, canonical
    if type(require_clock) is not bool:raise ValueError("Marketing scoped debug clock flag invalid")
    assert_scoped_debug_request(prepared["request"])
    if (prepared.get("capability") != "scoped_report" or type(prepared.get("scoped_marketing_policy_version")) is not int
        or prepared["scoped_marketing_policy_version"] != 3 or prepared.get("anchor_source_mode") != "inline"
        or len(prepared.get("scoped_read_results", [])) != len(prepared["request"]["targets"])
        or any(r.get("mode") != "inline" for r in prepared["scoped_read_results"])
        or len(prepared.get("portfolio_campaign_reads", [])) != 2
        or any(r.get("mode") != "inline" for r in prepared["portfolio_campaign_reads"])):
        raise ValueError("Marketing scoped debug source not accepted")
    if canonical(build_scoped_catalog(prepared)) != canonical(prepared):
        raise ValueError("Marketing scoped debug native context changed")
    if require_clock:
        from commerce_agent.marketing.scoped_debug_source import assert_debug_clock,validate_debug_source
        clock=prepared.get("scoped_debug_clock")
        if not isinstance(clock,dict) or set(clock)!={"mode","envelope"} or clock["mode"]!="inline":
            raise ValueError("Marketing scoped debug clock source not accepted")
        today=assert_debug_clock(clock["envelope"])
        if str(today)!=prepared["today"]:raise ValueError("Marketing scoped debug clock changed")
        from commerce_agent.marketing.scoped import MarketingScopedRequest
        request=MarketingScopedRequest.model_validate(prepared["request"])
        validate_debug_source("marketing/report-snapshot",prepared["anchor_snapshot"],{"product_keyword":None},request,today)
        for record in prepared["scoped_read_results"]:
            args={"date_from":str(request.date_from),"date_to":str(request.date_to),"sku_list":record["target"],"product_keyword":None,"period_days":request.days,"limit":100}
            validate_debug_source("marketing/sku-period-comparison",record["envelope"],args,request,today)
        for record in prepared["portfolio_campaign_reads"]:
            validate_debug_source(record["operation"],record["envelope"],record["arguments"],request,today)
    from commerce_agent.marketing.proposal_privacy import project_native_nm_ids, OMITTED
    drr = prepared["portfolio_campaign_reads"][0]["envelope"]["data"]
    if any(project_native_nm_ids(row.get("nm_ids")) == OMITTED for row in drr["campaign_rows"]):
        raise ValueError("Marketing scoped debug technical IDs contain unapproved text")
    prepared_scoped_model_input(prepared, include_export_contract=True)


def assert_scoped_debug_checkpoint(state):
    """Reject source/marker loss before the first resumed callback.

    Empty initial/skipped and known failed states are legitimate; a candidate
    or published business result without its validated source is not.
    """
    from commerce_agent.marketing.scoped import validate_scoped_publication,render_scoped_report,canonical
    from commerce_agent.models import DomainAnalysis
    values=state.values
    prepared=values.get("prepared_marketing")
    request=values.get("marketing_request")
    candidate=values.get("marketing_candidate")
    failure=values.get("marketing_failure")
    skipped=values.get("marketing_skipped",False)
    if failure not in {None,"invalid_request","needs_data","analysis_rejected"} or type(skipped) is not bool:
        raise ValueError("Marketing scoped debug checkpoint lifecycle unknown")
    for key in ("marketing_revision","marketing_base_attempt"):
        if key in values and (type(values[key]) is not int or values[key]<0):
            raise ValueError("Marketing scoped debug checkpoint revision invalid")
    if request:assert_scoped_debug_request(request)
    if prepared:
        assert_scoped_debug_prepared(prepared,require_clock=True)
        if request and canonical(request)!=canonical(prepared["request"]):
            raise ValueError("Marketing scoped debug checkpoint request changed")
        if candidate is not None:validate_scoped_publication(candidate,prepared)
    elif candidate is not None or (set(state.next)&{"collect_evidence","analyze","validate_and_publish"}
        and not skipped and not failure and ("collect_evidence" not in state.next or not request)):
        raise ValueError("Marketing scoped debug checkpoint source missing")
    analysis=values.get("marketing_analysis")
    if analysis is not None:
        analysis=DomainAnalysis.model_validate(analysis)
        if analysis.result is not None:
            if not prepared:raise ValueError("Marketing scoped debug publication source missing")
            if render_scoped_report(analysis.result,prepared)!=analysis.summary:
                raise ValueError("Marketing scoped debug publication changed")
        elif (analysis.status!="failed" or failure not in {"invalid_request","needs_data","analysis_rejected"}
            or analysis.summary!="Анализ рекламы недоступен: недостаточно проверенных данных или результат не прошёл проверку."):
            raise ValueError("Marketing scoped debug failed publication changed")
