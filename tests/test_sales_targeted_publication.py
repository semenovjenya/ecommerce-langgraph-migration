"""Named portfolio deep dives must survive publication, not just tool reads."""
from copy import deepcopy
import asyncio
from uuid import uuid4
import httpx
import pytest

from commerce_agent.sales.analytics import analytical_claims
from commerce_agent.sales.policy import SalesRequest
from commerce_agent.sales.claims import native_claim_catalog, render_sales_claim
from commerce_agent.sales.contracts import SalesDiagnosticResult
from commerce_agent.sales.publication import validate_sales_publication
from commerce_agent.sales.report import render_sales_report
from commerce_agent.sales.runtime import build_sales_subgraph
from commerce_agent.sales.semantics import SalesRequestedCheck, finding_presentation, render_requested_check, render_delegated_check
from commerce_agent.sales.targeted import targeted_scopes, required_targeted_claims
from commerce_agent.sales.model import narrative_schema, LangChainSalesModel, SalesModelSettings
from commerce_agent.serialization import checkpoint_serializer
from langgraph.checkpoint.memory import InMemorySaver
from test_sales_evidence import collect, CONTEXT, respond
from test_sales_runtime import dependencies, invocation


QUERY = "Отчет продаж по всему портфелю, отдельно SKU SKU-Z"


def native_target(path, payload):
    data = payload["data"]
    if data.get("focus_keyword") != "SKU-Z": return
    if path == "overview":
        data["top_revenue_rows"][0].update(nm_id=99, vendor_code="SKU-Z", sales_7d_units=9, revenue_7d=333.33)
    if path == "sku-revenue-change":
        data["rows"][0].update(nm_id=99, vendor_code="SKU-Z", current_revenue=333.33,
            previous_revenue=111.11, delta_revenue=222.22, delta_revenue_pct=200.0,
            current_sales=9, previous_sales=4, delta_sales=5, driver_role="growth_driver")


def test_historical_collector_demonstrates_named_target_publication_gap():
    prepared, seen = collect(SalesRequest(user_query=QUERY, targets=("SKU-Z",)), native_target)
    assert [args["product_keyword"] for path,args in seen if path == "sku-revenue-change"] == [None,"SKU-Z"]
    assert prepared["deep_dives"][0]["tables"]["sales_decomposition"]["rows"][0][0] == "SKU-Z"
    assert {f["sku"] for f in prepared["facts"]} == {"SKU-A"}


def test_named_target_outside_portfolio_rows_must_have_public_native_facts_and_claims():
    prepared, seen = collect(SalesRequest(user_query=QUERY, targets=("SKU-Z",)), native_target,publish_targeted=True)
    assert [args["product_keyword"] for path,args in seen if path == "sku-revenue-change"] == [None,"SKU-Z"]
    assert prepared["tables"]["sales_decomposition"]["rows"][0][0] == "SKU-A"
    assert {f["sku"] for f in prepared["facts"]} == {"SKU-A","SKU-Z"}
    claims = [c for c in analytical_claims(prepared) if c["sku"] == "SKU-Z"]
    assert {c["factor"] for c in claims} == {"Динамика выручки","Динамика продаж"}
    assert any("333.33 / 111.11 RUB" in c["text"] for c in claims)


def candidate_for(prepared):
    claims = required_targeted_claims(prepared)
    catalog = native_claim_catalog(prepared)
    findings, requests = [], []
    for scope in targeted_scopes(prepared):
        selected = [c for c in claims if c["requested_target"] == scope["target"]]
        observation = "\n".join(c["text"] for c in selected)
        index = next(i for i,c in enumerate(catalog) if render_sales_claim(c,prepared) == selected[0]["text"])
        factor,driver,confidence = finding_presentation(observation,prepared)
        findings.append(dict(factor=factor,driver_class=driver,confidence=confidence,observation=observation,
            evidence=selected[0]["evidence_refs"],cause="Не устанавливается этой проверкой.",impact="Нужна дополнительная проверка.",
            action=render_requested_check(SalesRequestedCheck(kind="source_completeness",claim_index=index),prepared)))
        requests.append(render_delegated_check(SalesRequestedCheck(kind="marketing",claim_index=index),prepared))
    return {"agent_name":"Sales Agent","summary":"\n".join(c["text"] for c in claims) or "Проверены доступные данные.",
        "facts":deepcopy(prepared["facts"]),**deepcopy(prepared["tables"]),"findings":findings,
        "cross_agent_requests":requests,"recommendations":[],"data_gaps":[]}


def targeted_fixture(targets=("SKU-Z",),change=native_target):
    prepared,_ = collect(SalesRequest(user_query=QUERY,targets=targets),change,publish_targeted=True)
    return prepared,candidate_for(prepared)


def test_targeted_publication_report_and_checks_preserve_portfolio_without_adding_totals():
    prepared,candidate = targeted_fixture()
    published = validate_sales_publication(candidate,prepared)
    assert published.sales_decomposition.rows[0][0] == "SKU-A"
    assert len(published.facts) == 8
    assert "SKU-Z" in published.summary
    request = published.cross_agent_requests[0]
    assert "SKU SKU-Z" in request.question and "2026-10-03/2026-10-09" in request.question
    report = render_sales_report(published,prepared)
    assert "Область: портфель" in report and "### Запрошенный SKU SKU-Z" in report
    assert "333.33 | 111.11 | 222.22" in report
    assert "Их нельзя прибавлять" in report


@pytest.mark.parametrize("field",["summary","findings","cross_agent_requests"])
def test_omitting_named_target_analysis_or_checks_is_rejected(field):
    prepared,candidate = targeted_fixture()
    candidate[field] = "Проверены доступные данные." if field == "summary" else []
    with pytest.raises(ValueError,match="explicitly requested"):
        validate_sales_publication(candidate,prepared)


@pytest.mark.parametrize("mutation",["native_value","table_value","native_bool","period","source","scope","version","fact","ref","missing_scope"])
def test_targeted_source_mutations_fail_closed(mutation):
    prepared,candidate = targeted_fixture()
    scope = prepared["targeted_sales_scopes"][0]
    if mutation == "native_value": scope["native_change"]["rows"][0]["current_revenue"] = 999
    if mutation == "table_value": scope["tables"]["sales_decomposition"]["rows"][0][1] = 999
    if mutation == "native_bool": scope["native_change"]["rows"][0]["current_sales"] = True
    if mutation == "period": scope["native_change"]["previous_period_end"] = "2026-10-01"
    if mutation == "source": scope["native_change"]["source_table"] = "finance.fake"
    if mutation == "scope": scope["target"] = "SKU-Q"
    if mutation == "version": prepared.pop("targeted_sales_policy_version")
    if mutation == "fact": prepared["facts"][-1]["value"] = 999
    if mutation == "ref": scope["facts"][0]["evidence_ref"] = "fake"
    if mutation == "missing_scope": prepared["targeted_sales_scopes"] = []
    with pytest.raises(ValueError): validate_sales_publication(candidate,prepared)


@pytest.mark.parametrize("field",["sales_overview","facts"])
def test_boolean_cannot_replace_native_one_before_publication_normalization(field):
    prepared,candidate = targeted_fixture()
    if field == "sales_overview": candidate[field]["rows"][0][1] = True
    else: candidate[field][2]["value"] = True
    with pytest.raises(ValueError): validate_sales_publication(candidate,prepared)


def test_same_sku_in_main_and_target_scope_keeps_both_native_claims():
    prepared,candidate = targeted_fixture(targets=("SKU-A",),change=None)
    claims = analytical_claims(prepared)
    assert len([c for c in claims if c["factor"] == "Динамика выручки"]) == 2
    assert len(validate_sales_publication(candidate,prepared).facts) == 8


def test_twenty_named_targets_are_not_reduced_by_old_narrative_caps():
    targets = tuple(f"T-{i}" for i in range(20))
    prepared,candidate = targeted_fixture(targets=targets,change=None)
    props = narrative_schema(prepared).model_json_schema()["properties"]
    assert props["grounded_claims"]["maxItems"] >= 46
    assert props["findings"]["maxItems"] >= 20
    assert props["cross_agent_requests"]["maxItems"] >= 20
    result = validate_sales_publication(candidate,prepared)
    assert len(result.findings) == 20 and len(result.cross_agent_requests) == 20
    assert len(result.facts) == 84


def test_missing_target_is_explicit_gap_not_zero_or_portfolio_failure():
    def fail_target(path,payload):
        if path == "overview" and payload["data"]["focus_keyword"] == "SKU-Z": payload["data"]["totals"] = None
    prepared,candidate = targeted_fixture(change=fail_target)
    result = validate_sales_publication(candidate,prepared)
    assert len(result.facts) == 4
    assert "needs_data:target_unavailable:SKU-Z" in prepared["warnings"]
    report = render_sales_report(result,prepared)
    assert "Данные этой области недоступны" in report


def test_unmapped_numeric_target_does_not_drop_valid_portfolio():
    from commerce_agent.adapters.business_tools import BusinessToolClient
    from commerce_agent.sales.evidence import collect_sales_evidence
    from test_sales_evidence import SETTINGS,envelope
    def handler(request):
        if request.url.path.endswith("vendor-code-map"):
            return httpx.Response(200,json=envelope({}))
        def missing_vendor(path,payload):
            if payload["data"].get("focus_keyword") == "99":
                rows = payload["data"].get("rows",[]) if path == "sku-revenue-change" else payload["data"].get("top_revenue_rows",[])
                for row in rows: row.update(nm_id=99,vendor_code=None)
        return respond(request,change=missing_vendor)
    async def exercise():
        async with BusinessToolClient(SETTINGS,transport=httpx.MockTransport(handler)) as client:
            return await collect_sales_evidence(client,CONTEXT,SalesRequest(user_query="Весь портфель, отдельно SKU 99",targets=("99",)),publish_targeted=True)
    prepared = asyncio.run(exercise())
    assert {f["sku"] for f in prepared["facts"]} == {"SKU-A"}
    assert "needs_data:target_unavailable:99" in prepared["warnings"]
    assert "Данные этой области недоступны" in render_sales_report(candidate_for(prepared),prepared)


class TargetedFixtureModel:
    def __init__(self): self.calls = []
    async def analyze(self,*,instructions,prepared):
        self.calls.append(deepcopy(prepared))
        return SalesDiagnosticResult.model_validate(candidate_for(prepared))


def test_disabled_targeted_capability_fails_before_any_tool_or_provider_call():
    model = TargetedFixtureModel()
    def forbidden(request): raise AssertionError("tools must not be called")
    result = asyncio.run(build_sales_subgraph(dependencies(model,forbidden)).ainvoke(invocation(QUERY),context=CONTEXT))
    assert result["sales_analysis"].status == "failed" and not model.calls


def test_targeted_subgraph_checkpoint_resume_does_not_repeat_collection():
    async def exercise():
        seen, model = [],TargetedFixtureModel()
        def handler(request): return respond(request,change=native_target,seen=seen)
        dep = dependencies(model,handler,enable_targeted_reports=True)
        saver = InMemorySaver(serde=checkpoint_serializer())
        config = {"configurable":{"thread_id":str(uuid4())}}
        first = build_sales_subgraph(dep,checkpointer=saver,interrupt_after=["collect_evidence"])
        await first.ainvoke(invocation(QUERY),config,context=CONTEXT)
        before = deepcopy(seen)
        second = build_sales_subgraph(dep,checkpointer=saver)
        output = await second.ainvoke(None,config,context=CONTEXT)
        assert seen == before and len(model.calls) == 1
        assert "Запрошенный SKU SKU-Z" in output["sales_analysis"].summary
        assert output["errors"] == []
    asyncio.run(exercise())


def test_targeted_debug_is_rejected_before_callbacks(monkeypatch):
    monkeypatch.setenv("COMMERCE_AGENT_ENVIRONMENT","development")
    monkeypatch.setenv("COMMERCE_AGENT_LANGSMITH_DEBUG","true")
    with pytest.raises(ValueError,match="separate debug acceptance"):
        build_sales_subgraph(dependencies(enable_targeted_reports=True))


def test_changed_debug_environment_is_rejected_before_callbacks_on_existing_graph(monkeypatch):
    from langchain_core.callbacks import BaseCallbackHandler
    class Counter(BaseCallbackHandler):
        def __init__(self): self.calls = 0
        def on_chain_start(self,*args,**kwargs): self.calls += 1
    graph = build_sales_subgraph(dependencies(enable_targeted_reports=True))
    monkeypatch.setenv("COMMERCE_AGENT_ENVIRONMENT","development")
    monkeypatch.setenv("COMMERCE_AGENT_LANGSMITH_DEBUG","true")
    counter = Counter()
    with pytest.raises(ValueError,match="separate debug acceptance"):
        asyncio.run(graph.ainvoke(invocation(QUERY),{"callbacks":[counter]},context=CONTEXT))
    assert counter.calls == 0


def test_targeted_checkpoint_cannot_resume_under_disabled_capability():
    async def exercise():
        saver = InMemorySaver(serde=checkpoint_serializer())
        config = {"configurable":{"thread_id":str(uuid4())}}
        model = TargetedFixtureModel()
        first = build_sales_subgraph(dependencies(model,enable_targeted_reports=True),checkpointer=saver,
                                     interrupt_after=["collect_evidence"])
        await first.ainvoke(invocation(QUERY),config,context=CONTEXT)
        second = build_sales_subgraph(dependencies(model),checkpointer=saver)
        with pytest.raises(ValueError,match="checkpoint capability"):
            await second.ainvoke(None,config,context=CONTEXT)
        assert not model.calls
    asyncio.run(exercise())


@pytest.mark.parametrize("enabled",[True,False])
@pytest.mark.parametrize("clone",["copy","with_config"])
def test_native_graph_clones_preserve_enabled_and_disabled_capabilities(enabled,clone):
    model = TargetedFixtureModel()
    graph = build_sales_subgraph(dependencies(model,enable_targeted_reports=enabled))
    copied = graph.copy() if clone == "copy" else graph.with_config({"recursion_limit":40})
    assert copied._targeted_enabled is enabled
    output = asyncio.run(copied.ainvoke(invocation(QUERY),context=CONTEXT))
    assert output["sales_analysis"].status == ("partial" if enabled else "failed")
    assert len(model.calls) == (1 if enabled else 0)


@pytest.mark.parametrize("targets",[("SKU-Z",),tuple(f"T-{i}" for i in range(20))])
@pytest.mark.parametrize("policy_version",[1,2])
def test_targeted_provider_wire_selects_claims_without_generating_business_values(targets,policy_version):
    from test_sales_model import response
    prepared,_ = targeted_fixture(targets=targets,change=native_target if len(targets)==1 else None)
    prepared["targeted_sales_policy_version"] = policy_version
    required = {c["text"] for c in required_targeted_claims(prepared)}
    catalog = native_claim_catalog(prepared)
    indices = [i for i,c in enumerate(catalog) if render_sales_claim(c,prepared) in required and c.kind == "analytic"]
    seen = []
    def transport(request):
        body = __import__("json").loads(request.content)
        seen.append(body)
        assert body["store"] is False and not body.get("tools")
        assert body["text"]["format"]["strict"] is True
        prompt = next(item for item in body["input"] if item["role"] == "user")
        assert "native_change" not in __import__("json").dumps(prompt)
        grouped = [indices[i:i+2] for i in range(0,len(indices),2)]
        return httpx.Response(200,json=response({"summary":"","grounded_claims":[{"claim_index":i} for i in indices],
            "findings":[{"observation":{"kind":"native","claims":[{"claim_index":i} for i in group]},
                "requested_check":({"kind":"source_completeness"} if policy_version==2 else
                    {"kind":"source_completeness","claim_index":group[0]}),"evidence_indices":[]} for group in grouped],
            "recommendations":[],"data_gaps":[],"cross_agent_requests":[{"kind":"marketing","claim_index":g[0]} for g in grouped]}))
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            model = LangChainSalesModel(SalesModelSettings(api_key="synthetic-only-model-key"),http_async_client=client)
            candidate = await model.analyze(instructions="fixture",prepared=prepared)
            published = validate_sales_publication(candidate,prepared)
            assert len(published.findings) == len(targets)
            assert len(published.cross_agent_requests) == len(targets)
            if len(targets)==1:
                assert "333.33 / 111.11 RUB" in published.summary
                assert "SKU SKU-Z" in published.cross_agent_requests[0].question
    asyncio.run(exercise())
    assert len(seen) == 1
