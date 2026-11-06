"""V6 nodes integrated into the fixed Marketing graph, preserving v1-v5 readers."""
import asyncio
from copy import deepcopy
from hashlib import sha256
from commerce_agent.marketing import compact, feedback, research, research_phases, research_owned
from commerce_agent.marketing.feedback_details import fit_read_budget, render_history
from commerce_agent.marketing.research_provider import ResponseRejected
from commerce_agent.sales.contracts import DiagnosticFinding, DriverClass


def checked(state, context):
    journal = state["marketing_investigation_v6"]
    research.assert_owner(journal, context)
    if state.get("prepared_marketing"):
        research.validate_journal(state["prepared_marketing"], state["marketing_investigation_reads"], journal)
    return journal


async def investigate(state, context, dependencies):
    journal = deepcopy(checked(state, context))
    reads, prepared = state["marketing_investigation_reads"], state["prepared_marketing"]
    phased = journal["wire_version"] in (2, 3)
    protocol = research_owned if journal["wire_version"] == 3 else research_phases
    raw = None
    if len(journal["events"]) >= 3 or phased and protocol.planning_exhausted(journal):
        return {"marketing_failure": "research_budget_exhausted"}
    try:
        async with asyncio.timeout(dependencies.model_timeout):
            raw = await dependencies.investigator.decide(prepared=deepcopy(prepared), reads=deepcopy(reads), journal=deepcopy(journal))
        event = (protocol.normalize_wire if phased else research.normalize)(raw, prepared, reads, journal)
    except research_phases.ContractError as exc:
        event = dict(status="rejected", reads_before=len(reads), error="model_invalid", issues=exc.issues)
    except ResponseRejected as exc:
        event = dict(status="rejected", reads_before=len(reads), error=exc.code)
    except TimeoutError:
        event = dict(status="rejected", reads_before=len(reads), error="provider_timeout")
    except (ValueError, TypeError, KeyError):
        event = dict(status="rejected", reads_before=len(reads), error="model_invalid")
    except Exception as exc:
        from langgraph.errors import GraphBubbleUp, NodeCancelledError
        if isinstance(exc, (GraphBubbleUp, NodeCancelledError)):
            raise
        event = dict(status="rejected", reads_before=len(reads), error="provider_failed")
    if phased and event["status"] == "rejected":
        event.setdefault("issues", [{"code": "contract_invalid", "path": "$"}])
        if journal["wire_version"] == 3:
            event["draft"] = research_owned.safe_draft(raw, journal)
            try:
                from commerce_agent.marketing.policy import assert_bounded_safe
                assert_bounded_safe({**journal, "events": [*journal["events"], event]}, max_bytes=262144-2048)
            except ValueError:
                event["draft"] = None
    journal["events"].append(event)
    research.validate_journal(prepared, reads, journal)
    update = {"marketing_investigation_v6": journal}
    # Transport failures are explicit terminal states, not automatic retries.
    if event["status"] == "rejected" and (len(journal["events"]) == 3 or event["error"] in research.TERMINAL_ERRORS
            or phased and protocol.planning_exhausted(journal)):
        update["marketing_failure"] = "research_"+event["error"]
    return update


async def execute(state, context, dependencies, graph=None):
    journal = checked(state, context)
    prepared = state["prepared_marketing"]
    reads = deepcopy(state["marketing_investigation_reads"])
    item = research.pending(journal, reads)[0]
    check = compact.ExecutionCheck.model_validate({k: item[k] for k in compact.ExecutionCheck.model_fields})
    try:
        async with asyncio.timeout(dependencies.collection_timeout):
            async with dependencies.client_factory() as client:
                read = await feedback.execute_check(client, context, check, prepared,
                    neighbors=dependencies.feedback_neighbors, graph=graph, details=True)
    except TimeoutError:
        read = {**check.model_dump(mode="json"), "arguments": feedback.check_arguments(check, prepared), "status": "timeout", "envelope": None}
    except Exception as exc:
        from langgraph.errors import GraphBubbleUp, NodeCancelledError
        if isinstance(exc, (GraphBubbleUp, NodeCancelledError)):
            raise
        read = {**check.model_dump(mode="json"), "arguments": feedback.check_arguments(check, prepared), "status": "failed", "envelope": None}
    read = fit_read_budget(read, prepared, reads)
    read.update(item)
    reads.append(read)
    research.validate_journal(prepared, reads, journal)
    return {"marketing_investigation_reads": reads}


def next_node(state):
    if state.get("marketing_failure") or state.get("marketing_skipped"):
        return "validate_and_publish"
    journal, reads = state["marketing_investigation_v6"], state["marketing_investigation_reads"]
    pending = research.pending(journal, reads)
    if pending:
        return {"sales": "check_sales", "supply": "check_supply", "price": "check_price"}.get(pending[0]["operation"], "execute_checks")
    decision = research.previous_decision(journal)
    return "validate_and_publish" if decision and decision["finished"] else "investigate"


def evidence(prepared, reads):
    catalog = compact.evidence_catalog(prepared, reads)
    return {ref: dict(evidence_ref="marketing-investigation-v6:"+("price:" if value.get("operation")=="price" else "")+ref+":"+sha256(compact.canonical(value).encode()).hexdigest(),
        domain="marketing", claim="Native source, not proven causality", source=(value.get("envelope") or {}).get("source", "marketing_research_"+ref),
        period=None if ref == "context" else prepared["period"], freshness="unknown") for ref, value in catalog.items()}


def publish(base, prepared, reads, journal, *, failure=None):
    research.validate_journal(prepared, reads, journal)
    result = base.model_copy(deep=True)
    refs = evidence(prepared, reads)
    final = research.previous_decision(journal)
    parts = ["# Marketing: исследование", "Поддержка гипотезы не доказывает причинность."]
    if failure:
        parts.append("Исследование не завершено: " + failure + ". Ниже сохранены проверенные исходные данные и ответы источников.")
        # Do not publish a failed/rejected model's prose or hide question gaps.
        from commerce_agent.sales.contracts import DataGap as ResultGap
        result.data_gaps.append(ResultGap(metric="marketing-investigation-v6:failure", reason="Исследование не завершено; причины и существенные части вопроса могут остаться непроверенными", impact=failure))
    else:
        if not final or not final["finished"]:
            raise ValueError("research_publication_pending")
        from commerce_agent.sales.contracts import DataGap as ResultGap
        for h in final["hypotheses"]:
            observation = "\n\n".join([h["observation"], h["revision"], h["limitation"]])
            result.findings.append(DiagnosticFinding(factor=h["explanation"], observation=observation,
                impact=h["priority"], cause=h["assessment"], action=h["limitation"],
                evidence=[refs[r]["evidence_ref"] for r in h["evidence"]], confidence="low", driver_class=DriverClass.NOT_A_DRIVER))
            result.data_gaps.append(ResultGap(metric="marketing-investigation-v6:"+h["id"], reason=h["limitation"], impact="Причинность не установлена"))
            parts.extend(["## "+h["explanation"], h["assessment"], observation])
        explanations = {}
        for event in journal["events"]:
            if event["status"] == "accepted":
                for h in event["decision"]["retired_hypotheses"]:
                    parts.append("Снята версия: "+explanations[h["id"]]+". "+h["reason"])
                explanations.update({h["id"]: h["explanation"] for h in event["decision"]["hypotheses"]})
    for read in reads:
        parts.append("Проверка "+read["operation"]+": "+read["status"])
        if read["status"] == "ok":
            data = read["envelope"]["data"]
            if read["operation"] in {"stock-history", "ad-history"}:
                parts.append(render_history(data))
            elif read["operation"] in {"sales", "supply", "price"}:
                parts.append(data["analysis"]["summary"])
            else:
                parts.append("Поиск: "+data["coverage"]+"\n"+compact.canonical({k: data[k] for k in ("current_days", "previous_days", "limitations")}))
    result.cross_agent_requests = [r for r in result.cross_agent_requests if not any(
        read["status"] == "ok" and r.target_agent.value == read["operation"] and r.vendor_codes == [read["target"]]
        and r.period == prepared["period"] for read in reads)]
    result.summary = "\n\n".join(parts)
    from commerce_agent.marketing.historical_sources import render
    historical_report = render(prepared)
    if historical_report:
        result.summary += "\n\n" + historical_report
    if prepared.get("coverage_manifest_version") == 1:
        from commerce_agent.marketing.coverage import render_coverage
        result.summary += "\n\n" + render_coverage(prepared, reads, journal)
    from commerce_agent.marketing.policy import assert_bounded_safe
    assert_bounded_safe(result.model_dump(mode="json"), max_bytes=524288)
    return result, result.summary
