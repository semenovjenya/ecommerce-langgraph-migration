"""Private v1 coverage projection for historical source groups, never all DB fields."""
from collections import defaultdict
from datetime import timedelta
from commerce_agent.adapters.data_coverage import CoverageManifest
from commerce_agent.adapters.historical_sources import DELIVERY_FIELDS


def historical_manifest(value):
    data = value.data
    if data.version != 2:
        raise ValueError("delivery field coverage requires historical source v2")
    groups = (
        ("funnel", DELIVERY_FIELDS, "nm_id × report_date; native selected/past windows"),
        ("search_details", ("avg_position_current", "visibility_current", "open_card_current", "add_to_cart_current", "orders_current"), "nm_id × snapshot × measurement window"),
        ("search_texts", ("avg_position_current", "visibility_current", "frequency_current", "week_frequency", "orders_current"), "nm_id × text × top_order_by × snapshot × window"),
        ("serp", ("source_found_any", "source_found_organic", "source_position_absolute", "source_position_organic", "max_position_scanned", "parser_status"), "run × query × region; scan metadata"),
    )
    sources = []
    scoped_read_executed = data.identity_status == "resolved"
    for source, (name, fields, grain) in zip(data.sources, groups):
        rows = getattr(data, name)
        refs = [str(i) for i in range(len(rows))]
        origins = defaultdict(list)
        windows = set(); snapshots = []; observations = []
        for i, row in enumerate(rows):
            origin = getattr(row, "observation_origin", "replay" if row.replay_batch else "unknown")
            origins[origin].append(str(i))
            for start, end in (("selected_period_start", "selected_period_end"), ("current_period_start", "current_period_end"), ("past_period_start", "past_period_end")):
                a, b = getattr(row, start, None), getattr(row, end, None)
                if a is not None and b is not None:
                    windows.add((a,b))
            at = getattr(row, "snapshot_hour", getattr(row, "observed_at", None))
            if at is not None:
                snapshots.append(at)
                if origin == "observed":
                    observations.append(at)
        if name == "funnel":
            start, end = data.previous_period.date_from, data.requested_period.date_to
            expected = (end-start).days+1
            missing = sorted(str(start+timedelta(days=i)) for i in range(expected) if start+timedelta(days=i) not in {r.day for r in rows})
            coverage = dict(status="partial" if missing else "complete", unit="days", expected=expected,
                observed=len(rows), missing_refs=missing, reason="returned_day_coverage_only_not_field_completeness")
            comparison = dict(status="incomparable", reason="no_valid_weights_for_weekly_delivery_or_localization; use native row windows")
        else:
            coverage = dict(status="unknown",unit="rows",expected=None,observed=len(rows),missing_refs=[],
                reason="collection_schedule_and_expected_snapshot_count_not_proven")
            comparison = dict(status="unknown", reason="query_region_window_and_observation_origin_require_scoped_comparison")
        if not scoped_read_executed:
            coverage = dict(status="unknown",unit=coverage["unit"],expected=None,observed=None,missing_refs=[],
                reason="source_query_not_executed_identity_"+data.identity_status)
            comparison = dict(status="unknown",reason="source_query_not_executed_identity_"+data.identity_status)
        nonnull = {}
        for field in fields:
            missing = [str(i) for i,row in enumerate(rows) if getattr(row, field) is None]
            nonnull[source+"."+field] = dict(status="partial" if missing else "complete" if rows else "unknown",
                unit="values",expected=len(rows) if rows else None,observed=len(rows)-len(missing) if scoped_read_executed else None,missing_refs=missing,
                reason="nonnull_returned_values_only; zero_is_preserved_not_proof_of_validity")
        sources.append(dict(source_id=source,field_ids=list(nonnull),sku=data.vendor_code,nm_id=data.nm_id,grain=grain,
            requested_window=data.requested_period.model_dump(),previous_window=data.previous_period.model_dump(),
            measurement_windows=[dict(date_from=a,date_to=b) for a,b in sorted(windows)],
            observed_at=max(observations) if observations else None,
            snapshot_at=max(snapshots) if snapshots else None,
            time_reason="stored_snapshot_may_be_replay_or_unknown; observed_at_requires_explicit_observed_origin",
            fetched_at=value.freshness.observed_at,
            rows_ref="/context/historical_sources/data/"+name,row_refs=refs,
            availability=dict(status="unknown" if not scoped_read_executed else "available" if rows else "empty",
                reason="source_query_not_executed_identity_"+data.identity_status if not scoped_read_executed else "typed_read_returned_rows" if rows else "typed_read_returned_no_rows"),
            row_coverage=coverage,field_nonnull_coverage=nonnull,comparability=comparison,
            origins=[dict(origin=origin,row_refs=indices,input_origins=[],reason="explicit_source_origin" if origin=="observed" else "lineage_marks_copy" if origin=="replay" else "no_authoritative_origin_record") for origin,indices in sorted(origins.items())],
            consumption=dict(status="fetched" if scoped_read_executed else "not_requested",
                reason="validated_source_read; projection_is_not_delivery_or_use" if scoped_read_executed else "source_query_not_executed_identity_"+data.identity_status,
                proof_scope="source_read" if scoped_read_executed else "unknown",
                boundary_ref=None,check_receipt_ref=None,observation_refs=[],accepted_event_ref=None),
            evidence_refs=[item.id for item in value.evidence if item.source==source]))
    return CoverageManifest(version=1,sources=sources,scope="historical_sources_only",limitations=[
        "not_a_complete_database_source_manifest", "nonnull_is_not_quality_or_comparability", "projection_is_not_provider_delivery",
        "accepted_observation_binding_is_not_semantic_acceptance"]).model_dump(mode="json")


def accepted_usage(prepared, reads, journal):
    """Derive selected field/status bindings only after canonical journal validation.

    Scripted and real investigators both produce accepted events; this explicitly
    does not attest transport. Coverage/status citations do not consume all fields.
    """
    from commerce_agent.marketing.research import validate_journal
    from commerce_agent.adapters.data_coverage import UsageManifest
    validate_journal(prepared, reads, journal)
    bindings = []
    for event_index, event in enumerate(journal["events"]):
        if event["status"] != "accepted":
            continue
        for hypothesis, observations in event.get("source_metadata", {}).items():
            for ref, item in observations.items():
                receipt = item["receipt"]
                if not receipt.startswith("r"):
                    continue
                read = reads[int(receipt[1:])]
                metric = item["metric"]
                kind = "status" if metric == "read_status" else "coverage" if metric == "coverage" else "data"
                source = (item.get("native_scope") or {}).get("source_table")
                # Delivery has an explicitly traced raw field. Aggregate fact
                # metrics and generic receipt pointers are not raw-column usage.
                fields = [source + "." + metric] if source and "_delivery_" in ref else []
                event_ref = "event:" + str(event_index)
                bindings.append(dict(observation_ref=ref, observation_kind=kind, source_id=source,
                    field_ids=fields,hypothesis_id=hypothesis,check_receipt_ref=receipt,read_key=read["read_key"],
                    sku=read["target"],requested_window={key:read["arguments"][key] for key in ("date_from", "date_to")},
                    accepted_event_ref=event_ref,consumption=dict(status="used_in_check",reason="accepted_observation_bound_to_executed_check; usability_and_causality_are_separate",
                        proof_scope="accepted_binding",boundary_ref=None,check_receipt_ref=receipt,observation_refs=[ref],accepted_event_ref=event_ref)))
    return UsageManifest(version=1,owner=journal["owner"],bindings=bindings,model_delivery="unverified",
        limitation="accepted_binding_does_not_prove_provider_delivery_or_semantic_acceptance").model_dump(mode="json")


def render_coverage(prepared, reads, journal):
    if prepared.get("coverage_manifest_version") != 1:
        return ""
    from commerce_agent.marketing.historical_sources import validate
    validate(prepared)
    usage = accepted_usage(prepared, reads, journal)
    lines = ["Покрытие исторических источников (наличие, покрытие и использование — разные статусы)."]
    for source in prepared["coverage_manifest"]["sources"]:
        coverage = source["row_coverage"]
        origin_counts = ", ".join(group["origin"]+":"+str(len(group["row_refs"])) for group in source["origins"]) or "no_returned_rows"
        selected = sorted({field for item in usage["bindings"] for field in item["field_ids"] if item["source_id"]==source["source_id"]})
        lines.append(f"{source['source_id']}: {source['availability']['status']}; coverage={coverage['status']} ({coverage['observed']}/{coverage['expected']} {coverage['unit']}); origin={origin_counts}; source_read={source['consumption']['status']}; accepted_field_refs={', '.join(selected) or 'none'}.")
    lines.append("Принятая ссылка не доказывает доставку реальному провайдеру или смысловую корректность. Неиспользованные поля остаются в исходных данных; непроверенные источники вне этого manifest не объявлены пустыми.")
    return "\n".join(lines)
