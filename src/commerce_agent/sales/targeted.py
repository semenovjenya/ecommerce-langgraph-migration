"""Separate requested Sales scopes: no portfolio filtering or aggregate math."""
from datetime import date, timedelta
import json

from commerce_agent.adapters.contracts import SalesRevenueChange
from commerce_agent.sales.contracts import EvidenceFact


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def targeted_scopes(prepared):
    version = prepared.get("targeted_sales_policy_version")
    if version is None:
        if "targeted_sales_scopes" in prepared:
            raise ValueError("Sales targeted source cannot lose its policy version")
        return []  # Historical checkpoints retain their recorded behavior.
    if type(version) is not int or version not in (1,2):
        raise ValueError("Sales unsupported targeted source policy")
    request = prepared["request"]
    scopes = prepared.get("targeted_sales_scopes")
    if request["scope"] != "portfolio" or not request["targets"] or not isinstance(scopes,list):
        raise ValueError("Sales targeted scopes require explicit portfolio targets")
    return validate_native_scopes(prepared, scopes)


def validate_native_scopes(prepared, scopes):
    """Validate exact native receipts, shared by distinct public scope policies."""
    request = prepared["request"]
    if len(scopes) > len(request["targets"]):
        raise ValueError("Sales targeted scopes exceed the request")
    seen = set()
    for scope in scopes:
        target = scope["target"]
        if target not in request["targets"] or target in seen:
            raise ValueError("Sales targeted source outside requested scope")
        seen.add(target)
        native = SalesRevenueChange.model_validate(scope["native_change"])
        start, end = date.fromisoformat(scope["period"]["date_from"]), date.fromisoformat(scope["period"]["date_to"])
        previous_end = start - timedelta(days=1)
        previous_start = start - timedelta(days=(end-start).days+1)
        if (native.source_table != "sales.wb_supplier_sales" or native.focus_keyword != target
            or native.date_from != start or native.date_to != end or end > date.fromisoformat(prepared["today"])
            or end < start or native.current_period_start != start or native.current_period_end != end
            or native.previous_period_start != previous_start or native.previous_period_end != previous_end
            or native.period_days != (end-start).days+1
            or not request.get("date_from") and native.period_days != request["days"]
            or not native.rows or native.row_count != len(native.rows)
            or request.get("date_from") and (request["date_from"] != str(start) or request["date_to"] != str(end))):
            raise ValueError("Sales targeted source period or identity changed")
        current = f"{start}/{end}"
        previous = f"{native.previous_period_start}/{native.previous_period_end}"
        refs = {f["evidence_ref"] for f in scope["facts"]}
        evidence = {e["evidence_ref"]:e for e in prepared["evidence"]}
        if len(refs) != 1 or any(ref not in evidence or evidence[ref]["source"] != native.source_table
            or evidence[ref]["period"] != current for ref in refs):
            raise ValueError("Sales targeted source provenance missing")
        ref = next(iter(refs))
        expected_facts, expected_rows = [], []
        for row in native.rows:
            if (str(row.nm_id) != target and (row.vendor_code or "").casefold() != target.casefold()
                or row.current_period_start != start or row.current_period_end != end
                or row.previous_period_start != native.previous_period_start or row.previous_period_end != native.previous_period_end):
                raise ValueError("Sales targeted native row escaped request")
            sku = row.vendor_code or scope["native_identities"].get(str(row.nm_id))
            if not sku or sku == "Артикул не сопоставлен":
                raise ValueError("Sales targeted source lacks canonical vendor code")
            expected_rows.append([sku,row.current_revenue,row.previous_revenue,row.delta_revenue,
                row.current_sales,row.previous_sales,row.delta_sales,current,previous,row.delta_revenue_pct,row.driver_role])
            for metric in ("current_revenue","previous_revenue","current_sales","previous_sales"):
                expected_facts.append(EvidenceFact(metric=metric,value=getattr(row,metric),
                    unit="RUB" if "revenue" in metric else "units",period=previous if metric.startswith("previous") else current,
                    sku=sku,target_type="sku",target_id=target,source_table=native.source_table,
                    evidence_ref=ref,source_agent="sales").model_dump(mode="json",exclude_none=True,exclude_defaults=True))
        table = scope["tables"]["sales_decomposition"]
        columns = ["Артикул","current_revenue","previous_revenue","delta_revenue","current_sales","previous_sales",
            "delta_sales","Текущий период","Предыдущий период","delta_revenue_pct","driver_role"]
        if (table["columns"] != columns or canonical(table["rows"]) != canonical(expected_rows)
            or canonical(scope["facts"]) != canonical(expected_facts)):
            raise ValueError("Sales targeted publication differs from native rows")
    selected = [fact for fact in prepared["facts"] if fact.get("target_type") == "sku" and fact.get("target_id") in request["targets"]]
    if canonical(selected) != canonical([fact for scope in scopes for fact in scope["facts"]]):
        raise ValueError("Sales public targeted facts differ from source scopes")
    if any(f"needs_data:target_unavailable:{target}" not in prepared["warnings"]
           for target in request["targets"] if target not in seen):
        raise ValueError("Sales missing target needs an explicit data gap")
    return scopes


def targeted_claims(prepared):
    return native_scope_claims(prepared, targeted_scopes(prepared))


def native_scope_claims(prepared, scopes, *, scope_label=None):
    from commerce_agent.sales.analytics import _window, _change
    claims = []
    for scope in scopes:
        native = SalesRevenueChange.model_validate(scope["native_change"])
        current = f"{native.current_period_start}/{native.current_period_end}"
        previous = f"{native.previous_period_start}/{native.previous_period_end}"
        _window(current,previous)
        refs = list(dict.fromkeys(f["evidence_ref"] for f in scope["facts"]))
        # Existing native ordering is preserved; never invent a SKU aggregate
        # from multiple nm_id rows or silently claim these are the full universe.
        for index,row in enumerate(native.rows[:5]):
            sku = row.vendor_code or scope["native_identities"].get(str(row.nm_id))
            label = f"SKU {sku}" + (f", отдельная строка источника {index+1}" if len(native.rows)>1 else "")
            if scope_label is not None:
                label += ", " + scope_label
            for name,now,before,delta,rate,unit in (("выручки",row.current_revenue,row.previous_revenue,row.delta_revenue,row.delta_revenue_pct,"RUB"),
                ("продаж",row.current_sales,row.previous_sales,row.delta_sales,None,"units")):
                claims.append({"text":_change(f"Динамика {name}",label,current,previous,now,before,delta,rate,unit),
                    "sku":sku,"period":current,"evidence_refs":refs,"factor":f"Динамика {name}","confidence":"medium",
                    "requested_target":scope["target"]})
    return claims


def required_targeted_claims(prepared):
    # At least both native families for every requested/available scope. All
    # rows remain in facts and the deterministic report; no grand totals.
    claims = targeted_claims(prepared)
    return [next(c for c in claims if c["requested_target"] == scope["target"] and c["factor"] == factor)
        for scope in targeted_scopes(prepared) for factor in ("Динамика выручки","Динамика продаж")]
