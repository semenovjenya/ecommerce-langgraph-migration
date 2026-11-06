"""Native day-grain Sales observations: no ratios, deltas, joins or causality."""
from datetime import date,timedelta
from hashlib import sha256
import json
import re

from commerce_agent.adapters.contracts import DailyPriceSalesComparison
from commerce_agent.sales.contracts import EvidenceFact
from commerce_agent.sales.policy import SalesRequest
from commerce_agent.sales.targeted import canonical

MART="agent_marts.v_agent_price_sku_daily"
FUNNEL="sales.wb_sales_funnel_products_daily"
FIELDS=(
    ("non_cancelled_order_units","Заказы без отмен","units",MART,"sales"),
    ("realized_sales_units","Реализованные продажи","units",MART,"sales"),
    ("realized_for_pay_rub","Выплата продавцу","RUB",MART,"sales"),
    ("base_price_rub","Базовая цена","RUB",MART,"price"),
    ("seller_discount_pct","Скидка продавца","native_pct",MART,"price"),
    ("seller_effective_price_rub","Цена продавца","RUB",MART,"price"),
    ("observed_transaction_seller_price_avg_rub","Наблюдаемая средняя цена продавца","RUB",MART,"price"),
    ("observed_buyer_price_avg_rub","Наблюдаемая средняя цена покупателя","RUB",MART,"price"),
    ("observed_spp_pct_avg","Наблюдаемая средняя СПП","native_pct",MART,"price"),
    ("open_count","Просмотры","count",FUNNEL,"funnel"),
    ("cart_count","Корзины","count",FUNNEL,"funnel"),
    ("funnel_order_count","Заказы воронки","count",FUNNEL,"funnel"),
    ("add_to_cart_conversion","open-to-cart","native_scale",FUNNEL,"funnel"),
    ("cart_to_order_conversion","cart-to-order","native_scale",FUNNEL,"funnel"),
    ("buyout_percent","Выкуп","native_scale",FUNNEL,"funnel"),
)
FLAGS=("business_day_complete","sales_day_complete","marketing_day_complete")
FACTORS={"sales":"Дневные продажи и заказы","price":"Дневной ценовой контекст","funnel":"Дневная воронка"}


def _native(scope,request,today):
    target=scope["target"]
    raw=scope["native_daily"]
    data=DailyPriceSalesComparison.model_validate(raw)
    if (canonical(data.model_dump(mode="json"))!=canonical(raw) or target not in request["targets"]
        or request.get("date_from") is not None or not 1<=request["days"]<=31
        or data.focus_keyword!=target or data.days!=request["days"] or data.source_tables!=[MART,FUNNEL]
        or data.row_count!=len(data.rows) or not data.rows or data.causal_valid is not False):
        raise ValueError("Sales daily native scope/period/source changed")
    last=max(row.report_date for row in data.rows)
    if last>date.fromisoformat(today):raise ValueError("Sales daily source date in future")
    first=last-timedelta(days=data.days-1)
    seen=set()
    for row in data.rows:
        key=(row.nm_id,row.vendor_code,row.report_date)
        if (key in seen or row.report_date<first or not row.vendor_code
            or str(row.nm_id)!=target and row.vendor_code.casefold()!=target.casefold()):
            raise ValueError("Sales daily native row identity/grain changed")
        seen.add(key)
        if (row.price_basis=="seller_snapshot_exact_day" and row.seller_effective_price_rub is None
            or row.price_basis=="observed_transaction_average" and row.observed_transaction_seller_price_avg_rub is None):
            raise ValueError("Sales daily price basis lacks its native value")
    if [row.report_date for row in data.rows]!=sorted((r.report_date for r in data.rows),reverse=True):
        raise ValueError("Sales daily native row order changed")
    refs=scope["source_refs"]
    signature=sha256(json.dumps(data.model_dump(mode="json"),sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:16]
    expected={MART}
    if any(getattr(row,key) is not None for row in data.rows for key,_,_,source,_ in FIELDS if source==FUNNEL):expected.add(FUNNEL)
    if (not isinstance(refs,list) or len(refs)!=len(expected) or {ref["source"] for ref in refs}!=expected
        or any(ref["domain"]!="sales" or ref["freshness"]!="unknown" or ref["claim"]!="Daily Sales signals (non-causal)"
            or ref["period"]!=f"{min(row.report_date for row in data.rows)}/{last}"
            or re.fullmatch(r"sales:[^:\r\n]{1,128}:sales\.daily_price_sales_comparison:[0-9a-f]{64}:[0-9a-f]{16}",ref["evidence_ref"]) is None
            or not ref["evidence_ref"].endswith(":"+signature) for ref in refs)):
        raise ValueError("Sales daily source provenance changed")
    return data


def source_facts(scope,request,today):
    data=_native(scope,request,today)
    refs={r["source"]:r["evidence_ref"] for r in scope["source_refs"]}
    facts=[]
    for index,row in enumerate(data.rows):
        period=f"{row.report_date}/{row.report_date}"
        for key,_,unit,source,_ in FIELDS:
            value=getattr(row,key)
            if value is None:continue  # Null remains in receipt/report, never becomes zero.
            facts.append(EvidenceFact(metric="daily."+key,value=value,unit=unit,period=period,sku=row.vendor_code,
                target_type="sku_day",target_id=f"{scope['target']}:{index}:{row.report_date}",
                source_table=source,evidence_ref=refs[source],source_agent="sales",
                confidence="low" if any(getattr(row,flag) is not True for flag in FLAGS) else "medium"
                ).model_dump(mode="json",exclude_none=True,exclude_defaults=True))
    return facts


def daily_scopes(prepared):
    version=prepared.get("daily_sales_policy_version")
    if version is None:
        if "daily_sales_scopes" in prepared or any(f.get("target_type")=="sku_day" or f.get("metric","").startswith("daily.") for f in prepared.get("facts",[])):
            raise ValueError("Sales daily source cannot lose policy")
        return []
    if type(version) is not int or version!=1:raise ValueError("Sales unsupported daily policy")
    request=prepared["request"];scopes=prepared.get("daily_sales_scopes")
    if canonical(SalesRequest.model_validate(request).model_dump(mode="json"))!=canonical(request):
        raise ValueError("Sales daily request types/scope changed")
    if not request["targets"] or not isinstance(scopes,list):raise ValueError("Sales daily source needs named targets")
    if len(scopes)>len(request["targets"]) or len({s["target"] for s in scopes})!=len(scopes):
        raise ValueError("Sales daily source duplicates requested target")
    evidence={e["evidence_ref"]:e for e in prepared["evidence"]}
    expected=[]
    for scope in scopes:
        facts=source_facts(scope,request,prepared["today"])
        if canonical(scope["facts"])!=canonical(facts):raise ValueError("Sales daily facts differ from native receipt")
        if any(canonical(evidence.get(ref["evidence_ref"]))!=canonical(ref) for ref in scope["source_refs"]):
            raise ValueError("Sales daily provenance absent from parent")
        expected.extend(facts)
    actual=[f for f in prepared["facts"] if f.get("target_type")=="sku_day"]
    if canonical(actual)!=canonical(expected):raise ValueError("Sales public daily facts differ from source")
    unsupported=request["date_from"] is not None or request["days"]>31
    skipped=unsupported and "scope:daily_signal_skipped_no_matching_period_capability" in prepared["warnings"]
    if any("needs_data:daily_signal_unavailable:"+target not in prepared["warnings"] for target in request["targets"]
        if target not in {s["target"] for s in scopes}) and not skipped:
        raise ValueError("Sales missing daily scope needs an explicit gap")
    return scopes


def daily_claims(prepared):
    claims=[]
    for scope in daily_scopes(prepared):
        refs={r["source"]:r["evidence_ref"] for r in scope["source_refs"]}
        data=DailyPriceSalesComparison.model_validate(scope["native_daily"])
        for index,row in enumerate(data.rows):
            for family in ("sales","price","funnel"):
                fields=[(key,label,unit,source) for key,label,unit,source,kind in FIELDS if kind==family]
                if not any(getattr(row,key) is not None for key,_,_,_ in fields):continue
                values="; ".join(f"{label}: {getattr(row,key)} {unit}" if getattr(row,key) is not None else f"{label}: нет данных"
                    for key,label,unit,_ in fields)
                flags="; ".join(f"{flag}: {getattr(row,flag)}" for flag in FLAGS)
                basis=(f"; price_basis: {row.price_basis}" if family=="price" else "")
                text=(f"{FACTORS[family]} SKU {row.vendor_code}, строка {index+1} на {row.report_date}: {values}{basis}; {flags}. "
                    "Наблюдение не устанавливает причину; полнота и шкала показателей не восстановлены предположениями.")
                claims.append({"text":text,"sku":row.vendor_code,"period":f"{row.report_date}/{row.report_date}",
                    "evidence_refs":[refs[fields[0][3]]],"factor":FACTORS[family],"family":family,
                    "confidence":"low" if any(getattr(row,flag) is not True for flag in FLAGS) else "medium",
                    "daily_target":scope["target"],"source_row":index})
    return claims


def required_daily_claims(prepared):
    # Latest returned row of each available family; ALL rows remain in report.
    # Different nm_id/day rows never become an invented aggregate.
    groups={}
    for claim in daily_claims(prepared):groups.setdefault((claim["daily_target"],claim["family"]),claim)
    return list(groups.values())
