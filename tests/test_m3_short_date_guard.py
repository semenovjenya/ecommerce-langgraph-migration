"""G03 wrong-window regression: never silently replace a short calendar range."""
import pytest

from commerce_agent.sales.policy import resolve_sales_request
from commerce_agent.marketing.policy import resolve_marketing_request
from commerce_agent.supply.policy import resolve_supply_request
from commerce_agent.finance.policy import resolve_finance_request

CASES=[(resolve_sales_request,"Отчёт продаж",{}),
    (resolve_marketing_request,"Отчёт рекламы",{}),
    (resolve_supply_request,"Оборачиваемость запасов",{"operation":"turnover"}),
    (resolve_finance_request,"Финансовый отчёт",{"operation":"report"})]


@pytest.mark.parametrize("resolve,prefix,constraints",CASES)
@pytest.mark.parametrize("window",["01.04-07.04","1.4 — 7.4","01/04–07/04","01.04 по 07.04","01.04.2026-07.04.2026",
    "2026/04/01 — 2026/04/07","2026.04.01 по 2026.04.07",
    "01-04 — 07-04","01-04-2026-07-04-2026","01.04","01-04","01-04-26"])
@pytest.mark.parametrize("trusted_operation",[False,True])
def test_short_dates_are_rejected_not_rolling_defaults(resolve,prefix,constraints,window,trusted_operation):
    with pytest.raises(ValueError):
        resolve(prefix+" за "+window,constraints if trusted_operation else None)


@pytest.mark.parametrize("resolve,prefix,constraints",CASES)
def test_explicit_trusted_iso_normalization_is_preserved(resolve,prefix,constraints):
    request=resolve(prefix+" за 01.04-07.04",{**constraints,"date_from":"2026-04-01","date_to":"2026-04-07"})
    assert str(request.date_from)=="2026-04-01" and str(request.date_to)=="2026-04-07"


def test_literal_date_like_sku_is_not_mistaken_for_query_window():
    request=resolve_sales_request("Отчёт продаж SKU 01.04-07.04 за последние 7 дней")
    assert request.targets==("01.04-07.04",) and request.days==7


@pytest.mark.parametrize("resolve,prefix,constraints",CASES)
@pytest.mark.parametrize("target",["01","01.04-07.04"])
def test_sku_mask_cannot_erase_separate_calendar_range(resolve,prefix,constraints,target):
    with pytest.raises(ValueError):
        resolve(prefix+f" SKU {target} за 01.04-07.04")
    with pytest.raises(ValueError):
        resolve(prefix+f" SKU {target} за 01.04-07.04",{**constraints,"scope":"single_sku","targets":[target]})


@pytest.mark.parametrize("resolve,prefix,constraints",CASES)
@pytest.mark.parametrize("window",["2026-09-01 — 2026-09-30","2026-09-01","2026/09/01 — 2026/09/30",
    "2026-09-01-2026-09-30","2026/09/01-2026/09/30"])
def test_operation_only_constraint_never_discards_explicit_calendar(resolve,prefix,constraints,window):
    with pytest.raises(ValueError):
        resolve(prefix+" за "+window,{**constraints,"days":7})
