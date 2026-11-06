"""Bounded triage, not a profitability formula or a write recommendation.

Sort native metrics; never calculate DRR, revenue, profit or attribution here.
The queue covers at most four distinct SKU / eight read-only neighbor checks.
"""


def prioritize_skus(rows, claims):
    if len({row.vendor_code for row in rows}) != len(rows):
        raise ValueError("Marketing duplicate SKU identity")
    active = [row for row in rows if row.current_spend_rub is not None and row.current_spend_rub > 0]
    spend = lambda row: (-row.current_spend_rub, row.vendor_code)
    queue = []

    def add(row, reason):
        existing = next((item for item in queue if item["sku"] == row.vendor_code), None)
        if existing is not None:
            existing["reasons"].append(reason)
            return
        if len(queue) == 4:
            return
        indices = [i for i, claim in enumerate(claims) if claim["kind"] == "sku" and claim["sku"] == row.vendor_code]
        if not indices:
            raise ValueError("Marketing priority lacks native claims")
        queue.append({"rank": len(queue) + 1, "sku": row.vendor_code, "nm_id": row.nm_id,
            "period": claims[indices[0]]["period"], "claim_indices": indices,
            "reasons": [reason], "domains": ["sales", "finance"],
            "spend_rub": row.current_spend_rub, "estimated_drr_pct": row.current_estimated_drr_pct,
            "ad_orders": row.current_ad_orders, "attribution": row.attribution_quality})

    missing = sorted([row for row in active if row.current_estimated_drr_pct is None], key=spend)
    if missing:
        add(missing[0], "Нет ДРР при положительном расходе: проверить полноту данных.")
    ranked = sorted([row for row in active if row.current_estimated_drr_pct is not None],
                    key=lambda row: (-row.current_estimated_drr_pct, *spend(row)))
    for row in ranked[:2]:
        add(row, "Один из двух наибольших ДРР среди доступных SKU; не доказательство убыточности.")
    zero = sorted([row for row in active if row.current_ad_orders == 0], key=spend)
    if zero:
        add(zero[0], "Наибольший расход среди SKU с нулём заказов в рекламной статистике.")
    declining = sorted([row for row in active if row.ad_orders_delta is not None and row.ad_orders_delta < 0], key=spend)
    if declining:
        add(declining[0], "Наибольший расход среди SKU со снижением заказов в рекламной статистике.")
    return queue
