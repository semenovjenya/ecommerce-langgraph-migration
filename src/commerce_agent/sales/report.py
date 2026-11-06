"""Readable Sales publication, rendered only from validated native evidence.

No formulas, causal inference, portfolio aggregation or provider-written prose.
The compact legacy result remains intact; this is DomainAnalysis.summary.
"""
from collections import defaultdict
from decimal import Decimal
from html import escape

from commerce_agent.sales.policy import MAX_RESULT_BYTES, assert_bounded_safe
from commerce_agent.sales.publication import validate_sales_publication
from commerce_agent.sales.targeted import targeted_scopes


def _cell(value):
    if value is None:
        return "нет данных"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Presentation only: never recompute a service delta or percentage.
        value = format(Decimal(str(value)), ",f").replace(",", " ")
    return escape(str(value)).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ").replace("`", "\\`").replace("[", "\\[").replace("]", "\\]").replace("*", "\\*").replace("_", "\\_")


def _table(columns, rows):
    return ["| " + " | ".join(_cell(c) for c in columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *("| " + " | ".join(_cell(c) for c in row) + " |" for row in rows)]


def render_sales_report(candidate, prepared):
    result = validate_sales_publication(candidate, prepared)
    lines = ["# Отчёт Sales", "", "## Общая картина", "",
             "Ниже — наблюдаемые продажи и выручка по данным бизнес-сервисов. "
             "Изменения не доказывают причины; финансовая прибыль здесь не рассчитана.", ""]
    snapshot = result.sales_report_snapshot
    # Closed layouts: never publish arbitrary source column names as metrics.
    scope_column = ["Запрошенный артикул"]
    layouts = (["Сравнение", "Периоды", "Продажи тек./пред.", "Δ продаж", "Выручка тек./пред.", "Δ выручки"],
               ["Период", "Продажи, шт", "Выручка, ₽"])
    if snapshot and any(snapshot.columns in (layout, scope_column + layout) for layout in layouts):
        lines += _table(snapshot.columns, snapshot.rows) if snapshot.rows else ["Сводка по периодам отсутствует; это не означает нулевые продажи."]
    else:
        lines += ["Сводка по периодам недоступна в поддерживаемом формате; значения не восстановлены предположениями."]
    lines += ["", "## Изменения по товарам", ""]
    groups = defaultdict(list)
    request = prepared["request"]
    default_scope = request["targets"][0] if request["scope"] == "single_sku" else "портфель"
    decomposition = result.sales_decomposition
    required = ["Артикул", "current_revenue", "previous_revenue", "delta_revenue", "current_sales",
                "previous_sales", "delta_sales", "Текущий период", "Предыдущий период", "delta_revenue_pct", "driver_role"]
    if decomposition and decomposition.columns in (required, scope_column + required):
        for row in decomposition.rows:
            if len(row) != len(decomposition.columns):
                raise ValueError("Sales report row width mismatch")
            item = dict(zip(decomposition.columns, row))
            key = (item.get("Запрошенный артикул", default_scope), item["Текущий период"], item["Предыдущий период"])
            groups[key].append(item)
    for (scope, current, previous), rows in groups.items():
        lines += [f"Область: {_cell(scope)}. Текущий период: {_cell(current)}; предыдущий: {_cell(previous)}.", ""]
        for title, sign in (("Наибольшее снижение среди полученных строк", -1), ("Наибольший рост среди полученных строк", 1)):
            selected = [r for r in rows if isinstance(r["delta_revenue"], (int, float)) and sign * r["delta_revenue"] > 0]
            selected.sort(key=lambda r: sign * r["delta_revenue"], reverse=True)
            lines += [title + ":", ""]
            if not selected:
                lines += ["Нет строк с подтверждённым изменением этого направления; это не доказывает отсутствие изменений вне выборки.", ""]
                continue
            lines += _table(["SKU", "Выручка текущая, ₽", "Выручка предыдущая, ₽", "Δ выручки, ₽", "Δ выручки, %", "Продажи текущие, шт", "Продажи предыдущие, шт", "Δ продаж, шт"],
                            [[r[k] for k in ("Артикул", "current_revenue", "previous_revenue", "delta_revenue", "delta_revenue_pct", "current_sales", "previous_sales", "delta_sales")] for r in selected[:5]])
            lines += [""]
    if not groups:
        lines += ["Сопоставимые изменения SKU отсутствуют; отсутствующие строки не трактуются как нули.", ""]
    requested_scopes = targeted_scopes(prepared)
    if prepared.get("scoped_sales_policy_version") in (1, 2, 3):
        from commerce_agent.sales.scoped import scoped_scopes
        available = {scope["target"] for scope in scoped_scopes(prepared)}
        missing = [target for target in request["targets"] if target not in available]
        if missing:
            lines += ["## Недоступные запрошенные товары", ""]
            for target in missing:
                lines += [f"- SKU {_cell(target)}: данные этой области недоступны; это не означает нулевые продажи."]
            lines += [""]
    if prepared.get("targeted_sales_policy_version") in (1,2):
        lines += ["## Отдельно запрошенные товары", "",
                  "Следующие строки относятся к отдельным запросам SKU, а не к дополнительному портфелю. "
                  "Их нельзя прибавлять к портфельным показателям или объединять между областями.", ""]
        by_target = {scope["target"]:scope for scope in requested_scopes}
        for target in prepared["request"]["targets"]:
            lines += [f"### Запрошенный SKU {_cell(target)}", ""]
            if target not in by_target:
                lines += ["Данные этой области недоступны; это не означает нулевые продажи.", ""]
                continue
            scope = by_target[target]
            lines += [f"Период: {_cell(scope['period']['date_from'])} — {_cell(scope['period']['date_to'])}.", "",
                      "Каждая строка сохранена отдельно из источника; одинаковые артикулы не сложены.", ""]
            table = scope["tables"]["sales_decomposition"]
            lines += _table(table["columns"],table["rows"])
            lines += [""]
    from commerce_agent.sales.daily import daily_scopes,FIELDS,FLAGS
    daily_sources=daily_scopes(prepared)
    if prepared.get("daily_sales_policy_version")==1:
        lines += ["## Дневные наблюдения", "",
            "Дни взяты из отдельного price/sales reader, а не восстановлены из недельных сумм. "
            "Заказы без отмен, реализованные продажи и заказы воронки — разные показатели. "
            "Неполные дни не названы полными; null не заменён нулём. "
            "Наблюдаемые цены не доказывают влияние цены на продажи; шкалы conversion сохранены нативными.", ""]
        by_target={s["target"]:s for s in daily_sources}
        for target in prepared["request"]["targets"]:
            lines += [f"### Дневные данные запрошенного SKU {_cell(target)}", ""]
            if target not in by_target:
                lines += ["Дневные данные недоступны в поддерживаемом scope/window; это не нулевые продажи.", ""]
                continue
            rows=by_target[target]["native_daily"]["rows"]
            for family,title in (("sales","Продажи и заказы"),("price","Наблюдаемый ценовой контекст"),("funnel","Воронка")):
                fields=[(key,label,unit) for key,label,unit,_,kind in FIELDS if kind==family]
                lines += [title+":", ""]
                keys=["report_date","vendor_code","nm_id",*(key for key,_,_ in fields)]
                labels=["День","SKU","nm_id",*(label+", "+unit for _,label,unit in fields)]
                if family=="price":keys.append("price_basis");labels.append("price_basis")
                lines += _table(labels,[[row[key] for key in keys] for row in rows])
                lines += [""]
            lines += ["Полнота отдельных дней:", "",*_table(["День","SKU","nm_id",*FLAGS],
                [[row[key] for key in ("report_date","vendor_code","nm_id",*FLAGS)] for row in rows]),""]
    lines += ["Показаны до пяти строк каждого направления в каждой области и паре периодов. "
              "Это не полный вклад товаров в изменение портфеля. Проценты и разницы взяты из сервиса; "
              "при нулевой предыдущей базе процент может отсутствовать, а высокая относительная динамика не доказывает большую абсолютную значимость.", "",
              "## Доказательность и ограничения", ""]
    sources = sorted({(e["source"], e.get("period"), e.get("freshness")) for e in prepared["evidence"]}, key=str)
    for source, period, freshness in sources:
        lines += [f"- Источник: {_cell(source)}; период: {_cell(period)}; актуальность: {_cell(freshness)}."]
    dates = sorted({str(f.source_snapshot_at) for f in result.facts if f.source_snapshot_at is not None})
    if dates:
        lines += ["- Даты исходных снимков: " + ", ".join(_cell(d) for d in dates) + "."]
    lines += ["- Полнота источника не подтверждена. Актуальность unknown не означает, что данные свежие.",
              "- Сводка отражает последние доступные данные; при запросе исторического периода она может отличаться от окна сравнения SKU.",
              "- Нельзя складывать пересекающиеся области или сравнивать рейтинги из разных окон как единый портфель."]
    if any(w.startswith("needs_data:") for w in prepared["warnings"]):
        lines += ["- Часть запрошенных данных недоступна; выводы по отсутствующим областям не сделаны."]
    if any(w.startswith("coverage:") for w in prepared["warnings"]):
        lines += ["- Получена ограниченная выборка товаров; полный портфель и товары вне лимита не проверены."]
    lines += ["", "## Аналитические наблюдения Sales", ""]
    for finding in result.findings:
        lines += [f"### {_cell(finding.factor)}", "", _cell(finding.observation), "",
                  "Следующая проверка: " + _cell(finding.action), ""]
    if not result.findings:
        lines += ["В этом запуске отдельные аналитические наблюдения не опубликованы.", ""]
    if result.cross_agent_requests:
        lines += ["## Запросы соседним агентам", ""]
        for request in result.cross_agent_requests:
            lines += [f"- {_cell(request.target_agent.value)}: {_cell(request.question)}"]
        lines += ["", "Это предложения на проверку, а не уже выполненные вызовы соседних агентов.", ""]
    lines += ["", "## Выводы и следующие проверки", "",
              "- Таблицы подтверждают размеры и направления изменений только в указанных окнах и областях.",
              "- Начать проверку товаров из таблицы снижения: подтвердить полноту выгрузки и сопоставимость периодов.",
              "- Причины изменений не установлены. Проверить рекламу с Marketing, остатки с Supply, цены с Price и прибыль с Finance; это запросы на проверку, не подтверждённые причины.",
              "- Изменение цен, рекламы или поставки по этому отчёту не выполняется. Для решения нужны междоменные evidence и отдельное согласование."]
    report = "\n".join(lines)
    from commerce_agent.sales.historical_funnel import render as render_historical
    historical = render_historical(prepared)
    if historical:
        report += "\n\n## Историческая дневная воронка\n\n" + historical
    assert_bounded_safe(report, max_bytes=MAX_RESULT_BYTES)
    return report
