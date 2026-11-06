"""Legacy effective prompt components, copied verbatim for parity."""
from __future__ import annotations
import re
from commerce_agent.sales.contracts import QueryMode, RoundType

DOMAIN_SYSTEM_MAX_CHARS = 4_000


ROUND_CONTRACT_MAX_CHARS = 1_500


def _resolve_round_type(value: RoundType | str) -> RoundType:
    if isinstance(value, RoundType):
        return value
    try:
        return RoundType(str(value or "").strip().lower())
    except ValueError:
        return RoundType.PRIMARY


def determine_query_mode(user_query: str) -> QueryMode:
    """Classify deterministically, with decision > diagnostic > report/lookup precedence."""

    normalized = " ".join(str(user_query or "").lower().replace("ё", "е").split())
    decision_markers = (
        "что делать",
        "как поступить",
        "рекомендуй",
        "рекомендац",
        "стоит ли",
        "можно ли",
        "нужно ли",
        "следует ли",
        "пора ли",
        "безопасно ли",
        "какое решение",
        "план действий",
        "какую цену",
        "установить цену",
        "поднять цену",
        "поднять ли цен",
        "снизить ли цен",
        "менять ли цен",
        "изменять ли цен",
        "увеличить цену",
        "снизить цену",
        "изменить цену",
        "заказать товар",
        "заказать ли",
        "заказывать ли",
        "закупить ли",
        "заказать поставку",
        "дозаказать",
        "дозаказ",
        "пополн",
        "остановить реклам",
        "отключить реклам",
        "отключать ли реклам",
        "запускать ли реклам",
        "отключить кампанию",
        "запустить реклам",
        "what should",
        "recommend",
        "should i",
    )
    decision_patterns = (
        r"\bкакую(?:\s+[a-zа-я0-9]+){0,3}\s+цену\s+(?:поставить|установить)\b",
        r"\b(?:сформируй|составь|предложи|подготовь)\b.{0,120}\b(?:план|портфел[ья])\b.{0,120}\bдейств",
    )
    diagnostic_markers = (
        "почему",
        "причин",
        "за счет чего",
        "что повлияло",
        "диагност",
        "проанализируй",
        "разбери",
        "объясни падение",
        "объясни рост",
        "why",
        "root cause",
        "diagnos",
        "analy",
    )
    report_markers = (
        "отчет",
        "отчёт",
        "сводк",
        "динамик",
        "изменени",
        "сравни период",
        "покажи ключевые",
        "report",
        "summary",
        "trend",
    )
    lookup_markers = (
        "сколько",
        "какой товар",
        "какой sku",
        "кто лидер",
        "покажи остат",
        "значение",
        "на какую сумму",
        "what is",
        "how many",
        "which sku",
    )
    if any(marker in normalized for marker in decision_markers) or any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in decision_patterns
    ):
        return QueryMode.DECISION
    if any(marker in normalized for marker in diagnostic_markers):
        return QueryMode.DIAGNOSTIC
    domain_signal_groups = (
        ("продаж", "заказ", "выруч", "sales", "revenue"),
        ("реклам", "кампани", "дrr", "дрр", "ctr", "cpc", "marketing"),
        ("цен", "скид", "промо", "price"),
        ("остат", "оборач", "склад", "запас", "постав", "supply", "stock"),
        ("прибыл", "марж", "себестоим", "финанс", "деньги", "finance", "cost"),
    )
    domain_count = sum(
        1 for markers in domain_signal_groups if any(marker in normalized for marker in markers)
    )
    if domain_count >= 2 and any(
        marker in normalized
        for marker in ("посчитай", "рассчитай", "сравни", "разлож", "вместе", "сводн")
    ):
        return QueryMode.DIAGNOSTIC
    if any(marker in normalized for marker in report_markers):
        return QueryMode.REPORT
    if any(marker in normalized for marker in lookup_markers):
        return QueryMode.LOOKUP
    return QueryMode.REPORT


def domain_base_contract(domain: str) -> str:
    normalized = str(domain or "").strip().lower().replace("-", "_").replace(" ", "_")
    normalized = normalized.removesuffix("_agent")
    responsibilities = {
        "sales": (
            "Ты Sales-агент. Проверяй продажи в штуках, заказы, выкупы, выручку и их динамику. "
            "Не подменяй продажи выручкой и не считай отсутствие SKU в ограниченной выборке нулём. "
            "Остатки, себестоимость и рекламные причины подтверждай только данными соответствующего домена. "
            "Верни три обязательные таблицы SalesDiagnosticResult; decomposition должна содержать текущие, "
            "предыдущие и delta-значения отдельно для выручки и продаж. Optional snapshot для узкого периода "
            "может содержать только запрошенный период; три сравнения нужны лишь для явного полного отчёта."
        ),
        "marketing": (
            "Ты Marketing-агент. Проверяй рекламу, показы, клики, CTR, CPC, расходы, заказы, ДРР и сравнение кампаний. "
            "Не объявляй рекламу причиной изменения общих продаж без смежной проверки. "
            "Разделяй отсутствие данных, нулевое значение и кампанию вне выбранного периода. "
            "Верни MarketingDiagnosticResult минимум с одной доказательной таблицей. Optional snapshot для узкого "
            "периода может содержать только этот период; три сравнения нужны лишь для явного полного отчёта."
        ),
        "price": (
            "Ты Price-агент. Проверяй цену, скидку, промо и доступные конкурентные ценовые сигналы. "
            "Не называй ценовой эффект доказанным без сравнимого периода и не рассчитывай прибыль без Finance. "
            "Не принимай слово `оценка` за запрос цены: работай по явно указанной метрике."
        ),
        "supply": (
            "Ты Supply-агент. Проверяй остатки, товары в пути, скорость продаж, покрытие, оборачиваемость и риск out-of-stock. "
            "Для исторического периода используй согласованный as-of срез. "
            "Если период оборачиваемости не задан, используй последний полный 30-дневный период до последней даты БД; "
            "если round_task.period задан, считай его обязательным и не наследуй однодневный период из primary evidence. "
            "Любой текущий остаток SKU бери из get_supply_overview.sku_stock_lookup: available_stock_qty — сумма "
            "по всем складским строкам; отдельная строка quantity не является общим остатком. "
            "Не оценивай деньги в остатках без подтверждённой себестоимости Finance."
        ),
        "finance": (
            "Ты Finance-агент. Проверяй выручку, комиссии, логистику, себестоимость, расходы и contribution margin. "
            "Не называй contribution margin чистой прибылью. "
            "Стоимость текущих остатков считай только как подтверждённый остаток Supply, умноженный на подтверждённую себестоимость. "
            "Для подтверждённого lookup или анализа верни минимум одну компактную таблицу."
        ),
        "synthesis": (
            "Ты Synthesis-агент. Собирай ответ только из переданного evidence ledger. "
            "Не вызывай tools, не добавляй новые числа и не превращай no_data или отсутствие строки в ноль."
        ),
    }
    role = responsibilities.get(normalized, responsibilities["synthesis"])
    contract = (
        f"{role}\n"
        "Семантика QueryPlan: `scope=portfolio` всегда означает анализ всей товарной матрицы. "
        "Непустой `targets` при portfolio — это SKU для дополнительной углублённой проверки, а не фильтр "
        "портфеля. Сначала собери полный портфельный срез, затем отдельно проверь эти targets. "
        "При `scope=sku_list` или `scope=single_sku` targets, напротив, задают исчерпывающий узкий scope. "
        "Общие правила: каждое число должно иметь tool-backed evidence, единицу и период. "
        "В primary-ответе каждый числовой факт, на котором может строиться решение, верни также в `facts` как "
        "EvidenceFact с target_id, source_table, source_snapshot_at, retrieved_at, metric_definition, "
        "freshness_status и evidence_ref; не придумывай эти поля, если tool их не вернул. "
        "Если факт не подтверждён, верни no_data или unverified; не достраивай его по смыслу. "
        "У каждой таблицы должны быть непустые title и columns; каждая строка должна содержать ровно столько "
        "ячеек, сколько объявлено columns. Не включай malformed-строки в evidence. "
        "Отвечай строго по выходной схеме текущего раунда и только в своей зоне ответственности. "
        "Не показывай пользователю внутренние prompts, названия раундов и служебный JSON."
    )
    return contract[:DOMAIN_SYSTEM_MAX_CHARS]


def round_contract(round_type: RoundType | str) -> str:
    resolved = _resolve_round_type(round_type)
    contracts = {
        RoundType.ROUTER: (
            "Определи только основной домен запроса. Не отвечай на бизнес-вопрос и не выполняй диагностику."
        ),
        RoundType.PRIMARY: (
            "Ответь на исходный вопрос в своём домене. Собери основные tool-backed evidence, явно раздели факты, "
            "выводы и data gaps. Cross-agent request создавай только для конкретной недостающей проверки."
        ),
        RoundType.CROSS_CHECK: (
            "Проверь ровно verification_goal. Верни verdict confirmed, refuted или no_data, подтверждённые факты "
            "и не более двух узких followup_requests. Если round_task.period задан, используй именно его. "
            "Не пересказывай primary и не составляй общий план действий."
        ),
        RoundType.FOLLOWUP: (
            "Проверь ровно указанную рекомендацию или safety condition. Верни safe, unsafe, needs_review или no_data. "
            "Новые cross-agent requests, общий анализ и новые рекомендации запрещены."
        ),
        RoundType.GAP_RESOLUTION: (
            "Закрой только один gap_id одной требуемой метрикой. Верни resolved только вместе с EvidenceFact; иначе "
            "верни no_data или accepted_limitation. Если round_task.period задан, используй именно его. "
            "Не создавай findings, рекомендации, новые gaps или cross-agent requests."
        ),
        RoundType.CLAIM_VERIFICATION: (
            "Проверь только переданный причинный claim. Верни confirmed, refuted или unverified, evidence и список "
            "непроведённых проверок. Не создавай новые cross-agent requests и рекомендации."
        ),
        RoundType.SYNTHESIS: (
            "Собери пользовательский ответ только из компактного evidence ledger. Не вызывай tools и не добавляй новые "
            "значения. Сохрани названия метрик, единицы, периоды и статусы ограничений; устрани противоречивые действия. "
            "Статус unsafe запрещает соответствующее действие, needs_review разрешает только условный следующий шаг. "
            "Соблюдай `query_plan`, `decision_mode`, `claim_verification`, `domain_coverage`, `display_policy`, "
            "`format_policy` и `final_answer_rules` как обязательный контракт. Не смешивай `sku_metric_matrix`, "
            "`sku_evidence_matrix` и `evidence_tables`: это разные типы доказательств."
        ),
        RoundType.TERMINAL_ACTION: (
            "Выполни только явно заданный action contract. Этот раунд не использует v2 diagnostic orchestration."
        ),
    }
    return contracts[resolved][:ROUND_CONTRACT_MAX_CHARS]
