"""Wire v2: free planning before reads, evidence review after reads.

This adapter fills protocol metadata, never business calculations or assessments.
Wire v1 remains available to the same reader without reinterpretation.
"""
from copy import deepcopy
import re
from typing import Literal

from pydantic import Field, ValidationError

from commerce_agent.marketing import compact, research
from commerce_agent.marketing.policy import assert_bounded_safe
from commerce_agent.marketing.research_evidence import observation_index


class PlanHypothesis(compact.Closed):
    id: str = Field(min_length=1, max_length=40)
    explanation: str = Field(min_length=8, max_length=600)
    priority: Literal["high", "medium", "low"]
    price_focus: Literal["none", "seller", "customer", "competitors", "market_summary"]
    expected_observation: str = Field(min_length=8, max_length=800)
    requested_checks: list[research.RequestedCheck] = Field(max_length=5)


class PlanDecision(compact.Closed):
    hypotheses: list[PlanHypothesis] = Field(min_length=1, max_length=4)


class ReviewHypothesis(research.Hypothesis):
    observations: list[str] = Field(max_length=12, description="Select existing observation IDs from evidence.observations only.")


class ReviewDecision(compact.Closed):
    hypotheses: list[ReviewHypothesis] = Field(min_length=1, max_length=4)
    retired_hypotheses: list[compact.Retired] = Field(max_length=4)
    finished: bool


# Only static codes/instructions and schema field paths may enter corrections.
INSTRUCTIONS = {
    "schema_invalid": "Верни JSON текущей фазы строго по схеме. До чтений нужны только гипотезы и выбранные проверки, без оценки и ссылок на доказательства.",
    "plan_requires_checks": "Выбери хотя бы одну доступную проверку для диагностических гипотез.",
    "research_evidence_invalid": "В evidence используй только существующие baseline, context и receipt refs из входа.",
    "research_observation_unbound": "Выбери существующий observation ID и включи его receipt в evidence этой гипотезы.",
    "research_layer_mismatch": "Наблюдение должно относиться к price_focus этой версии. Разные ценовые слои требуют отдельных версий.",
    "research_support_without_observation": "Усиление или ослабление требует нового доступного наблюдения, связанного с этой версией. Иначе unchanged и конкретное ограничение.",
    "research_tool_disabled": "Используй только инструменты из available_checks.",
    "research_round_budget": "Новые чтения больше недоступны. Заверши пересмотр на имеющихся данных и укажи непроверенное.",
    "research_read_budget": "Сократи выбор до оставшегося бюджета уникальных проверок; уже полученные ответы используй повторно.",
    "research_journal_budget": "Сократи пояснения, сохранив существенные выводы и ограничения. Исходные данные и ответы источников сокращать нельзя.",
    "research_completion_invalid": "При новых проверках finished=false, без новых проверок finished=true.",
    "research_final_unassessed": "Финальную непроверенную версию пометь unchanged с конкретным limitation.",
    "research_lifecycle_invalid": "Сохрани активные id; исчезнувшие версии явно сними через retired_hypotheses. Не повторяй и не возвращай снятые id.",
    "research_hypothesis_id_invalid": "Используй уникальный id: латинская буква, затем латинские буквы, цифры или подчёркивание, до сорока символов.",
    "research_prose_contains_numbers": "Объяснения должны быть качественными, без цифр, SKU и дат. Числа публикует код.",
    "contract_invalid": "Исправь ответ по текущей схеме и доступным данным. Не выдумывай доказательства.",
}
FIELDS = set(PlanHypothesis.model_fields) | set(ReviewHypothesis.model_fields) | set(ReviewDecision.model_fields) | {"tool", "purpose", "reason"}


class ContractError(ValueError):
    def __init__(self, code, path="$"):
        self.issues = [{"code": code if code in INSTRUCTIONS else "contract_invalid", "path": path}]
        super().__init__(self.issues[0]["code"])


def validate_issues(issues):
    if not isinstance(issues, list) or not 1 <= len(issues) <= 8:
        raise ValueError("research_issues_invalid")
    for issue in issues:
        if not isinstance(issue, dict) or set(issue) != {"code", "path"} or issue["code"] not in INSTRUCTIONS:
            raise ValueError("research_issues_invalid")
        path = issue["path"]
        if not isinstance(path, str) or len(path) > 200 or not re.fullmatch(r"\$(?:\.[a-z_]+|\[\d{1,2}\])*", path):
            raise ValueError("research_issues_invalid")
        if any(field not in FIELDS for field in re.findall(r"\.([a-z_]+)", path)):
            raise ValueError("research_issues_invalid")


def safe_path(location):
    path = "$"
    for part in location:
        if isinstance(part, str) and part in FIELDS:
            path += "." + part
        elif type(part) is int and 0 <= part < 100:
            path += f"[{part}]"
        else:
            break  # Unknown field names can themselves contain secrets.
    return path[:200] if len(path) <= 200 else "$"


def phase(journal):
    return "review" if research.previous_decision(journal) else "plan"


def schema(prepared, reads, journal, *, final_only=False):
    planning = phase(journal) == "plan"
    value = (PlanDecision if planning else ReviewDecision).model_json_schema()
    key = "PlanHypothesis" if planning else "ReviewHypothesis"
    props = value["$defs"][key]["properties"]
    tools = list(compact.available_tools(prepared))
    final = final_only or len(journal["events"]) >= 2 or research.rounds_used(journal) >= 2
    if final or len(reads) >= 5:
        tools = [tool for tool in tools if any(r["operation"] == tool for r in reads)]
    if tools:
        value["$defs"]["RequestedCheck"]["properties"]["tool"]["enum"] = tools
    else:
        props["requested_checks"]["maxItems"] = 0
    if not planning:
        if final:
            value["properties"]["finished"]["enum"] = [True]
            props["assessment"]["enum"] = ["strengthened", "weakened", "unchanged"]
        props["evidence"]["items"]["enum"] = list(compact.evidence_catalog(prepared, reads))
        refs = list(observation_index(prepared, reads))
        if refs:
            props["observations"]["items"]["enum"] = refs
        else:
            props["observations"]["maxItems"] = 0
    return value


def normalize_wire(raw, prepared, reads, journal):
    if journal["wire_version"] != 2:
        raise ValueError("research_wire_version")
    current_phase = phase(journal)
    try:
        assert_bounded_safe(raw, max_bytes=65536)
        value = (PlanDecision if current_phase == "plan" else ReviewDecision).model_validate(raw, strict=True).model_dump(mode="json")
    except ValidationError as exc:
        error = ContractError("schema_invalid")
        error.issues = [{"code": "schema_invalid", "path": safe_path(item["loc"])}
            for item in exc.errors(include_input=False, include_context=False, include_url=False)[:8]]
        raise error from None
    except (ValueError, TypeError):
        raise ContractError("contract_invalid") from None
    canonical = deepcopy(value)
    if current_phase == "plan":
        if not any(h["requested_checks"] for h in value["hypotheses"]):
            raise ContractError("plan_requires_checks", "$.hypotheses")
        for h in canonical["hypotheses"]:
            h.update(assessment="untested", evidence=["baseline"], observations=[],
                observation="Исходные факты доступны; дополнительные проверки ещё не выполнены.",
                revision="Гипотеза предложена для проверки; её поддержка пока не оценена.",
                limitation="Ожидаются ответы выбранных источников; причина пока не установлена.")
        canonical.update(retired_hypotheses=[], finished=False)
    else:
        index = observation_index(prepared, reads)
        for n, h in enumerate(canonical["hypotheses"]):
            expanded = []
            for ref in h["observations"]:
                item = index.get(ref)
                if item is None:
                    raise ContractError("research_observation_unbound", f"$.hypotheses[{n}].observations")
                expanded.append(dict(observation_ref=ref, native_comparison_ref=item["native_comparison_ref"], asserted_relation=item["relation"]))
            h["observations"] = expanded
    try:
        event = research.normalize(canonical, prepared, reads, journal)
    except (ValueError, TypeError, KeyError) as exc:
        code = str(exc)
        raise ContractError(code if code in INSTRUCTIONS else "contract_invalid") from None
    event.update(wire_phase=current_phase, wire_response=deepcopy(raw))
    try:
        # Reserve room for bounded rejection events. Never truncate native evidence.
        assert_bounded_safe({**journal, "events": [*journal["events"], event]}, max_bytes=262144-4096)
    except ValueError:
        raise ContractError("research_journal_budget") from None
    return event


def planning_exhausted(journal):
    # A third initial plan would consume the only decision left for evidence review.
    return phase(journal) == "plan" and len(journal["events"]) >= 2
