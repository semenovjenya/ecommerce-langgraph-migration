"""Private wire3: model chooses hypotheses/observations; code owns metadata.

Canonical wire1/2 decisions and replay are not reinterpreted. Source layers stay
on individual observations, including hypotheses comparing price with sales.
"""
from copy import deepcopy
import re
from typing import Literal

from pydantic import Field, ValidationError

from commerce_agent.marketing import compact, research, research_phases
from commerce_agent.marketing.policy import assert_bounded_safe
from commerce_agent.marketing.research_evidence import observation_index


class PlanHypothesis(compact.Closed):
    id: str = Field(min_length=1, max_length=40)
    explanation: str = Field(min_length=8, max_length=600)
    priority: Literal["high", "medium", "low"]
    observation: str = Field(min_length=8, max_length=1000)
    expected_observation: str = Field(min_length=8, max_length=800)
    limitation: str = Field(min_length=8, max_length=800)


class ReviewHypothesis(PlanHypothesis):
    assessment: Literal["strengthened", "weakened", "unchanged"]
    revision: str = Field(min_length=8, max_length=1000)
    observations: list[str] = Field(max_length=12)


class Check(research.RequestedCheck):
    hypothesis_ids: list[str] = Field(min_length=1, max_length=4)


class Plan(compact.Closed):
    hypotheses: list[PlanHypothesis] = Field(min_length=1, max_length=4)
    checks: list[Check] = Field(min_length=1, max_length=5)


class Review(compact.Closed):
    hypotheses: list[ReviewHypothesis] = Field(min_length=1, max_length=4)
    retired_hypotheses: list[compact.Retired] = Field(max_length=4)
    checks: list[Check] = Field(max_length=5)


INSTRUCTIONS = {
    **research_phases.INSTRUCTIONS,
    "schema_invalid": "Верни ответ по схеме текущей фазы. Служебные evidence, сравнения, ценовые слои и finished заполняет код.",
    "research_check_binding": "Каждая проверка должна ссылаться на уникальные активные id гипотез текущего ответа.",
    "research_duplicate_check": "Укажи инструмент один раз в общем checks и перечисли все связанные hypothesis_ids.",
    "research_observation_unbound": "Выбирай только существующие observation ID из evidence.observations. Источник и связь с гипотезой добавит код.",
    "research_lifecycle_invalid": "Сохрани все active_hypothesis_ids либо явно сними их через retired_hypotheses. Исправь отклонённый draft, не теряя остальные версии.",
}
FIELDS = set(PlanHypothesis.model_fields) | set(ReviewHypothesis.model_fields) | set(Review.model_fields) | set(Check.model_fields) | {"reason"}


class ContractError(research_phases.ContractError):
    def __init__(self, code, path="$"):
        self.issues = [{"code": code if code in INSTRUCTIONS else "contract_invalid", "path": path}]
        ValueError.__init__(self, self.issues[0]["code"])


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
            break
    return path if len(path) <= 200 else "$"


phase = research_phases.phase
planning_exhausted = research_phases.planning_exhausted


def model_type(journal):
    return Plan if phase(journal) == "plan" else Review


def safe_draft(raw, journal):
    """Only a bounded closed wire shape, never exception input/unknown keys."""
    try:
        value = model_type(journal).model_validate(raw, strict=True).model_dump(mode="json")
        assert_bounded_safe(value, max_bytes=16384)
        return value
    except (ValueError, TypeError):
        return None


def schema(prepared, reads, journal, *, final_only=False):
    value = model_type(journal).model_json_schema()
    tools = list(compact.available_tools(prepared))
    final = final_only or len(journal["events"]) >= 2 or research.rounds_used(journal) >= 2
    if phase(journal) == "plan" and final:
        raise ContractError("research_round_budget")
    if final or len(reads) >= 5:
        tools = [tool for tool in tools if any(r["operation"] == tool for r in reads)]
    if tools:
        value["$defs"]["Check"]["properties"]["tool"]["enum"] = tools
    else:
        value["properties"]["checks"]["maxItems"] = 0
    if phase(journal) == "review":
        props = value["$defs"]["ReviewHypothesis"]["properties"]
        refs = observation_index(prepared, reads)
        if refs:
            props["observations"]["items"]["enum"] = list(refs)
        else:
            props["observations"]["maxItems"] = 0
        if not any(item["usable"] and item["receipt"].startswith("r") for item in refs.values()):
            props["assessment"]["enum"] = ["unchanged"]
    return value


def normalize_wire(raw, prepared, reads, journal):
    if journal["wire_version"] != 3:
        raise ValueError("research_wire_version")
    try:
        assert_bounded_safe(raw, max_bytes=65536)
        value = model_type(journal).model_validate(raw, strict=True).model_dump(mode="json")
    except ValidationError as exc:
        error = ContractError("schema_invalid")
        error.issues = [{"code": "schema_invalid", "path": safe_path(item["loc"])}
            for item in exc.errors(include_input=False, include_context=False, include_url=False)[:8]]
        raise error from None
    except (ValueError, TypeError):
        raise ContractError("contract_invalid") from None
    planning = phase(journal) == "plan"
    active = [h["id"] for h in value["hypotheses"]]
    for n, identity in enumerate(active):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", identity) or active.count(identity) != 1:
            raise ContractError("research_hypothesis_id_invalid", f"$.hypotheses[{n}].id")
    checks = {}
    for n, check in enumerate(value["checks"]):
        if check["tool"] in checks:
            raise ContractError("research_duplicate_check", f"$.checks[{n}].tool")
        ids = check["hypothesis_ids"]
        if len(ids) != len(set(ids)) or not set(ids) <= set(active):
            raise ContractError("research_check_binding", f"$.checks[{n}].hypothesis_ids")
        checks[check["tool"]] = check
    index = observation_index(prepared, reads)
    source_metadata = {}
    canonical = dict(hypotheses=[], retired_hypotheses=value.get("retired_hypotheses", []),
        finished=not any(tool not in {r["operation"] for r in reads} for tool in checks))
    for n, source in enumerate(value["hypotheses"]):
        h = deepcopy(source)
        refs = h.pop("observations", [])
        observations, evidence, layers = [], [], set()
        requested = {tool: dict(tool=tool, purpose=c["purpose"]) for tool, c in checks.items() if h["id"] in c["hypothesis_ids"]}
        for ref in refs:
            item = index.get(ref)
            if item is None or ref in source_metadata.get(h["id"], {}):
                raise ContractError("research_observation_unbound", f"$.hypotheses[{n}].observations")
            source_metadata.setdefault(h["id"], {})[ref] = deepcopy(item)
            observations.append(dict(observation_ref=ref, native_comparison_ref=item["native_comparison_ref"], asserted_relation=item["relation"]))
            evidence.append(item["receipt"])
            layers.add(item["layer"])
            if item["receipt"].startswith("r"):
                read = reads[int(item["receipt"][1:])]
                requested.setdefault(read["operation"], dict(tool=read["operation"], purpose="Использовать уже полученное наблюдение источника"))
        h.update(evidence=list(dict.fromkeys(evidence)) or ["baseline"], observations=observations,
            price_focus=next(iter(layers)) if len(layers) == 1 else "none", requested_checks=list(requested.values()))
        if planning:
            h.update(assessment="untested", revision="Гипотеза предложена; дополнительные проверки ещё не выполнены.")
        canonical["hypotheses"].append(h)
    # Return exact hypothesis paths for actionable corrections, while the shared
    # canonical validator still owns read identity/budgets/support/lifecycle.
    for n, h in enumerate(canonical["hypotheses"]):
        if any(re.search(r"\d", h[k]) for k in ("explanation", "observation", "revision", "limitation", "expected_observation")):
            raise ContractError("research_prose_contains_numbers", f"$.hypotheses[{n}]")
        if h["assessment"] in {"strengthened", "weakened"} and not any(
                index[o["observation_ref"]]["usable"] and index[o["observation_ref"]]["receipt"].startswith("r") for o in h["observations"]):
            raise ContractError("research_support_without_observation", f"$.hypotheses[{n}].observations")
    try:
        event = research.normalize(canonical, prepared, reads, journal)
    except (ValueError, TypeError, KeyError) as exc:
        raise ContractError(str(exc)) from None
    event.update(wire_phase=phase(journal), wire_response=deepcopy(raw), source_metadata=source_metadata)
    try:
        assert_bounded_safe({**journal, "events": [*journal["events"], event]}, max_bytes=262144-4096)
    except ValueError:
        raise ContractError("research_journal_budget") from None
    return event
