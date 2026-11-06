"""V6 final-response selection. No first-block JSON fallback or repair heuristics."""
import json
from langchain_core.messages import AIMessage, AIMessageChunk
from commerce_agent.marketing.policy import assert_bounded_safe


class ResponseRejected(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ResponseRejected("duplicate_json_key")
        result[key] = value
    return result


def select_final(message):
    """Return a strict object plus selection metadata; never persist exception text."""
    if (not isinstance(message, AIMessage) or isinstance(message, AIMessageChunk)
            or message.response_metadata.get("status") != "completed"
            or message.response_metadata.get("incomplete_details")
            or message.additional_kwargs.get("refusal") or message.tool_calls or message.invalid_tool_calls):
        raise ResponseRejected("response_not_completed")
    blocks = message.content
    if isinstance(blocks, str):
        blocks = [{"type": "text", "text": blocks}]
    if not isinstance(blocks, list) or any(not isinstance(b, dict) for b in blocks):
        raise ResponseRejected("response_blocks_invalid")
    if any(b.get("type") in {"refusal", "tool_call", "function_call"} for b in blocks):
        raise ResponseRejected("response_refused_or_tool")
    texts = [(i, b) for i, b in enumerate(blocks) if b.get("type") in {"text", "output_text"}]
    finals = [(i, b) for i, b in texts if b.get("phase") == "final_answer"]
    if finals:
        candidates = finals
    elif len(texts) == 1 and texts[0][1].get("phase") is None:
        candidates = texts
    else:
        raise ResponseRejected("final_answer_missing")
    parsed = []
    for _, block in candidates:
        try:
            value = json.loads(block["text"], object_pairs_hook=_object,
                parse_constant=lambda _: (_ for _ in ()).throw(ResponseRejected("nonfinite_json")))
        except (KeyError, TypeError, json.JSONDecodeError):
            raise ResponseRejected("final_json_invalid") from None
        if not isinstance(value, dict):
            raise ResponseRejected("final_json_not_object")
        assert_bounded_safe(value, max_bytes=65536)
        parsed.append(value)
    canonical = lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len({canonical(v) for v in parsed}) != 1:
        raise ResponseRejected("ambiguous_final")
    return parsed[0], dict(policy_version=1, selected_index=candidates[0][0],
        identical_finals=len(candidates), ignored_text_blocks=len(texts)-len(candidates))
