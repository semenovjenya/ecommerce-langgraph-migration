"""Inspect native control payloads before source capability callbacks."""
from langgraph.types import Command, Send


def invocation_source_values(value, *, nested=False):
    """Do not flatten duplicate updates or interpret ordinary message text."""
    if isinstance(value, Command):
        for key, item in value._update_as_tuples():
            yield {key: item}
        yield from invocation_source_values(value.resume, nested=True)
        yield from invocation_source_values(value.goto, nested=True)
    elif isinstance(value, Send):
        yield from invocation_source_values(value.arg, nested=True)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from invocation_source_values(item, nested=nested)
    else:
        value = value.model_dump(mode="python") if hasattr(value, "model_dump") else value
        if isinstance(value, dict):
            yield value
            if nested:
                for key, item in value.items():
                    if key != "messages":
                        yield from invocation_source_values(item, nested=True)
