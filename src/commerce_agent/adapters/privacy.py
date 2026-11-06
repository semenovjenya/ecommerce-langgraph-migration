"""Direct-identifier projection for aggregate commerce tool payloads.

Only email and explicitly formatted international phone numbers are projected;
numeric SKU IDs, quantities and financial values are not phone heuristics.
Personal/customer records are outside the strict aggregate result schemas.
"""
import re

EMAIL = re.compile(r"(?<![\w@])[\w.+-]+@(?:[\w-]+\.)+(?:xn--[A-Za-z0-9-]{2,}|[^\W\d_]{2,})(?![\w@-])", re.IGNORECASE)
PHONE = re.compile(
    r"(?<![\w.+-])(?:\+[1-9]\d{0,2}|8(?=[ \t-]*\())[ \t-]*"
    r"(?:\(\d{1,4}\)|\d{1,4})(?:[ \t-]*\d{1,4}){1,5}"
    r"(?!\w|[.,]\d)"
)
MARKER = "[redacted-pii]"


class PrivacyError(ValueError):
    def __init__(self):
        super().__init__("personal identifiers are forbidden in tool payload")


def _is_phone(match):
    # Bounded international/direct dial identifiers, not arbitrary numeric
    # fields. Whole-token/decimal boundaries are enforced by PHONE itself.
    digits = sum(char.isdigit() for char in match.group())
    monetary_suffix = re.match(r"[ \t]*(?:[₽€$£]|руб|RUB\b|USD\b|EUR\b)",
                               match.string[match.end():], re.IGNORECASE)
    return 9 <= digits <= 15 and not monetary_suffix


def _contains(value):
    return bool(EMAIL.search(value) or any(_is_phone(match) for match in PHONE.finditer(value)))


def assert_no_direct_identifiers(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if _contains(str(key)):
                raise PrivacyError()
            assert_no_direct_identifiers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_no_direct_identifiers(item)
    elif isinstance(value, str) and _contains(value):
        raise PrivacyError()


def redact_direct_identifiers(value):
    """Return a new projection; never mutate native business results."""
    changed = False
    if isinstance(value, dict):
        projected = {}
        for key, item in value.items():
            # Do not rewrite dictionary identity keys or risk collisions.
            if _contains(str(key)):
                raise PrivacyError()
            projected[key], item_changed = redact_direct_identifiers(item)
            changed |= item_changed
        return projected, changed
    if isinstance(value, list):
        projected = []
        for item in value:
            clean, item_changed = redact_direct_identifiers(item)
            projected.append(clean)
            changed |= item_changed
        return projected, changed
    if isinstance(value, str):
        clean = PHONE.sub(lambda match: MARKER if _is_phone(match) else match.group(), EMAIL.sub(MARKER, value))
        return clean, clean != value
    return value, False

