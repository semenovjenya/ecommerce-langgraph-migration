"""Versioned native-only checkpoints; derived caches are local, not compressed.

Storage format is independent of analytical policy1/2/3. No implicit fallback
from missing derived fields: marker loss must fail closed, not become legacy.
"""
from copy import deepcopy

from commerce_agent.marketing.composite import canonical
from commerce_agent.marketing.policy import assert_bounded_safe

VERSION_KEY="catalog_storage_version"
VERSION=1
PREPARED_BYTES=262144
DERIVED_KEYS={"claims","facts","tables","required_groups","eligible_check_indices","required_query_check_indices","query_check_overflow"}


def build_composite_storage(base):
    from commerce_agent.marketing.composite_analysis import base_projection,derive_composite_catalog
    # Derivation reconstructs the exact native base, validating source types,
    # request, lineage, periods, warnings and the separate ephemeral bound.
    catalog=derive_composite_catalog(base)
    stored={**deepcopy(base_projection(catalog)),VERSION_KEY:VERSION}
    assert_bounded_safe(stored,max_bytes=PREPARED_BYTES)
    return stored


def load_composite_catalog(prepared):
    from commerce_agent.marketing.composite_analysis import base_projection,derive_composite_catalog
    if not isinstance(prepared,dict):raise ValueError("Marketing composite storage invalid")
    versioned=VERSION_KEY in prepared
    if versioned and (type(prepared[VERSION_KEY]) is not int or prepared[VERSION_KEY]!=VERSION):
        raise ValueError("Marketing composite storage version unknown")
    assert_bounded_safe(prepared,max_bytes=PREPARED_BYTES)
    base=base_projection(prepared)
    catalog=derive_composite_catalog(base)
    expected={**base_projection(catalog),VERSION_KEY:VERSION} if versioned else catalog
    if canonical(expected)!=canonical(prepared):
        raise ValueError("Marketing composite storage/catalog changed")
    # This is a locally reconstructed view only. Callers must not return it
    # into graph state; publication/provider apply their own unchanged limits.
    return catalog
