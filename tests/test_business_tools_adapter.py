from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from uuid import uuid4

import httpx
import pytest

from commerce_agent.adapters.business_tools import (
    BusinessToolClient, BusinessToolsSettings, ToolAdapterError,
)
from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.adapters.contracts import SalesSnapshot, ToolArtifactReference


CONTEXT = CommerceRuntimeContext(
    tenant_id="tenant-one", user_id="47", roles=("viewer",),
    request_id="cab2eab3-faa3-4fdc-9e2e-f613d816d673",
)
SETTINGS = BusinessToolsSettings(
    api_url="http://business.internal", service_token="synthetic-tools-token-" + "x" * 32,
)


def missing_result():
    return {
        "schema_version": 1,
        "data": {
            "report_date": None, "focus_keyword": None,
            "source_table": "sales.wb_supplier_sales", "required_fields": [],
            "calculations": {}, "rows": [], "markdown_table": "Данные продаж не найдены.",
            "report_table": {"title": "Sales-сводка", "columns": [], "rows": []},
        },
        "evidence": [], "freshness": {
            "as_of": None, "observed_at": "2026-10-10T12:00:00Z", "status": "missing",
        },
        "source": "sales.wb_supplier_sales", "period": None,
        "warnings": ["needs_data:sales_source_missing"],
    }


def test_adapter_maps_identity_and_validates_typed_result():
    def respond(request):
        assert request.url.path == "/internal/v1/agent-tools/sales/report-snapshot"
        assert request.headers["x-commerce-tenant-id"] == CONTEXT.tenant_id
        assert request.headers["x-commerce-user-id"] == "47"
        assert request.headers["x-commerce-role"] == "viewer"
        assert json.loads(request.content) == {"product_keyword": None}
        assert "authorization" not in request.headers
        return httpx.Response(200, json=missing_result())

    async def exercise():
        async with BusinessToolClient(SETTINGS, transport=httpx.MockTransport(respond)) as client:
            result = await client.sales_report_snapshot(CONTEXT)
        assert result.data.report_date is None
        assert result.warnings == ["needs_data:sales_source_missing"]
        assert SETTINGS.service_token not in repr(SETTINGS)

    asyncio.run(exercise())


@pytest.mark.parametrize("status,code,calls", [(401, "denied", 1), (403, "denied", 1),
    (422, "invalid_request", 1), (503, "unavailable", 2), (504, "unavailable", 2),
    (302, "invalid_response", 1)])
def test_safe_errors_and_only_bounded_transient_read_retry(status, code, calls):
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(status, text="lsv2_credential_like_canary_12345")

    async def exercise():
        async with BusinessToolClient(SETTINGS, transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(ToolAdapterError) as exc:
                await client.sales_report_snapshot(CONTEXT)
            assert exc.value.code == code
            assert "lsv2_" not in str(exc.value)
        assert len(seen) == calls

    asyncio.run(exercise())


def test_adapter_rejects_unsafe_response_and_request():
    seen = []

    def respond(request):
        seen.append(request)
        result = missing_result()
        result["data"]["markdown_table"] = "lsv2_credential_like_canary_12345"
        return httpx.Response(200, json=result)

    async def exercise():
        async with BusinessToolClient(SETTINGS, transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(ToolAdapterError, match="invalid_request"):
                await client.sales_report_snapshot(CONTEXT, product_keyword="lsv2_credential_like_canary_12345")
            assert seen == []
            with pytest.raises(ToolAdapterError, match="invalid_response"):
                await client.sales_report_snapshot(CONTEXT)

    asyncio.run(exercise())


def test_timeout_retries_once_and_cancellation_propagates():
    seen = []

    async def respond(request):
        seen.append(request)
        raise httpx.ReadTimeout("private connection details")

    async def cancelled(request):
        raise asyncio.CancelledError

    async def exercise():
        async with BusinessToolClient(SETTINGS, transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(ToolAdapterError, match="unavailable"):
                await client.sales_report_snapshot(CONTEXT)
        assert len(seen) == 2
        async with BusinessToolClient(SETTINGS, transport=httpx.MockTransport(cancelled)) as client:
            with pytest.raises(asyncio.CancelledError):
                await client.sales_report_snapshot(CONTEXT)

    asyncio.run(exercise())


def test_total_attempt_timeout_and_response_size_are_bounded():
    seen = []

    async def slow_response(request):
        seen.append(request)
        await asyncio.sleep(1)
        return httpx.Response(200, json=missing_result())

    async def exercise():
        settings = BusinessToolsSettings(
            api_url=SETTINGS.api_url, service_token=SETTINGS.service_token,
            timeout_seconds=0.01,
        )
        async with BusinessToolClient(settings, transport=httpx.MockTransport(slow_response)) as client:
            with pytest.raises(ToolAdapterError, match="unavailable"):
                await asyncio.wait_for(client.sales_report_snapshot(CONTEXT), timeout=0.5)
        assert len(seen) == 2
        async with BusinessToolClient(SETTINGS, transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=b"x" * 262_145)
        )) as client:
            with pytest.raises(ToolAdapterError, match="result_too_large"):
                await client.sales_report_snapshot(CONTEXT)

    asyncio.run(exercise())


def artifact_fixture():
    payload = missing_result()
    payload["data"]["markdown_table"] = "x" * 300_000
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ref = ToolArtifactReference(
        id=uuid4(), operation="sales.report_snapshot", result_schema="SalesSnapshot",
        sha256=sha256(encoded).hexdigest(), byte_size=len(encoded),
        expires_at=datetime.now(timezone.utc) + timedelta(days=30),
    )
    return ref, encoded, payload


def test_artifact_reference_has_explicit_full_payload_hydration():
    ref, encoded, payload = artifact_fixture()
    seen = []
    def respond(request):
        seen.append(request)
        if request.method == "POST":
            return httpx.Response(200, json=ref.model_dump(mode="json"))
        assert request.url.path == "/internal/v1/agent-tool-artifacts/" + str(ref.id)
        assert request.headers["accept-encoding"] == "identity"
        assert request.headers["x-commerce-user-id"] == "47"
        return httpx.Response(200, content=encoded, headers={
            "X-Artifact-SHA256": ref.sha256, "X-Artifact-Bytes": str(ref.byte_size),
        })
    async def exercise():
        async with BusinessToolClient(SETTINGS, transport=httpx.MockTransport(respond)) as client:
            reference = await client.sales_report_snapshot(CONTEXT)
            assert isinstance(reference, ToolArtifactReference) and len(seen) == 1
            result = await client.hydrate(CONTEXT, reference, result_type=SalesSnapshot)
            assert result.model_dump(mode="json", exclude_unset=True) == payload
        assert len(seen) == 2
    asyncio.run(exercise())


@pytest.mark.parametrize("mode", ["wrong_schema", "wrong_operation", "url", "too_large"])
def test_invalid_reference_rejected_in_tool_response(mode):
    ref, _, _ = artifact_fixture()
    payload = ref.model_dump(mode="json")
    if mode == "wrong_schema":
        payload["result_schema"] = "MarketingSnapshot"
    elif mode == "wrong_operation":
        payload["operation"] = "marketing.report_snapshot"
    elif mode == "url":
        payload["url"] = "https://untrusted.test"
    else:
        payload["byte_size"] = 10_485_761
    async def exercise():
        async with BusinessToolClient(SETTINGS, transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=payload)
        )) as client:
            with pytest.raises(ToolAdapterError, match="invalid_response"):
                await client.sales_report_snapshot(CONTEXT)
    asyncio.run(exercise())


@pytest.mark.parametrize("mode", ["hash", "truncated", "oversized", "schema", "secret", "pii", "pii_unicode", "headers", "compression"])
def test_artifact_hydration_rejects_corruption_and_unsafe_content(mode):
    ref, encoded, payload = artifact_fixture()
    if mode == "hash":
        encoded = encoded[:-1] + b" "
    elif mode == "truncated":
        encoded = encoded[:-1]
    elif mode == "oversized":
        encoded += b" "
    elif mode in {"schema", "secret", "pii", "pii_unicode"}:
        if mode == "schema":
            payload["data"]["rows"] = "bad shape"
        elif mode == "secret":
            payload["data"]["markdown_table"] = "lsv2_synthetic_credential_canary_12345"
        elif mode == "pii":
            payload["data"]["markdown_table"] = "person.fixture@example.com / +7 (999) 123-45-67"
        else:
            payload["data"]["markdown_table"] = "person.fixture@почта.рф / +7(999)123-45-67"
        encoded = json.dumps(payload).encode()
        ref = ref.model_copy(update={"sha256": sha256(encoded).hexdigest(), "byte_size": len(encoded)})
    headers = {"X-Artifact-SHA256": ref.sha256, "X-Artifact-Bytes": str(ref.byte_size)}
    if mode == "headers":
        headers = {}
    elif mode == "compression":
        headers["Content-Encoding"] = "identity-unexpected"
    async def exercise():
        async with BusinessToolClient(SETTINGS, transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=encoded, headers=headers)
        )) as client:
            with pytest.raises(ToolAdapterError) as error:
                await client.hydrate(CONTEXT, ref, result_type=SalesSnapshot)
            assert error.value.code == ("result_too_large" if mode == "oversized" else "invalid_response")
    asyncio.run(exercise())


@pytest.mark.parametrize("status,code,count", [(403, "denied", 1), (404, "artifact_unavailable", 1),
    (410, "artifact_unavailable", 1), (503, "unavailable", 2), (504, "unavailable", 2), (302, "invalid_response", 1)])
def test_artifact_hydration_safe_status_and_bounded_retry(status, code, count):
    ref, _, _ = artifact_fixture()
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(status, text="private internal details")
    async def exercise():
        async with BusinessToolClient(SETTINGS, transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(ToolAdapterError) as error:
                await client.hydrate(CONTEXT, ref, result_type=SalesSnapshot)
            assert error.value.code == code and str(error.value) == code
        assert len(calls) == count
    asyncio.run(exercise())


def test_artifact_hydration_expiry_timeout_cancellation_and_schema_input():
    ref, _, _ = artifact_fixture()
    calls = []
    async def timeout(request):
        calls.append(request)
        raise httpx.ReadTimeout("private network details")
    async def cancel(_):
        raise asyncio.CancelledError
    async def exercise():
        async with BusinessToolClient(SETTINGS, transport=httpx.MockTransport(timeout)) as client:
            with pytest.raises(ToolAdapterError, match="invalid_request"):
                await client.hydrate(CONTEXT, ref, result_type=ToolArtifactReference)
            with pytest.raises(ToolAdapterError, match="artifact_unavailable"):
                await client.hydrate(CONTEXT, ref.model_copy(update={"expires_at": datetime.now(timezone.utc)-timedelta(days=1)}), result_type=SalesSnapshot)
            assert calls == []
            with pytest.raises(ToolAdapterError, match="unavailable"):
                await client.hydrate(CONTEXT, ref, result_type=SalesSnapshot)
            assert len(calls) == 2
        async with BusinessToolClient(SETTINGS, transport=httpx.MockTransport(cancel)) as client:
            with pytest.raises(asyncio.CancelledError):
                await client.hydrate(CONTEXT, ref, result_type=SalesSnapshot)
    asyncio.run(exercise())


@pytest.mark.parametrize("hydrate", [False, True])
def test_slow_drip_body_obeys_absolute_attempt_deadline(hydrate):
    ref, encoded, _ = artifact_fixture()
    seen, closed = [], []
    class SlowDrip(httpx.AsyncByteStream):
        async def __aiter__(self):
            for offset in range(0, len(encoded), 100):
                await asyncio.sleep(0.005)  # Each I/O is below the timeout.
                yield encoded[offset:offset + 100]
        async def aclose(self):
            closed.append(True)
    def respond(request):
        seen.append(request)
        return httpx.Response(200, stream=SlowDrip(), headers={
            "X-Artifact-SHA256": ref.sha256, "X-Artifact-Bytes": str(ref.byte_size),
        })
    async def exercise():
        settings = BusinessToolsSettings(api_url=SETTINGS.api_url, service_token=SETTINGS.service_token,
                                         timeout_seconds=0.03)
        async with BusinessToolClient(settings, transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(ToolAdapterError, match="unavailable"):
                coroutine = client.hydrate(CONTEXT, ref, result_type=SalesSnapshot) if hydrate else client.sales_report_snapshot(CONTEXT)
                await asyncio.wait_for(coroutine, timeout=0.5)
        assert len(seen) == len(closed) == 2
    asyncio.run(exercise())


@pytest.mark.parametrize("field,value", [("byte_size", True), ("byte_size", 10_485_761), ("id", "../../untrusted")])
def test_hydration_revalidates_mutated_reference_before_http(field, value):
    ref, _, _ = artifact_fixture()
    seen = []
    def respond(request):
        seen.append(request)
        return httpx.Response(500)
    async def exercise():
        async with BusinessToolClient(SETTINGS, transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(ToolAdapterError, match="invalid_request"):
                await client.hydrate(CONTEXT, ref.model_copy(update={field: value}), result_type=SalesSnapshot)
        assert seen == []
    asyncio.run(exercise())


@pytest.mark.parametrize("pii", ["person.fixture@example.com / +7 (999) 123-45-67",
    "person.fixture@почта.рф / +7(999)123-45-67",
    "person.fixture@example.xn--p1ai / 8(999)123-45-67"])
def test_tool_request_and_inline_response_pii_rejected(pii):
    seen = []
    def respond(request):
        seen.append(request)
        payload = missing_result()
        payload["data"]["markdown_table"] = pii
        return httpx.Response(200, json=payload)
    async def exercise():
        async with BusinessToolClient(SETTINGS, transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(ToolAdapterError, match="invalid_request"):
                await client.sales_report_snapshot(CONTEXT, product_keyword=pii)
            assert seen == []
            with pytest.raises(ToolAdapterError, match="invalid_response"):
                await client.sales_report_snapshot(CONTEXT)
    asyncio.run(exercise())
