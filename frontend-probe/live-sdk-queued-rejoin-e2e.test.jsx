// @vitest-environment jsdom
import { expect, test } from "vitest";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { useState } from "react";
import { useChannel, useStream } from "@langchain/react";
import { createCommerceTransport } from "./commerce-transport";

const apiUrl = process.env.COMMERCE_AGENT_LIVE_URL;
const bearer = process.env.COMMERCE_AGENT_LIVE_BEARER;
const authHeaders = { Authorization: `Bearer ${bearer}` };

function transport(threadId, onRunId = () => {}) {
  return createCommerceTransport(apiUrl, async (url, init = {}) => {
    const headers = new Headers(init.headers);
    headers.set("Authorization", `Bearer ${bearer}`);
    const response = await fetch(url, { ...init, headers });
    if (new URL(url).pathname.endsWith("/commands") && response.ok) {
      const payload = await response.clone().json();
      if (payload.result?.run_id) onRunId(payload.result.run_id);
    }
    return response;
  }, threadId);
}

test.skipIf(!apiUrl || !bearer)(
  "queued run rejoins before its first checkpoint and completes in the same thread",
  async () => {
    let runId;
    const firstTransport = transport(undefined, (value) => { runId = value; });
    const first = renderHook(() => {
      const [threadId, setThreadId] = useState(null);
      return useStream({ transport: firstTransport, threadId, onThreadId: setThreadId });
    });
    let threadId;
    try {
      act(() => {
        void first.result.current.submit({
          messages: [{
            type: "human", content: "safe queued rejoin probe",
            id: "live-queued-rejoin-message",
          }],
        }).catch(() => {});
      });
      await waitFor(() => {
        expect(runId).toMatch(/^[0-9a-f-]{36}$/);
        expect(first.result.current.threadId).toMatch(/^[0-9a-f-]{36}$/);
      }, { timeout: 20_000 });
      threadId = first.result.current.threadId;
      const response = await fetch(`${apiUrl}/threads/${threadId}/state`, {
        headers: authHeaders,
      });
      expect(response.status).toBe(200);
      const state = await response.json();
      expect(state.run_status).toBe("queued");
      expect(state.run_id).toBe(runId);
      expect(state).not.toHaveProperty("next");
      expect(state.tasks).toEqual([]);
    } finally {
      first.unmount();
      await firstTransport.close();
      cleanup();
    }

    const secondTransport = transport(threadId);
    const second = renderHook(() => {
      const stream = useStream({ transport: secondTransport, threadId });
      return { stream, lifecycle: useChannel(stream, ["lifecycle"], undefined, { replay: true }) };
    });
    try {
      await waitFor(() => {
        expect(second.result.current.stream.error).toBeUndefined();
        expect(second.result.current.stream.isThreadLoading).toBe(false);
        expect(second.result.current.stream.isLoading).toBe(true);
      }, { timeout: 20_000 });
      const release = await fetch(`${apiUrl.replace(/\/v2\/agent$/, "")}/__test__/release-worker`, {
        method: "POST", headers: authHeaders,
      });
      expect(release.status).toBe(200);
      await waitFor(() => {
        expect(second.result.current.lifecycle.some(
          (event) => event.params?.data?.event === "completed",
        )).toBe(true);
        expect(second.result.current.stream.isLoading).toBe(false);
        expect(second.result.current.stream.values.messages?.[0]?.content).toBe(
          "safe queued rejoin probe",
        );
      }, { timeout: 20_000 });
    } finally {
      second.unmount();
      await secondTransport.close();
      cleanup();
    }
  },
  65_000,
);
