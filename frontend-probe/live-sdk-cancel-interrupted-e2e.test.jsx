// @vitest-environment jsdom
import { expect, test } from "vitest";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { useState } from "react";
import { useStream } from "@langchain/react";
import { createCommerceTransport } from "./commerce-transport";

const apiUrl = process.env.COMMERCE_AGENT_LIVE_URL;
const bearer = process.env.COMMERCE_AGENT_LIVE_BEARER;

function transport(onRunId = () => {}, calls = [], threadId) {
  return createCommerceTransport(apiUrl, async (url, init = {}) => {
    const headers = new Headers(init.headers);
    headers.set("Authorization", `Bearer ${bearer}`);
    const response = await fetch(url, { ...init, headers });
    calls.push({ path: new URL(url).pathname, status: response.status });
    if (new URL(url).pathname.endsWith("/commands") && response.ok) {
      const payload = await response.clone().json();
      if (payload.result?.run_id) onRunId(payload.result.run_id);
    }
    return response;
  }, threadId);
}

test.skipIf(!apiUrl || !bearer)(
  "cancelled paused run does not resurrect an approval on remount",
  async () => {
    let runId;
    const firstTransport = transport((value) => { runId = value; });
    const first = renderHook(() => {
      const [threadId, setThreadId] = useState(null);
      return useStream({ transport: firstTransport, threadId, onThreadId: setThreadId });
    });
    let threadId;
    try {
      await act(async () => {
        await first.result.current.submit({
          messages: [{
            type: "human", content: "safe paused cancel probe",
            id: "live-paused-cancel-message",
          }],
        });
      });
      await waitFor(() => {
        expect(first.result.current.interrupts[0]?.value?.action_id).toBe("synthetic-no-write");
        expect(first.result.current.isLoading).toBe(false);
      }, { timeout: 20_000 });
      threadId = first.result.current.threadId;
      expect(threadId).toMatch(/^[0-9a-f-]{36}$/);
      expect(runId).toMatch(/^[0-9a-f-]{36}$/);
      const cancellation = await fetch(
        `${apiUrl}/threads/${threadId}/runs/${runId}/cancel?wait=0&action=interrupt`,
        { method: "POST", headers: { Authorization: `Bearer ${bearer}` } },
      );
      expect(cancellation.status).toBe(204);
    } finally {
      first.unmount();
      await firstTransport.close();
      cleanup();
    }

    const persisted = await fetch(`${apiUrl}/threads/${threadId}/state`, {
      headers: { Authorization: `Bearer ${bearer}` },
    });
    expect(persisted.status).toBe(200);
    const state = await persisted.json();
    expect(state.run_status).toBe("cancelled");
    expect(state.run_id).toBe(runId);
    expect(state.tasks).toEqual([]);
    expect(state.next).toEqual([]);

    const calls = [];
    const secondTransport = transport(() => {}, calls, threadId);
    const second = renderHook(() => useStream({ transport: secondTransport, threadId }));
    try {
      await waitFor(() => {
        expect(second.result.current.error).toBeUndefined();
        expect(second.result.current.isThreadLoading).toBe(false);
        expect(second.result.current.isLoading).toBe(false);
        expect(second.result.current.interrupts).toHaveLength(0);
      }, { timeout: 20_000 });
      expect(calls.some(({ path }) => path.endsWith("/state"))).toBe(true);
      expect(calls.some(({ path }) => path.endsWith("/stream"))).toBe(false);
    } finally {
      second.unmount();
      await secondTransport.close();
      cleanup();
    }
  },
  65_000,
);
