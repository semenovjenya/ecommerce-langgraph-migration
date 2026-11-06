// @vitest-environment jsdom
import { expect, test } from "vitest";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { useState } from "react";
import { useStream } from "@langchain/react";
import { createCommerceTransport } from "./commerce-transport";

const apiUrl = process.env.COMMERCE_AGENT_LIVE_URL;
const bearer = process.env.COMMERCE_AGENT_LIVE_BEARER;
const requests = [];

function transport(threadId) {
  return createCommerceTransport(apiUrl, (url, init = {}) => {
    const headers = new Headers(init.headers);
    headers.set("Authorization", `Bearer ${bearer}`);
    const path = new URL(url).pathname;
    const method = path.endsWith("/commands") && typeof init.body === "string"
      ? JSON.parse(init.body).method : null;
    return fetch(url, { ...init, headers }).then((response) => {
      requests.push({ path, method, status: response.status });
      return response;
    });
  }, threadId);
}

test.skipIf(!apiUrl || !bearer)(
  "real useStream replays an interrupt after remount and resumes the same thread",
  async () => {
    const firstTransport = transport();
    const first = renderHook(() => {
      const [threadId, setThreadId] = useState(null);
      return useStream({ transport: firstTransport, threadId, onThreadId: setThreadId });
    });
    let threadId;
    let interruptId;
    try {
      await act(async () => {
        await first.result.current.submit({
          messages: [{ type: "human", content: "safe approval probe", id: "live-approval-message" }],
        });
      });
      await waitFor(() => {
        expect(first.result.current.interrupts[0]?.value?.action_id).toBe("synthetic-no-write");
        expect(first.result.current.isLoading).toBe(false);
      }, { timeout: 20_000 });
      threadId = first.result.current.threadId;
      interruptId = first.result.current.interrupts[0].id;
      expect(threadId).toMatch(/^[0-9a-f-]{36}$/);
    } finally {
      first.unmount();
      await firstTransport.close();
      cleanup();
    }

    const secondTransport = transport(threadId);
    const second = renderHook(() => useStream({ transport: secondTransport, threadId }));
    try {
      await waitFor(() => {
        expect(second.result.current.error).toBeUndefined();
        expect(second.result.current.isThreadLoading).toBe(false);
        expect(second.result.current.interrupts[0]?.id).toBe(interruptId);
      }, { timeout: 20_000 });
      expect(requests.some(({ path, status }) => path.endsWith("/state") && status === 200)).toBe(true);
      await act(async () => {
        await second.result.current.respond({ approved: false }, { interruptId });
      });
      await waitFor(() => {
        expect(second.result.current.interrupts).toHaveLength(0);
        expect(second.result.current.isLoading).toBe(false);
        expect(second.result.current.error).toBeUndefined();
      }, { timeout: 20_000 });
      expect(second.result.current.threadId).toBe(threadId);
      expect(requests.filter(({ method, status }) => method === "input.respond" && status === 200)).toHaveLength(1);
      expect(requests.filter(({ path, status }) => path.endsWith("/stream") && status === 200).length).toBeGreaterThanOrEqual(2);
    } finally {
      second.unmount();
      await secondTransport.close();
      cleanup();
    }
  },
  65_000,
);
