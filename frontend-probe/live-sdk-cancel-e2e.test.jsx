// @vitest-environment jsdom
import { expect, test } from "vitest";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { useState } from "react";
import { useChannel, useStream } from "@langchain/react";
import { createCommerceTransport } from "./commerce-transport";

const apiUrl = process.env.COMMERCE_AGENT_LIVE_URL;
const bearer = process.env.COMMERCE_AGENT_LIVE_BEARER;
const requests = [];

function transport() {
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
  });
}

test.skipIf(!apiUrl || !bearer)(
  "real useStream.stop cancels a durable run; remount replays its terminal event",
  async () => {
    const firstTransport = transport();
    // Stable option identities prevent useStream from rehydrating on every render.
    const authHeaders = { Authorization: `Bearer ${bearer}` };
    const first = renderHook(() => {
      const [threadId, setThreadId] = useState(null);
      const stream = useStream({
        transport: firstTransport, threadId, onThreadId: setThreadId,
        defaultHeaders: authHeaders,
      });
      return { stream, lifecycle: useChannel(stream, ["lifecycle"], undefined, { replay: true }) };
    });
    let threadId;
    let submitError;
    try {
      act(() => {
        void first.result.current.stream.submit({
          messages: [{ type: "human", content: "safe cancel probe", id: "live-cancel-message" }],
        }).catch((error) => {
          submitError = error;
        });
      });
      await waitFor(() => {
        if (submitError || first.result.current.stream.error) {
          throw new Error(`SDK submit failed: ${String(submitError ?? first.result.current.stream.error)
            .replaceAll(bearer, "[redacted]")}`);
        }
        expect(requests.some(({ method, status }) => method === "run.start" && status === 200)).toBe(true);
        expect(first.result.current.stream.isLoading).toBe(true);
        expect(first.result.current.lifecycle.some(
          (event) => event.params?.data?.event === "running",
        )).toBe(true);
      }, { timeout: 20_000 });
      expect(requests.filter(({ path }) => path.endsWith("/state")).length).toBeLessThan(5);
      threadId = first.result.current.stream.threadId;
      expect(threadId).toMatch(/^[0-9a-f-]{36}$/);
      await act(async () => {
        await first.result.current.stream.stop();
      });
      expect(first.result.current.stream.isLoading).toBe(false);
    } finally {
      first.unmount();
      await firstTransport.close();
      cleanup();
    }

    const secondTransport = transport();
    const second = renderHook(() => {
      const stream = useStream({
        transport: secondTransport, threadId,
        defaultHeaders: authHeaders,
      });
      return { stream, lifecycle: useChannel(stream, ["lifecycle"], undefined, { replay: true }) };
    });
    try {
      await waitFor(() => {
        expect(second.result.current.lifecycle.some(
          (event) => event.params?.data?.event === "cancelled",
        )).toBe(true);
      }, { timeout: 20_000 });
      expect(second.result.current.stream.threadId).toBe(threadId);
      expect(second.result.current.stream.isLoading).toBe(false);
    } finally {
      second.unmount();
      await secondTransport.close();
      cleanup();
    }
  },
  65_000,
);
