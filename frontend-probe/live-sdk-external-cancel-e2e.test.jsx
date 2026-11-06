// @vitest-environment jsdom
import { expect, test } from "vitest";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { useState } from "react";
import { useChannel, useStream } from "@langchain/react";
import { createCommerceTransport } from "./commerce-transport";

const apiUrl = process.env.COMMERCE_AGENT_LIVE_URL;
const bearer = process.env.COMMERCE_AGENT_LIVE_BEARER;

test.skipIf(!apiUrl || !bearer)(
  "external cancellation settles an active SDK submission and remains replayable",
  async () => {
    const requests = [];
    const transport = createCommerceTransport(apiUrl, async (url, init = {}) => {
      const headers = new Headers(init.headers);
      headers.set("Authorization", `Bearer ${bearer}`);
      const response = await fetch(url, { ...init, headers });
      const path = new URL(url).pathname;
      if (path.endsWith("/commands") && response.ok) {
        const payload = await response.clone().json();
        if (payload.result?.run_id) requests.push(payload.result.run_id);
      }
      return response;
    });
    const authHeaders = { Authorization: `Bearer ${bearer}` };
    let submitSettled = false;
    let submitError;
    const hook = renderHook(() => {
      const [threadId, setThreadId] = useState(null);
      const stream = useStream({
        transport, threadId, onThreadId: setThreadId,
        defaultHeaders: authHeaders,
      });
      return { stream, lifecycle: useChannel(stream, ["lifecycle"], undefined, { replay: true }) };
    });
    try {
      act(() => {
        void hook.result.current.stream.submit({
          messages: [{
            type: "human", content: "safe external cancel probe",
            id: "live-external-cancel-message",
          }],
        }).then(() => { submitSettled = true; }).catch((error) => {
          submitError = error;
          submitSettled = true;
        });
      });
      await waitFor(() => {
        if (submitError || hook.result.current.stream.error) {
          throw new Error(`SDK submit failed: ${String(submitError ?? hook.result.current.stream.error)
            .replaceAll(bearer, "[redacted]")}`);
        }
        expect(requests).toHaveLength(1);
        expect(hook.result.current.stream.threadId).toMatch(/^[0-9a-f-]{36}$/);
        expect(hook.result.current.stream.isLoading).toBe(true);
        expect(hook.result.current.lifecycle.some(
          (event) => event.params?.data?.event === "running",
        )).toBe(true);
      }, { timeout: 20_000 });
      const runId = requests[0];
      const threadId = hook.result.current.stream.threadId;
      const cancellation = await fetch(
        `${apiUrl}/threads/${threadId}/runs/${runId}/cancel?wait=0&action=interrupt`,
        { method: "POST", headers: authHeaders },
      );
      expect(cancellation.status).toBe(204);
      await waitFor(() => {
        expect(hook.result.current.lifecycle.some(
          (event) => event.params?.data?.event === "cancelled",
        )).toBe(true);
        expect(hook.result.current.stream.isLoading).toBe(false);
        expect(submitSettled).toBe(true);
      }, { timeout: 20_000 });
      expect(submitError).toBeUndefined();
    } finally {
      hook.unmount();
      await transport.close();
      cleanup();
    }
  },
  65_000,
);
