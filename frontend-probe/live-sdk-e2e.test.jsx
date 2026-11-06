// @vitest-environment jsdom
import { expect, test } from "vitest";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { useState } from "react";
import { useStream } from "@langchain/react";
import { createCommerceTransport } from "./commerce-transport";

const apiUrl = process.env.COMMERCE_AGENT_LIVE_URL;
const bearer = process.env.COMMERCE_AGENT_LIVE_BEARER;
const network = [];

function authenticatedTransport() {
  return createCommerceTransport(apiUrl, (url, init = {}) => {
    const headers = new Headers(init.headers);
    headers.set("Authorization", `Bearer ${bearer}`);
    const path = new URL(url).pathname;
    const command = path.endsWith("/commands") && typeof init.body === "string"
      ? JSON.parse(init.body) : null;
    return fetch(url, { ...init, headers }).then((response) => {
      network.push({
        path, status: response.status,
        command: command && {
          keys: Object.keys(command), id: command.id, method: command.method,
          assistantId: command.params?.assistant_id,
          paramsKeys: Object.keys(command.params ?? {}),
          inputKeys: Object.keys(command.params?.input ?? {}),
          messageKeys: (command.params?.input?.messages ?? []).map(Object.keys),
          messages: command.params?.input?.messages,
          config: command.params?.config,
        },
      });
      return response;
    });
  });
}

test.skipIf(!apiUrl || !bearer)(
  "real useStream submits, hydrates, and submits again through the authenticated BFF",
  async () => {
    const firstTransport = authenticatedTransport();
    const first = renderHook(() => {
      const [threadId, setThreadId] = useState(null);
      return useStream({ transport: firstTransport, threadId, onThreadId: setThreadId });
    });
    let threadId;
    try {
      await act(async () => {
        await first.result.current.submit({
          messages: [{ type: "human", content: "safe first question", id: "live-message-1" }],
        });
      });
      if (first.result.current.error) {
        throw new Error(JSON.stringify({
          error: String(first.result.current.error), network,
        }));
      }
      await waitFor(() => {
        const diagnostic = JSON.stringify({
          threadId: first.result.current.threadId,
          error: String(first.result.current.error ?? ""), network,
        });
        if (first.result.current.error) throw new Error(diagnostic);
        expect(first.result.current.values.schema_version, diagnostic).toBe(1);
        expect(first.result.current.messages.at(-1)?.content).toBe("safe first question");
        expect(first.result.current.values.agent_statuses.sales.status).toBe("planned");
        expect(first.result.current.isLoading).toBe(false);
      }, { timeout: 20_000 });
      threadId = first.result.current.threadId;
      expect(threadId).toMatch(/^[0-9a-f-]{36}$/);
      expect(first.result.current.values.agent_statuses.sales.status).toBe("planned");
      expect(network.some(({ path, status }) => path.endsWith("/stream") && status === 200)).toBe(true);
    } finally {
      first.unmount();
      await firstTransport.close();
      cleanup();
    }

    const secondTransport = authenticatedTransport();
    const second = renderHook(() => useStream({ transport: secondTransport, threadId }));
    try {
      await waitFor(() => {
        expect(second.result.current.messages.at(-1)?.content).toBe("safe first question");
        expect(second.result.current.values.schema_version).toBe(1);
      }, { timeout: 20_000 });
      await act(async () => {
        await second.result.current.submit({
          messages: [{ type: "human", content: "safe second question", id: "live-message-2" }],
        });
      });
      await waitFor(() => {
        expect(second.result.current.messages.map((message) => message.content)).toEqual([
          "safe first question", "safe second question",
        ]);
        expect(second.result.current.values.agent_statuses.finance.status).toBe("planned");
        expect(second.result.current.isLoading).toBe(false);
      }, { timeout: 20_000 });
      expect(second.result.current.threadId).toBe(threadId);
      expect(second.result.current.values.agent_statuses.finance.status).toBe("planned");
      expect(network.filter(
        ({ path, status }) => path.endsWith("/stream") && status === 200,
      ).length).toBeGreaterThanOrEqual(2);
    } finally {
      second.unmount();
      await secondTransport.close();
      cleanup();
    }
  },
  65_000,
);
