// @vitest-environment jsdom
import { afterEach, expect, test, vi } from "vitest";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { useState } from "react";
import { useStream } from "@langchain/react";
import { createCommerceTransport } from "./commerce-transport";
import runStartV1 from "./fixtures/run-start-v1.json";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function createProtocolFetch({ interrupt = false } = {}) {
  const requests = [];
  const subscribers = new Set();
  const journal = [];
  const encoder = new TextEncoder();
  const emit = (event) => {
    const recorded = { ...event, seq: journal.length, event_id: `probe:${journal.length}` };
    journal.push(recorded);
    for (const subscriber of subscribers) subscriber(recorded);
  };
  const fetchMock = vi.fn(async (url, init = {}) => {
    const path = new URL(url).pathname;
    const body = init.body ? JSON.parse(init.body) : null;
    requests.push({ path, method: init.method, body });
    if (path.endsWith("/state")) return new Response("", { status: 404 });
    if (path.endsWith("/commands")) {
      queueMicrotask(() => {
        emit({ type: "event", method: "lifecycle", params: {
          namespace: [], data: { event: "running" },
        } });
        emit({ type: "event", method: "values", params: {
          namespace: [], data: {
            schema_version: 1,
            messages: [{ type: "human", content: "safe question", id: "probe-message" }],
            request_id: "probe-request",
          },
        } });
        if (interrupt && body.method === "run.start") {
          emit({ type: "event", method: "input.requested", params: {
            namespace: [], data: {
              interrupt_id: "synthetic-interrupt",
              payload: { kind: "approval", action_id: "synthetic-action" },
            },
          } });
          emit({ type: "event", method: "lifecycle", params: {
            namespace: [], data: { event: "interrupted" },
          } });
        } else {
          emit({ type: "event", method: "lifecycle", params: {
            namespace: [], data: { event: "completed" },
          } });
        }
      });
      return Response.json({ type: "success", id: body.id, result: { run_id: "probe-run" } });
    }
    if (path.endsWith("/stream")) {
      const stream = new ReadableStream({
        start(controller) {
          const publish = (event) => {
            if (!body.channels.includes(event.method)
              && !(event.method === "input.requested" && body.channels.includes("input"))) return;
            controller.enqueue(encoder.encode(`id: ${event.event_id}\ndata: ${JSON.stringify(event)}\n\n`));
          };
          for (const event of journal) {
            if (event.seq > (body.since ?? -1)) publish(event);
          }
          subscribers.add(publish);
          init.signal?.addEventListener("abort", () => {
            subscribers.delete(publish);
            try { controller.close(); } catch { /* already closed */ }
          }, { once: true });
        },
      });
      return new Response(stream, { headers: { "content-type": "text/event-stream" } });
    }
    throw new Error(`Unexpected SDK path: ${path}`);
  });
  return { fetchMock, requests };
}

test("pinned SDK adapter uses the configured BFF command/state/stream paths", async () => {
  const { fetchMock, requests } = createProtocolFetch();
  const adapter = createCommerceTransport("https://example.test/v2/agent", fetchMock);
  adapter.setThreadId("probe-thread");
  try {
    expect(await adapter.getState()).toBeNull();
    const response = await adapter.send({
      type: "command", id: "probe-command", method: "run.start",
      params: { assistant_id: "commerce_agent", input: { messages: [] } },
    });
    expect(response.result.run_id).toBe("probe-run");
    const subscription = adapter.openEventStream({ channels: ["values"] });
    try {
      await subscription.ready;
      const next = await subscription.events[Symbol.asyncIterator]().next();
      expect(next.value.method).toBe("values");
    } finally {
      subscription.close();
    }
    expect(requests.map((request) => request.path)).toEqual([
      "/v2/agent/threads/probe-thread/state",
      "/v2/agent/threads/probe-thread/commands",
      "/v2/agent/threads/probe-thread/stream",
    ]);
  } finally {
    await adapter.close();
  }
});

test("useStream submits a new thread using the pinned SDK wire envelope", async () => {
  const { fetchMock, requests } = createProtocolFetch();
  const adapter = createCommerceTransport("https://example.test/v2/agent", fetchMock);
  const { result, unmount } = renderHook(() => {
    const [threadId, setThreadId] = useState(null);
    return useStream({
      transport: adapter,
      threadId,
      onThreadId: setThreadId,
    });
  });
  try {
    await act(async () => {
      await result.current.submit({
        messages: [{ type: "human", content: "safe question", id: "probe-message" }],
      });
    });
    await waitFor(() => expect(result.current.values.schema_version).toBe(1));
    expect(result.current.threadId).toMatch(/^[0-9a-f-]{36}$/);
    expect(result.current.messages[0].content).toBe("safe question");
    const command = requests.find((request) => request.path.endsWith("/commands"));
    expect(Object.keys(command.body).sort()).toEqual(["id", "method", "params"]);
    expect(command.body.method).toBe("run.start");
    expect(Object.keys(command.body.params).sort()).toEqual(["assistant_id", "config", "input"]);
    expect(command.body.params.assistant_id).toBe("commerce_agent");
    expect(command.body.params.config).toEqual({
      configurable: { thread_id: result.current.threadId },
    });
    expect(command.body.params.input.messages[0].content).toBe("safe question");
    expect({
      ...command.body,
      params: {
        ...command.body.params,
        config: { configurable: { thread_id: runStartV1.params.config.configurable.thread_id } },
      },
    }).toEqual(runStartV1);
    expect(requests.some((request) => request.path.endsWith("/stream"))).toBe(true);
  } finally {
    unmount();
    await adapter.close();
  }
});

test("pinned useStream sends input.respond for a protocol interrupt", async () => {
  const { fetchMock, requests } = createProtocolFetch({ interrupt: true });
  const adapter = createCommerceTransport("https://example.test/v2/agent", fetchMock);
  const { result, unmount } = renderHook(() => {
    const [threadId, setThreadId] = useState(null);
    return useStream({ transport: adapter, threadId, onThreadId: setThreadId });
  });
  try {
    await act(async () => {
      await result.current.submit({
        messages: [{ type: "human", content: "safe question", id: "probe-message" }],
      });
    });
    await waitFor(() => expect(result.current.interrupts[0]?.id).toBe("synthetic-interrupt"));
    await act(async () => {
      await result.current.respond({ approved: false }, { interruptId: "synthetic-interrupt" });
    });
    const response = requests.find((request) => request.body?.method === "input.respond");
    expect(response?.path).toMatch(/\/v2\/agent\/threads\/.+\/commands$/);
    expect(response?.body?.params).toEqual({
      namespace: [], interrupt_id: "synthetic-interrupt", response: { approved: false },
    });
    expect(typeof response?.body?.id).toBe("number");
  } finally {
    unmount();
    await adapter.close();
  }
});
