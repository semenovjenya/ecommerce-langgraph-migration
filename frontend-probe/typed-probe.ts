import { useMemo, useRef } from "react";
import type { BaseMessage } from "@langchain/core/messages";
import { useStream } from "@langchain/react";
import { createCommerceTransport } from "./commerce-transport";

type Domain = "sales" | "marketing" | "supply" | "price" | "finance";
type AgentStatus = { status: "pending" | "planned" | "running" | "completed" | "failed" };

interface CommerceState {
  schema_version: number;
  messages: BaseMessage[];
  agent_statuses: Record<Domain, AgentStatus>;
  sales_analysis: unknown | null;
  marketing_analysis: unknown | null;
  supply_analysis: unknown | null;
  price_analysis: unknown | null;
  finance_analysis: unknown | null;
  data_gaps: unknown[];
  pending_interrupt: unknown | null;
}

export function useCommerceProbe(
  threadId: string | null,
  onThreadId: (id: string | null) => void,
  authenticatedFetch: typeof fetch,
) {
  // A restored thread must be bound before the SDK's eager hydration fetch.
  // The first run may update threadId without recreating the transport.
  const restoredThreadId = useRef(threadId);
  const transport = useMemo(
    () => createCommerceTransport(
      `${window.location.origin}/v2/agent`, authenticatedFetch,
      restoredThreadId.current ?? undefined,
    ),
    [authenticatedFetch],
  );
  const stream = useStream<CommerceState>({
    transport,
    threadId,
    onThreadId,
  });
  return {
    stream,
    salesStatus: stream.values.agent_statuses?.sales?.status,
    gapCount: stream.values.data_gaps?.length ?? 0,
  };
}
