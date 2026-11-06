import { HttpAgentServerAdapter } from "@langchain/react";

/** Keep the self-managed service's graph ID stable with the typed custom adapter. */
export function createCommerceTransport(
  apiUrl: string, fetchOverride?: typeof fetch, threadId?: string,
) {
  return new HttpAgentServerAdapter({
    apiUrl,
    threadId,
    fetch: fetchOverride,
    paths: { stream: (id) => `/threads/${id}/stream` },
    onRequest: async (url, init) => {
      if (!url.pathname.endsWith("/commands") || typeof init.body !== "string") {
        return init;
      }
      const command: unknown = JSON.parse(init.body);
      if (
        typeof command !== "object" || command === null
        || !("method" in command) || command.method !== "run.start"
        || !("params" in command)
        || typeof command.params !== "object" || command.params === null
      ) {
        return init;
      }
      const params = command.params as Record<string, unknown>;
      if (params.assistant_id !== "_") return init;
      return {
        ...init,
        body: JSON.stringify({
          ...command,
          params: { ...params, assistant_id: "commerce_agent" },
        }),
      };
    },
  });
}
