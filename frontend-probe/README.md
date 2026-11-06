# Frontend SDK contract probe

A small React `useStream` transport and contract harness. This is not a complete application UI.

```bash
npm ci --ignore-scripts
npm run typecheck
npm test
```

Three offline SDK contract tests pass. Six live probes skip unless `COMMERCE_AGENT_LIVE_URL` and `COMMERCE_AGENT_LIVE_BEARER` are provided for an isolated test BFF/runtime. Never point them at a production environment or commit the bearer token.

The lockfile includes updated transitive dependencies; `npm audit` reported zero vulnerabilities during portfolio preparation. This is a point-in-time dependency audit, not a security guarantee.
