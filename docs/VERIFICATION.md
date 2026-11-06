# Verification and export status

Prepared on 2026-11-06 from the current working copy of `services/commerce-agent`. This snapshot includes uncommitted migration implementation, exported without private Git history. It is not a claim that the original production migration has been accepted.

## Reproducible checks

| Check | Command | Observed result |
| --- | --- | --- |
| Fresh dependency installation | `uv sync --frozen` | Passed with the supplied `uv.lock` on Python 3.11 |
| Standalone Python suite | `uv run --frozen python -m pytest -o addopts='' -q` | **804 passed, 24 skipped, 3 warnings** |
| In-memory checkpoint pause/resume | `uv run --frozen python -m examples.checkpoint_demo` | Paused at synthesis, resumed, zero external writes |
| Evidence benchmark | `uv run --frozen python -m benchmarks.evidence_payload` | Exact roundtrip; 76.02%, 67.37%, 0% evidence-byte reductions |
| Frontend dependency install | `npm ci --ignore-scripts` in `frontend-probe` | Passed |
| Frontend typing | `npm run typecheck` | Passed |
| Frontend SDK probes | `npm test` | **3 passed, 6 skipped** |
| Frontend dependency audit | `npm audit` | 0 vulnerabilities reported after transitive dependency updates |

Python skipped tests require a dedicated PostgreSQL. Frontend live probes require a running BFF/runtime and test bearer. Docker daemon was unavailable, so PostgreSQL/restart/live HTTP integration is not claimed.

Warnings are an intentional invalid-UUID serialization canary and LangGraph's experimental v3 streaming warnings. They were not suppressed. Test counts describe only the exported standalone suite.

## Export boundary

- 132 runtime Python files were verified byte-for-byte identical to the source working copy.
- The standalone export includes all runtime source and the lockfile, public examples, frontend probe and self-contained tests.
- 81 test modules requiring private evidence, monorepo business-core imports or dependent test helpers were excluded. Their names and reasons are in [excluded-private-tests.json](excluded-private-tests.json). Their prior results are not counted here.
- No private seller capture or historical trace output was copied.
- Credentials from the source environment were compared against exported text; no matches were found. Personal workstation paths and known client-name references were also checked. This is a bounded scan, not a comprehensive security certification.
- Synthetic secret canaries remain in security tests. Credential-looking values are generated from obvious dummy fragments where needed to avoid publishing token-shaped literals.
- The frontend lockfile updates vulnerable transitive dependencies; package runtime code is unchanged. Typechecking and offline SDK probes were repeated after the update.

An independent read-only review checked export boundaries, benchmark methodology and readiness claims. Its integration finding was addressed: the published backend snapshot does not contain `/internal/v1/agent-tools/*`. A compatible business-tools service is needed for live domain integration.

The original monorepo documentation contract checker passed: six migration documents, 22 stable state fields, links and code fences. This validates the source contract, not completion of migration milestones. Exported state/graph contract tests provide the standalone verification.

## Status of the larger migration

| Area | Portfolio evidence | Production gate |
| --- | --- | --- |
| Typed graph, reducers, guards | Code and standalone tests | Full operational acceptance separate |
| Fixed domain subgraphs | Source included, dependencies opt-in | Full semantic parity with private data not claimed |
| Compact evidence | Reproducible synthetic byte benchmark | Token/cost/latency evaluation not measured |
| Checkpoint recovery | In-memory example and unit checks | Dedicated PostgreSQL restart tests skipped here |
| Protocol and frontend transport | Offline contract tests | Live integration skipped here |
| General planning/synthesis/actions | Explicit stub nodes preserved | Not complete |
| Deep Research | Disabled path | Not enabled |
| Production cutover | No source service changes | Not performed |

The original private project and published backend remain unchanged. This export does not resume paused migration work, retire legacy services or change production routing. GitHub Actions is not installed because the available credential lacks workflow scope; all reported checks are local.
