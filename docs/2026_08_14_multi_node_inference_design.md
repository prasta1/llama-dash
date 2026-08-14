# Multi-Node Inference

Date: 2026-08-14

## Summary

llama-dash currently fronts exactly one inference backend instance. This design
extends it to front N llama-swap instances — concretely two, on separate hosts —
with a single dashboard, a single request log, and automatic `/v1/*` dispatch to
whichever instance owns the requested model.

The inference backend facade (`src/server/inference/backend.ts`) already
normalizes model shapes and gates behavior on capability flags, so runtime
specifics do not leak into the proxy or UI. What it does not have is *plurality*:
the facade is a module-level singleton bound to one base URL. This design
replaces that singleton with a registry and threads node identity through the
proxy, the data model, and the UI.

`docs/2026_05_03_inference_backends.md` anticipated this:

> A registry/selector becomes useful when there are multiple concrete backends
> [...] or when llama-dash supports multiple active runtimes at the same time.

## Goals

- Monitor two llama-swap instances on different hosts from one dashboard.
- Dispatch `/v1/*` traffic to whichever node serves the requested model, without
  clients changing their configuration.
- Attribute every logged request and every model load/unload event to a node.
- Degrade gracefully where a capability is inherently host-local (GPU stats,
  config-file editing) rather than faking it.

## Non-Goals

- No admin CRUD or UI for adding nodes. Nodes are declared in environment
  variables and read at boot. Two static hosts do not justify a table, a
  migration, an admin route group, and a settings page.
- No remote GPU telemetry. Node 2's GPU is out of scope for this pass.
- No load balancing, failover, or health-based routing. Dispatch is by model
  ownership, not by capacity.
- No second backend *kind*. Both nodes are llama-swap. The Ollama adapter
  sketched in the 2026-05-03 doc remains parked.

## Terminology: "node"

`endpoint` is already overloaded in this repo and must not take a third meaning:

- `ctx.endpoint` and `requests.endpoint` mean the **request path**
  (`/v1/chat/completions`).
- The `/endpoints` route means **client connection examples**.

A third sense would make `requests.endpoint` and `requests.endpoint_id` two
unrelated concepts one underscore apart. This design uses **node** throughout:
`INFERENCE_2_BASE_URL`, `nodeId`, `requests.node_id`, `InferenceNode`.

## Current single-node assumptions

An audit of what actually binds the codebase to one instance.

**Configuration is scalar** — `src/server/config.ts:20-23`. `inferenceBaseUrl`,
`inferenceInsecure`, and `inferenceConfigFile` are single strings from single
environment variables.

**The facade is a module-level singleton** — `src/server/inference/backend.ts:74`:

```ts
export const inferenceBackend = createInferenceBackend(config.inferenceBackend)
```

25 references across eight modules import it directly: `model-watcher.ts`,
`metrics.ts`, `proxy/context.ts`, `admin/routes/models.ts`,
`admin/routes/system.ts`, `admin/routes/config.ts`, `admin/model-detail.ts`,
and `lib/auth-functions.ts`.

**The client underneath has no base-URL seam** —
`src/server/llama-swap/client.ts:14,27` interpolate `config.inferenceBaseUrl`
directly on every call, as does `inference/backends/llama-swap.ts` at lines
76, 80, 121, and 123. There is no way to point an existing backend object at a
different host.

**The data model has no node identity** — `src/server/db/schema.ts`. Neither
`requests` nor `model_events` carries a node column. `model_events` is keyed on
`modelId` alone, and `model-watcher.ts:9` holds one global `knownRunning` set.

**The proxy has one default upstream, and the override is locked** —
`proxy/context.ts:47` seeds `defaultUpstream` from
`inferenceBackend.defaultProxyUpstream()`. Routing rules can override it via
`target.type === 'direct'` (`proxy/upstream.ts:20`), but
`src/lib/schemas/routing-rule.ts:108` restricts direct targets to an allow-list:

```ts
export const ALLOWED_DIRECT_UPSTREAM_HOSTS = new Set(['api.openai.com', 'api.anthropic.com'])
```

So a routing rule cannot currently target a second local llama-swap.

## Configuration

Node 1 keeps the existing variables, so an unchanged `.env` behaves exactly as
it does today. Additional nodes use an indexed prefix:

```text
INFERENCE_BASE_URL=http://gpu-box:8080
INFERENCE_LABEL=gpu-box
INFERENCE_CONFIG_FILE=/config/config.yaml

INFERENCE_2_BASE_URL=http://mac-mini:8080
INFERENCE_2_LABEL=mac-mini
INFERENCE_2_INSECURE=false
INFERENCE_2_CONFIG_FILE=
```

Loading rules:

- Node 1 is `INFERENCE_*`. Nodes 2..N are `INFERENCE_<n>_*`, scanned upward from
  2 and stopping at the first index with no `BASE_URL`.
- `id` is the slugified label; `label` defaults to the URL host if unset. Ids
  must be unique — a collision fails at boot rather than silently merging nodes.
- Priority is declaration order. Node 1 is primary.
- `INFERENCE_BACKEND` stays global. All nodes are the same kind in this pass.
- Omitting `INFERENCE_2_BASE_URL` yields a single-node registry that is
  behaviorally identical to today.

`INFERENCE_INSECURE` sets `process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0'` at
boot (`src/server.ts:6-7`). It is process-wide, not per-connection.
`INFERENCE_<n>_INSECURE` is therefore accepted but ORs into that same global
flag, and `/api/system` keeps reporting one `insecureTls` value
(`admin/routes/system.ts:62`). The spec records this rather than pretending the
setting is per-node.

## Node registry

New `src/server/inference/registry.ts`:

```ts
export type InferenceNode = {
  id: string
  label: string
  priority: number
  backend: InferenceBackend
}

export const nodes: ReadonlyArray<InferenceNode>
export const primaryNode: InferenceNode
export function getNode(id: string): InferenceNode | undefined
```

This requires turning two modules from singletons into factories:

- `llama-swap/client.ts` — `createLlamaSwapClient(baseUrl)` returning the same
  typed surface, instead of a module-level `llamaSwap` object reading config.
- `inference/backends/llama-swap.ts` — `createLlamaSwapBackend(nodeConfig)`
  taking its base URL and config-file path as arguments.
- `inference/llama-swap-config.ts` — config-derived hints (context lengths, log
  names, config snippets) become per-node, since each node has its own
  `config.yaml`.

To keep the change reviewable and the fork mergeable, `backend.ts` retains:

```ts
export const inferenceBackend = primaryNode.backend
```

as a deprecated shim. The eight consuming modules migrate to the registry in
later commits rather than in one sweeping diff.

## Model index

Dispatch needs a fast, synchronous answer to "which node serves this model?".

New `src/server/inference/model-index.ts` holds `Map<modelId, Array<nodeId>>`,
node ids sorted by priority. It is refreshed on the **existing** model-watcher
tick, which already polls every node every 15 seconds — no second poller, no
added latency in the request path.

A model missing from the index resolves to the primary node, which preserves
today's behavior for unknown models and lets llama-swap return its own 404.

## Proxy dispatch

`selectUpstream()` (`proxy/upstream.ts:14`) is called from `context.ts:90,93`,
which run *after* transforms resolve aliases and `rewrite_model`. The resolved
model is therefore already available at the point upstream is chosen. Dispatch
needs no change to the auth or body-buffering flow, so the
"don't buffer a body just to reject it" constraint in `CLAUDE.md` holds
unchanged.

Resolution order inside `selectUpstream()`:

1. Routing rule `target.type === 'direct'` — unchanged cloud passthrough.
2. Routing rule `target.type === 'node'` — **new**, the explicit pin/override.
3. Model index lookup — **new**, owning node, priority breaks ties.
4. Fallback — primary node.

The routing-rule target schema gains a `node` variant carrying `nodeId`.
`isAllowedDirectUpstream` is untouched: node targets are validated against the
registry, not against the direct-upstream host allow-list.

A rule referencing an unknown node id is rejected at write time. If a node is
later removed from the environment, a stored rule still naming it degrades to
default dispatch — model index, then primary — rather than failing every matched
request. It does not become non-matching: re-evaluating rules at that point would
break the single ordered resolution pass that `CLAUDE.md` requires.

## Data model

Two nullable columns and one migration. Null means "logged before this change".

- `requests.node_id` — which node served the request.
- `model_events.node_id` — which node the load/unload happened on.

`model-watcher.ts` must key its `knownRunning` set on `${nodeId}::${modelId}`.
Without this, two nodes running the same model name cancel each other's events
and the timeline silently goes wrong — the failure is invisible rather than
loud, which makes it the highest-risk item in this design.

Request-log retention (`request-log-maintenance.ts`) is unaffected; node id is
metadata, not a new retention class.

## Per-node capability degradation

`InferenceBackendInfo.capabilities` already exists per backend instance. It
becomes per node, and the UI reads it per node rather than globally.

**GPU** — `gpu-poller.ts` shells out to `nvidia-smi` / `rocm-smi` /
`system_profiler` on the llama-dash *host*. It describes one machine and cannot
describe node 2. The GPU panel is labelled with the host it belongs to; no
attempt is made to infer or fake node 2's GPU state.

**Config editor** — `INFERENCE_CONFIG_FILE` is a local filesystem path. A node
whose `config.yaml` is not reachable from the llama-dash container reports
`config: false` and the editor shows the existing not-configured fallback.

**Logs** — `/api/log-events` gains a `?node=` parameter. The Logs page gains a
node selector, defaulting to primary.

**Metrics** — `src/server/metrics.ts` gauges gain a `node` label. Without it
Prometheus sums both hosts into one series, which is wrong and silent.

## `/v1/models` merge

The proxy has no handler for `GET /v1/models`; it is forwarded like any other
`/v1/*` path. With two nodes, a client would see only the primary's catalog.
Models on the second node remain *reachable* — the model index dispatches them
correctly if named explicitly — but undiscoverable in any client's dropdown.

**Decision:** llama-dash intercepts `GET /v1/models`, queries every node, and
returns a merged list.

This is the first `/v1/*` route llama-dash answers rather than relays, and the
design records that deliberately. Consequences:

- **Duplicate model ids across nodes** — deduped to one entry, owned by the
  higher-priority node, matching how dispatch resolves the tie.
- **A node unreachable** — return the surviving nodes' models. A partially
  populated dropdown is more useful than a failed call, and the System page is
  where node health is reported.

This is the one item the user did not explicitly confirm; it is recorded as a
decision to be accepted or reversed at spec review.

## UI

- **Models** — node badge per row. A model on both nodes shows once, badged with
  both, consistent with `/v1/models` dedupe.
- **System** — one health/version/latency row per node, each showing its own
  capability flags.
- **Requests** — node column and node filter, backed by `requests.node_id`.
- **Dashboard** — running-model entries attributed to their node.
- **Logs** — node selector.

Node identity is a *category*, not a *state*, so per the status-vs-identity rule
in `CLAUDE.md` node badges use the `--series-*` ramp via `assignSeriesSteps()`,
never `--ok` / `--warn` / `--err`.

## Commit breakdown

Proxy and admin changes stay in separate commits per repo scope discipline.

1. Parameterize `llama-swap/client.ts` and `createLlamaSwapBackend()` — pure
   refactor, no behavior change.
2. Node registry plus indexed env loading, with the `inferenceBackend` shim.
   Still single-node in practice.
3. `model_events.node_id` migration and per-node watcher state.
4. Model index, populated from the watcher tick.
5. Proxy dispatch: `node` routing target, `selectUpstream()` resolution order,
   `requests.node_id` migration.
6. Admin API per node: models, system, config, log events.
7. UI: node badges, per-node System rows, Requests filter, Logs selector.
8. `GET /v1/models` merge.
9. Metrics `node` label.
10. Docs: `AGENTS.md`, `README.md`, `.env.example`, compose files.

## Testing

- `proxy/handler.test.ts` mocks `config.inferenceBaseUrl`; it moves to mocking
  the registry.
- New unit tests for `selectUpstream()` resolution order, covering: direct
  target wins over node target, node target wins over index, index wins over
  primary, unknown model falls back to primary, and duplicate-model tie-break by
  priority.
- New unit tests for env parsing: gap stops the scan, duplicate ids fail at
  boot, absent `INFERENCE_2_*` yields a one-node registry.
- New watcher test: same model id running on both nodes produces two independent
  event streams and neither cancels the other.
- Manual verification on the running dev server for the UI commits, per the
  browser-verification note in `CLAUDE.md`.

## Risks

- **Watcher key collision** (mitigated above) is the highest-impact silent
  failure in this design.
- **Fork divergence** — the repo has no `upstream` git remote, only `origin`.
  The additive shim in commit 2 exists so upstream fixes touching
  `inferenceBackend` still merge; rewriting the singleton outright would make
  every future merge painful.
- **Registry construction at import time** — the current singleton is built at
  module load. The registry keeps that, so a malformed `INFERENCE_2_BASE_URL`
  fails at boot. This is intentional and matches the existing
  `Unsupported INFERENCE_BACKEND` behavior.
