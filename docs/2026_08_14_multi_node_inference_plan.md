# Multi-Node Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let llama-dash front two llama-swap instances on separate hosts from one dashboard, dispatching `/v1/*` to whichever node owns the requested model.

**Architecture:** Replace the module-level `inferenceBackend` singleton with a registry of nodes built from indexed environment variables. A model index (`modelId -> nodeId[]`), refreshed on the existing 15s model-watcher tick, answers ownership queries synchronously. Upstream selection moves into `selectUpstream()`, which already runs after transforms resolve the model name.

**Tech Stack:** TypeScript, TanStack Start, Drizzle + better-sqlite3, Valibot, Vitest, Biome, tsgo.

**Spec:** [`docs/2026_08_14_multi_node_inference_design.md`](./2026_08_14_multi_node_inference_design.md)

## Global Constraints

- **Terminology:** the noun is **node**. Never `endpoint` — that already means request path (`ctx.endpoint`) and client connection examples (`/endpoints` route).
- **Formatting:** Biome — 2-space indent, single quotes, no semicolons, trailing commas. Server imports use explicit `.ts` extensions.
- **Before any task is "done":** `pnpm lint:fix && pnpm format:fix && pnpm typecheck` must each exit clean.
- **Tests:** `pnpm test` (vitest run). Single file: `pnpm vitest run <path>`.
- **Migrations:** edit `src/server/db/schema.ts`, then `pnpm db:generate`, then `pnpm db:migrate`. Commit the generated SQL in `drizzle/`.
- **Never start `pnpm dev` as a final smoke test.** If a task needs a running server, start it mid-task, verify, kill it.
- **Do not touch `ALLOWED_DIRECT_UPSTREAM_HOSTS`.** Node targets validate against the registry, not that allow-list.
- **Status vs identity colour:** node badges use `--series-*` via `assignSeriesSteps()`. Never `--ok`/`--warn`/`--err`.
- **Commit style:** one logical change per commit, message focused on *why*. Stage named paths only, never `git add -A`.
- **Branch:** `feature/multi-node-inference`. Auto-push allowed on `feature/*`.

## File Structure

**Created:**
- `src/server/inference/node-config.ts` — pure env → node config parsing
- `src/server/inference/node-config.test.ts`
- `src/server/inference/registry.ts` — the node registry
- `src/server/inference/model-index.ts` — `modelId -> nodeId[]` ownership map
- `src/server/inference/model-index.test.ts`
- `src/server/proxy/upstream.test.ts` — dispatch resolution order
- `src/server/proxy/models-merge.ts` — merged `GET /v1/models` handler
- `src/server/proxy/models-merge.test.ts`

**Modified:**
- `src/server/llama-swap/client.ts` — singleton → factory
- `src/server/inference/backends/llama-swap.ts` — takes node config
- `src/server/inference/llama-swap-config.ts` — config path per node
- `src/server/inference/backend.ts` — singleton becomes a shim over the registry
- `src/server/config.ts` — expose raw env for node parsing
- `src/server/model-watcher.ts` — per-node state, feeds the model index
- `src/server/proxy/upstream.ts` — resolution order, returns node id
- `src/server/proxy/context.ts` — carries `nodeId`, drops `defaultUpstream`
- `src/server/proxy/handler.ts` — `/v1/models` intercept, logs `nodeId`
- `src/server/proxy/forward.ts` — `nodeId` on the log input
- `src/server/proxy/log.ts` — writes `node_id`
- `src/server/db/schema.ts` — `requests.node_id`, `model_events.node_id`
- `src/lib/schemas/routing-rule.ts` — `node` target variant
- `src/server/admin/routes/{models,system,config}.ts`, `src/server/admin/model-detail.ts`
- `src/server/metrics.ts` — `node` label
- `src/features/{models,dashboard,requests,logs}/*` — node UI
- `AGENTS.md`, `README.md`, `.env.example`, `docker-compose.*.yaml`

---

### Task 1: Turn the llama-swap client into a factory

`client.ts` interpolates `config.inferenceBaseUrl` on every call, so there is no way to point it at a second host. This is a pure refactor — behavior must not change.

**Files:**
- Modify: `src/server/llama-swap/client.ts`
- Test: `src/server/llama-swap/client.test.ts` (create)

**Interfaces:**
- Produces: `createLlamaSwapClient(baseUrl: string): LlamaSwapClient` where `LlamaSwapClient` is the type of the current `llamaSwap` object (`listModels`, `listRunning`, `unloadModel`, `loadModel`, `unloadAll`, `health`, `version`).
- Produces: `llamaSwap` — retained export, `= createLlamaSwapClient(config.inferenceBaseUrl)`.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { createLlamaSwapClient } from './client'

describe('createLlamaSwapClient', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('ok', { status: 200 })),
    )
  })
  afterEach(() => vi.unstubAllGlobals())

  it('calls the base URL it was constructed with', async () => {
    const client = createLlamaSwapClient('http://node-two:8080')
    await client.health()
    expect(fetch).toHaveBeenCalledWith('http://node-two:8080/health', undefined)
  })

  it('keeps two clients independent', async () => {
    await createLlamaSwapClient('http://a:8080').health()
    await createLlamaSwapClient('http://b:8080').health()
    const calls = vi.mocked(fetch).mock.calls.map(([url]) => url)
    expect(calls).toEqual(['http://a:8080/health', 'http://b:8080/health'])
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm vitest run src/server/llama-swap/client.test.ts`
Expected: FAIL — `createLlamaSwapClient is not a function`.

- [ ] **Step 3: Implement the factory**

Replace the module body of `client.ts` below the imports:

```ts
export function createLlamaSwapClient(baseUrl: string) {
  const callText = async (path: string, init?: RequestInit): Promise<string> => {
    const res = await fetch(`${baseUrl}${path}`, init)
    if (!res.ok) {
      const body = await res.text().catch(() => '')
      throw new Error(`llama-swap ${path} -> ${res.status}: ${body.slice(0, 200)}`)
    }
    return res.text()
  }

  const callJson = async <T>(
    path: string,
    schema: v.BaseSchema<unknown, T, v.BaseIssue<unknown>>,
    init?: RequestInit,
  ): Promise<T> => {
    const res = await fetch(`${baseUrl}${path}`, init)
    if (!res.ok) {
      const body = await res.text().catch(() => '')
      throw new Error(`llama-swap ${path} -> ${res.status}: ${body.slice(0, 200)}`)
    }
    return v.parse(schema, await res.json())
  }

  return {
    listModels: () => callJson('/v1/models', ModelsListResponseSchema),
    listRunning: () => callJson('/running', RunningResponseSchema),
    unloadModel: (id: string) => callText(`/api/models/unload/${encodeURIComponent(id)}`, { method: 'POST' }),
    loadModel: (id: string) => callText(`/upstream/${encodeURIComponent(id)}/`),
    unloadAll: () => callJson('/api/models/unload', v.object({ msg: v.string() }), { method: 'POST' }),
    health: () => callText('/health'),
    version: () => callJson('/api/version', VersionResponseSchema),
  }
}

export type LlamaSwapClient = ReturnType<typeof createLlamaSwapClient>

export const llamaSwap = createLlamaSwapClient(config.inferenceBaseUrl)
```

- [ ] **Step 4: Run tests**

Run: `pnpm vitest run src/server/llama-swap/client.test.ts && pnpm test`
Expected: PASS, and no existing test regresses — `llamaSwap` still behaves identically.

- [ ] **Step 5: Verify clean**

Run: `pnpm lint:fix && pnpm format:fix && pnpm typecheck`

- [ ] **Step 6: Commit**

```bash
git add src/server/llama-swap/client.ts src/server/llama-swap/client.test.ts
git commit -m "refactor(llama-swap): make the client a factory over base URL

A second node needs its own client instance; the module-level singleton
read config directly and offered no seam to point elsewhere."
```

---

### Task 2: Make config-file helpers take a path

`llama-swap-config.ts` reads `config.inferenceConfigFile` from module scope. Each node has its own `config.yaml`, so the path becomes an argument.

**Files:**
- Modify: `src/server/inference/llama-swap-config.ts`
- Test: `src/server/inference/llama-swap-config.test.ts` (create)

**Interfaces:**
- Produces: `getLlamaSwapModelLogNames(configFile: string, modelId: string): Array<string>`
- Produces: `getLlamaSwapModelConfigSnippet(configFile: string, modelId: string): string | null`
- Produces: `getLlamaSwapConfigContextLengths(configFile: string): Map<string, number>`

An empty `configFile` returns the same not-configured results as today: `[modelId]`, `null`, and an empty `Map`.

- [ ] **Step 1: Write the failing test**

```ts
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import {
  getLlamaSwapConfigContextLengths,
  getLlamaSwapModelConfigSnippet,
  getLlamaSwapModelLogNames,
} from './llama-swap-config'

function writeConfig(body: string): string {
  const path = join(mkdtempSync(join(tmpdir(), 'llama-dash-')), 'config.yaml')
  writeFileSync(path, body, 'utf-8')
  return path
}

describe('llama-swap config helpers', () => {
  const configFile = writeConfig(
    ['models:', '  qwen3:', '    cmd: llama-server --model /models/qwen3.gguf --ctx-size 8192'].join('\n'),
  )

  it('reads context lengths from the given file', () => {
    expect(getLlamaSwapConfigContextLengths(configFile).get('qwen3')).toBe(8192)
  })

  it('reads log-name candidates from the given file', () => {
    expect(getLlamaSwapModelLogNames(configFile, 'qwen3')).toContain('qwen3.gguf')
  })

  it('returns a snippet for a known model', () => {
    expect(getLlamaSwapModelConfigSnippet(configFile, 'qwen3')).toContain('ctx-size')
  })

  it('degrades to not-configured results on an empty path', () => {
    expect(getLlamaSwapConfigContextLengths('')).toEqual(new Map())
    expect(getLlamaSwapModelLogNames('', 'qwen3')).toEqual(['qwen3'])
    expect(getLlamaSwapModelConfigSnippet('', 'qwen3')).toBeNull()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm vitest run src/server/inference/llama-swap-config.test.ts`
Expected: FAIL — the helpers currently take `(modelId)` only, so the context-length assertion reads the wrong argument.

- [ ] **Step 3: Thread the path through**

In `llama-swap-config.ts`: delete the `config` import, change `readParsedConfig()` to `readParsedConfig(configFile: string)` guarding on `if (!configFile) return null`, and add `configFile` as the first parameter of the three exported functions, passing it into `readParsedConfig`.

- [ ] **Step 4: Update the one caller**

`src/server/inference/backends/llama-swap.ts` currently passes these functions by reference:

```ts
modelLogNames: getLlamaSwapModelLogNames,
modelConfigSnippet: getLlamaSwapModelConfigSnippet,
modelContextLengthHints: getLlamaSwapConfigContextLengths,
```

Bind the path (Task 3 replaces `config.inferenceConfigFile` with the node's own):

```ts
modelLogNames: (modelId: string) => getLlamaSwapModelLogNames(config.inferenceConfigFile, modelId),
modelConfigSnippet: (modelId: string) => getLlamaSwapModelConfigSnippet(config.inferenceConfigFile, modelId),
modelContextLengthHints: () => getLlamaSwapConfigContextLengths(config.inferenceConfigFile),
```

- [ ] **Step 5: Run tests and verify clean**

Run: `pnpm test && pnpm lint:fix && pnpm format:fix && pnpm typecheck`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/server/inference/llama-swap-config.ts src/server/inference/llama-swap-config.test.ts src/server/inference/backends/llama-swap.ts
git commit -m "refactor(inference): pass the config path into llama-swap config helpers

Each node has its own config.yaml, so the path can no longer come from
module-scope config."
```

---

### Task 3: Make the backend adapter take a node config

**Files:**
- Modify: `src/server/inference/backends/llama-swap.ts`
- Modify: `src/server/inference/backend.ts`

**Interfaces:**
- Produces: `type LlamaSwapNodeConfig = { baseUrl: string; label: string; configFile: string }`
- Produces: `createLlamaSwapBackend(node: LlamaSwapNodeConfig): InferenceBackend`
- `InferenceBackendInfo.label` becomes the node's label rather than the constant `'llama-swap'`.

- [ ] **Step 1: Change the factory signature**

In `backends/llama-swap.ts`, delete the `config` import and rewrite the two config-reading functions:

```ts
export type LlamaSwapNodeConfig = {
  baseUrl: string
  label: string
  configFile: string
}

function getLlamaSwapInfo(node: LlamaSwapNodeConfig): InferenceBackendInfo {
  const upstreamUrl = new URL(node.baseUrl)
  return {
    kind: 'llama-swap',
    label: node.label,
    upstreamBaseUrl: node.baseUrl,
    upstreamHost: upstreamUrl.host,
    capabilities: {
      models: true,
      runningModels: true,
      lifecycle: true,
      logs: true,
      config: Boolean(node.configFile),
      metrics: true,
    },
  }
}

export function createLlamaSwapBackend(node: LlamaSwapNodeConfig): InferenceBackend {
  const client = createLlamaSwapClient(node.baseUrl)
  // ...body unchanged, but every `llamaSwap.` becomes `client.`,
  // every `config.inferenceBaseUrl` becomes `node.baseUrl`,
  // and the three config helpers bind `node.configFile`.
}
```

Note `config: Boolean(node.configFile)` — this is a real behavior change and it is intended. Per the spec, a node whose `config.yaml` is unreachable reports `config: false` so the UI shows the existing not-configured fallback instead of a broken editor.

- [ ] **Step 2: Keep `backend.ts` compiling**

```ts
function createInferenceBackend(kind: string): InferenceBackend {
  if (kind === 'llama-swap') {
    return createLlamaSwapBackend({
      baseUrl: config.inferenceBaseUrl,
      label: 'llama-swap',
      configFile: config.inferenceConfigFile,
    })
  }
  throw new Error(`Unsupported INFERENCE_BACKEND "${kind}". Supported backends: llama-swap`)
}
```

- [ ] **Step 3: Run tests and verify clean**

Run: `pnpm test && pnpm lint:fix && pnpm format:fix && pnpm typecheck`
Expected: PASS. Behavior is unchanged for a single node except the `config` capability flag, which now reflects whether `INFERENCE_CONFIG_FILE` is set.

- [ ] **Step 4: Commit**

```bash
git add src/server/inference/backends/llama-swap.ts src/server/inference/backend.ts
git commit -m "refactor(inference): construct the llama-swap backend from a node config

Also derives the config capability from whether a config path is set,
rather than hardcoding true."
```

---

### Task 4: Parse node configs from environment

Pure function over an env record, so it is testable without touching `process.env`.

**Files:**
- Create: `src/server/inference/node-config.ts`
- Create: `src/server/inference/node-config.test.ts`

**Interfaces:**
- Produces:

```ts
export type NodeConfig = {
  id: string
  label: string
  baseUrl: string
  configFile: string
  priority: number
}

export function parseNodeConfigs(env: Record<string, string | undefined>): Array<NodeConfig>
export function slugifyNodeLabel(label: string): string
```

Rules: node 1 is `INFERENCE_*`; nodes 2..N are `INFERENCE_<n>_*`, scanned upward from 2, stopping at the first index with no `BASE_URL`. `label` defaults to the URL host. `id` is the slugified label. Duplicate ids throw. Priority is 1-based declaration order.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from 'vitest'
import { parseNodeConfigs, slugifyNodeLabel } from './node-config'

describe('slugifyNodeLabel', () => {
  it('lowercases and hyphenates', () => {
    expect(slugifyNodeLabel('GPU Box')).toBe('gpu-box')
    expect(slugifyNodeLabel('mac-mini.local:8080')).toBe('mac-mini-local-8080')
  })
})

describe('parseNodeConfigs', () => {
  it('returns one node when only the legacy vars are set', () => {
    const nodes = parseNodeConfigs({ INFERENCE_BASE_URL: 'http://localhost:8080' })
    expect(nodes).toEqual([
      { id: 'localhost-8080', label: 'localhost:8080', baseUrl: 'http://localhost:8080', configFile: '', priority: 1 },
    ])
  })

  it('defaults node 1 to localhost:8080 when unset', () => {
    expect(parseNodeConfigs({})[0].baseUrl).toBe('http://localhost:8080')
  })

  it('reads an indexed second node', () => {
    const nodes = parseNodeConfigs({
      INFERENCE_BASE_URL: 'http://gpu-box:8080',
      INFERENCE_LABEL: 'gpu-box',
      INFERENCE_CONFIG_FILE: '/config/config.yaml',
      INFERENCE_2_BASE_URL: 'http://mac-mini:8080',
      INFERENCE_2_LABEL: 'mac-mini',
    })
    expect(nodes.map((n) => n.id)).toEqual(['gpu-box', 'mac-mini'])
    expect(nodes[1]).toMatchObject({ baseUrl: 'http://mac-mini:8080', configFile: '', priority: 2 })
  })

  it('strips trailing slashes from base URLs', () => {
    expect(parseNodeConfigs({ INFERENCE_BASE_URL: 'http://gpu-box:8080///' })[0].baseUrl).toBe('http://gpu-box:8080')
  })

  it('stops scanning at the first gap', () => {
    const nodes = parseNodeConfigs({
      INFERENCE_BASE_URL: 'http://a:8080',
      INFERENCE_3_BASE_URL: 'http://c:8080',
    })
    expect(nodes).toHaveLength(1)
  })

  it('throws on duplicate node ids', () => {
    expect(() =>
      parseNodeConfigs({
        INFERENCE_BASE_URL: 'http://a:8080',
        INFERENCE_LABEL: 'same',
        INFERENCE_2_BASE_URL: 'http://b:8080',
        INFERENCE_2_LABEL: 'same',
      }),
    ).toThrow(/duplicate node id/i)
  })

  it('throws on an unparseable base URL', () => {
    expect(() => parseNodeConfigs({ INFERENCE_BASE_URL: 'not-a-url' })).toThrow(/INFERENCE_BASE_URL/)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm vitest run src/server/inference/node-config.test.ts`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

```ts
const stripTrailingSlash = (s: string) => s.replace(/\/+$/, '')

export type NodeConfig = {
  id: string
  label: string
  baseUrl: string
  configFile: string
  priority: number
}

export function slugifyNodeLabel(label: string): string {
  return label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

/** Env var name for a node index: 1 uses the legacy unindexed names. */
function nodeVar(index: number, suffix: string): string {
  return index === 1 ? `INFERENCE_${suffix}` : `INFERENCE_${index}_${suffix}`
}

export function parseNodeConfigs(env: Record<string, string | undefined>): Array<NodeConfig> {
  const nodes: Array<NodeConfig> = []
  const seen = new Set<string>()

  for (let index = 1; ; index++) {
    const rawBaseUrl = env[nodeVar(index, 'BASE_URL')]
    // Node 1 keeps the historical localhost default; later nodes are opt-in.
    if (rawBaseUrl == null || rawBaseUrl.trim() === '') {
      if (index === 1) {
        nodes.push(makeNode(1, 'http://localhost:8080', env))
        continue
      }
      break
    }
    const node = makeNode(index, stripTrailingSlash(rawBaseUrl.trim()), env)
    if (seen.has(node.id)) {
      throw new Error(`Duplicate node id "${node.id}". Set a distinct ${nodeVar(index, 'LABEL')}.`)
    }
    seen.add(node.id)
    nodes.push(node)
  }

  return nodes
}

function makeNode(index: number, baseUrl: string, env: Record<string, string | undefined>): NodeConfig {
  let host: string
  try {
    host = new URL(baseUrl).host
  } catch {
    throw new Error(`${nodeVar(index, 'BASE_URL')} is not a valid URL: "${baseUrl}"`)
  }
  const label = env[nodeVar(index, 'LABEL')]?.trim() || host
  return {
    id: slugifyNodeLabel(label),
    label,
    baseUrl,
    configFile: env[nodeVar(index, 'CONFIG_FILE')]?.trim() ?? '',
    priority: index,
  }
}
```

The `index === 1` branch pushes the default node then `continue`s, so the loop still checks index 2 — that is what makes an unset `INFERENCE_BASE_URL` plus a set `INFERENCE_2_BASE_URL` behave sensibly. The duplicate-id guard runs only on explicitly configured nodes, which is fine because node 1's default id (`localhost-8080`) can only collide with an explicit node of the same name, and that collision is caught on the later node.

- [ ] **Step 4: Run tests**

Run: `pnpm vitest run src/server/inference/node-config.test.ts`
Expected: PASS (8 tests).

- [ ] **Step 5: Verify clean and commit**

```bash
pnpm lint:fix && pnpm format:fix && pnpm typecheck
git add src/server/inference/node-config.ts src/server/inference/node-config.test.ts
git commit -m "feat(inference): parse indexed node configs from the environment

Nodes are static homelab hosts, so they come from env at boot rather than
a table with CRUD behind it."
```

---

### Task 5: Build the node registry

**Files:**
- Create: `src/server/inference/registry.ts`
- Modify: `src/server/inference/backend.ts`

**Interfaces:**
- Produces:

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

- Produces (shim, retained): `inferenceBackend` from `backend.ts` — now `primaryNode.backend`.

- [ ] **Step 1: Write `registry.ts`**

```ts
import { config } from '../config.ts'
import type { InferenceBackend } from './backend.ts'
import { createLlamaSwapBackend } from './backends/llama-swap.ts'
import { parseNodeConfigs } from './node-config.ts'

export type InferenceNode = {
  id: string
  label: string
  priority: number
  backend: InferenceBackend
}

function createBackendForKind(kind: string, node: { baseUrl: string; label: string; configFile: string }) {
  if (kind === 'llama-swap') return createLlamaSwapBackend(node)
  throw new Error(`Unsupported INFERENCE_BACKEND "${kind}". Supported backends: llama-swap`)
}

export const nodes: ReadonlyArray<InferenceNode> = parseNodeConfigs(process.env).map((node) => ({
  id: node.id,
  label: node.label,
  priority: node.priority,
  backend: createBackendForKind(config.inferenceBackend, node),
}))

export const primaryNode: InferenceNode = nodes[0]

const byId = new Map(nodes.map((node) => [node.id, node]))

export function getNode(id: string): InferenceNode | undefined {
  return byId.get(id)
}
```

Registry construction happens at import time, matching the existing singleton. A malformed `INFERENCE_2_BASE_URL` therefore fails at boot — intentional, and consistent with the existing `Unsupported INFERENCE_BACKEND` behavior.

- [ ] **Step 2: Reduce `backend.ts` to types plus the shim**

Keep every exported type. Replace the bottom of the file:

```ts
// Deprecated: prefer `primaryNode` / `getNode()` from ./registry.ts.
// Retained so upstream merges touching inferenceBackend still apply cleanly.
export { primaryNode } from './registry.ts'
import { primaryNode } from './registry.ts'
export const inferenceBackend: InferenceBackend = primaryNode.backend
```

`createInferenceBackend` moves into `registry.ts` as `createBackendForKind` and is deleted from `backend.ts`. Watch for an import cycle: `registry.ts` imports the `InferenceBackend` *type* from `backend.ts` (erased at runtime) and `backend.ts` imports the `primaryNode` *value* from `registry.ts`. Use `import type` in `registry.ts` so the cycle has no runtime edge.

- [ ] **Step 3: Run the full suite**

Run: `pnpm test`
Expected: PASS. `handler.test.ts` mocks `config`, which still drives node 1, so it should be unaffected. If it fails on registry import, add a `vi.mock('../inference/registry.ts')` returning a single fake node.

- [ ] **Step 4: Verify clean and commit**

```bash
pnpm lint:fix && pnpm format:fix && pnpm typecheck
git add src/server/inference/registry.ts src/server/inference/backend.ts
git commit -m "feat(inference): add a node registry behind the existing singleton

inferenceBackend stays exported as primaryNode.backend so the eight
consuming modules migrate incrementally and upstream merges still apply."
```

---

### Task 6: Per-node model events

Two nodes running the same model name currently cancel each other's load/unload events. This is the highest-risk item in the design because it fails silently.

**Files:**
- Modify: `src/server/db/schema.ts`
- Modify: `src/server/model-watcher.ts`
- Create: `src/server/model-watcher.test.ts`
- Generated: `drizzle/*.sql`

**Interfaces:**
- Consumes: `nodes` from `registry.ts` (Task 5).
- Produces: `model_events.node_id` column, nullable text.

- [ ] **Step 1: Add the column**

In `src/server/db/schema.ts`, inside the `modelEvents` table definition:

```ts
nodeId: text('node_id'),
```

Nullable — existing rows predate nodes and must not be rewritten.

- [ ] **Step 2: Generate and apply the migration**

```bash
pnpm db:generate
pnpm db:migrate
```

Read the generated SQL in `drizzle/` before continuing. Expect a single `ALTER TABLE model_events ADD `node_id` text;`. If drizzle proposes a table rebuild instead, stop and investigate — a rebuild risks the existing event history.

- [ ] **Step 3: Write the failing test**

```ts
import { describe, expect, it } from 'vitest'
import { diffNodeRunning } from './model-watcher'

describe('diffNodeRunning', () => {
  it('emits a load for a newly running model', () => {
    const events = diffNodeRunning('gpu-box', new Set(), new Set(['qwen3']))
    expect(events).toEqual([{ nodeId: 'gpu-box', modelId: 'qwen3', event: 'load' }])
  })

  it('emits an unload for a model that stopped', () => {
    const events = diffNodeRunning('gpu-box', new Set(['qwen3']), new Set())
    expect(events).toEqual([{ nodeId: 'gpu-box', modelId: 'qwen3', event: 'unload' }])
  })

  it('emits nothing when state is unchanged', () => {
    expect(diffNodeRunning('gpu-box', new Set(['qwen3']), new Set(['qwen3']))).toEqual([])
  })

  it('does not let one node cancel another running the same model', () => {
    // gpu-box keeps qwen3 running; mac-mini starts its own copy.
    const gpu = diffNodeRunning('gpu-box', new Set(['qwen3']), new Set(['qwen3']))
    const mac = diffNodeRunning('mac-mini', new Set(), new Set(['qwen3']))
    expect(gpu).toEqual([])
    expect(mac).toEqual([{ nodeId: 'mac-mini', modelId: 'qwen3', event: 'load' }])
  })
})
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pnpm vitest run src/server/model-watcher.test.ts`
Expected: FAIL — `diffNodeRunning` is not exported.

- [ ] **Step 5: Rewrite the watcher around per-node state**

Extract the pure diff so it is testable, and key state by node:

```ts
export type NodeModelEvent = { nodeId: string; modelId: string; event: 'load' | 'unload' }

/** Pure diff of one node's running set against its last known set. */
export function diffNodeRunning(nodeId: string, known: Set<string>, current: Set<string>): Array<NodeModelEvent> {
  const events: Array<NodeModelEvent> = []
  for (const modelId of current) {
    if (!known.has(modelId)) events.push({ nodeId, modelId, event: 'load' })
  }
  for (const modelId of known) {
    if (!current.has(modelId)) events.push({ nodeId, modelId, event: 'unload' })
  }
  return events
}
```

Replace the module-level `let knownRunning = new Set<string>()` with `const knownRunning = new Map<string, Set<string>>()` keyed by node id.

`seedFromDb()` must group by node: select `nodeId`, `modelId`, and `event` ordered by `timestamp desc`, and track the newest row per `${nodeId}::${modelId}` pair. Rows with a null `nodeId` seed the primary node, so history written before this change is attributed to node 1 rather than dropped.

`diffRunning()` iterates `nodes`, calls each backend's `listRunning()`, diffs with `diffNodeRunning`, and inserts rows carrying `nodeId`. Wrap each node's poll in its own `try/catch` — one unreachable node must not stop the other from being polled, which is exactly what a single shared `try` around the whole loop would do.

- [ ] **Step 6: Run tests and verify clean**

Run: `pnpm vitest run src/server/model-watcher.test.ts && pnpm test && pnpm lint:fix && pnpm format:fix && pnpm typecheck`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/server/db/schema.ts src/server/model-watcher.ts src/server/model-watcher.test.ts drizzle/
git commit -m "feat(models): attribute load/unload events to a node

Keying known-running state on model id alone let two nodes running the
same model cancel each other's events, corrupting the timeline silently."
```

---

### Task 7: Model ownership index

**Files:**
- Create: `src/server/inference/model-index.ts`
- Create: `src/server/inference/model-index.test.ts`
- Modify: `src/server/model-watcher.ts`

**Interfaces:**
- Produces:

```ts
export function setModelOwners(modelId: string, nodeIds: Array<string>): void
export function replaceModelIndex(entries: Map<string, Array<string>>): void
export function getOwningNodeId(modelId: string): string | null
export function getModelOwners(modelId: string): ReadonlyArray<string>
```

`getOwningNodeId` returns the highest-priority owner, or `null` if the model is unknown. Callers translate `null` into "use the primary node".

- [ ] **Step 1: Write the failing test**

```ts
import { beforeEach, describe, expect, it } from 'vitest'
import { getModelOwners, getOwningNodeId, replaceModelIndex } from './model-index'

describe('model index', () => {
  beforeEach(() => replaceModelIndex(new Map()))

  it('returns null for an unknown model', () => {
    expect(getOwningNodeId('nope')).toBeNull()
  })

  it('returns the only owner', () => {
    replaceModelIndex(new Map([['qwen3', ['mac-mini']]]))
    expect(getOwningNodeId('qwen3')).toBe('mac-mini')
  })

  it('returns the first owner when several nodes serve the model', () => {
    replaceModelIndex(new Map([['qwen3', ['gpu-box', 'mac-mini']]]))
    expect(getOwningNodeId('qwen3')).toBe('gpu-box')
    expect(getModelOwners('qwen3')).toEqual(['gpu-box', 'mac-mini'])
  })

  it('drops models that disappear on refresh', () => {
    replaceModelIndex(new Map([['qwen3', ['gpu-box']]]))
    replaceModelIndex(new Map([['gemma3', ['mac-mini']]]))
    expect(getOwningNodeId('qwen3')).toBeNull()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm vitest run src/server/inference/model-index.test.ts`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

```ts
/**
 * modelId -> node ids that serve it, ordered by node priority.
 * Refreshed on the model-watcher tick; read synchronously by the proxy.
 */
let index = new Map<string, Array<string>>()

export function replaceModelIndex(entries: Map<string, Array<string>>): void {
  index = entries
}

export function setModelOwners(modelId: string, nodeIds: Array<string>): void {
  index.set(modelId, nodeIds)
}

export function getModelOwners(modelId: string): ReadonlyArray<string> {
  return index.get(modelId) ?? []
}

export function getOwningNodeId(modelId: string): string | null {
  return index.get(modelId)?.[0] ?? null
}
```

- [ ] **Step 4: Populate it from the watcher tick**

In `model-watcher.ts`, after the per-node poll loop, build the index from each node's `listModels()` and hand it to `replaceModelIndex`. Iterate `nodes` in priority order so the owner arrays come out ordered. A node that throws contributes nothing to that refresh but must not clear the whole index — accumulate into a fresh map and only call `replaceModelIndex` if at least one node responded.

- [ ] **Step 5: Run tests, verify clean, commit**

```bash
pnpm vitest run src/server/inference/model-index.test.ts && pnpm test
pnpm lint:fix && pnpm format:fix && pnpm typecheck
git add src/server/inference/model-index.ts src/server/inference/model-index.test.ts src/server/model-watcher.ts
git commit -m "feat(inference): index model ownership per node

Reuses the existing 15s watcher tick so proxy dispatch stays synchronous
and no second poller is introduced."
```

---

### Task 8: Add a `node` routing target

**Files:**
- Modify: `src/lib/schemas/routing-rule.ts`
- Modify: `src/server/proxy/transforms.ts` (`RoutingOutcome`)
- Modify: `src/server/admin/routing-rules.ts`
- Test: `src/server/admin/routing-rules.test.ts` (extend)

**Interfaces:**
- Produces: `NodeTargetSchema = v.object({ type: v.literal('node'), nodeId: v.pipe(v.string(), v.minLength(1), v.maxLength(120)) })`
- Produces: `RoutingOutcome.targetType` widens to `'llama_swap' | 'direct' | 'node' | null`
- Produces: `RoutingOutcome.targetNodeId: string | null`

- [ ] **Step 1: Write the failing test**

Add to `src/server/admin/routing-rules.test.ts`, matching the existing `makeRule` helper style in that file:

```ts
it('accepts a node target', () => {
  const parsed = v.safeParse(RoutingTargetSchema, { type: 'node', nodeId: 'mac-mini' })
  expect(parsed.success).toBe(true)
})

it('rejects a node target with an empty id', () => {
  const parsed = v.safeParse(RoutingTargetSchema, { type: 'node', nodeId: '' })
  expect(parsed.success).toBe(false)
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm vitest run src/server/admin/routing-rules.test.ts`
Expected: FAIL — the variant has no `node` member.

- [ ] **Step 3: Extend the schema**

```ts
export const NodeTargetSchema = v.object({
  type: v.literal('node'),
  nodeId: v.pipe(v.string(), v.minLength(1), v.maxLength(120)),
})

export const RoutingTargetSchema = v.variant('type', [LlamaSwapTargetSchema, DirectTargetSchema, NodeTargetSchema])
```

Leave `isSafeRoutingRule` alone — its `direct` checks are unrelated, and a node target carries no URL or credential, so it needs no safety gate here. Validate the id against the registry at write time instead (next step).

- [ ] **Step 4: Validate node ids on write**

In `src/server/admin/routing-rules.ts`, where a rule is created or updated, reject a `node` target whose `nodeId` is not in the registry:

```ts
if (target.type === 'node' && !getNode(target.nodeId)) {
  return { ok: false, error: `Unknown node "${target.nodeId}"` }
}
```

Match the surrounding error-return convention in that file rather than throwing.

- [ ] **Step 5: Carry the node id on the outcome**

Add `targetNodeId: string | null` to `RoutingOutcome`, default `null` in `emptyRoutingOutcome()`, and populate it in `routingOutcomeFromDecision()` from a `node` target. Widen `targetType` to include `'node'`.

- [ ] **Step 6: Run tests, verify clean, commit**

```bash
pnpm test && pnpm lint:fix && pnpm format:fix && pnpm typecheck
git add src/lib/schemas/routing-rule.ts src/server/proxy/transforms.ts src/server/admin/routing-rules.ts src/server/admin/routing-rules.test.ts
git commit -m "feat(routing): add a node routing target

Pins a model to a specific node when both serve it. Validated against the
registry rather than the direct-upstream host allow-list."
```

---

### Task 9: Dispatch to the owning node

`selectUpstream()` runs from `context.ts:90,93`, after transforms resolve aliases and `rewrite_model`, so the resolved model is available. This task does not touch the auth or body-buffering flow.

**Files:**
- Modify: `src/server/proxy/upstream.ts`
- Modify: `src/server/proxy/context.ts`
- Create: `src/server/proxy/upstream.test.ts`

**Interfaces:**
- Produces:

```ts
export type UpstreamSelection = { url: string; nodeId: string | null }

export function selectUpstream(
  routing: Pick<RoutingOutcome, 'targetType' | 'targetBaseUrl' | 'targetNodeId'>,
  endpoint: string,
  search: string,
  resolvedModel: string | null,
): UpstreamSelection
```

- Produces: `ProxyContext.nodeId: string | null`. `ProxyContext.defaultUpstream` is **removed** — it is referenced only within `context.ts` (lines 28, 54, 55, 90, 93).

Resolution order: direct target → node target → model index → primary node. `nodeId` is `null` for a direct (cloud) target, which is what tells the log that no local node served the request.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it, vi, beforeEach } from 'vitest'

vi.mock('../inference/registry.ts', () => {
  const node = (id: string, baseUrl: string) => ({
    id,
    label: id,
    priority: id === 'gpu-box' ? 1 : 2,
    backend: { defaultProxyUpstream: (p: string, s: string) => `${baseUrl}${p}${s}` },
  })
  const gpu = node('gpu-box', 'http://gpu-box:8080')
  const mac = node('mac-mini', 'http://mac-mini:8080')
  return {
    nodes: [gpu, mac],
    primaryNode: gpu,
    getNode: (id: string) => ({ 'gpu-box': gpu, 'mac-mini': mac })[id],
  }
})

import { replaceModelIndex } from '../inference/model-index'
import { selectUpstream } from './upstream'

const noTarget = { targetType: null, targetBaseUrl: null, targetNodeId: null }

describe('selectUpstream', () => {
  beforeEach(() => replaceModelIndex(new Map()))

  it('falls back to the primary node for an unknown model', () => {
    expect(selectUpstream(noTarget, '/v1/chat/completions', '', 'unknown')).toEqual({
      url: 'http://gpu-box:8080/v1/chat/completions',
      nodeId: 'gpu-box',
    })
  })

  it('routes to the node that owns the model', () => {
    replaceModelIndex(new Map([['gemma3', ['mac-mini']]]))
    expect(selectUpstream(noTarget, '/v1/chat/completions', '', 'gemma3')).toEqual({
      url: 'http://mac-mini:8080/v1/chat/completions',
      nodeId: 'mac-mini',
    })
  })

  it('breaks a tie by node priority', () => {
    replaceModelIndex(new Map([['qwen3', ['gpu-box', 'mac-mini']]]))
    expect(selectUpstream(noTarget, '/v1/chat/completions', '', 'qwen3').nodeId).toBe('gpu-box')
  })

  it('lets a node target override the index', () => {
    replaceModelIndex(new Map([['qwen3', ['gpu-box']]]))
    const routing = { targetType: 'node', targetBaseUrl: null, targetNodeId: 'mac-mini' }
    expect(selectUpstream(routing, '/v1/chat/completions', '', 'qwen3').nodeId).toBe('mac-mini')
  })

  it('lets a direct target win over everything and reports no node', () => {
    replaceModelIndex(new Map([['qwen3', ['mac-mini']]]))
    const routing = { targetType: 'direct', targetBaseUrl: 'https://api.anthropic.com/v1', targetNodeId: null }
    const result = selectUpstream(routing, '/v1/messages', '', 'qwen3')
    expect(result.nodeId).toBeNull()
    expect(result.url).toBe('https://api.anthropic.com/v1/messages')
  })

  it('falls back to the primary node when the target node id is unknown', () => {
    const routing = { targetType: 'node', targetBaseUrl: null, targetNodeId: 'deleted-node' }
    expect(selectUpstream(routing, '/v1/chat/completions', '', null).nodeId).toBe('gpu-box')
  })

  it('uses the primary node when no model is resolved', () => {
    expect(selectUpstream(noTarget, '/v1/models', '', null).nodeId).toBe('gpu-box')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm vitest run src/server/proxy/upstream.test.ts`
Expected: FAIL — `selectUpstream` currently takes `defaultUpstream` first and returns a string.

- [ ] **Step 3: Implement the resolution order**

```ts
export function selectUpstream(
  routing: Pick<RoutingOutcome, 'targetType' | 'targetBaseUrl' | 'targetNodeId'>,
  endpoint: string,
  search: string,
  resolvedModel: string | null,
): UpstreamSelection {
  if (routing.targetType === 'direct' && routing.targetBaseUrl) {
    return { url: buildDirectUpstream(routing.targetBaseUrl, endpoint, search), nodeId: null }
  }

  const pinned = routing.targetType === 'node' && routing.targetNodeId ? getNode(routing.targetNodeId) : undefined
  const owner = resolvedModel ? getNode(getOwningNodeId(resolvedModel) ?? '') : undefined
  const node = pinned ?? owner ?? primaryNode

  return { url: node.backend.defaultProxyUpstream(endpoint, search), nodeId: node.id }
}
```

An unknown pinned node id falls through to the owner or primary rather than erroring — a rule referencing a node deleted from `.env` degrades to default routing instead of 500ing every matched request.

- [ ] **Step 4: Update `context.ts`**

- Delete `defaultUpstream` from the `ProxyContext` type and from `createProxyContext`.
- Add `nodeId: string | null`, initialised to `null`.
- Initialise `upstream` in `createProxyContext` from `primaryNode.backend.defaultProxyUpstream(url.pathname, url.search)` so a request that never reaches `finalizeRoutingAndBody` still has a usable upstream.
- Replace both `selectUpstream` calls in `finalizeRoutingAndBody`:

```ts
const selection = selectUpstream(ctx.routingOutcome, ctx.endpoint, ctx.url.search, ctx.body?.reqModel ?? null)
ctx.upstream = selection.url
ctx.nodeId = selection.nodeId
```

Both branches of that function need it — the early `if (!ctx.body.hasBody)` return and the main path.

- Swap the `inferenceBackend` import for `primaryNode` from `../inference/registry.ts`.

- [ ] **Step 5: Run tests, verify clean, commit**

```bash
pnpm vitest run src/server/proxy/upstream.test.ts && pnpm test
pnpm lint:fix && pnpm format:fix && pnpm typecheck
git add src/server/proxy/upstream.ts src/server/proxy/upstream.test.ts src/server/proxy/context.ts
git commit -m "feat(proxy): dispatch to the node that owns the requested model

Selection happens after transforms resolve the model, so the auth and
body-buffering flow is untouched."
```

---

### Task 10: Log which node served each request

**Files:**
- Modify: `src/server/db/schema.ts`
- Modify: `src/server/proxy/forward.ts`
- Modify: `src/server/proxy/log.ts`
- Modify: `src/server/proxy/handler.ts`
- Generated: `drizzle/*.sql`

**Interfaces:**
- Consumes: `ctx.nodeId` (Task 9).
- Produces: `requests.node_id`, nullable text. `ProxyLogInput.nodeId: string | null`.

No index on `node_id`: with two nodes the cardinality is 2, so an index would cost writes and buy nothing.

- [ ] **Step 1: Add the column and migrate**

Add `nodeId: text('node_id'),` to the `requests` table in `schema.ts`, then:

```bash
pnpm db:generate && pnpm db:migrate
```

Read the generated SQL. Expect one `ALTER TABLE requests ADD `node_id` text;`.

- [ ] **Step 2: Thread it through the log input**

Add `nodeId: string | null` to `ProxyLogInput` in `forward.ts`, pass it into `writeRequestLog` inside `writeProxyLog`, and add it to the row written in `log.ts`.

- [ ] **Step 3: Populate it at every call site**

`handler.ts` calls `writeProxyLog` in six places (auth failure, transform rejection, credential-key-required, credential-injection failure, upstream error, and `rejectBodyTooLarge`) plus `forwardUpstreamAndLog`. Add `nodeId: ctx.nodeId` to each.

For the paths that run *before* `finalizeRoutingAndBody`, `ctx.nodeId` is still `null` — correct, because no node was selected and none served the request.

- [ ] **Step 4: Verify end to end against a real node**

This is the first task whose result is invisible to unit tests, so verify it live:

```bash
pnpm dev &
sleep 5
curl -s localhost:5173/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"<a model on node 1>","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' > /dev/null
sqlite3 data/dash.db 'select id, model, node_id from requests order by id desc limit 1;'
kill %1
```

Expected: `node_id` matches node 1's slug. Kill the dev server before continuing — do not leave it bound to :5173.

- [ ] **Step 5: Verify clean and commit**

```bash
pnpm test && pnpm lint:fix && pnpm format:fix && pnpm typecheck
git add src/server/db/schema.ts src/server/proxy/forward.ts src/server/proxy/log.ts src/server/proxy/handler.ts drizzle/
git commit -m "feat(requests): record which node served each request"
```

---

### Task 11: Per-node admin API

**Files:**
- Modify: `src/server/admin/routes/models.ts`
- Modify: `src/server/admin/routes/system.ts`
- Modify: `src/server/admin/routes/config.ts`
- Modify: `src/server/admin/model-detail.ts`
- Modify: `src/lib/schemas/model.ts` — model entries gain `nodes`
- Modify: `src/lib/schemas/system.ts` — system response gains `nodes`
- Modify: `src/lib/schemas/health.ts` — health response gains per-node entries
- Modify: `src/lib/auth-functions.ts`

**Interfaces:**
- Produces: each entry in the `/api/models` response gains `nodes: Array<{ id: string; label: string }>`.
- Produces: `/api/system` gains `nodes: Array<{ id, label, host, capabilities, reachable, version, latencyMs, error }>`, keeping the existing single `upstream` field populated from the primary node for backwards compatibility.
- Produces: `/api/models/:id/load`, `/unload`, and `/api/config` accept an optional `?node=<id>`, defaulting to primary.

- [ ] **Step 1: Update the response schemas first**

Schemas are the source of truth for API types in this repo — write the valibot schema, then derive the type with `v.InferOutput`. Never hand-write a type that duplicates a schema.

- [ ] **Step 2: Fan out `/api/models`**

Query every node's `listModels()` and `listRunning()` with `Promise.allSettled`, so one unreachable node yields a partial list rather than a failed page. Merge by model id; a model present on both nodes becomes one entry with two entries in `nodes`. Order `nodes` by node priority so it matches dispatch.

- [ ] **Step 3: Fan out `/api/system` and `/api/health`**

One `health()` call per node, again with `Promise.allSettled`. Keep `upstream` populated from the primary node so nothing that reads it breaks before Task 12 updates the UI.

- [ ] **Step 4: Scope lifecycle and config routes by node**

`load`, `unload`, `unloadAll`, `modelConfigSnippet`, log names, and context hints resolve their node from `?node=`, falling back to primary. An unknown node id returns `error(404, 'Unknown node')`. A node whose `capabilities.config` is false returns the existing `501` from `routes/config.ts`.

- [ ] **Step 5: Expose per-node capabilities to the client**

`src/lib/auth-functions.ts:21` currently sends `inference: { capabilities }` from the singleton. Send an array of `{ id, label, capabilities }` instead. The UI needs per-node flags to decide whether to show the config editor and logs for a given node.

- [ ] **Step 6: Verify against both nodes**

Configure `INFERENCE_2_*` in `.env` pointing at your second box, then:

```bash
pnpm dev &
sleep 5
curl -s localhost:5173/api/system | jq '.nodes[] | {id, reachable}'
curl -s localhost:5173/api/models | jq '[.[] | {id, nodes: [.nodes[].id]}] | .[0:5]'
kill %1
```

Expected: both nodes listed; models from both boxes present. Then stop node 2 and re-run — expect node 1's models still listed and node 2 marked unreachable, not a 500.

- [ ] **Step 7: Verify clean and commit**

```bash
pnpm test && pnpm lint:fix && pnpm format:fix && pnpm typecheck
git add src/server/admin/ src/lib/schemas/ src/lib/auth-functions.ts
git commit -m "feat(admin): report models, health, and capabilities per node

Uses allSettled so one unreachable node degrades to a partial response
rather than failing the whole page."
```

---

### Task 12: Node identity in the UI

**Files:**
- Modify: `src/features/models/ModelRow.tsx`, `ModelsPage.tsx`
- Modify: `src/features/dashboard/DashboardRunningModelsPanel.tsx`
- Modify: `src/features/requests/` (list filters and the detail panel)
- Modify: `src/features/logs/LogsPage.tsx`
- Modify: `src/routes/system.tsx` or its feature components

**Interfaces:**
- Consumes: `nodes` on `/api/models` entries and `/api/system` (Task 11); `nodeId` on request rows (Task 10).

- [ ] **Step 1: Read the existing series-colour pattern**

Read `src/features/dashboard/DashboardResidencyPanel.tsx:38,100`. It calls `assignSeriesSteps()` over a stable id list and `seriesVar(step)` for the colour. Node badges follow exactly this: assign steps over the sorted node id list once, reuse everywhere. Node identity is a category, so it must never use `--ok`/`--warn`/`--err`.

- [ ] **Step 2: Add node badges to the Models list**

One badge per node in the entry's `nodes` array. A model on both nodes shows one row with two badges.

- [ ] **Step 3: One row per node on the System page**

Each node gets its own health/version/latency row and its own capability flags. Label the GPU panel with the host it describes — per the spec, `gpu-poller.ts` polls the llama-dash host only and cannot describe node 2. Do not render an empty or zeroed GPU panel for node 2; omit it and say why.

- [ ] **Step 4: Node column and filter on Requests**

Add a node column to the list and a node filter alongside the existing routing and attribution filters. Rows with a null `node_id` display as `—`.

- [ ] **Step 5: Node selector on the Logs page**

`/api/log-events` takes `?node=`, defaulting to primary. Only offer nodes whose `capabilities.logs` is true.

- [ ] **Step 6: Verify the rendered UI**

Layout regressions in this repo are usually lost `min-h-0` / `flex-1` / `h-full` in nested flex containers, which static checks will not catch. Start the dev server, load `/models`, `/system`, `/requests`, and `/logs`, and confirm each main column still stretches to the bottom and scroll regions still own their overflow. Kill the server when done.

- [ ] **Step 7: Verify clean and commit**

```bash
pnpm lint:fix && pnpm format:fix && pnpm typecheck && pnpm test
git add src/features/ src/routes/
git commit -m "feat(ui): surface node identity across models, system, requests, logs"
```

---

### Task 13: Merge `GET /v1/models`

The proxy has no handler for this route today, so clients would see only the primary node's catalog. Models on node 2 stay reachable by name but undiscoverable in any client dropdown.

This is the first `/v1/*` route llama-dash answers rather than relays.

Each returned model reports `owned_by` set to its node's **label**, replacing whatever the upstream sent. `owned_by` is already required on `OpenAiModelSchema` (`src/server/llama-swap/schemas.ts:7`), so no shape changes and no client breaks. Model ids stay bare — prefixed addressable ids were considered and rejected.

**Files:**
- Create: `src/server/proxy/models-merge.ts`
- Create: `src/server/proxy/models-merge.test.ts`
- Modify: `src/server/proxy/handler.ts`

**Interfaces:**
- Produces: `mergeNodeModels(results: Array<{ nodeLabel: string; models: Array<OpenAiModel> } | null>): { object: 'list'; data: Array<OpenAiModel> }`
- Produces: `handleMergedModelsRequest(): Promise<Response>`

Callers pass results in node-priority order. `nodeLabel` (not id) is used because `owned_by` is a human-facing field.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from 'vitest'
import { mergeNodeModels } from './models-merge'

const model = (id: string, ownedBy = 'llama-swap') =>
  ({ id, object: 'model', created: 0, owned_by: ownedBy }) as never

describe('mergeNodeModels', () => {
  it('concatenates models across nodes', () => {
    const merged = mergeNodeModels([
      { nodeLabel: 'gpu-box', models: [model('qwen3')] },
      { nodeLabel: 'mac-mini', models: [model('gemma3')] },
    ])
    expect(merged.data.map((m) => m.id)).toEqual(['qwen3', 'gemma3'])
    expect(merged.object).toBe('list')
  })

  it('stamps owned_by with the node label', () => {
    const merged = mergeNodeModels([
      { nodeLabel: 'gpu-box', models: [model('qwen3')] },
      { nodeLabel: 'mac-mini', models: [model('gemma3')] },
    ])
    expect(merged.data.map((m) => m.owned_by)).toEqual(['gpu-box', 'mac-mini'])
  })

  it('overwrites whatever owned_by the upstream sent', () => {
    const merged = mergeNodeModels([{ nodeLabel: 'gpu-box', models: [model('qwen3', 'organization-owner')] }])
    expect(merged.data[0].owned_by).toBe('gpu-box')
  })

  it('dedupes a model served by both nodes, keeping the higher-priority node', () => {
    const merged = mergeNodeModels([
      { nodeLabel: 'gpu-box', models: [model('qwen3')] },
      { nodeLabel: 'mac-mini', models: [model('qwen3')] },
    ])
    expect(merged.data).toHaveLength(1)
    // owned_by names where dispatch will actually send it, not merely where it exists.
    expect(merged.data[0].owned_by).toBe('gpu-box')
  })

  it('does not mutate the caller-supplied model objects', () => {
    const original = model('qwen3', 'llama-swap')
    mergeNodeModels([{ nodeLabel: 'gpu-box', models: [original] }])
    expect((original as { owned_by: string }).owned_by).toBe('llama-swap')
  })

  it('returns surviving nodes when one failed', () => {
    const merged = mergeNodeModels([null, { nodeLabel: 'mac-mini', models: [model('gemma3')] }])
    expect(merged.data.map((m) => m.id)).toEqual(['gemma3'])
  })

  it('returns an empty list when every node failed', () => {
    expect(mergeNodeModels([null, null])).toEqual({ object: 'list', data: [] })
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm vitest run src/server/proxy/models-merge.test.ts`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the merge**

First-wins dedupe matches how `selectUpstream` breaks ties, because callers pass results in node-priority order. Copy each model before stamping `owned_by` (`{ ...m, owned_by: nodeLabel }`) — mutating the caller's objects would corrupt any cached upstream response, which the last test guards against.

`handleMergedModelsRequest()` fans out with `Promise.allSettled`, maps rejections to `null`, and passes results in node-priority order.

**Which `listModels` to call.** The backend facade's `listModels()` returns `BackendModel`, which is normalized and has no `owned_by` or `created` — it cannot produce an OpenAI-shaped response. You need the client's `listModels()`, which returns the full `OpenAiModel`. Build one per node from the base URL the registry already exposes:

```ts
import { createLlamaSwapClient } from '../llama-swap/client.ts'
import { nodes } from '../inference/registry.ts'

const perNode = await Promise.allSettled(
  nodes.map(async (node) => ({
    nodeLabel: node.label,
    models: (await createLlamaSwapClient(node.backend.info.upstreamBaseUrl).listModels()).data,
  })),
)
```

This names llama-swap directly inside the proxy layer, which the facade normally prevents. It is acceptable here only because a second backend *kind* is an explicit non-goal in the spec. Add a comment saying so, and saying that a second kind would need an optional `listOpenAiModels?()` on the facade instead. Do not add that method now — there is no second kind to justify it.

- [ ] **Step 4: Intercept the route in `handler.ts`**

At the top of `handleProxyRequest`, before `createProxyContext`:

```ts
const url = new URL(request.url)
if (request.method === 'GET' && url.pathname === '/v1/models') {
  return handleMergedModelsRequest()
}
```

Two consequences to accept deliberately: this response is not written to the request log (it is llama-dash's own answer, not a proxied exchange), and it bypasses API-key enforcement exactly as the forwarded version effectively did. If key enforcement on model listing is wanted, that is a separate change.

- [ ] **Step 5: Verify against both nodes**

```bash
pnpm dev &
sleep 5
curl -s localhost:5173/v1/models | jq '.data | length'
curl -s localhost:5173/v1/models | jq -r '.data[].id' | sort | head -20
kill %1
```

Expected: the union of both boxes' models, deduped. Stop node 2 and re-run — expect node 1's models still returned, not an error.

- [ ] **Step 6: Verify clean and commit**

```bash
pnpm test && pnpm lint:fix && pnpm format:fix && pnpm typecheck
git add src/server/proxy/models-merge.ts src/server/proxy/models-merge.test.ts src/server/proxy/handler.ts
git commit -m "feat(proxy): merge GET /v1/models across nodes

Without this, half the catalog is reachable by name but invisible to every
client dropdown. First /v1 route llama-dash answers rather than relays."
```

---

### Task 14: Label metrics by node

Without a `node` label Prometheus sums both hosts into one series — wrong, and silent.

**Files:**
- Modify: `src/server/metrics.ts`
- Modify: `src/server/metrics.test.ts`

**Interfaces:**
- Consumes: `nodes` from the registry.

- [ ] **Step 1: Read the affected gauges**

`metrics.ts:75` calls `inferenceBackend.ping()`; `metrics.ts:124` calls `inferenceBackend.listRunning?.()`. They feed exactly three gauges, emitted at `metrics.ts:180-189`:

```text
llama_dash_upstream_reachable
llama_dash_upstream_latency_seconds
llama_dash_models_running
```

These three become per-node. Everything else in the exporter stays as-is.

- [ ] **Step 2: Write the failing test**

Extend `src/server/metrics.test.ts`, mocking the registry the same way `upstream.test.ts` does (Task 9, Step 1):

```ts
it('emits one upstream series per node', async () => {
  const text = await renderMetrics()
  const reachable = text.split('\n').filter((line) => line.startsWith('llama_dash_upstream_reachable{'))
  expect(reachable).toHaveLength(2)
  expect(reachable.join('\n')).toContain('node="gpu-box"')
  expect(reachable.join('\n')).toContain('node="mac-mini"')
})

it('emits running-model counts per node', async () => {
  const text = await renderMetrics()
  expect(text).toContain('llama_dash_models_running{node="gpu-box"}')
  expect(text).toContain('llama_dash_models_running{node="mac-mini"}')
})

it('leaves llama-dash-owned counters unlabelled', async () => {
  const text = await renderMetrics()
  expect(text).toContain('llama_dash_log_queue_depth ')
  expect(text).not.toContain('llama_dash_log_queue_depth{node=')
})
```

Replace `renderMetrics` with whatever the exporter function is actually named in `metrics.ts` — read the existing test file for the current call convention before writing this.

- [ ] **Step 3: Run test to verify it fails**

Run: `pnpm vitest run src/server/metrics.test.ts`
Expected: FAIL — one unlabelled series per gauge, so the length assertion sees 0 matches for the `{`-suffixed prefix.

- [ ] **Step 4: Fan out the three gauges**

Loop the registry with `Promise.allSettled`, emitting one sample per node via the existing `metricLine(name, value, labels)` helper with `{ node: node.id }`. An unreachable node still emits `llama_dash_upstream_reachable{node="..."} 0` — a missing series and a zero series mean different things to Prometheus, and "down" is the useful signal.

Keep cardinality low: node id only, never host or URL. Leave request, token, latency, and queue metrics unlabelled — those count llama-dash's own work, not any node's.

- [ ] **Step 5: Verify the exporter output**

```bash
pnpm dev &
sleep 5
curl -s localhost:5173/metrics | grep -E 'node='
kill %1
```

Expected: reachability and running-model gauges present once per node.

- [ ] **Step 6: Verify clean and commit**

```bash
pnpm test && pnpm lint:fix && pnpm format:fix && pnpm typecheck
git add src/server/metrics.ts src/server/metrics.test.ts
git commit -m "feat(metrics): label upstream and running-model gauges by node"
```

---

### Task 15: Documentation and deployment config

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `docker-compose.amd.yaml`, `docker-compose.nvidia.yaml`
- Modify: `docs/2026_05_03_inference_backends.md`

- [ ] **Step 1: Document the env vars in `.env.example`**

```text
# Node 1 (required). Legacy unindexed names.
INFERENCE_BASE_URL=http://localhost:8080
INFERENCE_LABEL=
INFERENCE_CONFIG_FILE=

# Node 2+ (optional). Scanned upward from 2; the first gap stops the scan.
# INFERENCE_2_BASE_URL=http://mac-mini:8080
# INFERENCE_2_LABEL=mac-mini
# INFERENCE_2_CONFIG_FILE=
```

Note in a comment that `INFERENCE_INSECURE` is process-wide (`src/server.ts:6-7`) and that `INFERENCE_<n>_INSECURE` ORs into the same global flag rather than applying per node.

- [ ] **Step 2: Update `AGENTS.md`**

Update the repo layout for the new `src/server/inference/*` files, the "What's shipped" section for multi-node support and the `/v1/models` merge, the admin API list for `?node=` parameters, and the Entity IDs section is unaffected (nodes have slugs, not ULIDs). Record in the design-constraints section that `endpoint` means request path and `node` means inference instance.

- [ ] **Step 3: Update `README.md`**

Feature list and routes list, plus a short "monitoring two nodes" configuration example.

- [ ] **Step 4: Amend the backends doc**

`docs/2026_05_03_inference_backends.md:132-150` argues for a singleton over a registry. Add a short note that the registry landed for multi-node, linking to the multi-node design doc, so the two documents do not contradict each other.

- [ ] **Step 5: Leave compose files single-node by default**

Add the `INFERENCE_2_*` block commented out, with a note that a second node is typically another host and so is not part of the bundled stack.

- [ ] **Step 6: Verify clean and commit**

```bash
pnpm lint:fix && pnpm format:fix && pnpm typecheck
git add AGENTS.md README.md .env.example docker-compose.amd.yaml docker-compose.nvidia.yaml docs/
git commit -m "docs: document multi-node configuration"
```

---

## Final verification

- [ ] `pnpm test` — full suite green
- [ ] `pnpm lint:fix && pnpm format:fix && pnpm typecheck` — all clean
- [ ] With only `INFERENCE_BASE_URL` set, the app behaves exactly as before: one node, one health row, no node badges beyond the single node, `/v1/models` returns that node's list
- [ ] With `INFERENCE_2_*` set, both nodes appear on `/system`, models from both appear on `/models`, and a request naming a node-2-only model logs `node_id` = node 2
- [ ] Stop node 2: `/models` and `/system` still render, node 2 shows unreachable, node 1 traffic is unaffected
- [ ] Restart llama-dash with a deliberately malformed `INFERENCE_2_BASE_URL`: boot fails with a clear error naming the variable
