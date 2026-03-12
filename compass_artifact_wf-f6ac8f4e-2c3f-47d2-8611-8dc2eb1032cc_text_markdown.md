# Memory architecture for a parallel AI agent swarm

**The optimal memory architecture for Blitz-Swarm is a Redis-backed blackboard system with layered storage primitives, append-only event logging, and composite utility-based retention—buildable by a solo developer using Redis, SQLite, and an embedded vector store.** This design draws on the blackboard architecture pattern (validated by two 2025 papers showing 13–57% task improvement), CRDTs for conflict-free concurrent writes, and machine-native retention algorithms. The critical insight from the research: LLM API calls take 500–5,000ms, so storage latency differences between Redis (~0.1ms) and Postgres (~5ms) are noise. Optimize for **data model correctness and conflict semantics**, not raw speed.

## The blackboard pattern is the right foundation

The blackboard architecture, originating from Carnegie-Mellon's Hearsay-II speech system (1971–1976), maps precisely to Blitz-Swarm's requirements. Three components define it: a **blackboard** (shared memory surface), **knowledge sources** (specialized agents), and a **control component** (scheduler deciding who acts). Agents don't communicate directly—they read from and write to the shared blackboard, enabling loose coupling and true parallel execution.

Two landmark 2025 papers validate this pattern for modern LLM swarms. Google Research's blackboard MAS (Salemi et al., arXiv 2510.01285) achieved **13–57% relative improvement** in end-to-end task success by having a central agent post information requests to a shared blackboard while subordinate agents volunteer responses based on capability. LbMAS (Han & Zhang, arXiv 2507.01701) demonstrated that blackboard-based coordination consumes **fewer tokens** than alternative multi-agent approaches while remaining competitive with state-of-the-art. The blackboard's key advantage over orchestrator patterns is resilience—no single point of failure, agents contribute independently, and the system naturally supports the "middle-out" problem-solving that swarms excel at.

For Blitz-Swarm specifically, the blackboard becomes the single source of truth. Each agent reads the current blackboard state, performs its specialized work, and writes its contribution back. A controller runs rounds of parallel agent execution until convergence. The practical implementation uses **Redis hashes** as the blackboard surface with **Redis Pub/Sub** for change notifications—giving sub-millisecond read/write with built-in event broadcasting.

## Three storage layers handle different memory needs

No single storage primitive handles all multi-agent memory requirements. The research consistently points to a **layered hybrid architecture** where each layer serves a distinct purpose with appropriate latency, consistency, and query characteristics.

**Layer 1: Working memory via Redis.** This is the hot path—agent scratchpads (Redis Hashes with TTL), inter-agent notifications (Pub/Sub), task queues and event logs (Redis Streams with consumer groups), and distributed locks (Redlock). Redis delivers **~0.1ms local latency** with rich data structures: sorted sets for priority queues, streams for append-only event logs with automatic ID generation, and WATCH/MULTI for optimistic concurrency. Redis Streams are particularly valuable—they function as a lightweight event store with consumer groups, allowing multiple agents to process events reliably without message loss. The entire working memory layer runs on a single Redis instance, which handles **59,000+ messages/second** for pub/sub, more than enough for 5–20 agents.

**Layer 2: Persistent structured memory via SQLite + WAL.** SQLite in WAL (Write-Ahead Logging) mode enables concurrent reads alongside writes with zero operational overhead—no server process, no configuration. Enable it with three pragmas: `journal_mode=WAL`, `busy_timeout=5000`, `synchronous=NORMAL`. The constraint is **single-writer**—solve this with a dedicated async writer task and a queue. All agents submit writes to an `asyncio.Queue`; one writer batches and commits them. FTS5 provides full-text search; JSON1 handles structured queries. SQLite stores durable task results, agent decisions, SOPs, and the dependency graph.

**Layer 3: Semantic retrieval via LanceDB (or ChromaDB).** Semantic memory enables agents to find relevant prior observations by embedding similarity rather than exact key lookup. **LanceDB** is the recommended choice for a solo developer: zero-config (`pip install lancedb`), embedded (no server), Rust-backed columnar format, and CrewAI's current default backend. ChromaDB is the alternative with the largest community and recently gained a 4x performance boost from its 2025 Rust rewrite. Both handle the **5–50ms query latency** range at sub-million vector scale. Use this layer for agent knowledge bases, episodic memory, and observation retrieval—but never as the coordination layer.

The minimal viable stack is therefore **Redis + SQLite + LanceDB**—three dependencies, two of which require zero infrastructure.

## Conflicts dissolve with the right data model

The dominant conflict resolution strategy for agent swarms is not a single mechanism but a **tiered approach** matched to data type. The research reveals that most conflicts can be structurally eliminated rather than resolved.

**Append-only patterns eliminate conflicts entirely.** When agents write observations, findings, or events, use Redis Streams—an append-only log where concurrent writes never conflict. Each agent appends its contribution with an auto-generated ID and metadata (agent ID, timestamp, confidence score). No locking, no retries, no conflicts. This should be the default pattern for **~70% of agent writes**.

**Optimistic concurrency handles shared mutable state.** For the remaining cases where agents must update shared state (e.g., the current plan, a shared task list), Redis WATCH provides optimistic locking: an agent reads a key, processes it, then commits only if no other agent modified it. On conflict, the agent re-reads and retries with exponential backoff. This works well when contention is low—which it typically is, since agents usually write to different keys. TiDB's production experience warns that under high contention, OCC abort rates spike; but with 5–20 agents working on different subtasks, contention stays manageable.

**CRDTs provide conflict-free concurrent access for specific data structures.** Three CRDT types map directly to agent memory needs: **LWW-Register** (Last-Writer-Wins, timestamp-based) for agent status and simple state—the latest write silently wins; **OR-Set** (Observed-Remove Set) for shared collections like discovered entities or active tasks—concurrent add and remove operations merge deterministically; and **G-Counter** (Grow-only counter) for accumulated metrics like observation counts or confidence votes. These are simple enough to implement in application code—an LWW-Register is ~20 lines of Python, a G-Counter ~15 lines. For complex cases, the **Automerge** library (Rust core with Python bindings) provides a JSON-like API for CRDT-backed documents. Redis Enterprise includes native CRDT support, but the open-source route of implementing simple CRDTs in application code is sufficient for a solo developer.

**Semantic conflict resolution via arbiter agent** handles the rare case where two agents produce conflicting plans or strategies. Use an MV-Register (Multi-Value Register) CRDT that preserves all concurrent writes, then dispatch a dedicated arbiter agent to review the conflicts and choose or synthesize the best resolution. This is the most expensive path (requires an LLM call) but appropriate for high-stakes decisions.

## Machine-native retention replaces human memory analogies

Deciding what to store versus discard should use **composite utility scoring**, not human cognitive metaphors like "short-term" and "long-term" memory. The recommended approach combines four machine-native signals into a single eviction priority score.

**Access-frequency with time decay** tracks how often each memory entry is read, weighted by recency. The **ARC (Adaptive Replacement Cache)** algorithm is the strongest choice because it **self-tunes** between recency and frequency without parameters. ARC maintains four lists—recently-accessed (T1), frequently-accessed (T2), and two ghost lists (B1, B2) that track eviction metadata. Hits in the ghost lists dynamically adjust the balance. ARC outperforms LRU by **5–20%** across workloads. Redis implements approximated LFU using a clever 8-bit Morris counter (logarithmic probabilistic counting) that represents counts up to ~1 million in a single byte—a model worth emulating.

**Reward-based retention** back-propagates task success to contributing memory entries. When an agent's output improves (measured by downstream quality scores), the memory entries it retrieved before producing that output get a reward boost: `V(m) ← V(m) + α(R − V(m))`. The AgeMem paper (arXiv 2601.01885, 2026) demonstrated that RL-trained retention policies achieve **+13.9% to +21.7% improvement** over baselines by learning when to retain versus discard.

**Dependency graph tracking** models memory entries as nodes in a DAG where edges represent "depends on" relationships. Before evicting an entry, compute its transitive closure to assess downstream impact. Use **reference counting** as the primary GC mechanism (immediate, O(1) per operation) with **periodic mark-and-sweep** (O(V+E)) to catch cycles—exactly the hybrid pattern CPython uses. The `networkx` library provides the graph infrastructure; `nx.descendants()` computes transitive dependencies efficiently.

**Delta compression** stores only the diff between successive memory states. For iterative agent refinement where sequential states differ by <20%, delta compression achieves **5:1 to 20:1 compression** over full snapshots. The `DeepDiff` library (v8.6.1) provides a `Delta` class that can diff any Python object and reconstruct states: `state_v1 + delta == state_v2`. Store a chain of deltas with periodic full snapshots every N operations to bound reconstruction time.

The composite eviction score combines these signals: `priority = 0.3 × access_frequency + 0.3 × reward_score + 0.2 × dependency_importance + 0.2 × info_novelty`. Entries with the lowest composite score get evicted first. During compaction (LSM-tree style), apply utility scores to determine which versions to keep—making retention a natural part of the storage lifecycle rather than a separate process.

## What the cutting-edge research reveals

The most significant recent contribution is **G-Memory** (Zhang et al., arXiv 2506.07398, NeurIPS 2025 Spotlight)—the first hierarchical memory system designed specifically for multi-agent systems rather than adapted from single-agent architectures. It organizes memory into three graph tiers: an **Insight Graph** (high-level generalizable knowledge), a **Query Graph** (task-level associations), and an **Interaction Graph** (fine-grained agent collaboration traces). Bi-directional traversal retrieves both high-level insights and granular history. G-Memory achieved up to **20.89% improvement** in embodied action success rates and plugs into existing frameworks (ChatDev, MetaGPT, AutoGen) without modification. Its open-source implementation is at github.com/bingreeky/GMemory.

**A-MEM** (Xu et al., arXiv 2502.12110, NeurIPS 2025 Poster) takes a Zettelkasten-inspired approach where memory notes contain keywords, tags, and contextual descriptions, with the LLM autonomously generating links between memories and retroactively evolving them as new experiences arrive. It doubled performance on multi-hop reasoning versus baselines. **SRMT** (Sagirova et al., arXiv 2501.13200) demonstrates that in MARL settings, pooling individual agent memories into a shared workspace via cross-attention enables **emergent coordination without explicit communication protocols**—validating the shared-memory approach for swarms.

The comprehensive survey by Wu et al. ("Memory in LLM-based Multi-agent Systems," 2025, TechRxiv) identifies the critical design tradeoff: **hybrid topology** (combining agent-local memory for role-specific knowledge with shared memory for coordination) consistently outperforms pure-local or pure-shared approaches. The MacNet paper (Qian et al., ICLR 2025) scales this to 1,000+ agents by propagating only final artifacts between agents rather than full conversation histories—preventing context explosion while maintaining continuity.

## The recommended Blitz-Swarm implementation stack

Here is the concrete, buildable architecture for a solo developer:

```
┌─────────────────────────────────────────────────────────┐
│                 BLITZ-SWARM ARCHITECTURE                 │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│  │ Agent A  │ │ Agent B  │ │ Agent C  │ │ Agent D  │      │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘      │
│       │            │            │            │              │
│  ─────┴────────────┴────────────┴────────────┴──────────  │
│       BLACKBOARD CONTROLLER (asyncio event loop)          │
│       - Round management, convergence check               │
│       - Agent selection per round                         │
│       - Utility scoring + eviction                        │
│  ────────────────────┬──────────────────────────────────  │
│                      │                                     │
│  ┌───────────────────┴───────────────────────────────┐   │
│  │          REDIS (single instance)                    │   │
│  │  • Hashes: blackboard state (sub-ms r/w)           │   │
│  │  • Pub/Sub: change notifications                    │   │
│  │  • Streams: append-only event log + consumer groups │   │
│  │  • WATCH/MULTI: optimistic concurrency              │   │
│  └───────────────────┬───────────────────────────────┘   │
│                      │                                     │
│  ┌──────────┐  ┌─────┴──────┐  ┌────────────────────┐   │
│  │ SQLite   │  │ LanceDB    │  │ In-Memory           │   │
│  │ +WAL     │  │ (embedded) │  │ Dependency Graph    │   │
│  │          │  │            │  │ (networkx DAG)      │   │
│  │ Durable  │  │ Semantic   │  │ Ref counting +      │   │
│  │ state,   │  │ retrieval, │  │ mark-and-sweep GC   │   │
│  │ task log │  │ embeddings │  │                      │   │
│  └──────────┘  └────────────┘  └────────────────────┘   │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

**Core dependencies** (total: 6 Python packages + 1 service):

- `redis` (async) — working memory, pub/sub, streams
- `aiosqlite` — persistent storage with WAL
- `lancedb` — embedded vector search
- `deepdiff` — delta compression between states
- `networkx` — dependency graph management
- `asyncio` (stdlib) — parallel agent execution

**Conflict resolution strategy by write type:**

| Write type | Pattern | Mechanism |
|---|---|---|
| Agent observations/findings | Append-only | Redis Streams (no conflicts) |
| Agent status updates | Last-Writer-Wins | LWW-Register (~20 lines Python) |
| Shared entity/task lists | Concurrent add/remove | OR-Set CRDT (or Redis Sets) |
| Shared plan/strategy | Optimistic lock | Redis WATCH/MULTI with retry |
| Critical decisions | Arbiter resolution | MV-Register + arbiter agent |

**Retention pipeline:** Every memory write passes through: (1) WAL append for durability → (2) Bloom filter update for fast existence checks → (3) Dependency graph edge creation → (4) Utility score initialization. Eviction triggers when memory exceeds a threshold, scoring all entries by the composite formula and evicting the lowest-scoring entries that pass the dependency safety check (no critical downstream dependents).

## What to build first and what to skip

**Start with this minimal viable architecture:** A Python asyncio event loop running N agent coroutines in parallel, each reading from and writing to a Redis-backed blackboard (Hash + Pub/Sub). Use `asyncio.gather()` to run all agents simultaneously per round. Store durable results in SQLite+WAL via a single writer queue. Add LanceDB semantic search when agents need to retrieve relevant prior observations. This gets you a working parallel swarm with shared memory in under 500 lines of code.

**Add incrementally:** Delta compression (when memory growth becomes a problem), dependency graphs (when you need provenance tracking), CRDTs (when conflict rates increase), and the composite utility scorer (when you need intelligent eviction). Consider integrating G-Memory's three-tier hierarchy if you need cross-task learning, or Mem0 as a drop-in memory layer if you want LLM-powered memory extraction without building it yourself.

**Skip entirely:** Kafka (overkill for <20 agents), Raft/Paxos consensus (unnecessary—use Redis locks for the rare cases needing strong coordination), Kubernetes orchestration (asyncio coroutines handle 5–50 agents on one machine), custom vector search (use LanceDB), and graph databases like Neo4j/Memgraph (add complexity without proportionate benefit at this scale). The entire system should run on a single machine with Redis as the only external service—`docker run redis` and you're operational.

## Conclusion

The Blitz-Swarm memory architecture rests on three validated principles: the blackboard pattern for agent coordination (confirmed by Google Research and LbMAS in 2025), append-only event logging as the default write pattern (structurally eliminating most conflicts), and composite utility scoring for machine-native retention decisions. The most surprising finding from this research is that **storage latency is irrelevant**—LLM API calls dominate at 500–5,000ms, making the difference between Redis and SQLite (~5ms) a rounding error. The real architectural decisions are about data model design, conflict semantics, and memory lifecycle management. G-Memory's hierarchical approach (NeurIPS 2025 Spotlight) represents the current frontier for multi-agent memory, offering a framework-agnostic design that a solo developer could adapt. The recommended stack—Redis, SQLite+WAL, LanceDB, and Python asyncio—provides production-grade capabilities with minimal operational overhead, requiring only a single `docker run redis` command beyond standard Python dependencies.