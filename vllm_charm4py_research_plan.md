# Running Ray-vLLM on Charm4Py/Reconverse: feasibility and research plan

Date: 2026-09-13

## Executive recommendation

Pursue this as a staged research project, but do **not** pitch “vLLM runs after replacing `import ray`” as the research contribution. The compatibility work is valuable infrastructure and a strong demonstrator, but by itself it is unlikely to clear the novelty bar of a selective systems conference.

The promising paper is a controlled study and a new mechanism around a **replaceable, message-driven control plane for distributed LLM serving**. Current vLLM makes this unusually clean: its default Ray V2 executor uses Ray to place, identify, monitor, and terminate workers, but uses vLLM’s own shared-memory/TCP message queues for steady-state control and NCCL or the platform collective library for tensor communication. This lets Ray and Charm4Py/Reconverse be compared while holding PagedAttention, model execution, batching, and most of the data plane fixed. PagedAttention itself is not being reimplemented; it remains inside vLLM.[^1]

My recommended sequence is:

1. Build a versioned Ray Core compatibility contract and make the existing Charm4Py layer correct.
2. Target only vLLM’s default `RayExecutorV2`, not the legacy compiled-DAG executor or Ray Serve.
3. Prove unchanged vLLM worker orchestration first with fake workers, then on two GPUs, then across nodes.
4. Continue toward a paper only if measurements reveal a real systems opportunity: scalable actor/placement control, lower control-plane interference, faster failure detection/recovery, or a novel SLO-aware scheduling mechanism.

A realistic estimate for one experienced developer is **8–12 engineering weeks for a robust RayExecutorV2 MVP**, assuming access to a Linux GPU machine by the middle of the project. A credible conference result is more likely **4–6 months**, because it needs a novel mechanism, multi-node experiments, baselines, ablations, and failure/workload studies.

## Scope and audited snapshots

This plan is based on these local snapshots:

- Charm4Py `ray-dev`: `14bc1459372e7118bd3459efc11ff57428194071`
- vLLM `main`: `7fe8fc803d904795fd10b785776e20181aee64c5`
- Charm++/Reconverse branch: `f6c74074f0c505d21d1c898836c1509d951dd44a`
- nested Reconverse branch: `92cb815b3bd3b5b1492e8d164a8b29d6c16809b3`

The vLLM snapshot is fast-moving main-branch code. Its CUDA test requirements currently pin Ray 2.56.x, so the compatibility target should be explicitly named something like “Ray Core compatibility profile for vLLM `7fe8fc8` / Ray 2.56,” rather than claiming general compatibility with every Ray release.

The minimum target is:

- vLLM V1 engine;
- `distributed_executor_backend="ray"`;
- default `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1`;
- one vLLM engine, first one node and then multiple nodes;
- tensor and pipeline parallelism using vLLM/PyTorch communication;
- no Ray Serve, Ray compiled DAG, Ray channels, NIXL/RDT weight transfer, autoscaling, or elastic data-parallel control in the MVP.

## What the existing Charm4Py implementation does

### Runtime initialization

[`ray.init()`](/Users/ritvik/charm4py/charm4py/ray/api.py:52) creates a Charm `Group(ObjectStore)`, giving each PE a local object-store branch, and broadcasts the proxy into the `charm4py.ray.api` module. It ignores the address, CPU/GPU resource counts, runtime environment, namespace, and most normal Ray initialization arguments. It also assumes Charm4Py has already been started with `charm.start(...)`; therefore this is not currently a drop-in replacement for a normal Python/Ray entry point.

### Stateless tasks and scheduling

[`@ray.remote` functions](/Users/ritvik/charm4py/charm4py/ray/api.py:125) are submitted through `charm.pool.map_async`. A singleton `PoolScheduler` on PE 0 maintains a job list and a set of idle workers; it sends one task at a time to a fixed Charm array of workers on PEs 1 through N-1. Marking the function as a Charm coroutine lets nested `ray.get()` calls yield through greenlets.

This gives basic task execution, but it is not Ray-like resource scheduling:

- all tasks go through a centralized PE-0 scheduler;
- CPU, GPU, fractional, custom, memory, and placement resources are ignored;
- keyword arguments to remote functions are lost;
- task retries, cancellation, priorities, locality, runtime environments, and scheduling strategies do not exist;
- the pool has correctness defects in multiple-return and exception paths;
- the worker placement is static and does not currently invoke Charm’s measurement-based load balancing.

### Actors

[`@ray.remote` classes](/Users/ritvik/charm4py/charm4py/ray/api.py:149) are dynamically converted into Charm chare subclasses. Actor construction chooses `counter % charm.numPes()`, and [`RayProxy`](/Users/ritvik/charm4py/charm4py/ray/api.py:86) wraps exposed methods with `.remote()`.

This preserves state and serial method execution in simple examples, but several Ray actor properties are missing. Ray normally gives each actor a dedicated Python worker process and schedules it according to declared resources.[^2] The current implementation instead creates a singleton chare inside a Charm runtime process, ignores `.options()`, has no `ActorHandle` module/type, accepts no constructor keyword arguments, has no actor naming or lookup, cannot kill or restart an actor, and does not propagate actor failures to references. Its round-robin placement also bypasses the adaptive scheduling capabilities that would motivate using Charm++.

### `put`, `get`, `wait`, and references

[`ray.put()`](/Users/ritvik/charm4py/charm4py/ray/api.py:183) returns a Charm `Future(store=True)`. Remote tasks and actor methods also return stored futures. A nominal `ObjectRef` subclass exists, but the implementation never returns it, and there is no `ray.types.ObjectRef`.

[`ray.get()`](/Users/ritvik/charm4py/charm4py/ray/api.py:164) handles one Charm future or a list of them. It does not implement timeout behavior, correct exception behavior, tuples, generators, or Ray’s pass-through behavior for ordinary values. [`ray.wait()`](/Users/ritvik/charm4py/charm4py/ray/api.py:173) supports neither timeouts nor `fetch_local=False`, and computes the not-ready result through sets, losing input order and duplicate-reference semantics.

Ray references are more than futures: they name objects cached in a distributed shared-memory store, participate in distributed reference counting, and have owner/failure semantics.[^3] Full Ray fault tolerance also includes replica lookup and lineage-based reconstruction for eligible task results.[^4] Not all of that is needed for the first vLLM milestone, but API type identity, timeouts, deterministic ordering, failure completion, and thread safety are needed.

### Object store

The store is one [`CObjectStore`](/Users/ritvik/charm4py/charm4py/c_object_store.pyx:86) per PE. Each stored future receives an ID composed from its creator PE and future ID. The ID’s “home” PE tracks locations. Objects under 1 KiB according to shallow `sys.getsizeof()` are placed at home; larger objects remain where produced, and remote gets copy and retain a local replica. When a stored future is serialized as an argument, the receiver can delay the Charm message until referenced objects arrive.

This is a useful prototype of dependency-aware active messages, but it is not production safe:

- size classification is shallow Python-object size, not serialized or device-buffer size;
- there are no byte quotas, spilling, backpressure, pinning, zero-copy shared memory, CUDA buffers, or reconstruction;
- reference lifetime depends on Python `__del__` messages and does not cover failed processes;
- [`bulk_send_location()`](/Users/ritvik/charm4py/charm4py/object_store.py:76) calls a nonexistent Cython method;
- [`check_loc_requests_buffer()`](/Users/ritvik/charm4py/charm4py/c_object_store.pyx:178) sends to a nonexistent `bulk_request_location` entry method;
- failure paths can leave requests or dependencies unresolved forever.

For vLLM V2, this store is primarily needed for small startup RPC results and the pending worker-liveness reference, not for every inference step. That sharply lowers the MVP performance requirement, though correctness remains mandatory.

### Resource, placement, topology, and lifecycle state

This area is mostly stubbed:

- [`cluster_resources()`](/Users/ritvik/charm4py/charm4py/ray/api.py:47) dereferences a state object whose implementation is commented out;
- [`available_resources()`](/Users/ritvik/charm4py/charm4py/ray/api.py:188) reports every PE as one CPU, zero GPUs, and constant fake memory;
- [`PlacementGroup`](/Users/ritvik/charm4py/charm4py/ray/util/placement_group.py:14) calls undefined `map_bundles()` and lacks `ready()`, `bundle_specs`, an ID, removal, and table state;
- `get_current_placement_group()` incorrectly constructs an empty group rather than returning `None` outside one;
- [`RuntimeContext`](/Users/ritvik/charm4py/charm4py/ray/runtime_context.py:6) equates a PE with a node and lacks actor ID and accelerator assignments;
- `get_node_ip_address()` returns an empty string;
- `shutdown()` exits the entire program rather than releasing compatibility-layer resources.

Ray placement groups are gang reservations of resource bundles with PACK/SPREAD variants; actors are subsequently bound to a reserved bundle.[^5] Ray’s resource quantities are logical admission-control resources, while accelerator assignment additionally controls visible device IDs.[^6] vLLM depends on both parts.

## How current Ray-vLLM actually uses Ray

The default executor choice is visible in [`Executor.get_class()`](/Users/ritvik/vllm/vllm/v1/executor/abstract.py:48), and V2 is enabled by default in [`envs.py`](/Users/ritvik/vllm/vllm/envs.py:926). vLLM’s own description of [`RayExecutorV2`](/Users/ritvik/vllm/vllm/v1/executor/ray_executor_v2.py:237) is the decisive fact: it reuses the multiprocessing executor’s message-queue control plane and NCCL data plane while using Ray actors as workers. Official vLLM documentation also describes Ray as its multi-node distributed runtime.[^7]

The startup path is:

1. Initialize or join the runtime.
2. Discover cluster and per-node accelerator resources.
3. Create or reuse a placement group containing one accelerator bundle per world rank.
4. Read placement-group tables and live node IDs/IPs to order bundles.
5. Create one `RayWorkerProc` actor per bundle with `name`, `num_cpus=0`, accelerator resources, a placement-group scheduling strategy, and `runtime_env`.
6. From each actor, obtain node ID and physical accelerator IDs.
7. Initialize workers using keyword arguments, including local rank, distributed init endpoint, environment variables, and the node-local physical-device map.
8. Fetch response-message-queue handles from all workers.
9. call each actor’s long-running `run.remote()` and retain the pending `ObjectRef` as a liveness sentinel.
10. Poll those references with `ray.wait(..., timeout=5)`; use `ray.kill()` for failure injection and shutdown.

After startup, vLLM broadcasts work through its `MessageQueue` and uses platform collectives for model tensors. The Ray object store is not on the normal token-generation path. This is why implementing Ray compiled DAG or a high-performance tensor object store first would be effort in the wrong place.

The immediate API surface for this path is:

| Contract | Exact vLLM use | Current status |
|---|---|---|
| Import surface | `ray`, `ray.actor`, `ray.types`, `ray.exceptions`, `ray.util`, `ray.util.placement_group`, `ray.util.scheduling_strategies`, `ray._private.state` | Mostly absent |
| Init/state | `is_initialized`, `init(address, runtime_env, ...)`, `cluster_resources` | Partial; resource query broken |
| Topology | `nodes`, per-node available resources, node IP | Absent/incorrect |
| Placement | create, `ready`, `bundle_specs`, table, remove, current PG | Stub is unusable |
| Actors | `ray.remote(Class).options(...).remote(**kwargs)` | Superficially similar; kwargs/options/process model fail |
| Context | node ID, actor ID, accelerator IDs, runtime env | Mostly absent |
| References | `ray.get`, timed `ray.wait`, `ObjectRef` type | Partial; timeout/type/error semantics missing |
| Lifecycle | pending run ref, actor failure, `kill` | Absent |
| Runtime env | actor-specific environment propagation | Absent |

Three architectural incompatibilities matter more than individual missing APIs:

1. **Bootstrapping:** vLLM is an ordinary Python/CLI program; Charm4Py currently requires the application body to be entered through `charm.start(main)`. A one-line import change is not sufficient.
2. **Process model:** Ray actors have dedicated processes. vLLM relies on process-scoped environment, GPU context, failure, and termination behavior. Multiple chare actors in one Python process are not equivalent.
3. **Long-running actor call:** `RayWorkerProc.run()` enters a blocking worker loop. Running it directly as a normal chare entry method can monopolize the PE and prevent the Charm scheduler from processing health or kill messages. A greenlet does not help if the loop never yields to Charm.

## Recommended implementation architecture

### 1. Define a compatibility profile, not “all of Ray”

Create a machine-readable capability list tied to a Ray/vLLM snapshot. Every supported call should have the same signature, return shape, ordering, timeout, and exception behavior as real Ray. Unsupported options should fail loudly rather than being silently discarded.

Maintain two test modes from the same test vectors:

- an oracle run against Ray 2.56.x;
- a compatibility run against Charm4Py/Reconverse.

Do not begin with vLLM. First prove the contracts in isolation, including actor construction with kwargs, chained options, placement bundle binding, runtime context, pending references, timeout zero, actor failure, and cleanup.

### 2. Add a controlled import/bootstrap facade

vLLM contains direct imports from multiple `ray.*` modules, so changing `import ray` in one file is insufficient. Add a launcher such as `charmray-run` that:

- starts Charm4Py under the selected Charm++ build;
- installs compatibility modules into `sys.modules` for the narrowly supported `ray.*` namespace before importing vLLM;
- invokes the target module or console entry point only on the driver PE;
- leaves all other PEs in the Charm runtime as worker hosts;
- converts process exit and signals into an orderly collective shutdown.

This avoids publishing a misleading top-level package named `ray` and permits unchanged third-party source within the launched application. The stronger compatibility claim should be “source-unmodified under a CharmRay launcher,” not “only one import changes,” unless the bootstrap API is redesigned more deeply.

### 3. Separate Ray-facing objects from Charm internals

Introduce explicit objects with stable semantics:

- `ObjectRef`, containing an immutable globally unique ID and completion/error metadata;
- `RemoteFunction` and `ActorClass`, each retaining base options and returning a cloned wrapper from `.options()`;
- `ActorHandle`, containing actor ID and a proxy to its execution host;
- `ActorMethod`, whose `.remote(*args, **kwargs)` always returns an `ObjectRef`;
- exception classes including `GetTimeoutError` and `RayActorError`;
- a runtime context stored per executing task/actor rather than derived only from `charm.myPe()`.

Keep Charm `Future` as an internal waiting primitive. Do not make every Charm future pretend to be a Ray reference.

### 4. Build a real resource and placement service

Create a replicated or PE-0-authoritative cluster registry initially, with one record per physical node:

- stable node ID and routable IP;
- PE/process IDs hosted on the node;
- total and available logical CPUs;
- accelerator type, physical IDs, and allocation state;
- custom resources and liveness.

Placement-group creation must atomically reserve all bundles, implement at least PACK and SPREAD, expose `ready()` as a reference, preserve bundle indices, and produce the exact table shape vLLM reads. Actor creation with `PlacementGroupSchedulingStrategy` must consume the requested bundle and install its node/GPU assignment into the actor runtime context. Start with whole accelerators; fractional GPUs can wait.

Charm’s PE topology must not be exposed as Ray node topology. Several PEs can reside in one physical node, and vLLM uses that distinction to compute local rank and device maps.

### 5. Resolve actor process isolation before integrating vLLM

Run a short design spike comparing these choices:

| Design | Benefits | Risks |
|---|---|---|
| One actor per non-SMP Charm PE/process | Smallest implementation; process-scoped GPU state is naturally isolated | Long-running methods block the PE; kill/restart may affect the Charm job |
| Actor code on a dedicated native thread in each PE | Charm scheduler remains responsive; low launch cost | Environment and CUDA state remain process-wide; CPython/Charm thread-safety and cancellation are difficult |
| Charm-managed child process per actor | Closest Ray semantics; clean environment, GPU, failure, and kill isolation | More machinery and IPC; Charm schedules hosts rather than executing actor code directly |
| vLLM-specific Charm executor | Fastest route to a demo | Weak Ray-compatibility result and more vLLM source coupling |

For a general compatibility layer, **Charm-managed child process per actor is the safest target**. For an early vLLM proof, one actor per non-SMP PE may work if the long-running `run()` call is executed on a dedicated lane and graceful shutdown is proven. Treat this as a go/no-go experiment, not an implementation detail to postpone.

GPU-bearing actor hosts should be pinned and non-migratable. CUDA contexts, PyTorch process groups, and vLLM message queues are not transparently migratable. Charm migration is more plausible for lightweight schedulers, routers, request metadata, or explicitly serializable KV state than for a live model worker.

### 6. Make the object store correct before making it sophisticated

For the V2 MVP:

- store small control results and exceptions reliably;
- support concurrent driver threads calling `get`/`wait`;
- implement timeouts without blocking the Charm scheduler;
- retain input ordering and duplicates;
- make actor death resolve outstanding references with `RayActorError`;
- use explicit reference counting/leases rather than relying only on `__del__`;
- remove the known undefined entry-method paths;
- add bounded memory accounting.

Zero-copy tensor storage, spilling, lineage reconstruction, and GPU-aware object transfer belong to a later general-Ray milestone. They are not prerequisites for V2 token generation.

## Execution plan and exit criteria

Estimates assume one developer, code review time included, and no major Charm4Py thread-safety surprises.

### Phase 0 — Freeze the target and build the oracle suite (1 week)

Deliverables:

- pin the vLLM and Ray commits/versions;
- extract every V2 Ray call and expected data shape;
- make a minimal standalone “fake vLLM worker” reproducing V2’s actor lifecycle;
- run contract tests against real Ray;
- document unsupported features as explicit errors.

Exit criterion: a reviewed compatibility matrix with executable tests, not just an API list.

### Phase 1 — Core reference and actor correctness (1–2 weeks)

Implement:

- explicit wrappers and types;
- args/kwargs and immutable `.options()` behavior;
- scalar/list `get`, timed `get`, timed/order-preserving `wait`;
- exception propagation;
- runtime initialization and non-destructive shutdown;
- supported `ray.*` import facade.

Exit criterion: basic tasks and actors match the Ray oracle on one and four Reconverse PEs, including nested references and failures.

### Phase 2 — Topology, resources, and gang placement (2–3 weeks)

Implement:

- physical-node discovery independent of PE ID;
- node IP and accelerator inventory;
- atomic resource reservation;
- placement-group lifecycle/table APIs;
- placement-group scheduling strategy;
- actor runtime context and runtime-env variables.

Exit criterion: a fake four-worker TP/PP topology receives deterministic node IDs, local ranks, and distinct accelerator assignments; over-allocation blocks and times out correctly.

### Phase 3 — Actor isolation and lifecycle (2–3 weeks)

Implement the selected actor-host model, pending run references, kill, graceful exit, crash detection, and cleanup. Test a long-running method that coexists with runtime messages and can be terminated without killing the driver.

Exit criterion: the lifecycle tests modeled on [`test_ray_v2_executor.py`](/Users/ritvik/vllm/tests/distributed/test_ray_v2_executor.py:249) pass in a framework-independent harness, including external kill, monitor notification, and no leaked resource reservations.

This is the most important risk gate. Stop and reconsider the architecture if correct actor isolation requires bypassing Charm’s control plane or invasive vLLM changes.

### Phase 4 — vLLM V2 bring-up (2–3 weeks)

Bring up in this order:

1. import all required modules under the launcher;
2. run vLLM’s Ray cluster initialization with synthetic accelerator resources;
3. instantiate a stripped/fake `RayWorkerProc` and complete all nine startup steps;
4. run `facebook/opt-125m`, TP=1, through the executor;
5. run TP=2 and then PP=2 on a Linux GPU node;
6. exercise repeated create/generate/shutdown cycles;
7. run an OpenAI-compatible server and a small correctness workload.

Exit criterion: source-unmodified vLLM under the compatibility launcher produces matching deterministic outputs and survives 100 startup/shutdown cycles without hangs or resource leaks.

### Phase 5 — Multi-node correctness and hardening (2–4 weeks)

Add multi-node environment propagation, routable IP validation, device/rank assignment, actor failure behavior, logging, and CI. Test TP and PP separately because TP across slow links can obscure runtime behavior.

Exit criterion: at least two nodes and four or more GPUs, with output parity, clean shutdown, controlled worker failure, and no use of real Ray.

### Phase 6 — Research mechanism and evaluation (6–12 additional weeks)

Only begin the paper branch after the MVP exposes a measurable bottleneck or capability. Implement one mechanism, not several weak ones. The best initial candidate is a Reconverse-backed, priority-aware and distributed actor/placement control plane for many concurrent vLLM replicas, with fast failure detection and re-placement.

## Test and evaluation matrix

### Local Reconverse development

Use the installed `reconverse-darwin-arm8` build and the Charm4Py virtual environment. Multi-process runs should use:

```text
LCRUN=/Users/ritvik/charm_reconverse/reconverse-darwin-arm8/_deps/lci-src/lcrun
```

The Mac is appropriate for contract tests, object-store tests, fake actors, placement logic, and Reconverse message microbenchmarks. Current vLLM supports experimental CPU execution on Apple Silicon, but its normal Ray executor requires a platform accelerator resource key; the official macOS CPU path is not a clean proxy for a distributed GPU experiment.[^8] vLLM-Metal is an optional later demonstration, not a prerequisite, because it introduces another plugin and a custom `mlx` resource.

### Required GPU evaluation

Use a Linux cluster for the actual claim. At minimum:

- one node: 1, 2, 4, and preferably 8 GPUs;
- two or more nodes: TP and PP configurations;
- real Ray 2.56.x as the primary baseline;
- vLLM multiprocessing on one node as a non-Ray baseline;
- Charm4Py over standard Charm++ versus Charm4Py over Reconverse to isolate the substrate.

Measure:

- contract pass rate and source changes required;
- placement-group readiness and actor startup p50/p95/p99;
- remote-call and small-object latency/throughput;
- scheduler/control CPU utilization under concurrent replicas;
- model cold-start and time-to-ready;
- request throughput, TTFT, TPOT, and SLO goodput;
- stable-load versus bursty-load tail latency;
- worker-failure detection and time to restore capacity;
- steady-state overhead when no failure or scaling event occurs.

Do not expect a large steady-state token-throughput win from merely swapping runtimes: vLLM’s MQ/NCCL data plane and PagedAttention remain unchanged. A strong result is more likely under replica churn, failure, bursty control traffic, or large actor counts.

## Publication assessment

### What is not enough

“We implemented missing Ray APIs and ran vLLM” is a useful artifact, demo, experience report, or workshop paper. It is not by itself a strong MLSys/EuroSys/OSDI contribution. Likewise, generic claims that Charm++ supports overdecomposition, migratable objects, and adaptive load balancing are established capabilities.[^9] Charm4Py has also already been evaluated as a scalable Python programming model.[^10]

Transparent migration of live vLLM GPU workers is not a credible first novelty claim. Their CUDA context, model memory, collective membership, and message queues are process/device bound. Explicit request/KV-cache migration is possible, but Llumnix already demonstrated live request-state migration and dynamic scheduling,[^11] while DistServe and later systems have established prefill/decode disaggregation and topology-aware resource planning.[^12] Heterogeneous adaptive deployment is also crowded; ThunderServe, for example, combines heterogeneous placement with lightweight rescheduling.[^13]

### Strongest paper angle

Working title: **ReconServe: A Replaceable Message-Driven Control Plane for Distributed LLM Serving**.

Core question:

> Can an active-message, adaptive HPC runtime replace a cloud task/actor runtime beneath an unmodified LLM serving engine, and does that replacement enable materially better orchestration at scale or under disruption?

Potential contributions:

1. A versioned semantic bridge from a useful Ray Core subset to Charm4Py, including actors, references, gang placement, topology, and failure contracts.
2. A design that separates the LLM data plane from a replaceable runtime control plane, demonstrated with source-unmodified vLLM V2.
3. One new runtime mechanism: for example, priority-aware control messages plus distributed gang placement and rapid failure re-placement for many concurrent replicas.
4. A controlled evaluation holding the model engine and communication data plane constant across Ray and Charm4Py/Reconverse.

Reconverse provides a plausible systems lever because it modernizes the Converse substrate around asynchronous, multithreaded communication; its reported LCI integration improves latency and message rate over the compared legacy layers.[^14] That result is motivation, not evidence that vLLM will improve. The paper must show where these substrate properties affect serving-level outcomes.

The cleanest hypotheses to test are:

- H1: distributed active-message control reduces placement/launch overhead and controller CPU at high actor/replica counts;
- H2: priority-aware scheduling prevents background object/task traffic from inflating inference control-tail latency;
- H3: runtime-integrated liveness and re-placement restores serving capacity faster after worker/node failures;
- H4: the compatibility layer adds negligible overhead during stable serving because vLLM retains its optimized MQ/NCCL data plane.

Reject H1–H3 honestly if the data does not support them. H4 plus compatibility alone is an engineering result, not a top-tier research result.

### Alternative angle if control-plane wins do not appear

Broaden from vLLM to **retargeting Python actor applications onto adaptive HPC runtimes**. Use vLLM as the demanding case, add two or three unmodified Ray Core applications, publish a semantic conformance suite, and compare task/actor/object-store behavior at scale. This fits a programming-model/runtime venue better than an ML-systems venue. The novelty must be generality and a principled mapping, not API mimicry.

### Venue fit and timing

- **MLSys:** best if there is a serving-level mechanism and convincing TTFT/TPOT/goodput improvement. The MLSys 2027 deadline is October 30, 2026, which is too soon for this project from its present state.[^15]
- **IPDPS, HPDC, or Cluster:** best if the contribution is runtime architecture, scalable actors/gang scheduling, or HPC/AI interoperability. IPDPS 2027 abstracts and papers are due October 1 and 8, 2026, also too soon for a responsible full paper.[^16]
- **EuroSys/OSDI/SOSP:** appropriate only with a broadly important mechanism, rigorous fault/scaling story, and substantial evaluation. EuroSys 2027’s remaining September 2026 cycle is not realistic.[^17]
- **Workshop/demo/artifact:** appropriate for the compatibility implementation and initial vLLM result while the research mechanism matures.

Do not optimize the engineering schedule around the imminent 2027 deadlines. Target a workshop or artifact first, then the next full-conference cycle with complete evidence.

## Go/no-go gates

Continue after each gate only if its condition is met:

1. **After 2 weeks:** oracle tests pass for references, actors with kwargs/options, timeouts, and fake placement. Otherwise narrow to a vLLM-specific executor.
2. **After 5 weeks:** a long-running fake worker can coexist with Charm scheduling, be killed, and release resources. Otherwise redesign actor isolation before doing any vLLM build work.
3. **After 8–12 weeks:** TP=2 vLLM generation works on Linux GPUs with output parity and stable repeated shutdown. Otherwise the work is not ready to support a research claim.
4. **After initial measurement:** at least one research hypothesis has a clear, reproducible advantage or exposes a new general tradeoff. Otherwise publish/release the compatibility artifact, but do not force it into an ML-systems paper.

The project is a good idea if the goal is to build a reusable experimental platform and then let measurements select the paper. It is a poor bet if the only acceptable outcome is a conference paper based on compatibility alone.

## Sources

[^1]: Woosuk Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention,” SOSP 2023. https://arxiv.org/abs/2309.06180
[^2]: Ray documentation, “Actors.” https://docs.ray.io/en/latest/ray-core/actors.html
[^3]: Ray documentation, “Objects.” https://docs.ray.io/en/latest/ray-core/objects.html
[^4]: Ray documentation, “Object Fault Tolerance.” https://docs.ray.io/en/latest/ray-core/fault_tolerance/objects.html
[^5]: Ray documentation, “Placement Groups.” https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html
[^6]: Ray documentation, “Resources” and “Accelerator Support.” https://docs.ray.io/en/latest/ray-core/scheduling/resources.html ; https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html
[^7]: vLLM documentation, “Multi-Node Serving” and executor API. https://docs.vllm.ai/en/stable/examples/ray_serving/multi-node-serving/ ; https://docs.vllm.ai/en/stable/api/vllm/v1/executor/index.html
[^8]: vLLM documentation, “CPU.” https://docs.vllm.ai/en/stable/getting_started/installation/cpu/index.html
[^9]: Charm++ documentation, “The Charm++ Parallel Programming System.” https://charm.readthedocs.io/en/rtd.yaml/charm%2B%2B/manual.html
[^10]: Zane Fink et al., “Performance Evaluation of Python Parallel Programming Models: Charm4Py and mpi4py,” ESPM2 2021. https://charm.cs.illinois.edu/papers/22-04
[^11]: Biao Sun et al., “Llumnix: Dynamic Scheduling for Large Language Model Serving,” OSDI 2024. https://www.usenix.org/conference/osdi24/presentation/sun-biao
[^12]: Yinmin Zhong et al., “DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving,” OSDI 2024. https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf
[^13]: Youhe Jiang et al., “ThunderServe: High-performance and Cost-efficient LLM Serving in Cloud Environments,” MLSys 2025. https://proceedings.mlsys.org/paper_files/paper/2025/hash/c2a0e26dd9ee7d57e92bb1c24b39659a-Abstract-Conference.html
[^14]: Jiakun Yan, “Rethinking communication libraries for asynchronous multithreaded runtimes,” University of Illinois thesis, 2025. https://www.ideals.illinois.edu/items/139528
[^15]: MLSys 2027, “Call for Research Papers.” https://mlsys.org/Conferences/2027/CallForResearchPapers
[^16]: IPDPS 2027, “Call for Papers.” https://ww.ipdps.org/ipdps2027/2027-call-for-papers.html
[^17]: EuroSys 2027, “Call for Papers.” https://2027.eurosys.org/cfp.html
