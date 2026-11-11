# Design

Status: **draft, planning stage.** Nothing described here is implemented yet. This
document records the decisions taken so far and the questions still open. When the
code disagrees with it, one of the two is wrong and gets fixed.

## 1. What the suite is for

The suite answers two kinds of question about [BNGsim](https://github.com/lanl/bngsim):

- **Correctness:** does BNGsim give the same answer as an independent reference? The
  references are other simulators, analytic results, and standard test suites.
- **Cost:** how long does BNGsim take, relative to the tools a modeler would otherwise
  use, on the same problem?

Every comparison in the suite supports one of these claims:

1. BNGsim reproduces the results of the tool it replaces (BioNetGen) and of the
   standard references (libRoadRunner, AMICI).
2. It costs no more than those tools, and less where it matters. The main example is
   re-simulating a loaded model with new parameters, as in a fitting loop.
3. Its internal choices are sound:
   - Automatic linear-solver selection costs about the same as the better of dense
     and KLU.
   - Compiled and interpreted right-hand sides agree.
4. It conforms to the SBML standard.
5. Its approximate method (partial scaling, PSA) is accurate where it is used.

The suite is built so that anyone can run it:

- one job at a time on a laptop;
- many jobs on a single large node;
- unattended on a Slurm cluster.

## 2. Capabilities and comparisons

| Capability | Correctness reference | Cost comparison |
|---|---|---|
| ODE simulation, BNGL models | BioNetGen `run_network` | `run_network` |
| ODE simulation, SBML models | libRoadRunner | libRoadRunner; AMICI; COPASI (core only) |
| ODE settings: dense, KLU, automatic; compiled vs interpreted | each other | forced dense and forced KLU vs automatic |
| Re-simulation of a loaded model with new parameters | — | `run_network`, libRoadRunner, AMICI |
| Exact stochastic simulation (SSA) | `run_network` SSA (ensemble statistics); DSMTS analytic moments | `run_network` SSA; libRoadRunner; COPASI (core only) |
| Partial scaling (PSA) | exact SSA, as the critical population N_c varies; `run_network` PSA | `run_network` PSA; exact SSA |
| Network-free simulation | standalone NFsim; RuleMonkey; SSA on the generated network where it is finite | standalone NFsim |
| Forward sensitivities | AMICI; finite differences | AMICI; COPASI (core only) |
| Steady state | long-time integration; AMICI steady state | integration to steady state; AMICI |
| SBML compliance | SBML Test Suite semantic cases; DSMTS | — |
| Parameter fitting (application) | objective values at fixed parameter points | PyBNF with its legacy BioNetGen backend |

COPASI appears in the core tables and in the BioModels coverage count. It is not part
of breadth timing.

## 3. Layers

The suite has five layers. Each problem belongs to one layer, and the layer sets the
rule for what may go in it.

### 3.1 Standards

These are used whole, as published by their maintainers:

- the SBML Test Suite (semantic cases);
- DSMTS (discrete stochastic model test suite);
- AMICI's own forward-sensitivity test cases.

They are conformance sets, not representative models.

### 3.2 Core

About 15–20 published models, each tagged with the capabilities it exercises. Each
model is reused across capabilities wherever possible, so results for ODE,
sensitivity and steady state on the same model can be read together.

A core model must be **published and verified** (section 4.1).

The core as a whole should span:

- **Size:** from about 10 species to thousands. Include models on both sides of the
  automatic dense/KLU threshold, and one near it.
- **Dynamics:**
  - relaxation to steady state;
  - an oscillator;
  - a bistable switch.
- **Stiffness:** at least one strongly stiff model.
- **Format:**
  - BNGL-native and SBML-native models;
  - one or two models available in both formats, for a cross-format check.
- **Stochastic behaviour:**
  - low copy numbers;
  - one model with known analytic moments;
  - for PSA, species whose populations differ by orders of magnitude.
- **Network-free behaviour:**
  - aggregation models with no finite network;
  - at least one model whose network is finite, so network-free results can be
    checked against SSA on that network.
- **Sensitivities:** 10–200 parameters, preferably parameter-estimation problems with
  data.
- **Steady state:** monostable and bistable models, and one with conservation laws,
  whose Jacobian is singular.

### 3.3 Size series

These are variants of verified core models, used **only** for scaling claims:

- ODE cost against network size, for each linear-solver setting;
- SSA cost per reaction event against reaction count;
- the point at which network-free simulation overtakes simulation of the generated
  network.

Rules:

1. The knob is either one the publication itself varies, or a truncation of an
   infinite network (maximum chain length, maximum aggregate size). Nothing else in
   the model changes.
2. The published setting is one point of the series, and that point is the verified
   file.
3. Variants are produced by a committed generator from the verified file. They are
   never edited by hand. The knob value is part of the problem ID, e.g.
   `dushek2011@sites=8`.
4. Variants are labelled as variants wherever results are reported. They never feed
   claims about realistic problems.
5. Correctness checks run on every variant too.

Candidate families, all curated records in
[BNGL-Models](https://github.com/wshlavacek/BNGL-Models) with verification notebooks:

| Family | Knob | Notes |
|---|---|---|
| Dushek et al. 2011, membrane-anchored multisite phosphorylation | number of identical sites | sparse Jacobian; 20 sites give 143 species, 284 reactions |
| Rubenstein et al. 2007, nucleated prion polymerization | maximum chain length (truncation) | much denser coupling; reactions grow roughly as the square of the length |
| Yang et al. 2008, trivalent-ligand bivalent-receptor aggregation | aggregate-size cap for network generation | network-free cost depends on particle count, not network size |

Scaling claims that need no variant model vary a run setting instead:

- N_c for PSA;
- the number of parameters for sensitivities;
- the number of dose points for steady state;
- the number of replicates for stochastic methods.

### 3.4 Breadth

Whole public corpora, used for distributions:

- correctness pass rates;
- the spread of cost ratios.

Cost headlines use all public models in the breadth tier. The corpora are:

- **SBML:** the BioModels corpus that bngsim's `rr_parity` suite uses: 1323 models,
  1011 curated (`BIOMD…`) and 312 non-curated (`MODEL…`).
- **BNGL:** the corpus that bngsim's `bng_parity` suite vendors, 895 models from
  BNGL-Models, RuleHub and RuleMonkey.

The admission rule for breadth is section 4.3.

### 3.5 Application

Parameter fitting with [PyBNF](https://github.com/lanl/PyBNF), using published
fitting jobs from BNGL-Models `pybnf-jobs/`. Each job has a recorded validation. Each
job yields three results:

1. **Correctness:**
   - Evaluate the objective with both backends at a fixed set of parameter points:
     the published best fit plus about 20 sampled points.
   - The values must agree within tolerance.
   - This is deterministic, so optimizer randomness doesn't enter.
2. **Paired cost per objective evaluation:** measured on the same points, under the
   timing protocol of section 7.
3. **End-to-end fit:**
   - The job's committed configuration, at its fixed iteration budget, run once per
     backend.
   - Each fit is one job.
   - Both backends' fits of a job run on the same node type.

## 4. Provenance rules

### 4.1 Published and verified (core layer)

A model is **published** if it is described in a peer-reviewed publication. The file
may be:

- the authors' own release, or
- a curated re-implementation, if it is verified.

A file is **verified** if all three hold:

- the exact file reproduces something the publication states;
- the check does not use BNGsim;
- the comparison is committed and can be re-run.

Each core problem records the kind of evidence it has:

| Kind | Meaning |
|---|---|
| `author_file` | the authors' own released model: curated BioModels, the PEtab benchmark collection, or the authors' repository at a pinned commit |
| `equations` | agrees with an independent implementation of the paper's equations |
| `data` | agrees with published numbers (digitized figure or reported values) within a stated tolerance |
| `theory` | agrees with an analytic or limiting-case result |
| `structure` | the network's size or composition matches the paper. Accepted only with a written reason, such as "the paper reports no dynamics". |

A variant file counts as verified only if its verification covers that exact file.

### 4.2 Pinning

Every model file is identified by repository, commit, path and sha256. The suite
refers to files; it does not need to redistribute them.

### 4.3 Breadth admission and labels

A model may enter the breadth tier if it is publicly available at a pinned commit.
Each model carries a provenance class, for breakdowns:

- `peer_reviewed`: RuleHub `Published/`, curated BNGL-Models records, curated BioModels;
- `example`: RuleHub `Examples/` and `Tutorials/`;
- `test`: engine test cases, such as RuleMonkey `tests/`;
- `working`: unpublished working models;
- `noncurated_biomodels`: BioModels `MODEL…` entries.

Byte-identical files (the same sha256) are counted once.

## 5. Problems and jobs

### 5.1 Problem specification

A problem is a model plus a fixed protocol plus stored reference results. One small
YAML file per problem holds the spec, for example:

```yaml
id: dushek2011
layer: core
source: {repo: wshlavacek/BNGL-Models, commit: <sha>, path: models/<record>/<file>.bngl, sha256: <sha256>}
publication: {citation: "<authors, year, journal>", doi: "<doi>"}
verification: {kind: equations, evidence: models/<record>/<notebook>, metric: max_rel_err, value: <value>}
capabilities: [ode, ode_settings, sensitivity]
protocol:
  ode: {t_end: <t>, n_points: <n>, rtol: <rtol>, atol: <atol>}
  sensitivity: {parameters: <list or "all">}
references: {ode/run_network: <path to stored reference output>}
```

Stored references let anyone check BNGsim without installing BioNetGen, AMICI or
COPASI. Re-generating the references is a separate job type.

### 5.2 Jobs

A job is the unit of work:

> one problem × one capability × one engine (or engine pair, for paired timing) × one setting

Job IDs are deterministic and readable, for example
`ode/dushek2011@sites=8/bngsim-vs-run_network/auto`. Generating the job list from the
problem specs and the suite configuration is itself a command, so the list never
drifts from the specs.

Correctness jobs and timing jobs are separate:

- **Correctness jobs** can run anywhere, at any parallelism.
- **Timing jobs** follow the protocol of section 7.

A timing job runs only on problems whose correctness job passed.

## 6. Results

### 6.1 One file per job

Each job writes one JSON file, atomically, under a path derived from its job ID. The
file holds:

- the outcome, using one outcome vocabulary across all capabilities: PASS, DIFF,
  EXCEPTION, TIMEOUT, UNSUPPORTED, SKIP;
- the measured values, and every timing repetition rather than only the summary;
- provenance:
  - the BNGsim version and compiled build commit;
  - the reference engines' versions;
  - this repository's commit;
  - the environment lock or container digest;
  - the node name and hardware (section 7);
  - start and end time;
  - exit status.

### 6.2 Freshness

A result is current when its key matches what would run now. The key combines:

- the job spec;
- this repository's version;
- the BNGsim build;
- the reference engines' versions;
- for timing jobs, the node type.

Only stale or missing jobs run. An interrupted run loses at most the jobs in flight.

### 6.3 Gathering and publishing

- `gather` assembles per-job results into one report per capability and layer.
- A **result bundle** packages one complete set of results for one BNGsim build:
  - per-job results;
  - gathered reports;
  - the environment lock;
  - hardware records.

  Bundles are published as GitHub release assets and linked from
  [lanl/bngsim](https://github.com/lanl/bngsim). Each bundle comes with a rendered
  summary page.
- Raw outputs are never committed to git history.

## 7. Timing protocol

1. **Paired and interleaved:**
   - A timing job runs BNGsim and its comparator in the same process tree, pinned to
     one core.
   - Repetitions alternate between the two (A B A B …).
   - Warm-up repetitions are discarded, and every remaining repetition is kept.
   - The job reports the per-problem ratio, with a bootstrap confidence interval over
     pairs.
2. **Single-threaded:** `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and the equivalents
   are set to 1, unless a job measures parallelism on purpose.
3. **Cold vs warm:**
   - Cold cost (loading, code generation, compilation) is a separate phase, measured
     with an empty compile cache.
   - Compile caches live on node-local scratch, never on a shared filesystem.
4. **One node type per artifact:**
   - Timing results that are compared with each other come from one declared node
     type (Slurm `--constraint`, or a recorded node type for interactive runs).
   - Each result records the node name, CPU model, microcode, frequency governor and
     turbo state, kernel, and how many other jobs shared the node.
5. **Absolute times** are always reported with their node type. They are more
   sensitive than ratios to everything above.
6. **Node sharing is settled by measurement, not assumed** (section 9).

Run-to-run drift on a single machine can be as large as the effects being measured.
That is why ratios are formed inside one job, never across separate runs.

## 8. Running

One job list, three ways to run it, all writing the same result files:

| Mode | Use | Notes |
|---|---|---|
| serial | a laptop, one job at a time | `bench run <job-id>`, or `bench run --next N` |
| local pool | a checked-out compute node | `bench run --workers 64`; correctness jobs fill the node, timing jobs follow section 9 |
| Slurm | unattended runs | `bench submit` writes `sbatch` job arrays; many short jobs are packed into each array task; partition, account and time limit come from a config file |

Other commands:

- `bench list`: problems and jobs;
- `bench plan`: what is stale, and what it would cost;
- `bench status`: progress and failures;
- `bench gather`;
- `bench bundle`.

The command names are provisional.

An allocation has a maximum wall time, possibly around 10 hours on the cluster in
view (unconfirmed). So:

- **Bounded runs:** a run on a checked-out node takes a deadline, `--until` or
  `--budget`. It does not start a job whose estimated cost would overrun the deadline,
  and it stops cleanly when the deadline arrives.
- **No lost work:** completed jobs are already written, so the next allocation picks
  up the rest.
- **Array tasks fit the limit:** jobs are packed into Slurm array tasks by estimated
  cost, so no task approaches the time limit.
- **Long jobs:** a single job that cannot finish within one allocation is split, or
  else flagged before it is submitted.

The environment is pinned:

- a `pixi` lock for Python and the engines;
- the BioNetGen distribution, pinned by release tarball and sha256;
- a separate pinned interpreter for PyBNF;
- a container image built from the same lock, for clusters where installing is
  impractical.

Corpora are fetched at pinned versions and checked by sha256.

## 9. Validation before trusting cluster timings

Two checks run before any cluster timing is used:

1. **Laptop vs cluster:**
   - One corpus is timed on a laptop and on the cluster, both with the paired protocol.
   - The candidate corpus is BNGL models exported to SBML, BNGsim vs libRoadRunner.
   - It passes if the geometric mean of the per-problem ratio-of-ratios lies within a
     stated tolerance (for example 5%) and the per-problem ratios correlate tightly.
2. **Node sharing:**
   - The same timing jobs are run one per node, then packed onto a full node.
   - If the ratios agree, timing jobs may share nodes.
   - If they don't, each node runs fewer timing jobs at once, down to one.
   - Nodes are not shared with other users, so this is only about the suite's own
     concurrent jobs.

## 10. Relationship to lanl/bngsim

- bngsim keeps its own correctness suites (`parity_checks/`) and its nightly parity
  workflow. They guard bngsim's development; this repository measures releases.
- This repository should use the same comparison protocol as bngsim's
  `parity_checks/_core` (tolerances, ensemble verdicts, outcome vocabulary), so a
  verdict means the same thing in both places. How to share that code is open
  (section 12).
- Benchmark harnesses migrate here from bngsim's `benchmarks/suites/` and from
  harnesses maintained elsewhere. They migrate one capability at a time.

## 11. Migration order

1. **Infrastructure:**
   - problem specs;
   - job generation;
   - the serial and local-pool runners, then Slurm;
   - per-job results and `gather`;
   - the environment and container;
   - the validation of section 9.
2. **Breadth correctness:** parallel, with no timing constraints. It reuses bngsim's
   parity comparisons.
3. **Core selection and verification audit:** choose the core models, then confirm
   that each one's evidence covers the exact file used.
4. **Capabilities, in order:**
   1. ODE, with the size series;
   2. SSA and PSA;
   3. network-free;
   4. sensitivities;
   5. steady state;
   6. SBML compliance;
   7. fitting.
5. **Breadth timing.**
6. **First result bundle.**

## 12. Open questions

- **Cluster specifics.** Known so far:
  - Slurm;
  - an interactively checked-out compute node is the user's alone;
  - nodes have at least 64 CPUs;
  - GitHub is reachable;
  - allocations have a wall-time limit, possibly about 10 hours.

  Still to establish:
  - node CPU model and whether nodes are homogeneous;
  - glibc version (libRoadRunner's Linux wheel needs glibc 2.28 or newer);
  - node-local scratch size;
  - partitions, account, QOS and the exact time limits;
  - container support;
  - package-index reachability from compute nodes.

  [`tools/survey_cluster.sh`](../tools/survey_cluster.sh) collects all of these.
- **Sharing the comparison protocol with bngsim:** vendor a pinned copy of
  `parity_checks/_core`, depend on a bngsim checkout at a pinned commit, or have
  bngsim ship it as part of the package.
- **Where bundles live:** as release assets of this repository linked from
  lanl/bngsim, or as release assets of lanl/bngsim itself.
- **Core model selection:** which 15–20 models, and the knob ranges of each size
  series.
- **Tolerances:** per-capability correctness tolerances, and the acceptance tolerance
  for section 9.
