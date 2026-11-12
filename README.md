# bngsim-benchmarks

Benchmarks and cross-engine correctness checks for
[BNGsim](https://github.com/lanl/bngsim), the embeddable simulation library for
BioNetGen rule-based models.

This repository is new and still being set up. It will hold the harnesses that
measure BNGsim's cost and check its results against other simulators on
published models, so that anyone can re-run them. The comparisons are against
BioNetGen (`run_network`, NFsim), libRoadRunner, AMICI, COPASI and PyBNF.

What runs today: the job machinery, and one capability, ODE simulation of BNGL
models against BioNetGen's `run_network`, on five provisional pilot models in
[`problems/pilot/`](problems/pilot/).

## Goals

- **Runnable by anyone**, from a pinned, documented software environment.
- **One job at a time.** A job is one model × engine × method, so a single
  comparison can be run or re-run on its own.
- **In parallel on a cluster**, as well as serially on a laptop.
- **Published results**, linked from [lanl/bngsim](https://github.com/lanl/bngsim).

The design, and the decisions behind it, are in [docs/DESIGN.md](docs/DESIGN.md).

## Running it

Install [pixi](https://pixi.sh), then, from a checkout of this repository:

```bash
pixi install
```

```bash
pixi run fetch-bionetgen
```

```bash
pixi run doctor
```

`pixi install` builds the pinned environment, including the bngsim release under
test. `fetch-bionetgen` downloads BioNetGen 2.9.3 and checks its sha256. `doctor`
prints what every result will record about the environment, and says READY when
nothing is missing.

Then run any part of the suite. A job is one problem, capability, engine pair,
setting and kind (`correctness` or `timing`). Its id reads like
`ode/kinetic_proofreading_hlavacek2001/bngsim-vs-run_network/auto/correctness`,
and any part of an id selects jobs:

| command | does |
|---|---|
| `pixi run bench list` | every problem and job |
| `pixi run bench plan` | which jobs have a current result, and which would run |
| `pixi run bench run kinetic_proofreading` | the jobs whose id contains the text, one at a time |
| `pixi run bench run --workers 8` | every job without a current result, correctness jobs 8 at a time |
| `pixi run bench run --budget 9h30m` | stop starting jobs that would not finish within 9.5 hours |
| `pixi run bench run --until slurm` | the same, up to the end of the current Slurm allocation |
| `pixi run bench gather` | collect results into `reports/`, with a readable `reports/summary.md` |

Each job writes one file under `results/`, named after its id. A result stays
current until something it depends on changes: the model file, the protocol, the
harness, the bngsim build, BioNetGen, or, for a timing result, the node type.
Running again runs only what is missing or out of date, so an interrupted run
loses only the jobs in flight. Timing jobs run one at a time unless
`--timing-workers` says otherwise, and only after their correctness job passed.

`pixi run test` runs the unit tests in seconds. `pixi run test-integration`
runs one real pilot problem end to end.

## Running unattended on Slurm

`bench submit` takes the same job selection as `bench run` and submits the jobs
without a current result as one Slurm job array. Jobs are packed into array
tasks that fit the time limit, and each task runs `bench run` on its node until
the end of its allocation.

```bash
cp slurm.example.yaml slurm.yaml
```

Fill in `slurm.yaml` (time limit, CPUs per task, partition, account, ...), and
never commit it: it names local infrastructure, so it is git-ignored. Then, on a
login node:

| command | does |
|---|---|
| `pixi run bench submit --dry-run` | write the job array to `slurm/<time>/` without submitting it |
| `pixi run bench submit` | the same, then submit it with `sbatch` |

Before submitting, `bench submit` downloads and checks every model the array
needs, so compute nodes need no network once `pixi install` has run on the login
node. A job that does not finish within its task's allocation leaves no result;
`bench plan` shows what is left, and the next `bench submit` picks it up.

## Surveying a cluster

Before the first run on a new cluster, describe it with
[`tools/survey_cluster.sh`](tools/survey_cluster.sh). Run it once on a login node
and once on a compute node, inside an allocation:

```bash
bash tools/survey_cluster.sh
```

The script is read-only and takes about a minute. Each run writes a report to
`$HOME/bngsim-cluster-survey-<host>-<timestamp>.txt`, covering:

- CPUs, memory, scratch space and glibc;
- Slurm partitions, limits and accounts;
- containers and toolchains;
- network reachability.

The reports describe local infrastructure, so they stay outside the repository.
Don't commit them.

## License

MIT. See [LICENSE](LICENSE) for the full text and the Triad/LANL copyright notice.

© 2026. Triad National Security, LLC. All rights reserved. This is a Los Alamos National
Laboratory open-source release; LANL software release reference **O5098**.
