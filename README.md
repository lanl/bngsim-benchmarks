# bngsim-benchmarks

Benchmarks and cross-engine correctness checks for
[BNGsim](https://github.com/lanl/bngsim), the embeddable simulation library for
BioNetGen rule-based models.

This repository is new and still being set up. It will hold the harnesses that
measure BNGsim's cost and check its results against other simulators on
published models, so that anyone can re-run them. The comparisons are against
BioNetGen (`run_network`, NFsim), libRoadRunner, AMICI, COPASI and PyBNF.

## Goals

- **Runnable by anyone**, from a pinned, documented software environment.
- **One job at a time.** A job is one model × engine × method, so a single
  comparison can be run or re-run on its own.
- **In parallel on a cluster**, as well as serially on a laptop.
- **Published results**, linked from [lanl/bngsim](https://github.com/lanl/bngsim).

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
