# Published results

`published/` holds our own results, one snapshot per suite and BNGsim build, written by
`bench publish SUITE` and committed. Anyone can re-run the same suite (`bench run
SUITE`, or `bench submit SUITE` on a Slurm cluster) and compare what they get with what
is here.

```
published/<suite>/bngsim-<version>-<build commit>/
    manifest.json        what was measured, on what, with which engines and harness
    correctness.csv      one row per (problem, reference engine)
    timing.csv           one row per (problem, engine)
    jobs/<job id>.json   every job's full result record
```

A snapshot is complete: every job of the suite has a current result, all from one
BNGsim build, and every timing from one node type. `bench publish` refuses otherwise.
Records are scrubbed of where they ran (host names, scheduler job ids, local paths;
paths are rewritten to `.` for the repository, `~` for home and `$TMPDIR` for a job's
scratch directory), and `bench publish` refuses if anything of that kind survives.

## manifest.json

| key | meaning |
|---|---|
| `schema` | format version of this snapshot (1) |
| `suite`, `capabilities` | the suite and the capabilities its jobs exercise |
| `published` | when the snapshot was written (UTC) |
| `bngsim` | version and compiled build commit of the BNGsim under test |
| `engines` | each reference engine's version(s) across the records |
| `node_type` | the CPU model, architecture and OS of the timing node |
| `harness` | the harness version, the vendored verdict code, and the git commits of the harness that ran the jobs |
| `members` | the suite's models in position order: source pin (repo, commit, path, sha256), publications, verification evidence, size |
| `jobs` | every job: id, problem, kind, outcome, and the path of its record |

## correctness.csv

One row per (problem, reference engine). BNGsim is the system under test; each row is
BNGsim's agreement with one reference.

| column | meaning |
|---|---|
| `suite`, `position`, `problem` | the suite, the model's position in it (1-10), the problem id |
| `species`, `reactions` | size of the generated network |
| `outcome` | the job's outcome: `PASS` (agrees with every reference), `DIFF` (disagrees with one), `REFERENCE_FAILED` (BNGsim ran; some reference did not), `EXCEPTION` (BNGsim failed), `TIMEOUT` (BNGsim ran out of time), `BAD_TEST` (the model could not be prepared, or every engine failed) |
| `agreement` | references agreed / attempted, e.g. `FULL 4/4`, `PARTIAL 3/4 (copasi: crash)`; a reference that does not apply is named (`run_network: n/a`) but not counted |
| `engine` | the reference engine |
| `status` | `ok` (agrees), `DIFF` (ran, disagrees), `crash` (failed to run), `timeout` (ran past its per-engine wall budget), `n/a` (it cannot simulate this model faithfully, an `UNSUPPORTED` error; counted as attempted), `not_applicable` (the comparison does not exist for this model, so the reference was not run and is not counted: run_network on an SBML model with no faithful BNGL form) |
| `passed` | the verdict, empty when the reference did not run |
| `max_err_over_peak` | worst disagreement as a fraction of each species' peak |
| `max_rel` | the verdict's worst relative error over failing cells (0 on a pass) |
| `n_cells` | trajectory cells compared |
| `cold_seconds` | the reference's cold build + first solve, in seconds |
| `error` | why the reference has no verdict; for `not_applicable`, why it does not apply |

## timing.csv

One row per (problem, engine), BNGsim included. Times are absolute seconds; a speedup
is a ratio of two rows, not stored.

| column | meaning |
|---|---|
| `suite`, `position`, `problem`, `species`, `reactions`, `outcome` | as above; `outcome` is `PASS` when every timed engine reproduced its first run |
| `engine`, `role` | the engine, and how it was timed (`warm, in-process`, or `fresh subprocess` when every solve is a new process) |
| `status` | `timed`, or why it was not: `crash`, `timeout` (it ran past its budget in the correctness job), `n/a`, `not_applicable` (as in correctness.csv) |
| `presim_seconds` | the one-time cost before the timed solve, measured cold (BNGsim: network generation + code generation + load; a reference: its own code generation + load); empty for a fresh subprocess, whose load is inside every solve |
| `netgen_seconds`, `codegen_seconds`, `load_seconds`, `backend` | the parts of `presim_seconds`, and BNGsim's code-generation backend |
| `sim_median_seconds` | median warm solve over the timed rounds |
| `sim_ci_low_seconds`, `sim_ci_high_seconds` | bootstrap 95% confidence interval of that median |
| `rounds` | timed rounds (engines interleaved round-robin, docs/DESIGN.md section 7) |
| `reproducible` | whether every timed repetition reproduced the engine's first run |
| `error` | why the engine was not timed; for `not_applicable`, why it does not apply |

The full record of each job, under `jobs/`, carries everything the CSVs summarize, plus
the per-repetition times, the per-engine configuration and the environment.
