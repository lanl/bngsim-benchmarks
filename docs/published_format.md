# Published results

`published/` holds our own results, one snapshot per suite and BNGsim build, written by
`bench publish SUITE` and committed. Anyone can re-run the same suite (`bench run
SUITE`, or `bench submit SUITE` on a Slurm cluster) and compare what they get with what
is here.

```
published/README.md      every suite: its latest snapshot, tally and status (generated)
published/index.json     the same, machine-readable (generated)
published/<suite>/bngsim-<version>-<build commit>/
    manifest.json        what was measured, on what, with which engines and harness
    correctness.csv      one row per (problem, setting, reference engine)
    timing.csv           one row per (problem, setting, engine)
    README.md            the snapshot for people, generated from the three files above
    jobs/<job id>.json   every job's full result record
    jobs/<job id>.npz    a correctness job's stored trajectories, when the suite's
                         total is at most 5 MB (see Trajectories below)
```

The READMEs and index.json are generated, never edited: see Summaries and staleness
below.

A snapshot is complete: every job of the suite has a current result, all from one
BNGsim build, and every timing from one node type. `bench publish` refuses otherwise.
The one exception is a member whose correctness outcome does not let its timing run
(`DIFF`, `EXCEPTION`, `BAD_TEST`, `TIMEOUT`). Its timing is **withheld**: it is not
demanded, and its record is not in `jobs/`, even where an earlier run left one. It
gets one `timing.csv` row saying why, with no numbers. A timing is published only
beside a current `PASS` or `REFERENCE_FAILED` correctness result (docs/DESIGN.md
§5.2).
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
| `jobs` | every job: id, problem, kind, outcome, and the path of its record; for a correctness job that stored trajectories, `trajectories` (the `.npz` path, `in_snapshot`, `bytes`, `sha256`); for a withheld timing, `outcome` `WITHHELD`, `withheld` (the reason, e.g. `correctness DIFF`) and no record |
| `trajectories` | the suite's stored trajectories: `in_snapshot`, the number of `files`, their `total_bytes`, and the `limit_bytes` that decides (5,000,000) |

## correctness.csv

One row per (problem, reference engine). BNGsim is the system under test; each row is
BNGsim's agreement with one reference.

| column | meaning |
|---|---|
| `suite`, `position`, `problem` | the suite, the model's position in it (1-10), the problem id |
| `setting` | the job's setting (the fourth part of its id): `auto` for most capabilities; a member has one job per setting where a capability runs several (network-free `nf_exact` and `nf_reject`; sens-net's `auto` and `fitted`) |
| `species`, `reactions` | size of the generated network |
| `outcome` | the job's outcome: `PASS` (agrees with every reference), `DIFF` (disagrees with one that ran, whatever else failed), `REFERENCE_FAILED` (BNGsim ran; some reference did not, and every one that ran agrees), `EXCEPTION` (BNGsim failed), `TIMEOUT` (BNGsim ran out of time), `BAD_TEST` (the model could not be prepared, or every engine failed) |
| `agreement` | references agreed / attempted, e.g. `FULL 4/4`, `PARTIAL 3/4 (copasi: crash)`; a reference that does not apply is named (`run_network: n/a`) but not counted |
| `engine` | the reference engine |
| `status` | `ok` (agrees), `DIFF` (ran, disagrees), `crash` (failed to run), `timeout` (ran past its per-engine wall budget), `n/a` (it cannot simulate this model faithfully, an `UNSUPPORTED` error; counted as attempted), `not_applicable` (the comparison does not exist for this model, so the reference was not run and is not counted: run_network on an SBML model with no faithful BNGL form) |
| `passed` | the verdict, empty when the reference did not run; on ODE, true only when both halves of the gate pass (docs/DESIGN.md §6.4) |
| `failed_check` | for an ODE `DIFF`, which half of the gate failed: `vendored` (bngsim's deterministic verdict, `parity_core.differ`), `species` (the per-species check) or `both`; empty when the reference agreed or did not run, or its verdict has no two halves |
| `max_err_over_peak` | worst disagreement as a fraction of each species' peak |
| `max_rel` | bngsim's deterministic verdict's worst relative error over its failing cells (0 when it passes) |
| `n_cells` | trajectory cells compared |
| `cold_seconds` | the reference's cold build + first solve, in seconds |
| `error` | why the reference has no verdict; for `not_applicable`, why it does not apply |

## timing.csv

One row per (problem, engine), BNGsim included. Times are absolute seconds; a speedup
is a ratio of two rows, not stored.

| column | meaning |
|---|---|
| `suite`, `position`, `problem`, `setting`, `species`, `reactions`, `outcome` | as above; `outcome` is `PASS` when every timed engine reproduced its first run, and `WITHHELD` (not a job outcome) on the one row of a withheld timing |
| `engine`, `role` | the engine, and how it was timed (`warm, in-process`, or `fresh subprocess` when every solve is a new process) |
| `status` | `timed`, or why it was not: `crash`, `timeout` (it ran past its budget in the correctness job), `n/a`, `not_applicable` (as in correctness.csv), or `withheld`: the member's correctness outcome does not let it be timed, so the row has no engine and no numbers |
| `presim_seconds` | the one-time cost before the timed solve, measured cold (BNGsim: network generation + code generation + load; a reference: its own code generation + load); empty for a fresh subprocess, whose load is inside every solve |
| `netgen_seconds`, `codegen_seconds`, `load_seconds`, `backend` | the parts of `presim_seconds`, and BNGsim's code-generation backend |
| `sim_median_seconds` | median warm solve over the timed rounds |
| `sim_ci_low_seconds`, `sim_ci_high_seconds` | bootstrap 95% confidence interval of that median |
| `rounds` | timed rounds (engines interleaved round-robin, docs/DESIGN.md section 7) |
| `reproducible` | whether every timed repetition reproduced the engine's first run |
| `error` | why the engine was not timed; for `not_applicable`, why it does not apply; for `withheld`, the correctness outcome that withholds the timing (`correctness DIFF`) |

The full record of each job, under `jobs/`, carries everything the CSVs summarize, plus
the per-repetition times, the per-engine configuration and the environment.

## Trajectories

A correctness job stores the trajectories its verdict was computed from beside its
record, as `<job id>.npz` (compressed float64, readable without pickle): every engine's
compared trajectory under the engine's name, all in the same columns, with the output
times (`time`) and the compared species (`species`: BNGsim's species names on the `.net`
path, SBML ids on the SBML path), and `run_network_sparse` where the model asks for a
sparse run. They are what `bench rescore` re-scores a verdict from.

A suite's trajectories go into its snapshot, beside each record, when they total at most
5 MB (5,000,000 bytes); otherwise none of them do. Either way a record's `trajectories`
says what the job stored and whether it is here:

| key | meaning |
|---|---|
| `file` | the `.npz`'s name, relative to the record |
| `bytes`, `sha256` | its size and SHA-256, which identify the file in or out of the snapshot |
| `arrays` | the arrays it holds |
| `in_snapshot` | `true` when the file is beside the record in this snapshot, `false` when it is not |
| `note` | when it is not: why (the suite's total against the limit) |

A record whose job stored nothing (`stored: false`, with a `reason`) names no file.
`bench publish` checks every file it ships against the size and SHA-256 its record
names, and refuses to publish if one is missing or differs.

Measured on the core suites' correctness results of November 2026, neither fits: the
ten ode-sbml files total 13.3 MB (0.1 to 3.4 MB each), and ode-net's eight total 32.7 MB
(9.2 to 9.5 MB for each of its three largest networks), before its two largest models,
which had no current result to measure.

## Summaries and staleness

Three kinds of file under `published/` are generated, so that what people read cannot
drift from the data:

- `published/<suite>/<label>/README.md`, written by `bench publish SUITE` from that
  snapshot's `manifest.json`, `correctness.csv` and `timing.csv` alone: a header (the
  BNGsim version and build, the engine versions, the harness version and commits, the
  timing node, the date), a correctness table (each member's outcome and every
  reference's verdict, then one line per check that did not pass, naming the engine and
  why), and a timing table (absolute pre-sim and solve medians with their 95% CI, per
  engine). A speedup appears only as a view derived from that table (a reference's solve
  median over BNGsim's); it is not stored.
- `published/README.md` and `published/index.json`: every suite of `specs.SUITES`, its
  latest snapshot (by `published` time), the tally of its correctness outcomes, and its
  status: `CURRENT`, `OUT OF DATE` with every reason, or `NOT PUBLISHED`.
  `index.json` also records the newest bngsim release seen on PyPI
  (`bngsim_release`: `version`, and the day it was `checked`).

`bench publish SUITE` regenerates all of them; `bench publish --index` regenerates them
without publishing (`--offline`: judge by the release `index.json` records instead of
looking up PyPI).

A snapshot is **OUT OF DATE** (`bench.staleness`) when

| signal | from |
|---|---|
| a newer bngsim release is on PyPI than the snapshot's (pre-releases and yanked releases do not count) | PyPI's JSON API, looked up by `bench publish` and `bench staleness` |
| the environment pins another bngsim | `pixi.toml` |
| the suite's members or their positions differ | `problems/` against `manifest.json` `members` |
| a member's jobs differ: a job id defined now is not in the snapshot, or one in it is no longer defined (an engine added, a setting changed) | the job list (`bench list`) against `manifest.json` `jobs` |
| a job's definition differs from the one its record was keyed on: model file (sha256), PEtab tables, protocol, verdict, harness version, setting, timing protocol, input conversion | `Job.definition` now against each record's `key_parts.job.definition` |
| the harness the records were keyed on differs: the bench version, the vendored verdict code, the environment the harness forces | `env.harness_identity()` against each record's `key_parts.environment.harness` |
| a reference engine's version differs from the one the environment pins | `pixi.toml` exact pins (libRoadRunner, COPASI, AMICI, PEtab, Antimony) and `bench.fetch` (BioNetGen, RuleMonkey) against the records' `reference_versions` and BioNetGen |

All but the first are read from committed files, never from the installed environment,
so every machine gets the same answer: a job reads its engines' versions from the
environment it runs in, and here they are replaced by what the environment pins. A
withheld timing has no record, so only its correctness job's definition is compared.

Two guards keep this honest:

- `tests/test_summary.py` regenerates every README and `index.json` from the committed
  data (with the release `index.json` records) and fails on any difference, in CI's
  unit tests. A change to a spec, a snapshot or the code that would change a status
  fails there until `bench publish --index` is run and its output committed.
- `bench staleness` prints every suite's status and reasons, looking up PyPI (exit 0
  when every published suite is current, 1 when one is out of date, 2 when PyPI cannot
  be read; `--offline` uses the recorded release). `--markdown FILE` writes the
  out-of-date suites as an issue body; `--write` also regenerates the summaries.
