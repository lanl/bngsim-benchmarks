# Core suites

The core layer is a set of suites, each one bngsim feature on one input format, with
exactly 10 published, verified models in a fixed order (DESIGN §3.2). This file records
each suite's members and what they cover. The specs under `problems/core/` are the
source of truth for membership; `bench list` prints every suite in order, and a suite
name selects its jobs (`bench run ode-net`).

## ode-net: ODE on the `.net` generated from published BNGL

Members are ordered by network size. Positions 1, 3, 5, 7 and 9 also seed the Antimony
ODE suite (#38).

| # | problem | species / reactions | dynamics | stiffness ratio, peak (sustained) | evidence | how it entered |
|---|---|---|---|---|---|---|
| 1 | `noise_resistant_circadian_oscillator_vilar2002` | 9 / 16 | oscillator (limit cycle, ~25.6 h) | 8.6e6 (8.8e3) ‡ | `equations` + `data` | top-up: small size; second oscillator |
| 2 | `tcr_signaling_bistability_lipniacki2008` | 37 / 97 | bistable switch | 1.2e3 (1.2e3) ‡ | `data` + `equations` | top-up: bistable switch; the paper vetted it for its SSA and PSA tables |
| 3 | `cell_cycle_oscillator_lang2024` (`_fitted`) | 73 / 332 | oscillator | 1.7e7 (6.1e6) ‡ | `author_file` + `equations` | paper seed, at the authors' fitted parameters |
| 4 | `scaffolded_mapk_cascade_kocieniewski2012` | 85 / 487 | relaxation (monostable) | 3.0e3 (2.8e3) † | `data` + `theory` | paper seed |
| 5 | `shp2_regulation_and_function_barua2007` | 149 / 1032 | relaxation (monostable) | 2.1e4 (2.1e4) † | `data` + `structure` | paper seed |
| 6 | `combinatorial_egfr_signaling_blinov2006` | 356 / 3749 | transient, then relaxation | 5.8e4 (2.7e2) † | `data` | paper seed |
| 7 | `beta_catenin_destruction_complex_barua2013` | 409 / 2737 | relaxation (monostable) | 6.9e6 (5.2e6) † | `data` | paper seed |
| 8 | `endothelial_vegfr2_and_cd47_signaling_zhang2023` | 870 / 5602 | transient | 7.3e9 (1.3e8) ‡ | `data` + `structure` | top-up: fills the gap between 409 and 1612 species |
| 9 | `mapk_adaptive_resistance_frohlich2023` (`_preeq`) | 1612 / 17516 | transient (EGF-induced pERK pulse in vemurafenib-adapted cells) | 5.8e9 (1.7e9) ‡ | `author_file` + `equations` | replaces the paper's Faeder 2003 `_fyn` variant, which is not published; seeded with the authors' steady state |
| 10 | `ksr1_raf_mek_inhibitor_response_imoto2026` | 2709 / 27457 | relaxation with overshoot | 2.6e8 (2.6e8) ‡ | `data` | top-up: thousands of species |

Stiffness ratio is max |Re λ| / min over nonzero |Re λ| of the ODE Jacobian along the
model's own trajectory. Peak is the maximum over time; sustained is the median.

- † From the BNGsim paper's table of representative models: conservation-reduced
  Jacobian, from bngsim's Jacobian characterization.
- ‡ Computed for this suite on the record's committed reference `.net`: bngsim's
  analytical Jacobian at 16–40 points of the model's own horizon (rtol = atol = 1e-8),
  with conservation modes (|λ| < 1e-10 · max |λ|) dropped. That cutoff caps the ratio at
  10^10, and 8 and 9 come within a factor of 2 of it. For 9 the dropped modes are exactly
  its 10 conservation laws at every sampled time (n − rank S), with a gap of at least 50×
  to the smallest kept mode. For 8 the peak is its first sample, where the smallest kept
  mode sits just above the cutoff, so its sustained value is the robust number.

### What the suite spans (DESIGN §3.2, ODE)

| axis | members |
|---|---|
| size | 9 to 2709 species, roughly log-spaced |
| linear-solver policy | the paper's runs put the seed on all three branches of bngsim's automatic policy: built-in dense (3, 4), KLU (5), BLAS dense (6, 7). Each correctness result records the branch every member takes. |
| relaxation to steady state | 4, 5, 7, 10 |
| oscillator | 1, 3 |
| bistable switch | 2 |
| strongly stiff | every member reaches a peak ratio above 10^3; 2 only just, 8, 9 and 10 above 10^8 |

### Membership decisions

- **Seed:** the paper's six representative BNGL models, chosen there as "published,
  real-world, stiff systems lying along the density–size trend". Four are kept as they
  are.
- **Faeder 2003 `_fyn` dropped:** its own header states that the Fyn extension is not
  part of the published paper, so it cannot meet the published-and-verified rule (§4.1).
  `mapk_adaptive_resistance_frohlich2023` takes its place in the same size range.
- **Lang 2024, `_fitted` in place of `_fused`:** the paper's table used `_fused`, but no
  committed check covered that file. A check against the authors' release
  (`paulflang/cell_cycle_petab`, MIT from `362b666`) found that `_fused` has the authors'
  exact network but their pre-fit nominal parameters, and 13 promoter initial values that
  differ from their PEtab problem. BNGL-Models now adds `_fitted`, written mechanically
  from the authors' optimized table and PEtab wt condition. It reproduces the authors' own
  simulation output to 5.4e-6, and it is the model the paper's figures show.
- **Fröhlich 2023, `_preeq` in place of the primary:** the record's committed checks did
  not qualify: the comparison with the authors' model read a CSV that nothing committed
  generates, and the Fig. 2A comparison asserted no tolerance. A re-runnable check
  against the authors' MARM2.0 release (`labsyspharm/marm2-supplement` @ `7e97789`,
  wshlavacek/BNGL-Models#59) found the primary has the authors' exact model (parameters
  to 1.3e-13, identical network). Its seeds, though, are a 24 h integration, where the
  authors pre-equilibrate to steady state, so under their protocol it misses their
  simulation by up to 1.1e-2. BNGL-Models now adds `_preeq`, the same model seeded with
  the authors' steady state and written mechanically. Its run_network output reproduces
  the authors' own simulation to 1.8e-6.
- **Top-ups** fill what the seed lacks: a small model, a bistable switch, the gap between
  409 and 1612 species, and a network in the thousands.

## ode-sbml: ODE on published SBML, read as it is

Every member is a curated BioModels file, pinned at `sys-bio/temp-biomodels`
`6a09daf4`, the commit lanl/bngsim's rr_parity corpus pins. Its protocol is the time
course of the SED-ML that sits beside it there, whose `<model source>` is the pinned
file. Members are ordered by size. Positions 2, 4, 6, 8 and 10 also seed the Antimony
ODE suite (#38).

| # | problem | BioModels | species / reactions | events | output (from t=0) | evidence | how it entered |
|---|---|---|---|---|---|---|---|
| 1 | `brain_energy_metabolism_winter2017` | BIOMD0000000627 | 66 (65) / 64 | none | t=100–400, 1000 steps | `author_file`: Fig. 3 | top-up |
| 2 | `map3k8_vemurafenib_resistance_giani2019` | BIOMD0000000883 | 75 / 85 | none | 0–14400, 1000 | `author_file`: Fig. 2C, the with-inhibitor curve | top-up |
| 3 | `alpha_synuclein_parkinsons_bst_sass2009` | BIOMD0000000575 | 76 / 58 | none | 0–100, 1000 | `author_file`: Fig. 15, the baseline model | top-up |
| 4 | `glucose_phosphotransferase_system_nishio2008` | BIOMD0000000571 | 76 / 92 | 1 (t=500) | 0–1000, 1000 | `author_file`: Fig. 3A | top-up |
| 5 | `hepatocyte_calcium_propagation_verma2016` | BIOMD0000000834 | 76 (75) / 134 | 1 (t=200) | t=400–700, 43 | `author_file`: Fig. 2 ‡ | paper seed |
| 6 | `shear_stress_nitric_oxide_koo2013` | BIOMD0000000468 | 79 / 74 | none | 0–12000, 1000 | `author_file`: Fig. 6, the shear-stress curve | paper seed |
| 7 | `alpha_synuclein_neuronal_homeostasis_ouzounoglou2014` | BIOMD0000000559 | 90 / 136 | none | 0–600000, 1000 | `author_file`: Fig. 7 of Supplementary material 5 | paper seed |
| 8 | `mapk_signalling_control_hornberg2005` | BIOMD0000000667 | 103 / 148 | none | 0–6000, 100 | `author_file`: Fig. 2 | paper seed |
| 9 | `insulin_signalling_oxidative_stress_smith2013` | BIOMD0000000474 | 133 / 367 | 4 (t=0+, 15, 2785, 2800) | 0–2805, 1000 | `author_file`: Fig. 3A, from this events variant | paper seed |
| 10 | `segment_polarity_network_vondassow2000` | BIOMD0000001065 | 148 / 480 | none | 0–1100, 220 | `author_file` † | paper seed |

Species are counted as bngsim's `Model.from_sbml` counts them, the count the suite is
ranked by; where the SBML's own count differs it is in parentheses. Winter 2017's
count includes its rate-ruled compartment, and Verma 2016's the parameter its event
sets. Each `author_file` row is a curated BioModels entry whose curators reproduced
the named figure of the paper; the spec quotes what the curation reproduced and how it
covers the pinned file.

- † von Dassow 2000's curation reproduced the paper's Fig. 2, a radial plot over the
  model's parameters, and its SED-ML names no figure. So its 0–1100 time course is the
  curators' protocol rather than a published figure. The same folder holds that time
  course from AMICI, COPASI, tellurium and VCell.
- ‡ The curators of Verma 2016 describe their reproduction of Fig. 2 as similar in
  pattern but not exact.

### Protocol and comparison

- The output window and its steps are the SED-ML's. Winter 2017 and Verma 2016 output
  from t=100 and t=400, after integrating from t=0, so the protocol has an
  `output_start` beside its `t_start`.
- `rtol` is the SED-ML's: 1e-6, and 1e-8 for von Dassow 2000. The SED-MLs give no
  absolute tolerance, only COPASI's adjustment factor, so `atol` is 1e-9, the value
  lanl/bngsim's rr_parity job manifest uses for them. BNGsim takes it as a plain float,
  its own per-species ceiling, as an SBML user passes it; on the `.net` path it is held
  exactly instead, to match run_network.
- **Nishio 2008's `atol` is 1e-15 instead.** 34 of its 59 compared species are promoter
  binding sites, free or bound, in molar units, whose peaks lie between 1.5e-11 and
  1e-9: wholly below 1e-9. At 1e-9, the engines that hold atol exactly do not resolve them:
  - AMICI and run_network land 2–6% of a species' peak off the converged solution
    (rtol 1e-10, atol 1e-18), and so does BNGsim given `bngsim.ExactAtol`;
  - BNGsim's float atol, libRoadRunner and COPASI scale it per species, and are
    within 1e-5.

  At 1e-15, every compared species peaks at 1.5e4 atol or more, and every engine is
  within 7.5e-4 of the converged solution: run_network is the worst, AMICI is at
  1.1e-5. To see the disagreement, set the spec's `atol` back to 1e-9 and re-run the
  correctness job. run_network and AMICI are then each 10.4 atol off, on species that
  peak at 180 and 460 atol (IICB_Mlc, 6% of its peak; mRNA_ptsG, 2%). That is just
  past the per-species check's floor of 10 atol, so a protocol this loose leaves the
  check little to judge.
- What is compared is each **state species**: every species the SBML neither declares
  constant nor sets by an assignment rule. Each engine reports them as concentrations,
  by SBML id, at exactly the protocol's output times, or it has failed.
- A reference agrees with BNGsim only when bngsim's vendored verdict **and** a
  per-species check pass (DESIGN §6.4). The per-species check holds each species to
  1e-2 of its own peak, or 10 atol where that is larger. On this suite the vendored
  verdict alone judges few species: 1 of Nishio 2008's 59, for example, and 16 of
  Smith 2013's 105.
- libRoadRunner, COPASI and AMICI read the pinned SBML. **run_network** reads a network
  derived from it. `bngsim.convert.sbml_to_bngl` writes compartmental BNGL, in which a
  fixed-time event becomes a change of state between `simulate` phases, and BNG2.pl
  generates the `.net`. The phases then run as BNG2.pl runs them: a fresh run_network
  per phase, from the state the last one left. The derivation is model preparation,
  charged to no engine. It gates run_network: BNGsim on the derived network must
  reproduce BNGsim on the SBML at a tight tolerance. Where the SBML has no faithful
  BNGL form, or the check fails, run_network is *not applicable*. That is recorded with
  its reason, shown in the reports, and left out of the count of references. Winter
  2017 is the one such member: the converter refuses its amount-valued species in
  compartments whose volume is not 1.
- At an instant where the derived protocol changes the state (an event's time on the
  output grid: Nishio 2008's t=500, Smith 2013's t=0), what run_network reports is the
  conversion's convention, before or after the change. Those rows are left out of
  run_network's verdict, and named in it.

### What the suite spans (DESIGN §3.2, ODE)

| axis | members |
|---|---|
| size | 66 to 148 species: no curated BioModel with a figure-reproducing SED-ML has more than 148, so this suite cannot reach the thousands the `.net` suite does |
| events | 4 (one), 5 (one, before the output starts), 9 (four) |
| output after the integration start | 1, 5 |
| compartments that are not of volume 1 | 1, 2, 9 |
| linear-solver policy | bngsim's automatic policy takes KLU on all ten |

### Membership decisions

- **Criterion:** the paper's own: the largest curated BioModels that carry a
  figure-reproducing SED-ML horizon. That is the `figure_sedml` tier of lanl/bngsim's
  rr_parity corpus, 424 models, ranked by bngsim's species count and then by SBML
  reactions. This ranking reproduces the paper's six exactly, including a three-way
  tie at 76 species that Verma 2016 wins on reactions. The six take positions 5–10, and
  the next four by the same ranking fill positions 1–4.
- **Pinning:** Giani 2019 is missing from lanl/bngsim's temp-biomodels manifest; its
  sha256 was checked against the file at the pinned commit.
- **Smith 2013** is pinned as the curators' events variant
  (`MODEL1212210000_eventsForFigure3A.xml`), not the main model: it adds the two insulin
  administrations Fig. 3A needs, and the curation figure comes from it.
- **Koo 2013:** its SED-ML is named for the no-shear-stress condition, but it runs the
  file unchanged, which by the curation notes gives the shear-stress curve of Fig. 6.

## ss-net: steady state on the `.net` generated from published BNGL

Members are ordered by network size. Each runs a dose-response: a scan of one
parameter, every point solved from the file's seed species at that point's parameters,
as BNG's `parameter_scan` does (`bench.capabilities.steady`). Where the file has its own
`parameter_scan` and the verification checks those points, the scan is the file's;
otherwise it is the verification's.

| # | problem | species / reactions | dose (points) | mono / bistable | evidence | how it entered |
|---|---|---|---|---|---|---|
| 1 | `genetic_switch_gardner2000` | 2 / 4 | alpha_2, 1–500 linear (100, the file's scan) | bistable for 14.76 < alpha_2 < 80.57 (13 points) | `equations`: exact fixed points of Box 1 Eq. 1, both stable branches and the saddle | paper seed |
| 2 | `lac_operon_dreisigmeyer2008` | 3 / 8 | l_ext, 1e-2–1e6 uM log (100, the file's scan) | monostable (relabelled) | `equations`: SciPy Eq. 1 at all 100 doses | paper seed |
| 3 | `kinetic_proofreading_hlavacek2001` | 9 / 14 | koff, 1e-3–1 /s log (20) | monostable | `theory`: closed form D_i/D = alpha^i (1 − alpha) at every dose | paper seed |
| 4 | `tcr_signaling_bistability_lipniacki2008` | 37 / 97 | N_agonist, 1–1000 log (20) | bistable for 8.85 < N < 772 (13 points) | `equations` (independent 37-state ODE, both branches) + `data` (Fig. 7A) | top-up: the second switch |
| 5 | `scaffolded_mapk_cascade_kocieniewski2012` | 85 / 487 | Stot, 1e1–1e6 log (61, the file's scan) | monostable, biphasic (prozone) | `data`: digitized Fig. 2 | paper seed |
| 6 | `camkii_holoenzyme_activation_michalski2012_pp1` | 132 / 868 | CaM4_tot, 1e-4–1 uM log (25, the file's scan), a clamped species | monostable, non-equilibrium | `data` (Fig. 5a at three PP1 levels) + `equations` | top-up |
| 7 | `membrane_anchored_multisite_phosphorylation_dushek2011` | 143 / 284 | r = E/F, 0.01–100 log (41, the file's scan) | monostable (linear network) | `theory`: exact null space at every dose + `data` (Fig. 2) | top-up: an exact reference at every dose |
| 8 | `shp2_regulation_and_function_barua2007` | 149 / 1032 | Rtot, 0.05–0.5 uM linear (10, the file's scan), through R_dim = Rtot/2 | monostable | `data`: digitized Fig. 3 | paper seed |
| 9 | `beta_catenin_destruction_complex_barua2013` | 409 / 2737 | APCtot, 31855–3185540 log (11, the file's scan) | monostable; 4 write-only sinks | `data`: digitized Fig. 5A | paper seed |
| 10 | `ksr1_raf_mek_inhibitor_response_imoto2026` | 2709 / 27457 | RAFi1_0, 0–50 nM (20 of the record's 27-dose ladder), a clamped species, at the paper's type II factors | monostable | `data`: digitized Fig. 3 (725 steady states) | top-up: thousands of species |

### Protocol and comparison

- The system under test is BNGsim's `steady_state(method="newton")` (a CVODE burst,
  then KINSOL); BNGsim's integration to steady state is run and recorded beside it,
  not scored. The references are run_network integrating to the protocol's horizon
  (the integration reference, which must have settled), libRoadRunner's
  `steadyState`, COPASI's steady-state task and AMICI's steady state by simulation, on
  the SBML exported from the same `.net`; and two engine-free checks, the residual and
  every conservation law's initial total (DESIGN §5, `bench.capabilities.steady`).
- A reference that returns a state the integration from the initial state does not
  reach (a saddle, the other branch of a switch), a state that fails an engine-free
  check, or no state is `REFERENCE_FAILED` with its reason, and is not scored.
- Horizons and tolerances are the verification's or the file's scan's, lengthened
  where a file's own scan does not settle: Gardner 2000 (t = 1000, not 20), kinetic
  proofreading (20000 s, not 1000), the lac operon (5000 min, not 500), Barua 2007
  (1e4 s, not 1e3) and Barua 2013 (1e6 s, not 2.5e5).
- Barua 2013's four write-only sinks grow forever, so the full system has no steady
  state; they are left out of every comparison and check (BNGsim's `mask`).
- Imoto 2026 is solved from the seed at every dose (its record's driver continues from
  the previous dose), so each point is a full relaxation of a 2709-species network. It
  keeps 20 doses with its own per-engine budget (`engine_budget_seconds` in its
  protocol), and needs a long allocation.

### What the suite spans (DESIGN §3.2, steady state)

| axis | members |
|---|---|
| size | 2 to 2709 species |
| bistable | 1, 4 (each run on the branch the seed reaches; the verification covers both) |
| conservation laws (a singular Jacobian) | 3, 4, 5, 6, 7, 8, 9, 10 |
| write-only sinks (masked) | 9 |
| the dose is a clamped species | 6, 10 |
| the dose sets a conserved total or an initial amount | 4, 5, 7, 8, 9 |
| an exact reference at every dose | 3, 7 |

### Membership decisions

- **Seed:** the paper's six steady-state models, all kept. The lac operon is labelled
  bistable in the paper's table and in bngsim's own suite; it is monostable at the
  published growth rate (one root at every dose), as its record says, and is kept
  with that label. Barua 2013 is kept with its sinks masked.
- **Top-ups** (proposal D6, accepted): Lipniacki 2008 for a second genuine switch,
  CaMKII `_pp1` and Dushek 2011 between 37 and 149 species (Dushek also gives an exact
  reference at every dose), and Imoto 2026 for thousands of species.
- **Seed evidence closed upstream** (D6c): Gardner's second branch and saddle
  (wshlavacek/BNGL-Models#67), the closed form at all 20 kinetic proofreading doses
  (#68), and both branches of Lipniacki 2008 against the equations and Fig. 7A (#69).
- **Not run yet:** the proposal's continuation along each branch of the bistable
  members, `steady_states()` multistart root counts, the file's `parameter_scan`
  replayed by `run_bngl`, and steady-state sensitivities. The capability solves each
  point from the seed only; these are settings to add.
