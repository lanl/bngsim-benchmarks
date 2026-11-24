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

## ssa-sbml: exact SSA on published SBML, amounts as molecule counts

Ten SBML models whose amounts are molecule counts, run with each engine's exact SSA:
BNGsim, libRoadRunner (gillespie) and COPASI (direct method) read the pinned file;
run_network reads a network derived from it where that network has the SBML's jumps
(DESIGN §5.3). Ordered by species, then reactions. Horizons are the published ones
(the curation figures, or DSMTS's settings), never a placeholder SED-ML's. Pins: the
SBML Test Suite at `sbmlteam/sbml-test-suite` 473e119d for DSMTS, `sys-bio/temp-biomodels`
6a09daf4 for the BioModels.

| # | problem | source | species / reactions | events | horizon, steps | replicates | references | evidence |
|---|---|---|---|---|---|---|---|---|
| 1 | `dimerisation_dsmts_003_01` | DSMTS 003-01 | 2 / 2 | none | 50, 50 | 10^4 | rr, COPASI, exact moments (table); run_network n/a | `theory` |
| 2 | `abeta_dimer_aggregation_proctor2012` | BIOMD0000000462 | 6 / 8 | none | 100 y, 100 | 5000 | rr, COPASI; run_network n/a | `author_file`: Figs 5A, 5B (COPASI stochastic) |
| 3 | `repressilator_elowitz2000` | BIOMD0000000012 | 6 / 12 | none | 1000 min, 100 | 10^4 | rr, COPASI; run_network n/a | `author_file`: Fig. 1c |
| 4 | `mirna_positive_feedback_osteoarthritis_proctor2017` | BIOMD0000000862 | 9 / 11 | 2 (t = 3600, 7200 s) | 8 h, 100 | 10^4 | rr (time-event warning), COPASI; run_network n/a | `author_file`: Suppl. Fig. A(i) |
| 5 | `noise_resistant_circadian_oscillator_vilar2002_sbml` | BIOMD0000000035 | 10 / 16 | none | 400 h, 200 | 10^4 | run_network (derived), rr, COPASI | `author_file`: Fig. 2AB |
| 6 | `p21_ros_senescence_passos2010` | BIOMD0000000287 | 23 / 37 | 3 (t = 172800, 172860, 691200 s) | 10 d, 100 | 100 | rr (time-event warning), COPASI; run_network n/a | `author_file`: Fig. S17 (Fig. 3B stochastic) |
| 7 | `ras_camp_pka_oscillations_besozzi2012` | BIOMD0000000478 | 33 / 39 | none | 1400 a.u., 140 | 200 | rr, COPASI; run_network n/a | `author_file`: Fig. 6 (COPASI stochastic) |
| 8 | `ubiquitin_proteasome_system_proctor2007` | BIOMD0000000105 | 39 / 94 | none | 20 h, 100 | 10^4 | rr, COPASI; run_network n/a | `author_file`: Fig. 3 (Gillespie2) |
| 9 | `telomere_uncapping_checkpoint_proctor2007` | BIOMD0000000087 | 55 / 43 | 9 state-triggered | 12 h, 100 | 4000 | rr, COPASI; run_network n/a (state events) | `author_file`: Fig. 5 (Gillespie 2.0) |
| 10 | `uchl1_protein_aggregation_proctor2010` | BIOMD0000000293 | 136 / 316 | none | 20 h, 100 | 200 | rr, COPASI; run_network n/a | `author_file`: Fig. 2 (COPASI stochastic) |

run_network n/a: the network `bngsim.convert.sbml_to_bngl` derives keeps the ODE but
writes each law that is not mass action as per-species flux reactions, so its jumps
are not the SBML's (members 1-4, 7, 8, 10); it refuses state-triggered events (9);
and on 6 it would set a parameter it promotes to a species to 0.6, which run_network's
SSA rounds to 1. Each result records the reason.

### What the suite spans (DESIGN §3.2, SSA)

| axis | members |
|---|---|
| size | 2 to 136 species, 2 to 316 reactions |
| exact moments | 1 (the DSMTS tables) |
| repeated reactants, written `k*X*(X-1)/2` | 1, 2, 7, 8, 10 |
| time-triggered events | 4, 6 (rows at an event instant left out) |
| state-triggered events | 9 |
| cost per trajectory | 1 ms (1) to minutes (6: about 1.35e8 events) |

### Membership decisions

- Members, horizons and pins are those of the stochastic-suites proposal (§2.2 and
  Appendix B), accepted 2026-11-23, with 287 and 105 in for the proposal's S07/S08 and
  087 for 344.
- **Ensemble sizes** below the 10^4 default (462: 5000, 087: 4000, 478, 293: 200,
  287: 100) are a cost decision (2026-11-24); each verdict records its replicates.
  287's correctness job runs each engine as its own array task (its parts).
- **libRoadRunner on time-triggered events** (4, 6) is run, not screened out: 2.11
  fires them, at its first reaction event after the trigger, and warns that this is
  not precise; the warning is in its configuration.
- **COPASI on 087** looked wrong in a wave-1 probe (n = 10); a DIFF it causes is
  recorded, not worked around.

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

## ss-sbml: steady state on published SBML, read as it is

Every member but one is a curated BioModels file, pinned at `sys-bio/temp-biomodels`
`6a09daf4` (the commit ode-sbml pins), whose curation note names a published
*steady-state* figure: a dose-response curve or a bifurcation diagram. Member 8 is a
PEtab benchmark problem whose measurements are all at steady state, pinned at
`Benchmarking-Initiative/Benchmark-Models-PEtab` `4d20850`. Members are ordered by SBML
species count (dynamic species in parentheses). The suite's design is
`suites-deterministic.md` §2.4 (decision D5).

| # | problem | source | species / reactions | laws | dose (points) | roots at the file's own parameters | evidence |
|---|---|---|---|---|---|---|---|
| 1 | `inflammation_rheumatoid_arthritis_baker2013` | BIOMD0000000550 | 2 (rate rules) / 0 | 0 | alpha2 (`parameter_2`), 1–25 linear (20) | 1 stable; bistable for ~7.5 < alpha2 < ~15.5 | `author_file`: Fig. 3a |
| 2 | `smallest_bistable_reaction_wilhelm2009` | BIOMD0000000233 | 4 (2) / 4 | 0 | the constant species S, 0.1–2 linear (20) | 2 stable + 1 saddle | `author_file`: Fig. 1 |
| 3 | `mutual_activation_switch_tyson2003` | BIOMD0000000311 | 4 (1) / 5 | 0 | the boundary signal S, 0–20 linear (20) | 2 stable + 1 saddle | `author_file`: Fig. 1e |
| 4 | `mapk_double_phosphorylation_ordered_markevich2004` | BIOMD0000000027 | 5 (3) / 4 | 1 | the constant kinase MAPKK, 5–100 nM linear (20) | 2 stable + 1 saddle | `author_file`: Fig. 3 |
| 5 | `tunable_reversible_receptor_switch_palani2011` | BIOMD0000000325 | 6 (5) / 10 | 0 | the ligand L, 1e-3–10 log (20) | 1 stable | `author_file`: Fig. 1B |
| 6 | `embryonic_stem_cell_switch_chickarmane2006` | BIOMD0000000204 | 12 (5) / 10 | 0 | the signal A, 0–100 linear (20) | 2 stable + 1 saddle | `author_file`: Fig. 10 |
| 7 | `mapk_double_phosphorylation_random_markevich2004` | BIOMD0000000028 | 16 / 17 | 3 | MEK, 300–400 nM linear (20), at ERK 1000 nM and MKP3 180 nM | 1 stable | `author_file`: Fig. 5 |
| 8 | `histone_h4_acetylation_motifs_blasi2016` | PEtab `Blasi_CellSystems2016` | 16 / 32 | 1 | none: one point, the PEtab nominal parameters | 1 stable | `author_file`: the PEtab problem, 252 measurements at t = inf |
| 9 | `aspartate_metabolism_curien2009` | BIOMD0000000212 | 31 (8) / 18 | 1 | V_max of the tRNA synthetases (`V_AA_RS`), 0.15–2 uM/s linear (20) | 1 stable | `author_file`: Fig. 7c |
| 10 | `carbohydrate_flux_oxidative_stress_ralser2007` | BIOMD0000000247 | 31 (24) / 25 | 2 | relative TPI activity (`k_rel_TPI`), 0.25–1 linear (20) | 1 stable | `author_file`: Fig. 5a |

Laws: conservation laws among the dynamic species, from the SBML's stoichiometry.
Roots: `steady_states(n_starts=200)` at the file's own parameters (the proposal's
probe). Each `author_file` row quotes its curation note in the spec's evidence; the
curation figure is the curators' re-simulation, not a numeric oracle, so the suite's
numeric correctness rests on the engines agreeing with each other, with long-time
integration and with the engine-free checks.

### Protocol and comparison

- Each point starts from the file's initial state at that point's doses (BNG
  `parameter_scan` with `reset_conc`), as on the `.net` path. A dose is an SBML global
  parameter or, for members 2–7, a species' initial concentration (`scan.species`).
  Member 7 also sets ERK and MKP3 at every point (`species`), to the totals the
  curation note gives for its MEK scan; the file's own values are 800 and 100 nM.
- `rtol` 1e-8, `atol` 1e-10 on every member; the horizon is 1e6, and 1e8 for Curien
  2009, whose threonine pool takes about 1e7 s to settle at the low end of its scan.
- The compared species are the SBML's state species, by id, as concentrations.
  run_network integrates each point's network derived from the SBML
  (`ode.derive_network`, its faithfulness check included) to the horizon; it applies to
  all ten members.
- The engine-free checks read the derived network for the residual, and the SBML's own
  stoichiometry for the conservation laws (the converter writes each reaction as
  signed per-species fluxes, so the derived network's stoichiometry carries no laws).
  Three refinements over the `.net` verdict, each found on these members: the
  residual's floor also scales with the flux at the point's initial state (Wilhelm
  2009's off state is the origin); species moved by a rate rule are left out of the
  residual (Baker 2013 has only those); a law passes within 10 atol absolute
  (Chickarmane 2006's genes are held at 0).
- **Continuation is not yet run.** The design asks for both branches of the bistable
  members (each point seeded from the last, or from the seeds the curation notes name:
  X(0) = 1 and 4 for Wilhelm 2009, L = 0.001 and 0.2 for Palani 2011); every point here
  starts from the file's initial state, so each dose-response follows the branch that
  state reaches.

### What the suite spans

| axis | members |
|---|---|
| size | 2 to 31 species (at most 24 dynamic): curated BioModels with a steady-state curation are small, and the larger PEtab pre-equilibration problems read the clock, which `steady_state()` refuses (proposal §2.4) |
| bistable at the file's parameters | 2, 3, 4, 6 |
| a switch crossed by the dose from the initial state | 1, 2, 3, 5, 6 |
| conservation laws (a singular full Jacobian) | 4, 7, 8, 9, 10 |
| rate rules only | 1 |
| species doses (constant or boundary species) | 2, 3, 4, 5, 6, 7 |
| assignment rules | 3, 8, 9, 10 |
| two compartments | 10 |
| steady-state data | 8 |

### Membership decisions

- **Criterion** (D5): curated BioModels whose curation reproduces a published
  steady-state curve or bifurcation diagram, that `steady_state()` accepts (no events,
  no clock), plus Blasi 2016 from PEtab for steady-state data; ordered by species count.
- **Blasi 2016** replaces BIOMD0000000210 (Chickarmane, Peterson 2008), the first
  alternate. It runs at the PEtab problem's nominal parameters, which the SBML holds
  rounded to 3–4 digits.
- **Dose grids** follow the curation notes' ranges where they give one (1, 2, 4, 7, 9),
  at 20 points. Where they do not (3, 5, 6, 10), the grid was chosen to cross the
  curated figure's response from the file's initial state: Ralser 2007 stops at
  `k_rel_TPI` = 0.25 because at 0.1 BNGsim finds no steady state within 1e6 (DHAP is
  still growing, past 4e6), while 0.2 converges.

## ssa-net: exact SSA on the `.net` generated from published BNGL

Members are ordered by network size, as in `ode-net`. Each correctness job compares BNGsim's
exact SSA ensemble with run_network's, libRoadRunner's and COPASI's (and with the network's
exact moments where `moments: linear` is declared), by the permutation max-T test of DESIGN
§5.3.

| # | problem | species / reactions | copy numbers | protocol (t_end; ensemble) | stochastic behaviour | evidence | how it entered |
|---|---|---|---|---|---|---|---|
| 1 | `bursty_autoregulated_gene_expression_lin2016` | 2 / 4 | mRNA 0–5, protein up to ~1100 | 3600 s; 10^4 | bursty autoregulation; Hill law | `data` (Fig. 3b) | paper seed (B07) |
| 2 | `demographic_noise_predator_prey_cycles_mckane2005` | 3 / 5 | N = 3200 individuals | 1200; 10^3 | noise-sustained quasi-cycles | `theory` (LNA spectrum) | paper seed (B08) |
| 3 | `three_stage_stochastic_gene_expression_shahrezaei2008` | 4 / 6 | DNA 1, mRNA 0–4, protein 0–155 | 2e4 s; 10^4, exact moments | promoter switching, bursts | `theory` (CME) | paper seed (B09) |
| 4 | `noise_induced_bistable_futile_cycle_samoilov2005_unordered_pair` | 7 / 10 | E 0–20, N 0–10 beside X ~1.8e3 | 10 s; 200 | noise-induced bistability; a repeated reactant | `data` (Fig. 3A) | replaces the seed's primary (B10) |
| 5 | `noise_resistant_circadian_oscillator_vilar2002` | 9 / 16 | genes 1, mRNA < 1 on average, proteins ~2e3 | 400 h; 10^3 | limit cycle under noise | `equations` + `data` | top-up; BNGL twin of `ssa-sbml`'s BIOMD0000000035 |
| 6 | `two_state_gene_expression_noise_munsky2012` | 10 / 14 | promoters 1, mRNA 0–134 | 6000 s; 10^4 | bimodal, long-tailed and Poisson transcripts; `if()` laws | `theory` (FSP, Eq. 2) | paper seed (B11) |
| 7 | `tcr_signaling_bistability_lipniacki2008` | 37 / 97 | pMHC 30 … ERK 3e5 | 600 s; 100 | bistable switch | `data` (Fig. 7B, ODE) | paper seed (B13) |
| 8 | `prion_nucleated_polymerization_rubenstein2007` | 41 / 483 | 3 seed trimers beside 600 PrPC | 365 d; 150 | nucleation or extinction | `data` (Fig. 2, SSA) | replaces the seed's `_benchmark` (B14) |
| 9 | `membrane_anchored_multisite_phosphorylation_dushek2011` | 143 / 284 | one substrate molecule | 1000 s; 2000 | Bernoulli occupancies; 20 identical sites | `data` + `theory` | top-up |
| 10 | `early_fceri_signaling_faeder2003` | 354 / 3680 | Lyn 28 … Lig 6000 | 120 s; 100 | early response, thousands of reactions | `data` (Fig. 3) | top-up, in place of B12 |

### Protocol and comparison

- A protocol is the record's own stochastic action (McKane, Samoilov, Vilar, Lipniacki,
  Rubenstein, Faeder), run as an ensemble from the record's seed state rather than as its
  one seeded trajectory, or, where the record samples a stationary distribution from one
  long trajectory or has no SSA action, an ensemble relaxing from the seed state over
  a stated window (Lin, Shahrezaei, Munsky, Dushek). Each spec says which and why.
- Horizons and ensemble sizes are set so that each engine's ensemble fits the harness's
  one-hour budget per engine, from a one-trajectory probe of every engine on a Darwin node.
  Two members are cut short. Lipniacki 2008's SSA action runs 3 h, at 70 s per BNGsim
  trajectory and 133 s per run_network one, so its protocol is the first 10 min, in which
  the deterministic switch peaks (libRoadRunner still takes about 16 s a trajectory there).
  Faeder 2003 runs its file's condition for 2 min rather than 10: COPASI's direct method
  takes about 22 s a trajectory on its 3680 reactions whatever the horizon (the task's
  initialisation, repeated for each trajectory's seed), and libRoadRunner about 25 s per
  10 min.
- libRoadRunner is UNSUPPORTED on member 4 (a repeated reactant, DESIGN §5.3).
- Exact moments are declared only for member 3. Member 9's network is first order in one
  molecule, so its moments are known too, but 143 species is past the linear-moment
  solver's limit of 40.

### What the suite spans (DESIGN §3.2, SSA)

| axis | members |
|---|---|
| copy numbers | one molecule (9); single-copy genes or promoters (1, 3, 5, 6); up to ~3e5 (7) |
| size | 2–354 species, 4–3680 reactions |
| known moments | in the job: exact moments (3); in the records' checks: LNA spectrum (2), CME (3), FSP (6), the Markov chain's null space (9) |
| dynamics | stationary distributions (1, 3, 6); limit cycle (5); quasi-cycles (2); noise-induced bistability (4); bistable switch (7); nucleation and extinction (8); relaxation (9, 10) |
| rate laws | mass action; a repeated reactant (4); functional: Hill (1), `if()` (6) |

### Membership decisions

- **Seed:** the paper's BNGL exact-SSA set B07–B14, models simulated with exact SSA in the
  source study. Five are kept as they are.
- **Samoilov 2005, `_unordered_pair` for the primary:** the primary's driver `N + N` fires
  at twice the convention under which `_unordered_pair` reproduces the paper's Fig. 3A
  (one mode where the paper has two); BNGL-Models#72 asserts that check.
- **Kochańczyk 2017 out:** its record declares no exact-SSA action, and exact SSA is about
  6e8 events per hour of its protocol. It stays in `psa-net`.
- **Rubenstein 2007, the primary for `_benchmark`:** `_benchmark` has no stochastic check;
  the primary's Fig. 2 check is of SSA. `_benchmark` moves to `psa-net`.
- **Top-ups:** Vilar 2002 (asserted in BNGL-Models#73), Dushek 2011 (#71) and Faeder 2003
  (Fig. 3 data check, #70) in place of Kochańczyk.

## psa-net: partial scaling (PSA) on the `.net`

Every correctness job draws BNGsim's and run_network's ensembles at exact SSA and at N_c =
10, 30, 100, 300 and 1000, and gates like for like and exactness where nothing scaled
(DESIGN §5.4); PSA's error against exact SSA is reported per N_c.

| # | problem | species / reactions | populations | protocol (t_end; ensemble) | run_network PSA | how it entered |
|---|---|---|---|---|---|---|
| 1 | `bursty_autoregulated_gene_expression_lin2016` | 2 / 4 | mRNA 0–5 beside protein ~1100 | 3.6e4 s; 10^4 | aborts at t = 0 (zeroth-order reaction) | top-up |
| 2 | `noise_induced_bistable_futile_cycle_samoilov2005_unordered_pair` | 7 / 10 | E, N ≤ 20 beside X ~1.8e3 | 10 s; 100 | runs | top-up |
| 3 | `noise_resistant_circadian_oscillator_vilar2002` | 9 / 16 | genes 1 beside proteins ~2e3 | 400 h; 500 | runs | top-up |
| 4 | `mapk_relaxation_oscillations_kochanczyk2017` | 34 / 65 | 6e3–3e6 | 600 s; 200, exact 40 | runs | paper seed (B12) |
| 5 | `tcr_signaling_bistability_lipniacki2008` | 37 / 97 | 10–3e5 | 600 s; 200, exact 100 | runs | paper seed (B13) |
| 6 | `car_cd3zeta_phosphorylation_rohrs2018` | 65 / 192 | LCK 20 beside CD3ζ 2e4 | 60 min; 1000 | segfaults or hangs (exact SSA too) | top-up |
| 7 | `scaffolded_mapk_cascade_kocieniewski2012` | 85 / 487 | 1e5–5e5 | 50 s; 100 | runs | top-up |
| 8 | `prion_nucleated_polymerization_rubenstein2007_benchmark` | 121 / 3843 | 3 seed trimers beside PrPC 3.8e3 | 10 d; 200 | segfaults at t = 0 | paper seed (B14) |
| 9 | `early_fceri_signaling_faeder2003` | 354 / 3680 | Lyn 28 … Lig 6000 | 120 s; 100 | runs | top-up |
| 10 | `combinatorial_egfr_signaling_blinov2006` | 356 / 3749 | 1.3e4–1.2e6 | 10 s; 100 | runs | top-up |

### Protocol and comparison

- *exact N* (`exact_replicates`) is a smaller ensemble for the two exact-SSA ensembles
  alone, where an exact trajectory costs tens of seconds and a scaled one seconds
  (Kochańczyk: 49 s exact, 1.5–5.5 s scaled, per 10 min).
- Kochańczyk, Kocieniewski and Blinov run the start of their own protocol (the record's PSA
  hour, the ODE protocol's 500 s and 120 s) so that twelve ensembles fit one job: BNGsim's
  PSA runs interpreted propensities and is no faster than exact SSA on Kocieniewski and
  Blinov at the larger N_c. The specs give the probe's numbers.
- **run_network's PSA segfaults on networks with null-species reactions** (Munsky 2012, both
  Rubenstein 2007 files, Rohrs 2018 in the probe), beside its documented abort on a
  zeroth-order reaction (Lin 2016). These are recorded as REFERENCE_FAILED, never worked
  around; BNGsim's PSA is then gated against exact SSA only where nothing scaled.

### What the suite spans (DESIGN §3.2, PSA)

| axis | members |
|---|---|
| populations spanning orders of magnitude | every member; 2–3 orders (1–3, 6, 8, 9), 3 or more (4, 5, 10) |
| low copies beside scaled ones | single genes (3), low-copy enzymes or drivers (2, 6, 9), seed aggregates (8), mRNA (1) |
| uniformly large populations | 7, 10 |
| size | 2–356 species, 4–3843 reactions |
| the three published models of Lin et al. 2019 | 4, 5, 8 |

### Membership decisions

- **Seed:** B12–B14, the three published models of Lin, Feng & Hlavacek (2019), kept whole.
- **Top-ups**, by Lin et al.'s own rule (populations large enough to make exact SSA slow),
  with populations spanning two to three orders and exact SSA feasible as the N_c = ∞
  reference: Lin 2016, Samoilov `_unordered_pair`, Vilar 2002, Rohrs 2018, Kocieniewski 2012,
  Faeder 2003 and Blinov 2006.

## sens-net: forward sensitivities on the `.net` generated from published BNGL

BNGsim's CVODES forward sensitivities `dx/dp` against AMICI's, central finite
differences from `run_network` and COPASI's (DESIGN §5.6). Every new member is pinned at
`wshlavacek/BNGL-Models` `7d6d366`, where the upstream checks of #60–#66 are merged;
members 6, 9 and 10 keep their `ode-net` pins (the same files). Each protocol is the
member's published experiment: one measured phase from the network's seeds, or an
ordered list of finite phases (`phases`: a duration, the parameters each sets, the
species each resets; D3b). Members are ordered by N x N_p as the proposal counted it
(`suites-deterministic.md` §2.1); with the parameter counts the harness takes from the
generated networks, 4 and 5 swap (1800 against 1760).

| # | problem | species / reactions | N_p (convention) | protocol | evidence | how it entered |
|---|---|---|---|---|---|---|
| 1 | `nfkb_reduced_onoff_jaruszewiczblonska2023` | 7 / 15 | 13 (fitted) | one phase, 0–43200 s, 144 steps; TNF on for 0–7200 s through a clock-gated rate | `author_file`: the authors' S1 Code file, 7.7e-7 of peak | paper seed (B22) |
| 2 | `igf1r_ligand_binding_erickson2019` | 27 / 96 | 11 (independent; the fitted 7 a setting) | one phase, 0–14400 s, 48 | `author_file`: the authors' `IGF1R_fit.bngl` at Table 1, 1.2e-7 of peak | paper seed (B21), the curated record's file |
| 3 | `atf2_phosphoswitch_kirsch2020` | 48 / 152 | 21 (independent; the fitted 4 a setting) | 20000 s at Stim = 0, then Stim = 1 over 0–2466 s, 18 | `author_file`: the authors' Supplementary Software 1, difference 0 | paper seed (B23), at its stimulated protocol |
| 4 | `multisite_egfr_phosphorylation_salazarcavazos2020` | 75 / 618 | 24 (independent; the fitted 6 a setting) | one phase, 25 nM EGF, 0–300 s, 300 | `data`: the PEtab collection's published fit, 1/2 chi^2 within 4.2e-6 | paper seed (B27), the curated record's file |
| 5 | `cstar_skmel133_melanoma_rukhlenko2022` | 44 / 56 | 40 (fitted, less 3 derived) | 86400 s without inhibitors, then the ERK inhibitor at dose 1 over 0–86400 s, 864 | `author_file`: the authors' `SKMEL-133-3.bngl`, 15 of 16 phases bit-identical, 1 at 2.7e-14 | paper seed (B41), at its published protocol |
| 6 | `shp2_regulation_and_function_barua2007` | 149 / 1032 | 22 (independent) | one phase, 0–1000 s, 200 | `data` + `structure` (`ode-net` 5) | paper seed (B03) |
| 7 | `cstar_trka_ngf_rukhlenko2022` | 82 / 405 | 88 (independent; the fitted 8 a setting) | 50000 s, S reset to 2*S0, 50000 s, then Lig_on = 1 over 0–2700 s, 270 | `author_file`: the authors' `TrkA_S_model.bngl`, all 9 phases bit-identical | paper seed (B39), at its published protocol |
| 8 | `cstar_trkb_bdnf_rukhlenko2022` | 114 / 645 | 91 (independent; the fitted 8 a setting) | as 7 | `author_file`: the authors' `TrkB_S_model.bngl`, all 9 phases bit-identical | paper seed (B40), at its published protocol |
| 9 | `cell_cycle_oscillator_lang2024` (`_fitted`) | 73 / 332 | 158 (fitted: 114 rate constants and 47 initial values, less 3 at zero) | one phase, 0–136471.6 s, 300 | `author_file` + `equations` (`ode-net` 3) | paper seed (B38), `_fitted` in place of `v3_2_0` |
| 10 | `combinatorial_egfr_signaling_blinov2006` | 356 / 3749 | 43 (independent) | one phase, 0–120 s, 120 | `data` (`ode-net` 6) | paper seed (B04) |

N_p is the number of parameters differentiated, as the harness selects them from the
generated `.net` (`sens.select_parameters`): `independent` is every parameter written as a
number that reaches a rate or an initial amount, less one the protocol sets, one that sizes
a compartment, and one whose value is zero without a `parameter_scales` entry; `fitted` is
the problem's fitted set. Where the fitted set is below 10, every independent parameter is
differentiated and the fitted set is a job setting of its own (D3c).

### What the suite spans (DESIGN §3.2, sensitivities)

| axis | members |
|---|---|
| size | 7 to 356 species; N_p 11 to 158; N x N_p 91 to 15.3k |
| multi-phase protocol, sensitivities carried across | 3 and 5 (a parameter changed between phases), 7 and 8 (and a species reset: S := 2*S0) |
| derived-parameter chains to rates and initial amounts | 2, 4; initial values differentiated in 2, 3, 4, 6, 7, 8, 9, 10 |
| rates that switch during a run | 1 (TNF on, then off, through a clock species) |
| piecewise `if()` rates of the state | 5, 7, 8 (the phenotype coordinate's force) |
| oscillator | 9 |
| N_p above 50 | 7, 8, 9 |

### Membership decisions

- **Seed:** the BNGsim paper's sensitivity table, eleven rows: eight PyBNF-fitted BNGL
  problems (B21 igf1r, B22 reduced_onoff, B23 p38atf2_binding, B27 egfr_simpull, B38
  v3_2_0, B39 cstar_trka, B40 cstar_trkb, B41 cstar_skmel133_bmra_exact) and three large
  networks differentiated over every independent parameter (B03 Barua 2007, B04 Blinov
  2006, B06 Faeder 2003 `_fyn`). Proposal §2.1, decisions D3, D3b, D3c (§4.2).
- **B06 Faeder 2003 `_fyn` dropped:** not published (its own header says so), as in
  `ode-net`. That leaves exactly ten, so there is no top-up (§4.1).
- **B38 `v3_2_0` replaced by `cell_cycle_oscillator_lang2024_fitted`:** `v3_2_0` is the
  `_fused` network at the pre-fit nominal point, where the paper's row ran; `_fitted`
  (`ode-net` 3) is the same network at the authors' optimum (§4.1).
- **B21: the curated record's file in place of the job file:** the job ships the
  Kiselyov et al. 2009 pre-fit point; the record is at Erickson et al.'s Table 1 (§4.1).
- **B22, B21, B23, B27, B39, B40 and B41 verified upstream** (D3): one BNGL-Models PR
  each, a pinned author file or published fit and a BNG2.pl check with an asserted
  tolerance (#63, #64, #65, #66, #60, #61, #62). Every one passed, so no fallback of §2.1
  was needed. B23's data comparison is reported, not asserted (the Source Data's error
  bars admit no derived tolerance); its evidence is the authors' own file, which the job
  file reproduces exactly.
- **The published protocols, not unstimulated relaxations:** the paper's rows ran B23,
  B39, B40 and B41 from the seed state without their stimulus. Here B23 equilibrates and
  then stimulates; B41 equilibrates for the authors' 86400 s and then adds the ERK
  inhibitor (the job's first experiment and the authors' first 86400 s arm; the job file
  routes that inhibitor through the parameter `I_ERK_conc`, so the change is a parameter
  set, and the inhibitor species it seeds takes part in no reaction); B39 and B40 run the
  authors' 50000 s, S := 2*S0, 50000 s, then ligand. Every equilibration is finite:
  p38-ATF2 has an absorbing state, and from the seeds the cSTAR Trk networks' steady state
  lies in another basin of S than the authors' protocol reaches (upstream check).
- **Parameters:** B41's three force coefficients `beta_SRC`, `beta_mTOR`, `beta_PKC` are
  fitted but written as derived parameters (`x / phen_scale`), so they cannot be
  differentiated as parameters: 40 of its 43. Lang's fitted set has three rate constants
  at zero at the optimum, which have no relative step: 158 of 161. Salazar-Cavazos counts
  24 independent parameters, the proposal 23.
- **Grids and tolerances:** the measured grid is coarser than the authors' output grid
  for 7 and 8 (10 s, not 1 s) and than the `ode-net` protocol for 9 and 10 (every fourth
  point; 1 s, not 0.2 s), so that one engine's sensitivities stay at tens of MB. The
  cSTAR members and Lang run at rtol 1e-10 / atol 1e-12 and their finite differences at
  1e-12 / 1e-14: BioNetGen's defaults err up to 1.45e-2 of a species' peak on TrkB
  (upstream check).
- **Expected references that do not apply:** COPASI's time-course sensitivities cannot
  start from an earlier phase's, so it is `not_applicable` on 3, 5, 7 and 8. BNGsim 0.17.2
  refuses to carry sensitivities across a species reset (7, 8), which is BNGsim's result
  to record.

## sens-sbml: forward sensitivities on SBML, from PEtab benchmark problems

BNGsim's forward sensitivities on the SBML of ten PEtab benchmark problems, each applied
as PEtab defines it, against AMICI's own PEtab importer and central finite differences
from libRoadRunner (`bench.capabilities.sens_sbml`). Every member is pinned at
`Benchmarking-Initiative/Benchmark-Models-PEtab` `4d20850` (the commit BNGL-Models' Grein
job and AMICI v1.1.0's CI resolved): the SBML is the spec's `source`, and its `petab:` block
pins the YAML and the four tables by sha256 (D12). Members are ordered by N x N_p, N the
SBML species and N_p the sensitivity set (`suites-deterministic.md` §2.2, decision D4).

| # | problem | PEtab id | species / reactions | conditions (measurements) | estimated / N_p | what the problem adds | AMICI v1.1.0 CI |
|---|---|---|---|---|---|---|---|
| 1 | `sphingolipid_rheostat_apoptosis_armistead2024` | `Armistead_CellDeathDis2024` | 4 / 8 | 2 (58), each setting a switch and four initial values | 14 / 10 | initial values set per condition | G |
| 2 | `stat5_dimerization_isoforms_boehm2014` | `Boehm_JProteomeRes2014` | 8 / 9 | 1 (48) | 9 / 6 | two compartments of unequal volume; initial assignments | G + L |
| 3 | `ip3_receptor_type2_dynamics_sneyd2002` | `Sneyd_PNAS2002` | 6 / 10 | 9 (135), IP3 and Ca doses | 15 / 14 | a dose-response read as nine conditions | G + L |
| 4 | `akt_pathway_low_pass_filter_fujita2010` | `Fujita_SciSignal2010` | 9 / 11 | 6 (144), EGF steps and ramps | 19 / 16 | a time-dependent input set per condition; scaling parameters | G + L |
| 5 | `il13_signaling_lymphoma_raia2011` | `Raia_CancerResearch2011` | 14 / 16 | 4 (205), IL13 doses | 39 / 18 | two compartments (10 / 100); a noise model that reads the state | G |
| 6 | `covid19_italy_interventions_giordano2020` | `Giordano_Nature2020` | 13 / 13 | 1 (313) | 50 / 43 | 55 `piecewise` terms in time (rates switched at intervention dates) | G |
| 7 | `histone_h3_k27_k36_methylation_zheng2012` | `Zheng_PNAS2012` | 15 / 60 | 1 after a pre-equilibration (60) | 46 / 45 | **pre-equilibration**: sensitivities carried from a steady state | G + L |
| 8 | `jak2_stat5_dual_feedback_bachmann2011` | `Bachmann_MSB2011` | 25 / 36 | 36 (541) | 113 / 27 | initial values set per condition through initial assignments; log10 observables | L |
| 9 | `insulin_signalling_oxidative_stress_smith2013_petab` | `Smith_BMCSystBiol2013` | 133 / 367 | 35 (62), species and parameters set | 25 / 16 | **3 events** (insulin pulses); 5 compartments of 6.4e-14 to 1.65e-11; amount-only species | L (AMICI's own regression value) |
| 10 | `smad_complex_gene_expression_lucarelli2018` | `Lucarelli_CellSystems2018` | 33 / 57 | 16 (1755) | 84 / 69 | three estimated totals reach initial values through the condition table; 65 log10 observables | L |

Evidence: every member is `author_file` (the collection holds the authors' model, data and
fitted nominal parameters). AMICI v1.1.0's benchmark test checks each at that commit: G, the
likelihood gradient against finite differences at nominal plus 5% noise; L, the
log-likelihood at nominal against a stored value, which the specs copy as `llh_reference`.

### Protocol and comparison

- **The PEtab problem is applied, not just its SBML** (the paper's rows read none of the
  tables). For each simulation condition of the measurement table: the SBML's values,
  overridden by every SBML parameter's nominal value in the parameter table, overridden
  by the condition table (a number, or a parameter-table id standing for its nominal
  value); initial assignments evaluated at those values; the condition's species
  initial values; the condition's measurement times from t = 0. A pre-equilibration
  runs the pre-equilibration condition to steady state on the same model, then the
  simulation condition's parameters take effect and the state, with its
  sensitivities, carries over. The harness reads the tables itself (`bench.petab_problem`,
  no `petab` library); AMICI reads them through `petab` 0.10.0, an independent application.
- **Sensitivity set:** the estimated parameters that are the model's own SBML parameters,
  in parameter-table order: not a noise, scaling or offset parameter (those never reach the
  state), not a condition-table column (the experiment's). The spec writes the set down; the
  harness and a unit test re-derive it from the pinned tables and refuse a spec that differs.
- **Engines:** BNGsim (`Model.from_sbml`, both CVODES correctors), its pre-equilibration by
  its own recipe (`run(steady_state=True)`, then `carry_sensitivities=True`); AMICI's PEtab
  importer (`amici.importers.petab.v1`, `simulate_petab` with forward sensitivities, its
  steady-state pre-equilibration; sensitivities on the PEtab scale put back on a linear
  one); libRoadRunner central differences at steps 1e-3 and 5e-4 of each parameter, at
  rtol 1e-10 and atol 1e-12. COPASI is not wired for PEtab conditions (recorded).
- **Outputs:** the state species, by SBML semantics (an amount for a species with
  `hasOnlySubstanceUnits`, a concentration otherwise; BNGsim reports concentrations, which
  the harness multiplies by the constant compartment size for Smith's amount-only species),
  and every measurement row's observable, at each condition's measurement times, the
  conditions end to end. The verdict is `sens-net`'s column error of the semi-scaled
  sensitivity (tol 1e-3, with its floors, refutation and coverage rules). An observable is
  formed from the states by the harness for BNGsim and the finite differences, and by AMICI
  itself for AMICI.
- **Problem-level checks**, part of the outcome: BNGsim's PEtab log-likelihood at nominal
  against AMICI's (atol 1e-3 + rtol 1e-6; AMICI 1.1.0's cost functions keep the Jacobian term
  of a log-transformed observable, as PEtab's do) and against AMICI's stored value; its
  gradient over the sensitivity set against AMICI's forward gradient, within 1e-3 of the
  size of the rows' contributions (at the fitted point the gradient is a cancelled sum).
- `rtol` 1e-8 and `atol` 1e-10 on every member; a pre-equilibration may integrate to 1e6.
- **Timing** interleaves one evaluation of the whole problem per engine: both BNGsim
  correctors, AMICI's forward sensitivities and its adjoint gradient of the PEtab
  log-likelihood (D7, timing only), and the plain solves of BNGsim and libRoadRunner that a
  finite-difference gradient pays N_p + 1 or 2 N_p times.

### What the suite spans (DESIGN §3.2, sensitivities)

| axis | members |
|---|---|
| size | 4 to 133 species; N_p 6 to 69; N x N_p 40 to 2277 |
| conditions | 1 to 36 simulation conditions; 48 to 1755 measurements |
| pre-equilibration, sensitivities carried from a steady state | 7 |
| events | 9 |
| the clock read by an assignment rule (a time-dependent input, `piecewise` rates) | 4, 6; 9's events trigger on time |
| compartments of unequal size | 2, 5, 9 |
| initial values set per condition (directly or through initial assignments) | 1, 8, 9, 10 |
| log-transformed observables | 8, 10 |
| observable and noise parameters (scalings, offsets, a noise model of the state) | 4, 5, 8, 9, 10 |

### Membership decisions

- **Seed:** the paper's six PEtab rows. Kept: Armistead (P04), Boehm (P01), Sneyd (P02), Raia
  (P05), each now at its PEtab nominal parameters and every condition (the paper ran
  Armistead at the SBML's 1.0 placeholders, Sneyd at IP3 = 0, Raia unstimulated). Replaced:
  Perelson 1996 (P03; N_p = 2, a nominal 210 log-likelihood units from the optimum) and
  Crauste 2017 (P06; AMICI's own test excludes its gradient for numerical failures; the
  paper's manifest named Bruno 2016, whose per-condition estimated initial values BNGsim's
  sensitivity path cannot route). Boehm stays below the 10-parameter floor (D4).
- **Top-ups** by the criterion of proposal §2.2: at least 10 estimated parameters, N_p <= 200,
  validated by AMICI v1.1.0's CI at the pinned commit, the largest N x N_p: Giordano, Zheng,
  Bachmann, Isensee, Smith, Lucarelli.
- **Isensee 2018 replaced by Fujita 2010** (D4: record the refusal once, then replace). BNGsim
  does not reach a steady state in Isensee's pre-equilibration: the model reads the clock, and
  BNGsim never marks such a model steady (bngsim issue #710), so its correctness job is
  UNSUPPORTED in BNGsim's words. Its spec stays (`pka_rii_phosphorylation_camp_isensee2018`,
  no suite) so the refusal is reproducible. Fujita 2010 is the first alternate BNGsim can
  pre-equilibrate (Weber 2015 reads the clock too); it has no pre-equilibration, and takes
  its place by N x N_p (4).
- **Smith 2013** is the PEtab problem's file, not `ode-sbml`'s BIOMD0000000474: the problem's
  parameter table sets 54 of the model's parameters and its conditions three species.
