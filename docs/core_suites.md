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

## nf-xml-aggregation: network-free on BNG-XML, multivalent aggregation

Network-free simulation of the BNG-XML that BNG2.pl's `writeXML()` writes from published
BNGL, on the multivalent-aggregation models of the bngsim paper's network-free set (its
B15–B20, B31 and B35, kept whole) plus two top-ups: the strict finite-network anchor and
polymerization with turnover (suites-stochastic.md §2.4). Every job runs one of BNGsim's
two network-free engines (`nf_reject`, `nf_exact`, two settings) against standalone NFsim
driven by BNG2.pl, standalone RuleMonkey (same lineage as `nf_exact`, so it cannot pass or
refute BNGsim alone; DESIGN §5.5) and, on a finite or negligibly truncated network, exact
SSA by run_network and by BNGsim. Members are ordered by particle count as the proposal
ranked them (at the largest dose of a scan); see the protocol notes for the one dose each
scan member runs.

| # | problem | particles in the protocol | network | protocol (one network-free run) | replicates | evidence | how it entered |
|---|---|---|---|---|---:|---|---|
| 1 | `blbr_heterogeneity_goldstein1980` | 50 antibodies (excess-ligand mode) | unbounded chains | default dose 1e-8 M, 0–10 s, 50 steps, -bscb | 1000 | `theory`: Goldstein–Wofsy Eqs. 13–14 (Fig. 5) | paper seed (B18) |
| 2 | `prion_nucleated_polymerization_rubenstein2007` | 600 PrPC + 3 seed trimers → ~3.7e4 PrPSc | unbounded chains; SSA on the 40-subunit truncation (41 / 483) | 0–365 days, 73 steps | 150 | `data`: Fig. 2, by SSA (shared with ssa-net 8) | top-up: polymerization with turnover |
| 3 | `tnf_tnfr1_sequential_binding_mcmillan2021` | 3,613 | **finite, 9 / 24** | 0–10,000 s, 100 steps | 1000 | `data`: Fig. 1b Source Data, by ODE | top-up: the strict finite anchor |
| 4 | `tlbr_solution_macken1982` | 4,500 | unbounded, acyclic | own: 0–500 s, 250 steps, -bscb | 1000 | `theory`: Eq. 11, Eq. 21, Theorem 1 | paper seed (B16) |
| 5 | `tlbr_yang2008` | 4,500 | unbounded | own: 0–3000 s, 300 steps | 1000 | `theory`: mean-field bond ODE | paper seed (B15) |
| 6 | `blbr_dembo1978` | 33,000 | unbounded chains | own: 0–3000 s, 300 steps, -bscb | 500 | `theory`: Dembo–Goldstein equilibrium | paper seed (B17) |
| 7 | `blbr_rings_posner1995` | 33,000 | unbounded, with rings | own: 0–3000 s, 300 steps, -bscb -utl 5 | 250 | `theory`: Posner Eqs. 14a–i | paper seed (B19) |
| 8 | `tlbr_steric_monine2010` | 4,515 at 7 nM (scan to ~1.3e5) | unbounded, acyclic | the record's single-dose run: 7 nM, 0–5000 s, 250 steps, -bscb | 500 | `data`: Fig. 2a | paper seed (B35) |
| 9 | `blbr_cooperativity_posner2004` | 2,707 at 1 nM (scan to ~1.8e5) | unbounded | default 1 nM, 0–10 s, 50 steps | 1000 | `theory`: Wofsy–Goldstein Eqs. 10, 15, 17 | paper seed (B20) |
| 10 | `egfr_oligomerization_mitra2019` | ~3.0e5 EGF + 900 receptors (30 nM) | unbounded, with rings | the file's 30 nM time course, cut to 0–150 s, 15 steps, -bscb -utl 5 | 120 | `data`: Kozer 2013 | paper seed (B31) |

### Protocol and comparison

- **One network-free run per member.** Every engine replays the protocol, so it can hold
  only network-free `simulate` steps and `setParameter` (bench.nfactions). The action
  blocks of Goldstein, Monine and Posner 2004 are a `parameter_scan` (so is Mitra's first
  protocol), which no engine but BNG2.pl can replay. Each of those runs one dose: the
  file's default (Goldstein 1e-8 M, near the peak of Fig. 5's bell; Posner 2004 1 nM),
  the record's own single-dose alternative (Monine 7 nM, commented out in its action
  block), or the record's time course (Mitra). A scan is a series of independent runs, so
  one dose loses no engine path; the dose curve stays the record's own check.
- **Mitra 2019 runs the first 150 s of its 600 s time course.** One engine spends about a
  minute on the whole course (2.1e6 events at a steady ~3.5e3 per simulated second; RuleMonkey
  55 s and `nf_exact` 60 s per trajectory on a general node), so 100 trajectories would
  outrun the per-engine budget below, and a lower dose does not help (1 nM costs as much as
  30 nM: the cost is in the receptors' clustering and phosphorylation, not the free EGF). The
  window is cut, the same for every engine, rather than the model; clustering is still
  evolving at 150 s.
- **NFsim flags.** `-bscb` and `-cb` are the same in NFsim 1.14.3 (one seed, one
  trajectory under either; measured on Yang 2008 and Posner 1995), and BNG2.pl passes
  `-cb` by default, so every member blocks same-complex binding in every engine, whether
  or not its file writes `-bscb`. Posner 1995 and Mitra pass `-utl 5` (RuleWorld/nfsim#52).
- **Ensemble sizes** fit each engine's ensemble inside the harness's per-engine budget
  of one hour (`nf.ENGINE_BUDGET_CAP_SECONDS`): 1000 trajectories where one costs under
  about 2 s in the slowest engine, fewer where it costs more (measured per engine on a
  general node, 2026-11-24): Dembo 1978 and Monine 2010 about 5 s, Posner 1995 9 s,
  Rubenstein 2007 15 s and Mitra 2019 (150 s) 15 s.
- **NFsim's final state is not read back.** BNG2.pl reads NFsim's species file back after
  a run with `get_final_state=>1` (its default); on Dembo 1978's chains that took BNG2.pl
  over three minutes per trajectory against NFsim's 3 s. The harness runs the protocol's
  last step with `get_final_state=>0`, since no step follows it; the trajectory is the
  same (DESIGN §5.5).
- **Rubenstein 2007** is the one member whose network is truncated: exact SSA runs on the
  network `generate_network` writes at `max_stoich` 40, the file's own. The chain-length
  tail is negligible: in a BioNetGen SSA run to 365 days no chain ever exceeded 27
  subunits, and at day 365 2e-4 of the aggregates held 20 or more. Its NF route has no
  action of its own (the file runs SSA only), so the protocol is its SSA run as a
  network-free one, output every five days.

### Membership decisions

- The seed is the paper's aggregation group whole (Q11, Q13); Yang 2008 is kept beside
  its near-duplicate Macken 1982 as the rejection-method paper behind `nf_reject`.
- TNF 2021 was the network-free capability's pilot; it entered unchanged as the strict
  finite anchor; both of its pilot jobs passed, correctness and timing.
- Each member's network-free check of the exact file, not run through BNGsim, is the
  upstream work of suites-stochastic.md §3.2 items 1–3, one BNGL-Models branch per
  record. A spec names its branch under `verification.pending` until it is merged and
  pinned. No branch changes a `.bngl`, so every member's source pin stays 7d6d366.

### Evidence (BNG2.pl NFsim 1.14.3 or exact SSA, never BNGsim)

| # | BNGL-Models branch @ commit | what is asserted | result |
|---|---|---|---|
| 1 | `verify-blbr_heterogeneity_goldstein1980-asserts` @ 168a072 | 18 doses x 64 seeds against Eqs. 13–14, 4.32 SE + 0.01 | worst 0.41 of the tolerance |
| 2 | — (ssa-net's check) | SSA against Fig. 2's saturated levels, < 5% of full scale | 3.91% |
| 3 | — (pilot) | ODE against Fig. 1b Source Data | as the pilot |
| 4 | `verify-tlbr_solution_macken1982-asserts` @ 0c3b431 | 3 doses x 200 seeds against Eq. 11 kinetics and Eq. 21 | max \|z\| 3.76 of 5.33; Eq. 21 within 2.0% |
| 5 | `verify-tlbr_yang2008-seeds-bscb` @ d4cceef | 200 seeds against the mean-field bond ODE; Fig. 3 caption's rejected fractions | bonds 293.13 ± 0.17 vs 293.21 |
| 6 | `verify-blbr_dembo1978-execute-asserts` @ 05f6bb4 | 3 doses x 20 seeds against the exact chain theory and Perelson–DeLisi kinetics | max \|z\| 1.66 and 2.16 |
| 7 | `verify-blbr_rings_posner1995-asserts` @ 78d7f9a | a measured ring identity, equilibrium and kinetics against Eqs. 14a–i, 60 seeds | max \|z\| 1.20, 1.58, 2.37 |
| 8 | `verify-tlbr_steric_monine2010-replicates` @ 936660b | 24 fresh scans against Fig. 2a, RMS λ < 0.02 | 0.0153 ± 0.0005 |
| 9 | `verify-blbr_cooperativity_posner2004-seeds-bscb` @ 2cd5a16 | 120 seeds x 20 doses against Wofsy–Goldstein | max \|q − theory\| 0.0016 |
| 10 | `verify-egfr_oligomerization_mitra2019-replicates` @ 5b8ebb1 | Kozer 2013 data, χ² < 30 | 13.90 ± 0.50 |
