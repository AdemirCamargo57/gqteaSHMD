# The gqteashmd project website: design

**Status:** design approved section by section by the owner on 2026-11-26; this spec awaits the owner's review.
**Branch:** `feat/project-website` (from `main` `280a612`).

## 0. Owner decisions (2026-11-26, multiple choice)

1. **Model:** the page follows the style and structure of the gqteaWinToga site (https://ademircamargo57.github.io/gqteaWinToga/): one `index.html`, `style.css`, `app.js`, served by GitHub Pages from `docs/` on `main`.
2. **Hero figure:** a real gqteashmd trajectory of ethylene, played back in the sister site's ball-and-stick style, with the active state and the hop shown.
3. **Validation:** a "How it was checked" section with each result marked (test, pre-registered, report) and a "Known limitations" box on the page, not only behind a link.
4. **Credits:** the same group, the same six names and the same contact as gqteaWinToga.
5. **AI credit:** one sentence in "Who makes it" saying the code was written with the help of an AI coding assistant (Claude, Anthropic).
6. The structure (section 2), the figures (section 3) and the validation content (section 4) were approved as presented in the conversation.

## 1. Files and hosting

| File | What it is |
|---|---|
| `docs/index.html` | the page |
| `docs/style.css` | gqteaWinToga's stylesheet (palette, fonts, dark mode, header, hero, tabs, cards, steps, footer) plus the few rules the new parts need (section 3, 4): the trajectory controls, the populations figure, the result badges, the limitations box |
| `docs/app.js` | the latest-release button, the tabs (gqteaWinToga's code), the trajectory player |
| `docs/trajectory.json` | the hero trajectory (section 3.1), written by the script |
| `docs/icon.svg` | the brand icon and favicon (an ethylene glyph in the site's colours; SVG, so no image tool is needed) |
| `docs/.nojekyll` | no Jekyll processing, as on gqteaWinToga |
| `tools/site/make_figures.py` | reads the ensemble's files and writes `trajectory.json` and the populations SVG (section 3) |

- **No external scripts.** The only outside resources are Google Fonts (STIX Two Text, Public Sans, JetBrains Mono), as on gqteaWinToga, and the GitHub API call of the release button (public, no credentials; the button keeps its fallback link when the call fails).
- **Publishing:** after the merge the owner enables GitHub Pages (Settings → Pages → Deploy from a branch → `main`, `/docs`). The address is `https://ademircamargo57.github.io/gqteaSHMD/`. The repository's About field may then link to it (owner).
- `docs/development.md` and `docs/superpowers/` are served by Pages as plain files too; they are already public in the repository (gqteaWinToga has the same arrangement).
- **Privacy:** the page contains only public material: the six names and `ajc@ueg.br` (already on gqteaWinToga), the repository links, and computed data. `make_figures.py` takes every path as an argument; no local path, host name or user name is committed in it or in `trajectory.json`.

## 2. Structure (gqteaWinToga's order)

1. **Header:** the icon and "gqteashmd"; links Workflow (`#workflow`), What's new (`#new`), Validation (`#checked`), Install (`#install`), Manual (`USER_MANUAL.md` on GitHub), Source code (the repository).
2. **Hero:**
   - `h1` "gqteashmd"; lede "Automated surface-hopping molecular dynamics with PySCF."; one paragraph: from one TOML input file, the program optimizes the molecule, computes its frequencies, samples initial conditions, computes the excitations, selects the initial states, runs the surface-hopping trajectories and analyses them, in one resumable run that survives crashes.
   - The yellow button: "Get gqteashmd 0.1.0", pointing at `releases/latest`. `app.js` asks the GitHub API for the latest release, points the button at its `.whl` asset and fills the line below it with the version, the wheel size and the release date; on failure it keeps the fallback. The line below reads "Linux only. Free and open source, MIT license." The second line links "Release notes and earlier versions".
   - The figure: the trajectory player (section 3.1).
3. **"What it does" (`#workflow`):** one intro sentence (one input file, eight stages, each stage resumable), then the sister site's tab box with five tabs. Each tab has a lead sentence and a list; the items come from the manual (sections 4 and 8):
   - **Input & control:** the strict TOML input and `validate`; preflight checks; parallel tasks with per-stage resources; crash-safe resume, `stop`, review pauses (`pause_after`); change detection with `--rerun-from`; provenance and the build stamp.
   - **Ground state:** `prepare`; `optimize` (geomeTRIC, optional); `frequencies` (analytic SCF Hessian or finite differences for SA-CASSCF, Molden file); the methods SCF, SA-CASSCF (with the start rule), TDA, TDDFT, CIS, TDHF.
   - **Initial conditions:** `sampling` (harmonic Wigner, exact thermal widths); `excitations` (vertical energies, oscillator strengths, the absorption spectrum as CSV); `selection` (bright-state selection by f/dE², or a fixed state).
   - **Dynamics:** `dynamics` (SHARC-type or Tully hopping, local diabatization from overlaps or SA-CASSCF couplings, EDC decoherence, step rejection with halving, a restart after every step, outcomes per trajectory, extension by raising `tmax_fs`).
   - **Analysis:** `analysis` (classical and quantum populations with 95 % intervals, the censored estimator by default, the delayed-exponential lifetime with bootstrap intervals, hop statistics, energy diagnostics; CSV, HDF5, JSON, Markdown).
4. **Spotlight "New in version 0.1.0: one input file" (`#new`, alternate background):** tag "First release, 0.1.0"; heading "One input file, one command"; a paragraph and the feature list (resume after a crash, stop and continue, pauses for review, change detection with `--rerun-from`, provenance of every run, the build stamp that names the commit); links to the manual's input and command sections. The figure is the real input of the ensemble in section 3 (`tools/sharc_compare/p9b/ethylene_seed2028.toml`, shown in the sister site's `input-sample` style); a test checks that the shown text equals that file and that the file validates (section 5).
5. **Spotlight "Dynamics that run to the end, or say why they stopped" (`#dynamics`):** tag "`dynamics` and `analysis` stages"; the feature list (overlaps every step, rejection signals and halving, outcomes, extension, the censored estimator); the populations figure (section 3.2).
6. **Cards "Other commands" (`#commands`):** `init`, `validate`, `run`, `status`, `stop`, `single-point`, `model-dynamics` (Tully's models, and LVC models with triplets and spin-orbit coupling), each with one or two sentences and a manual link.
7. **"How it was checked" (`#checked`, alternate background)** and the **"Known limitations"** box (section 4).
8. **Install (`#install`):** four steps (download `environment.lock.txt` and the wheel from the release; `conda create -n gqteashmd-0.1.0 --file environment.lock.txt`; `pip install --no-deps` the wheel; `gqteashmd init` then `gqteashmd run input.toml`); the "Requirements" box (Linux 64-bit, tested on Oracle Linux 9.5 under WSL2; conda (Miniforge); PySCF and every other dependency come from the lock file; SHARC and ORCA are needed only for the comparison tests; on Windows use WSL2 and keep projects off `/mnt/c` and OneDrive); "Install for development" folded inside it (clone, `conda env create -f environment.yml`, `pip install -e . --no-deps`). The commands are those of USER_MANUAL 2.2.
9. **"Learn to use it" and "Who makes it" (`#docs`):** the manual and its worked examples (water, the ethylene studies); issues; gQTEA at the State University of Goiás (UEG), the six names, `ajc@ueg.br`; the AI sentence: "gqteashmd was written by the gQTEA team with the help of an AI coding assistant (Claude, Anthropic); every result on this page comes from the program's tests and recorded runs."
10. **Footer:** "gqteashmd is free software released under the MIT License. © 2026 the gqteashmd developers, State University of Goiás." (the licence's holder).

## 3. The two figures (real data)

Both come from the P9b ethylene ensemble computed with gqteashmd ("ours", SA3-CASSCF(2,2)/6-31G\*, seed 2028, 100 trajectories, 100 fs at 0.5 fs, `coupling = "overlap"`; USER_MANUAL 8.17). With the default settings 0.1.0 gives the same numbers (USER_MANUAL: the `dynamics` version 3 changes no number under the default `hop_energy`), so no version caveat is printed.

### 3.1 The hero trajectory

- **Data:** `traj_00033`: `finished`, one hop S1→S0 at 28.0 fs, no rejection, max drift 0.069 eV, the C=C twist about 82° at the hop. `make_figures.py PROJECT TASK --out docs/` writes `trajectory.json`: `{"source", "method", "dt_fs", "symbols", "time_fs", "active", "state_labels", "hop_index", "positions_angstrom"}` with positions rounded to 3 decimals (201 frames).
- **Player** (`app.js`): the sister site's ball-and-stick drawing (radial-gradient atoms, painter's order, bonds by covalent radii recomputed every frame, drag to rotate, pause off screen). Playback about 10 s per 100 fs, looping, with a play/pause button and a time slider (`input type="range"`) whose track marks the hop. An overlay reads `t = 28.0 fs · S1` with the state in its colour (S1 the site's blue link colour, S0 the ink colour). With `prefers-reduced-motion` it does not auto-play: a still frame and the slider. Without `trajectory.json` (or JavaScript) the figure shows its caption and a short fallback text.
- **Caption:** "Ethylene after excitation to S1: one trajectory of a 100-trajectory gqteashmd ensemble (SA3-CASSCF(2,2)/6-31G\*, 0.5 fs steps). The C=C bond twists and the molecule hops to S0 at 28 fs. Drag to rotate."

### 3.2 The populations figure

- **Data:** the ensemble's `07_analysis/populations.csv` (P8/P9b rule, `estimator = "complete"`, 67 of 100 trajectories included, 95 % Wilson intervals) and `analysis.json` (fit t0 12.8 fs, tau 80.1 fs). `make_figures.py` writes the SVG (P(S0), P(S1), P(S2) against time, 0-100 fs, shaded bands, the fitted curve dashed, axis labels, a legend); colours through CSS variables so it follows dark mode; `role="img"` with an `aria-label` and a `<title>`.
- **Caption:** "Populations of the 67 of 100 trajectories that ran to 100 fs, with 95 % intervals; the dashed line is the fitted delayed exponential (tau 80 fs). The other 33 ended early after a hop. The default censored estimator also uses them and gives a shorter lifetime (62 fs here); which value is closer to the truth is not known (see Known limitations)." The 62.49 fs is the censored re-analysis of this project (censored-analysis report).

## 4. "How it was checked" and "Known limitations"

Every number comes from `CHANGELOG.md` 0.1.0 (checked against the manual by the release's final review). One intro sentence: gqteashmd is research software at version 0.1, and every result below is marked by how it was checked. Badges: **Test** (an automated test fails when the bound is missed), **Pre-registered** (a pass rule fixed before the run, checked by a tool run by hand), **Report** (a measurement with no pass rule). Seven cards, each linking to its manual section:

1. Electronic structure: energies, gradients, NACs against SHARC 4.1 (ethylene SA2-CASSCF(2,2)/6-31G\*, 1.4e-8 Eh, 1.4e-8 Eh/bohr, 9.1e-8 1/bohr up to sign); finite-difference gradients (Test) (8.6).
2. Frequencies and sampling: ORCA 6.1.1 within 0.052 cm⁻¹; Wigner sampling against `wigner.py`, means within 1.6 and standard deviations within 1.4 standard errors (Test) (8.7, 8.8).
3. Excitations and selection: against `SHARC_PYSCF.py` and `excite.py`, dE ≤ 8.7e-8 Eh (Test) (8.9).
4. Dynamics core: FSSH and local diabatization against exact wavepackets on Tully's models (Test); FSSH against an exact 2D LVC solver, largest differences 0.049 and 0.072 (Report); single trajectories against SHARC 4.1, nuclear motion 7.5e-11 bohr (Test) (8.14, 8.15).
5. The `dynamics` stage: restart and extension bit-identical with an analytic backend and to round-off with PySCF; SHARC's driver on our backend dq 2.1e-7 bohr over 41 steps (Test) (8.16).
6. Ensembles against SHARC (ethylene, 100 trajectories): swap ensemble, all four rules pass, 48 paired (Pre-registered); our population functions against `populations.py` (Pre-registered); the direct comparison inconclusive, 13 paired (Report) (8.17).
7. The analysis: the censored estimator unbiased and covering on a synthetic Markov model (Test); least biased of three under artificial random losses (Pre-registered) (8.19).

**Known limitations** (a box beside or below the cards): closed-shell singlets only, no triplets or spin-orbit coupling from PySCF, Linux only, one workstation without a batch scheduler; 33 of 100 ethylene trajectories end early after a hop (CASSCF solution switches); the lifetime is uncertain (censored 62 fs, complete-only 80 fs, the truth not known); the direct SHARC comparison is inconclusive, a replayed trajectory disagrees with SHARC after its hop under the default hop-energy rule, and the lifetime fit is not compared with `make_fit.py`; no plots (CSV, HDF5, JSON, Markdown); a link to the full list in `CHANGELOG.md` and the manual.

## 5. Checks

- `tests/site/test_site.py` (fast tier, no network, no PySCF): every local `href`/`src` in `docs/index.html` exists; every `#id` link has its target; every `USER_MANUAL.md#anchor` link matches a heading of `USER_MANUAL.md` (GitHub's slug rule); the sample input shown equals `tools/sharc_compare/p9b/ethylene_seed2028.toml` (after HTML unescaping) and that file passes `gqteashmd validate` when copied into `tmp_path` beside `tests/data/ethylene.xyz` (the geometry it names, `ethylene.xyz`, sat beside it in the run directory and is not in `tools/sharc_compare/p9b/`); `trajectory.json` has the keys of 3.1, consistent shapes, `active[hop_index] != active[hop_index - 1]`; the page loads no script other than `app.js`; no local path (`/home/`, `/mnt/`, `C:\`) appears in `docs/`.
- `tools/site/make_figures.py` has a unit test on a small written `output.h5` and `populations.csv` (h5py, as the other `tools/` tests).
- A manual browser check by the owner before enabling Pages: `python -m http.server -d docs 8000`, light and dark mode, a phone-width window (no horizontal scroll at 375 px), keyboard use of the tabs and the player controls.
- ruff on the new Python; the HTML/CSS/JS are not linted (as on gqteaWinToga).

## 6. Documents

README.md gets the website link at its top; USER_MANUAL.md section 1 or 2 names the site; CLAUDE.md gets a short block (the files, the script, the data source, Pages from `main` `/docs`, the test) and a status line. No behaviour, stage version or fingerprint changes.

## 7. Out of scope

A custom domain; analytics or cookies; a Portuguese version; plots inside the program (still CSV only); more figures than the two of section 3; changes to the gqteaWinToga site.

## 8. Risks

- **The trajectory looks unlike the textbook picture** (pyramidalization, hydrogen migration): the task is chosen in section 3.1 with the measured twist; if the playback reads badly, another listed one-hop trajectory may be chosen (`traj_00004`, hop 50.0 fs, twist 83°), recorded as a ruling.
- **GitHub anchors:** the manual's headings contain backticks and symbols; the test of section 5 computes the slugs, so a wrong anchor fails the test rather than the reader.
- **The API call** of the release button is rate-limited per visitor (60 per hour); the fallback link always works.
