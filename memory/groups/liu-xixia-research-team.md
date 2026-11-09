---
display_name: "Liu Xixia research team"
summary: "The user's graduation thesis research team at Hubei Normal University; the user only does dual-T-line LFIA optimization for SM2 + ENR within its dual-target immunochromatography work."
rank: 1
updated: "2026-11-08"
---

# Liu Xixia research team

## Purpose
- The team the user does their graduation thesis with at Hubei Normal University.
- The thesis work is single-target sulfamethazine (SM2) immunoassay testing: antibody preparation, performance evaluation (ic-ELISA and IC50), and immunochromatography.
- Current immunochromatography work as of 2026-11-06 is dual-target: two T lines detecting SM2 and enrofloxacin (ENR); whether this expands the graduation thesis or is a separate experiment is unconfirmed.
- The user's own part is dual-T-line optimization, just starting as of 2026-11-06.
- Scope (user, 2026-11-07): only dual-T-line LFIA optimization for ENR and SM2, including ic-ELISA checkerboard/IC50 evaluation and strip-condition optimization — not the AMD/TYL part of Xiao's thesis, nor the other chapters.

## Members
- [Liu Xixia](../people/liu-xixia.md): leads the team.
- [Xiao Xiaoyue](../people/xiao-xiaoyue.md): advises the user's thesis.
- [Wen Shuang](../people/wen-shuang.md): SM2 IC50 measurements in October 2026.

## Dynamics
- Liu Xixia leads the team; Xiao Xiaoyue advises the user's thesis.

## Themes
- Single-target SM2 immunoassay testing.
- ic-ELISA experiments (checkerboard titrations, standard curves) and OD data processing in Excel.
- Dual-target SM2/ENR fluorescent LFIA (DDF-LFIA): AQB/AQBT probe preparation with EDC only (no NHS) at 25°C and 180 rpm, red AQB for SM2 and green AQBT for ENR per Xiao's paper excerpt.
- PB buffer: 0.2 mol/L NaH2PO4/Na2HPO4 stocks already prepared; prepare fresh in small amounts from pH 6.0, dilute 20× to 0.01 mol/L.
- Dual-T-line route: colloidal gold LFIA first, then red/green fluorescent microsphere LFIA, for accurate multi-target quantification; competitive principle, T = SM2-BSA, C = goat anti-mouse IgG.
- SM2 checkerboard titration, done as of 2026-11-06: mAb and SM2-BSA at 1, 0.3, 0.1, 0.03 μg/mL, positive 10 ng/mL SM2; best IR 79.75% at 0.03/0.03 μg/mL, negative OD only 0.948 (low signal).
- SM2 ic-ELISA standard curve, done as of 2026-11-06: 0–50 ng/mL, inhibition 0% to 85%, S-shaped, IC50 3.19 ng/mL, 3-replicate CV under 10%.
- Three SM2 IC50 plates (2026-10-26, 2026-10-27, 2026-10-31): the first two barely inhibited (max 53.5% and 46.5%); the 2026-10-31 plate matches the user's own file, IC50 1.47 ng/mL, not the PPT's 3.19 ng/mL; which plate the PPT used is still to ask.
- Xiao's full doctoral thesis, deep-read 2026-11-06: ch. 5 covers 4 targets (SM2, AMD, ENR, TYL); T2 coating antigen is ENR-BSA, not ENR-OVA; "AQC" is a typo for AQB; PNPB is 5% sucrose, 2% trehalose, 1% PEG 20000, 1% casein, 0.25% Tween-20 in 0.01 M BB buffer; ch. 5 antibodies are from Jiangxi Weibang, so the paper's SM2 IC50 0.69 ng/mL is not the team's target.
- Colloidal gold SM2 pre-experiment, done as of 2026-11-06: 10 ng/mL, T line almost invisible (92.01% inhibition); 1.25 ng/mL, ~59% inhibition, chosen for T-line position optimization.
- ENR checkerboard titration ran 2026-11-08: antibody and coating ENR-BSA at 1, 0.3, 0.1, 0.03 μg/mL, positive ENR 10 ng/mL, single well per combination. Best combination antibody 1 × coating 1 μg/mL, IR 81.3%, 阳 OD 1.355 — unlike SM2 (best 0.03/0.03), ENR needs high antibody; the 0.03 row is near-background noise. Selection criterion (§2.4.4.1): negative OD > 1.0 and highest spiked IR; both met.
- Thesis-style ENR checkerboard figure built 2026-11-08 (heatmap + bar chart, per thesis Fig 2.48–2.51): heatmap of 阳 OD with IR circles, red circle on the best working point; bar chart of IR by coating concentration.
- Next: ENR IC50 plate at antibody 1 × coating 1 μg/mL in triplicate (0–50 ng/mL); IC50 from the middle 4 points in the ~20%–80% IR range. ENR checkerboard + IC50 template built 2026-11-06, with formula-linked plot tables and auto-computed slope/intercept. After that: quantum-dot fluorescent microsphere LFIA pre-experiment and literature review.
- Experimental logic: checkerboard → IC50 → probe prep → colloidal gold pre-experiment (T-line drug 1.25 ng/mL) → dual-T-line optimization → fluorescent LFIA quantification (RGB + four-parameter fitting). If sensitivity falls short, loop back to the checkerboard combination or probe amount; if lines cross-talk, adjust spacing or coating amount.
- ENR standard solution: the user plans to prepare it themselves (2026-11-07); ask Xiao or lab seniors for the lab's method — none given in chat, including whether the standard is diluted like the coating antigen for storage.
- ENR coating antigen stock confirmed 2026-11-07: 5.2 mg/mL, to be diluted to 2 mg/mL for storage (5 μL stock + 8 μL PBS pH 7.4). Paper gives no storage protocol; coating diluted in 50 mM CBS pH 9.6 (§2.3.3), optimal T-line ENR-BSA 0.2 mg/mL (§5.4.2.3). Working dilution: 1 μg/mL in CBS pH 9.6, 3-fold to 0.333, 0.111, 0.037 μg/mL.
- ENR mAb stock confirmed 2026-11-07 from the user's tube photo: Weibang enrofloxacin antibody, batch AB-60…, 1 mg total, 19.5 mg/mL (unit partly cut off), ≈ 51 μL. At 19.5 mg/mL, 2 μg is ~0.10 μL, so pre-dilution is essential. Dilution to 1 μg/mL clarified (user, 2026-11-08): the notebook's 19.5 μg/mL is a fridge stock from a 1000× pre-dilution of the 19.5 mg/mL original, so 40 μL + 740 μL PBS gives 1 μg/mL. Serial gradient still open: 1300 μL with 260 μL transfers computes as 6-fold, not the intended 3-fold; user said coating antigen, notebook places the figures in the antibody box.
- Blocking solution for the ENR ic-ELISA is 0.2% gelatin in CBS pH 9.6 (§2.3.3), 200 μL/well in the lab, not the paper's 250 μL/well. 500 mL prep (2026-11-08): Na2CO3 0.795 g + NaHCO3 1.465 g + gelatin 1 g in ultrapure water.
- TMB solution: on 2026-11-08 the user mixed 3000 μL A + 600 μL B (5:1, 3600 μL total) and asked whether the ratio is correct; kit instructions unavailable.

## Open threads
- [ ] Confirm which bottles (5 mg/mL stock vs 1 mg/mL working solution) were used for antibody and SM2 on the 2026-10-23 plate, plus a three-replicate retest, before locking the combination.
- [ ] Confirm whether to adopt the paper excerpt's conditions, which differ from the user's notebook: EDC 25 vs 22.5 μL, antibody SM2 5 μg and ENR 2 μg vs 7.5 μg, ENR labelling pH 8.0 vs 6.0. Draft to Xiao prepared 2026-11-07; update numbers for 19.5 mg/mL.
- [x] T2 coating antigen resolved on 2026-11-06 from Xiao's full thesis: ENR-BSA, not ENR-OVA.
- [ ] Whether to dilute the stock first for a 2 μg ENR mAb dose (~0.10 μL of the 19.5 mg/mL stock) before the next labelling.
- [x] ENR checkerboard titration measured 2026-11-08; best combination antibody 1 × coating 1 μg/mL, IR 81.3%.
- [ ] Run the ENR IC50 plate at antibody 1 × coating 1 μg/mL in triplicate (0–50 ng/mL) to validate the best combination.
- [x] ENR coating antigen stock confirmed 2026-11-07: ENR, 5.2 mg/mL, to be diluted to 2 mg/mL for storage at microlitre scale (5 μL stock + 8 μL PBS pH 7.4).
- [x] ENR mAb stock concentration confirmed 2026-11-07 from the user's tube photo: Weibang enrofloxacin antibody, 19.5 mg/mL, 1 mg total, batch AB-60… (unit partly cut off — confirm it reads mg/mL).
- [x] ENR antibody dilution to 1 μg/mL resolved 2026-11-08: the 19.5 μg/mL fridge stock is a 1000× pre-dilution of the 19.5 mg/mL original, so 40 μL + 740 μL PBS gives 1 μg/mL.
- [ ] Resolve which reagent the ENR serial gradient belonged to (user said coating antigen on 2026-11-08; the notebook places the figures in the antibody box). Also the standard step-2 diluent (PBS vs ultrapure water).
- [ ] Find out which plate/dataset the PPT's SM2 IC50 of 3.19 ng/mL came from; the 2026-10-31 file computes 1.47 ng/mL.
- [ ] Ask Xiao or lab seniors how to prepare the ENR standard solution before the ENR IC50 plate; the user plans to prepare it themselves.
- [ ] Confirm the TMB A/B ratio with Xiao or lab seniors (user mixed 3000 μL A + 600 μL B, 5:1, on 2026-11-08; kit instructions unavailable).
