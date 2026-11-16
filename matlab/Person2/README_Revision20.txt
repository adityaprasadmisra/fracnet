OTFS FRACTIONAL-DOPPLER ESTIMATOR — REVISION 20
================================================

Purpose
-------
Revision 20 is a deliberately narrow update to the Revision 19 estimator.
It targets the report's recurring weak point: a Nelder-Mead endpoint can
become the globally selected solution while remaining unpolished by the
analytic variable-projection LM solver.

Why this change
---------------
In the Revision 19 20 dB diagnostics, the worst gain-estimation trials
(8, 38 and 48) selected "Nelder-Mead fallback (unpolished)". Their local
gain-uncertainty warnings were also true. Trial 38 additionally had a
near-degenerate/ambiguous close-path solution.

Revision 20 change
------------------
Whenever a raw Nelder-Mead endpoint beats the current global best ML
objective, it is now passed through the existing analytic variable-
projection LM retry routine. The polished endpoint replaces the raw
endpoint only if it achieves a strictly lower valid ML residual.

This is a targeted polish, not a new objective:
- The final path-location objective remains the same unregularized ML
  residual.
- The condition validity test is unchanged.
- The final ridge gain estimation is unchanged.
- The channel model, dictionary, training design, and Person 1 TX/RX
  implementation are unchanged.
- Existing seed quotas, search budgets, LM limits, and Nelder-Mead limits
  are unchanged.

Expected effect
---------------
This removes an avoidable "unpolished winner" case. It may improve
fractional-Doppler and gain estimates in trials where a raw simplex
endpoint narrowly wins but the analytic LM can still reduce the residual.
It cannot guarantee recovery of a statistically ambiguous/ill-conditioned
path pair. In particular, Trial 38 may remain difficult because its true
pair is highly ill-conditioned and its competing hypotheses have nearly
indistinguishable likelihoods.

Installation
------------
1. Back up Person2.
2. Extract this package directly into Person2; MATLAB files are at the
   archive root.
3. Keep Person1 unchanged.
4. Run:

   clear; clc;
   check_revision20_installation
   test_channel_estimator

5. Compare Revision 20 against Revision 19 with identical random seeds,
   SNR grid, training frames, channel realizations, and noise realizations.

Important
---------
The source package has been statically inspected and packaged. MATLAB/
Octave execution was not available during preparation, so no measured
performance improvement or runtime result is claimed. First run the
noiseless checks, then the 50-trial comparison. Inspect Trials 8, 38, and
48 at 20 dB, plus all-SNR RMSE, gain NMSE median/p95, recovery rates,
ambiguity, and runtime.
