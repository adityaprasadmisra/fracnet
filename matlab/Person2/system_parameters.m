function cfg = system_parameters()
% SYSTEM_PARAMETERS (Revision 20)
% Common Person 2 configuration for OTFS fractional-Doppler estimation.
% Person 1 parameters.m remains the source for the TX/RX implementation.

    cfg.N = 12;
    cfg.M = 12;
    cfg.num_symbols = cfg.N * cfg.M;

    cfg.fc = 3e9;
    cfg.delta_f = 7.5e3;
    cfg.T = 1 / cfg.delta_f;
    cfg.fs = cfg.M * cfg.delta_f;
    cfg.Ts = 1 / cfg.fs;

    cfg.c = 3e8;
    cfg.v_max = 150 / 3.6;
    cfg.fd_max = cfg.v_max * cfg.fc / cfg.c;
    cfg.doppler_resolution = 1 / (cfg.N * cfg.T);
    cfg.nu_max = cfg.fd_max / cfg.doppler_resolution;

    % Current physical channel model: integer sample-delay taps.
    cfg.l_max = 2;
    cfg.k_max = 2; % Metadata only; not the physical Doppler bound.
    cfg.max_delay_samples = cfg.l_max;
    cfg.max_delay_seconds_in_model = cfg.max_delay_samples * cfg.Ts;
    cfg.max_delay = 2000e-9; % Retained from original specification.

    cfg.P = 2;
    cfg.pilot_amplitude = 1;
    cfg.pilot_seed = 2026;

    % IMPORTANT: estimator-identification experiment uses contiguous known
    % OTFS training frames. Doppler phase is continuous across frame edges.
    % Set this to 1 to recover the original single-frame experiment.
    % This is NOT the paper's single-cell/superimposed-pilot overhead design.
    cfg.num_training_frames = 4;

    % The longer observation aperture makes the original very fine grid
    % unnecessary. Continuous LM refinement supplies sub-grid precision.
    cfg.estimator_doppler_step = 0.01;
    cfg.estimator_pair_min_denominator = 1e-8;
    % Base pair-condition cap used at low SNR. The estimator adapts this
    % limit using the estimated observation SNR, up to the ceiling below.
    cfg.estimator_max_pair_condition = 25;
    cfg.estimator_adaptive_condition_ceiling = 1000;
    cfg.estimator_condition_snr_exponent = 1.0;
    % Begin relaxing the condition cap only above 2.5x linear observation SNR
    % (approximately 4 dB), limiting unstable fits at moderate SNR while
    % still admitting very close pairs at high SNR.
    cfg.estimator_condition_snr_reference_linear = 2.5;

    % Complementary initializations for close and weak second paths.
    cfg.estimator_residual_seed_enable = true;
    cfg.estimator_residual_seed_top_atoms = 6;
    cfg.estimator_residual_seed_second_atoms = 1;

    % Extra coarse hypotheses for same-delay paths with close Dopplers.
    % These only add refinement starts; the final ML objective is unchanged.
    cfg.estimator_close_pair_seed_enable = true;
    cfg.estimator_close_pair_min_grid_steps = 0.5;
    cfg.estimator_close_pair_max_grid_steps = 10.0;
    cfg.estimator_close_pair_seeds_per_delay = 3;
    % Stratify same-delay seed candidates so very-close and moderately-close
    % Doppler pairs both enter continuous refinement.
    cfg.estimator_close_pair_separation_bands_steps = ...
        [0.5 1.5; 1.5 3.0; 3.0 10.0];
    % Permit a small, explicitly reserved close-pair pool with its own
    % SNR-dependent condition cap. This avoids admitting highly ill-conditioned
    % close pairs indiscriminately at low SNR. Final ML selection is unchanged.
    cfg.estimator_close_pair_condition_ceiling = 600;
    cfg.estimator_close_pair_condition_snr_exponent = 1.3;
    cfg.estimator_close_pair_seed_condition_penalty = 1.0;

    % Preserve the Revision 18 nonlinear refinement depth and LM budget.
    % Reallocate one pattern slot from ordinary coarse search to close-pair
    % coverage without reducing the total number of pattern seeds.
    cfg.estimator_pattern_seed_budget = 24;
    cfg.estimator_pattern_coarse_budget = 11;
    cfg.estimator_pattern_close_budget = 9;
    cfg.estimator_pattern_residual_budget = 4;
    cfg.estimator_close_pair_refine_levels = 4;

    % Gain regularization is applied only after locations are selected.
    cfg.estimator_ridge_multiplier = 1.0;
    cfg.estimator_condition_penalty = 0.0; % backward compatibility only

    % Search budget. Keep the established Revision 18 iteration limits so
    % high-SNR and different-delay cases retain their previous refinement.
    cfg.estimator_refine_top_k = 24;
    cfg.estimator_seeds_per_delay_pair = 4;
    cfg.estimator_seed_min_separation_steps = 1.0;
    cfg.estimator_refine_levels = 6;
    cfg.estimator_lm_top_k = 12;
    cfg.estimator_fminsearch_top_k = 2;
    cfg.estimator_nonlinear_seeds_per_delay_pair = 3;
    cfg.estimator_refined_seed_separation_steps = 0.15;
    cfg.estimator_fminsearch_max_iter = 80;
    cfg.estimator_fminsearch_max_fun_evals = 160;
    cfg.estimator_fminsearch_tol_x = 1e-9;
    cfg.estimator_fminsearch_tol_fun = 1e-12;
    cfg.estimator_lm_max_iter = 45;
    cfg.estimator_lm_initial_damping = 1e-3;
    cfg.estimator_lm_max_step = 0.05;
    cfg.estimator_lm_gradient_tolerance = 1e-9;
    cfg.estimator_lm_step_tolerance = 1e-10;
    % Controlled alternate-damping retries for stalled analytic LM calls.
    cfg.estimator_lm_retry_enable = true;
    cfg.estimator_lm_retry_budget = 1; % total retries per estimator call
    % Fallback selection reserves starts for close/residual/high-condition
    % candidates; the soft penalty is used only inside Nelder-Mead.
    cfg.estimator_fallback_condition_trigger = 10;
    cfg.estimator_fallback_condition_penalty = 0.02;
    cfg.estimator_condition_warning = 20;

    % Close same-delay pairs use center/separation coordinates
    % in the derivative-free fallback. The solution bank reports competing
    % hypotheses whose residual gaps are small relative to noise variance.
    cfg.estimator_close_pair_parameterization_enable = true;
    cfg.estimator_close_pair_nm_max_separation = 0.12; % normalized Doppler bins
    cfg.estimator_close_pair_nm_min_separation = 0.001;
    % Use a center/separation coordinate pattern search for
    % same-delay pairs up to 0.12 normalized Doppler bins apart. This covers
    % the difficult 0.08-0.10-bin cases observed in Revision 16 as well as
    % the very-close 0.01-bin cases. This improves search coverage but cannot
    % overcome the information limit of a short observation aperture.
    cfg.estimator_close_pair_search_max_separation = 0.12;
    cfg.estimator_close_pair_search_min_separation = ...
        max(0.25*cfg.estimator_doppler_step,1e-4);
    cfg.estimator_ambiguity_delta_log_likelihood = 2.3;
    cfg.estimator_ambiguity_min_location_separation = 0.5*cfg.estimator_doppler_step;
    cfg.estimator_ambiguity_condition_warning = 20;
    cfg.estimator_local_information_warning = 1e8;
    cfg.estimator_gain_uncertainty_warning = 0.50;

    % Monte Carlo settings; the channel bank is reused at every SNR.
    cfg.SNR_dB = 0:2:20;
    cfg.test_num_trials = 50;
    cfg.random_seed = 42;
    cfg.noise_seed = 20261104;

    % Definitions for reporting.
    cfg.doppler_recovery_tolerance = 0.02;
    cfg.doppler_reporting_tolerances = [0.01 0.02 0.03 0.05];
    cfg.gain_relative_error_tolerance = 0.25;

    % Retained metadata for other project scripts.
    cfg.num_samples = 1000;
    cfg.output_dir = 'dataset';
end
