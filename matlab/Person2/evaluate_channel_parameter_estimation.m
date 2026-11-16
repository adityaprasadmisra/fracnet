function results = evaluate_channel_parameter_estimation( ...
    Xpilot,cfg,dopplerGrid,delayGrid,snrVec,numTrials,dictionary)
% EVALUATE_CHANNEL_PARAMETER_ESTIMATION
% Controlled Monte Carlo evaluation of physical two-path estimates.
%
% Revision 20:
%   - Reuses identical channels and identical standardized noise samples
%     across SNR points (common-random-numbers comparison).
%   - Matches paths using delay/Doppler locations only; gains do not decide
%     which physical path is paired with which estimate.
%   - Reports path recovery at several Doppler tolerances.
%   - Retains per-trial true/estimated parameters for failure diagnosis.
%   - Reports achieved time-domain SNR and raw observation-noise NMSE.
%   - Adds an oracle-location least-squares residual diagnostic. The oracle
%     uses true locations only for evaluation, never for estimation.
%
% Metrics:
%   Doppler and delay RMSE, overall and conditional complex-gain NMSE,
%   response NMSE, delay/path/full recovery rates, pair conditioning, and
%   oracle-location residual diagnostics. It reports the full upper tail of
%   per-trial gain NMSE so close-path outliers are not hidden by the median.

    if ~isstruct(cfg) || ~isfield(cfg,'P') || cfg.P ~= 2
        error('This evaluator requires a valid cfg structure with cfg.P = 2.');
    end
    if ~isnumeric(Xpilot) || isempty(Xpilot) || any(~isfinite(Xpilot(:)))
        error('Xpilot must be a nonempty finite numeric training grid.');
    end
    if size(Xpilot,1) ~= cfg.N || size(Xpilot,2) ~= cfg.M
        error('Xpilot dimensions must match cfg.N and cfg.M.');
    end
    if ~isnumeric(snrVec) || ~isreal(snrVec) || ...
            ~isvector(snrVec) || isempty(snrVec) || ...
            any(~isfinite(snrVec(:)))
        error('snrVec must be a nonempty finite numeric vector.');
    end
    snrVec = snrVec(:).';
    if nargin < 6 || isempty(numTrials)
        if ~isfield(cfg,'test_num_trials')
            error('numTrials is required when cfg.test_num_trials is not defined.');
        end
        numTrials = cfg.test_num_trials;
    end
    if nargin < 7 || isempty(dictionary)
        [A,atomNorms,pairCache] = fractional_doppler_dictionary( ...
            Xpilot,cfg,dopplerGrid,delayGrid);
    else
        requiredDictionaryFields = {'A','atomNorms','pairCache'};
        if ~isstruct(dictionary)
            error('dictionary must be the structure returned by the dictionary builder.');
        end
        for fieldIndex = 1:numel(requiredDictionaryFields)
            if ~isfield(dictionary,requiredDictionaryFields{fieldIndex})
                error('dictionary is missing field %s.', ...
                    requiredDictionaryFields{fieldIndex});
            end
        end
        A = dictionary.A;
        atomNorms = dictionary.atomNorms;
        pairCache = dictionary.pairCache;
    end
    if size(A,1) ~= numel(Xpilot) || size(A,2) ~= numel(atomNorms)
        error('The supplied dictionary dimensions do not match Xpilot/atomNorms.');
    end
    if ~isnumeric(numTrials) || ~isreal(numTrials) || ...
            ~isscalar(numTrials) || ~isfinite(numTrials) || numTrials < 1 || numTrials ~= floor(numTrials)
        error('numTrials must be a finite positive integer scalar.');
    end

    pilotTx = modulate_training_sequence(Xpilot,cfg);
    numObservations = numel(pilotTx);
    fprintf('Using %d contiguous known training frames (%d samples total).\n', ...
        get_cfg(cfg,'num_training_frames',1),numObservations);

    % Generate one fixed channel bank.
    originalRng = rng;
    restoreRng = onCleanup(@() rng(originalRng)); %#ok<NASGU>
    rng(cfg.random_seed,'twister');
    channelBank = cell(numTrials,1);
    for trial = 1:numTrials
        channelBank{trial} = generate_channel_params(cfg);
    end

    % Generate one standardized complex noise vector per trial. The same
    % vector is scaled to the required variance at every SNR. This paired
    % noise comparison reduces Monte Carlo variation between SNR points.
    noiseSeed = get_cfg(cfg,'noise_seed',cfg.random_seed+100000);
    noiseBank = cell(numTrials,1);
    for trial = 1:numTrials
        rng(noiseSeed+trial,'twister');
        noiseBank{trial} = (randn(numObservations,1) + ...
            1i*randn(numObservations,1))/sqrt(2);
    end

    dopplerTolerance = get_cfg(cfg,'doppler_recovery_tolerance', ...
        max(1.5*median(diff(dopplerGrid)),0.02));
    gainRelativeTolerance = get_cfg(cfg,'gain_relative_error_tolerance',0.25);
    conditionWarning = get_cfg(cfg,'estimator_condition_warning',100);
    reportingTolerances = get_cfg(cfg,'doppler_reporting_tolerances', ...
        [0.01 0.02 0.03 0.05]);
    reportingTolerances = unique(sort(reportingTolerances(:).'));

    numSNR = numel(snrVec);
    dopplerRMSE = zeros(size(snrVec));
    delayRMSE = zeros(size(snrVec));
    gainNMSE = zeros(size(snrVec));
    conditionalGainNMSE = NaN(size(snrVec));
    medianTrialGainNMSE = zeros(size(snrVec));
    p95TrialGainNMSE = zeros(size(snrVec));
    p90TrialGainNMSE = zeros(size(snrVec));
    maxTrialGainNMSE = zeros(size(snrVec));
    maxTrialGainIndex = zeros(size(snrVec));
    exactDelayRate = zeros(size(snrVec));
    pathLocationRate = zeros(size(snrVec));
    fullParameterRecoveryRate = zeros(size(snrVec));
    pathLocationRatesByTolerance = zeros(numel(reportingTolerances),numSNR);
    responseNMSE = zeros(size(snrVec));
    responseNMSEStd = zeros(size(snrVec));
    meanPairCondition = zeros(size(snrVec));
    p95PairCondition = zeros(size(snrVec));
    illConditionedRate = zeros(size(snrVec));
    meanPairCoherence = zeros(size(snrVec));
    meanMeasuredSNR = zeros(size(snrVec));
    p05MeasuredSNR = zeros(size(snrVec));
    p95MeasuredSNR = zeros(size(snrVec));
    meanRawNoiseNMSE = zeros(size(snrVec));
    meanEstimatorErrorToNoise = zeros(size(snrVec));

    trialDopplerError = zeros(numTrials,cfg.P,numSNR);
    trialDelayError = zeros(numTrials,cfg.P,numSNR);
    trialGainNMSE = zeros(numTrials,numSNR);
    trialResponseNMSE = zeros(numTrials,numSNR);
    trialPairCondition = zeros(numTrials,numSNR);
    trialPairCoherence = zeros(numTrials,numSNR);
    trialMeasuredSNR = zeros(numTrials,numSNR);
    trialRawNoiseNMSE = zeros(numTrials,numSNR);
    trialEstimatorErrorToNoise = zeros(numTrials,numSNR);
    trialLocationRecovered = false(numTrials,numSNR);
    trialFullRecovered = false(numTrials,numSNR);
    trialTrueDoppler = zeros(numTrials,cfg.P,numSNR);
    trialEstimatedDoppler = zeros(numTrials,cfg.P,numSNR);
    trialTrueDelay = zeros(numTrials,cfg.P,numSNR);
    trialEstimatedDelay = zeros(numTrials,cfg.P,numSNR);
    trialTrueGain = complex(zeros(numTrials,cfg.P,numSNR));
    trialEstimatedGain = complex(zeros(numTrials,cfg.P,numSNR));
    trialRefinementSource = cell(numTrials,numSNR);
    trialLMIterations = zeros(numTrials,numSNR);
    trialLMAcceptedSteps = zeros(numTrials,numSNR);
    trialLMRetryCalls = zeros(numTrials,numSNR);
    trialNelderMeadStarts = zeros(numTrials,numSNR);
    trialNelderMeadProjectedEndpoints = zeros(numTrials,numSNR);
    trialOracleObjectiveGapdB = zeros(numTrials,numSNR);
    trialOraclePairCondition = zeros(numTrials,numSNR);
    trialOracleGainNMSE = zeros(numTrials,numSNR);
    trialMinimumPathSNRdB = zeros(numTrials,numSNR);
    trialMaxPairConditionUsed = zeros(numTrials,numSNR);
    trialClosePairConditionCeiling = zeros(numTrials,numSNR);
    trialEstimatedObservationSNRdB = zeros(numTrials,numSNR);
    trialValidPairCount = zeros(numTrials,numSNR);
    trialNewlyAdmittedPairCount = zeros(numTrials,numSNR);
    trialClosePairSeedCount = zeros(numTrials,numSNR);
    trialResidualSeedCount = zeros(numTrials,numSNR);
    trialCoarseSeedCount = zeros(numTrials,numSNR);
    trialRefinedCandidateCount = zeros(numTrials,numSNR);
    trialEstimatorRuntimeSeconds = zeros(numTrials,numSNR);
    trialAmbiguityFlag = false(numTrials,numSNR);
    trialMultiHypothesisAmbiguous = false(numTrials,numSNR);
    trialNearDegeneratePair = false(numTrials,numSNR);
    trialAmbiguityDeltaLogLikelihood = Inf(numTrials,numSNR);
    trialAmbiguityAlternativeCondition = Inf(numTrials,numSNR);
    trialAmbiguityCompetitorCount = zeros(numTrials,numSNR);
    trialAmbiguityAlternativeDoppler = NaN(numTrials,cfg.P,numSNR);
    trialAmbiguityAlternativeDelay = NaN(numTrials,cfg.P,numSNR);
    trialAmbiguityAlternativeSource = cell(numTrials,numSNR);
    trialClosePairNMStarts = zeros(numTrials,numSNR);
    trialClosePairPatternStarts = zeros(numTrials,numSNR);
    trialLocalDopplerStdBins = NaN(numTrials,cfg.P,numSNR);
    trialLocalGainRelativeStd = NaN(numTrials,cfg.P,numSNR);
    trialLocalInformationCondition = Inf(numTrials,numSNR);
    trialLocalGainUncertaintyFlag = true(numTrials,numSNR);

    for iSNR = 1:numSNR
        currentSNR = snrVec(iSNR);
        sumDopplerSquaredError = 0;
        sumDelaySquaredError = 0;
        sumGainSquaredError = 0;
        sumTrueGainPower = 0;
        conditionalGainError = 0;
        conditionalGainPower = 0;
        sumResponseError = 0;
        sumResponseReferencePower = 0;
        exactDelayCount = 0;
        locationCount = 0;
        fullRecoveryCount = 0;
        toleranceCounts = zeros(numel(reportingTolerances),1);
        measuredSNRThisPoint = zeros(numTrials,1);
        rawNoiseNMSEThisPoint = zeros(numTrials,1);
        errorToNoiseThisPoint = zeros(numTrials,1);

        fprintf('\nParameter evaluation: SNR = %g dB (%d/%d)\n', ...
            currentSNR,iSNR,numSNR);

        for trial = 1:numTrials
            channel = channelBank{trial};
            trueDoppler = channel.doppler(:);
            trueDelay = channel.delay(:);
            trueGain = channel.gain(:);

            rClean = apply_fractional_doppler_channel(pilotTx,channel,cfg);
            Yclean = demodulate_training_sequence(rClean,cfg);
            signalPower = mean(abs(rClean(:)).^2);
            noiseVariance = signalPower/10^(currentSNR/10);

            % Identical standardized complex AWGN per trial, scaled for SNR.
            noiseTime = sqrt(noiseVariance)*noiseBank{trial};
            rNoisy = rClean + noiseTime;
            Ynoisy = demodulate_training_sequence(rNoisy,cfg);
            noiseDD = demodulate_training_sequence(noiseTime,cfg);

            % Audit achieved SNR rather than assuming the generated noise
            % vector has exactly unit empirical power in a finite frame.
            measuredSNR = 10*log10(sum(abs(rClean(:)).^2) / ...
                (sum(abs(noiseTime(:)).^2)+eps));
            rawNoiseNMSE = norm(noiseDD(:))^2/(norm(Yclean(:))^2+eps);

            estimatorTimer = tic;
            [H_est,estimatedPaths,estimatedGains,~,info] = ...
                fractional_doppler_coarse_estimation( ...
                Ynoisy,A,dopplerGrid,delayGrid,cfg, ...
                atomNorms,pairCache,noiseVariance);
            trialEstimatorRuntimeSeconds(trial,iSNR) = toc(estimatorTimer);
            trialAmbiguityFlag(trial,iSNR) = info.ambiguity_flag;
            trialMultiHypothesisAmbiguous(trial,iSNR) = ...
                info.multi_hypothesis_ambiguous;
            trialNearDegeneratePair(trial,iSNR) = ...
                info.near_degenerate_pair_flag;
            trialAmbiguityDeltaLogLikelihood(trial,iSNR) = ...
                info.ambiguity_delta_log_likelihood;
            trialAmbiguityAlternativeCondition(trial,iSNR) = ...
                info.ambiguity_alternative_condition;
            trialAmbiguityCompetitorCount(trial,iSNR) = ...
                info.ambiguity_competitor_count;
            trialAmbiguityAlternativeDoppler(trial,:,iSNR) = ...
                info.ambiguity_alternative_doppler;
            trialAmbiguityAlternativeDelay(trial,:,iSNR) = ...
                info.ambiguity_alternative_delay;
            trialAmbiguityAlternativeSource{trial,iSNR} = ...
                info.ambiguity_alternative_source;
            trialClosePairNMStarts(trial,iSNR) = info.close_pair_nm_starts;
            trialClosePairPatternStarts(trial,iSNR) = ...
                info.close_pair_pattern_search_starts;
            trialLocalDopplerStdBins(trial,:,iSNR) = ...
                info.local_doppler_std_bins;
            trialLocalGainRelativeStd(trial,:,iSNR) = ...
                info.local_relative_gain_std_approx;
            trialLocalInformationCondition(trial,iSNR) = ...
                info.local_information_condition;
            trialLocalGainUncertaintyFlag(trial,iSNR) = ...
                info.local_gain_uncertainty_flag;

            % Diagnostic only: fit the noisy observation using the TRUE
            % path locations, while re-estimating complex gains by LS.
            % This is not passed to the estimator and is never used to
            % select its output. It tells us whether a miss is due to the
            % optimizer failing to reach the true-location basin (positive
            % gap) or the noisy ML objective preferring another pair
            % (negative gap / noise-fitting ambiguity).
            [oracleObjective,oracleCondition,oracleGains,pathSNRdB] = ...
                oracle_location_fit(pilotTx,Ynoisy,trueDoppler,trueDelay, ...
                trueGain,noiseVariance,cfg);
            oracleGapdB = 10*log10((info.best_objective+eps) / ...
                (oracleObjective+eps));
            oracleGainError = oracleGains(:)-trueGain(:);
            oracleGainNMSE = sum(abs(oracleGainError).^2) / ...
                (sum(abs(trueGain).^2)+eps);

            estimatedDoppler = estimatedPaths(:,1);
            estimatedDelay = estimatedPaths(:,2);
            estimatedGains = estimatedGains(:);

            % Match by physical path location only. Gain values must not
            % influence path identity; gain error is measured afterwards.
            directCost = assignment_cost(trueDoppler,trueDelay, ...
                estimatedDoppler,estimatedDelay,cfg.nu_max);
            swappedCost = assignment_cost(trueDoppler,trueDelay, ...
                flipud(estimatedDoppler),flipud(estimatedDelay),cfg.nu_max);
            if directCost <= swappedCost
                assignment = [1;2];
            else
                assignment = [2;1];
            end

            matchedDoppler = estimatedDoppler(assignment);
            matchedDelay = estimatedDelay(assignment);
            matchedGain = estimatedGains(assignment);

            dopplerError = matchedDoppler-trueDoppler;
            delayError = matchedDelay-trueDelay;
            gainError = matchedGain-trueGain;
            relativeGainError = abs(gainError)./max(abs(trueGain),1e-12);

            trialDopplerError(trial,:,iSNR) = dopplerError(:).';
            trialDelayError(trial,:,iSNR) = delayError(:).';
            trialGainNMSE(trial,iSNR) = sum(abs(gainError).^2)/ ...
                (sum(abs(trueGain).^2)+eps);
            trialResponseNMSE(trial,iSNR) = ...
                norm(H_est(:)-Yclean(:))^2/(norm(Yclean(:))^2+eps);
            trialPairCondition(trial,iSNR) = info.selected_pair_gram_condition;
            trialRefinementSource{trial,iSNR} = info.selected_refinement_source;
            trialLMIterations(trial,iSNR) = info.lm_total_iterations;
            trialLMAcceptedSteps(trial,iSNR) = info.lm_total_accepted_steps;
            trialLMRetryCalls(trial,iSNR) = info.lm_retry_calls;
            trialNelderMeadStarts(trial,iSNR) = ...
                numel(info.nelder_mead_candidate_indices);
            trialNelderMeadProjectedEndpoints(trial,iSNR) = ...
                info.nelder_mead_projected_endpoints;
            trialOracleObjectiveGapdB(trial,iSNR) = oracleGapdB;
            trialOraclePairCondition(trial,iSNR) = oracleCondition;
            trialOracleGainNMSE(trial,iSNR) = oracleGainNMSE;
            trialMinimumPathSNRdB(trial,iSNR) = min(pathSNRdB);
            trialMaxPairConditionUsed(trial,iSNR) = ...
                info.max_pair_condition_used;
            trialClosePairConditionCeiling(trial,iSNR) = ...
                info.close_pair_condition_ceiling;
            trialEstimatedObservationSNRdB(trial,iSNR) = ...
                info.estimated_observation_snr_db;
            trialValidPairCount(trial,iSNR) = ...
                info.valid_pair_count_after_adaptation;
            trialNewlyAdmittedPairCount(trial,iSNR) = ...
                info.newly_admitted_pair_count;
            trialClosePairSeedCount(trial,iSNR) = ...
                info.close_pair_seed_count;
            trialResidualSeedCount(trial,iSNR) = ...
                info.residual_seed_count;
            trialCoarseSeedCount(trial,iSNR) = info.coarse_seed_count;
            trialRefinedCandidateCount(trial,iSNR) = ...
                info.nonlinear_refinement_starts;
            trialPairCoherence(trial,iSNR) = info.selected_atom_coherence;
            trialMeasuredSNR(trial,iSNR) = measuredSNR;
            trialRawNoiseNMSE(trial,iSNR) = rawNoiseNMSE;
            trialEstimatorErrorToNoise(trial,iSNR) = ...
                norm(H_est(:)-Ynoisy(:))^2/(norm(noiseDD(:))^2+eps);
            measuredSNRThisPoint(trial) = measuredSNR;
            rawNoiseNMSEThisPoint(trial) = rawNoiseNMSE;
            errorToNoiseThisPoint(trial) = ...
                trialEstimatorErrorToNoise(trial,iSNR);
            trialTrueDoppler(trial,:,iSNR) = trueDoppler(:).';
            trialEstimatedDoppler(trial,:,iSNR) = matchedDoppler(:).';
            trialTrueDelay(trial,:,iSNR) = trueDelay(:).';
            trialEstimatedDelay(trial,:,iSNR) = matchedDelay(:).';
            trialTrueGain(trial,:,iSNR) = trueGain(:).';
            trialEstimatedGain(trial,:,iSNR) = matchedGain(:).';

            sumDopplerSquaredError = sumDopplerSquaredError + ...
                sum(abs(dopplerError).^2);
            sumDelaySquaredError = sumDelaySquaredError + ...
                sum(abs(delayError).^2);
            sumGainSquaredError = sumGainSquaredError + sum(abs(gainError).^2);
            sumTrueGainPower = sumTrueGainPower + sum(abs(trueGain).^2);

            delayCorrect = all(delayError == 0);
            locationCorrect = delayCorrect && ...
                all(abs(dopplerError) <= dopplerTolerance);
            fullCorrect = locationCorrect && ...
                all(relativeGainError <= gainRelativeTolerance);

            if delayCorrect
                exactDelayCount = exactDelayCount+1;
            end
            for t = 1:numel(reportingTolerances)
                if delayCorrect && all(abs(dopplerError) <= reportingTolerances(t))
                    toleranceCounts(t) = toleranceCounts(t)+1;
                end
            end
            if locationCorrect
                locationCount = locationCount+1;
                conditionalGainError = conditionalGainError + ...
                    sum(abs(gainError).^2);
                conditionalGainPower = conditionalGainPower + ...
                    sum(abs(trueGain).^2);
            end
            if fullCorrect
                fullRecoveryCount = fullRecoveryCount+1;
            end

            trialLocationRecovered(trial,iSNR) = locationCorrect;
            trialFullRecovered(trial,iSNR) = fullCorrect;

            responseError = norm(H_est(:)-Yclean(:))^2;
            responseReferencePower = norm(Yclean(:))^2;
            sumResponseError = sumResponseError+responseError;
            sumResponseReferencePower = sumResponseReferencePower+ ...
                responseReferencePower;
        end

        totalPaths = numTrials*cfg.P;
        dopplerRMSE(iSNR) = sqrt(sumDopplerSquaredError/totalPaths);
        delayRMSE(iSNR) = sqrt(sumDelaySquaredError/totalPaths);
        gainNMSE(iSNR) = sumGainSquaredError/(sumTrueGainPower+eps);
        if conditionalGainPower > 0
            conditionalGainNMSE(iSNR) = conditionalGainError/ ...
                (conditionalGainPower+eps);
        end
        exactDelayRate(iSNR) = exactDelayCount/numTrials;
        pathLocationRate(iSNR) = locationCount/numTrials;
        fullParameterRecoveryRate(iSNR) = fullRecoveryCount/numTrials;
        pathLocationRatesByTolerance(:,iSNR) = toleranceCounts/numTrials;
        responseNMSE(iSNR) = sumResponseError/(sumResponseReferencePower+eps);
        responseNMSEStd(iSNR) = std(trialResponseNMSE(:,iSNR));
        medianTrialGainNMSE(iSNR) = median(trialGainNMSE(:,iSNR));
        p90TrialGainNMSE(iSNR) = percentile_simple(trialGainNMSE(:,iSNR),90);
        p95TrialGainNMSE(iSNR) = percentile_simple(trialGainNMSE(:,iSNR),95);
        [maxTrialGainNMSE(iSNR),maxTrialGainIndex(iSNR)] = ...
            max(trialGainNMSE(:,iSNR));
        meanPairCondition(iSNR) = mean(trialPairCondition(:,iSNR));
        p95PairCondition(iSNR) = percentile_simple(trialPairCondition(:,iSNR),95);
        illConditionedRate(iSNR) = mean(trialPairCondition(:,iSNR)>conditionWarning);
        meanPairCoherence(iSNR) = mean(trialPairCoherence(:,iSNR));
        meanMeasuredSNR(iSNR) = mean(measuredSNRThisPoint);
        p05MeasuredSNR(iSNR) = percentile_simple(measuredSNRThisPoint,5);
        p95MeasuredSNR(iSNR) = percentile_simple(measuredSNRThisPoint,95);
        meanRawNoiseNMSE(iSNR) = mean(rawNoiseNMSEThisPoint);
        meanEstimatorErrorToNoise(iSNR) = mean(errorToNoiseThisPoint);

        fprintf('Doppler RMSE                       = %.6f bins\n',dopplerRMSE(iSNR));
        fprintf('Delay RMSE                         = %.6f samples\n',delayRMSE(iSNR));
        fprintf('Physical complex-gain NMSE         = %.6e (%.2f dB)\n', ...
            gainNMSE(iSNR),10*log10(gainNMSE(iSNR)+eps));
        fprintf('Conditional gain NMSE (location OK)= %.6e\n', ...
            conditionalGainNMSE(iSNR));
        fprintf('Trial gain NMSE median / p90 / p95 = %.3e / %.3e / %.3e\n', ...
            medianTrialGainNMSE(iSNR),p90TrialGainNMSE(iSNR), ...
            p95TrialGainNMSE(iSNR));
        fprintf('Worst trial gain NMSE               = %.3e (trial %d)\n', ...
            maxTrialGainNMSE(iSNR),maxTrialGainIndex(iSNR));
        fprintf('Exact delay-pair recovery          = %.1f %%\n',100*exactDelayRate(iSNR));
        fprintf('Path-location recovery (%.3f bins)  = %.1f %%\n', ...
            dopplerTolerance,100*pathLocationRate(iSNR));
        fprintf('Full parameter recovery             = %.1f %%\n', ...
            100*fullParameterRecoveryRate(iSNR));
        fprintf('Pooled pilot-response NMSE          = %.6e (%.2f dB)\n', ...
            responseNMSE(iSNR),10*log10(responseNMSE(iSNR)+eps));
        fprintf('Pair condition mean / median / p95 / max = %.2f / %.2f / %.2f / %.2f\n', ...
            meanPairCondition(iSNR),median(trialPairCondition(:,iSNR)), ...
            p95PairCondition(iSNR),max(trialPairCondition(:,iSNR)));
        fprintf('Adaptive condition limit mean / max = %.2f / %.2f\n', ...
            mean(trialMaxPairConditionUsed(:,iSNR)), ...
            max(trialMaxPairConditionUsed(:,iSNR)));
        fprintf('Newly admitted dictionary pairs (mean) = %.1f\n', ...
            mean(trialNewlyAdmittedPairCount(:,iSNR)));
        fprintf('Coarse / close / residual seeds (mean) = %.1f / %.1f / %.1f\n', ...
            mean(trialCoarseSeedCount(:,iSNR)), ...
            mean(trialClosePairSeedCount(:,iSNR)), ...
            mean(trialResidualSeedCount(:,iSNR)));
        fprintf('Close-pair condition ceiling (mean / max) = %.1f / %.1f\n', ...
            mean(trialClosePairConditionCeiling(:,iSNR)), ...
            max(trialClosePairConditionCeiling(:,iSNR)));
        fprintf('Nonlinear candidates refined (mean) = %.1f\n', ...
            mean(trialRefinedCandidateCount(:,iSNR)));
        fprintf('Estimator runtime mean / median / p95 = %.3f / %.3f / %.3f s\n', ...
            mean(trialEstimatorRuntimeSeconds(:,iSNR)), ...
            median(trialEstimatorRuntimeSeconds(:,iSNR)), ...
            percentile_simple(trialEstimatorRuntimeSeconds(:,iSNR),95));
        fprintf('Oracle-location gain NMSE (mean) = %.3e\n', ...
            mean(trialOracleGainNMSE(:,iSNR)));
        fprintf('Weakest true-path SNR (median / p05) = %.2f / %.2f dB\n', ...
            median(trialMinimumPathSNRdB(:,iSNR)), ...
            percentile_simple(trialMinimumPathSNRdB(:,iSNR),5));
        fprintf('Trials with condition > %.1f        = %.1f %%\n', ...
            conditionWarning,100*illConditionedRate(iSNR));
        fprintf('Measured SNR mean / 5th-95th pct     = %.2f / %.2f to %.2f dB\n', ...
            meanMeasuredSNR(iSNR),p05MeasuredSNR(iSNR),p95MeasuredSNR(iSNR));
        fprintf('Raw noise NMSE / estimator-error-noise ratio = %.3e / %.3e\n', ...
            meanRawNoiseNMSE(iSNR),meanEstimatorErrorToNoise(iSNR));
        fprintf('Oracle-location fit gap (mean / positive) = %.3f dB / %.1f %%\n', ...
            mean(trialOracleObjectiveGapdB(:,iSNR)), ...
            100*mean(trialOracleObjectiveGapdB(:,iSNR)>0));
        fprintf('Multi-hypothesis ambiguous / near-degenerate trials = %.1f / %.1f %%\n', ...
            100*mean(trialMultiHypothesisAmbiguous(:,iSNR)), ...
            100*mean(trialNearDegeneratePair(:,iSNR)));
        fprintf('Close-pair center/separation pattern starts (mean) = %.2f\n', ...
            mean(trialClosePairPatternStarts(:,iSNR)));
        fprintf('Close-pair center/separation NM starts (mean)      = %.2f\n', ...
            mean(trialClosePairNMStarts(:,iSNR)));
        maxLocalNuStd = max(trialLocalDopplerStdBins(:,:,iSNR),[],2);
        maxLocalGainRelStd = max(trialLocalGainRelativeStd(:,:,iSNR),[],2);
        fprintf('Local max Doppler std (median / p95) = %.3e / %.3e bins\n', ...
            median(maxLocalNuStd),percentile_simple(maxLocalNuStd,95));
        fprintf('Local max relative-gain std (median / p95) = %.3e / %.3e\n', ...
            median(maxLocalGainRelStd),percentile_simple(maxLocalGainRelStd,95));
        fprintf('Local gain-uncertainty warning rate = %.1f %%\n', ...
            100*mean(trialLocalGainUncertaintyFlag(:,iSNR)));
    end

    % Explicitly expose the high-SNR gain-error tail. The p95 is a standard
    % linearly interpolated sample percentile; with 50 trials it lies between
    % the 3rd- and 4th-worst observations, so the maximum can be much larger.
    lastSNR = numSNR;
    [~,worstOrder] = sort(trialGainNMSE(:,lastSNR),'descend');
    fprintf('\nWORST GAIN-ESTIMATION TRIALS AT %.1f dB\n',snrVec(lastSNR));
    fprintf('Trial | Gain NMSE | True sep | Est sep | Pair cond | Oracle cond | Ambiguous\n');
    for worstIndex = 1:min(5,numTrials)
        trialIndex = worstOrder(worstIndex);
        trueSep = abs(diff(trialTrueDoppler(trialIndex,:,lastSNR)));
        estSep = abs(diff(trialEstimatedDoppler(trialIndex,:,lastSNR)));
        fprintf('%5d | %9.3e | %8.5f | %7.5f | %9.2f | %11.2f | %d\n', ...
            trialIndex,trialGainNMSE(trialIndex,lastSNR),trueSep,estSep, ...
            trialPairCondition(trialIndex,lastSNR), ...
            trialOraclePairCondition(trialIndex,lastSNR), ...
            trialMultiHypothesisAmbiguous(trialIndex,lastSNR));
    end

    results.snr_dB = snrVec;
    results.doppler_rmse = dopplerRMSE;
    results.delay_rmse = delayRMSE;
    results.gain_nmse = gainNMSE;
    results.conditional_gain_nmse = conditionalGainNMSE;
    results.median_trial_gain_nmse = medianTrialGainNMSE;
    results.p90_trial_gain_nmse = p90TrialGainNMSE;
    results.p95_trial_gain_nmse = p95TrialGainNMSE;
    results.max_trial_gain_nmse = maxTrialGainNMSE;
    results.max_trial_gain_trial_index = maxTrialGainIndex;
    results.exact_delay_rate = exactDelayRate;
    results.path_location_recovery_rate = pathLocationRate;
    results.full_parameter_recovery_rate = fullParameterRecoveryRate;
    results.path_location_recovery_tolerances = reportingTolerances;
    results.path_location_recovery_by_tolerance = pathLocationRatesByTolerance;
    results.response_nmse = responseNMSE;
    results.response_nmse_std = responseNMSEStd;
    results.mean_pair_condition = meanPairCondition;
    results.p95_pair_condition = p95PairCondition;
    results.ill_conditioned_rate = illConditionedRate;
    results.mean_pair_coherence = meanPairCoherence;
    results.measured_snr_dB = meanMeasuredSNR;
    results.measured_snr_p05_dB = p05MeasuredSNR;
    results.measured_snr_p95_dB = p95MeasuredSNR;
    results.raw_noise_nmse = meanRawNoiseNMSE;
    results.estimator_error_to_noise_ratio = meanEstimatorErrorToNoise;
    results.doppler_tolerance = dopplerTolerance;
    results.gain_relative_error_tolerance = gainRelativeTolerance;
    results.condition_warning_threshold = conditionWarning;
    results.num_trials = numTrials;
    results.num_training_frames = get_cfg(cfg,'num_training_frames',1);
    results.observation_samples = numObservations;
    results.channel_bank = channelBank;
    results.trial_doppler_error = trialDopplerError;
    results.trial_delay_error = trialDelayError;
    results.trial_gain_nmse = trialGainNMSE;
    results.trial_response_nmse = trialResponseNMSE;
    results.trial_pair_condition = trialPairCondition;
    results.trial_pair_coherence = trialPairCoherence;
    results.trial_measured_snr_dB = trialMeasuredSNR;
    results.trial_raw_noise_nmse = trialRawNoiseNMSE;
    results.trial_estimator_error_to_noise_ratio = trialEstimatorErrorToNoise;
    results.trial_location_recovered = trialLocationRecovered;
    results.trial_full_recovered = trialFullRecovered;
    results.trial_true_doppler = trialTrueDoppler;
    results.trial_estimated_doppler = trialEstimatedDoppler;
    results.trial_true_delay = trialTrueDelay;
    results.trial_estimated_delay = trialEstimatedDelay;
    results.trial_true_gain = trialTrueGain;
    results.trial_estimated_gain = trialEstimatedGain;
    results.trial_refinement_source = trialRefinementSource;
    results.trial_lm_iterations = trialLMIterations;
    results.trial_lm_accepted_steps = trialLMAcceptedSteps;
    results.trial_lm_retry_calls = trialLMRetryCalls;
    results.trial_nelder_mead_starts = trialNelderMeadStarts;
    results.trial_nelder_mead_projected_endpoints = ...
        trialNelderMeadProjectedEndpoints;
    results.trial_oracle_objective_gap_dB = trialOracleObjectiveGapdB;
    results.trial_oracle_pair_condition = trialOraclePairCondition;
    results.trial_oracle_gain_nmse = trialOracleGainNMSE;
    results.trial_minimum_path_snr_dB = trialMinimumPathSNRdB;
    results.trial_max_pair_condition_used = trialMaxPairConditionUsed;
    results.trial_close_pair_condition_ceiling = trialClosePairConditionCeiling;
    results.trial_estimated_observation_snr_dB = trialEstimatedObservationSNRdB;
    results.trial_valid_pair_count = trialValidPairCount;
    results.trial_newly_admitted_pair_count = trialNewlyAdmittedPairCount;
    results.trial_close_pair_seed_count = trialClosePairSeedCount;
    results.trial_residual_seed_count = trialResidualSeedCount;
    results.trial_coarse_seed_count = trialCoarseSeedCount;
    results.trial_refined_candidate_count = trialRefinedCandidateCount;
    results.trial_estimator_runtime_seconds = trialEstimatorRuntimeSeconds;
    results.mean_estimator_runtime_seconds = mean(trialEstimatorRuntimeSeconds,1);
    results.total_estimator_runtime_seconds = sum(trialEstimatorRuntimeSeconds,1);
    results.trial_ambiguity_flag = trialAmbiguityFlag;
    results.trial_multi_hypothesis_ambiguous = trialMultiHypothesisAmbiguous;
    results.trial_near_degenerate_pair = trialNearDegeneratePair;
    results.trial_ambiguity_delta_log_likelihood = ...
        trialAmbiguityDeltaLogLikelihood;
    results.trial_ambiguity_alternative_condition = ...
        trialAmbiguityAlternativeCondition;
    results.trial_ambiguity_competitor_count = trialAmbiguityCompetitorCount;
    results.trial_ambiguity_alternative_doppler = ...
        trialAmbiguityAlternativeDoppler;
    results.trial_ambiguity_alternative_delay = ...
        trialAmbiguityAlternativeDelay;
    results.trial_ambiguity_alternative_source = ...
        trialAmbiguityAlternativeSource;
    results.trial_close_pair_nm_starts = trialClosePairNMStarts;
    results.trial_close_pair_pattern_starts = trialClosePairPatternStarts;
    results.trial_local_doppler_std_bins = trialLocalDopplerStdBins;
    results.trial_local_gain_relative_std = trialLocalGainRelativeStd;
    results.trial_local_information_condition = trialLocalInformationCondition;
    results.trial_local_gain_uncertainty_flag = trialLocalGainUncertaintyFlag;
    results.multi_hypothesis_ambiguity_rate = mean(trialMultiHypothesisAmbiguous,1);
    results.near_degenerate_pair_rate = mean(trialNearDegeneratePair,1);

    make_summary_plots(snrVec,results);
end

function [objective,gramCondition,oracleGains,pathSNRdB] = ...
    oracle_location_fit(pilotTx,Ynoisy,trueDoppler,trueDelay,trueGain, ...
    noiseVariance,cfg)
% ORACLE_LOCATION_FIT
% Evaluation-only baseline: known path locations, unknown LS gains.
% Also returns the LS gain estimate and per-path observability SNR.
% This function must never be called from the estimator's decision path.
    y = Ynoisy(:);
    numPaths = numel(trueDoppler);
    B = complex(zeros(numel(y),numPaths));
    rawNorms = zeros(numPaths,1);
    for p = 1:numPaths
        onePath.delay = trueDelay(p);
        onePath.doppler = trueDoppler(p);
        onePath.gain = 1;
        rx = apply_fractional_doppler_channel(pilotTx,onePath,cfg);
        atom = demodulate_training_sequence(rx,cfg);
        atom = atom(:);
        rawNorms(p) = norm(atom);
        if rawNorms(p) <= eps
            error('Oracle diagnostic generated a zero-norm path atom.');
        end
        B(:,p) = atom/rawNorms(p);
    end
    coefficients = B\y;
    residual = y-B*coefficients;
    objective = real(residual'*residual);
    gramCondition = cond(B'*B);
    oracleGains = coefficients(:)./rawNorms;

    if noiseVariance > 0
        pathAveragePower = abs(trueGain(:)).^2 .* rawNorms.^2 / numel(y);
        pathSNR = pathAveragePower/noiseVariance;
        pathSNRdB = 10*log10(max(pathSNR,realmin));
    else
        pathSNRdB = Inf(numPaths,1);
    end
end

function cost = assignment_cost(trueDoppler,trueDelay,estDoppler,estDelay,nuMax)
% Assignment must be based on path location, not the gains being evaluated.
    dopplerScale = max(nuMax,0.1);
    delayWeight = 4;
    cost = sum(((trueDoppler-estDoppler)/dopplerScale).^2) + ...
        delayWeight*sum((trueDelay-estDelay).^2);
end

function value = percentile_simple(x,percentile)
% Standard Type-7 linearly interpolated sample percentile (MATLAB prctile default).
    x = sort(x(:));
    if isempty(x)
        value = NaN;
        return;
    end
    position = 1+(numel(x)-1)*percentile/100;
    lowerIndex = floor(position);
    upperIndex = ceil(position);
    if lowerIndex == upperIndex
        value = x(lowerIndex);
    else
        value = x(lowerIndex)+(position-lowerIndex)* ...
            (x(upperIndex)-x(lowerIndex));
    end
end

function make_summary_plots(snrVec,results)
    figure('Name','Physical path-estimation performance');
    plot(snrVec,results.doppler_rmse,'-o','LineWidth',1.5);
    grid on; xlabel('SNR (dB)'); ylabel('Doppler RMSE (bins)');
    title('Fractional-Doppler Parameter Estimation');

    figure('Name','Delay estimation performance');
    plot(snrVec,results.delay_rmse,'-o','LineWidth',1.5);
    grid on; xlabel('SNR (dB)'); ylabel('Delay RMSE (samples)');
    title('Integer-Sample Delay Estimation');

    figure('Name','Physical complex-gain estimation');
    semilogy(snrVec,results.gain_nmse,'-o','LineWidth',1.5);
    hold on;
    semilogy(snrVec,results.conditional_gain_nmse,'--s','LineWidth',1.5);
    grid on; xlabel('SNR (dB)'); ylabel('Complex-gain NMSE');
    legend('All trials','Correctly localized trials','Location','best');
    title('Physical Complex Path-Gain Estimation');

    figure('Name','Complete response reconstruction');
    plot(snrVec,10*log10(results.response_nmse+eps),'-o','LineWidth',1.5);
    grid on; xlabel('SNR (dB)'); ylabel('Response NMSE (dB)');
    title('Complete DD Response Reconstruction');

    figure('Name','Path recovery at multiple tolerances');
    hold on;
    for k = 1:numel(results.path_location_recovery_tolerances)
        plot(snrVec,100*results.path_location_recovery_by_tolerance(k,:), ...
            '-o','LineWidth',1.2);
    end
    grid on; ylim([0 100]); xlabel('SNR (dB)'); ylabel('Recovery rate (%)');
    labels = arrayfun(@(x)sprintf('Doppler tol. %.3f',x), ...
        results.path_location_recovery_tolerances,'UniformOutput',false);
    legend(labels,'Location','best');
    title('Path-Location Recovery Versus Doppler Tolerance');

    figure('Name','Selected-pair conditioning');
    semilogy(snrVec,results.mean_pair_condition,'-o','LineWidth',1.5);
    hold on;
    semilogy(snrVec,results.p95_pair_condition,'--s','LineWidth',1.5);
    grid on; xlabel('SNR (dB)'); ylabel('Gram condition number');
    legend('Mean','95th percentile','Location','best');
    title('Selected Dictionary-Pair Conditioning');
end

function value = get_cfg(cfg,fieldName,defaultValue)
    if isfield(cfg,fieldName) && ~isempty(cfg.(fieldName))
        value = cfg.(fieldName);
    else
        value = defaultValue;
    end
end
