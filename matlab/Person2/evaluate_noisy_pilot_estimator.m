function [nmse_dB, nmse_linear, results] = evaluate_noisy_pilot_estimator( ...
    Xpilot, cfg, dopplerGrid, delayGrid, snrVec, numTrials, dictionary)
% EVALUATE_NOISY_PILOT_ESTIMATOR
% Controlled Monte Carlo evaluation of complete DD-response reconstruction.
%
% The same channel realizations are reused for every SNR; only AWGN changes.
% This avoids confounding the SNR curve with different channel difficulty.
% The reference is always the noiseless response to the same known grid.

    if nargin < 6 || isempty(numTrials)
        numTrials = cfg.test_num_trials;
    end
    if nargin < 7 || isempty(dictionary)
        [A, atomNorms, pairCache] = fractional_doppler_dictionary( ...
            Xpilot, cfg, dopplerGrid, delayGrid);
    else
        A = dictionary.A;
        atomNorms = dictionary.atomNorms;
        pairCache = dictionary.pairCache;
    end
    if numTrials < 1 || numTrials ~= floor(numTrials)
        error('numTrials must be a positive integer.');
    end

    pilotTx = modulate_training_sequence(Xpilot,cfg);
    numObservations = numel(pilotTx);

    originalRng = rng;
    restoreRng = onCleanup(@() rng(originalRng)); %#ok<NASGU>
    rng(cfg.random_seed, 'twister');
    channelBank = cell(numTrials, 1);
    for trial = 1:numTrials
        channelBank{trial} = generate_channel_params(cfg);
    end

    noiseSeed = cfg.noise_seed;
    noiseBank = cell(numTrials,1);
    for trial = 1:numTrials
        rng(noiseSeed+trial,'twister');
        noiseBank{trial} = (randn(numObservations,1) + ...
            1i*randn(numObservations,1))/sqrt(2);
    end

    nmse_linear = zeros(size(snrVec));
    nmse_std = zeros(size(snrVec));
    meanResidualPower = zeros(size(snrVec));
    meanMeasuredSNR = zeros(size(snrVec));
    meanRawNoiseNMSE = zeros(size(snrVec));
    trialNMSE = zeros(numTrials, numel(snrVec));
    trialResidualPower = zeros(numTrials, numel(snrVec));
    trialMeasuredSNR = zeros(numTrials,numel(snrVec));
    trialRawNoiseNMSE = zeros(numTrials,numel(snrVec));

    for iSNR = 1:numel(snrVec)
        currentSNR = snrVec(iSNR);
        sumErrorPower = 0;
        sumReferencePower = 0;
        measuredSNRThisPoint = zeros(numTrials,1);
        rawNoiseNMSEThisPoint = zeros(numTrials,1);
        fprintf('\nResponse-estimation evaluation: SNR = %g dB\n', currentSNR);

        for trial = 1:numTrials
            channel = channelBank{trial};
            rClean = apply_fractional_doppler_channel(pilotTx, channel, cfg);
            Yclean = demodulate_training_sequence(rClean,cfg);
            signalPower = mean(abs(rClean(:)).^2);
            noiseVariance = signalPower / 10^(currentSNR/10);

            noiseTime = sqrt(noiseVariance)*noiseBank{trial};
            rNoisy = rClean + noiseTime;
            Ynoisy = demodulate_training_sequence(rNoisy,cfg);
            noiseDD = demodulate_training_sequence(noiseTime,cfg);
            trialMeasuredSNR(trial,iSNR) = 10*log10( ...
                sum(abs(rClean(:)).^2)/(sum(abs(noiseTime(:)).^2)+eps));
            trialRawNoiseNMSE(trial,iSNR) = norm(noiseDD(:))^2/ ...
                (norm(Yclean(:))^2+eps);
            measuredSNRThisPoint(trial) = trialMeasuredSNR(trial,iSNR);
            rawNoiseNMSEThisPoint(trial) = trialRawNoiseNMSE(trial,iSNR);

            [H_est, ~, ~, residual] = fractional_doppler_coarse_estimation( ...
                Ynoisy, A, dopplerGrid, delayGrid, cfg, ...
                atomNorms, pairCache, noiseVariance);

            errorPower = norm(H_est(:)-Yclean(:))^2;
            referencePower = norm(Yclean(:))^2;
            trialNMSE(trial,iSNR) = errorPower / (referencePower + eps);
            trialResidualPower(trial,iSNR) = norm(residual(:))^2 / ...
                (norm(Ynoisy(:))^2 + eps);
            sumErrorPower = sumErrorPower + errorPower;
            sumReferencePower = sumReferencePower + referencePower;
        end

        nmse_linear(iSNR) = sumErrorPower / (sumReferencePower + eps);
        nmse_std(iSNR) = std(trialNMSE(:,iSNR));
        meanResidualPower(iSNR) = mean(trialResidualPower(:,iSNR));
        meanMeasuredSNR(iSNR) = mean(measuredSNRThisPoint);
        meanRawNoiseNMSE(iSNR) = mean(rawNoiseNMSEThisPoint);
        fprintf('Pooled response NMSE = %.6e (%.2f dB); trial SD = %.3e\n', ...
            nmse_linear(iSNR), 10*log10(nmse_linear(iSNR)+eps), ...
            nmse_std(iSNR));
        fprintf('Measured SNR = %.2f dB; raw noise NMSE = %.3e\n', ...
            meanMeasuredSNR(iSNR),meanRawNoiseNMSE(iSNR));
    end

    nmse_dB = 10*log10(nmse_linear + eps);
    results.snr_dB = snrVec;
    results.nmse_linear = nmse_linear;
    results.nmse_dB = nmse_dB;
    results.nmse_std = nmse_std;
    results.mean_residual_power = meanResidualPower;
    results.measured_snr_dB = meanMeasuredSNR;
    results.raw_noise_nmse = meanRawNoiseNMSE;
    results.trial_nmse = trialNMSE;
    results.trial_residual_power = trialResidualPower;
    results.trial_measured_snr_dB = trialMeasuredSNR;
    results.trial_raw_noise_nmse = trialRawNoiseNMSE;
    results.channel_bank = channelBank;
    results.noise_bank = noiseBank;
    results.num_trials = numTrials;
    results.num_training_frames = cfg.num_training_frames;
    results.observation_samples = numObservations;

    figure('Name', 'Pilot-response NMSE');
    plot(snrVec, nmse_dB, '-o', 'LineWidth', 1.5);
    grid on;
    xlabel('SNR (dB)');
    ylabel('Pilot-response NMSE (dB)');
    title('Joint Two-Path Fractional-Doppler Estimation');
end
