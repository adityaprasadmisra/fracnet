
function data = generate_fracnet_dataset(outputFile, numSamples, snrMode, seed)
% GENERATE_FRACNET_DATASET
% Generate estimator-compatible labelled OTFS data for FracNet.
%
% Supervised learning:
%   Input  = H_est  (coarse channel estimate)
%   Target = Yclean (clean received DD response)
%
% Usage:
%   D = generate_fracnet_dataset('fracnet_generated.mat',100,'balanced',1234);
%
% snrMode:
%   'random'   : Random SNR from 0:2:20 dB
%   'balanced' : Cycle through 0:2:20 dB
%   numeric    : Fixed SNR
%
% Outputs:
%   Xpilot         : N x M x K x S
%   Yclean         : N x M x K x S
%   Ynoisy         : N x M x K x S
%   H_est          : N x M x K x S
%   path_params    : 4 x P x S
%   path_doppler   : P x S
%   path_delay     : P x S
%   path_gain      : P x S
%   SNR_dB         : 1 x S
%   noise_variance : 1 x S
%   sample_id      : 1 x S

    %% 1. Default arguments

    if nargin < 1 || isempty(outputFile)
        outputFile = 'fracnet_generated.mat';
    end

    if nargin < 2 || isempty(numSamples)
        numSamples = 1000;
    end

    if nargin < 3 || isempty(snrMode)
        snrMode = 'random';
    end

    if nargin < 4 || isempty(seed)
        seed = 1234;
    end

    if numSamples < 1 || numSamples ~= floor(numSamples)
        error('numSamples must be a positive integer.');
    end

    %% 2. Random seed

    previousRng = rng;
    restoreRng = onCleanup(@() rng(previousRng));
    rng(seed,'twister');

    %% 3. System configuration

    cfg = system_parameters();
    cfg.num_training_frames = 4;

    K = cfg.num_training_frames;
    N = cfg.N;
    M = cfg.M;
    P = cfg.P;

    snrLevels = 0:2:20;

    fprintf('\n====================================\n');
    fprintf(' FRACNET DATASET GENERATION\n');
    fprintf('====================================\n');

    fprintf('Samples       : %d\n',numSamples);
    fprintf('Grid          : %d x %d\n',N,M);
    fprintf('Training frames: %d\n',K);
    fprintf('Paths          : %d\n',P);
    fprintf('Doppler range  : +/- %.4f bins\n',cfg.nu_max);

    %% 4. Generate one fixed pilot and dictionary

    fprintf('\nGenerating fixed training pilot...\n');

    % Same deterministic pilot used by Person 2's test pipeline
    Xfixed = create_estimation_pilot(cfg);

    % Build the dictionary once for this pilot
    [dopplerGrid, delayGrid] = estimation_grids(cfg);

    fprintf('Building fractional-Doppler dictionary...\n');

    [A, atomNorms, pairCache] = ...
        fractional_doppler_dictionary( ...
        Xfixed, cfg, dopplerGrid, delayGrid);

    fprintf('Pilot size      : %s\n',mat2str(size(Xfixed)));
    fprintf('Dictionary size : %s\n',mat2str(size(A)));

    %% 5. Preallocate arrays

    Xpilot = complex(zeros(N,M,K,numSamples,'single'));
    Yclean = complex(zeros(N,M,K,numSamples,'single'));
    Ynoisy = complex(zeros(N,M,K,numSamples,'single'));
    H_est = complex(zeros(N,M,K,numSamples,'single'));

    path_params = zeros(4,P,numSamples,'single');

    path_doppler = zeros(P,numSamples,'single');
    path_delay = zeros(P,numSamples,'int16');

    path_gain = complex(zeros(P,numSamples,'single'));

    SNR_dB = zeros(1,numSamples,'single');
    noise_variance = zeros(1,numSamples,'single');

    sample_id = 1:numSamples;

    %% 6. Generate samples

    for s = 1:numSamples

        fprintf('Generating sample %d / %d\n',s,numSamples);

        %% 6.1 Use the same fixed pilot

        X = Xfixed;

        %% 6.2 Generate P-path channel

        channel.delay = randi([0 cfg.l_max],1,P);

        if rand < 0.20

            center = (2*rand-1)*(cfg.nu_max-0.06);

            separation = 0.005 + ...
                (0.12-0.005)*rand;

            channel.doppler = ...
                [center-separation/2, ...
                 center+separation/2];

            if rand < 0.5
                channel.doppler = fliplr(channel.doppler);
            end

            channel.doppler = min( ...
                max(channel.doppler,-cfg.nu_max), ...
                cfg.nu_max);

        else

            channel.doppler = ...
                (2*rand(1,P)-1)*cfg.nu_max;

        end

        channel.doppler_hz = ...
            channel.doppler*cfg.doppler_resolution;

        channel.doppler_integer = ...
            round(channel.doppler);

        channel.doppler_fraction = ...
            channel.doppler-channel.doppler_integer;

        %% 6.3 Generate complex path gains

        channel.gain = ...
            (randn(1,P)+1i*randn(1,P))/sqrt(2);

        channel.gain = channel.gain/norm(channel.gain);

        %% 6.4 Canonical path ordering

        [~,order] = sortrows( ...
            [channel.delay(:),channel.doppler(:)],[1 2]);

        channel.delay = channel.delay(order);
        channel.doppler = channel.doppler(order);
        channel.doppler_hz = channel.doppler_hz(order);

        channel.doppler_integer = ...
            channel.doppler_integer(order);

        channel.doppler_fraction = ...
            channel.doppler_fraction(order);

        channel.gain = channel.gain(order);

        %% 6.5 OTFS modulation

        tx = modulate_training_sequence(X,cfg);

        %% 6.6 Apply fractional-Doppler channel

        rxClean = apply_fractional_doppler_channel( ...
            tx,channel,cfg);

        %% 6.7 OTFS demodulation

        Yc = demodulate_training_sequence(rxClean,cfg);

        %% 6.8 Select SNR

        if ischar(snrMode) || isstring(snrMode)

            mode = char(snrMode);

            if strcmpi(mode,'random')

                currentSNR = ...
                    snrLevels(randi(numel(snrLevels)));

            elseif strcmpi(mode,'balanced')

                currentSNR = ...
                    snrLevels(mod(s-1,numel(snrLevels))+1);

            else

                error('snrMode must be random, balanced, or numeric.');

            end

        elseif isnumeric(snrMode) && ...
                isscalar(snrMode) && isfinite(snrMode)

            currentSNR = double(snrMode);

        else

            error('Invalid snrMode.');

        end

        %% 6.9 Add AWGN

        signalPower = mean(abs(rxClean).^2);

        noiseVar = signalPower/10^(currentSNR/10);

        noiseTime = sqrt(noiseVar)* ...
            (randn(size(rxClean))+ ...
             1i*randn(size(rxClean)))/sqrt(2);

        rxNoisy = rxClean+noiseTime;

        Yn = demodulate_training_sequence(rxNoisy,cfg);

        %% 6.10 Run Person 2 coarse channel estimator

        [H, detectedPaths, gains, residual] = ...
            fractional_doppler_coarse_estimation( ...
            Yn, A, dopplerGrid, delayGrid, cfg, ...
            atomNorms, pairCache, noiseVar);

        %#ok<NASGU>
        % detectedPaths, gains and residual are retained as estimator
        % outputs for debugging; H is the supervised input.

        %% 6.11 Store observations

        Xpilot(:,:,:,s) = single(X);

        Yclean(:,:,:,s) = single(Yc);

        Ynoisy(:,:,:,s) = single(Yn);

        H_est(:,:,:,s) = single(H);

        %% 6.12 Store channel labels

        path_params(1,:,s) = single(channel.doppler);

        path_params(2,:,s) = single(channel.delay);

        path_params(3,:,s) = single(real(channel.gain));

        path_params(4,:,s) = single(imag(channel.gain));

        path_doppler(:,s) = single(channel.doppler(:));

        path_delay(:,s) = int16(channel.delay(:));

        path_gain(:,s) = single(channel.gain(:));

        SNR_dB(s) = single(currentSNR);

        noise_variance(s) = single(noiseVar);

    end

    %% 7. Assemble dataset

    data.Xpilot = Xpilot;
    data.Yclean = Yclean;
    data.Ynoisy = Ynoisy;
    data.H_est = H_est;

    data.path_params = path_params;

    data.path_doppler = path_doppler;
    data.path_delay = path_delay;
    data.path_gain = path_gain;

    data.SNR_dB = SNR_dB;
    data.noise_variance = noise_variance;

    data.sample_id = sample_id;

    data.dopplerGrid = dopplerGrid;
    data.delayGrid = delayGrid;

    %% 8. Save dataset

    save(outputFile,'-struct','data','-v7.3');

    fprintf('\n====================================\n');
    fprintf(' DATASET GENERATION COMPLETED\n');
    fprintf('====================================\n');

    fprintf('Samples saved : %d\n',numSamples);
    fprintf('Output file   : %s\n',outputFile);

    fprintf('Xpilot shape  : %s\n',mat2str(size(Xpilot)));
    fprintf('Yclean shape  : %s\n',mat2str(size(Yclean)));
    fprintf('Ynoisy shape  : %s\n',mat2str(size(Ynoisy)));
    fprintf('H_est shape   : %s\n',mat2str(size(H_est)));

    fprintf('\nDataset saved successfully!\n');

end

