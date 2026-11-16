% Load the raw dataset
data = load('fracnet_generated.mat');

% Extract one sample
Ynoisy = data.Ynoisy(:,:,:,1);
Yclean = data.Yclean(:,:,:,1);

% Use the dictionary already generated
noiseVariance = data.noise_variance(1);

% Run coarse channel estimation
[H_est, detectedPaths, gains, residual] = ...
    fractional_doppler_coarse_estimation( ...
    Ynoisy, A, dopplerGrid, delayGrid, cfg, ...
    atomNorms, pairCache, noiseVariance);

% Inspect dimensions
fprintf('Ynoisy size: ');
disp(size(Ynoisy));

fprintf('Yclean size: ');
disp(size(Yclean));

fprintf('H_est size: ');
disp(size(H_est));

fprintf('H_est NaN count: %d\n', sum(isnan(H_est(:))));
fprintf('H_est Inf count: %d\n', sum(isinf(H_est(:))));