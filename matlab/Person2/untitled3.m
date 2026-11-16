D = load('fracnet_generated.mat');

size(D.Xpilot)
size(D.Yclean)
size(D.Ynoisy)

D = load('fracnet_generated.mat');

fprintf('NaN in Xpilot: %d\n', sum(isnan(D.Xpilot(:))));
fprintf('NaN in Yclean: %d\n', sum(isnan(D.Yclean(:))));
fprintf('NaN in Ynoisy: %d\n', sum(isnan(D.Ynoisy(:))));

fprintf('Inf in Xpilot: %d\n', sum(isinf(D.Xpilot(:))));
fprintf('Inf in Yclean: %d\n', sum(isinf(D.Yclean(:))));
fprintf('Inf in Ynoisy: %d\n', sum(isinf(D.Ynoisy(:))));

fprintf('\nSNR values:\n');
cd('E:\fracnet\FracNet\matlab\Person2')

dbtype test_channel_estimator 1:180

dbtype estimation_grids


dbtype create_estimation_pilot