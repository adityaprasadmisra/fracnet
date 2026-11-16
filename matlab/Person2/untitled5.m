cd('E:\fracnet\FracNet\matlab\Person2')

data = generate_fracnet_dataset( ...
    'fracnet_train.mat', 100, 'balanced', 1234);

check = load('fracnet_test_fixed.mat');

disp(size(check.Xpilot));
disp(size(check.H_est));
disp(size(check.Yclean));

fprintf('H_est NaN: %d\n',sum(isnan(check.H_est(:))));
fprintf('H_est Inf: %d\n',sum(isinf(check.H_est(:))));

fprintf('Yclean NaN: %d\n',sum(isnan(check.Yclean(:))));
fprintf('Yclean Inf: %d\n',sum(isinf(check.Yclean(:))));

fprintf('Unique pilot count: %d\n', ...
    size(unique(reshape(check.Xpilot,[],10).','rows'),1));

clear check

check = load('fracnet_train.mat');

fprintf('H_est size: ');
disp(size(check.H_est));

fprintf('Yclean size: ');
disp(size(check.Yclean));

fprintf('H_est NaN: %d\n', sum(isnan(check.H_est(:))));
fprintf('H_est Inf: %d\n', sum(isinf(check.H_est(:))));

fprintf('Yclean NaN: %d\n', sum(isnan(check.Yclean(:))));
fprintf('Yclean Inf: %d\n', sum(isinf(check.Yclean(:))));

fprintf('SNR distribution:\n');
disp(unique(check.SNR_dB));