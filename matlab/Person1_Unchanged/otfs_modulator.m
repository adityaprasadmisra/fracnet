
function [s, X_tf] = otfs_modulator(X_dd, params)
% ==========================================================
% OTFS FRACTIONAL DOPPLER CHANNEL ESTIMATION PROJECT
% OTFS Modulation: ISFFT + Heisenberg Transform
% ==========================================================

N = params.N;
M = params.M;

% Check DD grid dimensions
if ~isequal(size(X_dd), [N, M])
    error('Input DD grid must be N x M.');
end

%% 1. ISFFT
X_tf = sqrt(N/M) * fft(ifft(X_dd, [], 1), [], 2);

%% 2. Heisenberg Transform
x_time_matrix = sqrt(M) * ifft(X_tf, [], 2);

%% 3. Serialize time-domain samples
s = reshape(x_time_matrix.', [], 1);

end
