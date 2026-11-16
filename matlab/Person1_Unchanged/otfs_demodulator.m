  
function [Y_dd, Y_tf] = otfs_demodulator(r, params)
% ==========================================================
% OTFS FRACTIONAL DOPPLER CHANNEL ESTIMATION PROJECT
% OTFS Demodulation: Wigner + SFFT
% ==========================================================

N = params.N;
M = params.M;

% Convert received signal into a column vector
r = r(:);

% Check received signal length
if length(r) ~= N*M
    error('Received signal must contain N*M samples.');
end

%% 1. Reconstruct time-domain matrix
% Reverse the serialization performed in the modulator
r_matrix = reshape(r, M, N).';

%% 2. Wigner Transform
% Rectangular receive pulse
Y_tf = fft(r_matrix, [], 2) / sqrt(M);

%% 3. SFFT: Time-Frequency to Delay-Doppler
Y_dd = sqrt(M/N) * fft(ifft(Y_tf, [], 2), [], 1);

end
