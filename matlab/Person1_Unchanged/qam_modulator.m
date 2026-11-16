
function symbols = qam_modulator(bits, params)
% ==========================================================
% OTFS FRACTIONAL DOPPLER CHANNEL ESTIMATION PROJECT
% 4-QAM Modulation
% ==========================================================

% Reshape bits into pairs
bit_pairs = reshape(bits, params.bits_per_symbol, []).';

% Separate the two bits
b1 = bit_pairs(:,1);
b2 = bit_pairs(:,2);

% Convert bit pairs into 4-QAM complex symbols
symbols = ((1 - 2*b1) + 1j*(1 - 2*b2)) / sqrt(2);

% Apply data amplitude
symbols = params.data_amplitude * symbols;

% Display results
disp('First 10 transmitted 4-QAM symbols:');
disp(symbols(1:10));

fprintf('Total number of 4-QAM symbols: %d\n', length(symbols));


end
