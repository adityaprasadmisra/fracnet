
function bits = generate_bits(params)
% ==========================================================
% OTFS FRACTIONAL DOPPLER CHANNEL ESTIMATION PROJECT
% Random Binary Data Generation
% ==========================================================

% Total number of bits required
num_bits = params.num_symbols * params.bits_per_symbol;

% Generate random binary data (0s and 1s)
bits = randi([0 1], num_bits, 1);

% Display generated bits
disp('Generated Binary Data:');
disp(bits.');

% Display dimensions
fprintf('\nNumber of generated bits: %d\n', length(bits));
fprintf('Size of bit vector: %d x %d\n', size(bits,1), size(bits,2));

end
