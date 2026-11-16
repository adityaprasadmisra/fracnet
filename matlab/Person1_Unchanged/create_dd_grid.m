
function X_dd = create_dd_grid(combined_symbols, params)
% ==========================================================
% OTFS FRACTIONAL DOPPLER CHANNEL ESTIMATION PROJECT
% Delay-Doppler Grid Construction
% ==========================================================

% Check input length
if length(combined_symbols) ~= params.num_symbols
    error('Number of input symbols must equal N*M.');
end

% Arrange symbols into an N x M matrix
X_dd = reshape(combined_symbols, params.N, params.M);

% Display grid dimensions
fprintf('Delay-Doppler grid size: %d x %d\n', ...
    size(X_dd,1), size(X_dd,2));

end
