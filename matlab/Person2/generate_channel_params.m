function channel = generate_channel_params(cfg)
% GENERATE_CHANNEL_PARAMS
% Generate a P-path channel using the same convention as
% apply_fractional_doppler_channel.m.
%
% channel.delay  : integer sample delays
% channel.doppler: total normalized Doppler in Doppler-bin units
% channel.gain   : normalized complex path gains

    required = {'P', 'l_max', 'fd_max', 'doppler_resolution'};
    for k = 1:numel(required)
        if ~isfield(cfg, required{k})
            error('Missing cfg.%s.', required{k});
        end
    end

    P = cfg.P;
    if P < 1 || P ~= floor(P)
        error('cfg.P must be a positive integer.');
    end

    channel.delay = randi([0 cfg.l_max], 1, P);

    % Sample physical Doppler uniformly within the configured limit.
    channel.doppler_hz = (2*rand(1, P) - 1) * cfg.fd_max;
    channel.doppler = channel.doppler_hz / cfg.doppler_resolution;

    % Integer/fractional decomposition is retained for reporting.
    channel.doppler_integer = round(channel.doppler);
    channel.doppler_fraction = ...
        channel.doppler - channel.doppler_integer;

    % Independent circular complex Gaussian gains, normalized to unit
    % total channel power for consistent Monte Carlo comparisons.
    channel.gain = (randn(1, P) + 1j*randn(1, P)) / sqrt(2);
    gainNorm = norm(channel.gain);
    if gainNorm <= eps
        error('Generated channel gains have zero norm.');
    end
    channel.gain = channel.gain / gainNorm;
end
