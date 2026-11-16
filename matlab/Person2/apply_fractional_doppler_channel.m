function received_signal = apply_fractional_doppler_channel(transmitted_signal, channel, cfg)
% APPLY_FRACTIONAL_DOPPLER_CHANNEL
% Apply an integer-delay, fractional-Doppler multipath channel to a sample
% stream of arbitrary positive length.
%
% For a concatenation of OTFS frames, the Doppler phase uses the GLOBAL
% sample index n=0,...,Lstream-1. Thus phase is continuous across frame
% boundaries, and a delayed sample at a frame boundary correctly comes
% from the preceding frame. The original one-frame behavior is unchanged.
%
% r[n] = sum_p h[p] s[n-delay[p]] exp(j*2*pi*fd[p]*n*Ts)
% channel.doppler is in normalized Doppler-bin units.

    s = transmitted_signal(:);
    if isempty(s)
        error('transmitted_signal must not be empty.');
    end

    if ~isstruct(cfg) || ~isfield(cfg,'Ts') || ~isfield(cfg,'N') || ...
            ~isfield(cfg,'T') || ~isscalar(cfg.Ts) || ~isfinite(cfg.Ts) || ...
            cfg.Ts<=0 || ~isscalar(cfg.N) || ~isfinite(cfg.N) || cfg.N<1 || ...
            ~isscalar(cfg.T) || ~isfinite(cfg.T) || cfg.T<=0
        error('cfg must contain valid positive Ts, N and T fields.');
    end
    Ts = cfg.Ts;
    L = numel(s);

    if ~isstruct(channel) || ~isfield(channel,'gain') || ~isfield(channel,'delay')
        error('Channel must be a structure containing gain and delay fields.');
    end
    if isfield(channel,'doppler')
        dopplerIndices = channel.doppler;
    elseif isfield(channel,'doppler_integer') && isfield(channel,'doppler_fraction')
        dopplerIndices = channel.doppler_integer + channel.doppler_fraction;
    else
        error('Channel must contain doppler or integer/fractional Doppler fields.');
    end

    if ~isnumeric(channel.gain) || isempty(channel.gain) || ...
            ~isnumeric(channel.delay) || ~isnumeric(dopplerIndices) || ...
            any(~isfinite(channel.gain(:))) || ...
            any(~isfinite(channel.delay(:))) || any(~isfinite(dopplerIndices(:)))
        error('Channel gain, delay and Doppler must be finite numeric arrays.');
    end
    P = numel(channel.gain);
    if numel(channel.delay) ~= P || numel(dopplerIndices) ~= P
        error('Channel gain, delay and Doppler arrays must have equal lengths.');
    end

    dopplerResolution = 1/(cfg.N*cfg.T);
    n = (0:L-1).';
    received_signal = complex(zeros(L,1));

    for p = 1:P
        delaySamples = channel.delay(p);
        if ~isscalar(delaySamples) || ~isfinite(delaySamples) || ...
                delaySamples < 0 || delaySamples ~= floor(delaySamples)
            error('Each delay must be a nonnegative integer number of samples.');
        end
        if delaySamples >= L
            continue;
        end
        delayedSignal = complex(zeros(L,1));
        validIndices = (delaySamples+1):L;
        delayedSignal(validIndices) = s(1:L-delaySamples);

        fd = dopplerIndices(p)*dopplerResolution;
        received_signal = received_signal + channel.gain(p) * ...
            delayedSignal .* exp(1j*2*pi*fd*n*Ts);
    end
end
