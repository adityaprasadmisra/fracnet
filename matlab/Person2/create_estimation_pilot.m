function Xpilot = create_estimation_pilot(cfg)
% CREATE_ESTIMATION_PILOT
% Create K deterministic, independent full-frame known QPSK training grids.
% Each frame is modulated separately; the resulting time-domain frames are
% concatenated and passed through one continuous time-varying channel.
%
% This is an identification/validation waveform. It is NOT the paper's
% single-cell superimposed-pilot overhead design.
% The global MATLAB random-number state is restored before returning.

    if nargin < 1 || isempty(cfg)
        cfg = system_parameters();
    end
    if ~isfield(cfg,'pilot_seed')
        error('cfg.pilot_seed is required.');
    end
    if ~isfield(cfg,'num_training_frames') || cfg.num_training_frames < 1 || ...
            cfg.num_training_frames ~= floor(cfg.num_training_frames)
        error('cfg.num_training_frames must be a positive integer.');
    end

    previousState = rng;
    restoreState = onCleanup(@() rng(previousState)); %#ok<NASGU>
    rng(cfg.pilot_seed,'twister');

    K = cfg.num_training_frames;
    Xpilot = complex(zeros(cfg.N,cfg.M,K));
    for k = 1:K
        bits = randi([0 1],cfg.N*cfg.M,2);
        qpskSymbols = ((1-2*bits(:,1)) + 1j*(1-2*bits(:,2)))/sqrt(2);
        Xpilot(:,:,k) = cfg.pilot_amplitude * reshape(qpskSymbols,cfg.N,cfg.M);
    end
end
