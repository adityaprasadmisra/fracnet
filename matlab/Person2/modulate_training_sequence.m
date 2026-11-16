function txStream = modulate_training_sequence(Xpilot, cfg)
% MODULATE_TRAINING_SEQUENCE
% Modulate K known OTFS DD training frames and concatenate them in time.
% The output is a single continuous sample stream, frame 1 followed by
% frame 2, ... . The time-varying channel therefore has a continuous
% Doppler phase across frame boundaries.

    L = cfg.N * cfg.M;
    if size(Xpilot,1) ~= cfg.N || size(Xpilot,2) ~= cfg.M
        error('Xpilot must have dimensions cfg.N-by-cfg.M-by-K.');
    end
    K = size(Xpilot,3);
    if K < 1 || K ~= floor(K)
        error('Xpilot must contain at least one training frame.');
    end
    if isfield(cfg,'num_training_frames') && cfg.num_training_frames ~= K
        error('Xpilot contains %d frames, but cfg.num_training_frames is %d.', ...
            K,cfg.num_training_frames);
    end

    txStream = complex(zeros(L*K,1));
    for k = 1:K
        frame = otfs_modulator(Xpilot(:,:,k),cfg);
        if numel(frame) ~= L
            error('otfs_modulator returned an unexpected frame length.');
        end
        indices = (k-1)*L + (1:L);
        txStream(indices) = frame(:);
    end
end
