function Yframes = demodulate_training_sequence(rxStream, cfg)
% DEMODULATE_TRAINING_SEQUENCE
% Split a continuous received sample stream into OTFS frames and demodulate
% each frame independently. Output is N-by-M-by-K.
% The time-domain channel must be applied to the complete concatenated
% stream BEFORE this function is called, so Doppler phase remains continuous
% and delay samples may cross frame boundaries.

    L = cfg.N * cfg.M;
    rxStream = rxStream(:);
    if isempty(rxStream) || mod(numel(rxStream),L) ~= 0
        error('Received stream length must be a positive multiple of N*M.');
    end
    K = numel(rxStream)/L;
    if isfield(cfg,'num_training_frames') && cfg.num_training_frames ~= K
        error('Received stream contains %d frames, but cfg.num_training_frames is %d.', ...
            K,cfg.num_training_frames);
    end

    Yframes = complex(zeros(cfg.N,cfg.M,K));
    for k = 1:K
        indices = (k-1)*L + (1:L);
        Yframes(:,:,k) = otfs_demodulator(rxStream(indices),cfg);
    end
end
