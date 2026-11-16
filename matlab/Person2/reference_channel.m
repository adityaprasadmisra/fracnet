function H = reference_channel(channel, cfg)
% REFERENCE_CHANNEL
% Construct the effective DD-domain channel matrix using the actual
% Person 1 OTFS modulator/demodulator and Person 2 time-domain channel.
%
% The resulting matrix satisfies, up to numerical precision:
%   vec(Ydd) = H * vec(Xdd)
%
% Person 1 files are called as-is; this function does not duplicate their
% transform equations.

    N = cfg.N;
    M = cfg.M;
    L = N * M;

    H = complex(zeros(L, L));

    for col = 1:L
        basisDD = complex(zeros(N, M));
        basisDD(col) = 1;

        tx = otfs_modulator(basisDD, cfg);
        rx = apply_fractional_doppler_channel(tx, channel, cfg);
        Ydd = otfs_demodulator(rx, cfg);

        H(:, col) = Ydd(:);
    end
end
