function [dopplerGrid, delayGrid] = estimation_grids(cfg)
% ESTIMATION_GRIDS
% Construct the physically valid fractional-Doppler and delay grids.
%
% Doppler is expressed in normalized Doppler-bin units. The grid spans
% exactly [-cfg.nu_max, +cfg.nu_max]. Delay is an integer sample index.

    if ~isfield(cfg, 'nu_max') || cfg.nu_max <= 0
        error('cfg.nu_max must be positive.');
    end
    if ~isfield(cfg, 'estimator_doppler_step') || ...
            cfg.estimator_doppler_step <= 0
        error('cfg.estimator_doppler_step must be positive.');
    end
    if ~isfield(cfg, 'l_max') || cfg.l_max < 0 || ...
            cfg.l_max ~= floor(cfg.l_max)
        error('cfg.l_max must be a nonnegative integer.');
    end

    numberOfPoints = ceil((2 * cfg.nu_max) / ...
        cfg.estimator_doppler_step) + 1;

    dopplerGrid = linspace(-cfg.nu_max, cfg.nu_max, numberOfPoints);
    delayGrid = 0:cfg.l_max;
end
