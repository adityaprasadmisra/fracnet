function [A, atomNorms, pairCache] = fractional_doppler_dictionary( ...
    Xpilot, cfg, dopplerGrid, delayGrid)
% FRACTIONAL_DOPPLER_DICTIONARY
% Build exact unit-norm stacked DD-response atoms and cache pair products.
%
% Each atom is generated from K concatenated OTFS frames through the actual
% Person 1 modulator/demodulator and the Person 2 continuous-time-index channel
% model. Doppler phase is never reset at frame boundaries. The dictionary
% ordering is delay outer, Doppler inner:
%   atomIndex = (delayIndex-1)*numel(dopplerGrid)+dopplerIndex.
%
% Outputs:
%   A          : (N*M*K)-by-numAtoms matrix of unit-norm atoms
%   atomNorms  : unnormalised atom norms; physical gain = coeff/atomNorm
%   pairCache  : pair indices, inner products, condition numbers and the
%                known training waveform used for off-grid refinement

    if ~isstruct(cfg) || ~isfield(cfg,'N') || ~isfield(cfg,'M') || ...
            ~isfield(cfg,'Ts') || ~isfield(cfg,'T') || ~isfield(cfg,'nu_max')
        error('cfg must contain N, M, Ts, T and nu_max.');
    end
    N = cfg.N;
    M = cfg.M;
    if ~isscalar(N) || ~isscalar(M) || ~isfinite(N) || ~isfinite(M) || ...
            N < 1 || M < 1 || N ~= floor(N) || M ~= floor(M) || ...
            ~isscalar(cfg.Ts) || ~isfinite(cfg.Ts) || cfg.Ts <= 0 || ...
            ~isscalar(cfg.T) || ~isfinite(cfg.T) || cfg.T <= 0 || ...
            ~isscalar(cfg.nu_max) || ~isfinite(cfg.nu_max) || cfg.nu_max <= 0
        error('cfg N/M must be positive integers; T, Ts and nu_max must be positive finite scalars.');
    end
    L = N * M;
    if ~isnumeric(Xpilot) || isempty(Xpilot) || ...
            any(~isfinite(Xpilot(:)))
        error('Xpilot must be a nonempty finite numeric N-by-M-by-K array.');
    end
    if size(Xpilot,1) ~= N || size(Xpilot,2) ~= M
        error('Xpilot must have dimensions N-by-M-by-K.');
    end
    numTrainingFrames = size(Xpilot,3);
    if numTrainingFrames < 1
        error('Xpilot must contain at least one training frame.');
    end
    if isfield(cfg,'num_training_frames') && ...
            cfg.num_training_frames ~= numTrainingFrames
        error('Xpilot frame count does not match cfg.num_training_frames.');
    end
    if isempty(dopplerGrid) || isempty(delayGrid)
        error('dopplerGrid and delayGrid must be nonempty.');
    end
    if ~isnumeric(dopplerGrid) || ~isreal(dopplerGrid) || ...
            ~isvector(dopplerGrid) || ~isnumeric(delayGrid) || ...
            ~isreal(delayGrid) || ~isvector(delayGrid)
        error('dopplerGrid and delayGrid must be numeric vectors.');
    end
    dopplerGrid = dopplerGrid(:).';
    delayGrid = delayGrid(:).';
    if any(~isfinite(dopplerGrid)) || any(~isfinite(delayGrid))
        error('Doppler and delay grids must be finite.');
    end
    if numel(dopplerGrid)<2 || any(diff(dopplerGrid)<=0)
        error('dopplerGrid must be strictly increasing and contain at least two points.');
    end
    if any(delayGrid < 0) || any(delayGrid ~= floor(delayGrid)) || ...
            numel(unique(delayGrid)) ~= numel(delayGrid)
        error('delayGrid must contain unique nonnegative integer sample delays.');
    end
    if any(delayGrid >= L*numTrainingFrames)
        error('Every delayGrid value must be smaller than the training stream length.');
    end
    if any(abs(dopplerGrid) > cfg.nu_max + 1e-10)
        error('Doppler grid exceeds the configured physical range.');
    end

    pilotTx = modulate_training_sequence(Xpilot,cfg);
    if numel(pilotTx) ~= L*numTrainingFrames
        error('modulate_training_sequence must return N*M*K time samples.');
    end

    numDoppler = numel(dopplerGrid);
    numDelay = numel(delayGrid);
    numAtoms = numDoppler * numDelay;

    A = complex(zeros(L*numTrainingFrames, numAtoms));
    atomNorms = zeros(numAtoms, 1);

    atomIndex = 0;
    for iDelay = 1:numDelay
        for iDoppler = 1:numDoppler
            atomIndex = atomIndex + 1;
            [atom, atomNorms(atomIndex)] = make_atom( ...
                pilotTx, dopplerGrid(iDoppler), delayGrid(iDelay), cfg);
            A(:, atomIndex) = atom;
        end
    end

    if any(~isfinite(A(:))) || any(~isfinite(atomNorms))
        error('Dictionary construction produced NaN or Inf values.');
    end

    % Precompute pair correlations once. For unit-norm atoms a_i and a_j,
    % the 2x2 Gram matrix is [1 g; conj(g) 1].
    pairIndices = nchoosek(1:numAtoms, 2);
    pairI = pairIndices(:, 1);
    pairJ = pairIndices(:, 2);

    gram = A' * A;
    pairCross = gram(sub2ind([numAtoms numAtoms], pairI, pairJ));
    pairDenominator = max(0, 1 - abs(pairCross).^2);
    pairCondition = (1 + abs(pairCross)) ./ ...
        max(1 - abs(pairCross), eps);

    if isfield(cfg, 'estimator_pair_min_denominator')
        minDenominator = cfg.estimator_pair_min_denominator;
    else
        minDenominator = 1e-8;
    end
    if isfield(cfg, 'estimator_max_pair_condition')
        maxPairCondition = cfg.estimator_max_pair_condition;
    else
        maxPairCondition = 1e5;
    end

    validPair = pairDenominator > minDenominator & ...
        pairCondition <= maxPairCondition;
    if ~any(validPair)
        error(['No numerically distinguishable dictionary pairs remain. ', ...
            'Relax estimator_max_pair_condition or refine the training grid.']);
    end

    pairCache.pairI = pairI;
    pairCache.pairJ = pairJ;
    pairCache.cross = pairCross;
    pairCache.denominator = pairDenominator;
    pairCache.condition = pairCondition;
    pairCache.valid = validPair;
    pairCache.numAtoms = numAtoms;
    pairCache.numPairs = size(pairIndices, 1);
    pairCache.numValidPairs = nnz(validPair);
    pairCache.minDenominator = minDenominator;
    pairCache.maxPairCondition = maxPairCondition;
    pairCache.maxCoherence = max(abs(pairCross));
    pairCache.pilotTx = pilotTx;
    pairCache.training_grid_size = size(Xpilot);
    pairCache.num_training_frames = numTrainingFrames;
    pairCache.stream_length = numel(pilotTx);
end

function [atom, atomNorm] = make_atom(pilotTx, doppler, delay, cfg)
    candidateChannel.delay = delay;
    candidateChannel.doppler = doppler;
    candidateChannel.gain = 1;

    rx = apply_fractional_doppler_channel(pilotTx, candidateChannel, cfg);
    Ycandidate = demodulate_training_sequence(rx,cfg);
    rawAtom = Ycandidate(:);
    atomNorm = norm(rawAtom);

    if atomNorm <= eps
        error('A dictionary atom has approximately zero norm.');
    end
    atom = rawAtom / atomNorm;
end
