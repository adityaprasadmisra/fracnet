function [H_est, detected_paths, gains_est, residual, info] = ...
    fractional_doppler_coarse_estimation( ...
    Ypilot, A, dopplerGrid, delayGrid, cfg, atomNorms, pairCache, noiseVariance)
% FRACTIONAL_DOPPLER_COARSE_ESTIMATION
% Joint two-path fractional-Doppler estimator (Revision 20).
%
% Main design retained from Revision 11, with Revision 12-14 additions:
%   1) Uses two-atom maximum-likelihood variable projection for path-location
%      search: for each candidate pair, the complex coefficients are solved
%      by unregularized least squares and residual energy is minimized.
%   2) Retains multiple distinct coarse minima per delay-pair class and uses
%      a joint two-dimensional pattern search to enter several basins.
%   3) Adds analytic-derivative damped Gauss-Newton / Levenberg-Marquardt
%      refinement. The complex gains are eliminated at every iteration.
%   4) Keeps a limited number of Nelder-Mead runs as an independent fallback;
%      only raw endpoints that improve their seed and beat the prior best are
%      sent to an additional analytic LM polish. The lowest unregularized
%      residual among all accepted candidates is retained.
%   5) Applies noise-aware ridge regularization only to the final physical
%      gain estimate, after path locations have been selected.
%   6) Supports K contiguous known OTFS frames in one observation. The
%      Doppler phase is continuous over all K*N*M samples.
%   7) Preserves the exact Person 1 TX/RX chain and per-frame OTFS transforms.
%   8) Uses an SNR-adaptive condition-number ceiling so very close paths are
%      not automatically discarded at high SNR.
%   9) Recomputes pair validity from cached correlations/conditions, so a
%      dictionary built with a stricter old threshold does not hide candidates.
%  10) Adds a reserved near-diagonal seed pool for same-delay, close-Doppler
%      hypotheses, so close path pairs receive explicit nonlinear refinement.
%  11) Exposes the effective condition limit and adaptive candidate counts
%      for every observation; these are needed to diagnose ambiguous trials.
%  12) Keeps the ML estimate unchanged: ambiguous physical decompositions are
%      not forced toward a presumed close-path answer.
%  13) Uses an SNR reference point for condition-limit relaxation, avoiding
%      overly permissive high-condition pairs at moderate SNR.
%  14) Applies an explicit budget to expensive pattern-search seeds and makes
%      LM/Nelder-Mead budgets truly configurable to control runtime.
%  15) Retains the Revision 15 source-diverse fallback and bounded LM retries.
%  16) Uses center/separation coordinates for same-delay close-pair simplex
%      refinement, which improves the numerical geometry of close-path search.
%  17) Maintains a multi-hypothesis solution bank and reports noise-scaled
%      likelihood gaps, alternative locations, and ambiguity diagnostics.
%  18) Keeps the final estimator output ML-selected; ambiguity is reported
%      rather than hidden by an unsupported condition-number tie-break.
%  19) Uses a dedicated center/separation pattern search for same-delay
%      close paths, not only a center/separation Nelder-Mead fallback.
%  20) Adds a small high-condition close-pair seed pool at low SNR, with a
%      noise-scaled ranking penalty; ordinary candidate admission is unchanged.
%  21) Uses a compact 3x3 center/separation pattern and center/separation LM
%      coordinates, replacing repeated 3x5 scans and poorly scaled Doppler steps.
%  22) Avoids LM-polishing every Nelder-Mead endpoint; only useful endpoints
%      receive a second LM pass.
%  23) Reports local Fisher-information uncertainty for Doppler and complex gains.
%  24) Uses an orthogonalized two-column projection score to avoid cancellation
%      when coarse dictionary atoms are highly correlated.
%  25) Allocates reserved close-pair seeds across Doppler-separation bands so
%      the nearest-separation region cannot be crowded out by wider pairs.
%  26) Makes the reserved high-condition close-pair ceiling increase with SNR;
%      noisy observations no longer admit the same very ill-conditioned pairs.
%  27) Uses augmented QR least squares for the final ridge gain solve, avoiding
%      the normal-equation condition-number penalty.
%  28) Reuses the already computed A'*y correlations in residual seed search.
%  29) Removes a redundant least-squares projection after accepted LM steps.
%
% Inputs:
%   Ypilot       : N-by-M-by-K received DD responses (K training frames)
%   A            : unit-norm dictionary from fractional_doppler_dictionary
%   dopplerGrid  : physically bounded normalized-Doppler grid
%   delayGrid    : integer sample-delay candidates
%   cfg          : system configuration (cfg.P must equal 2)
%   atomNorms    : original, unnormalized dictionary atom norms
%   pairCache    : pair cache from fractional_doppler_dictionary
%   noiseVariance: complex AWGN variance per time/DD sample; optional
%
% Outputs:
%   H_est          : reconstructed complete DD response
%   detected_paths : rows [normalized Doppler, integer delay]
%   gains_est      : estimated physical complex path gains
%   residual       : Ypilot-H_est
%   info           : search and conditioning diagnostics

    if nargin < 8 || isempty(noiseVariance)
        noiseVariance = 0;
    end
    if ~isstruct(cfg) || ~isfield(cfg,'P') || cfg.P ~= 2
        error('cfg must be a valid structure with cfg.P = 2.');
    end
    requiredCfgFields = {'N','M','T','Ts','nu_max'};
    for fieldIndex = 1:numel(requiredCfgFields)
        if ~isfield(cfg,requiredCfgFields{fieldIndex}) || ...
                ~isscalar(cfg.(requiredCfgFields{fieldIndex})) || ...
                ~isfinite(cfg.(requiredCfgFields{fieldIndex}))
            error('cfg.%s must be a finite scalar.',requiredCfgFields{fieldIndex});
        end
    end
    if cfg.N < 1 || cfg.M < 1 || cfg.N ~= floor(cfg.N) || ...
            cfg.M ~= floor(cfg.M) || cfg.T <= 0 || cfg.Ts <= 0 || cfg.nu_max <= 0
        error('cfg dimensions and sample/time parameters must be positive and valid.');
    end
    numTrainingFrames = get_cfg(cfg,'num_training_frames',1);
    if ~isscalar(numTrainingFrames) || ~isfinite(numTrainingFrames) || ...
            numTrainingFrames < 1 || numTrainingFrames ~= floor(numTrainingFrames)
        error('cfg.num_training_frames must be a positive integer.');
    end
    expectedObservations = cfg.N*cfg.M*numTrainingFrames;
    if ~isnumeric(Ypilot) || isempty(Ypilot) || ...
            numel(Ypilot) ~= expectedObservations || any(~isfinite(Ypilot(:)))
        error('Ypilot must be a finite numeric vector/array with N*M*K observations.');
    end
    if ~isnumeric(A) || ~ismatrix(A) || size(A,1) ~= expectedObservations || ...
            size(A,2) ~= numel(atomNorms)
        error('Dictionary dimensions and atomNorms do not match the training sequence.');
    end
    if ~isnumeric(atomNorms) || ~isvector(atomNorms) || isempty(atomNorms) || ...
            any(~isfinite(atomNorms(:))) || any(atomNorms(:) <= 0)
        error('atomNorms must contain finite positive values.');
    end
    requiredPairFields = {'numAtoms','pilotTx','pairI','pairJ','cross','condition'};
    if ~isstruct(pairCache)
        error('pairCache must be the structure returned by fractional_doppler_dictionary.');
    end
    for fieldIndex = 1:numel(requiredPairFields)
        if ~isfield(pairCache,requiredPairFields{fieldIndex})
            error('pairCache is missing field %s. Rebuild the dictionary.', ...
                requiredPairFields{fieldIndex});
        end
    end
    if pairCache.numAtoms ~= size(A,2) || ...
            numel(pairCache.pilotTx) ~= expectedObservations
        error('pairCache does not match the supplied dictionary/training stream.');
    end
    pairCount = numel(pairCache.pairI);
    if pairCount == 0 || ~isnumeric(pairCache.pairI) || ...
            ~isnumeric(pairCache.pairJ) || ~isnumeric(pairCache.cross) || ...
            ~isnumeric(pairCache.condition) || ...
            numel(pairCache.pairJ) ~= pairCount || ...
            numel(pairCache.cross) ~= pairCount || ...
            numel(pairCache.condition) ~= pairCount || ...
            ~isnumeric(pairCache.pilotTx) || ...
            numel(pairCache.pilotTx) ~= expectedObservations || ...
            any(~isfinite(pairCache.pairI(:))) || ...
            any(~isfinite(pairCache.pairJ(:))) || ...
            any(~isfinite(pairCache.cross(:))) || ...
            any(~isfinite(pairCache.condition(:))) || ...
            any(~isfinite(pairCache.pilotTx(:)))
        error('pairCache arrays are empty, nonnumeric, nonfinite or inconsistent. Rebuild the dictionary.');
    end
    if any(pairCache.pairI(:) < 1) || any(pairCache.pairJ(:) > size(A,2)) || ...
            any(pairCache.pairI(:) >= pairCache.pairJ(:)) || ...
            any(pairCache.pairI(:) ~= floor(pairCache.pairI(:))) || ...
            any(pairCache.pairJ(:) ~= floor(pairCache.pairJ(:)))
        error('pairCache contains invalid unordered dictionary-pair indices.');
    end
    dopplerGrid = dopplerGrid(:).';
    delayGrid = delayGrid(:).';
    if ~isnumeric(dopplerGrid) || ~isreal(dopplerGrid) || ...
            ~isvector(dopplerGrid) || numel(dopplerGrid) < 2 || any(~isfinite(dopplerGrid)) || ...
            any(diff(dopplerGrid) <= 0)
        error('dopplerGrid must be finite, strictly increasing and contain at least two points.');
    end
    if ~isnumeric(delayGrid) || ~isreal(delayGrid) || ...
            ~isvector(delayGrid) || isempty(delayGrid) || any(~isfinite(delayGrid)) || ...
            any(delayGrid < 0) || any(delayGrid ~= floor(delayGrid)) || ...
            numel(unique(delayGrid)) ~= numel(delayGrid)
        error('delayGrid must contain unique nonnegative integer sample delays.');
    end
    if any(delayGrid >= expectedObservations)
        error('Every delayGrid value must be smaller than the observation stream length.');
    end
    if size(A,2) ~= numel(dopplerGrid)*numel(delayGrid)
        error('Dictionary atom count does not match dopplerGrid and delayGrid. Rebuild the dictionary.');
    end
    if ~isnumeric(noiseVariance) || ~isscalar(noiseVariance) || ...
            ~isreal(noiseVariance) || ~isfinite(noiseVariance) || noiseVariance < 0
        error('noiseVariance must be a finite nonnegative real scalar.');
    end
    if any(~isfinite(Ypilot(:))) || any(~isfinite(A(:)))
        error('Estimator input contains NaN or Inf.');
    end
    if any(abs(dopplerGrid(:)) > cfg.nu_max + 1e-10)
        error('Doppler grid exceeds the configured physical range.');
    end

    y = Ypilot(:);
    yNormSquared = real(y' * y);
    numAtoms = size(A,2);
    numDoppler = numel(dopplerGrid);
    numDelay = numel(delayGrid);

    % Path locations are always selected using unregularized ML residuals.
    % The base condition limit is retained at low SNR. At higher SNR, the
    % estimator is allowed to consider more correlated pairs because the
    % observation contains more information about their small difference.
    % This is a bounded relaxation, not a claim that arbitrarily close paths
    % become identifiable.
    minPairDenominator = get_cfg(cfg,'estimator_pair_min_denominator',1e-8);
    baseMaxPairCondition = get_cfg(cfg,'estimator_max_pair_condition',25);
    conditionCeiling = get_cfg(cfg, ...
        'estimator_adaptive_condition_ceiling',1000);
    conditionSNRExponent = get_cfg(cfg, ...
        'estimator_condition_snr_exponent',1.0);
    conditionSNRReference = get_cfg(cfg, ...
        'estimator_condition_snr_reference_linear',2.5);

    if ~isscalar(baseMaxPairCondition) || ~isfinite(baseMaxPairCondition) || ...
            baseMaxPairCondition <= 1
        error('estimator_max_pair_condition must be a finite scalar greater than 1.');
    end
    if ~isscalar(conditionCeiling) || ~isfinite(conditionCeiling) || ...
            conditionCeiling < baseMaxPairCondition
        error('estimator_adaptive_condition_ceiling must be finite and >= the base condition limit.');
    end
    if ~isscalar(conditionSNRExponent) || ~isfinite(conditionSNRExponent) || ...
            conditionSNRExponent < 0
        error('estimator_condition_snr_exponent must be finite and nonnegative.');
    end
    if ~isscalar(conditionSNRReference) || ~isfinite(conditionSNRReference) || ...
            conditionSNRReference <= 0
        error('estimator_condition_snr_reference_linear must be finite and positive.');
    end

    if noiseVariance > 0
        % The OTFS transforms are unitary in this project, so average DD
        % observation power estimates average time-domain signal power.
        estimatedSignalPower = max(yNormSquared/expectedObservations - ...
            noiseVariance,eps);
        estimatedObservationSNRLinear = estimatedSignalPower/noiseVariance;
        estimatedObservationSNRdB = 10*log10(estimatedObservationSNRLinear);
        % Do not relax the pair condition limit until the observation SNR
        % exceeds the configured reference. This is more conservative than
        % scaling directly from 0 dB, where noise-fitting pairs otherwise
        % become admissible too early.
        snrScale = max(1,estimatedObservationSNRLinear / ...
            conditionSNRReference)^conditionSNRExponent;
        maxPairCondition = min(conditionCeiling, ...
            baseMaxPairCondition*snrScale);
    else
        estimatedSignalPower = yNormSquared/expectedObservations;
        estimatedObservationSNRLinear = Inf;
        estimatedObservationSNRdB = Inf;
        maxPairCondition = conditionCeiling;
    end
    maxPairCondition = max(baseMaxPairCondition,maxPairCondition);

    % Coarse two-atom ML projection score. A and its columns are unit norm.
    % For each candidate pair, this is ||P_B*y||^2, obtained from the 2x2
    % Gram matrix. The pair cache stores all pair correlations and conditions.
    % Recompute validity here instead of trusting pairCache.valid, because
    % that mask may have been built earlier using only the low-SNR base cap.
    correlations = A' * y;
    pairI = pairCache.pairI(:);
    pairJ = pairCache.pairJ(:);
    cross = pairCache.cross(:);
    pairCondition = pairCache.condition(:);
    if isfield(pairCache,'denominator')
        pairDenominator = pairCache.denominator(:);
    else
        pairDenominator = max(0,1-abs(cross).^2);
    end
    ci = correlations(pairI);
    cj = correlations(pairJ);
    % Stable two-column QR projection.  With rho=a_i'*a_j, the component
    % of a_j orthogonal to a_i is (a_j-a_i*rho).  Therefore
    % ||P_[ai aj] y||^2 = |ai'*y|^2 + |aj'*y-conj(rho)*(ai'*y)|^2/(1-|rho|^2).
    % This avoids subtracting nearly equal O(||y||^2) terms in the expanded
    % Gram formula when |rho| is close to one.
    denominator = max(1 - abs(cross).^2, eps);
    orthogonalCorrelation = cj - conj(cross).*ci;
    projectionScore = abs(ci).^2 + ...
        abs(orthogonalCorrelation).^2 ./ denominator;
    % Projection energy is mathematically nonnegative. Clamp tiny negative
    % round-off values and reject non-finite scores before seed ranking.
    projectionScore = max(real(projectionScore),0);
    pairScore = projectionScore(:);
    pairScore(~isfinite(pairScore)) = -Inf;
    if numel(pairDenominator) ~= numel(pairScore) || ...
            numel(pairCondition) ~= numel(pairScore)
        error('pairCache denominator and condition arrays must match candidate pairs.');
    end
    rawPairScore = pairScore;
    validPair = isfinite(pairCondition) & isfinite(pairDenominator) & ...
        pairDenominator > minPairDenominator & ...
        pairCondition <= maxPairCondition;
    cachedValidCount = NaN;
    if isfield(pairCache,'valid') && numel(pairCache.valid)==numel(pairScore)
        cachedValidCount = nnz(pairCache.valid(:));
    end
    newlyAdmittedPairCount = nnz(validPair) - ...
        nnz(validPair & isfield_valid_mask(pairCache,numel(pairScore)));
    pairScore(~validPair) = -Inf;

    % A close same-delay pair can be physically correct even when its Gram
    % condition exceeds the ordinary low-SNR admission cap. Keep only a small
    % reserved pool of such pairs, and rank it with a noise-scaled penalty.
    % This does NOT relax the condition cap for the ordinary coarse/residual
    % pools. The reserved pool is bounded by the global configured ceiling.
    configuredCloseConditionCeiling = get_cfg(cfg, ...
        'estimator_close_pair_condition_ceiling',600);
    closeConditionSNRExponent = get_cfg(cfg, ...
        'estimator_close_pair_condition_snr_exponent',1.3);
    if ~isscalar(configuredCloseConditionCeiling) || ...
            ~isfinite(configuredCloseConditionCeiling) || ...
            configuredCloseConditionCeiling < baseMaxPairCondition
        error('estimator_close_pair_condition_ceiling must be finite and >= the base condition limit.');
    end
    if ~isscalar(closeConditionSNRExponent) || ...
            ~isfinite(closeConditionSNRExponent) || closeConditionSNRExponent < 0
        error('estimator_close_pair_condition_snr_exponent must be finite and nonnegative.');
    end
    if noiseVariance > 0
        closePairSNRScale = max(1,estimatedObservationSNRLinear / ...
            conditionSNRReference)^closeConditionSNRExponent;
        closePairConditionCeiling = min(conditionCeiling, ...
            max(maxPairCondition,baseMaxPairCondition*closePairSNRScale));
    else
        closePairConditionCeiling = min(conditionCeiling, ...
            max(maxPairCondition,configuredCloseConditionCeiling));
    end
    closePairConditionCeiling = min(closePairConditionCeiling, ...
        configuredCloseConditionCeiling);
    closeSeedPenaltyWeight = get_cfg(cfg, ...
        'estimator_close_pair_seed_condition_penalty',1.0);
    closeSeedValid = isfinite(pairCondition) & isfinite(pairDenominator) & ...
        pairDenominator > minPairDenominator & ...
        pairCondition <= closePairConditionCeiling;
    closePairScore = rawPairScore;
    if noiseVariance > 0 && closeSeedPenaltyWeight > 0
        conditionExcess = max(0,log(max(pairCondition,1)./max(maxPairCondition,1)));
        closePairScore = closePairScore - ...
            closeSeedPenaltyWeight*noiseVariance.*conditionExcess;
    end
    closePairScore(~closeSeedValid) = -Inf;

    gridStep = median(diff(dopplerGrid));
    if isempty(gridStep) || ~isfinite(gridStep) || gridStep <= 0
        gridStep = get_cfg(cfg,'estimator_doppler_step',0.0125);
    end
    topK = round(get_cfg(cfg,'estimator_refine_top_k',8));
    if ~isscalar(topK) || ~isfinite(topK) || topK < 1
        error('cfg.estimator_refine_top_k must be a positive integer.');
    end
    minSeedSeparation = get_cfg(cfg, ...
        'estimator_seed_min_separation_steps',2.5) * gridStep;
    seedsPerDelayPair = get_cfg(cfg, ...
        'estimator_seeds_per_delay_pair',3);
    candidatePairIndices = select_diverse_pair_seeds( ...
        pairScore,pairI,pairJ,numAtoms,numDoppler,numDelay, ...
        dopplerGrid,topK,minSeedSeparation,seedsPerDelayPair);

    % Reserve additional seeds for same-delay paths separated by only a few
    % coarse Doppler bins. These are precisely the hypotheses that can be
    % underrepresented by ordinary local-maximum selection when the two
    % dictionary columns are highly coherent. The additional seeds are not
    % forced to win; they simply receive a chance to enter continuous search.
    closeSeedEnable = get_cfg(cfg,'estimator_close_pair_seed_enable',true);
    closeSeedMinSteps = get_cfg(cfg,'estimator_close_pair_min_grid_steps',1.0);
    closeSeedMaxSteps = get_cfg(cfg,'estimator_close_pair_max_grid_steps',4.0);
    closeSeedsPerDelay = get_cfg(cfg,'estimator_close_pair_seeds_per_delay',2);
    closeSeparationBands = get_cfg(cfg, ...
        'estimator_close_pair_separation_bands_steps', ...
        [0.5 1.5;1.5 3;3 6;6 10]);
    if ~isnumeric(closeSeparationBands) || ~isreal(closeSeparationBands) || ...
            size(closeSeparationBands,2)~=2 || isempty(closeSeparationBands) || ...
            any(~isfinite(closeSeparationBands(:))) || ...
            any(closeSeparationBands(:,1)<closeSeedMinSteps) || ...
            any(closeSeparationBands(:,2)<=closeSeparationBands(:,1))
        error('estimator_close_pair_separation_bands_steps must be a finite K-by-2 increasing band matrix.');
    end
    closePairSeedIndices = zeros(0,1);
    if closeSeedEnable
        closePairSeedIndices = select_close_same_delay_seeds( ...
            closePairScore,pairI,pairJ,numAtoms,numDoppler,dopplerGrid, ...
            gridStep,closeSeedMinSteps,closeSeedMaxSteps, ...
            closeSeedsPerDelay,closeSeparationBands);
    end

    % Sequential residual seeds complement the joint pair-score search.
    % The strongest one-atom hypotheses are fitted first; the residual is
    % then correlated against the full dictionary to expose a weaker second
    % path that can be masked in the raw received-signal correlations.
    residualSeedEnable = get_cfg(cfg,'estimator_residual_seed_enable',true);
    residualFirstAtomCount = get_cfg(cfg, ...
        'estimator_residual_seed_top_atoms',6);
    residualSecondAtomCount = get_cfg(cfg, ...
        'estimator_residual_seed_second_atoms',1);
    residualPairSeedIndices = zeros(0,1);
    if residualSeedEnable
        residualPairSeedIndices = select_residual_pair_seeds( ...
            numAtoms,correlations,pairI,pairJ,cross,validPair, ...
            residualFirstAtomCount,residualSecondAtomCount);
    end

    % The candidate pools can contain many more seeds than can be afforded
    % by a full 2-D pattern search. Keep separate quotas for the three seed
    % sources so close-pair and weak-path hypotheses are not crowded out.
    coarsePatternBudget = max(1,round(get_cfg(cfg, ...
        'estimator_pattern_coarse_budget',18)));
    closePatternBudget = max(0,round(get_cfg(cfg, ...
        'estimator_pattern_close_budget',6)));
    residualPatternBudget = max(0,round(get_cfg(cfg, ...
        'estimator_pattern_residual_budget',6)));
    patternSeedBudget = max(1,round(get_cfg(cfg, ...
        'estimator_pattern_seed_budget',30)));

    coarsePatternSeeds = top_pair_seed_subset( ...
        candidatePairIndices,pairScore,coarsePatternBudget);
    closePatternSeeds = ordered_seed_subset( ...
        closePairSeedIndices,closePatternBudget);
    % Preserve residual-search order: it is ranked by sequential residual
    % correlation, not by the raw-observation two-atom score.
    residualPatternSeeds = ordered_seed_subset( ...
        residualPairSeedIndices,residualPatternBudget);
    candidatePairIndices = unique([coarsePatternSeeds(:); ...
        closePatternSeeds(:);residualPatternSeeds(:)],'stable');
    if numel(candidatePairIndices) > patternSeedBudget
        candidatePairIndices = candidatePairIndices(1:patternSeedBudget);
    end
    if isempty(candidatePairIndices)
        error(['No finite path-pair candidates survived the configured ', ...
            'condition and seed filters. Relax the pair-condition threshold ', ...
            'or inspect the dictionary/training signal.']);
    end
    % Keep diagnostics and candidate-source flags aligned with the actual
    % seeds entering pattern search, not the larger pre-budget pools.
    closePairSeedIndices = closePatternSeeds;
    residualPairSeedIndices = residualPatternSeeds;
    topK = numel(candidatePairIndices);

    refineLevels = max(1,round(get_cfg(cfg,'estimator_refine_levels',7)));

    % First stage: joint 3x3 pattern search from diverse coarse seeds.
    candidates = repmat(struct( ...
        'objective',Inf,'nu',[0 0],'delay',[0 0],'B',[], ...
        'rawNorms',[],'condition',Inf,'seedIndex',0, ...
        'seedPairScore',-Inf,'seedPairCoherence',NaN, ...
        'isCloseSeed',false,'isResidualSeed',false, ...
        'conditionLimit',maxPairCondition),topK,1);
    totalPairEvaluations = 0;
    totalAtomEvaluations = 0;
    closePairPatternStarts = 0;

    for seedNumber = 1:topK
        pairNumber = candidatePairIndices(seedNumber);
        atomIndices = [pairI(pairNumber), pairJ(pairNumber)];
        nu = zeros(1,2);
        delay = zeros(1,2);
        for p = 1:2
            [nu(p),delay(p)] = atom_index_to_parameters( ...
                atomIndices(p),dopplerGrid,delayGrid);
        end

        B = A(:,atomIndices);
        rawNorms = atomNorms(atomIndices(:));
        isReservedCloseSeed = any(closePairSeedIndices==pairNumber) && ...
            delay(1)==delay(2);
        seedConditionLimit = maxPairCondition;
        if isReservedCloseSeed
            seedConditionLimit = closePairConditionCeiling;
        end
        [objective,~,~,currentCondition] = evaluate_pair_ml( ...
            B,y,minPairDenominator,seedConditionLimit);
        totalPairEvaluations = totalPairEvaluations+1;

        step = gridStep;
        closePatternMaxSeparation = get_cfg(cfg, ...
            'estimator_close_pair_search_max_separation',0.12);
        closePatternMinSeparation = get_cfg(cfg, ...
            'estimator_close_pair_search_min_separation', ...
            max(0.25*gridStep,1e-4));
        % Spend the extra center/separation pattern-search budget only on
        % seeds explicitly reserved by the close-pair selector. Other
        % same-delay candidates still receive center/separation LM later,
        % without paying for a second dense local stencil here.
        useCenterSeparationPattern = isReservedCloseSeed && ...
            delay(1)==delay(2) && ...
            abs(diff(nu)) <= closePatternMaxSeparation;
        seedRefineLevels = refineLevels;
        if useCenterSeparationPattern
            closePairPatternStarts = closePairPatternStarts+1;
            seedRefineLevels = min(refineLevels,max(1,round(get_cfg(cfg, ...
                'estimator_close_pair_refine_levels',4))));
        end

        for level = 1:seedRefineLevels
            levelBestObjective = objective;
            levelBestNu = nu;
            levelBestB = B;
            levelBestNorms = rawNorms;
            levelBestCondition = currentCondition;

            if useCenterSeparationPattern
                % For same-delay close paths, search in center/separation
                % coordinates. Independent Doppler perturbations produce a
                % skewed lattice in these coordinates and can undersample
                % the narrow separation direction of a coherent pair.
                center = mean(nu);
                separation = abs(diff(nu));
                centerLimit = max(cfg.nu_max-separation/2,0);
                trialCenters = unique(min(centerLimit,max(-centerLimit, ...
                    center+[-step 0 step])));
                % A compact 3x3 stencil explores the coupled centre and
                % separation directions. The previous 3x5 stencil performed
                % 15 pair projections per level, often repeating nearly
                % identical atoms, while adding little after analytic LM.
                trialSeparations = unique(min(closePatternMaxSeparation, ...
                    max(closePatternMinSeparation, ...
                    separation+[-step 0 step])));
                trialNu1 = zeros(numel(trialCenters)*numel(trialSeparations),1);
                trialNu2 = trialNu1;
                pairCursor = 0;
                for a = 1:numel(trialCenters)
                    for b = 1:numel(trialSeparations)
                        trialSeparation = trialSeparations(b);
                        trialCenterLimit = max(cfg.nu_max-trialSeparation/2,0);
                        trialCenter = min(trialCenterLimit,max(-trialCenterLimit, ...
                            trialCenters(a)));
                        pairCursor = pairCursor+1;
                        trialNu1(pairCursor) = trialCenter-trialSeparation/2;
                        trialNu2(pairCursor) = trialCenter+trialSeparation/2;
                    end
                end
                trialNu1 = trialNu1(1:pairCursor);
                trialNu2 = trialNu2(1:pairCursor);
                uniqueNu = unique([trialNu1;trialNu2]);
                atomCache = cell(numel(uniqueNu),1);
                normCache = zeros(numel(uniqueNu),1);
                for u = 1:numel(uniqueNu)
                    [atomCache{u},normCache(u)] = make_atom( ...
                        pairCache.pilotTx,uniqueNu(u),delay(1),cfg);
                end
                totalAtomEvaluations = totalAtomEvaluations+numel(uniqueNu);
                [~,map1] = ismember(trialNu1,uniqueNu);
                [~,map2] = ismember(trialNu2,uniqueNu);
                for q = 1:pairCursor
                    Btrial = [atomCache{map1(q)},atomCache{map2(q)}];
                    normTrial = [normCache(map1(q));normCache(map2(q))];
                    [trialObjective,~,~,trialCondition] = evaluate_pair_ml( ...
                        Btrial,y,minPairDenominator,seedConditionLimit);
                    totalPairEvaluations = totalPairEvaluations+1;
                    if trialObjective < levelBestObjective
                        levelBestObjective = trialObjective;
                        levelBestNu = [trialNu1(q),trialNu2(q)];
                        levelBestB = Btrial;
                        levelBestNorms = normTrial;
                        levelBestCondition = trialCondition;
                    end
                end
            else
                % For non-close or different-delay paths, retain the standard
                % independent two-Doppler local pattern search.
                trialNu1 = unique(min(cfg.nu_max,max(-cfg.nu_max, ...
                    nu(1)+[-step 0 step])));
                trialNu2 = unique(min(cfg.nu_max,max(-cfg.nu_max, ...
                    nu(2)+[-step 0 step])));

                atoms1 = cell(numel(trialNu1),1);
                norms1 = zeros(numel(trialNu1),1);
                atoms2 = cell(numel(trialNu2),1);
                norms2 = zeros(numel(trialNu2),1);
                for a = 1:numel(trialNu1)
                    [atoms1{a},norms1(a)] = make_atom( ...
                        pairCache.pilotTx,trialNu1(a),delay(1),cfg);
                end
                for b = 1:numel(trialNu2)
                    [atoms2{b},norms2(b)] = make_atom( ...
                        pairCache.pilotTx,trialNu2(b),delay(2),cfg);
                end
                totalAtomEvaluations = totalAtomEvaluations + ...
                    numel(trialNu1)+numel(trialNu2);

                for a = 1:numel(trialNu1)
                    for b = 1:numel(trialNu2)
                        Btrial = [atoms1{a},atoms2{b}];
                        [trialObjective,~,~,trialCondition] = evaluate_pair_ml( ...
                            Btrial,y,minPairDenominator,seedConditionLimit);
                        totalPairEvaluations = totalPairEvaluations+1;
                        if trialObjective < levelBestObjective
                            levelBestObjective = trialObjective;
                            levelBestNu = [trialNu1(a),trialNu2(b)];
                            levelBestB = Btrial;
                            levelBestNorms = [norms1(a);norms2(b)];
                            levelBestCondition = trialCondition;
                        end
                    end
                end
            end

            nu = levelBestNu;
            B = levelBestB;
            rawNorms = levelBestNorms;
            objective = levelBestObjective;
            currentCondition = levelBestCondition;
            step = step/2;
            % Re-evaluate the branch after each level because the candidate
            % may move into or out of the configured close-pair region.
            useCenterSeparationPattern = isReservedCloseSeed && ...
                delay(1)==delay(2) && ...
                abs(diff(nu)) <= closePatternMaxSeparation;
        end

        candidates(seedNumber).objective = objective;
        candidates(seedNumber).nu = nu;
        candidates(seedNumber).delay = delay;
        candidates(seedNumber).B = B;
        candidates(seedNumber).rawNorms = rawNorms;
        candidates(seedNumber).condition = currentCondition;
        candidates(seedNumber).seedIndex = seedNumber;
        candidates(seedNumber).seedPairScore = pairScore(pairNumber);
        candidates(seedNumber).seedPairCoherence = abs(cross(pairNumber));
        candidates(seedNumber).isCloseSeed = ...
            any(closePairSeedIndices==pairNumber);
        candidates(seedNumber).isResidualSeed = ...
            any(residualPairSeedIndices==pairNumber);
        candidates(seedNumber).conditionLimit = seedConditionLimit;
    end

    % Maintain every distinct finite hypothesis produced by the search.
    % This bank is used only for confidence/ambiguity reporting; the selected
    % estimate remains the minimum unregularized-ML-residual solution.
    solutionBank = empty_solution_bank();
    solutionDuplicateTolerance = max(1e-6,0.10*gridStep);
    for q = 1:numel(candidates)
        if isfinite(candidates(q).objective)
            solutionBank = add_solution_to_bank(solutionBank, ...
                candidates(q).objective,candidates(q).nu,candidates(q).delay, ...
                candidates(q).condition,'pattern search', ...
                candidates(q).seedIndex,solutionDuplicateTolerance);
        end
    end

    % Second stage: retain several minima per delay class, then refine each
    % candidate with an analytic-derivative variable-projection LM solver.
    % This replaces the previous reliance on Nelder-Mead alone. The exact
    % derivative is obtained by differentiating the time-domain Doppler
    % phase and passing it through the same OTFS demodulator as each atom.
    minRefinedSeparation = get_cfg(cfg, ...
        'estimator_refined_seed_separation_steps',0.25) * gridStep;
    lmCandidateBudget = max(1,round(get_cfg(cfg,'estimator_lm_top_k',18)));
    lmCandidatesPerDelayClass = max(1,round(get_cfg(cfg, ...
        'estimator_nonlinear_seeds_per_delay_pair',4)));
    refineIndices = select_nonlinear_refinement_candidates( ...
        candidates,lmCandidateBudget,lmCandidatesPerDelayClass, ...
        minRefinedSeparation);

    % Keep the best pattern-search solution as a guaranteed fallback.
    [bestObjective,bestIndex] = min([candidates.objective]);
    cBest = candidates(bestIndex);
    best.nu = cBest.nu;
    best.delay = cBest.delay;
    best.B = cBest.B;
    best.rawNorms = cBest.rawNorms;
    best.condition = cBest.condition;
    best.conditionLimit = cBest.conditionLimit;
    best.seedIndex = cBest.seedIndex;
    best.seedPairScore = cBest.seedPairScore;
    best.seedPairCoherence = cBest.seedPairCoherence;
    best.source = 'pattern-search fallback';

    lmMaxIter = get_cfg(cfg,'estimator_lm_max_iter',80);
    lmInitialDamping = get_cfg(cfg,'estimator_lm_initial_damping',1e-3);
    lmMaxStep = get_cfg(cfg,'estimator_lm_max_step',0.05);
    lmGradientTolerance = get_cfg(cfg, ...
        'estimator_lm_gradient_tolerance',1e-9);
    lmStepTolerance = get_cfg(cfg,'estimator_lm_step_tolerance',1e-10);

    lmTotalIterations = 0;
    lmTotalAcceptedSteps = 0;
    lmCalls = 0;
    lmRetryCalls = 0;
    lmRetryBudget = max(0,round(get_cfg(cfg,'estimator_lm_retry_budget',2)));
    lmRetryEnable = get_cfg(cfg,'estimator_lm_retry_enable',true);
    lmFinalGradients = zeros(numel(refineIndices),1);
    lmFinalObjectives = Inf(numel(refineIndices),1);
    lmRetryAttempted = false(numel(refineIndices),1);
    lmRetryStatuses = cell(numel(refineIndices),1);

    % Refine every retained coarse minimum. When the analytic LM solver
    % stalls, allow a small number of alternate-damping retries. Retries are
    % limited globally so difficult observations do not cause an unbounded
    % runtime increase.
    for r = 1:numel(refineIndices)
        c = candidates(refineIndices(r));
        [nuLM,BLM,normLM,objLM,condLM,lmStats,retryCount] = ...
            refine_pair_variable_projection_lm_with_retry( ...
            pairCache.pilotTx,y,c.nu,c.delay,cfg, ...
            minPairDenominator,c.conditionLimit,lmMaxIter, ...
            lmInitialDamping,lmMaxStep,lmGradientTolerance, ...
            lmStepTolerance,lmRetryEnable && lmRetryCalls<lmRetryBudget);
        lmRetryCalls = lmRetryCalls+retryCount;
        lmCalls = lmCalls+1+retryCount;
        lmTotalIterations = lmTotalIterations+lmStats.total_iterations;
        lmTotalAcceptedSteps = lmTotalAcceptedSteps+lmStats.total_accepted_steps;
        lmFinalGradients(r) = lmStats.final_gradient;
        lmFinalObjectives(r) = objLM;
        lmRetryAttempted(r) = lmStats.retry_attempted;
        lmRetryStatuses{r} = lmStats.retry_status;

        if isfinite(objLM)
            solutionBank = add_solution_to_bank(solutionBank,objLM,nuLM, ...
                c.delay,condLM,'analytic LM',c.seedIndex, ...
                solutionDuplicateTolerance);
        end

        if isfinite(objLM) && objLM < bestObjective
            bestObjective = objLM;
            best.nu = nuLM;
            best.delay = c.delay;
            best.B = BLM;
            best.rawNorms = normLM;
            best.condition = condLM;
            best.conditionLimit = c.conditionLimit;
            best.seedIndex = c.seedIndex;
            best.seedPairScore = c.seedPairScore;
            best.seedPairCoherence = c.seedPairCoherence;
            best.source = 'analytic variable-projection LM';
        end
    end

    % A small number of Nelder-Mead starts are retained as an independent
    % derivative-free cross-check. LM polishing is selective: a raw
    % endpoint must materially improve its own seed and beat the current best.
    % This guards against occasional poor local quadratic approximations while
    % avoiding the large cost of running fminsearch from every seed.
    fallbackConditionPenalty = get_cfg(cfg, ...
        'estimator_fallback_condition_penalty',0.02);
    if ~isscalar(fallbackConditionPenalty) || ...
            ~isfinite(fallbackConditionPenalty) || fallbackConditionPenalty < 0
        error('estimator_fallback_condition_penalty must be finite and nonnegative.');
    end
    fallbackConditionTrigger = get_cfg(cfg, ...
        'estimator_fallback_condition_trigger',10);
    if ~isscalar(fallbackConditionTrigger) || ...
            ~isfinite(fallbackConditionTrigger) || fallbackConditionTrigger<1
        error('estimator_fallback_condition_trigger must be finite and >= 1.');
    end
    fallbackIndices = select_fallback_refinement_candidates( ...
        candidates,refineIndices, ...
        max(0,round(get_cfg(cfg,'estimator_fminsearch_top_k',4))), ...
        fallbackConditionTrigger);
    fminStartCount = numel(fallbackIndices);
    fminStartSeedNumbers = zeros(fminStartCount,1);
    fminStartConditions = zeros(fminStartCount,1);
    fminProjectedEndpoints = 0;
    fminProjectionEvaluations = 0;
    if fminStartCount > 0
        fminOptions = optimset('Display','off', ...
            'MaxIter',get_cfg(cfg,'estimator_fminsearch_max_iter',180), ...
            'MaxFunEvals',get_cfg(cfg,'estimator_fminsearch_max_fun_evals',360), ...
            'TolX',get_cfg(cfg,'estimator_fminsearch_tol_x',1e-10), ...
            'TolFun',get_cfg(cfg,'estimator_fminsearch_tol_fun',1e-13));
        fminExitFlags = zeros(fminStartCount,1);
        fminEvaluations = 0;

        for r = 1:fminStartCount
            c = candidates(fallbackIndices(r));
            fminStartSeedNumbers(r) = c.seedIndex;
            fminStartConditions(r) = c.condition;
            useCenterSeparation = get_cfg(cfg, ...
                'estimator_close_pair_parameterization_enable',true) && ...
                c.delay(1)==c.delay(2) && ...
                abs(diff(c.nu)) <= get_cfg(cfg, ...
                'estimator_close_pair_nm_max_separation',0.08);
            if useCenterSeparation
                minimumSeparation = get_cfg(cfg, ...
                    'estimator_close_pair_nm_min_separation', ...
                    max(1e-4,0.10*gridStep));
                maximumSeparation = get_cfg(cfg, ...
                    'estimator_close_pair_nm_max_separation',0.08);
                u0 = encode_center_separation(c.nu,cfg.nu_max, ...
                    minimumSeparation,maximumSeparation);
                objectiveFunction = @(u) center_separation_fallback_objective( ...
                    u,c.delay(1),pairCache.pilotTx,y,cfg, ...
                    minPairDenominator,c.conditionLimit, ...
                    fallbackConditionPenalty,minimumSeparation, ...
                    maximumSeparation);
            else
                nu0 = min(max(c.nu/cfg.nu_max,-0.999999),0.999999);
                u0 = atanh(nu0(:));
                objectiveFunction = @(u) variable_projection_fallback_objective( ...
                    u,c.delay,pairCache.pilotTx,y,cfg, ...
                    minPairDenominator,c.conditionLimit,fallbackConditionPenalty);
            end
            % A close same-delay pair is optimized in (center,separation)
            % coordinates. This avoids asking a simplex to move two highly
            % correlated Dopplers independently. Other hypotheses retain the
            % original independent-Doppler parameterization.
            [uOpt,~,exitflag,output] = fminsearch( ...
                objectiveFunction,u0,fminOptions);
            fminExitFlags(r) = exitflag;
            if isfield(output,'funcCount')
                fminEvaluations = fminEvaluations+output.funcCount;
            end

            if useCenterSeparation
                nuFmin = decode_center_separation(uOpt,cfg.nu_max, ...
                    minimumSeparation,maximumSeparation);
            else
                nuFmin = (cfg.nu_max*tanh(uOpt(:))).';
            end
            % Always evaluate the raw simplex endpoint independently. In
            % Revision 14 it was considered only when LM polishing failed to
            % beat the global best, which could discard a better raw endpoint.
            [Bfmin,normFmin] = make_pair_atoms( ...
                pairCache.pilotTx,nuFmin,c.delay,cfg);
            [objFmin,~,~,condFmin] = evaluate_pair_ml( ...
                Bfmin,y,minPairDenominator,c.conditionLimit);

            % If the simplex ends outside the strict condition-valid region,
            % walk back along the line from its valid start to its endpoint.
            % This recovers the best valid point on that segment instead of
            % throwing away the entire derivative-free fallback result.
            if ~isfinite(objFmin)
                projectedNu = c.nu;
                projectedObjective = c.objective;
                projectedB = c.B;
                projectedNorms = c.rawNorms;
                projectedCondition = c.condition;
                projectionImproved = false;
                alphaValues = linspace(0.95,0.05,19);
                for alphaIndex = 1:numel(alphaValues)
                    alpha = alphaValues(alphaIndex);
                    nuTrial = c.nu+alpha*(nuFmin-c.nu);
                    [Btrial,normTrial] = make_pair_atoms( ...
                        pairCache.pilotTx,nuTrial,c.delay,cfg);
                    [objTrial,~,~,condTrial] = evaluate_pair_ml( ...
                        Btrial,y,minPairDenominator,c.conditionLimit);
                    fminProjectionEvaluations = fminProjectionEvaluations+1;
                    if isfinite(objTrial) && objTrial<projectedObjective
                        projectedNu = nuTrial;
                        projectedObjective = objTrial;
                        projectedB = Btrial;
                        projectedNorms = normTrial;
                        projectedCondition = condTrial;
                        projectionImproved = true;
                    end
                end
                if projectionImproved
                    nuFmin = projectedNu;
                    Bfmin = projectedB;
                    normFmin = projectedNorms;
                    objFmin = projectedObjective;
                    condFmin = projectedCondition;
                    fminProjectedEndpoints = fminProjectedEndpoints+1;
                else
                    % The original pattern seed remains a valid fallback.
                    nuFmin = c.nu;
                    Bfmin = c.B;
                    normFmin = c.rawNorms;
                    objFmin = c.objective;
                    condFmin = c.condition;
                end
            end

            if isfinite(objFmin)
                solutionBank = add_solution_to_bank(solutionBank,objFmin, ...
                    nuFmin,c.delay,condFmin, ...
                    'Nelder-Mead raw endpoint',c.seedIndex, ...
                    solutionDuplicateTolerance);
            end

            rawEndpointBeatsExistingBest = isfinite(objFmin) && ...
                objFmin < bestObjective;
            if rawEndpointBeatsExistingBest
                bestObjective = objFmin;
                best.nu = nuFmin;
                best.delay = c.delay;
                best.B = Bfmin;
                best.rawNorms = normFmin;
                best.condition = condFmin;
                best.conditionLimit = c.conditionLimit;
                best.seedIndex = c.seedIndex;
                best.seedPairScore = c.seedPairScore;
                best.seedPairCoherence = c.seedPairCoherence;
                best.source = 'Nelder-Mead fallback (unpolished)';
            end

            % A second LM solve is expensive and usually redundant because
            % the same basin has already received an LM start. Only polish a
            % simplex endpoint if it materially improves its own seed. This
            % preserves the independent raw endpoint in the hypothesis bank.
            % Revision20: polish every raw Nelder-Mead endpoint that can
            % become the global best.  The previous threshold compared the
            % endpoint only with its own seed; a small local improvement could
            % therefore win globally yet remain unpolished.  The analytic LM
            % result is accepted only when its strict ML objective improves,
            % so this cannot replace the raw endpoint with a worse fit.
            shouldPolish = isfinite(objFmin) && rawEndpointBeatsExistingBest;
            if shouldPolish
                [nuPolished,Bpolished,normPolished,objPolished, ...
                    condPolished,lmStats,retryCount] = ...
                    refine_pair_variable_projection_lm_with_retry( ...
                    pairCache.pilotTx,y,nuFmin,c.delay,cfg, ...
                    minPairDenominator,c.conditionLimit,lmMaxIter, ...
                    lmInitialDamping,lmMaxStep,lmGradientTolerance, ...
                    lmStepTolerance,lmRetryEnable && lmRetryCalls<lmRetryBudget);
                lmRetryCalls = lmRetryCalls+retryCount;
                lmCalls = lmCalls+1+retryCount;
                lmTotalIterations = lmTotalIterations+lmStats.total_iterations;
                lmTotalAcceptedSteps = lmTotalAcceptedSteps+lmStats.total_accepted_steps;

                if isfinite(objPolished)
                    solutionBank = add_solution_to_bank(solutionBank,objPolished, ...
                        nuPolished,c.delay,condPolished, ...
                        'Nelder-Mead plus LM',c.seedIndex, ...
                        solutionDuplicateTolerance);
                end
                if isfinite(objPolished) && objPolished < bestObjective
                    bestObjective = objPolished;
                    best.nu = nuPolished;
                    best.delay = c.delay;
                    best.B = Bpolished;
                    best.rawNorms = normPolished;
                    best.condition = condPolished;
                    best.conditionLimit = c.conditionLimit;
                    best.seedIndex = c.seedIndex;
                    best.seedPairScore = c.seedPairScore;
                    best.seedPairCoherence = c.seedPairCoherence;
                    best.source = 'Nelder-Mead followed by analytic LM';
                end
            end
        end
    else
        fminExitFlags = zeros(0,1);
        fminEvaluations = 0;
    end

    % Sort/compare distinct refined hypotheses. The delta log-likelihood
    % equals the residual-energy gap divided by complex-noise variance for
    % circular complex AWGN. A small gap means the observation does not
    % strongly distinguish the two path decompositions.
    ambiguityDeltaThreshold = get_cfg(cfg, ...
        'estimator_ambiguity_delta_log_likelihood',2.3);
    ambiguityMinSeparation = get_cfg(cfg, ...
        'estimator_ambiguity_min_location_separation', ...
        max(0.5*gridStep,1e-3));
    ambiguity = analyze_solution_ambiguity(solutionBank,bestObjective, ...
        best.nu,best.delay,best.condition,noiseVariance, ...
        ambiguityDeltaThreshold,ambiguityMinSeparation, ...
        get_cfg(cfg,'estimator_ambiguity_condition_warning',20));

    % ================================================================
    % FINAL PATH-SPECIFIC GAIN ESTIMATION
    % ================================================================
    %
    % The dictionary atoms are unit-normalized. With an approximately
    % equal-power channel prior E[|h_p|^2] = 1/P, the prior variance of
    % each unit-atom coefficient c_p = h_p*||a_p|| is ||a_p||^2/P.
    % The corresponding diagonal ridge parameter is therefore
    % lambda_p = noiseVariance*P/||a_p||^2.
    %
    % This regularization is applied only after path locations have been
    % selected. The location search above remains unregularized ML.

    rawNormsSafe = real(best.rawNorms(:));
    if numel(rawNormsSafe) ~= cfg.P || ...
            any(~isfinite(rawNormsSafe)) || ...
            any(rawNormsSafe <= sqrt(eps))
        error('Invalid raw atom norms for final gain estimation.');
    end

    ridgeMultiplier = get_cfg(cfg,'estimator_ridge_multiplier',1.0);
    if ~isscalar(ridgeMultiplier) || ~isfinite(ridgeMultiplier) || ...
            ridgeMultiplier < 0
        error('cfg.estimator_ridge_multiplier must be finite and nonnegative.');
    end

    % Path-specific noise-aware regularization.
    lambdaVector = ridgeMultiplier * noiseVariance * cfg.P ./ ...
        (rawNormsSafe.^2);

    % Solve the ridge problem as an augmented least-squares system instead
    % of forming B'*B.  This is algebraically identical, but avoids squaring
    % the condition number of a close-path dictionary pair.
    if all(lambdaVector==0)
        finalCoefficients = best.B \ y;
    else
        augmentedDictionary = [best.B;diag(sqrt(lambdaVector))];
        augmentedObservation = [y;zeros(cfg.P,1)];
        finalCoefficients = augmentedDictionary \ augmentedObservation;
    end

    % Convert normalized dictionary coefficients into physical gains.
    gains_est = finalCoefficients(:) ./ rawNormsSafe;

    % Local Fisher-information diagnostic. This is not used to change the ML
    % point estimate; it estimates how sensitive the two gains and Dopplers
    % are to noise in the selected local basin.
    % Estimate local identifiability around the unregularized ML fit. The
    % point estimate returned to the user remains the noise-regularized gain,
    % but the Fisher diagnostic should describe the local ML geometry rather
    % than inherit the shrinkage of the final ridge estimator.
    mlCoefficientsForInfo = best.B\y;
    localInfo = estimate_local_parameter_uncertainty( ...
        pairCache.pilotTx,best.nu,best.delay,cfg,mlCoefficientsForInfo, ...
        rawNormsSafe,noiseVariance);

    % Reconstruct the complete multi-frame received DD response.
    estimatedVector = best.B * finalCoefficients(:);
    residualVector = y - estimatedVector;

    H_est = reshape(estimatedVector,cfg.N,cfg.M,numTrainingFrames);
    residual = reshape(residualVector,cfg.N,cfg.M,numTrainingFrames);
    detected_paths = [best.nu(:),best.delay(:)];

    % ================================================================
    % ESTIMATOR DIAGNOSTICS (kept compatible with Revision 9 evaluator)
    % ================================================================

    info.selected_seed_number = best.seedIndex;
    info.selected_seed_pair_score = best.seedPairScore;
    info.selected_seed_pair_coherence = best.seedPairCoherence;
    info.best_objective = bestObjective;
    info.fit_nmse = norm(residualVector)^2/(yNormSquared+eps);

    info.selected_atom_coherence = abs(best.B(:,1)'*best.B(:,2));
    info.selected_pair_gram_condition = best.condition;
    info.selected_pair_condition_limit = best.conditionLimit;
    info.path_doppler_separation_bins = abs(diff(best.nu));
    info.ambiguity_flag = ambiguity.ambiguity_flag;
    info.multi_hypothesis_ambiguous = ambiguity.multi_hypothesis_ambiguous;
    info.near_degenerate_pair_flag = ambiguity.near_degenerate_pair_flag;
    info.ambiguity_delta_log_likelihood = ambiguity.delta_log_likelihood;
    info.ambiguity_delta_residual_energy = ambiguity.delta_residual_energy;
    info.ambiguity_alternative_doppler = ambiguity.alternative_doppler;
    info.ambiguity_alternative_delay = ambiguity.alternative_delay;
    info.ambiguity_alternative_condition = ambiguity.alternative_condition;
    info.ambiguity_alternative_source = ambiguity.alternative_source;
    info.ambiguity_competitor_count = ambiguity.competitor_count;
    info.ambiguity_competitors_within_threshold = ...
        ambiguity.competitors_within_threshold;
    info.ambiguity_delta_log_likelihood_threshold = ...
        ambiguityDeltaThreshold;
    info.ambiguity_min_location_separation = ambiguityMinSeparation;
    info.solution_hypotheses_retained = numel(solutionBank);
    info.close_pair_nm_starts = nnz(arrayfun(@(idx) ...
        candidates(idx).delay(1)==candidates(idx).delay(2) && ...
        abs(diff(candidates(idx).nu)) <= get_cfg(cfg, ...
        'estimator_close_pair_nm_max_separation',0.08),fallbackIndices));

    % Scalar field retained for compatibility; vector gives each path.
    info.regularization_lambda = mean(lambdaVector);
    info.regularization_lambda_per_path = lambdaVector;
    info.raw_atom_norms_per_path = rawNormsSafe;
    info.final_coefficient_magnitudes = abs(finalCoefficients(:));
    info.local_doppler_std_bins = localInfo.doppler_std_bins;
    info.local_complex_gain_std_approx = localInfo.gain_std_approx;
    info.local_relative_gain_std_approx = localInfo.relative_gain_std_approx;
    info.local_information_condition = localInfo.information_condition;
    info.local_information_rank = localInfo.information_rank;
    info.local_covariance_valid = localInfo.covariance_valid;
    info.local_gain_uncertainty_flag = localInfo.gain_uncertainty_flag || ...
        ambiguity.ambiguity_flag;

    info.noise_variance = noiseVariance;
    info.num_training_frames = numTrainingFrames;
    info.observation_samples = expectedObservations;

    info.refinement_evaluations = totalPairEvaluations;
    info.refinement_atom_evaluations = totalAtomEvaluations;
    info.nonlinear_refinement_evaluations = fminEvaluations;
    info.nonlinear_refinement_exitflags = fminExitFlags;
    info.nelder_mead_candidate_indices = fallbackIndices;
    info.nelder_mead_seed_numbers = fminStartSeedNumbers;
    info.nelder_mead_start_conditions = fminStartConditions;
    info.nelder_mead_projected_endpoints = fminProjectedEndpoints;
    info.nelder_mead_projection_evaluations = fminProjectionEvaluations;
    info.fallback_condition_penalty = fallbackConditionPenalty;
    info.fallback_condition_trigger = fallbackConditionTrigger;
    info.selected_refinement_source = best.source;

    info.lm_calls = lmCalls;
    info.lm_retry_calls = lmRetryCalls;
    info.lm_retry_budget = lmRetryBudget;
    info.lm_retry_enable = lmRetryEnable;
    info.lm_retry_attempted_by_candidate = lmRetryAttempted;
    info.lm_retry_status_by_candidate = lmRetryStatuses;
    info.lm_total_iterations = lmTotalIterations;
    info.lm_total_accepted_steps = lmTotalAcceptedSteps;
    info.lm_final_gradients = lmFinalGradients;
    info.lm_candidate_objectives = lmFinalObjectives;

    info.refinement_levels = refineLevels;
    info.top_candidate_pairs = topK;
    info.nonlinear_refinement_starts = numel(refineIndices);
    info.derivative_method = ...
        'analytic time-domain Doppler derivative through stacked OTFS frame demodulators';

    info.coarse_seed_count = numel(coarsePatternSeeds);
    info.close_pair_seed_count = numel(closePairSeedIndices);
    info.close_pair_pattern_search_starts = closePairPatternStarts;
    info.close_pair_search_max_separation = closePatternMaxSeparation;
    info.close_pair_pattern_refine_levels = get_cfg(cfg, ...
        'estimator_close_pair_refine_levels',4);
    info.close_pair_pattern_search_enabled = true;
    info.residual_seed_count = numel(residualPairSeedIndices);
    info.residual_seed_enable = residualSeedEnable;
    info.residual_seed_top_atoms = residualFirstAtomCount;
    info.residual_seed_second_atoms = residualSecondAtomCount;
    info.close_pair_seed_enable = closeSeedEnable;
    info.close_pair_seed_min_grid_steps = closeSeedMinSteps;
    info.close_pair_seed_max_grid_steps = closeSeedMaxSteps;
    info.doppler_grid_spacing = gridStep;
    info.coarse_seed_limit = get_cfg(cfg,'estimator_refine_top_k',8);
    info.coarse_seeds_per_delay_pair = seedsPerDelayPair;
    info.coarse_seed_min_separation_steps = ...
        get_cfg(cfg,'estimator_seed_min_separation_steps',2.5);
    info.pattern_refinement_levels = refineLevels;
    info.nonlinear_seed_limit = ...
        get_cfg(cfg,'estimator_fminsearch_top_k',12);
    info.nonlinear_seeds_per_delay_pair = ...
        get_cfg(cfg,'estimator_nonlinear_seeds_per_delay_pair',2);
    info.refined_seed_min_separation_steps = ...
        get_cfg(cfg,'estimator_refined_seed_separation_steps',0.5);

    info.candidate_seed_objectives = [candidates.objective].';
    info.candidate_seed_dopplers = reshape([candidates.nu],2,[]).';
    info.candidate_seed_delays = reshape([candidates.delay],2,[]).';
    info.candidate_seed_is_close = [candidates.isCloseSeed].';
    info.candidate_seed_is_residual = [candidates.isResidualSeed].';

    info.refinement_method = ...
        'budgeted coarse/residual/close-pair search + center/separation LM + selective Nelder-Mead fallback';
    info.path_location_objective = ...
        'unregularized least-squares residual energy';
    info.condition_number_role = ...
        'bounded SNR-adaptive pair-validity safeguard; condition is also exposed in ambiguity diagnostics';
    info.max_pair_condition_used = maxPairCondition;
    info.base_max_pair_condition = baseMaxPairCondition;
    info.adaptive_condition_ceiling = conditionCeiling;
    info.condition_snr_exponent = conditionSNRExponent;
    info.condition_snr_reference_linear = conditionSNRReference;
    info.pattern_seed_budget = patternSeedBudget;
    info.pattern_coarse_seed_budget = coarsePatternBudget;
    info.pattern_close_seed_budget = closePatternBudget;
    info.pattern_residual_seed_budget = residualPatternBudget;
    info.pattern_seeds_used = topK;
    info.close_pair_condition_ceiling = closePairConditionCeiling;
    info.close_pair_seed_condition_penalty = closeSeedPenaltyWeight;
    info.close_pair_condition_snr_exponent = closeConditionSNRExponent;
    info.close_pair_separation_bands_steps = closeSeparationBands;
    if isempty(closePairSeedIndices)
        info.close_pair_seed_separation_steps = zeros(0,1);
    else
        closeI = pairI(closePairSeedIndices);
        closeJ = pairJ(closePairSeedIndices);
        closeNuI = dopplerGrid(mod(closeI-1,numDoppler)+1);
        closeNuJ = dopplerGrid(mod(closeJ-1,numDoppler)+1);
        info.close_pair_seed_separation_steps = ...
            abs(closeNuI(:)-closeNuJ(:))/gridStep;
    end
    info.coarse_projection_score_method = ...
        'orthogonalized two-column QR projection';
    info.final_gain_solver = 'augmented least-squares QR ridge solve';
    info.lm_candidate_budget = lmCandidateBudget;
    info.lm_candidates_per_delay_class = lmCandidatesPerDelayClass;
    info.estimated_observation_snr_db = estimatedObservationSNRdB;
    info.estimated_observation_snr_linear = estimatedObservationSNRLinear;
    info.estimated_signal_power = estimatedSignalPower;
    info.valid_pair_count_after_adaptation = nnz(validPair);
    info.valid_pair_count_in_cached_mask = cachedValidCount;
    info.newly_admitted_pair_count = newlyAdmittedPairCount;
    info.physical_gain_convention = ...
        'path-specific ridge unit-atom coefficients ./ raw atom norms';
end

function bank = empty_solution_bank()
    bank = repmat(struct('objective',{},'nu',{},'delay',{}, ...
        'condition',{},'source',{},'seedIndex',{}),0,1);
end

function bank = add_solution_to_bank(bank,objective,nu,delay,condition, ...
    source,seedIndex,duplicateTolerance)
% Canonicalize path order and retain the best objective for each basin.
    if ~isfinite(objective) || any(~isfinite(nu)) || ...
            any(~isfinite(delay)) || ~isfinite(condition)
        return;
    end
    pair = sortrows([delay(:),nu(:)],[1 2]);
    nuCanonical = pair(:,2).';
    delayCanonical = pair(:,1).';
    for q = 1:numel(bank)
        if isequal(bank(q).delay,delayCanonical) && ...
                max(abs(bank(q).nu-nuCanonical)) <= duplicateTolerance
            if objective < bank(q).objective
                bank(q).objective = objective;
                bank(q).nu = nuCanonical;
                bank(q).condition = condition;
                bank(q).source = source;
                bank(q).seedIndex = seedIndex;
            end
            return;
        end
    end
    item.objective = objective;
    item.nu = nuCanonical;
    item.delay = delayCanonical;
    item.condition = condition;
    item.source = source;
    item.seedIndex = seedIndex;
    bank(end+1,1) = item;
end

function result = analyze_solution_ambiguity(bank,bestObjective,bestNu, ...
    bestDelay,bestCondition,noiseVariance,deltaThreshold,minLocationSeparation, ...
    conditionWarning)
% Compare the selected solution against distinct retained refined hypotheses.
    result.ambiguity_flag = false;
    result.multi_hypothesis_ambiguous = false;
    result.near_degenerate_pair_flag = bestCondition >= conditionWarning;
    result.delta_log_likelihood = Inf;
    result.delta_residual_energy = Inf;
    result.alternative_doppler = [NaN NaN];
    result.alternative_delay = [NaN NaN];
    result.alternative_condition = Inf;
    result.alternative_source = 'none';
    result.competitor_count = 0;
    result.competitors_within_threshold = 0;
    bestPair = sortrows([bestDelay(:),bestNu(:)],[1 2]);
    bestNu = bestPair(:,2).';
    bestDelay = bestPair(:,1).';
    if isempty(bank)
        result.ambiguity_flag = result.near_degenerate_pair_flag;
        return;
    end
    objectives = [bank.objective].';
    [~,order] = sort(objectives,'ascend');
    bestAlternativeIndex = 0;
    bestAlternativeGap = Inf;
    for q = 1:numel(order)
        idx = order(q);
        candidate = bank(idx);
        sameDelay = isequal(candidate.delay,bestDelay);
        if sameDelay
            locationDistance = max(abs(candidate.nu-bestNu));
            isDistinct = locationDistance >= minLocationSeparation;
        else
            isDistinct = true;
        end
        if ~isDistinct
            continue;
        end
        result.competitor_count = result.competitor_count+1;
        gap = max(0,candidate.objective-bestObjective);
        if noiseVariance > 0
            deltaLL = gap/noiseVariance;
        else
            deltaLL = Inf;
        end
        if deltaLL <= deltaThreshold
            result.competitors_within_threshold = ...
                result.competitors_within_threshold+1;
        end
        if gap < bestAlternativeGap
            bestAlternativeGap = gap;
            bestAlternativeIndex = idx;
            result.delta_log_likelihood = deltaLL;
            result.delta_residual_energy = gap;
        end
    end
    if bestAlternativeIndex > 0
        alt = bank(bestAlternativeIndex);
        result.alternative_doppler = alt.nu;
        result.alternative_delay = alt.delay;
        result.alternative_condition = alt.condition;
        result.alternative_source = alt.source;
        result.multi_hypothesis_ambiguous = ...
            result.delta_log_likelihood <= deltaThreshold;
    end
    result.near_degenerate_pair_flag = bestCondition >= conditionWarning;
    result.ambiguity_flag = result.multi_hypothesis_ambiguous || ...
        result.near_degenerate_pair_flag;
end

function u = encode_center_separation(nu,nuMax,minSeparation,maxSeparation)
    nu = sort(nu(:));
    separation = min(maxSeparation,max(minSeparation,nu(2)-nu(1)));
    center = mean(nu);
    centerLimit = max(nuMax-separation/2,eps);
    center = min(max(center,-centerLimit),centerLimit);
    u = [atanh(min(max(center/nuMax,-0.999999),0.999999)); ...
        log(separation)];
end

function nu = decode_center_separation(u,nuMax,minSeparation,maxSeparation)
    center = nuMax*tanh(u(1));
    separation = min(maxSeparation,max(minSeparation,exp(u(2))));
    centerLimit = max(nuMax-separation/2,0);
    center = min(max(center,-centerLimit),centerLimit);
    nu = [center-separation/2,center+separation/2];
end

function objective = center_separation_fallback_objective(u,delay, ...
    pilotTx,y,cfg,minPairDenominator,maxPairCondition,conditionPenalty, ...
    minSeparation,maxSeparation)
    if numel(u)~=2 || any(~isfinite(u))
        objective = 1e6+real(y'*y);
        return;
    end
    nu = decode_center_separation(u,cfg.nu_max,minSeparation,maxSeparation);
    [B,~] = make_pair_atoms(pilotTx,nu,[delay delay],cfg);
    cross = B(:,1)'*B(:,2);
    denominator = max(0,1-abs(cross)^2);
    condition = (1+abs(cross))/max(1-abs(cross),eps);
    if ~isfinite(denominator) || denominator<=minPairDenominator || ...
            ~isfinite(condition)
        objective = 1e6+real(y'*y);
        return;
    end
    coeff = B\y;
    residual = y-B*coeff;
    objective = real(residual'*residual);
    if ~isfinite(objective)
        objective = 1e6+real(y'*y);
        return;
    end
    if condition>maxPairCondition && conditionPenalty>0
        objective = objective+conditionPenalty*real(y'*y)* ...
            log(condition/maxPairCondition)^2;
    end
end

function selected = ordered_seed_subset(indices,maxCount)
% Preserve source ranking for seeds produced by dedicated search procedures.
    indices = unique(indices(:),'stable');
    maxCount = max(0,round(maxCount));
    selected = indices(1:min(maxCount,numel(indices)));
end

function selected = top_pair_seed_subset(indices,pairScore,maxCount)
% Return unique candidate indices ranked by their coarse pair projection score.
    indices = unique(indices(:),'stable');
    maxCount = max(0,round(maxCount));
    if isempty(indices) || maxCount == 0
        selected = zeros(0,1);
        return;
    end
    finiteMask = isfinite(pairScore(indices));
    indices = indices(finiteMask);
    if isempty(indices)
        selected = zeros(0,1);
        return;
    end
    [~,order] = sort(pairScore(indices),'descend');
    selected = indices(order(1:min(maxCount,numel(order))));
end

function selected = select_diverse_pair_seeds(pairScore,pairI,pairJ, ...
    numAtoms,numDoppler,numDelay,dopplerGrid,maxSeeds,minSeparation, ...
    seedsPerDelayPair)
% Select multiple local maxima from every unordered delay-pair class.
% The pair score is a two-atom variable-projection likelihood score.
% Selecting local maxima, instead of only the highest raw scores, preserves
% alternative Doppler hypotheses that may lead to different continuous
% minima during the subsequent nonlinear refinement.

    pairScore = pairScore(:);
    pairI = pairI(:);
    pairJ = pairJ(:);
    dopplerGrid = dopplerGrid(:);

    atomLinearIndex = (1:numAtoms).';
    atomDelayIndex = floor((atomLinearIndex-1)/numDoppler)+1;
    atomDopplerIndex = mod(atomLinearIndex-1,numDoppler)+1;
    delayI = atomDelayIndex(pairI);
    delayJ = atomDelayIndex(pairJ);
    dopplerI = atomDopplerIndex(pairI);
    dopplerJ = atomDopplerIndex(pairJ);
    nuI = dopplerGrid(dopplerI);
    nuJ = dopplerGrid(dopplerJ);

    selected = zeros(0,1);

    % The atom ordering is delay-major and Doppler-minor. Therefore pairI
    % always belongs to the lower/equal delay class, and the pair's Doppler
    % coordinates can be placed directly in a 2-D score map.
    for d1 = 1:numDelay
        for d2 = d1:numDelay
            group = find(isfinite(pairScore) & ...
                delayI==d1 & delayJ==d2);
            if isempty(group)
                continue;
            end

            scoreMap = -Inf(numDoppler,numDoppler);
            pairMap = zeros(numDoppler,numDoppler);
            for q = 1:numel(group)
                pairNumber = group(q);
                i = dopplerI(pairNumber);
                j = dopplerJ(pairNumber);
                scoreMap(i,j) = pairScore(pairNumber);
                pairMap(i,j) = pairNumber;
            end

            % Extract 8-neighbour local maxima in this delay-pair class.
            isLocalMaximum = false(numDoppler,numDoppler);
            for i = 1:numDoppler
                iRange = max(1,i-1):min(numDoppler,i+1);
                for j = 1:numDoppler
                    if ~isfinite(scoreMap(i,j))
                        continue;
                    end
                    jRange = max(1,j-1):min(numDoppler,j+1);
                    neighbourhood = scoreMap(iRange,jRange);
                    isLocalMaximum(i,j) = ...
                        scoreMap(i,j) >= max(neighbourhood(:));
                end
            end

            peakPairs = pairMap(isLocalMaximum);
            peakPairs = peakPairs(peakPairs>0);
            if isempty(peakPairs)
                peakPairs = group;
            end
            [~,peakOrder] = sort(pairScore(peakPairs),'descend');
            peakPairs = peakPairs(peakOrder);

            classSelected = zeros(0,1);
            for q = 1:numel(peakPairs)
                candidate = peakPairs(q);
                duplicate = false;
                for k = 1:numel(classSelected)
                    previous = classSelected(k);
                    separation = max(abs([ ...
                        nuI(candidate)-nuI(previous), ...
                        nuJ(candidate)-nuJ(previous)]));
                    if separation < minSeparation
                        duplicate = true;
                        break;
                    end
                end
                if ~duplicate
                    classSelected(end+1,1) = candidate; %#ok<AGROW>
                    if numel(classSelected)>=seedsPerDelayPair
                        break;
                    end
                end
            end

            % If the number of local maxima was insufficient, fill from
            % the best remaining samples while applying the same separation.
            if numel(classSelected)<seedsPerDelayPair
                [~,fallbackOrder] = sort(pairScore(group),'descend');
                fallbackPairs = group(fallbackOrder);
                for q = 1:numel(fallbackPairs)
                    candidate = fallbackPairs(q);
                    if any(classSelected==candidate)
                        continue;
                    end
                    duplicate = false;
                    for k = 1:numel(classSelected)
                        previous = classSelected(k);
                        separation = max(abs([ ...
                            nuI(candidate)-nuI(previous), ...
                            nuJ(candidate)-nuJ(previous)]));
                        if separation < minSeparation
                            duplicate = true;
                            break;
                        end
                    end
                    if ~duplicate
                        classSelected(end+1,1) = candidate; %#ok<AGROW>
                        if numel(classSelected)>=seedsPerDelayPair
                            break;
                        end
                    end
                end
            end

            selected = [selected;classSelected]; %#ok<AGROW>
        end
    end

    if isempty(selected)
        error('No valid pair seeds were selected. Check the dictionary pair cache.');
    end

    [~,globalOrder] = sort(pairScore(selected),'descend');
    selected = selected(globalOrder);
    selected = selected(1:min(maxSeeds,numel(selected)));
end

function selected = select_close_same_delay_seeds(pairScore,pairI,pairJ, ...
    numAtoms,numDoppler,dopplerGrid,gridStep,minSeparationSteps, ...
    maxSeparationSteps,seedsPerDelay,separationBands)
% SELECT_CLOSE_SAME_DELAY_SEEDS
% Reserve same-delay candidates across separation bands.  The previous
% global top-score policy could spend the entire close-pair quota on a
% relatively wide separation range and leave the 0.5-1.5-grid-step region
% unexplored.  One high-scoring candidate is first reserved from each band,
% then any remaining quota is filled by the best diverse candidates.
%
% The selected set is still only an initialization pool.  The final result
% is determined by the original unregularized ML residual.

    pairScore = pairScore(:);
    pairI = pairI(:);
    pairJ = pairJ(:);
    dopplerGrid = dopplerGrid(:);
    atomLinearIndex = (1:numAtoms).';
    atomDelayIndex = floor((atomLinearIndex-1)/numDoppler)+1;
    atomDopplerIndex = mod(atomLinearIndex-1,numDoppler)+1;
    delayI = atomDelayIndex(pairI);
    delayJ = atomDelayIndex(pairJ);
    nuI = dopplerGrid(atomDopplerIndex(pairI));
    nuJ = dopplerGrid(atomDopplerIndex(pairJ));
    separationSteps = abs(nuI-nuJ)/gridStep;

    selected = zeros(0,1);
    sameDelayClasses = unique(delayI(delayI==delayJ));
    for q = 1:numel(sameDelayClasses)
        d = sameDelayClasses(q);
        eligible = find(isfinite(pairScore) & delayI==d & delayJ==d & ...
            separationSteps>=minSeparationSteps & ...
            separationSteps<=maxSeparationSteps);
        if isempty(eligible)
            continue;
        end
        classSelected = zeros(0,1);

        % Reserve coverage from the smallest separation band first.  The
        % half-open band convention avoids selecting a boundary pair twice.
        for bandIndex = 1:size(separationBands,1)
            if numel(classSelected)>=seedsPerDelay
                break;
            end
            lo = max(minSeparationSteps,separationBands(bandIndex,1));
            hi = min(maxSeparationSteps,separationBands(bandIndex,2));
            if hi<=lo
                continue;
            end
            if bandIndex==size(separationBands,1)
                inBand = separationSteps(eligible)>=lo & ...
                    separationSteps(eligible)<=hi;
            else
                inBand = separationSteps(eligible)>=lo & ...
                    separationSteps(eligible)<hi;
            end
            bandCandidates = eligible(inBand);
            if isempty(bandCandidates)
                continue;
            end
            [~,bandOrder] = sort(pairScore(bandCandidates),'descend');
            bandCandidates = bandCandidates(bandOrder);
            for k = 1:numel(bandCandidates)
                candidate = bandCandidates(k);
                if any(classSelected==candidate)
                    continue;
                end
                if is_close_seed_duplicate(candidate,classSelected,nuI,nuJ, ...
                        0.75*gridStep)
                    continue;
                end
                classSelected(end+1,1) = candidate; %#ok<AGROW>
                break;
            end
        end

        % Fill any unoccupied band slots from the global ranked list while
        % preserving Doppler-pair diversity.  Coarse seeds are allowed here
        % too because the caller needs to mark them for close-path refinement.
        if numel(classSelected)<seedsPerDelay
            [~,order] = sort(pairScore(eligible),'descend');
            ranked = eligible(order);
            for k = 1:numel(ranked)
                candidate = ranked(k);
                if any(classSelected==candidate)
                    continue;
                end
                if is_close_seed_duplicate(candidate,classSelected,nuI,nuJ, ...
                        0.75*gridStep)
                    continue;
                end
                classSelected(end+1,1) = candidate; %#ok<AGROW>
                if numel(classSelected)>=seedsPerDelay
                    break;
                end
            end
        end
        selected = [selected;classSelected(:)]; %#ok<AGROW>
    end
end

function duplicate = is_close_seed_duplicate(candidate,classSelected,nuI,nuJ,tolerance)
    duplicate = false;
    candidateNu = sort([nuI(candidate),nuJ(candidate)]);
    for q = 1:numel(classSelected)
        previous = classSelected(q);
        previousNu = sort([nuI(previous),nuJ(previous)]);
        if max(abs(candidateNu-previousNu))<tolerance
            duplicate = true;
            return;
        end
    end
end

function selected = select_residual_pair_seeds(numAtoms,singleCorrelation, ...
    pairI,pairJ,pairCross,validPair,numFirstAtoms,numSecondAtoms)
% SELECT_RESIDUAL_PAIR_SEEDS
% Build OMP-style pair initializations. For each strong single-atom
% hypothesis, subtract its least-squares contribution and find atoms that
% explain the residual. Return the corresponding unordered pair indices.
%
% These are only starting points: the final estimator still jointly
% optimizes both Dopplers and both complex gains using the same ML objective.

    numAtoms = round(numAtoms);
    if ~isscalar(numAtoms) || ~isfinite(numAtoms) || numAtoms < 2
        error('Residual seed search requires at least two dictionary atoms.');
    end
    numFirstAtoms = max(1,min(numAtoms,round(numFirstAtoms)));
    numSecondAtoms = max(1,min(numAtoms-1,round(numSecondAtoms)));

    singleCorrelation = singleCorrelation(:);
    pairI = pairI(:);
    pairJ = pairJ(:);
    pairCross = pairCross(:);
    validPair = logical(validPair(:));
    if numel(singleCorrelation)~=numAtoms || numel(pairI)~=numel(pairJ) || ...
            numel(pairI)~=numel(pairCross) || numel(pairI)~=numel(validPair)
        error('Residual seed correlations and pair-cache arrays have inconsistent dimensions.');
    end
    if any(pairI<1 | pairJ>numAtoms | pairI>=pairJ) || ...
            any(~isfinite(pairCross))
        error('Residual seed pair cache must contain finite unordered pairs i<j.');
    end
    [~,firstOrder] = sort(abs(singleCorrelation).^2,'descend');
    firstOrder = firstOrder(1:numFirstAtoms);

    % Reconstruct the Hermitian Gram matrix from the pair cache once.  The
    % dictionary builder has already computed these inner products as part of
    % pair-cache construction.  This replaces one dense A'*residual product
    % per first-atom hypothesis with a sparse Gram-column multiply.
    pairGram = sparse([pairI;pairJ],[pairJ;pairI], ...
        [pairCross;conj(pairCross)],numAtoms,numAtoms);
    pairGram = pairGram + speye(numAtoms);

    % Sparse lookup maps an unordered atom pair directly to pairCache row.
    pairLookup = sparse(pairI,pairJ,(1:numel(pairI)).',numAtoms,numAtoms);
    selected = zeros(0,1);

    for q = 1:numel(firstOrder)
        firstAtom = firstOrder(q);
        firstCoefficient = singleCorrelation(firstAtom);
        residualCorrelation = abs(singleCorrelation - ...
            pairGram(:,firstAtom)*firstCoefficient).^2;
        residualCorrelation(firstAtom) = -Inf;
        [~,secondOrder] = sort(residualCorrelation,'descend');

        addedForFirst = 0;
        for k = 1:numel(secondOrder)
            secondAtom = secondOrder(k);
            if secondAtom == firstAtom || ...
                    ~isfinite(residualCorrelation(secondAtom))
                continue;
            end
            lo = min(firstAtom,secondAtom);
            hi = max(firstAtom,secondAtom);
            pairNumber = full(pairLookup(lo,hi));
            if pairNumber < 1 || ~validPair(pairNumber) || ...
                    any(selected==pairNumber)
                continue;
            end
            selected(end+1,1) = pairNumber; %#ok<AGROW>
            addedForFirst = addedForFirst+1;
            if addedForFirst >= numSecondAtoms
                break;
            end
        end
    end
    selected = unique(selected(:),'stable');
end

function selected = select_nonlinear_refinement_candidates( ...
    candidates,maxCount,maxPerDelayPair,minSeparation)
% Keep multiple distinct pattern-search minima per delay-pair class.
% Always take the best candidate from each class first, then fill remaining
% slots with additional nonduplicate minima ranked by residual objective.

    selected = zeros(0,1);
    if isempty(candidates)
        return;
    end

    objectives = [candidates.objective].';
    delayPairs = zeros(numel(candidates),2);
    for k = 1:numel(candidates)
        delayPairs(k,:) = sort(candidates(k).delay(:)).';
    end
    uniquePairs = unique(delayPairs,'rows','stable');
    [~,globalOrder] = sort(objectives,'ascend');
    classCounts = zeros(size(uniquePairs,1),1);

    % First guarantee representation of every unordered delay-pair class.
    for classIndex = 1:size(uniquePairs,1)
        group = find(delayPairs(:,1)==uniquePairs(classIndex,1) & ...
            delayPairs(:,2)==uniquePairs(classIndex,2));
        [~,order] = sort(objectives(group),'ascend');
        for q = 1:numel(order)
            candidate = group(order(q));
            if ~isfinite(objectives(candidate))
                continue;
            end
            if is_duplicate_refined_candidate(candidate,selected, ...
                    candidates,delayPairs,minSeparation)
                continue;
            end
            selected(end+1,1) = candidate; %#ok<AGROW>
            classCounts(classIndex) = 1;
            break;
        end
    end

    % Give reserved close-pair and residual-based hypotheses a guaranteed
    % refinement opportunity when they produced distinct finite minima. This
    % does not force them to win: every refined solution uses the same ML loss.
    isReserved = ([candidates.isCloseSeed].' | ...
        [candidates.isResidualSeed].');
    reservedOrder = find(isReserved & isfinite(objectives));
    if ~isempty(reservedOrder)
        [~,reservedRank] = sort(objectives(reservedOrder),'ascend');
        reservedOrder = reservedOrder(reservedRank);
        for q = 1:numel(reservedOrder)
            if numel(selected) >= maxCount
                break;
            end
            candidate = reservedOrder(q);
            if any(selected==candidate)
                continue;
            end
            classIndex = find(uniquePairs(:,1)==delayPairs(candidate,1) & ...
                uniquePairs(:,2)==delayPairs(candidate,2),1);
            if isempty(classIndex) || classCounts(classIndex)>=maxPerDelayPair
                continue;
            end
            if is_duplicate_refined_candidate(candidate,selected, ...
                    candidates,delayPairs,minSeparation)
                continue;
            end
            selected(end+1,1) = candidate; %#ok<AGROW>
            classCounts(classIndex) = classCounts(classIndex)+1;
        end
    end

    % Then retain additional local minima, including multiple hypotheses
    % from the same delay class, up to the configured per-class limit.
    for q = 1:numel(globalOrder)
        if numel(selected) >= maxCount
            break;
        end
        candidate = globalOrder(q);
        if any(selected==candidate) || ~isfinite(objectives(candidate))
            continue;
        end
        classIndex = find(uniquePairs(:,1)==delayPairs(candidate,1) & ...
            uniquePairs(:,2)==delayPairs(candidate,2),1);
        if isempty(classIndex) || classCounts(classIndex)>=maxPerDelayPair
            continue;
        end
        if is_duplicate_refined_candidate(candidate,selected, ...
                candidates,delayPairs,minSeparation)
            continue;
        end
        selected(end+1,1) = candidate; %#ok<AGROW>
        classCounts(classIndex) = classCounts(classIndex)+1;
    end

    if numel(selected)>maxCount
        selected = selected(1:maxCount);
    end
end

function duplicate = is_duplicate_refined_candidate(candidate,selected, ...
    candidates,delayPairs,minSeparation)
    duplicate = false;
    for q = 1:numel(selected)
        previous = selected(q);
        if all(delayPairs(candidate,:)==delayPairs(previous,:))
            separation = max(abs(candidates(candidate).nu- ...
                candidates(previous).nu));
            if separation < minSeparation
                duplicate = true;
                return;
            end
        end
    end
end


function selected = select_fallback_refinement_candidates( ...
    candidates,eligibleIndices,maxCount,conditionTrigger)
% SELECT_FALLBACK_REFINEMENT_CANDIDATES
% Select derivative-free starts by source diversity and conditioning rather
% than blindly taking the first entries of the LM candidate list.
    selected = zeros(0,1);
    maxCount = max(0,round(maxCount));
    eligibleIndices = unique(eligibleIndices(:),'stable');
    if maxCount==0 || isempty(eligibleIndices)
        return;
    end

    objectives = [candidates.objective].';
    conditions = [candidates.condition].';
    finiteEligible = eligibleIndices(isfinite(objectives(eligibleIndices)) & ...
        isfinite(conditions(eligibleIndices)));
    if isempty(finiteEligible)
        return;
    end

    % Priority 1: best overall ML basin.
    selected = append_best_fallback_candidate(selected,finiteEligible, ...
        objectives,maxCount);

    % Priority 2/3: explicitly include close-pair and residual-search basins.
    closePool = finiteEligible([candidates(finiteEligible).isCloseSeed].');
    selected = append_best_fallback_candidate(selected,closePool, ...
        objectives,maxCount);
    residualPool = finiteEligible([candidates(finiteEligible).isResidualSeed].');
    selected = append_best_fallback_candidate(selected,residualPool, ...
        objectives,maxCount);

    % Priority 4: include a numerically difficult (high-condition) basin.
    highConditionPool = finiteEligible(conditions(finiteEligible)>=conditionTrigger);
    if ~isempty(highConditionPool) && numel(selected)<maxCount
        [~,order] = sort(conditions(highConditionPool),'descend');
        selected = append_one_fallback_candidate(selected, ...
            highConditionPool(order(1)),maxCount);
    end

    % Fill remaining slots with the best candidate from additional delay
    % classes first, then with globally ranked distinct candidates.
    delayPairs = zeros(numel(candidates),2);
    for k=1:numel(candidates)
        delayPairs(k,:) = sort(candidates(k).delay(:)).';
    end
    classList = unique(delayPairs(finiteEligible,:),'rows','stable');
    for k=1:size(classList,1)
        if numel(selected)>=maxCount
            break;
        end
        mask = delayPairs(finiteEligible,1)==classList(k,1) & ...
            delayPairs(finiteEligible,2)==classList(k,2);
        pool = finiteEligible(mask);
        selected = append_best_fallback_candidate(selected,pool, ...
            objectives,maxCount);
    end

    [~,globalOrder] = sort(objectives(finiteEligible),'ascend');
    for k=1:numel(globalOrder)
        if numel(selected)>=maxCount
            break;
        end
        selected = append_one_fallback_candidate(selected, ...
            finiteEligible(globalOrder(k)),maxCount);
    end
end

function selected = append_best_fallback_candidate( ...
    selected,pool,objectives,maxCount)
    if isempty(pool) || numel(selected)>=maxCount
        return;
    end
    pool = pool(~ismember(pool,selected));
    if isempty(pool)
        return;
    end
    [~,order] = sort(objectives(pool),'ascend');
    selected = append_one_fallback_candidate(selected,pool(order(1)),maxCount);
end

function selected = append_one_fallback_candidate(selected,candidate,maxCount)
    if numel(selected)<maxCount && ~any(selected==candidate)
        selected(end+1,1) = candidate; %#ok<AGROW>
    end
end

function [nu,B,rawNorms,objective,pairCondition,stats,retryCount] = ...
    refine_pair_variable_projection_lm_with_retry( ...
    pilotTx,y,nu0,delay,cfg,minPairDenominator,maxPairCondition, ...
    maxIter,initialDamping,maxStep,gradientTolerance,stepTolerance, ...
    retryEnable)
% Run the primary analytic LM solve and, only when it stalls, one solve with
% an alternate initial damping. The better *valid ML* solution is retained.
    [nu,B,rawNorms,objective,pairCondition,primaryStats] = ...
        refine_pair_variable_projection_lm( ...
        pilotTx,y,nu0,delay,cfg,minPairDenominator,maxPairCondition, ...
        maxIter,initialDamping,maxStep,gradientTolerance,stepTolerance);

    stats = primaryStats;
    stats.total_iterations = primaryStats.iterations;
    stats.total_accepted_steps = primaryStats.accepted_steps;
    stats.retry_attempted = false;
    stats.retry_status = 'not needed';
    stats.primary_status = primaryStats.status;
    retryCount = 0;

    if ~retryEnable
        stats.retry_status = 'disabled or retry budget exhausted';
        return;
    end

    gradientScale = max(1,norm(y)^2);
    notConverged = ~isfinite(primaryStats.final_gradient) || ...
        primaryStats.final_gradient > ...
        100*gradientTolerance*gradientScale;
    stalledStatus = any(strcmp(primaryStats.status,{ ...
        'trust-region stalled','nonfinite LM step', ...
        'invalid pair during refinement'}));
    exhaustedWithoutConvergence = ...
        primaryStats.iterations>=maxIter && notConverged;
    noUsefulStep = primaryStats.accepted_steps==0 && notConverged;
    retryNeeded = stalledStatus || exhaustedWithoutConvergence || noUsefulStep;

    if ~retryNeeded
        return;
    end

    retryCount = 1;
    stats.retry_attempted = true;
    if stalledStatus || primaryStats.accepted_steps>0
        retryDamping = min(max(initialDamping*10,1e-8),1e6);
    else
        retryDamping = max(initialDamping*0.1,1e-8);
    end
    retryIterations = max(12,round(0.75*maxIter));
    % Continue from the best point reached by the first LM run rather than
    % restarting from its original coarse seed.
    retryStartNu = nu;
    [nuRetry,BRetry,normRetry,objRetry,condRetry,retryStats] = ...
        refine_pair_variable_projection_lm( ...
        pilotTx,y,retryStartNu,delay,cfg,minPairDenominator,maxPairCondition, ...
        retryIterations,retryDamping,maxStep,gradientTolerance,stepTolerance);

    stats.total_iterations = primaryStats.iterations+retryStats.iterations;
    stats.total_accepted_steps = primaryStats.accepted_steps+ ...
        retryStats.accepted_steps;
    stats.retry_status = retryStats.status;
    stats.retry_damping = retryDamping;
    stats.retry_gradient = retryStats.final_gradient;

    if isfinite(objRetry) && (~isfinite(objective) || objRetry<objective)
        nu = nuRetry;
        B = BRetry;
        rawNorms = normRetry;
        objective = objRetry;
        pairCondition = condRetry;
        stats = retryStats;
        stats.total_iterations = primaryStats.iterations+retryStats.iterations;
        stats.total_accepted_steps = primaryStats.accepted_steps+ ...
            retryStats.accepted_steps;
        stats.retry_attempted = true;
        stats.retry_status = retryStats.status;
        stats.retry_damping = retryDamping;
        stats.retry_gradient = retryStats.final_gradient;
        stats.primary_status = primaryStats.status;
    end
end

function objective = variable_projection_objective( ...
    unconstrainedNu,delay,pilotTx,y,cfg,minPairDenominator,maxPairCondition)
% Compatibility wrapper retaining the original hard-validity objective.
    objective = variable_projection_fallback_objective( ...
        unconstrainedNu,delay,pilotTx,y,cfg,minPairDenominator, ...
        maxPairCondition,0);
end

function objective = variable_projection_fallback_objective( ...
    unconstrainedNu,delay,pilotTx,y,cfg,minPairDenominator, ...
    maxPairCondition,conditionPenaltyWeight)
% Smooth condition-aware objective used only by derivative-free fallback.
% Below the condition limit this is exactly the unregularized ML loss. Above
% the limit it adds a smooth log-condition penalty, rather than returning a
% flat constant. The final candidate is always re-evaluated using the strict
% ML validity test in evaluate_pair_ml before it can replace the best result.
    nu = (cfg.nu_max*tanh(unconstrainedNu(:))).';
    [B,~] = make_pair_atoms(pilotTx,nu,delay,cfg);
    cross = B(:,1)'*B(:,2);
    denominator = max(0,1-abs(cross)^2);
    pairCondition = (1+abs(cross))/max(1-abs(cross),eps);

    if ~isfinite(denominator) || denominator <= minPairDenominator || ...
            ~isfinite(pairCondition)
        objective = 1e6 + real(y'*y);
        return;
    end

    coeff = B\y;
    residual = y-B*coeff;
    objective = real(residual'*residual);
    if ~isfinite(objective)
        objective = 1e6 + real(y'*y);
        return;
    end

    if pairCondition > maxPairCondition && conditionPenaltyWeight > 0
        logExcess = log(pairCondition/maxPairCondition);
        objective = objective + conditionPenaltyWeight*real(y'*y)*logExcess^2;
    end
end

function [objective,coeff,residual,pairCondition] = evaluate_pair_ml( ...
    B,y,minPairDenominator,maxPairCondition)
    cross = B(:,1)'*B(:,2);
    denominator = max(0,1-abs(cross)^2);
    pairCondition = (1+abs(cross))/max(1-abs(cross),eps);
    if denominator <= minPairDenominator || ~isfinite(pairCondition) || ...
            pairCondition > maxPairCondition
        objective = Inf;
        coeff = zeros(2,1);
        residual = y;
        return;
    end
    % QR-based least squares is more numerically robust than explicitly
    % inverting the 2x2 Gram matrix when the two paths are correlated.
    coeff = B\y;
    residual = y-B*coeff;
    objective = real(residual'*residual);
end

function [nu,B,rawNorms,objective,pairCondition,stats] = ...
    refine_pair_variable_projection_lm(pilotTx,y,nu0,delay,cfg, ...
    minPairDenominator,maxPairCondition,maxIter,initialDamping, ...
    maxStep,gradientTolerance,stepTolerance)
% REFINE_PAIR_VARIABLE_PROJECTION_LM
% Damped Gauss-Newton refinement of the two continuous Doppler variables.
% The complex path gains are analytically eliminated at every iteration by
% least squares (variable projection). Atom derivatives are analytic.

    nu = min(max(nu0(:).',-cfg.nu_max),cfg.nu_max);
    useCenterSeparation = delay(1)==delay(2) && ...
        abs(diff(nu)) <= get_cfg(cfg, ...
        'estimator_close_pair_search_max_separation',0.12);
    if useCenterSeparation
        nu = sort(nu);
        center = mean(nu);
        separation = max(abs(diff(nu)),get_cfg(cfg, ...
            'estimator_close_pair_search_min_separation',1e-4));
        minSeparation = get_cfg(cfg, ...
            'estimator_close_pair_search_min_separation',1e-4);
        maxSeparation = get_cfg(cfg, ...
            'estimator_close_pair_search_max_separation',0.12);
    end
    damping = max(initialDamping,1e-12);
    [B,rawNorms] = make_pair_atoms(pilotTx,nu,delay,cfg);
    [objective,coeff,residual,pairCondition] = evaluate_pair_ml( ...
        B,y,minPairDenominator,maxPairCondition);

    stats.iterations = 0;
    stats.accepted_steps = 0;
    stats.final_gradient = Inf;
    stats.status = 'initialization';

    if ~isfinite(objective)
        stats.status = 'invalid initial pair';
        return;
    end

    for iteration = 1:maxIter
        [B,rawNorms,dAtom1,dAtom2] = make_pair_atoms_with_derivatives( ...
            pilotTx,nu,delay,cfg);
        [objective,coeff,residual,pairCondition] = evaluate_pair_ml( ...
            B,y,minPairDenominator,maxPairCondition);
        if ~isfinite(objective)
            stats.status = 'invalid pair during refinement';
            break;
        end

        % Project the coefficient-weighted atom derivatives onto the
        % orthogonal complement of span(B), as required by variable
        % projection when the complex gains are re-optimized at each step.
        [Q,~] = qr(B,0);
        D = [dAtom1*coeff(1), dAtom2*coeff(2)];
        Jnu = D-Q*(Q'*D);
        if useCenterSeparation
            % d(nu1,nu2)/d(center,separation) = [1 -1/2; 1 1/2].
            transform = [1 -0.5;1 0.5];
            J = Jnu*transform;
        else
            J = Jnu;
        end
        gradient = -2*real(J'*residual);
        hessianGN = 2*real(J'*J);
        stats.final_gradient = norm(gradient,inf);
        stats.iterations = iteration;

        if stats.final_gradient <= gradientTolerance*max(1,norm(y)^2)
            stats.status = 'gradient converged';
            break;
        end

        diagonalScale = max(diag(hessianGN),1e-12);
        dampedHessian = hessianGN+damping*diag(diagonalScale);
        if rcond(dampedHessian) < 1e-14
            dampedHessian = dampedHessian + ...
                max(damping,1e-6)*eye(2);
        end
        step = -(dampedHessian\gradient);
        if any(~isfinite(step))
            stats.status = 'nonfinite LM step';
            break;
        end

        % Bound the local trust-region step in normalized Doppler bins.
        step = max(-maxStep,min(maxStep,step));
        accepted = false;
        alpha = 1;
        oldNu = nu;
        oldObjective = objective;
        if useCenterSeparation
            oldCenter = center;
            oldSeparation = separation;
        end

        for lineSearch = 1:16
            if useCenterSeparation
                trialCenter = oldCenter+alpha*step(1);
                trialSeparation = oldSeparation+alpha*step(2);
                trialSeparation = min(maxSeparation, ...
                    max(minSeparation,trialSeparation));
                trialCenterLimit = max(cfg.nu_max-trialSeparation/2,0);
                trialCenter = min(trialCenterLimit,max(-trialCenterLimit,trialCenter));
                trialNu = [trialCenter-trialSeparation/2, ...
                           trialCenter+trialSeparation/2];
            else
                trialNu = min(max(oldNu+alpha*step(:).',-cfg.nu_max),cfg.nu_max);
            end
            if norm(trialNu-oldNu,inf) <= stepTolerance
                break;
            end
            [trialB,trialNorms] = make_pair_atoms( ...
                pilotTx,trialNu,delay,cfg);
            [trialObjective,trialCoeff,trialResidual,trialCondition] = ...
                evaluate_pair_ml(trialB,y,minPairDenominator,maxPairCondition);

            if isfinite(trialObjective) && trialObjective < oldObjective
                nu = trialNu;
                if useCenterSeparation
                    center = trialCenter;
                    separation = trialSeparation;
                end
                B = trialB;
                rawNorms = trialNorms;
                objective = trialObjective;
                pairCondition = trialCondition;
                coeff = trialCoeff;
                residual = trialResidual;
                accepted = true;
                stats.accepted_steps = stats.accepted_steps+1;
                damping = max(damping/3,1e-12);
                break;
            end
            alpha = alpha/2;
        end

        if ~accepted
            damping = min(damping*10,1e12);
            if damping >= 1e12
                stats.status = 'trust-region stalled';
                break;
            end
        elseif norm(alpha*step,inf) <= stepTolerance || ...
                abs(oldObjective-objective) <= ...
                1e-13*max(1,oldObjective)
            stats.status = 'step/objective converged';
            break;
        end
    end

    % Recompute final atoms and objective to keep all returned quantities
    % consistent, including the unnormalized atom norms for physical gains.
    [B,rawNorms] = make_pair_atoms(pilotTx,nu,delay,cfg);
    [objective,~,~,pairCondition] = evaluate_pair_ml( ...
        B,y,minPairDenominator,maxPairCondition);
    if strcmp(stats.status,'initialization')
        stats.status = 'maximum iterations reached';
    end
end


function result = estimate_local_parameter_uncertainty( ...
    pilotTx,nu,delay,cfg,coefficients,rawNorms,noiseVariance)
% Approximate local covariance from the real-valued Fisher information.
% Parameters are [Re(c1), Im(c1), Re(c2), Im(c2), nu1, nu2], where c are
% coefficients on unit-norm atoms. Gain standard deviations are mapped back
% by the raw atom norms. This local covariance does not include uncertainty
% across distinct likelihood basins; the ambiguity flag reports that separately.
    result.doppler_std_bins = [Inf Inf];
    result.gain_std_approx = [Inf Inf];
    result.relative_gain_std_approx = [Inf Inf];
    result.information_condition = Inf;
    result.information_rank = 0;
    result.covariance_valid = false;
    result.gain_uncertainty_flag = true;
    if numel(coefficients)~=2 || numel(rawNorms)~=2 || ...
            any(~isfinite(coefficients(:))) || any(~isfinite(rawNorms(:))) || ...
            any(rawNorms(:)<=0)
        return;
    end
    [B,~,dAtom1,dAtom2,normDerivatives] = ...
        make_pair_atoms_with_derivatives(pilotTx,nu,delay,cfg);
    D = [dAtom1*coefficients(1),dAtom2*coefficients(2)];
    J = [B(:,1),1i*B(:,1),B(:,2),1i*B(:,2),D];
    information = real(J'*J);
    information = 0.5*(information+information.');
    singularValues = svd(information);
    if isempty(singularValues) || ~isfinite(singularValues(1)) || ...
            singularValues(1)<=0
        return;
    end
    tolerance = max(size(information))*eps(singularValues(1));
    result.information_rank = nnz(singularValues>tolerance);
    result.information_condition = singularValues(1)/ ...
        max(singularValues(end),realmin);
    if noiseVariance<=0 || result.information_rank<6
        return;
    end
    covariance = (noiseVariance/2)*pinv(information,tolerance);
    covariance = real(0.5*(covariance+covariance.'));
    diagonal = max(real(diag(covariance)),0);
    result.doppler_std_bins = sqrt(diagonal(5:6)).';

    % Delta-method conversion from normalized-atom coefficients c_p to
    % physical gains h_p=c_p/||a_raw,p(nu_p)||. This includes the covariance
    % between each coefficient and its Doppler estimate, as well as the
    % Doppler dependence of the raw atom norm. Omitting these terms can
    % understate gain sensitivity for close or weak paths.
    gainJacobian = zeros(4,6);
    for pathIndex = 1:2
        reIndex = 2*pathIndex-1;
        imIndex = 2*pathIndex;
        nuIndex = 4+pathIndex;
        atomNorm = rawNorms(pathIndex);
        atomNormDerivative = normDerivatives(pathIndex);
        coefficient = coefficients(pathIndex);
        gainJacobian(2*pathIndex-1,reIndex) = 1/atomNorm;
        gainJacobian(2*pathIndex-1,nuIndex) = ...
            -real(coefficient)*atomNormDerivative/(atomNorm^2);
        gainJacobian(2*pathIndex,imIndex) = 1/atomNorm;
        gainJacobian(2*pathIndex,nuIndex) = ...
            -imag(coefficient)*atomNormDerivative/(atomNorm^2);
    end
    physicalGainCovariance = gainJacobian*covariance*gainJacobian.';
    physicalGainCovariance = real(0.5*(physicalGainCovariance+ ...
        physicalGainCovariance.'));
    gainDiagonal = max(real(diag(physicalGainCovariance)),0);
    result.gain_std_approx = [ ...
        sqrt(gainDiagonal(1)+gainDiagonal(2)), ...
        sqrt(gainDiagonal(3)+gainDiagonal(4))];
    physicalGains = coefficients(:)./rawNorms(:);
    result.relative_gain_std_approx = result.gain_std_approx ./ ...
        max(abs(physicalGains(:).'),1e-12);
    result.covariance_valid = all(isfinite(diagonal)) && ...
        all(isfinite(result.doppler_std_bins)) && ...
        all(isfinite(result.relative_gain_std_approx));
    informationWarning = get_cfg(cfg,'estimator_local_information_warning',1e8);
    gainWarning = get_cfg(cfg,'estimator_gain_uncertainty_warning',0.50);
    result.gain_uncertainty_flag = ~result.covariance_valid || ...
        result.information_condition>informationWarning || ...
        any(result.relative_gain_std_approx>gainWarning);
end

function [B,rawNorms,dAtom1,dAtom2,normDerivatives] = ...
    make_pair_atoms_with_derivatives(pilotTx,nu,delay,cfg)
% Build unit-norm atoms, derivatives and raw-norm derivatives.
    [atom1,norm1,dAtom1,normDerivative1] = make_atom_with_derivative( ...
        pilotTx,nu(1),delay(1),cfg);
    [atom2,norm2,dAtom2,normDerivative2] = make_atom_with_derivative( ...
        pilotTx,nu(2),delay(2),cfg);
    B = [atom1,atom2];
    rawNorms = [norm1;norm2];
    normDerivatives = [normDerivative1;normDerivative2];
end

function [atom,atomNorm,atomDerivative,atomNormDerivative] = ...
    make_atom_with_derivative(pilotTx,doppler,delay,cfg)
% The derivative follows directly from d exp(j*w*n)/d nu.
    L = numel(pilotTx);
    s = pilotTx(:);
    delayedSignal = complex(zeros(L,1));
    if delay < 0 || delay ~= floor(delay) || delay >= L
        error('Refinement delay must be an integer in [0,N*M-1].');
    end
    validIndices = (delay+1):L;
    delayedSignal(validIndices) = s(1:L-delay);

    n = (0:L-1).';
    dopplerResolution = 1/(cfg.N*cfg.T);
    phaseSlope = 2*pi*dopplerResolution*cfg.Ts;
    phase = exp(1j*phaseSlope*doppler*n);
    dopplerSignal = delayedSignal.*phase;
    derivativeSignal = 1j*phaseSlope*n.*dopplerSignal;

    rawAtom = demodulate_training_sequence(dopplerSignal,cfg);
    derivativeRawAtom = demodulate_training_sequence(derivativeSignal,cfg);
    rawAtom = rawAtom(:);
    derivativeRawAtom = derivativeRawAtom(:);
    atomNorm = norm(rawAtom);
    if atomNorm <= eps
        error('Refinement generated a zero-norm atom.');
    end
    atom = rawAtom/atomNorm;
    normDerivative = real(atom'*derivativeRawAtom);
    atomNormDerivative = normDerivative;
    atomDerivative = (derivativeRawAtom-atom*normDerivative)/atomNorm;
end

function [B,rawNorms] = make_pair_atoms(pilotTx,nu,delay,cfg)
    [B1,norm1] = make_atom(pilotTx,nu(1),delay(1),cfg);
    [B2,norm2] = make_atom(pilotTx,nu(2),delay(2),cfg);
    B = [B1,B2];
    rawNorms = [norm1;norm2];
end

function [doppler,delay] = atom_index_to_parameters(atomIndex,dopplerGrid,delayGrid)
    numDoppler = numel(dopplerGrid);
    dopplerIndex = mod(atomIndex-1,numDoppler)+1;
    delayIndex = floor((atomIndex-1)/numDoppler)+1;
    doppler = dopplerGrid(dopplerIndex);
    delay = delayGrid(delayIndex);
end

function [atom,atomNorm] = make_atom(pilotTx,doppler,delay,cfg)
    channel.delay = delay;
    channel.doppler = doppler;
    channel.gain = 1;
    rx = apply_fractional_doppler_channel(pilotTx,channel,cfg);
    Y = demodulate_training_sequence(rx,cfg);
    rawAtom = Y(:);
    atomNorm = norm(rawAtom);
    if atomNorm <= eps
        error('Refinement generated a zero-norm atom.');
    end
    atom = rawAtom/atomNorm;
end

function mask = isfield_valid_mask(pairCache,numPairs)
% Return the cached validity mask when it is present and dimensionally valid.
    if isfield(pairCache,'valid') && numel(pairCache.valid)==numPairs
        mask = logical(pairCache.valid(:));
    else
        mask = false(numPairs,1);
    end
end

function value = get_cfg(cfg,fieldName,defaultValue)
    if isfield(cfg,fieldName) && ~isempty(cfg.(fieldName))
        value = cfg.(fieldName);
    else
        value = defaultValue;
    end
end
