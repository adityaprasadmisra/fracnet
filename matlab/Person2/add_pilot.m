function [X_dd, pilot_dd] = add_pilot(data_dd, params)
% Single-cell overlaid pilot, as in Guo and Zhu, WCNC 2024, Fig. 3(b).
N=params.N; M=params.M;
if ~isequal(size(data_dd),[N M]), error('data_dd must be N-by-M.'); end
if isfield(params,'pilot_position'), pos=params.pilot_position; else, pos=[1 1]; end
if numel(pos)~=2 || any(pos~=round(pos)) || pos(1)<1 || pos(1)>N || pos(2)<1 || pos(2)>M
    error('pilot_position must be valid MATLAB [row,column] indices.');
end
if ~isfield(params,'pilot_amplitude') || params.pilot_amplitude==0
    error('A nonzero params.pilot_amplitude is required.');
end
pilot_dd=complex(zeros(N,M));
pilot_dd(pos(1),pos(2))=params.pilot_amplitude;
X_dd=data_dd+pilot_dd;
end
