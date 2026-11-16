
function params = parameters()
% ==========================================================
% OTFS FRACTIONAL DOPPLER CHANNEL ESTIMATION PROJECT
% Person 1 - OTFS Transmitter and Receiver
% Common System Parameters
% ==========================================================

%% 1. OTFS Grid Parameters

params.N = 12;
params.M = 12;

params.num_symbols = params.N * params.M;

%% 2. Carrier and Modulation Parameters

params.fc = 3e9;              % Carrier frequency: 3 GHz
params.delta_f = 7.5e3;       % Subcarrier spacing: 7.5 kHz

params.modulation = '4-QAM';
params.bits_per_symbol = 2;

%% 3. Channel Parameters

params.max_velocity = 150;    % Maximum velocity in km/h
params.max_delay = 2000e-9;   % Maximum delay in seconds

params.max_delay_index = 2;
params.max_doppler_index = 2;
params.max_fractional_doppler = 0.2;

params.num_paths = 2;

params.doppler_neighborhood = [-1 0 1];

%% 4. Sampling Parameters

params.Fs = 90e3;             % Sampling frequency: 90 kHz
params.Ts = 1 / params.Fs;    % Sampling interval

params.T_symbol = 1 / params.delta_f;

%% 5. Pilot and Data Parameters

params.pilot_amplitude = 1;
params.data_amplitude = 1;

% Pilot sequence and placement will be finalized
% jointly with Person 2.

%% 6. Signal Dimensions

params.DD_size = [params.N params.M];
params.TF_size = [params.N params.M];

params.signal_length = params.N * params.M;

%% 7. Derived Parameters

params.max_velocity_ms = params.max_velocity / 3.6;

c = 3e8;                      % Speed of light (m/s)

params.max_doppler_hz = ...
    (params.max_velocity_ms * params.fc) / c;

params.doppler_bin_spacing = ...
    1 / (params.N * params.T_symbol);

%% 8. Display Parameters

disp('======================================');
disp('       OTFS SYSTEM PARAMETERS');
disp('======================================');

fprintf('N                    : %d\n', params.N);
fprintf('M                    : %d\n', params.M);
fprintf('Carrier Frequency    : %.2f GHz\n', params.fc/1e9);
fprintf('Subcarrier Spacing   : %.2f kHz\n', params.delta_f/1e3);
fprintf('Modulation           : %s\n', params.modulation);
fprintf('Sampling Frequency   : %.2f kHz\n', params.Fs/1e3);
fprintf('Sampling Interval    : %.4f us\n', params.Ts*1e6);
fprintf('Symbol Duration      : %.4f us\n', params.T_symbol*1e6);
fprintf('Maximum Doppler      : %.2f Hz\n', params.max_doppler_hz);
fprintf('Doppler Bin Spacing  : %.2f Hz\n', params.doppler_bin_spacing);
fprintf('Number of Paths      : %d\n', params.num_paths);
fprintf('Signal Length        : %d\n', params.signal_length);

disp('======================================');

end
