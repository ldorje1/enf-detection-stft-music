% ENF Estimation from Noisy Audio Using STFT and MUSIC Algorithms
%
% Author: xx
% Date: [feb 19 2025]
%
% Description:
    % This script estimates the Electric Network Frequency (ENF) from noisy 
    % audio signals using two methods: Short-Time Fourier Transform (STFT) 
    % and Multiple Signal Classification (MUSIC). 
    % The input signals are resampled to a target sampling frequency of 300 Hz. 
    % The script processes noisy and reference audio files, computes ENF estimates, 
    % and saves the results in .mat files. 
    % It then plots the differences in ENF estimates over time for both methods, 
    % along with the correlation coefficients between them. 
    % This is useful for comparing the accuracy of ENF estimation in noisy 
    % conditions and validating the performance of the algorithms.

    % Input: 
    % - Noisy and reference audio files (e.g., '001.wav' and '001_ref.wav')
    % - Target sampling frequency for resampling: 300 Hz
    % - Harmonic number for ENF estimation: 2
    % - Nominal frequency: 50 Hz (can be adjusted)
    % - Frequency band size: 0.2 Hz (adjustable)
    % - Butterworth filter order: 10 (adjustable)
    % - Window size for STFT analysis: 16 samples (adjustable)

    % Output:
    % - ENF estimates for noisy and reference signals (using STFT and MUSIC)
    % - Plots showing differences in ENF estimates and correlation coefficients
    
    % Dependencies:
    % - `enf_STFTQuad`: Function for ENF estimation using STFT
    % - `enf_MUSIC`: Function for ENF estimation using MUSIC

%%
clc; clear; close all;

%% Load Audio File and Resample
% Download these audio recording from 
% https://github.com/ghua-ac/ENF-WHU-Dataset/tree/master/ENF-WHU-Dataset.
% Following noisy_files and ref_files are actually audio recording the from
% the dataset.

%%%%%%%%%%%%%%% Just change this part to estimate ENF %%%%%%%%%%%%%%%%%%%%%%
% Note: You can add or remove audio files from noisy_files and ref_files.
% If remove something in noisy_files, then make sure remove its
% corresponding file ref_files too
%   For example: If remove '002.wav' then also remove '002_ref.wav'. 

% --- Auto-load file lists from folders ---
noisy_dir = fullfile(pwd, 'audiodata', 'H1');
ref_dir   = fullfile(pwd, 'audiodata', 'H1_ref');

noisy_listing = dir(fullfile(noisy_dir, '*.wav'));
ref_listing   = dir(fullfile(ref_dir,   '*.wav'));

% Sort by the first number in the filename (e.g., 001, 002, ...)
getNum = @(s) str2double(regexp(s, '\d+', 'match', 'once'));

[~, in] = sort(arrayfun(@(d) getNum(d.name), noisy_listing));
[~, ir] = sort(arrayfun(@(d) getNum(d.name), ref_listing));

noisy_listing = noisy_listing(in);
ref_listing   = ref_listing(ir);

% Build full paths as cell arrays (what audioread likes)
noisy_files = fullfile(noisy_dir, {noisy_listing.name});
ref_files   = fullfile(ref_dir,   {ref_listing.name});


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% We will resample both the noisy and reference audio at 400 Hz to ensure 
% that both datasets have the same sampling rate, 
% since the original noisy audio is sampled at 400 Hz, while the original 
% reference audio files are sampled at 8000 Hz.
[data_n1, fs_n] = audioread(noisy_files{1});
[data_ref1, fs_ref] = audioread(ref_files{1});

fs_target = 400; % New sampling rate

[p_n, q_n] = rat(fs_target / fs_n);
[p_r, q_r] = rat(fs_target / fs_ref);

data_noisy = cell(1, length(noisy_files));
data_ref = cell(1, length(noisy_files));

% Containers for audio data and estimated enf using STFT and MUSIC 
data_noisy = cell(1, length(noisy_files));
data_ref = cell(1, length(noisy_files));
enf_noisy_STFT = [];
enf_ref_STFT = [];
enf_noisy_MUSIC = [];
enf_ref_MUSIC = [];

% Parameters for ENF estimation
fs = fs_target;
nominal_freq = 50;
freq_band_size = 0.2;
harmonic_n = 2;
order = 10;
window_size = 16; 
N_fft = 100e3;

% In the following for loop, first the audio data is resampled at 400 and then
% ENF are estimate using enf_STFTQuad() and enf_MUSIC(). 
%   for example: (1) we will first select a noisy data, 001.wav, then
%   estimated ENF using our two algorithm and then store them in
%   enf_noisy_STFT{i} and enf_noisy_MUSIC{i}.
%   
%   (2) We will select the corresponding reference data, 001_ref.wav, then
%   estimated ENF using our two algorithm and then store them in
%   enf_ref_STFT{i} and enf_ref_MUSIC{i}. enf_ref_STFT{i} 
for i = 1:length(noisy_files)
    % (1) STFT and MUSIC ENF Noisy audio (H1)
    [data_noisy{i}] = audioread(noisy_files{i});
    data_noisy{i} = resample(data_noisy{i}, p_n, q_n);

    enf_noisy_STFT{i} = enf_STFTQuad(data_noisy{i}, fs, harmonic_n, ...
        nominal_freq, freq_band_size, order, window_size);
    enf_noisy_MUSIC{i} = enf_MUSIC(data_noisy{i}, fs, harmonic_n, ...
        nominal_freq, freq_band_size, order, window_size, N_fft);

    % (2) STFT and MUSIC ENF Noisy audio (H0) Reference audio
    [data_ref{i}] = audioread(ref_files{i});
    data_ref{i} = resample(data_ref{i}, p_r, q_r);

    enf_ref_STFT{i} = enf_STFTQuad(data_ref{i}, fs, harmonic_n, ...
        nominal_freq, freq_band_size, order, window_size);
    enf_ref_MUSIC{i} = enf_MUSIC(data_ref{i}, fs, harmonic_n, ...
        nominal_freq, freq_band_size, order, window_size, N_fft);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here all the esimated ENF are concatenated to form a single row vector.
%   For example: enf_noisy_STFT contains concenated STFT estimated noisy ENF 
%   from all audio data in noisy_files and enf_ref_STFT contains concenated 
%   STFT estimated reference ENF from all audio data in ref_files. 

% You can skip first seven esimation ENF from each vector to get a better 
% ENF reading. First 7 ENF values are way out of bound, not sure why?
enf_noisy_STFT = [enf_noisy_STFT{:}];  % Concatenate all noisy ENFs into a single row vector
%enf_noisy_STFT = enf_noisy_STFT(7:end);
enf_ref_STFT = [enf_ref_STFT{:}];      
%enf_ref_STFT = enf_ref_STFT(7:end);
enf_noisy_MUSIC = [enf_noisy_MUSIC{:}];  
%enf_noisy_MUSIC = enf_noisy_MUSIC(7:end);
enf_ref_MUSIC = [enf_ref_MUSIC{:}];      
%enf_ref_MUSIC = enf_ref_MUSIC(7:end);


% Finally, we have all ENF estimated from noisy_files and ref_files.
%   For example: First row of enf_all_STFT contains enf_noisy_STFT and
%   second row contains enf_ref_STFT estimated via STFT using noisy_files
%   and ref_files. 
enf_all_STFT = [enf_noisy_STFT; enf_ref_STFT];
enf_all_MUSIC = [enf_noisy_MUSIC; enf_ref_MUSIC];


% Save the estimated ENFs
save('enf_all_STFT.mat', 'enf_all_STFT');  % Save the matrix to a .mat file
save('enf_all_MUSIC.mat', 'enf_all_MUSIC');  % Save the matrix to a .mat file

disp('Estimated ENF files saved!');  % Print a confirmation message



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Twin ENF Estimation Algos : STFT and MUSIC
%--------------------------------------------------------------
function enf = enf_STFTQuad(data, fs, harmonic_n, nominal_freq, freq_band_size, order, window_size)
    % enf_STFTQuad: Estimate the Electric Network Frequency (ENF) from 
    % a signal using STFT and Quadratic Interpolation
    %
    % Inputs:
    %   data - Input signal
    %   fs - Sampling frequency
    %   harmonic_n - Harmonic number for the target frequency component
    %   nominal_freq - Nominal frequency of the ENF (e.g., 50Hz or 60Hz)
    %   freq_band_size - Bandwidth around the nominal frequency to analyze
    %   order - Order of the Butterworth filter
    %   window_size - Window size in seconds for the STFT analysis
    %
    % Outputs:
    %   enf - Estimated ENF values over time
    
    % Cutoff frequencies
    nyq = fs / 2;
    locut = harmonic_n * (nominal_freq - freq_band_size);
    hicut = harmonic_n * (nominal_freq + freq_band_size);
    low = locut / nyq;
    high = hicut / nyq;

    % Design Butterworth Bandpass Filter
    [A,B,C,D] = butter(order, [low high], 'bandpass');
    sos = ss2sos(A,B,C,D); 
    filtered_data = filtfilt(sos, 1, data); % Zero-phase filtering using SOS

    %% STFT Parameters and Computation
    window_size_seconds = window_size;
    %N_fft = 2^(ceil(log2(1000 * fs)));
    nperseg = fs * window_size_seconds;
    noverlap = fs * (window_size_seconds - 1);
    [s, f, t] = spectrogram(filtered_data, nperseg, noverlap, [], fs);

    % Quadratic Interpolation for Peak Frequency Estimation
    bin_size = f(2) - f(1);
    %quadratic_interpolation = @(data, max_idx, bin_size) ...
    %    0.5 * (data(max_idx-1) - data(max_idx+1)) / (data(max_idx-1) - 2*data(max_idx) + data(max_idx+1)) * bin_size + max_idx * bin_size;
    quadratic_interpolation = @(f, data, max_idx) ...
        f(max_idx) + 0.5 * (data(max_idx - 1) - data(max_idx + 1)) / ...
        (data(max_idx - 1) - 2 * data(max_idx) + data(max_idx + 1)) * (bin_size);

    % Find the peaks in each STFT spectrum and apply quadratic interpolation
    max_freqs = zeros(1, length(t));
    for i = 1:length(t)
        spectrum = abs(s(:, i)); % Get magnitude of spectrum for each time slice
        [~, max_freq_idx] = max(spectrum); % Find index of maximum amplitude

        % Only apply quadratic interpolation if necessary (e.g., peak is not on a grid point)
        if max_freq_idx > 1 && max_freq_idx < length(spectrum)           
            max_freq = quadratic_interpolation(f, spectrum, max_freq_idx);
        else
            max_freq = f(max_freq_idx); % Use directly if peak is already on a grid point
        end
        max_freqs(i) = max_freq; 
    end
    enf = max_freqs / harmonic_n;
end

%--------------------------------------------------------------
function enf = enf_MUSIC(data, fs, harmonic_n, nominal_freq, freq_band_size, order, window_size, N_fft)
% ENF_MUSIC Estimates the Electric Network Frequency (ENF) from an audio signal
    % using the MUSIC algorithm.
    %
    % INPUTS:
    %   data            - Input signal (e.g., audio data).
    %   fs              - Sampling frequency of the input signal (Hz).
    %   harmonic_n      - Harmonic number of the ENF component of interest.
    %   nominal_freq    - Nominal frequency of the power grid (e.g., 50 Hz or 60 Hz).
    %   freq_band_size  - Frequency band size around the nominal frequency for filtering.
    %   order           - Order of the Butterworth bandpass filter.
    %   window_size     - Window size in seconds for sliding window analysis.
    %   N_fft           - Number of FFT points for the MUSIC algorithm.
    %
    % OUTPUT:
    %   enf 

    % Cutoff frequencies
    nyq = 0.5 * fs;
    locut = harmonic_n * (nominal_freq - freq_band_size);
    hicut = harmonic_n * (nominal_freq + freq_band_size);
    low = locut / nyq;
    high = hicut / nyq;

    % Design Butterworth Bandpass Filter (SOS)
    [A, B, C, D] = butter(order, [low high], 'bandpass');
    sos = ss2sos(A, B, C, D);  % Second-order sections representation
    filtered_data = filtfilt(sos, 1, data); % Zero-phase filtering for no phase distortion

    %% Sliding Window and MUSIC Spectrum Calculation
    window_size_seconds = window_size;                % Window size in seconds
    nperseg = fs * window_size_seconds;      % Samples per segment
    noverlap = fs * (window_size_seconds - 1); % Overlap between segments
    step_size = nperseg - noverlap;          % Step size for sliding window

    %% Quadratic Interpolation Function
    quadratic_interpolation = @(data, max_idx, bin_size) ...
        0.5 * (data(max_idx-1) - data(max_idx+1)) / (data(max_idx-1) - 2*data(max_idx) + data(max_idx+1)) * bin_size + max_idx * bin_size;

    %% Initialize Variables for MUSIC Spectrum Calculation
    num_windows = floor((length(filtered_data) - nperseg) / step_size) + 1;
    music_spectrum_matrix = [];
    max_freqs = zeros(1, num_windows);
    max_psd_values = zeros(num_windows, 1);

    p_order = 2; % Order of the MUSIC algorithm

    % Process each window
    for i = 1:num_windows
        segment = filtered_data((i - 1) * step_size + (1:nperseg));
        [psd, freq] = pmusic(segment, p_order, N_fft, fs);
        music_spectrum_matrix(:, i) = abs(psd);
        [max_psd, max_freq_idx] = max(abs(psd));
        max_freq = quadratic_interpolation(abs(psd), max_freq_idx, freq(2) - freq(1));
  
        max_freqs(i) = max_freq;
        max_psd_values(i) = max_psd;
    end

    enf = max_freqs / harmonic_n;
end



%------ IGNORE PLEASE!$
%{
%% Plotting the absolute difference over time can help visualize the discrepancies.
% Calculate differences (not absolute differences)
diff_STFT = (enf_ref_STFT - enf_noisy_STFT);
diff_MUSIC = (enf_ref_MUSIC - enf_noisy_MUSIC);

% Calculate the mean difference for STFT and MUSIC
mean_diff_STFT = mean(diff_STFT);
mean_diff_MUSIC = mean(diff_MUSIC);

% Time vector
t = (0:length(enf_ref_STFT) - 1);

% Plot the results
figure;

% Plot for STFT
subplot(2,1,1);
hold on;
plot(t, diff_STFT, 'r', 'LineWidth', 1.2);  % Difference (not absolute)
plot(t, repmat(mean_diff_STFT, size(t)), 'b--', 'LineWidth', 1.5);  % Mean difference line (dashed)
xlabel('Time (samples)');
ylabel('Difference (Hz)');
title('Reference and Noisy ENF Difference Over Time (STFT)');
legend('Difference', ['Mean Difference (', num2str(mean_diff_STFT, '%.6f'), ' Hz)'], 'Location', 'Best');
grid on;
hold off;

% Plot for MUSIC
subplot(2,1,2);
hold on;
plot(t, diff_MUSIC, 'b', 'LineWidth', 1.2);  % Difference (not absolute)
plot(t, repmat(mean_diff_MUSIC, size(t)), 'r--', 'LineWidth', 1.5);  % Mean difference line (dashed)
xlabel('Time (samples)');
ylabel('Difference (Hz)');
title('Reference and Noisy ENF Difference Over Time (MUSIC)');
legend('Difference', ['Mean Difference (', num2str(mean_diff_MUSIC, '%.6f'), ' Hz)'], 'Location', 'Best');
grid on; 
hold off;


%% A high correlation (close to 1) indicates that the two ENF estimates are strongly related.
% Calculate correlation coefficients
corr_STFT = corrcoef(enf_ref_STFT, enf_noisy_STFT); 
corr_MUSIC = corrcoef(enf_ref_MUSIC, enf_noisy_MUSIC);

% Compute correlation between STFT and MUSIC for reference signals
corr_ref = corrcoef(enf_ref_STFT(6:end), enf_ref_MUSIC(6:end));
corr_noisy = corrcoef(enf_noisy_STFT, enf_noisy_MUSIC);

corr_MUSIC_cros = corrcoef(enf_ref_STFT, enf_noisy_MUSIC);
corr_STFT_cros = corrcoef(enf_ref_MUSIC, enf_noisy_STFT);

fprintf('Correlation coefficient (STFT): %.6f\n', corr_STFT(1,2));
fprintf('Correlation coefficient (MUSIC): %.6f\n', corr_MUSIC(1,2));
fprintf('Correlation coefficient between reference STFT and MUSIC ENF: %.4f\n', corr_ref(1, 2));
fprintf('Correlation coefficient between noisy STFT and MUSIC ENF: %.4f\n', corr_noisy(1, 2));

fprintf('"cross" Correlation coefficient between reference STFT and noisy MUSIC ENF: %.4f\n', corr_MUSIC_cros(1, 2));
fprintf('"cross" Correlation coefficient between reference MUSIC and nosiy STFT ENF: %.4f\n', corr_STFT_cros(1, 2));
%% Plot reference and noisy estimated via STFT and MUSIC

% Time axes for STFT and MUSIC
t_STFT = (0:length(enf_noisy_STFT) - 1);
t_MUSIC = (0:length(enf_noisy_MUSIC) - 1);

figure;

% First plot for STFT
subplot(2, 1, 1);
hold on;
plot(t_STFT, enf_noisy_STFT, 'r', 'LineWidth', 1.2);
plot(t_STFT, enf_ref_STFT, 'b', 'LineWidth', 1.2);
xlabel('Time (samples)');
ylabel('Frequency (Hz)');
title(sprintf('Audio ENF Comparison: Noise vs Reference (STFT) | Corr: %.2f', corr_STFT(1, 2)));
legend('ENF Noisy', 'ENF Reference');
grid on;
hold off;

% Second plot for MUSIC
subplot(2, 1, 2);
hold on;
plot(t_MUSIC, enf_noisy_MUSIC, 'r', 'LineWidth', 1.2);
plot(t_MUSIC, enf_ref_MUSIC, 'b', 'LineWidth', 1.2);
xlabel('Time (samples)');
ylabel('Frequency (Hz)');
title(sprintf('Audio ENF Comparison: Noise vs Reference (MUSIC) | Corr: %.6f', corr_MUSIC(1, 2)));
legend('ENF Noisy', 'ENF Reference');
grid on;
hold off;
%}