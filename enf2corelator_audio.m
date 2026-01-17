%% ENF Detection Evaluation and AUC Curve Generation
    % adapted from enf2corelato.m  (Prof. Li, feb 2024)

    % This MATLAB script evaluates the performance of ENF (Electrical Network Frequency)
    % estimation methods, specifically comparing two approaches: STFT (Short-Time Fourier Transform) 
    % and MUSIC (Multiple Signal Classification). It calculates correlation coefficients, detection 
    % accuracy, and false alarm rates for various segment lengths (Tc) of ENF data, and generates 
    % AUC (Area Under Curve) plots to visualize the detection performance.
    %
    % The script follows these steps:
    % 1. **Load ENF Data**: Loads pre-saved ENF estimates from `.mat` files (`enf_all_STFT.mat` and 
    %    `enf_all_MUSIC.mat`) which contain reference and noisy ENF signals.
    % 2. **Data Selection**: Based on the `data_load` variable, the user can specify which dataset 
    %    to use (STFT, MUSIC, or combinations of noisy and reference signals from different methods).
    % 3. **Correlation Calculation**: Computes correlation between noisy and reference ENF signals 
    %    for different segment durations (`Tc`). Both same-time segment correlations and cross-correlations 
    %    between random segments are calculated.
    % 4. **Detection Performance**: For each segment length, calculates detection accuracy and false 
    %    alarm rate (FAR) by sorting correlation values and applying a decision threshold.
    % 5. **AUC Curve Generation**: Generates AUC plots comparing detection accuracy and FAR for different 
    %    segment lengths, and visualizes the results alongside a reference AUC curve from a 2021 paper (Hua et al.).

%% 
clc; clear;

%% Load ENF estimates from your saved data
load('enf_all_STFT.mat');  
load('enf_all_MUSIC.mat'); 

% Select the reference and test ENF. 
% where we use f1 as refence, to detect if an arbitrary segment of f2 is recorded
% at the same time as a segment of f1 
data_load = 3;  % 1 for noise ENF from STFT and ref ENF from STFT,
                % 2 for noise ENF from MUSIC and ref ENF from MUSIC 
                % 3 for noisy ENF from MUSIC & ref ENF from STFT, 
                % 4 for noise ENF from STFT & ref ENF from MUSIC

% we use f1 as refence, to detect if an arbitrary segment of f2 is recorded
% at the same time as a segment of f1
% Based on the selection, assign the appropriate data
switch data_load
    case 1  % STFT
        f1 = enf_all_STFT(2, :);  % reference ENF
        f2 = enf_all_STFT(1, :);  % noisy ENF
    case 2  % MUSIC
        f1 = enf_all_MUSIC(2, :);  % Reference ENF
        f2 = enf_all_MUSIC(1, :);  % noisy ENF
    case 3  % Noisy from STFT, Reference from MUSIC
        f1 = enf_all_MUSIC(2, :);  % reference enf
        f2 = enf_all_STFT(1, :);   % noisy enf
    case 4  % Noisy from Music, Reference from MUSIC
        f1 = enf_all_STFT(2, :);   % reference enf 
        f2 = enf_all_MUSIC(1, :);  % noisy enf
    otherwise
        error('Invalid input. Please select 1 for STFT, 2 for MUSIC, 3 for noise from MUSIC & ref from STFT, or 4 for noise from STFT & ref from MUSIC.');
end


L = length(f1);   % Total number of ENF estimates

% Define segment time duration for correlation
dd = [];
for ii = 1 : 3   % Draw 3 AUC curves for different Tc values
    if ii == 1, Tc = 5; end  % 25 seconds segment
    if ii == 2, Tc = 150; end  % 150 seconds segment
    if ii == 3, Tc = 250; end  % 250 seconds segment
    %if ii == 4, Tc = 2500; end % 2500 seconds segment

    % Collect all the correlation data:

    % All of the correlation C(n,n) - correlation of two segments recorded at the same time
    c1 = zeros(1, L-Tc+1);
    parfor i = 1 : L-Tc+1
        c1(i) = f1(i:i+Tc-1)*f2(i:i+Tc-1)'/norm(f1(i:i+Tc-1))/norm(f2(i:i+Tc-1));
    end

    % For each segment of f1(n), calculate its correlation with a random segment of f2(m)
    ind = randperm(L-Tc); 
    if length(ind)>1000, ind = ind(1:1000); end % Pick 1000 random segments from f2

    % Parallel calculation for C(n,m)
    c2 = zeros(1, length(ind)*(L-Tc-1));  % L-Tc-1: total # of segments in f1
    parfor k = 1 : length(ind)*(L-Tc-1)-1
        f1i = floor(k/(L-Tc-1))+1;   % f1 index ind(i)
        f2j = mod(k, length(ind))+1; % f2 index j
        if abs(f2j - ind(f1i)) < Tc, continue; end  % Skip same-time segments
        c21 = f1(ind(f1i):ind(f1i)+Tc-1)*f2(f2j:f2j+Tc-1)' / ...
            norm(f1(ind(f1i):ind(f1i)+Tc-1)) / norm(f2(f2j:f2j+Tc-1));
        c2(k) = c21;  % Store C(n,m) to c2
    end
    c2(c2==0) = [];  % Remove uncaptured correlations

    % Calculate FAR and ACC based on a decision threshold
    FAR = 0.1;  % Desired false alarm rate
    c2s = sort(c2);  % Sort from low to high
    gamma2 = c2s(round((1-FAR)*length(c2s)));
    disp(['Iteration ii=' num2str(ii) ': Tc=' num2str(Tc) '. Find decision threshold gamma = ' num2str(gamma2)])

    % Calculate detection accuracy and false alarm rate
    acc = sum(c1 > gamma2) / length(c1);
    far = sum(c2 > gamma2) / length(c2);
    disp(['Detection accuracy is ' num2str(acc*100, 2) '% and false alarm rate is ' num2str(far*100, 2) '%'])

    % Collect FAR and ACC for drawing AUC curves, and save data
    acci = []; fari = [];
    for far = 0:0.01:1
        FAR = far;  % Desired false alarm rate
        ind = round((1-FAR) * length(c2s));
        if ind > 0
            gamma2 = c2s(round((1-FAR) * length(c2s)));
            acci = [acci sum(c1 > gamma2) / length(c1)];
            fari = [fari sum(c2 > gamma2) / length(c2)];
        else
            acci = [acci 1]; fari = [fari 1];
        end
    end

    % Append FAR and ACC to dd matrix
    dd = [dd; fari; acci];
end

%% Plots
% 2021 paper Hua'2021
pp2 = [0,   0.05, 0.1, 0.2, 0.3,  0.4, 1
       0.5, 0.65, 0.7, 0.95, 0.98, 1, 1];
   
% Draw AUC curves comparing results
figure
h1 = plot([0, 1], [0, 1], 'k--', 'LineWidth', 1); % Make sure to assign to a handle
hold on
h2 = plot(dd(1, :), dd(2, :), 'r-', 'LineWidth', 1); % Our Results (T=5s)
h3 = plot(dd(3, :), dd(4, :), 'b-', 'LineWidth', 1); % Our Results (T=150s)
h4 = plot(dd(5, :), dd(6, :), 'g-', 'LineWidth', 1); % Our Results (T=250s)
%h5 = plot(dd(7, :), dd(8, :), 'k-', 'LineWidth', 2); % Our Results (T=2500s)
h5 = plot(pp2(1, :), pp2(2, :), 'c--', 'LineWidth', 1); % Hua2021 (T=150s)

grid
xlabel('False Alarm Rate (P_F)'), ylabel('Detection Accuracy (P_D)')
%title('AUC Curves for ENF Detection on Hua2021 Audio Dataset (STFT)')

% Create a legend without the first plot (Random Classifier)
%legend([h2, h3, h4, h5, h6], {'Our Results (T=5s)', 'Our Results (T=150s)', 'Our Results (T=250s)','Our Results (T=2500s)','Hua2021 (T=150s)'}, 'Location', 'Best')
legend([h2, h3, h4, h5], {'Our Results (T=5s)', 'Our Results (T=150s)', 'Our Results (T=250s)','Hua2021 (T=150s)'}, 'Location', 'Best')
