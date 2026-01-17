# About
MATLAB implementation of ENF-based audio forensics: extracts ENF from ENF-WHU noisy/reference recordings (https://github.com/ghua-ac/ENF-WHU-Dataset) using STFT+quadratic interpolation and MUSIC, then evaluates synchronization/time-of-recording detection by segment correlation and ROC/AUC analysis, including baseline comparison with Hua et al. (2021) (https://ieeexplore.ieee.org/document/9143185).


## Results

### ENF extraction examples
<img src="results/stft_music_oneaudio_enf.png" width="550">
<br>
<img src="results/enf_difference_oneAudio.png" width="550">

### Detection performance
<img src="results/2_min_enf.png" width="550">

### AUC / ROC curves
<img src="results/dect_accu_stft.png" width="550">
<br>
<img src="results/dect_accu_musicref.png" width="550">
