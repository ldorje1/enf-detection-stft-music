# About
MATLAB implementation of ENF-based audio forensics: extracts ENF from ENF-WHU noisy/reference recordings (https://github.com/ghua-ac/ENF-WHU-Dataset) using STFT+quadratic interpolation and MUSIC, then evaluates synchronization/time-of-recording detection by segment correlation and ROC/AUC analysis, including baseline comparison with Hua et al. (2021) (https://ieeexplore.ieee.org/document/9143185).


## Results

### ENF extraction examples
![STFT vs MUSIC ENF (one audio)](results/stft_music_oneaudio_enf.png)

![ENF difference (one audio)](results/enf_difference_oneAudio.png)

### Detection performance
![Detection accuracy (STFT)](results/dect_accu_stft.png)

![Detection accuracy (MUSIC ref)](results/dect_accu_musicref.png)

### AUC / ROC curves
![AUC curves](results/2_min_enf.png)
