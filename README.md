# About
MATLAB implementation of ENF-based audio forensics: extracts ENF from ENF-WHU noisy/reference recordings (https://github.com/ghua-ac/ENF-WHU-Dataset) using STFT+quadratic interpolation and MUSIC, then evaluates synchronization/time-of-recording detection by segment correlation and ROC/AUC analysis, including baseline comparison with Hua et al. (2021) (https://ieeexplore.ieee.org/document/9143185).

## Run the algorithms (Step-by-step)

### 1) Download the dataset
Download the ENF-WHU dataset from:
- https://github.com/ghua-ac/ENF-WHU-Dataset/tree/master/ENF-WHU-Dataset

You only need these folders:
- `H1/` (noisy recordings)
- `H1_ref/` (reference recordings)

### 2) Place the audio into this repo
Copy the dataset folders into `audio-data/` so the structure becomes:

```text
audio-data/
├─ H1/
│  ├─ 001.wav
│  ├─ 002.wav
│  └─ ...
└─ H1_ref/
   ├─ 001_ref.wav
   ├─ 002_ref.wav
   └─ ...
```
### 3) Run ENF extraction (STFT + MUSIC)
```source/twin_algo_main.mat```

### 4) Run detection evaluation + AUC curves
```source/enf2correlator_audio.mat```
## Results

### ENF extraction examples
<img src="results/stft_music_oneaudio_enf.png" width="550">
<br>
<img src="results/enf_difference_oneAudio.png" width="550">

### Detection performance
<img src="results/2_min_enf.png" width="550">

### AUC / ROC curves
#### Using STFT
<img src="results/dect_accu_stft.png" width="550">
<br>

#### Using MUSIC
<img src="results/dect_accu_musicref.png" width="550">


## Citation

Part of this repository was used in our paper:

**Mohsen Hatami, Lhamo Dorje, Xiaohua Li, Yu Chen**, *“Is ENF a Good Physical Fingerprint?”* (accepted at **IEEE CCNC 2026**).  
The paper PDF is included in this repo: `Does_ENF_Work_.pdf`.

If you find this code useful, please cite:

```bibtex
@inproceedings{hatami2026enf,
  title     = {Is ENF a Good Physical Fingerprint?},
  author    = {Hatami, Mohsen and Dorje, Lhamo and Li, Xiaohua and Chen, Yu},
  booktitle = {2026 IEEE Consumer Communications \& Networking Conference (CCNC)},
  year      = {2026},
  note      = {Accepted for publication at IEEE CCNC 2026},
}



