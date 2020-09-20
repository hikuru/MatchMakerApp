# MatchMakerApp: A Deep Learning Command Line Application for Drug Synergy Prediction
MatchMakerApp is a command line application that predict Loewe synergy score of two given drugs. MatchMakerApp uses the our deep learning model <a href="https://www.biorxiv.org/content/10.1101/2020.05.24.113241v3">**MatchMaker**</a>

---
## Abstract
Drug combination therapies have been a viable strategy for the treatment of complex diseases such as cancer due to increased efficacy and reduced side effects. However, experimentally validating all possible combinations for synergistic interaction even with highthroughout screens is intractable due to vast combinatorial search space. Computational techniques can reduce the number of combinations to be evaluated experimentally by prioritizing promising candidates. We present MatchMaker that predicts drug synergy scores using drug chemical structure information and gene expression profiles of cell lines in a deep learning framework. For the first time, our model utilizes the largest known drug combination dataset to date, DrugComb. We compare the performance of MatchMaker with the state-of-the-art models and observe up to ∼ 20% correlation and ∼ 40% mean squared error (MSE) improvements over the next best method. We investigate the cell types and drug pairs that are relatively harder to predict and present novel candidate pairs.

---

## Authors
Halil Ibrahim Kuru, Oznur Tastan, A. Ercument Cicek

---

## Instructions Manual

### Requirements
- Python 3.7
- Numpy 1.18.1 
- Scipy 1.4.1
- Pandas 1.0.1
- Tensorflow 2.1.0
- Tensorflow-gpu 2.1.0
- Scikit-Learn 0.22.1
- keras-metrics 1.1.0
- h5py 2.10.0
- cudnn 7.6.5 (for gpu support only)


### Drug Synergy Prediction with MatchMakerApp
Download pretrained weights from <a href="https://drive.google.com/open?id=1QtMw0unMI-ZY-0z6_1bF76Cf627zXDWz">**link**</a>, and extract it into `matchmaker/`

Download drug information data from <a href="https://drive.google.com/file/d/1qo40N6sh14TRgrs9UawJ1GEuT-ulqur0/view?usp=sharing">**link**</a>, extract all files into `data/`

Run main Python script
```shell
$ python main.py
```

---

## References
- Kuru, Halil Ibrahim, Oznur Tastan, and A. Ercument Cicek. "MatchMaker: A Deep Learning Framework for Drug Synergy Prediction." bioRxiv (2020).


## License

- **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**