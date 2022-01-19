## VIGNet: A Deep Convolutional Neural Network for EEG-based Driver Vigilance Estimation
<p align="center"><img width="50%" src="files/framework.png" /></p>

This repository provides a TensorFlow implementation of the following paper:
> **VIGNet: A Deep Convolutional Neural Network for EEG-based Driver Vigilance Estimation**<br>
> [Wonjun Ko](https://scholar.google.com/citations?user=Fvzg1_sAAAAJ&hl=ko&oi=ao)<sup>1</sup>, [Kwanseok Oh](https://scholar.google.com/citations?user=EMYHaHUAAAAJ&hl=ko)<sup>2</sup>, [Eunjin Jeon](https://scholar.google.com/citations?user=U_hg5B0AAAAJ&hl=ko)<sup>1</sup>, [Heung-Il Suk](https://scholar.google.co.kr/citations?user=dl_oZLwAAAAJ&hl=ko)<sup>1, 2</sup><br/>
> (<sup>1</sup>Department of Brain and Cognitive Engineering, Korea University) <br/>
> (<sup>2</sup>Department of Artificial Intelligence, Korea University) <br/>
> [[Official version]](https://ieeexplore.ieee.org/abstract/document/9403717)
> [[Official version]](https://ieeexplore.ieee.org/abstract/document/9061668)
> Presented in the 8th IEEE International Winter Conference on Brain-Computer Interface (BCI)
> 
> **Abstract:** *Estimating driver fatigue is an important issue for traffic safety and user-centered brain–computer interface. In this paper, based on differential entropy (DE) extracted from electroencephalography (EEG) signals, we develop a novel deep convolutional neural network to detect driver drowsiness. By exploiting DE of EEG samples, the proposed network effectively extracts class-discriminative deep and hierarchical features. Then, a densely-connected layer is used for the final decision making to identify driver condition. To demonstrate the validity of our proposed method, we conduct classification and regression experiments using publicly available SEED-VIG dataset. Further, we also compare the proposed network to other competitive state-of-the-art methods with an appropriate statistical analysis. Furthermore, we inspect the real-world usability of our method by visualizing a change in the probability of driver status and confusion matrices.*


## Dependencies
* [Python 3.6+](https://www.continuum.io/downloads)
* [TensorFlow 2.0.0+](https://www.tensorflow.org/)

## Downloading datasets
To download GIST-MI dataset
* http://gigadb.org/dataset/100295

To download KU-MI/SSVEP dataset
* http://gigadb.org/dataset/100542

To download SEED-VIG dataset
* https://bcmi.sjtu.edu.cn/~seed/seed-vig.html

To download CHB-MIT dataset
* https://physionet.org/content/chbmit/1.0.0/

## Citation
If you find this work useful for your research, please cite our [paper](https://ieeexplore.ieee.org/abstract/document/9403717):
```
@article{ko2021multi,
  title={Multi-scale neural network for EEG representation learning in BCI},
  author={Ko, Wonjun and Jeon, Eunjin and Jeong, Seungwoo and Suk, Heung-Il},
  journal={IEEE Computational Intelligence Magazine},
  volume={16},
  number={2},
  pages={31--45},
  year={2021},
  publisher={IEEE}
}
```

## Acknowledgements
This work was supported by Institute for Information & Communications Technology Promotion (IITP) grant funded by the Korea government under Grant 2017-0-00451 (Development of BCI based Brain and Cognitive Computing Technology for Recognizing User’s Intentions using Deep Learning) and Grant 2019-0-00079 (Department of Artificial Intelligence, Korea University).
