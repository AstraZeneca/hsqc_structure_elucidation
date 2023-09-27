# HSQC Spectra Simulation and Matching Tool
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WawzugDDloQSxrToI6T66gm9cfXZqWAm?usp=sharing
)
![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)

Implementation of the following publication: 

**Advancing HSQC Spectral Matching: A Comparative Study of Peak-Matching and Simulation Techniques for Molecular Identification**

Link to the paper: [link](https://dummy.com) (dummy)

<img src="./dump/graphical_abstract.png" width="500">

*The tool allows for identification of correct molecules among similar analogues (molecules with the very similar molecular weight or different regio- or stereoisomers)*



## What is this?
![Alt text](./dump/Paper_1_Figure_v3_GITHUB.png)

This tool provides a comprehensive platform for simulating and matching Heteronuclear Single Quantum Coherence (HSQC) spectra which can be used to facilitate molecular structure elucidation. 
The tool provides an implementation of a machine learning based 1H and 13C NMR prediction with a graph-based neural network (ML) which was published as follows:
**Scalable graph neural network for NMR chemical shift prediction**  [URL](https://pubs.rsc.org/en/Content/ArticleLanding/2022/CP/D2CP04542G)
from *Jongmin Han, Hyungu Kang, Seokho Kang, Youngchun Kwon, Dongseon Lee and Youn-Suk Choi* 

Furthermore, it incorporates four distinct HSQC simulation techniques: ACD-Labs (ACD), MestReNova (MNova), Gaussian NMR calculations (DFT), and a graph-based neural network (ML). For DFT and ML, we've supplemented the techniques with a self-implemented 2D HSQC reconstruction logic. We've also devised three peak-matching strategies—Minimum-Sum (MinSum), Euclidean-Distance (EucDist), and Hungarian-Distance (HungDist)—which are combined with three padding approaches—zero-padding (Zero), peak-truncated (Trunc), and nearest-neighbor double assignment (NN) which can be selected for peak matching. 
<img src="./dump/peak_padding.png" width="800">

The tool is adept at handling molecules with very similar molecular weight or different regio- or stereoisomers, thereby facilitating the identification of correct molecules among similar analogues. Additionally, our methodology shows robust performance in resolving ambiguous structural assignments, as demonstrated on a set of previously misassigned molecules.
The tool is linked with a Google Colab notebook that allows users to apply our methodology to their own data, run the ML NMR prediction, and learn how to generate simulated spectra with commercial software. It also provides instructions on processing real spectra and conducting similarity comparisons using the algorithms we've implemented. This hands-on, interactive tool is designed to enhance user understanding and practical application of the methodologies used.

Try it out yourself using the following Google Colab Notebook:
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WawzugDDloQSxrToI6T66gm9cfXZqWAm?usp=sharing
) 
https://colab.research.google.com/drive/1WawzugDDloQSxrToI6T66gm9cfXZqWAm?usp=sharing
## Want to see a short video demonstration and user tutorials?
| Tutorial: Colab Notebook |  Tutorial: ACD HSQC Simulation |  Tutorial: MNova HSQC Simulation |  Tutorial: Experimental Data Preparation |
|:-:|:-:|:-:|:-:|
| [![](./dump/Colab_Notebook.PNG)](https://youtu.be/w59bVTpJmZY) | [![](./dump/MNova.PNG)](https://youtu.be/RyMQuRYtpbM) | [![](./dump/ACD.PNG)](https://youtu.be/xymw0ZRF8Xo) | [![](./dump/experimental_peak_picking.PNG)](https://youtu.be/NYW5bve198U) |




