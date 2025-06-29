# Benchmarking Artificial Intelligence Models for Dissolved Gas Forecasting in Power Transformers
This repository builds upon the [TSLib](https://github.com/thuml/Time-Series-Library) framework. For details regarding environment setup and baseline implementations, please refer to the original TSLib repository.

## Data
The data used for this study can be downloaded from: [OneDrive](https://indiana-my.sharepoint.com/:u:/g/personal/meocakir_iu_edu/EZhjgHwuw1BDlE6AQHBk8m8B42_WykHjwViLyh3Rvm-SwQ?e=KCKGi8)

All the csv files should be put under `dataset/transformers`

To reproduce the results presented in our paper, execute the script: `scripts/run_transformer_training.sh`.


## Leaderboard
| Model         | MSE (30 Days)     | MAE (30 Days)     | MSE (60 Days)     | MAE (60 Days)     | MSE (90 Days)     | MAE (90 Days)     | MSE (120 Days)    | MAE (120 Days)    |
|:--------------|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| Reformer      | 0.788             | 0.666             | 0.993             | 0.742             | 1.178             | 0.824             | 1.297             | 0.894             |
| Informer      | 0.778             | 0.642             | 0.959             | 0.713             | 1.178             | 0.814             | 1.200             | 0.894             |
| Pyraformer    | 0.732             | 0.606             | 0.980             | 0.716             | 1.145             | 0.790             | 1.155             | 0.822             |
| Transformer   | 0.668             | 0.636             | 0.820             | 0.733             | 0.906             | 0.788             | 1.020             | 0.837             |
| Nonstationary | 0.655             | 0.535             | 0.784             | 0.605             | 0.887             | 0.661             | **0.933**         | 0.681             |
| LightTS       | 0.601             | 0.505             | 0.851             | 0.626             | 1.064             | 0.717             | 1.196             | 0.785             |
| Autoformer    | 0.571             | 0.486             | 0.718             | 0.510             | 0.943             | 0.590             | 1.118             | 0.649             |
| TSMixer       | 0.568             | 0.504             | 0.772             | 0.613             | 0.996             | 0.721             | 1.151             | 0.804             |
| DLinear       | 0.560             | 0.500             | 0.738             | 0.568             | 0.942             | 0.646             | 1.051             | 0.675             |
| TimesNet      | 0.525             | 0.419             | 0.692             | 0.490             | 0.881             | 0.570             | 1.012             | **0.614**         |
| FEDformer     | 0.509             | 0.414             | 0.685             | 0.488             | 0.959             | 0.580             | 1.123             | 0.629             |
| Crossformer   | 0.501             | 0.427             | 0.751             | 0.556             | 0.909             | 0.659             | 1.026             | 0.732             |
| MICN          | 0.499             | 0.451             | 0.735             | 0.585             | 1.042             | 0.734             | 1.235             | 0.823             |
| FiLM          | 0.489             | 0.372             | 0.693             | 0.466             | 0.982             | 0.564             | 1.055             | 0.617             |
| SegRNN        | 0.468             | 0.383             | _0.642_           | 0.467             | **0.853**         | _0.562_           | 1.001             | 0.621             |
| ETSformer     | 0.461             | 0.383             | 0.674             | 0.482             | 0.935             | 0.603             | 1.103             | 0.656             |
| TiDE          | 0.449             | 0.362             | 0.647             | **0.444**         | 0.927             | **0.556**         | 1.102             | 0.617             |
| iTransformer  | 0.441             | _0.359_           | 0.653             | _0.454_           | 0.929             | 0.565             | 1.111             | 0.625             |
| TimeXer       | 0.440             | **0.358**         | 0.649             | 0.455             | 0.912             | 0.564             | 1.089             | 0.620             |
| TimeMixer     | 0.433             | 0.364             | **0.639**         | 0.462             | _0.877_           | 0.566             | 1.010             | _0.616_           |
| PatchTST      | _0.429_           | 0.365             | 0.643             | 0.461             | 0.888             | 0.570             | 1.020             | 0.625             |
| PAttn         | **0.424**         | _0.359_           | _0.642_           | 0.465             | **0.853**         | 0.578             | _0.959_           | 0.622             |



* List of models that are included in this repository. 

  - [x] **TimeXer** - TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables [[NeurIPS 2024]](https://arxiv.org/abs/2402.19072) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeXer.py)
  - [x] **TimeMixer** - TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting [[ICLR 2024]](https://openreview.net/pdf?id=7oLshfEIC2) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py).
  - [x] **TSMixer** - TSMixer: An All-MLP Architecture for Time Series Forecasting [[arXiv 2023]](https://arxiv.org/pdf/2303.06053.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TSMixer.py)
  - [x] **iTransformer** - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [[ICLR 2024]](https://arxiv.org/abs/2310.06625) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py).
  - [x] **PatchTST** - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers [[ICLR 2023]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py).
  - [x] **TimesNet** - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis [[ICLR 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py).
  - [x] **DLinear** - Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]](https://arxiv.org/pdf/2205.13504.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py).
  - [x] **LightTS** - Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures [[arXiv 2022]](https://arxiv.org/abs/2207.01186) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py).
  - [x] **ETSformer** - ETSformer: Exponential Smoothing Transformers for Time-series Forecasting [[arXiv 2022]](https://arxiv.org/abs/2202.01381) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/ETSformer.py).
  - [x] **Non-stationary Transformer** - Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/pdf?id=ucNDIDRNjjv) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Nonstationary_Transformer.py).
  - [x] **FEDformer** - FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting [[ICML 2022]](https://proceedings.mlr.press/v162/zhou22g.html) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FEDformer.py).
  - [x] **Pyraformer** - Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting [[ICLR 2022]](https://openreview.net/pdf?id=0EXmFzUn5I) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Pyraformer.py).
  - [x] **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [[NeurIPS 2021]](https://openreview.net/pdf?id=I55UqU-M11y) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py).
  - [x] **Informer** - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [[AAAI 2021]](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py).
  - [x] **Reformer** - Reformer: The Efficient Transformer [[ICLR 2020]](https://openreview.net/forum?id=rkgNKkHtvB) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Reformer.py).
  - [x] **Transformer** - Attention is All You Need [[NeurIPS 2017]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py).
  - [x] **MultiPatchFormer** - A multiscale model for multivariate time series forecasting [[Scientific Reports 2025]](https://www.nature.com/articles/s41598-024-82417-4) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MultiPatchFormer.py)
  - [x] **WPMixer** - WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting [[AAAI 2025]](https://arxiv.org/abs/2412.17176) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/WPMixer.py)
  - [x] **PAttn** - Are Language Models Actually Useful for Time Series Forecasting? [[NeurIPS 2024]](https://arxiv.org/pdf/2406.16964) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PAttn.py)
  - [x] **Mamba** - Mamba: Linear-Time Sequence Modeling with Selective State Spaces [[arXiv 2023]](https://arxiv.org/abs/2312.00752) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Mamba.py)
  - [x] **SegRNN** - SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting [[arXiv 2023]](https://arxiv.org/abs/2308.11200.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/SegRNN.py).
  - [x] **Koopa** - Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors [[NeurIPS 2023]](https://arxiv.org/pdf/2305.18803.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Koopa.py).
  - [x] **FreTS** - Frequency-domain MLPs are More Effective Learners in Time Series Forecasting [[NeurIPS 2023]](https://arxiv.org/pdf/2311.06184.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FreTS.py).
  - [x] **MICN** - MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=zt53IDUR1U)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MICN.py).
  - [x] **Crossformer** - Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=vSVLM2j9eie)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Crossformer.py).
  - [x] **TiDE** - Long-term Forecasting with TiDE: Time-series Dense Encoder [[arXiv 2023]](https://arxiv.org/pdf/2304.08424.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TiDE.py).
  - [x] **SCINet** - SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction [[NeurIPS 2022]](https://openreview.net/pdf?id=AyajSjTAzmg)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/SCINet.py).
  - [x] **FiLM** - FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/forum?id=zTQdHSQUQWc)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FiLM.py).
  - [x] **TFT** - Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting [[arXiv 2019]](https://arxiv.org/abs/1912.09363)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TemporalFusionTransformer.py). 
 
