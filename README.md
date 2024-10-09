# Compact Model Parameter Extraction via Derivative-Free Optimization

This repository accompanies the IEEE Access Journal ["Compact Model Parameter Extraction via Derivative-Free Optimization."](https://ieeexplore.ieee.org/abstract/document/10662920)

**Abstract:**

In this paper, we address the problem of compact model parameter extraction to simultaneously extract tens of parameters via derivative-free optimization. Traditionally, parameter extraction is performed manually by dividing the complete set of parameters into smaller subsets, each targeting different operational regions of the device, a process that can take several days or weeks. Our approach streamlines this process by employing derivative-free optimization to identify a good parameter set that best fits the compact model without performing an exhaustive number of simulations. We further enhance the optimization process to address three critical issues in device modeling by carefully choosing a loss function that focuses on relative errors rather than absolute errors to ensure consistent performance across different orders of magnitude, prioritizes accuracy in key operational regions above a specific threshold, and reduces sensitivity to outliers. Furthermore, we utilize the concept of train-test split to assess the model fit and avoid overfitting. We demonstrate the effectiveness of our approach by successfully modeling a diamond Schottky diode with the SPICE diode model and a GaN-on-SiC HEMT with the ASM-HEMT model. For the latter, which involves extracting 35 parameters for the ASM-HEMT DC model, we identified the best set of parameters in under 6,000 trials. Additional examples using both devices are provided to demonstrate robustness to outliers, showing that an excellent fit is achieved even with over 25% of the data purposely corrupted. These examples demonstrate the practicality of our approach, highlighting the benefits of derivative-free optimization in device modeling.

## Prerequisites

The examples in this repository can be run using:

1) [Keysight EDA Device Modeling](http://www.keysight.com/find/device-modeling) by running a transform. Note that a license is required, which can obtained directly through Keysight.
2) [Ngspice](https://ngspice.sourceforge.io/) (an open-source SPICE circuit simulator) using a Jupyter Notebook through Google Colab.

For both options, it is necessary to install [Optuna](https://optuna.org/) (a hyperparameter optimization framework).

The easiest way to get started is by exploring our Google Colab examples for a [diode](https://colab.research.google.com/drive/1vlVsj_4leCJ4xJy8r8VxMdwLVOM-USWN?usp=drive_link) and a [GaN HEMT](https://colab.research.google.com/drive/1M9QDa1-GiKbrjiNZ2k_QYIkSa30PyKHl?usp=drive_link), which use the SPICE diode model and the ASM-HEMT model, respectively.

You can install Optuna via pip:

```sh
$ pip install optuna
