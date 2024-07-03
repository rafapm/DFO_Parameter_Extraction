# Compact Model Parameter Extraction via Derivative-Free Optimization

This repository accompanies the manuscript [Compact Model Parameter Extraction via Derivative-Free Optimization](https://arxiv.org/abs/2406.16355).

**Abstract:**

In this paper, we address the problem of compact model parameter extraction to simultaneously extract tens of parameters via derivative-free optimization. Traditionally, parameter extraction is performed manually by dividing the complete set of parameters into smaller subsets, each targeting different operational regions of the device, a process that can take several days or even weeks. Our approach streamlines this process by employing derivative-free optimization to identify a good parameter set that best fits the compact model without performing an exhaustive number of simulations. We further enhance the optimization process to address critical issues in device modeling by carefully choosing a loss function that evaluates model performance consistently across varying magnitudes by focusing on relative errors (as opposed to absolute errors), prioritizing accuracy in key operational regions of the device above a certain threshold, and reducing sensitivity to outliers. Furthermore, we utilize the concept of train-test split to assess the model fit and avoid overfitting. This is done by fitting 80% of the data and testing the model efficacy with the remaining 20%. We demonstrate the effectiveness of our methodology by successfully modeling two semiconductor devices: a diamond Schottky diode and a GaN-on-SiC HEMT, with the latter involving the ASM-HEMT DC model, which requires simultaneously extracting 35 model parameters to fit the model to the measured data. These examples demonstrate the effectiveness of our approach and showcase the practical benefits of derivative-free optimization in device modeling.

## Prerequisites

The examples are performed using [Keysight EDA Device Modeling](http://www.keysight.com/find/device-modeling) by simply running a transform. A license can obtained directly through Keysight.
