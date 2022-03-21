# Training SNNs from Scractch

In this tutorial, we will investigate and optimize the firing rate response of a simple spiking neural network. We will define a single neuron with a single input channel, then measure the mean firing rate of the spiking neuron throughout the simulation duration. Specifically, we will explore

* How to build an SNN using Rockpool torch-backend Leaky Integrate and Fire (LIF) model namely `LIFTorch`
* How to optimize parameters of an SNNs directly using Back Propagation Through Time (BPTT)
* Why gradients vanish at spikes and how the problem can be alleviated using surrogate gradients

Below there is a diagram visualizing the model of interest.

<img src=https://raw.githubusercontent.com/synsense/snn-workshop-amld-2022/master/3.%20Training%20SNNs%20from%20scracth/figures/network.png width="1024">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JhBci7CkqlCTS4u5dXesSTNQ_xqP2wfn#scrollTo=uy3yH2T691Na)
