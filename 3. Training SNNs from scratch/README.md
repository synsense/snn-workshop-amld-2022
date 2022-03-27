# Training SNNs from scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/synsense/snn-workshop-amld-2022/blob/master/3.%20Training%20SNNs%20from%20scratch/training_snn.ipynb)

**AMLD EPFL 2022**: Spiking Neural Networks for Low-Power Real-Time Inference

**By SynSense**: Ugurcan Cakal, Hannah Bos, Saeid Haghighatshoar, Dylan Muir

**Estimated duration**: 30 minutes

In this tutorial, we investigate and optimize the firing rate response of a simple spiking neural network (SNN). We define a single neuron with a single input channel and measure its mean firing rate. Specifically, we will explore

* how to build an SNN using the Rockpool torch-backend Leaky Integrate and Fire (LIF) model (`LIFTorch`)
* how to optimize the parameters of an SNNs using Back Propagation Through Time (BPTT)
* why error backpropagation is difficult in SNNs and how it can be tackled using surrogate gradients

<img src=https://raw.githubusercontent.com/synsense/snn-workshop-amld-2022/master/3.%20Training%20SNNs%20from%20scratch/figures/network.png width="1024">


## Xylo hardware 

The presented slides about the Xylo hardware can be found [here](https://docs.google.com/presentation/d/1yRfiK_XOehWH7qUWYqHbfLZEQ-QoqLxEGC7iQEjvl4o/edit?usp=sharing)
