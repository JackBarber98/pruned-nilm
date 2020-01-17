# prunedNILM
Contempory pruning algorithms applied to transferable NILM models

# Pruning Algorithms for Seq2Point Energy Disaggregation

This code base implements three contemporary pruning algorithms designed to reduce the size of a typical sequence-to-point deep learning model [1] for use in energy disaggreation / non-intrusive load monitoring.

Support is also provided for transfer learning. 

## Pruning Algorithms

### Structured Probabilistic Pruning

A pruning algorithm proposed by Wang et al. [2]. The probability of a non-specific weight in a layer / convolutional filter being pruned is calculated. A weight mask for each layer is produced using Monte Carlo sampling that, in turn, is used to set insignificant weights to zero.

SPP pruning can be utilised during training using the following:
```bash
python train_main.py --pruning_algorithm="spp"
```

### Entropy-Based Pruning

A pruning algorithm proposed by Hur et al. [3]. Gaussian distribution is used to calculate the information gain potential of a specific weight in a layer. By comparing this value to the overall entropy of a layer, weights that have a small value and are less likely to gain information as training continues can be safely removed from the network. 

Entropic pruning can be utilised during training using the following:
```bash
python train_main.py --pruning_algorithm="entropic"
```

### Low Magnitude Pruning

The default pruning algorithm utilised by the Tensorflow Model Optimisation Toolkit [4]. In this implementation, the smallest 50% of weights are removed from the model during training.

TFMOT pruning can be utilised during training using the following:
```bash
python train_main.py --pruning_algorithm="tfmot"
```

## References

**[1]** Zhang, C., Zhong., Wang, Z., Goddard, N., Sutton, S. (2017) _Sequence-to_Point Learning with Neural Networks for Non-Intrusive Load Monitoring_. Available from https://arxiv.org/pdf/1612.09106.pdf [accessed 17 January 2020].

**[2]** Wang, H., Zhang, Q., Wang, Y., Hu, H. (2018) _Structured Probabilistic Pruning for Convolutional Neural Network Acceleration_. Zhejiang University, China. Available from https://arxiv.org/pdf/1709.06994.pdf [accessed 17 January 2020].

**[3]** Hur, C., Kang, S. (2018) Entropy-Based Pruning Method For Convolutional Neural Networks. _The Journal of Supercomputing_, 75:2950–2963. Available from https://link-springer-com/content/pdf/10.1007/s11227-018-2684-z.pdf [accessed 17 January 2020].

**[4]** Tensorflow (undated). _Magnitude-Based Weight Pruning with Keras_. Available from https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras [accessed 17 January 2020].
