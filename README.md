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

### Relative Threshold Pruning

An implementation of the relative threshold approach to pruning devised by Ashouri et al. [4]. The value of the n-th percentile weight is selected as a threshold, with any weights with a value less than this being set to zero during the pruning process. It is possible to conduct pruning with this algorithm after training has occurred.

Relative threshold pruning can be utilised during training using the following:
```bash
python train_main.py --pruning_algorithm="threshold"
```

### Low Magnitude Pruning

The default pruning algorithm utilised by the Tensorflow Model Optimisation Toolkit [5]. In this implementation, the smallest 50% of weights are removed from the model during training.

TFMOT pruning can be utilised during training using the following:
```bash
python train_main.py --pruning_algorithm="tfmot"
```

## Usage

### Training

```python train_main.py``` will train a seq2point model either with or without a pruning algorithm. The following options are available:

```
  --appliance_name APPLIANCE_NAME
                        The name of the appliance to train the network with. Default is kettle. 
                        Available are: kettle, fridge, washing machine,
                        dishwasher, and microwave.
  --batch_size BATCH_SIZE
                        The batch size to use when training the network.
                        Default is 1000.
  --crop CROP           The number of rows of the dataset to take training
                        data from. Default is 10000.
  --pruning_algorithm PRUNING_ALGORITHM
                        The pruning algorithm that the network will train
                        with. Default is none. Available are: spp, entropic,
                        threshold.
```

### Testing

```python test_main.py``` will test a pre-trained seq2point model. The following options are available:

```
  --appliance_name APPLIANCE_NAME
                        The name of the appliance to perform disaggregation
                        with. Default is kettle. Available are: kettle,
                        fridge, dishwasher, microwave.
  --transfer_domain TRANSFER_DOMAIN
                        The appliance used to train the model that you would
                        like to test (i.e. transfer learning).
  --batch_size BATCH_SIZE
                        The batch size to use when training the network.
                        Default is 1000.
  --crop CROP           The number of rows of the dataset to take training
                        data from. Default is 10000.
```


## References

**[1]** Zhang, C., Zhong., Wang, Z., Goddard, N., Sutton, S. (2017) _Sequence-to_Point Learning with Neural Networks for Non-Intrusive Load Monitoring_. Available from https://arxiv.org/pdf/1612.09106.pdf [accessed 17 January 2020].

**[2]** Wang, H., Zhang, Q., Wang, Y., Hu, H. (2018) _Structured Probabilistic Pruning for Convolutional Neural Network Acceleration_. Zhejiang University, China. Available from https://arxiv.org/pdf/1709.06994.pdf [accessed 17 January 2020].

**[3]** Hur, C., Kang, S. (2018) Entropy-Based Pruning Method For Convolutional Neural Networks. _The Journal of Supercomputing_, 75:2950–2963. Available from https://link-springer-com/content/pdf/10.1007/s11227-018-2684-z.pdf [accessed 17 January 2020].

**[4]** Tensorflow (undated). _Magnitude-Based Weight Pruning with Keras_. Available from https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras [accessed 17 January 2020].
