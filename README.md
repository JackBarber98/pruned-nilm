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

**[4]** Asouri, A. H., Abdelrahman, T. S., Remedios, A. D. (2019) Retraining-Free Methods for Fast On-the-Fly Pruning of Convolutional Neural Networks. _Neurocomputing_, 370 56-59. Available from https://pdf.sciencedirectassets.com/271597/1-s2.0-S0925231219X00405/1-s2.0-S0925231219312019/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEGUaCXVzLWVhc3QtMSJIMEYCIQDpMg0rKLoZUhh%2BxuG4x2OZGixbPGELgt5LvXKAvYYWaAIhALvx1yf3HZ2tL5IkU%2FPocIPKz2YltzZESoMHha85CDurKr0DCO3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQAhoMMDU5MDAzNTQ2ODY1IgxDKHKcYGfUWeEttHoqkQNlWV0O%2BATZgfLHBX208x%2FDKawz3A7VA4IWqUNa6tmX%2Ftw3DpBSCG00CLoY1tmKTzY4jFPcVW27KDJ7Fph0BEbfYZUk%2FR1cPEtmE7u8rXQL%2BYfKVbFB9%2FvyjLlOGF%2Ft283wRkPc93D7sRhe1revbh4wCEjDRe%2Fbm%2FGj%2B6qA4Vyt7LlX9YXsbTbDb3zYB7l339k4N%2BJKugHDR6lDAPEv74uPX%2FdDonSEvGl88fIv7atOxrlAUHO6WxRW0CzRE35WuJOO%2BNVCCAL45MBpeqF2XmJCOM89fxTdZOgEvE7kr5rFnySaC%2BASl0rf7a7%2Bv%2BCef265WpwQ%2BFq8KUzhnnEtLbXAN%2BWrxX10I37eoxhuz8ENy%2FIAnPmMYy7Vrmem74hVDMo%2B5tCJAWfl3HHq6pnOpdd%2B8Ewhwwj2YQbSrEvR%2Fh%2Fivrc%2Bfoald4TFvzgqRLUtjs8TSFmNkLwnYyqRYdWja7v2jjpF1aLjYRR7J8IfHwCBto7hF%2FgDbqadqBWNdMmm3OFrB8io4tCa4NQJTuKaidyYzDCRr5bxBTrqAWPEEIwzA1Hz5RwJu65xpp9%2BfSrquOnke%2FZ3q4ugl2goA7%2FwJueEWu8zPxc5FgPjLu9sIgNMizlw3cw8%2FQWk0v46%2BCYHm%2Bysb9JlChCubLDFGtHnTAD4W4t%2FnZlrC3DwtLwzqr%2BjisJOF7YJpxY9mYoVh2IUaloutPmPE1E4cZOPPQV95%2BILsXy9Ysi845mmloH5XwcgH8Fmku1URIShRMpUp%2BUssNAOxaA8RASXoXAawIyq2tYGVemqhOFOy%2BQSDtAcLotUnGnnSxh1EomSxN%2F3fzGLdfaNRchQZP%2B175Tya4oOOcQ%2FCejQoA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20200120T130550Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYXPIU7AHF%2F20200120%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=12a194d9fc8dbf2c0afdf5becb3484095122f29cb152b8fe9e29b64369f95d5f&hash=7e57517e09976b83550e004fe80295590eeabd8a510dbd78db0eace50b8c83a8&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0925231219312019&tid=spdf-e1b93937-398d-4d0f-8352-75920f99b58b&sid=5cc7b8d51dadd847c79a4ce19b8a91634fefgxrqb&type=client [accessed 20 January 2020].

**[5]** Tensorflow (undated). _Magnitude-Based Weight Pruning with Keras_. Available from https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras [accessed 17 January 2020].
