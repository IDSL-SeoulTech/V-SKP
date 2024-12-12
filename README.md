# Vectorized Structured Kernel Pruning

<p align="center">
  <img src="https://github.com/IDSL-SeoulTech/V-SKP/blob/master/figure/fig1_overall_structure.png" align="center" width="776" height="400"/>
</p>
## Abstract

In recent years, kernel pruning, which offers the advantages of both weight and filter pruning
methods, has been actively conducted. Although kernel pruning must be implemented as structured pruning
to obtain the actual network acceleration effect on GPUs, most existing methods are limited in that they
have been implemented as unstructured pruning. To compensate for this problem, we propose vectorized
kernel-based structured kernel pruning (V-SKP), which has a high FLOPs reduction effect with minimal
reduction in accuracy while maintaining a 4D weight structure. V-SKP treats the kernel of the convolution
layer as a vector and performs pruning by extracting the feature from each filter vector of the convolution
layer. Conventional L1/L2 norm-based pruning considers only the size of the vector and removes parameters
without considering the direction of the vector, whereas V-SKP removes the kernel by considering both
the feature of the filter vector and the size and direction of the vectorized kernel. Moreover, because the
kernel-pruned weight cannot be utilized when using the typical convolution, in this study, the kernel-pruned
weights and the input channels are matched by compressing and storing the retained kernel index in the
kernel index set during the proposed kernel pruning scheme. In addition, a kernel index convolution method
is proposed to perform convolution operations by matching the input channels with the kernel-pruned weights
on the GPU structure. Experimental results show that V-SKP achieves a significant level of parameter
and FLOPs reduction with acceptable accuracy degradation in various networks, including ResNet-50, and
facilitates real acceleration effects on the GPUs, unlike conventional kernel pruning techniques.
