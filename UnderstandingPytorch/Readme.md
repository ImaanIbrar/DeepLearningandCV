# Introduction to Basics of PyTorch

This repository contains Jupyter notebooks that provide an introduction to the basics of PyTorch.

## Notebooks

### 1. Understanding Tensors

#### Definition:
Tensors are a specialized data structure that are very similar to arrays and matrices. In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model's parameters.

![Tensors](![image](https://github.com/ImaanIbrar/RadarLab/assets/123624886/b0d48c8d-5ea4-4886-aac4-15d9f60e5350)
)

#### Initializing a Tensor:
- **Directly from data:**
- **From a NumPy Array:**
- **From another tensor:**

#### Attributes of Tensor:
Firstly initializing a Tensor:
tensor = torch.rand(3, 4)

- **Size:** `tensor.size()`
- **Shape:** `tensor.shape`
- **Datatype:** `tensor.dtype`
- **Device:** `tensor.device`

### Operations on Tensor:

There are over 100 operations that can be performed on tensors but in this notebook, the basic ones are explored:

- **Indexing and slicing:** (similar to Arrays)
- **Joining of Tensors:** `torch.cat([t1, t2, t3], dim=1)`
- **Matrix multiplication:** `@`, `.matmul()`
- **Transpose:** `.T`
- **Multiplication:** `*`, `.mul`
- **Sum of all entries:** `.sum()`

### Note:

When changes are made to the operand, they are known as in-place and are always denoted by a `_` suffix, like `tensor.add_` and `tensor.copy_(y)` and `x.t_()`.

Changes made to NumPy arrays are reflected on tensors and vice versa.

In the end, an exercise is provided to explore tensors and also the time computations between using the direct method and implementation of dot product from scratch.

### 2. Understanding DataLoader and Datasets

`Dataset` stores the samples and their corresponding labels, and `DataLoader` wraps an iterable around the `Dataset` to enable easy access to the samples.

#### Data primitives:

- `torch.utils.data.DataLoader`
- `torch.utils.data.Dataset`

### 3. Datasets & DataLoading Ways

This notebook explores two ways to create and handle data:

- Using `ImageFolder`
- Using Custom Dataset

Images of dogs and cats are used to demonstrate the use of `ImageFolder`. The Landmark dataset, a subset of the Google Landmark Data v2, is used for further demonstration.

### 4. Build Model 

This notebook explores basic model creation and explores some features of `torch.nn` api.
### 4. Optimization  

This notebook explores loss function and Optimizers and trains the FashionMNIST dataset on a neural network
and explores its accuracy with different learning rates and optimizers.  
