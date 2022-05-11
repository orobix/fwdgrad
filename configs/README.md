# Configuration

## MNIST

### config.yaml
Calls the default configuration for dataset,
model and optimization.

Also sets the device to use when CUDA is available.

### dataset
The default configuration initializes a Pytorch DataLoader with the following arguments:

* `batch_size`: how many samples per batch to load.
* `shuffle`: set to `true` to have the data reshuffled at every epoch.
* `num_workers`: how many subprocesses to use for data loading.
* `pin_memory`: if `true`, the data loader will copy Tensors into CUDA pinned memory before returning them.

Other keyword arguments for the DataLoader object can be specified.
Check [Pytorch DataLoader documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) 
for more info.

### model
The default configuration initializes a NeuralNet object with the
attributes specified in the `fwdgrad/model.py` module.

Since this object needs the initialization of other objects, like the activation
function, the recursive flag is set to `true`.

* `activation_function`: the activation function for the layers of a NeuralNet object.
Needs to be initialized.
* `hidden_sizes`: a list of integers. For each entry of this list, a layer is created
with the specified dimension.

#### Convolutional
Another configuration file is available with the name `conv.yaml`. It is needed for running the experiments
using a convolutional neural network.

It initializes a ConvNet, as specified in the paper, with the following architecture: four convolutional layers with 3x3 kernels and 64 channels, followed by two linear layers of sizes 1024 and 10. All convolutions and the first linear layer are followed by ReLU activation and there are two max-pooling layers with 2x2 kernel after the second and fourth convolutions.

### optimization
Specify some parameters for the optimization.

* `epochs`: number of epochs for the training.
* `learning_rate`: learning rate for SGD updates.

## Global Optimization
### global_optim_config.yaml
* `function`: initialize the function on which optimization will be performed.
These functions can be found under the `fwdgrad/global_optim_functions` module.
The partial flag is needed since we initialize the function without parameters.

* `seed`: seed for the Pytorch random generator.
* `learning_rate`: learning rate for SGD updates.
* `iterations`: total number of iterations to perform before stopping.