# AdvSegLearn
## Descriptor
 This package creates an easy to use api for training models on small datasets with adversarial learning.
 Important classes are:
 * Mutliclass_dataset - allows for both paired and unpaired images to be loaded, with built in support for initalized data augmentation and lazy loading data augmentation.
 * train_GAN - provides api to easily load in different models, datasets and loss functions to the model. Subclassing allows for easy modification of the training loop.
 * semi_supervised_loss - a loss function for the semisupervised training loop.
 * adversarial_loss - a loss function for hte unsupervised training loop.

# License 
This package is licensed under the [MIT LICENSE](LICENSE).
