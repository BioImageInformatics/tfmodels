## Experiments
Here are some demo's to use the built-in model definitions.
This section is in constant progress, as the model implementations are constantly being updated.
In most cases, the base implementations are for MNIST data, i.e. 28 x 28 squares.
Also, in most cases there is a demo template for customized model definitions, where it would be appropriate to change a few constants to make the dimensions work out.

### Structure
Note that this section resides outside the `tfmodels` package.
This choice facilitates what I think is a convenient way to use `tfmodels` in experiments and projects where you might have customized model definitions as extensions of the core classes.
Each script begins by adding the `tfmodels` package to python's path.

### Data
Generative models (vae, gan, wgan) use MNIST data, packaged up under assets.
Multiple instance models also use MNIST data, and the BaggedMNIST dataset class.
(Note, there's an updated implementation of multiple-instance networks [HERE].)
Segmentation, and image-regression will have to be data supplied by yourself, for now.
There's a thought to include an interface to some public benchmark dataset such as CamVid, but this is low priority.
