## Variational Auto Encoder

This is the normal variational autoencoder, where an Encoder maps some input onto a lower dimensional space.
The generator samples from the z-space and applies a series of deconvolutions to finally produce an image.
The stack is trained using the L2 reconstruction loss between the output image and input image.
VAE's are additionally trained with a loss term conditioning the latent variable, z, to conform to a certain distribution. 
