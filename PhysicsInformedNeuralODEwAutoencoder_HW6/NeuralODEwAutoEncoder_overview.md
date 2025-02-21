---
created: 2025-02-17T11:47:44 (UTC -07:00)
tags: []
source: https://flow.byu.edu/me595r/schedule/hw6/
author: 
---

# HW · ME 595R

> ## Excerpt
> due 2/20/2025 before midnight via Learning Suite
25 possible points

---
## HW 6: Neural ODE with Autoencoder (and physics loss)

due 2/20/2025 before midnight via Learning Suite 25 possible points

___

Autoencoders are used in many architectures to either reduce or increase dimensionality. In this problem we solve a high dimensional ODE problem by first projecting it into a low dimensional space (encoder). This low dimensional space is called the latent space. Next we solve the neural ODE in the latent space. Finally, we project the solution back to the original high-dimensional space (decoder). Our objective will simultaneously train the neural net in the ODE and the neural nets for the encoder/decoder.

[This paper](https://www.nature.com/articles/s41598-023-36799-6) uses that approach but also adds a physics-based loss term. That allows us to have more accurate solutions outside the training data. Read the methods section to better understand the formulation.

We’ll reproduce the first example: the lifted duffing oscillator. In [this file](https://flow.byu.edu/me595r/schedule/pinodedata.py) I’ve written functions that generate training data, testing data, and collocation points. I’ve reproduced the results in the paper, but it takes a little while to run. To make things easier I’ve expanded the range of the training data a bit (rather than limiting to just the red region, I expanded the training data to a little over half the domain with collocation points on the remainder of the domain). That should allow you to get by with significantly less training data and collocation points (I used about 600 training points, and 10,000 collocation points but could have gotten away with less). Even still, you may find it beneficial to use a GPU. Our goal is to get the MSE error for the test set (100 test points) below 0.1.

You should also be able to produce a plot like the lower right of Fig 3 (except we won’t worry about coloring separate regions). I provided a function true\_encoder that you can use (the paper also uses the true encoder for the visualization). We can use our trained encoder for this projection, but it won’t necessarily match since there will be many possible solutions for a good latent space. So in general this isn’t something that one would know, it just helps in this particularly case where we know what the projection looks like to see if our training is on track.

Tips:

-   torchdiffeq uses float64 by default, and you’ll probably want to keep it that way, especially if using a discrete adjoint. But if you want to use float32 you can change it by passing `options={'dtype': torch.float32}` to odeint.
-   Figure 1 seems to suggest that the prediction loss only compares the final time step, but you’ll have the full trajectory and should compare all of it (which is what they actually do as shown in Eq 5). Similarly, that figure seems to suggest that the reconstruction loss uses only the initial time step, but like Eq 4 shows, you’d want to check reconstruction for the full trajectory.
-   Make sure you understand how `reshape` works and make sure you aren’t using it when you should be using `permute`.
-   If you want to batch, you’ll need to create two data loaders since the training data and the collocation data have different sizes. To sync them up you could preselect the number of batches you want (`nbatches` below) then use that number to calculate what batch size you need for each.
    
    ```
      <span>batch_size_train</span> <span>=</span> <span>int</span><span>(</span><span>ntrain</span> <span>/</span> <span>nbatches</span><span>)</span>
      <span>batch_size_collocation</span> <span>=</span> <span>int</span><span>(</span><span>ncol</span> <span>/</span> <span>nbatches</span><span>)</span>
    ```
    
-   You can get dz/dx in a similar manner to what we’ve done on previous homeworks with autograd.grad (you’ll need two separate calls to autograd.grad since you have two different batched outputs for the two dimensions of z). You’ll then need to use torch.matmul or torch.bmm to get to dz/dt. There is a Jacobian function that can be used instead of two calls to grad, and there is also a jvp function (Jacobian vector product) that could be used to combine the grad and matrix multiply steps in one line. But understanding these in a batched mode takes more explanation so I’d just use autograd.grad, and if you’re interested in the other options we can discuss on Slack.
-   Per usual, start small. Get things running with a small number of collocation points, and perhaps weights.
-   If you want to use a GPU, Google Colab is a good option. In the upper right of Colab you need to change your runtime type to the GPU. In your code you need to set `device = "cuda"` (to allow it to run on the GPU or locally without changing code you can use `device = 'cuda' if torch.cuda.is_available() else 'cpu'`). You also need to moved all the data in your torch tensors, including those in the model, to the gpu device. (i.e., `model.double().to(device)`, `torch.tensor(x, dtype=torch.float64, device=device)`). Note that Google has time limits on free GPU usage.
