---
created: 2025-02-24T13:36:02 (UTC -07:00)
tags: []
source: https://flow.byu.edu/me595r/schedule/hw7/
author: 
---

# HW · ME 595R

> ## Excerpt
> due 2/27/2025 before midnight via Learning Suite
25 possible points

---
## HW 7: Deep Koopman

due 2/27/2025 before midnight via Learning Suite 25 possible points

___

We’ll use deep learning to learn a Koopman operator using the methodology of [this paper](https://www.nature.com/articles/s41467-018-07210-0) (though we’ll simplify it somewhat for this homework). At a minimum you’ll want to refer to figure 1, the subsection “Deep learning to identify Koopman eigenfunctions” under Results, and the subsection “Explicit loss function” under Methods. We’ll make the following simplifications: we won’t need the auxiliary network to learn eigenvalues, and so won’t need to construct a block diagonal K (this is a nice approach for better accuracy and for explainability of the results, but we won’t worry about it in this case and will just learn the whole matrix K directly). I did not add the infinity norm loss in (15), for eq (13) I used the entire time horizon (so eq (13) and (14) use the same sum across time).

The data comes from glycolysis pathway dynamics. I ran the simulations and pretabulated the data [here](https://flow.byu.edu/me595r/schedule/kdata.txt) to save time. The following commands will load the data:

```
<span>ntraj</span> <span>=</span> <span>2148</span>  <span># number of trajectories
</span><span>nt</span> <span>=</span> <span>50</span>  <span># number of time steps
</span><span>ny</span> <span>=</span> <span>7</span>  <span># number of states
</span>
<span>tvec</span> <span>=</span> <span>np</span><span>.</span><span>linspace</span><span>(</span><span>0</span><span>,</span> <span>350</span><span>,</span> <span>nt</span><span>)</span>
<span>Y</span> <span>=</span> <span>np</span><span>.</span><span>loadtxt</span><span>(</span><span>'</span><span>kdata.txt</span><span>'</span><span>).</span><span>reshape</span><span>(</span><span>ntraj</span><span>,</span> <span>nt</span><span>,</span> <span>ny</span><span>)</span>
<span>Ytrain</span> <span>=</span> <span>Y</span><span>[:</span><span>2048</span><span>,</span> <span>:,</span> <span>:]</span>  <span># 2048 training trajectories
</span><span>Ytest</span> <span>=</span> <span>Y</span><span>[</span><span>2048</span><span>:,</span> <span>:,</span> <span>:]</span>  <span># 100 testing trajectoreis
</span>
```

You should be able to find a linear mapping (matrix K) that reasonably reproduces the dynamics in the testing data set. Plot the dynamics for the first trajectory in the dataset against your trained model (data with dashed line, and model solid lines). Only plot the first three states just to keep the plot less busy.

Tips:

-   It’s generally a good idea to start by overfitting the data. Use only a relatively small number of training trajectories so that things run fast, and make sure you can train the model to reproduce your training data (ignore testing data for now). Until you can fit the training data there is no point trying to generalize to the testing data. And by keeping it small you can iterate quickly and make sure your loss functions, etc. are setup properly.
-   I’d then add more training data and modify hyperparameters until you can get good predictions (still with training data). Then try to generalize to the testing data. You might not need to use all 2048 data trajectories. I just keep doubling the number of training data points until I was able to get my test set error down.
-   A GPU (google colab) will be helpful when you start using more data.
-   For the first set of epochs I only train the autoencoder, and then add on the losses for linearity and prediction.
-   If you create a `nn.Parameter` for K within your `nn.Module` then when you pass `model.parameters()` to your optimizer it automatically includes K along with all the model weights and biases. Or, even easier, you can create a linear layer with bias=False.
-   When you initialize K, keep its weights small. Since xk+1\=Kxk, if the entires in K are large your dynamics will blow up quickly.
-   If you’re struggling with generalization (training error is going down but testing error is going up) a few techniques you can try including using more data (if you haven’t already maxed that out), using dropout, adding a penalty on model weights.
