---
created: 2025-03-12T10:31:19 (UTC -06:00)
tags: []
source: https://flow.byu.edu/me595r/schedule/hw8/
author: 
---

# HW · ME 595R

> ## Excerpt
> due 3/13/2025 before midnight via Learning Suite
25 possible points

---
## HW 8: Convolutional neural net for super resolution (with physics)

due 3/13/2025 before midnight via Learning Suite 25 possible points

___

We will use convolutional neural nets to perform super resolution of coarse MRI data using the methodology in this [paper](https://pubs.aip.org/aip/pof/article-abstract/33/7/073603/1076765/Super-resolution-and-denoising-of-fluid-flow-using). We’ll just focus on the very first case (summarized in Figure 3). Read the Methodology section (section II) and Known Boundary Condition (III.B.1), which is the case we’ll focus on. The goal is to reproduce Fig 3c (CNN).

You can download the data in these two files: [sr\_lfdata.npy](https://flow.byu.edu/me595r/schedule/sr_lfdata.npy), and [sr\_hfdata.npy](https://flow.byu.edu/me595r/schedule/sr_hfdata.npy) and load as follows:

```
<span>import</span> <span>numpy</span> <span>as</span> <span>np</span>
<span>import</span> <span>matplotlib.pyplot</span> <span>as</span> <span>plt</span>
<span>from</span> <span>matplotlib</span> <span>import</span> <span>cm</span>

<span># load low resolution data, which serves as input to our model
</span><span>lfdata</span> <span>=</span> <span>np</span><span>.</span><span>load</span><span>(</span><span>"</span><span>sr_lfdata.npy</span><span>"</span><span>)</span>
<span>lfx</span> <span>=</span> <span>lfdata</span><span>[</span><span>0</span><span>,</span> <span>:,</span> <span>:]</span>  <span># size 14 x 9  (height x width)
</span><span>lfy</span> <span>=</span> <span>lfdata</span><span>[</span><span>1</span><span>,</span> <span>:,</span> <span>:]</span>
<span>lfu</span> <span>=</span> <span>lfdata</span><span>[</span><span>4</span><span>,</span> <span>:,</span> <span>:]</span>
<span>lfv</span> <span>=</span> <span>lfdata</span><span>[</span><span>5</span><span>,</span> <span>:,</span> <span>:]</span>

<span># plot the low resolution data (like fig 3a except we are using MRI noise here rather than Gaussian noise so it will look a bit different)
</span><span>plt</span><span>.</span><span>figure</span><span>()</span>
<span>plt</span><span>.</span><span>pcolormesh</span><span>(</span><span>lfx</span><span>,</span> <span>lfy</span><span>,</span> <span>np</span><span>.</span><span>sqrt</span><span>(</span><span>lfu</span><span>**</span><span>2</span> <span>+</span> <span>lfv</span><span>**</span><span>2</span><span>),</span> <span>cmap</span><span>=</span><span>cm</span><span>.</span><span>coolwarm</span><span>,</span> <span>vmin</span><span>=</span><span>0.0</span><span>,</span> <span>vmax</span><span>=</span><span>1.0</span><span>)</span>
<span>plt</span><span>.</span><span>colorbar</span><span>()</span>

<span># load high resolution grids and mapping from low resolution to high resolution grid
</span><span>hfdata</span> <span>=</span> <span>np</span><span>.</span><span>load</span><span>(</span><span>"</span><span>sr_hfdata.npy</span><span>"</span><span>)</span>
<span>Jinv</span> <span>=</span> <span>hfdata</span><span>[</span><span>0</span><span>,</span> <span>:,</span> <span>:]</span>  <span># size 77 x 49 (height x width)
</span><span>dxdxi</span> <span>=</span> <span>hfdata</span><span>[</span><span>1</span><span>,</span> <span>:,</span> <span>:]</span>
<span>dxdeta</span> <span>=</span> <span>hfdata</span><span>[</span><span>2</span><span>,</span> <span>:,</span> <span>:]</span>
<span>dydxi</span> <span>=</span> <span>hfdata</span><span>[</span><span>3</span><span>,</span> <span>:,</span> <span>:]</span>
<span>dydeta</span> <span>=</span> <span>hfdata</span><span>[</span><span>4</span><span>,</span> <span>:,</span> <span>:]</span>
<span>hfx</span> <span>=</span> <span>hfdata</span><span>[</span><span>5</span><span>,</span> <span>:,</span> <span>:]</span>
<span>hfy</span> <span>=</span> <span>hfdata</span><span>[</span><span>6</span><span>,</span> <span>:,</span> <span>:]</span>

<span>ny</span><span>,</span> <span>nx</span> <span>=</span> <span>hfx</span><span>.</span><span>shape</span>  <span>#(77 x 49)
</span><span>h</span> <span>=</span> <span>0.01</span>  <span># grid spacing in high fidelity (needed for derivatives)
</span>
<span>plt</span><span>.</span><span>show</span><span>()</span>
```

I’ve also done (part of) the differentiation for you in the code below. The convolution filters shown in Appendix A aren’t actually used because they won’t work at the boundaries of the grid. We still use central finite differencing in the interior, but use one-sided finite differencing on the edges. The only difference from finite differencing you’ve done before is that we are using a larger stencil (5 pts for interior, 4 for edges). Note that the below code takes derivatives in the transformed Cartesian coordinates. You then need to apply the coordinate transformation using the formulas shown in eq 10a/b using the provided tensors above (Jinv which is written as 1/J in the text, dxdxi, dxdeta, dydxi, dydeta). Please note that the below assumes a grid given in height x width (not ‘x’ x ‘y’) just to match the ordering in [pytorch convolution layers](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).

```
<span># see https://en.wikipedia.org/wiki/Finite_difference_coefficient
# or https://web.media.mit.edu/~crtaylor/calculator.html
</span>
<span># f should be a tensor of size: nbatch x nchannels x height (y or eta) x width (x or xi)
# This is written in a general way if one had more data, but for this case there is only 1 data sample, and there are only a few channels it might be clearer to you to separate the channels out into separate variables, in which case the below could be simplified (i.e., you remove the first two dimensions from everything so that input is just height x width if you desire).
</span><span>def</span> <span>ddxi</span><span>(</span><span>f</span><span>,</span> <span>h</span><span>):</span>
    <span># 5-pt stencil
</span>    <span>dfdx_central</span> <span>=</span> <span>(</span><span>f</span><span>[:,</span> <span>:,</span> <span>:,</span> <span>0</span><span>:</span><span>-</span><span>4</span><span>]</span> <span>-</span> <span>8</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>:,</span> <span>1</span><span>:</span><span>-</span><span>3</span><span>]</span> <span>+</span> <span>8</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>:,</span> <span>3</span><span>:</span><span>-</span><span>1</span><span>]</span> <span>-</span> <span>f</span><span>[:,</span> <span>:,</span> <span>:,</span> <span>4</span><span>:])</span> <span>/</span> <span>(</span><span>12</span><span>*</span><span>h</span><span>)</span>
    <span># 1-sided 4pt stencil
</span>    <span>dfdx_left</span> <span>=</span> <span>(</span><span>-</span><span>11</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>:,</span> <span>0</span><span>:</span><span>2</span><span>]</span> <span>+</span> <span>18</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>:,</span> <span>1</span><span>:</span><span>3</span><span>]</span> <span>-</span><span>9</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>:,</span> <span>2</span><span>:</span><span>4</span><span>]</span> <span>+</span> <span>2</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>:,</span> <span>3</span><span>:</span><span>5</span><span>])</span> <span>/</span> <span>(</span><span>6</span><span>*</span><span>h</span><span>)</span>
    <span>dfdx_right</span> <span>=</span> <span>(</span><span>-</span><span>2</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>:,</span> <span>-</span><span>5</span><span>:</span><span>-</span><span>3</span><span>]</span> <span>+</span> <span>9</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>:,</span> <span>-</span><span>4</span><span>:</span><span>-</span><span>2</span><span>]</span> <span>-</span><span>18</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>:,</span> <span>-</span><span>3</span><span>:</span><span>-</span><span>1</span><span>]</span> <span>+</span> <span>11</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>:,</span> <span>-</span><span>2</span><span>:])</span> <span>/</span> <span>(</span><span>6</span><span>*</span><span>h</span><span>)</span>

    <span>return</span> <span>torch</span><span>.</span><span>cat</span><span>((</span><span>dfdx_left</span><span>,</span> <span>dfdx_central</span><span>,</span> <span>dfdx_right</span><span>),</span> <span>dim</span><span>=</span><span>3</span><span>)</span>

<span>def</span> <span>ddeta</span><span>(</span><span>f</span><span>,</span> <span>h</span><span>):</span>
    <span># 5-pt stencil
</span>    <span>dfdy_central</span> <span>=</span> <span>(</span><span>f</span><span>[:,</span> <span>:,</span> <span>0</span><span>:</span><span>-</span><span>4</span><span>,</span> <span>:]</span> <span>-</span> <span>8</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>1</span><span>:</span><span>-</span><span>3</span><span>,</span> <span>:]</span> <span>+</span> <span>8</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>3</span><span>:</span><span>-</span><span>1</span><span>,</span> <span>:]</span> <span>-</span> <span>f</span><span>[:,</span> <span>:,</span> <span>4</span><span>:,</span> <span>:])</span> <span>/</span> <span>(</span><span>12</span><span>*</span><span>h</span><span>)</span>
    <span># 1-sided 4pt stencil
</span>    <span>dfdy_bot</span> <span>=</span> <span>(</span><span>-</span><span>11</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>0</span><span>:</span><span>2</span><span>,</span> <span>:]</span> <span>+</span> <span>18</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>1</span><span>:</span><span>3</span><span>,</span> <span>:]</span> <span>-</span><span>9</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>2</span><span>:</span><span>4</span><span>,</span> <span>:]</span> <span>+</span> <span>2</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>3</span><span>:</span><span>5</span><span>,</span> <span>:])</span> <span>/</span> <span>(</span><span>6</span><span>*</span><span>h</span><span>)</span>
    <span>dfdy_top</span> <span>=</span> <span>(</span><span>-</span><span>2</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>-</span><span>5</span><span>:</span><span>-</span><span>3</span><span>,</span> <span>:]</span> <span>+</span> <span>9</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>-</span><span>4</span><span>:</span><span>-</span><span>2</span><span>,</span> <span>:]</span> <span>-</span><span>18</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>-</span><span>3</span><span>:</span><span>-</span><span>1</span><span>,</span> <span>:]</span> <span>+</span> <span>11</span><span>*</span><span>f</span><span>[:,</span> <span>:,</span> <span>-</span><span>2</span><span>:,</span> <span>:])</span> <span>/</span> <span>(</span><span>6</span><span>*</span><span>h</span><span>)</span>

    <span>return</span> <span>torch</span><span>.</span><span>cat</span><span>((</span><span>dfdy_bot</span><span>,</span> <span>dfdy_central</span><span>,</span> <span>dfdy_top</span><span>),</span> <span>dim</span><span>=</span><span>2</span><span>)</span>
```

Tips:

-   Although they used 3 separate networks that each produce 1 output channel (as shown in Fig 1), I just used 1 network with 3 output channels. I can be effective to learn shared features if output quantities are related. That would in general require adjusting the layer dimensions since we’ve reduced the available parameters with only 1 network. In this particular case I found that the default hyperparameters were still sufficient for the 1-network-3-channel-output version.
-   To better understand the two coordinate systems compare figure 2 and figure 9. Figure 2 is the actual physical space (x and y), and figure 9 is the same grid but mapped to a Cartesian space (ξ and η). We do most of our work in the Cartesian space where it is easier to compute finite difference derivatives, impose boundary conditions, and apply convolution filters. We then just need to map back to the regular space to impose our physics loss function (and plot). Mapping the velocities and pressures back is automatic as the grid connectivity is the same, and mapping derivatives back is what equations 10 a/b do (referred to earlier).
-   The paper only states some of the boundary conditions, and it does not explicitly state how they are imposed (because they are well known for the audience of this journal). So, I expand on them here. To impose boundary conditions, we simply set values for the velocities and pressures on the edges (bottom, left, top, right). The bottom edge (η = 0) is inflow with conditions: u\=0,v\=1,dp/dη\=0. So if u, v, p were tensors of size neta x nxi we would set: `u[0, :] = 0; v[0, :] = 1; p[0, :] = p[1, :]`. The latter forces the pressure gradient to be zero at the inlet (which just means it is at some unknown constant pressure). The left and right edges are walls with conditions: u\=0,v\=0,dp/dξ\=0 (the latter is a result from incompressible boundary layer theory). At the top (outlet) we set du/dη\=0,dv/dη\=0,p\=0 (the outlet pressure is unknown, but pressure is only defined relative to a reference point so we arbitrarily choose the outlet as a zero reference).
-   There are two ways you could do this: 1) predict results on a 77x49 grid then overwrite all the boundary values predicted by the network, or 2) predict results on a 75x47 grid then zero pad to create the boundaries (see [nn.ConstantPad2d](https://pytorch.org/docs/stable/generated/torch.nn.ConstantPad2d.html)). The size would now be 77x49 and you can then modify those added boundary values as needed (many of the boundary conditions are zero and so won’t require further modification).
-   In your neural network you first step should be to upsample the coarse input using bicubic interpolation like shown in the paper. That is your best estimate of the flow field on the high dimensional grid, then apply convolutional layers from there. Using your best estimate will allow the neural net portion to learn a simpler function. You can upscale using [nn.Upsample](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html) with mode=’bicubic\`. The size you upscale to depends on whether you add boundaries or modify boundaries as discussed in the prior bullet.
-   In the case we are analyzing, the only loss comes from the physics residuals shown in eq (4). I’ve expanded them in Cartesian coordinates for you below.
    
    dudx+dvdy\=0ududx+vdudy+1ρdpdx−ν(d2udx2+d2udy2)\=0udvdx+vdvdy+1ρdpdy−ν(d2vdx2+d2vdy2)\=0
    
-   To plot your final results you can use the following where u and v are your predicted flow fields from your network.
    
    ```
      <span>plt</span><span>.</span><span>figure</span><span>()</span>
      <span>plt</span><span>.</span><span>pcolormesh</span><span>(</span><span>hfx</span><span>,</span> <span>hfy</span><span>,</span> <span>np</span><span>.</span><span>sqrt</span><span>(</span><span>u</span><span>**</span><span>2</span> <span>+</span> <span>v</span><span>**</span><span>2</span><span>),</span> <span>cmap</span><span>=</span><span>cm</span><span>.</span><span>coolwarm</span><span>,</span> <span>vmin</span><span>=</span><span>0.0</span><span>,</span> <span>vmax</span><span>=</span><span>1.0</span><span>)</span>
      <span>plt</span><span>.</span><span>colorbar</span><span>()</span>
    ```
    
-   After finishing, it would be worth skimming through section III.C (at least III.C.1). We just learned to super resolve 1 data instance through an optimization process. But if we instead parameterized the inflow, and trained a model with multiple datasets, we could instead learn a model that could super resolve any MRI input without further training (just model evaluation).
