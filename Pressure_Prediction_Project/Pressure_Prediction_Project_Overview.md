# Pressure Prediction Project

Jacob Child

March 31st, 2025

**Project Statement:** [This](https://pubs.aip.org/aip/pof/article/33/7/073603/1076765/Super-resolution-and-denoising-of-fluid-flow-using) paper used a physics informed convolutional neural network to upscale a low resolution CFD simulation. They were essentially able to get this using a bicubic interpolation to upscale, and then the Navier-Stokes momentum and continuity equations to train (the "physics informed" portion) the net. No data comparisons were used to train, just physics. The pressure field that comes out of the paper is incorrect, and the paper says "The current framework cannot accurately recover the hidden pressure field because of the implicit coupling of pressure and velocity (i.e., the pressure appears as a source term in the momentum equations). The governing equation needs to be reformulated to strongly couple the pressure and velocity predictions based on, e.g., elliptic equation of pressure [81](https://www.taylorfrancis.com/books/mono/10.1201/9781482234213/numerical-heat-transfer-fluid-flow-suhas-patankar) and Rhie–Chow interpolation.[75](https://arc.aiaa.org/doi/10.2514/3.8284)" 

My goal then is to recreate the pressure field by changing the governing equations in the loss function to include the elliptic equation of pressure, Rich-Chow interpolation, or the [Pressure Poisson equations](https://en.wikipedia.org/wiki/Poisson%27s_equation#:~:text=adaptive%20octree.-,Fluid%20dynamics,-%5Bedit%5D).

Follow up goals include: 

- Extending this to work on a simple 2D channel flow cfd simulation, without the focus on upscaling, simply the PINN portion to learn pressure from velocity

- Using Symbolic Regression or Kolmogorov Arnold to come up with a possible equation to predict a pressure field from velocity fields

**Notes:** Code will be in this repo, and research/internet tabs will be under the DeepLearning workspace on my personal laptop. Important updates and notes will be placed here.

## Approach:

1. Literature Review: Look at the two relevant references, what are the take aways? Break down and understand? How would I implement as a governing equation in code?
   
   1. Elliptic Equation of Pressure (Pressure Poisson Equations)
      
      1. I requested access to the originally referenced book. For now I have started  an AI thread ([what is the elliptic equation of pressure](https://www.perplexity.ai/search/what-is-the-elliptic-equation-yBst_IWXS1.X9f22k101pw)) that should be helpful. If I want to dive into theory it looks like here ([What are Hyperbolic, parabolic, and elliptic equations in CFD?](https://cfd.university/learn/10-key-concepts-everyone-must-understand-in-cfd/what-are-hyperbolic-parabolic-and-elliptic-equations-in-cfd/)) is probably a good place to go. It does look like "Elliptic Equation of Pressure" is likely just a reference to the pressure poisson equations 
      
      2. [CFD Intro - Essential skills for reproducible research computing](https://barbagroup.github.io/essential_skills_RRC/numba/4/) has a great discussion of the Pressure Poisson equations, and later even has an iterative solver code. I could use the iterative solver somehow in the code, but instead maybe I have it just compare against the actual data, then the training won't take so long?
      
      3. Pressure Poisson Equation: $\Delta p = -\rho \nabla \cdot (V \cdot \Delta V)$ [Pressure Poisson equations](https://en.wikipedia.org/wiki/Poisson%27s_equation#:~:text=adaptive%20octree.-,Fluid%20dynamics,-%5Bedit%5D) or from Barba git or MIT lecture notes etc, TODO: look later. 
   
   2. Rhie-Chow Interpolation
   
   3. My understand of the equations and possible implementation

2. Coding: 
   
   1. Copy over the code from CNNForSuperResCFDwphysics_HW8, output the current predicted pressure plot and actual pressure plot, also create a pressure loss term/plot that shows how far off the current prediction is.
   
   2. Begin to attempt to implement newer governing equations from above
   
   3. Hyper-parameter tuning
   
   4. Adapt to work on 2D channel flow, train on different Reynolds Numbers?
   
   5. Symbolic Regression? 

3. Play around with hyper-parameter tuning until it seems like I plateau on improvements, then look at Rhie-Chow interpolation and see if I can better couple pressure and velocity in my governing equations. Try training on that. Once I can consistently produce a reasonable pressure field for homework 8, I have 3 options to extend the project:
   
   1. Use the same equations and methodology on a new 2D channel flow and see if I can predict that, train at different Reynolds numbers etc
   2. Use the same equations amd methodology on Nathan Welker's PIV data and see if I can predict reasonable pressure fields -> I need to see if he has pressure fields I can compare to
   3. Stick with homework 8, but now apply symbolic regression (or Kolmogorov Arnold?) to see if I can figure out what equations are used to predict the pressure field. This would probably require a standalone pressure prediction network instead of one that is also learning upscaling of u and v too.

## Project Presentation

---

Your final deliverable consists of two parts:

1. Prepare and record a presentation summarizing your project.
2. Watch 8 other project presentations and provide peer reviews. We won’t have lecture for the last two class periods to make time for this.

You will record a **10-minute max presentation** (15-minutes max for a group of 2) and make it available to the class via a link on a private online spreadsheet. The presentation will be peer-reviewed based on the following rubric (I will incorporate weights when averaging all the peer reviews, you will just grade based on the numeric categories):

- **Motivation (15%)**
  
  - **5: Clearly defines the problem, provides context, and explains significance.**
  - 4: Mostly clear but missing some background details.

- **Methods (25%)**
  
  - **5: Uses appropriate methods and best practices. Explains approach with technical rigor (e.g., model architecture, loss functions, training details).**
  - 4: Mostly solid work, but some minor methodological gaps.

- **Results (40%)**
  
  - **5: Presents results with clear visualizations, proper metrics, and insightful interpretation. Clearly meets or exceeds scope.**
  - 4: Good results, but missing some explanation or comparisons.

- **Presentation (20%)**
  
  - **5: Presentation is well-structured, logically organized, and easy to follow.**
  - 4: Mostly clear, but some minor issues.
