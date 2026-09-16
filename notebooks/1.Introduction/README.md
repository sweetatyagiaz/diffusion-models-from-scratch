# 1: An Introduction to Diffusion Models

Welcome to Unit 1 of the Hugging Face Diffusion Models Course! In this unit, you will learn the basics of how diffusion 
models work and how to create your own using the 🤗 Diffusers library.

## Start this Unit 🚀

Here are the steps for this unit:

- Read through the introductory material below as well as any of the additional resources that sound interesting.
- Check out the _**Introduction to Diffusers**_  notebook below to put theory into practice with the 🤗 Diffusers library.
- Train and share your own diffusion model using the notebook or the linked training script.
- (Optional) Dive deeper with the _**Diffusion Models from Scratch**_ notebook if you're interested in seeing a minimal from-scratch implementation and exploring the different design decisions involved.

 
## What Are Diffusion Models?

Diffusion models are a relatively recent addition to a group of algorithms known as 'generative models'. The goal of generative modeling is to learn to **generate** data, such as images or audio, given a number of training examples. A good generative model will create a **diverse** set of outputs that resemble the training data without being exact copies. How do diffusion models achieve this? Let's focus on the image generation case for illustrative purposes.

<p align="center">
    <img src="https://user-images.githubusercontent.com/10695622/174349667-04e9e485-793b-429a-affe-096e8199ad5b.png" width="800"/>
    <br>
    <em> Figure from DDPM paper (https://arxiv.org/abs/2006.11239). </em>
<p>

The secret to diffusion models' success is the iterative nature of the diffusion process. Generation begins with random noise, but this is gradually refined over a number of steps until an output image emerges. At each step, the model estimates how we could go from the current input to a completely denoised version. However, since we only make a small change at every step, any errors in this estimate at the early stages (where predicting the final output is extremely difficult) can be corrected in later updates. 

Training the model is relatively straightforward compared to some other types of generative model. We repeatedly
1) Load in some images from the training data
2) Add noise, in different amounts. Remember, we want the model to do a good job estimating how to 'fix' (denoise) both extremely noisy images and images that are close to perfect.
3) Feed the noisy versions of the inputs into the model
4) Evaluate how well the model does at denoising these inputs
5) Use this information to update the model weights

To generate new images with a trained model, we begin with a completely random input and repeatedly feed it through the model, updating it each time by a small amount based on the model prediction. As we'll see, there are a number of sampling methods that try to streamline this process so that we can generate good images with as few steps as possible.

We will show each of these steps in detail in the hands-on notebooks here in unit 1. In unit 2, we will look at how this process can be modified to add additional control over the model outputs through extra conditioning (such as a class label) or with techniques such as guidance. And units 3 and 4 will explore an extremely powerful diffusion model called Stable Diffusion, which can generate images given text descriptions.  


