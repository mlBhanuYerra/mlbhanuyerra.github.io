---
layout: post
title: An Introduction to Objective Functiosn Used in Machine Learning
subtitle: Whats the difference between Maximum Likelihood & Cross-Entropy?
tags: [Theory, Objective Functions, Loss Functions, Cross-Entropy, Maximum Likelihood Estimation, Statistical Framework]
comments: true
---

Objective Functions Used in Machine Learning
Developing machine learning applications can be viewed as consisting of three components [1]: a representation of data, an evaluation function, and an optimization method to estimate the parameter of the machine learning model. This is true for supervised, unsupervised and reinforcement learning.
The primary focus of this article is the evaluation component (objective functions or loss functions) of the ML tasks, and is divided into the following sections:
Objective functions for regression
Objective functions for classification
Objective functions used in reinforcement learning and generative adversarial networks (GANs) are not covered in this article. And the objective functions from various popular ML methods and papers, like Triplet Loss Function, Neural Style Transfer Loss Function, YOLO Loss Function, will be presented as a follow-up article.
1. Objective Functions for Regression
A select objective functions commonly used in linear regression models are presented in this section. For reference, see a list of available loss functions from Keras Library at https://keras.io/losses/
The problem entails establishing a linear relationship between x, an independent variable from a D-dimensional space ℝᴰ, and y, the dependent variable ∈ ℝ¹ as shown below

Where, 𝜀ᵢ is random error due to (i) wrong choice of a linear relationship, (ii) omission of othe relevant independent variables, (iii) meaurement error, and (iv) instrumental variables — measured variables are proxies of “real” variables. The objective of linear regression is to estimate the ws given a random sample of the population.

and the difference between the true dependent variable yᵢ and the model estimated dependent variable ŷᵢ is referred as residual error: eᵢ = yᵢ -ŷᵢ. The parameters ws are estimated by minimizing an objective function, which takes various functional forms as presented below.
Mean Squared Error:
Mostly commonly used in linear regression models. The mean squared error, sometimes called residual sum of squares is:

Mean Absolute Error:
Similar to MSE, mean absolute error is as shown below:

Mean Absolute Percentage Error:
Similar to mean absolute error, mean absolute percentage error is as shown below:

Mean Squared Logarithmic Error:
Similar to MSE, but calculated on a logarithmic scale. Target variable should be non-negative, but can be equal to zero. If the target is never zero, the addition of 1 in the logarithm can be dropped.

Regularization:
In cases where the model complexity or limited data points leads to over-fitting of regression models, a common technique known as regularization is employed. Regularization adds a penalty to the objective function to control the magnitude of the parameters.

Where R(w) represent regularization function on the parameters w and 𝜆 represents a hyper-parameter that needs to be tuned as part of the model calibration.
Ridge Regression: Adds a L2-norm of the parameters to the objective function. Also known as weight decay, as the learning algorithm leads the weights to decay towards zero.

Lasso Regression: Adds an L-1 norm of the parameters to the objective function. Lasso regression leads to a sparse model as it drives some of the parameters to zero.

For more on regularization, see: https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a
2. Objective Functions for Classification
Unlike regression that handles continuous dependent variable, classification problem handles dependent variables that are classes and is focused on estimating the probability of an observation belonging to each class. Dependent variables in classification problem are discrete and mutually exclusive groups or classes. To formally state the classification problem, let y be the class indicator from a group of K mutually exclusive classes and is represented as a one-hot vector. Let x be the independent variables for that data point. That means,

The idea is to establish a probability function that predicts the class of y given x and the parameters w of the probability function. Representing this probability as a K-dimensional Bernoulli, we get:

where,

Note the above condition in itself does not restrict the functional form of the probability function p(yk = 1|x, w).
There are two approaches to solving the above classification problem: Maximum Likelihood Estimation and Cross Entropy. Both leads to the same objective function (and solution) but there is considerable amount of confusion as the ML community adopts the “cross entropy” explanation, while the folks with statistics background see this as maximum likelihood estimation. Interestingly, linear regression also has two approaches that lead to the same solution (thankfully a closed form solution): Ordinary Least Squares (uses Mean Squared Error, see above) and Maximum Likelihood Estimation.
Maximum Likelihood Estimation: The K-dimensional Bernoulli shown above for a single observation (or data point) can be extended to the whole dataset/sample. Say there are N independent and identically distributed (iid) data points (Y, X), a likelihood function for the sample can be written as:

Note that this likelihood function represent the probability of coming across the observed sample set for various setting of the parameter w. In the frequentist paradigm of statistics, the parameters w are estimated such that they yield highest probability of observing the sample set. Now, maximizing the likelihood function in setting parameters w is same as minimizing the monotonically decreasing negative logarithm of the likelihood function. Negative log likelihood of this probability is:

Cross entropy: This approach of classification problem has its roots in information theory. In particular, the concepts of self-information, entropy and, of course, cross entropy. Self-information can be viewed as the degree of surprise or amount of information we learn from observing a random event and is based on the probability of that event. We learn less from a highly probable event compared to what we learn from observing a highly improbable event. This means self-information is a monotonically decreasing over probability space. And we learn twice as much of information from observing two such independent events as compared to just one. These requirements leads to the following functional form relating self-information h(x) to probability of an event x, p(x):

To extend this concept to the classification problem, lets state the classification problem as approximating the true probabilities y=(y₁,y₂…yk)ᵀ using the approximated probabilities ŷ = (𝜇₁,𝜇₂…𝜇𝑘)ᵀ. This means the self-information approximated by the classification for observing y belonging to class k is -log(𝜇k). Since we know the true probabilities as y=(y₁,y₂…yk)ᵀ, the expected self-information of observing y can be written as:

This is known as cross-entropy and can also be viewed as how closely the true probability distribution y is represented by the approximate probability distribution ŷ. If we apply this to all of our N observations in the data set we get total cross-entropy as:

See that the cross-entropy objective function looks exactly the same as the objective functions we formulated using MLE.
References:
A few useful things to know about machine learning, Pedro Domingos, Department of Computer Science and Engineering, UW, https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf
https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a
https://www.quora.com/What-are-the-differences-between-maximum-likelihood-and-cross-entropy-as-a-loss-function
https://medium.com/swlh/shannon-entropy-in-the-context-of-machine-learning-and-ai-24aee2709e32
