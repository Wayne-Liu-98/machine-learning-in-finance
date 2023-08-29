# %% [markdown]
# <a href="https://colab.research.google.com/gist/jteichma/a6e797d71b2af745fb7c39606e1f2290/lecture_3_fs2021.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Inverse Problems

# %% [markdown]
# In the sequel we shall study inverse problems, where training appears as a particular instance of an inverse problem. We shall take the point of view of optimization problems but we shall always emphasize a Bayesian perspective on it.

# %% [markdown]
# We start with a very generic point of view: consider a non-negative measurable function $f$ on a probability space $ (\Theta,\pi_0) $, then calculating the essential supremum of $ f $ corresponds to calculating its $\infty$-norm, for which a well-known formula exists
# $$
# \lim_{n \to \infty} {|| f ||}_n = {||f||}_{\infty} \, .
# $$
# Let us first assume that $ f $ is essentially bounded a number $M>0$, then $ f/ {||f||}_{\infty} $ is essentially bounded by $1$ and we can without restriction assume that $ {||f||}_{\infty}= 1 $ (with respect to the measure $ \pi_0$). Let $ 0 < \epsilon < 1 $ be fixed, then $ \pi_0[\{f > 1-\epsilon \}] > 0 $, hence
# $$
# {|| f ||}_n \geq (1-\epsilon) {(\pi_0[\{f > 1 - \epsilon \}])}^{1/n} \, ,
# $$
# whence for all $ n $ large enough greater than, say, $ 1 - 2 \epsilon $. Since $ \epsilon $ was arbitrary, this yields the proof.
# If $ f $ is unbounded, then $ \pi_0[\{f > k \}] > 0 $ for all $ k $, hence
# $$
# {|| f ||}_n \geq k {(\pi_0[\{f > k \}])}^{1/n} \, .
# $$
# So we can find for every $ k $ a number $ n_k $ such that the right hand side is large than $ k/2 $, and we conclude again.

# %% [markdown]
# We can interpret this equality in the case $ f = \exp(-L) $ for a finite measurable function $L$ and obtain immediately
# $$
# \operatorname{ess-inf}_{\theta \in \Theta} L(\theta) = - \lim_{n \to \infty} \frac{1}{n} \log \Big( \int \exp \big(- n L(\theta) \big) \pi_0(d\theta) \Big) \; \; \;  {\mathrm (EQ)} \, .
# $$
# Assume additionally that $ L $ is bounded from below. Then we expect the integrand to concentrate at arguments where values close to infima are taken, in other words it is worth investigating the probability measure
# $$
# \pi_n (d \theta) := \frac{\exp \big(-n L(\theta) \big) \pi_0(d\theta)}{\int \exp \big(- n L(\theta) \big) \pi_0(d\theta)} \, .
# $$
# Concentration around arguments where values close to infima are taken can be interpreted by proving that
# $$
# \lim_{n \to \infty} \pi_n [A] = 0
# $$
# for all measurable sets $ A $ such that there exists $ \epsilon > 0 $ and $ L(\theta) > \operatorname{ess-inf}_{\theta \in \Theta} L(\theta) + \epsilon $ for $ \theta \in A $. This in turn is equivalent to proving that
# $$
# \frac{\int_A \exp \big(- n L(\theta) \big) \pi_0(d\theta)}{\int \exp \big(- n L(\theta) \big) \pi_0(d\theta)} \rightarrow_{n \to \infty} 0 \, ,
# $$
# for such $A$, which follows immediately from equation $ \mathrm (EQ) $. Indeed, we obtain
# $$
# -\frac{1}{n} \log \Big(\int_A \exp \big(- n L(\theta) \big) \pi_0(d\theta) \Big) \geq \operatorname{ess-inf}_{\theta \in \Theta} L(\theta) + \epsilon
# $$
# for all $ n $, whence for all $ n $ large enough
# $$
# -\frac{1}{n} \log \frac{\int_A \exp \big(- n L(\theta) \big) \pi_0(d\theta)}{\int \exp \big(- n L(\theta) \big) \pi_0(d\theta)} \geq \epsilon/2
# $$
# holds true. This means that the limit of the term inside the logarithm has to vanish as $n$ goes to infinity.

# %% [markdown]
# Summing up this yields the following statement: for a measurable function $L$ essentially bounded from below the measure $ \pi_n $ concentrates at arguments where values close to the infimum are taken. This statement has a Bayesian interpretation. Consider $ \pi_0 $ as prior on $ \Theta $ and consider $ L $ a (negative) log-likelihood, then $ \pi_1 $ is the posterior calculated by Bayes formula (when do not have data in the moment), $ \pi_n $ appears as interation of this procedure and concentrates at arguments for the likelihood maximizes.
# 
# Let us introduce data in the next step, i.e. we consider the function $L$ as a measurable function of two variables $ z $, the data, and $ \theta $, the parameter, on a product space $ Z \times \Theta $, where $ Z $ only has a measurable structure. We shall write $L^z := L(z,.) $ and assume this function to be bounded from below and measurable. Then we can apply the previous considerations in a data-dependent way.
# 
# As a remark we add: for fixed $ \theta \in \Theta $ we can sometimes view $ z \mapsto \exp(-L^z(\theta)) $ as density of a random variable $Z$ with respect to some reference measure $ \nu $ on $Z$. In this case the fully Bayesian interpretation takes place, but actually we do not need this here.

# %% [markdown]
# A parameter-dependent optimization problem is an inverse problem: we are interested in describing a map
# $$
# z \mapsto \theta^*(z) \in \operatorname{arginf}_{\theta \in \Theta} L^z(\theta)
# $$
# for some $ z \in Z $. We give ourselves additionally topologies on $ Z $ and $ \Theta $ with corresponding sigma algebras being the Borel sigma algebras. We can require, following [Jacques Hadamard](https://en.wikipedia.org/wiki/Jacques_Hadamard), the following properties of such a map:
# 1. Existence for a large subset of $Z$.
# 2. Uniqueness for a large subset of $Z$.
# 3. Stability where it is uniquely defined, i.e. continuity as a map from $Z$ to $\Theta$.
# 
# Usually it is delicate to guarantee the three properties, which, however, are important if $z \in Z $ are considered data and $ \theta^*(z) \in \Theta $ a selected model (identifified by a parameter). Often those properties can be achieved if the problem is replaced by a regularized problem by adding a regularization term $ P: \Theta \to \mathbb{R}_{>0} $, i.e. we consider
# $$
# \operatorname{inf}_{\theta \in \Theta} L^z(\theta) + \lambda P(\theta) \, ,
# $$
# where $ \lambda > 0 $ is an additional parameter.
# 
# This by now classical theory has been developed in many directions, we shall compare it here with the above developed Bayesian perspective. Consider a reference measure $ \nu $ on $ \Theta $ and define a prior
# $$
# \pi_0 (d \theta) = \frac{\exp(- \lambda P(\theta)) \nu (d \theta)}{\int \exp(- \lambda P(\theta)) \nu (d \theta)} \, .
# $$
# Then the posterior
# $$
# \pi_n (d \theta) = \frac{\exp(- n L^z(\theta) - \lambda P(\theta)) \nu (d \theta)}{\int \exp(-n L^z(\theta) - \lambda P(\theta)) \nu (d \theta)}
# $$
# can be considered a generalized solution of the inverse problem (depending on parmaters $n$ and $\lambda$), which concentrates at arguments, where values of $L^z+\frac{\lambda}{n}P$ are close it their infimum. Notice that it is relatively easy to guarantee that $\pi_n$ depends continously on $ z $.
# 
# We do not go into detail which regularization functionals $P$ have to be chosen to guarantee Hadamard's properties, or under which assumptions the Bayesian solution satisfies them. We just want to point out that solving the optimization problem
# $$
# z \mapsto \theta^*(z) \in \operatorname{arginf}_{\theta \in \Theta} L^z(\theta)
# $$
# has to be either regularized or interpreted in a Bayesian way to provide useful solutions. We shall see in the sequel that the Bayesian interpretation is related to certain simulation alogrithms of gradient descent type.

# %% [markdown]
# # Training

# %% [markdown]
# The most enigmatic procedure in machine learning is training of neural networks, or, in general, parametric families of functions. This is of course an inverse problem, on which we have developed two perspectives above: the optimization perspective (with regularization) and the Bayesian perspective.
# 
# Essentially training is described as minimization of loss, i.e. for a given loss function $ L $ on a space of functions $ f $ parametrized by a set of parameters $ \theta \in \Theta $
# $$
# \operatorname{arginf}_{\theta \in \Theta} L(f^\theta)
# $$
# is searched.

# %% [markdown]
# Assume that $ \Theta $ is some open subset of points in $ \mathbb{R}^d $ and $ U : \Theta \to \mathbb{R} $, $ \theta \mapsto L(f^\theta) $ a sufficiently regular function with a unique minimum $ \theta^* \in \Theta $, then one can describe essentially one local and one global method to find the infimum:
# 
# 1. If $ U $ is strictly convex and $ C^2 $ in a neighborhood of the unique minimizer $ \theta^* $, then
# $$
# d \theta_t = - \nabla_\Theta U(\theta_t) dt
# $$
# converges to $ \theta^* $ as $ t \to \infty $. For any $ t \geq 0 $ it holds that
# $$
# d U(\theta_t) = - {|| \nabla U(\theta_t) ||}^2 dt \, ,
# $$
# i.e. the value of $ U $ is strictly increasing along the path $ t \mapsto \theta_t $. Together with the fact that $ U $ is strictly convex we obtain a convergence of $ || \theta_ t - \theta^* || \leq C \exp(- \lambda_{\text{min}} t ) $ as $ t \to \infty $, where $ \lambda_{\mathrm min} $ is the minimum of the smallest eigenvalue of the Hessian of $ U $ on $ \Theta $. This holds remarkably for any starting point $ \theta_0 \in \Theta $ and is the basis of all sorts of gradient descent algorithms.

# %% [markdown]
# 2. A far reaching generalization is given by the following consideration: consider $ U $ on $ \Theta $ having a unique minimizer $ \theta^* \in \Theta $, then the probability measure given by the density with respect to Lebesgue measure on $ \mathbb{R}^d$
# $$
# p_\epsilon := \frac{1}{Z_\epsilon} \exp \big( -\frac{U}{\epsilon} \big)
# $$
# tends in law to $ \delta_{\theta^*} $ as $ \epsilon \to 0 $. The denominator $ Z_\epsilon $ is just the integral $ \int_{\Theta} \exp(-U(\theta)/\epsilon) d \lambda(\theta)  $ and the above statement nothing else than the fact that the described density function concentrates at $ \theta^* $. If one manages to sample from the measure $ p_\epsilon d \lambda $, then one can approximate empirically $ \theta^* $.
# 
# The measure $ p_\epsilon d \lambda $ is the invariant measure of the stochastic differential equation
# $$
# d \theta_t = - \frac{1}{2} \nabla U(\theta_t) dt + \sqrt{\epsilon} dW_t \, ,
# $$
# which is just checked by the following equality
# $$
# \int_{\Theta} \big ( - \frac{1}{2} \nabla U(\theta) \nabla f(\theta) + \frac{\epsilon}{2} \Delta f(\theta) \big) p_\epsilon (\theta) d \lambda (\theta) = 0
# $$
# for all test functions $ f $.
# 
# One method, which generalizes this thought in a time-dependent way, is to sample from a measure concentrating at $\theta^*$ is to simulate from a stochastic differential equation (for $ \Theta = \mathbb{R}^N $) of the type
# $$
# d \theta_t = - \nabla U(\theta_t) dt + \alpha(t) dW_t
# $$
# where $ W $ is an $ N $-dimensional Brownian motion and the non-negative quantity, called cooling schedule, $ \alpha(t) = O(\frac{1}{\log(t)}) $ as $ t \to \infty $. For appropriate constants we obtain that $ \theta_t $ converges in law to $ \delta_{\theta^*} $ as $ t \to \infty $. This procedure is called simulated annealing and is the fundament for global mimimization algorithms of the type 'steepest descent plus noise'.

# %%
import numpy as np

# Trajectories of d \theta = - \nabla U (\theta) dt + \alpha d W

N=2000 # time disrectization
theta0=1 # initial value of theta
T=100 # maturity
alpha = 0.1 # volatility in Black Scholes

R=10**5 # number of Trajectories

def U(theta):
  return theta**2*(theta-2)**2
  #return theta**2*(theta-2)**2*np.sin(theta)**2

def gradU(theta):
  return 2*theta*(theta-2)**2 + theta**2*2*(theta-2)
  #return (2*theta*(theta-2)**2 + theta**2*2*(theta-2))*np.sin(theta)**2+2*np.sin(theta)*np.cos(theta)*theta**2*(theta-2)**2

theta = np.zeros((N,R))
theta[0,]=theta0*np.ones((1,R))


for j in range(N-1):
    increment = np.random.normal(0,np.sqrt(alpha)*np.sqrt(T)/np.sqrt(N),(1,R))
    theta[j+1,:] =theta[j,:]+ increment - 0.5*gradU(theta[j,:])*(T/N)

# %%
import matplotlib.pyplot as plt

for i in range(10):
   plt.plot(theta[:,i])
plt.show()

# %%
a=4
x=np.linspace(-a,a,100)
plt.hist(theta[N-1,:],density=True,bins=100)
plt.plot(x,np.exp(-U(x)/alpha)/(2*a/100*sum(np.exp(-U(x)/alpha))))
plt.show()

# %% [markdown]
# In machine learning applications, however, an algorithm, which traces back to work of Robbins-Monro (Kiefer-Wolfowitz respectively) in the fifties of the last century, is applied, the so called [stochastic approximation algorithm](https://en.wikipedia.org/wiki/Stochastic_approximation).
# 
# The stochastic gradient descent algorithm essentially says for a function of expectation type $ U(\theta) = E \big[ V(\theta) \big] $
# $$
# \theta_{n+1} = \theta_n - \gamma_n \nabla V(\theta_n,\omega_n)
# $$
# for independently chosen samples $ \omega_n $ converges in law to $ \theta^* $. Notice first that all our examples, in particular the ones from mathematical finance, are of expectation type, where the samples $ \omega $ are usually seen as elements from the training data set.

# %% [markdown]
# There are several proofs of this statement, but I want to focus one particular aspect which connects stochastic approximation with simulated annealing.
# 
# By the central limit theorem and appropriate sub-sampling one can understand
# $$
# \nabla V(\theta,\omega) = \nabla U(\theta) + \text{ 'Gaussian noise with a certain covariance structure' } \Sigma(\theta)
# $$
# where $ \Sigma(\theta) $ is essentially given by
# $$
# \operatorname{cov}(\nabla V(\theta)) \, .
# $$
# Whence one can understand stochastic gradient descent as simulated annealing with a space dependent covariance structure.
# 
# Such simulated annealing algorithms exist but are more sophisticated in nature, in particular one needs to work with geometric structures on $ \Theta $.

# %%



