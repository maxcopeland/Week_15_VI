import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import seaborn as sb
import pandas as pd

#generate data from a polynomial with random gaussian noise
def curve_gen(x,order):
  coeffs = []
  #generate curve
  return_val = 0
  for exp in range(order+1):
    coeff = np.random.randint(low=-10,high=10)
    coeffs.append(coeff) #store for checking results against later
    return_val += (coeff*(x**exp))

  #add randomness
  for i in range(len(return_val)):
    return_val[i] += np.random.normal(scale=10**order)
  return return_val,coeffs


def design_mat(x, order):
  design = []
  for i in range(order+1):
    design.append(x**i)
  return np.array(design).T


def likelihood(beta,x,y):
  #order is one less than the length of beta
  des_mat = design_mat(x,len(beta)-1)
  y_pred = des_mat@beta
  # TODO: remove return statement
  return scipy.stats.norm(loc=y_pred, scale=10**order).logpdf(y)


def prior(beta):
  return scipy.stats.norm(loc=0,scale=10**order).pdf(beta)


def posterior(beta,x,y):
  like = likelihood(beta,x,y)
  pr = prior(beta)
  return np.sum(like)*np.ones(len(pr))+np.log(pr)


def mh(order,steps=10000):
  betas = [np.zeros(order+1)]
  for i in range(steps):
    if (i+1)%(steps/100) == 0:print("Step ",i+1,"/",steps,sep='')
    u = np.random.uniform(size=order+1)
    beta_prop = np.random.normal(size=order+1,loc=betas[-1],scale=0.1)
    while np.abs(beta_prop).max()>10: 
      beta_prop = np.random.normal(size=order+1,loc=betas[-1],scale=0.1)
    #since the proposal is symmetric, no need for q in acceptance
    acc = np.minimum(np.ones(order+1),np.exp(posterior(beta_prop,x,y)-posterior(betas[-1],x,y)))

    beta_selector = (u<acc).astype(int)
    beta_selectee = np.append([betas[-1]],[beta_prop],axis=0)

    new_beta = np.diagonal(beta_selectee.T[:,beta_selector])
    betas.append(new_beta)
  return betas