#Optimization pypop7 for Pytorch
#https://pypi.org/project/pypop7/
#pip3 install pypop7
#needs a virtual environment from Conda:
#Conda list environments:
#conda env list
#conda create --name pypop7
#conda activate pypop7
#Has functions for converting the model weights to and from a 1D numpy array,
#and an (example) objective or error function for the algorithms in pypop7
#Gerald Schuller, July 2023

#choose optimizers around line 110.
#import xnes 
#All also suitable for high dimensions:
from pypop7.optimizers.es.lmmaes import LMMAES #path in pypop7, replace "/" by "." and remove ".py"
from pypop7.optimizers.nes.snes import SNES #Separable Natural Evolution Strategy, 2011
from pypop7.optimizers.es.es import ES
from pypop7.optimizers.es.mmes import MMES #from 2021
from pypop7.optimizers.rs.bes import BES #from 2022
from pypop7.optimizers.nes.r1nes import R1NES # 

import torch
import numpy as np

def model2weightsarray(model):
   #turns the weights of model to a 1D numpy array, 
   #returns the numpy array
   #Turn weights into 1D numpy array
   #global model
   weightsarray=[]
   for param_tensor in model.state_dict():
      #print(param_tensor, "\t", model.state_dict()[param_tensor].numpy().shape)
      weights=model.cpu().state_dict()[param_tensor] #reading weights from model, convert to numpy
      #print("weights=", weights)
      weights=weights.numpy()
      weightsfl=np.reshape(weights,(-1)) #flattened weights tensor
      #print("weightsfl=", weightsfl)
      weightsarray=np.hstack((weightsarray,weightsfl)) #make a long 1D vector out of it
   #print("weightsarray=", weightsarray)
   print("model2weightsarray: weightsarray.shape=", weightsarray.shape)
   return weightsarray
      
def weightsarray2model(weightsarray, model):
   #turns a weights array to a pytorch model
   #global model
   #print("weightsarray=", weightsarray)
   pointer=0
   for param_tensor in model.state_dict():
      sh=model.cpu().state_dict()[param_tensor].numpy().shape #shape of current layer
      numnweights= np.prod(sh) #number of weights to read
      weightsfl=weightsarray[pointer+np.arange(numnweights)] #flattened weights  
      weightsre=np.reshape(weightsfl,sh) #reconstructed tensor
      #print("weightsre=", weightsre)
      weightsretorch=torch.from_numpy(weightsre)
      #print("weightsretorch=", weightsretorch)
      model.state_dict()[param_tensor].data.copy_(weightsretorch) #write pytorch structure back to model
      pointer+=numnweights
   return   
      
def torcherrorfunction(weightsarray, args):
   #implements an error function for optimization of random direction
   #weights needs to be a numpy tensor
   #Return 1D numpy weightsarray back into pytorch structure:
   #global model
   #global loss_fn
   #print("weightsarray torcherrorfunction=", weightsarray)
   model=args[0]
   loss_fn=args[1]
   X=args[2]
   Y=args[3]
   weightsarray2model(weightsarray, model)
   Ypred=model(X)#.cpu()
   loss=loss_fn(Ypred, Y).item()#.cpu() #.detach().numpy()
   #print("loss errfn:", loss)
   return loss
   
def optimizer(model, loss_fn, X, Y, iterations=100, optimizer_choice='mmes'):
   #This is the interface function for Pytorch betworks
   #It replaces the optimizer for loop
   #Arguments: 
   #model: object for the neural network whose weights are to be optimized
   #loss_fn: the pytorch loss function
   #X: Training set
   #Y: Target
   #iterations: the number of iteration for the method of random directions
   #startingscale: starting standard deviation for the random steps in X
   #endscale: The standard deviation is slowly reduced of the iterations to reach endscale
   #The best values for the start- and end-scale depend on the network, some trials are useful.
   
   weightsarray=model2weightsarray(model)
   print("optimizer weightsarray.shape=", weightsarray.shape, "weightsarray=",weightsarray)

   #Here now the actual optimization of xnes:
   #weightsmin=optimrandomdir.optimrandomdir(torcherrorfunction, weightsarray, args=(model, loss_fn, X, Y), iterations=iterations, startingscale=startingscale, endscale=endscale)
   #Natural Evolution Strategie:
   args=(model, loss_fn, X, Y)
   lambdafunc=lambda x: torcherrorfunction(x, args) 
   ndim_problem = len(weightsarray)
   print('ndim_problem=', ndim_problem)
   problem = {'fitness_function': lambdafunc,  # cost function
        'ndim_problem': ndim_problem,  # dimension
        'lower_boundary': -5*np.ones((ndim_problem,)),  # search boundary
        'upper_boundary': 5*np.ones((ndim_problem,))}
   options = {'fitness_threshold': 1e-10,  # terminate when the best-so-far fitness is lower than this threshold
        'max_runtime': iterations,  # seconds (terminate when the actual runtime exceeds it)
        'seed_rng': 0,  # seed of random number generation (which must be explicitly set for repeatability)
        'x': weightsarray,  # initial mean of search (mutation/sampling) distribution
        'sigma': 0.3,  # initial global step-size of search distribution (not necessarily optimal)
        'verbose': 500}
        
        

   #print("pypop7 optimizer_choice=", optimizer_choice)
   print("Pypop7 optimizer: "+ optimizer_choice)
   #List: mmes, r1nes, lmmaes, snes, bes

   if optimizer_choice=='mmes':
      optimizerfunc = MMES
   if optimizer_choice=='r1nes':
      optimizerfunc = R1NES
   if optimizer_choice=='lmmaes':
      optimizerfunc = LMMAES
   if optimizer_choice=='snes':
      optimizerfunc = SNES
   if optimizer_choice=='bes':
      optimizerfunc = BES
      
   results = optimizerfunc(problem, options).optimize()

   print("results['best_so_far_y']=", results['best_so_far_y'])
   
   weightsmin=results['best_so_far_x']
   #print("xmin=", xmin)
   
   #update model with new weights:
   weightsarray2model(weightsmin, model)
   
