#Optimization of random directions for Pytorch
#Has functions for converting the model weights to and from a 1D numpy array,
#and an (example) objective or error function for optimrandomdir
#Gerald Schuller, August 2019

#import optimrandomdir_parallel as optimrandomdir #with parallel cpu processing, good for complex error functions
#import optimrandomdir_parallel_linesearch as optimrandomdir
import GLD_fast #with no parallel cpu processing, good for simple error functions

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
   #print("weightsarray.shape=", weightsarray.shape)
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
   
def optimizer(model, loss_fn, X, Y, iterations=100, startingscale=1.0, endscale=0.0):
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
   #optimrandomdir.iterations=iterations
   #optimrandomdir.startingscale=startingscale
   #optimrandomdir.endscale=endscale
   #Here now the actual optimization of random directions:
   weightsmin=GLD_fast.gldfast(torcherrorfunction, weightsarray, args=(model, loss_fn, X, Y), iterations=iterations, startingscale=startingscale)
   
   #update model with new weights:
   weightsarray2model(weightsmin, model)
   
