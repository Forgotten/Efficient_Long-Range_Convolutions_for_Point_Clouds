

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os.path
from os import path
import h5py
import sys
import json
import time


from data_gen_3d import genDataPer3DMixed
from nufft_layers_3d import NUFFTLayerMultiChannel3D
from utilities_3d import genDistInvPerNlistVec3D, trainStepList, computInterList2DOpt
from utilities_3d import MyDenseLayer, pyramidLayer



import os



os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
nameScript = sys.argv[0].split('/')[-1]



# we are going to give all the arguments using a Json file
nameJson = sys.argv[1]
print("=================================================")
print("Executing " + nameScript + " following " + nameJson, flush = True)
print("=================================================")





# we open Json file 
jsonFile = open(nameJson) 
data = json.load(jsonFile)   



# loading the input data from the json file

# here we assume the data is generated within some cells. The number of cells in
# each dimension is "Ncells". "Np" shows the number of particles in per cell. 
# For simiplicity, we assume they are generated randomly uniformly.   
Ncells = data["Ncells"]                  # number of cells
Np = data["Np"]                          # number of particles per cell
Nsamples = data["Nsamples"]              # number of samples 
Lcell = data["lengthCell"]               # length of each cell
mu1 = data["mu1"]                        # first characteristic interaction length 
mu2 = data["mu2"]                        # second characteristic interaction length
weight1 = data["weight1"]                # weight of first component
weight2 = data["weight2"]                # weight of second component      
minDelta = data["minDelta"]              # the minimal distance between two particles
descriptorNet = data["descriptorNet"]    # size of descriptor network       
fittingNet = data["fittingNet"]          # size of fitting network
epochsPerStair = data["epochsPerStair"]  # decay step of learning rate   
learningRate = data["learningRate"]      # initial learning rate
decayRate = data["decayRate"]            # decay rate of learning rate
dataFolder = data["dataFolder"]          # data folder
loadFile = data["loadFile"]              # load file
Nepochs = data["numberEpoch"]            # epoch
batchSizeArray = data["batchSize"]       # batchsize      
maxNumNeighs = data["maxNumNeighbors"]   # maximal number of neighbors
radious = data["radious"]                # short-range interaction radious 
NpointsFourier = data["NpointsFourier"]  # the number of Fourier modes 
fftChannels = data["fftChannels"]        # the number of FFT channels 
DataType = data["DataType"]              # data type
L = Lcell*Ncells
xLims = [0.0, L]

dataFile = dataFolder + "data_3D_"+ DataType + \
                        "_Ncells_" + str(Ncells) + \
                        "_Np_" + str(Np) + \
                        "_mu1_" + str(mu1) + \
                        "_mu2_" + str(mu2) + \
                        "_weight1_" + str(weight1) + \
                        "_weight2_" + str(weight2) + \
                        "_minDelta_%.4f"%(minDelta) + \
                        "_Nsamples_" + str(Nsamples) + ".h5"



checkFolder  = "/"
checkFile = checkFolder + "checkpoint_3D_" + \
                          "potential_"+ DataType + \
                          "_Ncells_" + str(Ncells) + \
                          "_Np_" + str(Np) + \
                          "_mu1_" + str(mu1) + \
                          "_mu2_" + str(mu2) + \
                          "_weight1_" + str(weight1) + \
                          "_weight2_" + str(weight2) + \
                          "_minDelta_%.4f"%(minDelta) + \
                          "_Nsamples_" + str(Nsamples)

print("Using data in %s"%(dataFile))

# if the file doesn't exist we create it
if not path.exists(dataFile):
  print("Data file does not exist, we create a new one")
  
  if DataType == "Periodic":
    print("Creating %s data"%(DataType))
    pointsArray, \
    potentialArray, \
    forcesArray  = genDataPer3DMixed(Ncells, Np, mu1, mu2, Nsamples, minDelta, 
                                     Lcell, weight1, weight2)
  else: 
    print("only periodic exponential data supported")


  hf = h5py.File(dataFile, 'w') 
  
  hf.create_dataset('points', data=pointsArray)   
  hf.create_dataset('potential', data=potentialArray) 
  hf.create_dataset('forces', data=forcesArray)

  hf.close()



# we extract the data
hf = h5py.File(dataFile, 'r')

pointsArray = hf['points'][:]
forcesArray = hf['forces'][:]
potentialArray = hf['potential'][:]


Rinput = tf.Variable(pointsArray, name="input", dtype = tf.float32)

# we only consider the first 100 
Rin = Rinput[:100,:,:]
Rinnumpy = Rin.numpy()


Idx = computInterList2DOpt(Rinnumpy, L,  radious, maxNumNeighs)
neigh_list = tf.Variable(Idx)
Npoints = Np*Ncells**3

# we compute the generated coordinate in order to obtain av and std
gen_coordinates = genDistInvPerNlistVec3D(Rin, neigh_list, L)
filter = tf.cast(tf.reduce_sum(tf.abs(gen_coordinates), axis = -1)>0, tf.int32)
numNonZero =  tf.reduce_sum(filter, axis = 0).numpy()
numTotal = gen_coordinates.shape[0]  


av = tf.reduce_sum(gen_coordinates, 
                    axis = 0, 
                    keepdims =True).numpy()[0]/numNonZero
std = np.sqrt((tf.reduce_sum(tf.square(gen_coordinates - av), 
                               axis = 0, 
                               keepdims=True).numpy()[0] 
                - av**2*(numTotal-numNonZero)) /numNonZero)

#print("mean of the inputs are %.8f and %.8f"%(av[0], av[1]))
#
#print("std of the inputs are %.8f and %.8f"%(std[0], std[1]))
#




class DeepMDsimpleEnergyNUFFT(tf.keras.Model):

  """Combines the encoder and decoder into an end-to-end model for training."""
  def __init__(self,
               Npoints,
               L, 
               maxNumNeighs = 4,
               descripDim = [2, 4, 8, 16, 32],
               fittingDim = [16, 8, 4, 2, 1],
               mu1 = 10.0,
               mu2 = 1.0,
               av = [0.0, 0.0],
               std = [1.0, 1.0],
               NpointsFourier = 500, 
               fftChannels = 4,
               xLims = [0.0, 10.0],
               name='deepMDsimpleEnergyNUFFT',
               **kwargs):
    super(DeepMDsimpleEnergyNUFFT, self).__init__(name=name, **kwargs)
    self.L = L
    self.Npoints = Npoints
    # maximum number of neighbors
    self.maxNumNeighs = maxNumNeighs
    # we normalize the inputs (should help for the training)
    self.av = av
    self.std = std
    self.descripDim = descripDim
    self.fittingDim = fittingDim
    self.descriptorDim = descripDim[-1]
    self.fftChannels = fftChannels
    self.mu1 = mu1
    self.mu2 = mu2
    # we may need to use the tanh here
    self.layerPyramid = pyramidLayer(descripDim, actfn = tf.nn.tanh)    
    self.layerPyramidDir = pyramidLayer(descripDim, actfn = tf.nn.tanh)    
    self.NUFFTLayer = NUFFTLayerMultiChannel3D(fftChannels,NpointsFourier, 
                                               xLims, mu1, mu2)
    self.layerPyramidLongRange = pyramidLayer(descripDim, actfn = tf.nn.relu)
    self.fittingNetwork = pyramidLayer(fittingDim, actfn = tf.nn.tanh)
    self.linfitNet = MyDenseLayer(1)    

  @tf.function
  def call(self, inputs, neighList):

    with tf.GradientTape() as tape:
      # we watch the inputs 
      tape.watch(inputs)
      # shape (Nsamples, Npoints)
      gen_coordinates = genDistInvPerNlistVec3D(inputs, 
                                          neighList, self.L, 
                                          self.av, self.std)
      # (Nsamples*Npoints*maxNumNeighs, 3)
      L1 = self.layerPyramid(gen_coordinates[:,:1])*gen_coordinates[:,:1]
      # (Nsamples*Npoints*maxNumNeighs, descriptorDim)
      L2 = self.layerPyramidDir(gen_coordinates[:,1:])*gen_coordinates[:,:1]
      LL = tf.concat([L1, L2], axis = 1)
      # (Nsamples*Npoints*maxNumNeighs, descriptorDim)
      Dtemp = tf.reshape(LL, (-1, self.maxNumNeighs, 2*self.descriptorDim))
      # (Nsamples*Ncells*Np, maxNumNeighs, descriptorDim)
      D = tf.reduce_sum(Dtemp, axis = 1)
      # (Nsamples*Npoints, descriptorDim*descriptorDim)
      long_range_coord = self.NUFFTLayer(inputs)
      long_range_coord2 = tf.reshape(long_range_coord, (-1, self.fftChannels))
      # (Nsamples*Ncells*Np, 1)
      L3   = self.layerPyramidLongRange(long_range_coord2)
      DLongRange = tf.concat([D, L3], axis = 1)
      F2 = self.fittingNetwork(DLongRange)
      F = self.linfitNet(F2)
      Energy = tf.reduce_sum(tf.reshape(F, (-1, self.Npoints)),
                             keepdims = True, axis = 1)
    Forces = -tape.gradient(Energy, inputs)
    return Energy, Forces

avTF = tf.constant(av, dtype=tf.float32)
stdTF = tf.constant(std, dtype=tf.float32)
## Define the model
model = DeepMDsimpleEnergyNUFFT(Npoints, L, maxNumNeighs,
                                descriptorNet, fittingNet, 
                                mu1,mu2,
                                avTF, stdTF,
                                NpointsFourier, fftChannels, 
                                xLims)
# quick run of the model to check that it is correct.
Rin2 = Rinput[:2,:,:] 
Rinnumpy = Rin2.numpy()
Idx = computInterList2DOpt(Rinnumpy, L,  radious, maxNumNeighs)
neigh_list2 = tf.Variable(Idx)

E, F = model(Rin2, neigh_list2)
model.summary()



# Create checkpointing directory if necessary
if not os.path.exists(checkFolder):
    os.mkdir(checkFolder)
    print("Directory " , checkFolder ,  " Created ")
else:    
    print("Directory " , checkFolder ,  " already exists :)")



# sometimes we need to load an older saved model
if loadFile: 
  print("Loading the weights the model contained in %s"(loadFile), flush = True)
  model.load_weights(loadFile)


print("Training cycles in number of epochs")
print(Nepochs)
print("Training batch sizes for each cycle")
print(batchSizeArray)



### optimization parameters ##
mse_loss_fn = tf.keras.losses.MeanSquaredError()



initialLearningRate = learningRate
lrSchedule = tf.keras.optimizers.schedules.ExponentialDecay(
             initialLearningRate,
             decay_steps=(Nsamples//batchSizeArray[0])*epochsPerStair,
             decay_rate=decayRate,
             staircase=True)



optimizer = tf.keras.optimizers.Adam(learning_rate=lrSchedule)

loss_metric = tf.keras.metrics.Mean()



##################train-test split #############################



points_train = pointsArray
forces_train = forcesArray
potential_train = potentialArray

points_test, \
potential_test, \
forces_test  = genDataPer3DMixed(Ncells, Np, mu1, mu2, 100, minDelta, Lcell,
                                     weight1, weight2)

Idx_test = computInterList2DOpt(points_test, L, radious, maxNumNeighs)

# (Nsamples, Npoints and MaxNumneighs)
neigh_list_test = tf.Variable(Idx_test)

rin_test = tf.Variable(points_test, dtype=tf.float32)
forces_test = tf.Variable(forces_test, dtype=tf.float32)



###################training loop ##################################

for cycle, (epochs, batchSizeL) in enumerate(zip(Nepochs, batchSizeArray)):



  print('++++++++++++++++++++++++++++++', flush = True) 
  print('Start of cycle %d' % (cycle,))
  print('Total number of epochs in this cycle: %d'%(epochs,))
  print('Batch size in this cycle: %d'%(batchSizeL,))

  weightE = 0.0
  weightF = 1.0

  x_train = (points_train, potential_train, forces_train)

  train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
  train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batchSizeL)

  # Iterate over epochs.
  for epoch in range(epochs):
    start = time.time()
    print('============================', flush = True) 
    print('Start of epoch %d' % (epoch,))

    loss_metric.reset_states()

    # Iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(train_dataset):

      Rinnumpy = x_batch_train[0].numpy()
      Idx = computInterList2DOpt(Rinnumpy, L,  radious, maxNumNeighs)
      neighList = tf.Variable(Idx)

      loss = trainStepList(model, optimizer, mse_loss_fn,
                           x_batch_train[0], neighList,
                           x_batch_train[1], x_batch_train[2], weightE, weightF)
      loss_metric(loss)


      if step % 100 == 0:
        print('step %s: mean loss = %s' % (step, str(loss_metric.result().numpy())))

    Idx = computInterList2DOpt(points_train[:10,:,:], L,  radious, maxNumNeighs)
    neighList = tf.Variable(Idx)

    pottrain, forcetrain = model(points_train[:10,:,:],neighList)
    errtrain = tf.sqrt(tf.reduce_sum(tf.square(forcetrain - forces_train[:10,:,:])))\
               /tf.sqrt(tf.reduce_sum(tf.square(forcetrain)))
    print("Relative Error in the trained forces is " +str(errtrain.numpy()))

    end = time.time()
    print('time elapsed %.4f'%(end - start))
    # mean loss saved in the metric
    meanLossStr = str(loss_metric.result().numpy())
    # learning rate using the decay 
    lrStr = str(optimizer._decayed_lr('float32').numpy())
    print('epoch %s: mean loss = %s  learning rate = %s'%(epoch, meanLossStr,
                                                          lrStr))

  print("saving the weights")
  model.save_weights(checkFile+".h5")

  # compute the test error using batches
  x_test = (points_test, potential_test, forces_test)
  test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
  test_dataset = test_dataset.shuffle(buffer_size=10000).batch(batchSizeL)

  err = tf.Variable(0.0, dtype=tf.float32)
  norm = tf.Variable(0.0, dtype=tf.float32)

  for step, x_batch_test in enumerate(test_dataset):

    Rinnumpy = x_batch_test[0].numpy()
    Idx = computInterList2DOpt(Rinnumpy, L,  radious, maxNumNeighs)
    neigh_list_test = tf.Variable(Idx)

    pot_pred, force_pred = model(x_batch_test[0], neigh_list_test)
    err.assign_add(tf.reduce_sum(tf.square(force_pred - x_batch_test[2])))
    norm.assign_add(tf.reduce_sum(tf.square(x_batch_test[2])))

  errTotal = tf.sqrt(err)/tf.sqrt(norm)
  print("Relative Error in the forces is " +str(errTotal.numpy()))
