import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os.path
from os import path
import h5py
import sys
import json
import csv
import time

from data_gen_1d import genDataYukawaPerMixed,gen_data_Per_Mixed
from utilities_1d import genDistInvPerNlistVec, trainStepList, computInterListOpt
from utilities_1d import MyDenseLayer, pyramidLayer
from nufft_layers_1d import NUFFTLayerMultiChannelInitMixed



import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
nameScript = sys.argv[0].split('/')[-1]


# we are going to give all the arguments using a Json file
nameJson = sys.argv[1]
print("=================================================")
print("Executing " + nameScript + " following " + nameJson, flush = True)
print("=================================================")

# opening Json file 
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




dataFile = dataFolder + "data_1D"+ DataType + \
                        "_Ncells_" + str(Ncells) + \
                        "_Np_" + str(Np) + \
                        "_mu1_" + str(mu1) + \
                        "_mu2_" + str(mu2) + \
                        "_weight1_" + str(weight1) + \
                        "_weight2_" + str(weight2) + \
                        "_minDelta_%.4f"%(minDelta) + \
                        "_Nsamples_" + str(Nsamples) + ".h5"

checkFolder  = "/"
checkFile = checkFolder + "checkpoint_1D_" + \
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
  # TODO: encapsulate all this in a function
  print("Data file does not exist, we create a new one")

  if DataType == "Periodic":

    print("Creating %s data"%(DataType))
    pointsArray, \
    potentialArray, \
    forcesArray  = gen_data_Per_Mixed(Ncells, Np, 
                                mu1, mu2, Nsamples, 
                                minDelta, Lcell, weight1, weight2)

  elif DataType == "YukawaPeriodic":

    print("Creating %s data"%(DataType))
    pointsArray, \
    potentialArray, \
    forcesArray  = genDataYukawaPerMixed(Ncells, Np, 
                                      mu1, mu2, Nsamples, 
                                      minDelta, Lcell,weight1,weight2)
    
    
    
  hf = h5py.File(dataFile, 'w') 
  
  hf.create_dataset('points', data=pointsArray)   
  hf.create_dataset('potential', data=potentialArray) 
  hf.create_dataset('forces', data=forcesArray)
  
  hf.close()

# extracting the data
hf = h5py.File(dataFile, 'r')

pointsArray = hf['points'][:]
forcesArray = hf['forces'][:]
potentialArray = hf['potential'][:]


Rinput = tf.Variable(pointsArray, name="input", dtype = tf.float32)
Rin = Rinput[0:100,:]
Rinnumpy = Rin.numpy()


Idx = computInterListOpt(Rinnumpy, L,  radious, maxNumNeighs)
# compute the neighbor list. shape:(Nsamples, Npoints and MaxNumneighs)
neighList = tf.Variable(Idx)
Npoints = Np*Ncells

genCoordinates = genDistInvPerNlistVec(Rin, neighList, L)
# compute the generated coordinates
filter = tf.cast(tf.reduce_sum(tf.abs(genCoordinates), axis = -1)>0, tf.int32)
numNonZero =  tf.reduce_sum(filter, axis = 0).numpy()
numTotal = genCoordinates.shape[0] 

av = tf.reduce_sum(genCoordinates, 
                    axis = 0, 
                    keepdims =True).numpy()[0]/numNonZero
std = np.sqrt((tf.reduce_sum(tf.square(genCoordinates - av), 
                             axis = 0, 
                             keepdims=True).numpy()[0] - av**2*(numTotal-numNonZero)) /numNonZero)


print("mean of the inputs are %.8f and %.8f"%(av[0], av[1]))
print("std of the inputs are %.8f and %.8f"%(std[0], std[1]))



class DeepMDsimpleForces(tf.keras.Model):
  """Combines the encoder and decoder into an end-to-end model for training."""

  def __init__(self,
               Npoints,
               L, 
               maxNumNeighs = 4,
               descripDim = [2, 4, 8, 16, 32],
               fittingDim = [16, 8, 4, 2, 1],
               mu1 = 1.0,
               mu2 = 1.0,
               av = [0.0, 0.0],
               std = [1.0, 1.0],
               NpointsFourier = 500, 
               fftChannels = 4,
               xLims = [0.0, 10.0], 
               name='deepMDsimpleForces',
               **kwargs):
    super(DeepMDsimpleForces, self).__init__(name=name, **kwargs)

    print("xLims = %f, %f"%(xLims[0], xLims[1]) )

    # this should be done on the fly, for now we will keep it here
    self.L = L
    self.Npoints = Npoints
    self.maxNumNeighs = maxNumNeighs
    # we normalize the inputs (should help for the training)
    self.av = av
    self.std = std
    self.mu1 = mu1
    self.mu2 = mu2
    self.descripDim = descripDim
    self.fittingDim = fittingDim
    self.descriptorDim = descripDim[-1]
    self.NpointsFourier = NpointsFourier
    self.fftChannels    = fftChannels 
    # we may need to use the tanh here
    self.layerPyramid   = pyramidLayer(descripDim, 
                                       actfn = tf.nn.tanh)
    self.layerPyramidInv  = pyramidLayer(descripDim, 
                                       actfn = tf.nn.tanh)

    self.NUFFTLayerMultiChannelInitMixed = NUFFTLayerMultiChannelInitMixed(fftChannels, \
      NpointsFourier, xLims, mu1,mu2)
    self.layerPyramidLongRange  = pyramidLayer(descripDim, 
                                       actfn = tf.nn.relu)
    self.fittingNetwork = pyramidLayer(fittingDim, 
                                       actfn = tf.nn.tanh)
    self.linfitNet      = MyDenseLayer(1)    

  @tf.function
  def call(self, inputs, neighList):

    with tf.GradientTape() as tape:
      # we watch the inputs 

      tape.watch(inputs)
      # (Nsamples, Npoints)

      genCoordinates = genDistInvPerNlistVec(inputs, neighList, self.L,self.av,self.std)

      L1   = self.layerPyramid(genCoordinates[:,1:])*genCoordinates[:,0:1]
      # (Nsamples*Npoints*maxNumNeighs, descriptorDim)
      L2   = self.layerPyramidInv(genCoordinates[:,0:1])*genCoordinates[:,0:1]
      # (Nsamples*Npoints*maxNumNeighs, descriptorDim)
      LL = tf.concat([L1, L2], axis = 1)
      # (Nsamples*Npoints*maxNumNeighs, 2*descriptorDim)
      Dtemp = tf.reshape(LL, (-1, self.maxNumNeighs, 2*self.descriptorDim ))
      # (Nsamples*Npoints, maxNumNeighs, 2*descriptorDim)
      D = tf.reduce_sum(Dtemp, axis = 1)
      # (Nsamples*Npoints, 2*descriptorDim)
      
      longRangewCoord = self.NUFFTLayerMultiChannelInitMixed(inputs)
      longRangewCoord2 = tf.reshape(longRangewCoord, (-1, self.fftChannels))
      # (Nsamples*Ncells*Np, 1)
      L3   = self.layerPyramidLongRange(longRangewCoord2)
      # (Nsamples*Ncells*Np, descriptorDim)

      DLongRange = tf.concat([D, L3], axis = 1)

      F2 = self.fittingNetwork(DLongRange)
      F = self.linfitNet(F2)

      Energy = tf.reduce_sum(tf.reshape(F, (-1, self.Npoints)),
                              keepdims = True, axis = 1)

    Forces = -tape.gradient(Energy, inputs)

    return Energy, Forces


## Defining the model
model = DeepMDsimpleForces(Npoints, L, maxNumNeighs,
                           descriptorNet, fittingNet, 
                           mu1,mu2,
                           av, std, 
                           NpointsFourier, fftChannels, xLims)


E,F = model(Rin,neighList)
model.summary()


errorlist = []
losslist = []
# Create checkpointing directory if necessary
if not os.path.exists(checkFolder):
    os.mkdir(checkFolder)
    print("Directory " , checkFolder ,  " Created ")
else:    
    print("Directory " , checkFolder ,  " already exists :)")

# sometimes we need to load an older saved model
if loadFile: 
  print("Loading the weights the model contained in %s"%(loadFile), flush = True)
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

if DataType == "Periodic":

    pointsTest, \
    potentialTest, \
    forcesTest  = gen_data_Per_Mixed(Ncells, Np, 
                                mu1, mu2, 100, 
                                minDelta, Lcell, weight1, weight2)

if DataType == "YukawaPeriodic":

    pointsTest, \
    potentialTest, \
    forcesTest  = genDataYukawaPerMixed(Ncells, Np, 
                                      mu1, mu2, 100, 
                                      minDelta, Lcell,weight1,weight2)
    
IdxTest = computInterListOpt(pointsTest, L,  radious, maxNumNeighs)
neighListTest = tf.Variable(IdxTest)

for cycle, (epochs, batchSizeL) in enumerate(zip(Nepochs, batchSizeArray)):

  print('++++++++++++++++++++++++++++++', flush = True) 
  print('Start of cycle %d' % (cycle,))
  print('Total number of epochs in this cycle: %d'%(epochs,))
  print('Batch size in this cycle: %d'%(batchSizeL,))

  weightE = 0.0
  weightF = 1.0

  x_train = (pointsArray, potentialArray, forcesArray)

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
      Idx = computInterListOpt(Rinnumpy, L,  radious, maxNumNeighs)
      neighList = tf.Variable(Idx)
        
      loss = trainStepList(model, optimizer, mse_loss_fn,
                        x_batch_train[0], neighList,
                        x_batch_train[1], 
                        x_batch_train[2],
                        weightE, weightF)
      loss_metric(loss)
  
      if step % 100 == 0:
        print('step %s: mean loss = %s' % (step, str(loss_metric.result().numpy())))
    

    potPred, forcePred = model(pointsTest, neighListTest)
    err = tf.sqrt(tf.reduce_sum(tf.square(forcePred - forcesTest)))/tf.sqrt(tf.reduce_sum(tf.square(forcePred)))
    print("Relative Error in the forces is " +str(err.numpy()))
    end = time.time()
    print('time elapsed %.4f'%(end - start))
    
    # mean loss saved in the metric
    errorlist.append(err.numpy())        
    with open('error'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist)
        
        
    meanLossStr = str(loss_metric.result().numpy())
    # learning rate using the decay 
    lrStr = str(optimizer._decayed_lr('float32').numpy())
    print('epoch %s: mean loss = %s  learning rate = %s'%(epoch,
                                                          meanLossStr,
                                                          lrStr))
    
    losslist.append(loss_metric.result().numpy())
    with open('loss'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(losslist)
        
    print("saving the weights")
    model.save_weights(checkFile+".h5")




