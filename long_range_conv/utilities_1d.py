import tensorflow as tf
import numpy as np 
from numba import jit 

@tf.function
def trainStepList(model, optimizer, loss,
                    inputs, neighList, outputsE, outputsF, 
                    weightE, weightF):

  with tf.GradientTape() as tape:
    # we use the model the predict the outcome
    predE, predF = model(inputs, neighList, training=True)

    # fidelity loss usin mse
    lossE = loss(predE, outputsE)
    lossF = loss(predF, outputsF)/outputsF.shape[-1]


    total_loss = weightE*lossE + weightF*lossF

  # compute the gradients of the total loss with respect to the trainable variables
  gradients = tape.gradient(total_loss, model.trainable_variables)
  # update the parameters of the network
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return total_loss

class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  initializer=tf.initializers.GlorotNormal(),
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])
    self.bias = self.add_weight("bias",
                                initializer=tf.initializers.zeros(),    
                                shape=[self.num_outputs,])
  @tf.function
  def call(self, input):
    return tf.matmul(input, self.kernel) + self.bias


class pyramidLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs, actfn = tf.nn.relu):
    super(pyramidLayer, self).__init__()
    self.num_outputs = num_outputs
    self.actfn = actfn

  def build(self, input_shape):
    self.kernel = []
    self.bias = []
    self.kernel.append(self.add_weight("kernel",
                       initializer=tf.initializers.GlorotNormal(),
                       shape=[int(input_shape[-1]),
                              self.num_outputs[0]]))
    self.bias.append(self.add_weight("bias",
                       initializer=tf.initializers.GlorotNormal(),
                       shape=[self.num_outputs[0],]))

    for n, (l,k) in enumerate(zip(self.num_outputs[0:-1],  
                                  self.num_outputs[1:])) :

      self.kernel.append(self.add_weight("kernel"+str(n),
                         shape=[l, k]))
      self.bias.append(self.add_weight("bias"+str(n),
                         shape=[k,]))


  @tf.function
  def call(self, input):
    x = self.actfn(tf.matmul(input, self.kernel[0]) + self.bias[0])
    for k, (ker, b) in enumerate(zip(self.kernel[1:], self.bias[1:])):
      if self.num_outputs[k] == self.num_outputs[k+1]:
        x += self.actfn(tf.matmul(x, ker) + b)  ###ResNet
      else :
        x = self.actfn(tf.matmul(x, ker) + b)
    return x



@tf.function
def genDistInvPerNlistVec(Rin, neighList, L, 
                          av = tf.constant([0.0, 0.0], dtype = tf.float32),
                          std =  tf.constant([1.0, 1.0], dtype = tf.float32)):

    # This function follows the same trick 
    # function to generate the generalized coordinates for periodic data

    Nsamples = Rin.shape[0]
    maxNumNeighs = neighList.shape[-1]
    # define an indicator
    mask = neighList > -1
    
    RinRep  = tf.tile(tf.expand_dims(Rin, -1),[1 ,1,maxNumNeighs] )
    RinGather = tf.gather(Rin, neighList, batch_dims = 1, axis = 1)

    # compute the periodic distance
    R_Diff = RinGather - RinRep
    R_Diff = R_Diff - L*tf.round(R_Diff/L)

    bnorm = (tf.abs(R_Diff) - av[1])/std[1]
    binv = (tf.math.reciprocal(tf.abs(R_Diff)) - av[0])/std[0], 



    zeroDummy = tf.zeros_like(bnorm)
    # add zero when the actual number of neighbors are less than maxNumNeigh
    bnorm_safe = tf.where(mask, bnorm, zeroDummy)
    binv_safe = tf.where(mask, binv, zeroDummy)

    
    R_total = tf.concat([tf.reshape(binv_safe, (-1,1)), 
                              tf.reshape(bnorm_safe, (-1,1))], axis = 1)
    return R_total


@jit(nopython=True)
def computInterListOpt(Rinnumpy, L,  radious, maxNumNeighs):
  # function to compute the interaction lists 
  Nsamples, Npoints = Rinnumpy.shape



  # compute the relative coordinates
  DistNumpy = np.abs(Rinnumpy.reshape(Nsamples,Npoints,1) \

              - Rinnumpy.reshape(Nsamples,1,Npoints))


  # periodic the distance
  # work around some quirks of numba with the np.round function
  out = np.zeros_like(DistNumpy)
  np.round(DistNumpy/L, 0, out)
  DistNumpy = DistNumpy - L*out



  # add the padding and loop over the indices 
  Idx = np.zeros((Nsamples, Npoints, maxNumNeighs), dtype=np.int32) -1 
  for ii in range(0,Nsamples):
    for jj in range(0, Npoints):
      ll = 0 
      for kk in range(0, Npoints):
        if jj!= kk and np.abs(DistNumpy[ii,jj,kk]) < radious:
          # checking that we are not going over the max number of
          # neighboors, if so we break the loop
          if ll >= maxNumNeighs:
            print("Number of neighboors is larger than the max number allowed")
            break
          Idx[ii,jj,ll] = kk
          ll += 1 
  return Idx