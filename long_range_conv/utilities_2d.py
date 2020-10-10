

import tensorflow as tf
import numpy as np 
from numba import jit 
# compute the generated coordinates
@tf.function
def genDistInvPerNlistVec2D(Rin, neighList, L, 
                            av = tf.constant([0.0, 0.0], dtype = tf.float32),
                            std =  tf.constant([1.0, 1.0], dtype = tf.float32)):
    # This function follows the same trick 
    # function to generate the generalized coordinates for periodic data
    # neighList is a (Nsample, Npoints, maxNeigh)
    Nsamples = Rin.shape[0]
    maxNumNeighs = neighList.shape[-1]
    # define an indicator
    mask = neighList > -1


    RinRepX  = tf.tile(tf.expand_dims(Rin[:,:,0], -1), [1 ,1, maxNumNeighs])
    RinGatherX = tf.gather(Rin[:,:,0], neighList, batch_dims = 1, axis = 1)

    RinRepY  = tf.tile(tf.expand_dims(Rin[:,:,1], -1), [1 ,1, maxNumNeighs])
    RinGatherY = tf.gather(Rin[:,:,1], neighList, batch_dims = 1, axis = 1)

    # compute the periodic distance
    R_DiffX = RinGatherX - RinRepX
    R_DiffX = R_DiffX - L*tf.round(R_DiffX/L)
    R_DiffY = RinGatherY - RinRepY
    R_DiffY = R_DiffY - L*tf.round(R_DiffY/L)
    norm = tf.sqrt(tf.square(R_DiffX) + tf.square(R_DiffY))
    

    
    binv = tf.math.reciprocal(norm) 
    bx = tf.math.multiply(R_DiffX, binv)
    by = tf.math.multiply(R_DiffY, binv)

    zeroDummy = tf.zeros_like(norm)
    # add zero when the actual number of neighbors are less than maxNumNeigh 
    binv_safe = tf.where(mask, (binv- av[0])/std[0], zeroDummy)
    bx_safe = tf.where(mask, bx, zeroDummy)
    by_safe = tf.where(mask, by, zeroDummy)
    
    R_total = tf.concat([tf.reshape(binv_safe, (-1,1)), 
                         tf.reshape(bx_safe, (-1,1)), 
                         tf.reshape(by_safe, (-1,1))], axis = 1)

    return R_total


# the function for training
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

# compute the neighbor list
@jit(nopython=True)
def computInterList2DOpt(Rinnumpy, L,  radious, maxNumNeighs):
  # function to compute the interaction lists 
  Nsamples, Npoints, dimension = Rinnumpy.shape
  # compute the relative coordinates
  DistNumpy = Rinnumpy.reshape(Nsamples,Npoints,1, dimension) \
              - Rinnumpy.reshape(Nsamples,1, Npoints,dimension)

  # periodic the distance
  # work around some quirks of numba with the np.round function
  out = np.zeros_like(DistNumpy)
  np.round(DistNumpy/L, 0, out)
  DistNumpy = DistNumpy - L*out

  # compute the distance
  DistNumpy = np.sqrt(np.sum(np.square(DistNumpy), axis = -1))

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

# dense layer
class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs, 
                     initializer = tf.initializers.GlorotNormal()):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs
    self.initializer = initializer

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  initializer=self.initializer,
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])
    self.bias = self.add_weight("bias",
                                initializer=tf.initializers.zeros(),    
                                shape=[self.num_outputs,])
  @tf.function
  def call(self, input):
    return tf.matmul(input, self.kernel) + self.bias


class pyramidLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs, 
                     actfn = tf.nn.relu,
                     initializer=tf.initializers.GlorotNormal() ):
    super(pyramidLayer, self).__init__()
    self.num_outputs = num_outputs
    self.actfn = actfn
    self.initializer = initializer

  def build(self, input_shape):
    self.kernel = []
    self.bias = []
    self.kernel.append(self.add_weight("kernel",
                       initializer=self.initializer,
                       shape=[int(input_shape[-1]),
                              self.num_outputs[0]]))
    self.bias.append(self.add_weight("bias",
                       initializer=tf.zeros_initializer,
                       shape=[self.num_outputs[0],]))

    for n, (l,k) in enumerate(zip(self.num_outputs[0:-1], \
                                  self.num_outputs[1:])) :

      self.kernel.append(self.add_weight("kernel"+str(n),
                         shape=[l, k]))
      self.bias.append(self.add_weight("bias"+str(n),
                         shape=[k,]))

  @tf.function
  def call(self, input):
    # first application
    x = self.actfn(tf.matmul(input, self.kernel[0]) + self.bias[0])
    
    # run the loop
    for k, (ker, b) in enumerate(zip(self.kernel[1:], self.bias[1:])):
      
      # if input equals to output use a shortcut connection
      if self.num_outputs[k] == self.num_outputs[k+1]:
        x += self.actfn(tf.matmul(x, ker) + b)
      else :
        x = self.actfn(tf.matmul(x, ker) + b)
    return x