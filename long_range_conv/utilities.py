import tensorflow as tf
import numpy as np 
from numba import jit 

## Layers ##

class DenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(DenseLayer, self).__init__()
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


## Training functions ##

@tf.function
def train_step(model, optimizer, loss,
                    inputs, neigh_list, output_E, output_f, 
                    weight_e, weight_f):

  with tf.GradientTape() as tape:
    # we use the model the predict the outcome
    predE, predF = model(inputs, neigh_list, training=True)

    # fidelity loss usin mse
    lossE = loss(predE, output_E)
    lossF = loss(predF, output_f)/output_f.shape[-1]


    total_loss = weight_e*lossE + weight_f*lossF

  # compute the gradients of the total loss with respect to the trainable variables
  gradients = tape.gradient(total_loss, model.trainable_variables)
  # update the parameters of the network
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return total_loss  
  
## Interaction Lists ##  

@jit(nopython=True)
def comput_inter_list(r_in, L,  radious, max_num_neighs):
  # function to compute the interaction lists 
  n_samples, n_points, dimension = r_in.shape

  # compute the relative coordinates
  dist = np.abs( r_in.reshape(n_samples, n_points, dimension) \
                -r_in.reshape(n_samples, dimension, n_points))

  # periodic the distance
  # work around some quirks of numba with the np.round function
  out = np.zeros_like(dist)
  np.round(dist/L, 0, out)
  dist = dist - L*out

  # compute the distance
  dist = np.sqrt(np.sum(np.square(dist), axis = -1))

  # add the padding 
  Idx = np.zeros((n_samples, n_points, max_num_neighs), dtype=np.int32)-1 

  # and loop over the indices 
  for ii in range(0,n_samples):
    for jj in range(0, n_points):
      ll = 0 
      for kk in range(0, n_points):
        if jj!= kk and np.abs(dist[ii,jj,kk]) < radious:
          # checking that we are not going over the max number of
          # neighboors, if so we break the loop
          if ll >= max_num_neighs:
            pr_int("Number of neighboors is larger than the max number allowed")
            break
          Idx[ii,jj,ll] = kk
          ll += 1 

  return Idx

## Generalized coordinates ##

@tf.function
def gen_coor_1d(r_in, neigh_list, L, 
                av = tf.constant([0.0, 0.0], dtype = tf.float32),
                std =  tf.constant([1.0, 1.0], dtype = tf.float32)):

    # This function follows the same trick 
    # function to generate the generalized coordinates for periodic data

    n_samples = r_in.shape[0]
    max_num_neighs = neigh_list.shape[-1]
    # define an indicator
    mask = neigh_list > -1
    
    r_in_rep  = tf.tile(tf.expand_dims(r_in, -1),[1 ,1, max_num_neighs] )
    r_in_gath = tf.gather(r_in, neigh_list, batch_dims = 1, axis = 1)

    # compute the periodic distance
    r_diff_ = r_in_gath - r_in_rep
    r_diff_ = r_diff_ - L*tf.round(r_diff_/L)

    bnorm = (tf.abs(r_diff_) - av[1])/std[1]
    binv = (tf.math.reciprocal(tf.abs(r_diff_)) - av[0])/std[0], 

    zero_dummy = tf.zeros_like(bnorm)
    # add zero when the actual number of neighbors are less than maxNumNeigh
    bnorm_safe = tf.where(mask, bnorm, zero_dummy)
    binv_safe = tf.where(mask, binv, zero_dummy)

    r_total = tf.concat([tf.reshape(binv_safe, (-1,1)), 
                         tf.reshape(bnorm_safe, (-1,1))], axis = 1)
    return r_total


@tf.function
def gen_coor_2d(r_in, neigh_list, L, 
                av = tf.constant([0.0, 0.0], dtype = tf.float32),
                std =  tf.constant([1.0, 1.0], dtype = tf.float32)):
    # This function follows the same trick 
    # function to generate the generalized coordinates for periodic data
    # neigh_list is a (Nsample, Npoints, maxNeigh)

    n_samples = r_in.shape[0]
    max_num_neighs = neigh_list.shape[-1]
    # define an indicator
    mask = neigh_list > -1

    r_in_rep_X  = tf.tile(tf.expand_dims(r_in[:,:,0], -1), [1 ,1, max_num_neighs])
    r_in_gath_X = tf.gather(r_in[:,:,0], neigh_list, batch_dims = 1, axis = 1)

    r_in_rep_Y  = tf.tile(tf.expand_dims(r_in[:,:,1], -1), [1 ,1, max_num_neighs])
    r_in_gath_Y = tf.gather(r_in[:,:,1], neigh_list, batch_dims = 1, axis = 1)

    # compute the periodic distance
    r_diff_X = r_in_gath_X - r_in_rep_X
    r_diff_X = r_diff_X - L*tf.round(r_diff_X/L)
    r_diff_Y = r_in_gath_Y - r_in_rep_Y
    r_diff_Y = r_diff_Y - L*tf.round(r_diff_Y/L)
    norm = tf.sqrt(tf.square(r_diff_X) + tf.square(r_diff_Y))
    
    
    binv = tf.math.reciprocal(norm) 
    bx = tf.math.multiply(r_diff_X, binv)
    by = tf.math.multiply(r_diff_Y, binv)

    zeroDummy = tf.zeros_like(norm)
    # add zero when the actual number of neighbors are less than maxNumNeigh 
    binv_safe = tf.where(mask, (binv- av[0])/std[0], zeroDummy)
    bx_safe = tf.where(mask, bx, zeroDummy)
    by_safe = tf.where(mask, by, zeroDummy)
    
    r_total = tf.concat([tf.reshape(binv_safe, (-1,1)), 
                         tf.reshape(bx_safe, (-1,1)), 
                         tf.reshape(by_safe, (-1,1))], axis = 1)

    return r_total

@tf.function
def gen_coor_3d(r_in, neigh_list, L, 
                            av = tf.constant([0.0, 0.0], dtype = tf.float32),
                            std =  tf.constant([1.0, 1.0], dtype = tf.float32)):

    # This function follows the same trick 
    # function to generate the generalized coordinates for periodic data
    # neigh_list is a (Nsample, Npoints, maxNeigh)

    Nsamples = r_in.shape[0]
    max_num_neighs = neigh_list.shape[-1]


    # define an indicator
    mask = neigh_list > -1

    # extract per_dimension the repeated and gathered entries
    r_in_rep_X  = tf.tile(tf.expand_dims(r_in[:,:,0], -1),
                       [1 ,1, max_num_neighs] )   
    r_in_gath_X = tf.gather(r_in[:,:,0], neigh_list, 
                           batch_dims = 1, axis = 1)
    r_in_rep_Y  = tf.tile(tf.expand_dims(r_in[:,:,1], -1),
                       [1 ,1, max_num_neighs] )
    r_in_gath_Y = tf.gather(r_in[:,:,1], neigh_list, 
                           batch_dims = 1, axis = 1)
    r_in_rep_Z  = tf.tile(tf.expand_dims(r_in[:,:,2], -1),
                       [1 ,1, max_num_neighs] )
    r_in_gath_Z = tf.gather(r_in[:,:,2], neigh_list, 
                           batch_dims = 1, axis = 1)


    # compute the periodic dimension wise distance
    r_diff_X = r_in_gath_X - r_in_rep_X
    r_diff_X = r_diff_X - L*tf.round(r_diff_X/L)
    r_diff_Y = r_in_gath_Y - r_in_rep_Y
    r_diff_Y = r_diff_Y - L*tf.round(r_diff_Y/L)
    r_diff_Z = r_in_gath_Z - r_in_rep_Z
    r_diff_Z = r_diff_Z - L*tf.round(r_diff_Z/L)

    norm = tf.sqrt(tf.square(r_diff_X) + tf.square(r_diff_Y) + tf.square(r_diff_Z))

    binv = tf.math.reciprocal(norm) 
    bx = tf.math.multiply(r_diff_X, binv)
    by = tf.math.multiply(r_diff_Y, binv)
    bz = tf.math.multiply(r_diff_Z, binv)

    zeroDummy = tf.zeros_like(norm)
    # add zero when the actual number of neighbors are less than maxNumNeigh
    binv_safe = tf.where(mask, (binv- av[0])/std[0], zeroDummy)
    bx_safe = tf.where(mask, bx, zeroDummy)
    by_safe = tf.where(mask, by, zeroDummy)
    bz_safe = tf.where(mask, bz, zeroDummy)
    
    r_total = tf.concat([tf.reshape(binv_safe, (-1,1)), 
                         tf.reshape(bx_safe,   (-1,1)), 
                         tf.reshape(by_safe,   (-1,1)),
                         tf.reshape(bz_safe,   (-1,1)) ], axis = 1)

    return r_total

