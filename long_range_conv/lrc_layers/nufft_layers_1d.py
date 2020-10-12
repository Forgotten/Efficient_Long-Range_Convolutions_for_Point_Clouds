import tensorflow as tf
import numpy as np 

@tf.function 
def gaussianPer(x, tau, L = 2*np.pi):
  return tf.exp( -tf.square(x  )/(4*tau)) + \
         tf.exp( -tf.square(x-L)/(4*tau)) + \
         tf.exp( -tf.square(x+L)/(4*tau))
         
@tf.function 
def gaussianDeconv(k, tau):
  return tf.sqrt(np.pi/tau)*tf.exp(tf.square(k)*tau)


class NUFFTLayerMultiChannelInitMixed(tf.keras.layers.Layer):
  def __init__(self, nChannels, NpointsMesh, xLims, mu1 = 1.0, mu2=0.5):
    super(NUFFTLayerMultiChannelInitMixed, self).__init__()
    self.nChannels = nChannels
    self.NpointsMesh = NpointsMesh 
    self.mu1 = tf.constant(mu1, dtype=tf.float32)
    self.mu2 = tf.constant(mu2, dtype=tf.float32)
    # we need the number of points to be odd 
    assert NpointsMesh % 2 == 1

    
    self.xLims = xLims
    self.L = np.abs(xLims[1] - xLims[0])
    self.tau = tf.constant(12*(self.L/(2*np.pi*NpointsMesh))**2, 
                           dtype = tf.float32)# the size of the mollifications
    self.kGrid = tf.constant((2*np.pi/self.L)*\
                              np.linspace(-(NpointsMesh//2), 
                                            NpointsMesh//2, 
                                            NpointsMesh), 
                              dtype = tf.float32)
    # we need to define a mesh betwen xLims[0] and xLims[1]
    self.xGrid =  tf.constant(np.linspace(xLims[0], 
                                          xLims[1], 
                                          NpointsMesh+1)[:-1], 
                              dtype = tf.float32)


  def build(self, input_shape):

    print("building the channels")
    # we initialize the channel multipliers
    self.shift = []
    for ii in range(2):
      self.shift.append(self.add_weight("std_"+str(ii),
                       initializer=tf.initializers.ones(),
                       shape=[1,]))
    self.amplitud = []
    for ii in range(2):
      self.amplitud.append(self.add_weight("bias_"+str(ii),
                       initializer=tf.initializers.ones(),
                       shape=[1,]))

  @tf.function
  def call(self, input):
    # we need to add an iterpolation step
    Npoints = input.shape[-1]
    batch_size = input.shape[0]
    diff = tf.expand_dims(input, -1) - tf.reshape(self.xGrid, (1,1, self.NpointsMesh))
    # (batch_size, Np*Ncells, NpointsMesh)
    array_gaussian = gaussianPer(diff, self.tau, self.L)
     # (batch_size, Np*Ncells, NpointsMesh)
    array_Gaussian_complex = tf.complex(array_gaussian, 0.0)
    #  (batch_size, Np*Ncells, NpointsMesh)
    fftGauss = tf.signal.fftshift(tf.signal.fft(array_Gaussian_complex),axes=-1)    
    # (batch_size, Np*Ncells, NpointsMesh)
    Deconv = tf.complex(tf.expand_dims(tf.expand_dims(gaussianDeconv(self.kGrid, self.tau), 0),0),0.0)

    rfft = tf.multiply(fftGauss, Deconv)
    #(batch_size, Np*Ncells,NpointsMesh)
    Rerfft = tf.math.real(rfft)
    Imrfft = tf.math.imag(rfft)
    multiplier1 = tf.expand_dims(tf.expand_dims(self.amplitud[0]*4*np.pi*\
                                tf.math.reciprocal( tf.square(self.kGrid) + \
                                tf.square(self.mu1*self.shift[0])), 0),0)
    multiplierRe1 = tf.math.real(multiplier1)    
    multReRefft = tf.multiply(multiplierRe1,Rerfft)
    multImRefft = tf.multiply(multiplierRe1,Imrfft)

    multfft = tf.complex(multReRefft,multImRefft)
    ##(batch_size, Np*Ncells, NpointsMesh)
    # an alternative method:   
    #    fft = tf.complex(self.multipliersRe[0],self.multipliersIm[0])
    #    multFFT = tf.multiply(rfft,fft)
    multiplier2 = tf.expand_dims(tf.expand_dims(self.amplitud[1]*4*np.pi*\
                                tf.math.reciprocal( tf.square(self.kGrid) + \
                                tf.square(self.mu2*self.shift[1])), 0),0)
    multiplierRe2 = tf.math.real(multiplier2)
    
    multReRefft2 = tf.multiply(multiplierRe2,Rerfft)
    multImRefft2 = tf.multiply(multiplierRe2,Imrfft)
    multfft2 = tf.complex(multReRefft2, multImRefft2)


    multfftDeconv1 = tf.multiply(multfft, Deconv)
    multfftDeconv2 = tf.multiply(multfft2, Deconv)
    irfft1 = tf.math.real(tf.signal.ifft(tf.signal.ifftshift(multfftDeconv1,axes=-1)))/(2*np.pi*self.NpointsMesh/self.L)/(2*np.pi)
    irfft2 = tf.math.real(tf.signal.ifft(tf.signal.ifftshift(multfftDeconv2,axes=-1)))/(2*np.pi*self.NpointsMesh/self.L)/(2*np.pi)
    ##(batch_size, Np*Ncells, NpointsMesh)
    diag_sum1 = tf.reduce_sum(irfft1*array_gaussian,axis=-1)
    ##(batch_size,Np*Ncells) part energy
    total1 = tf.reduce_sum(tf.reduce_sum(irfft1,axis=1,keepdims=True)*array_gaussian,axis=-1)
    ##(batch_size,Np*Ncells) 
    energy1 = total1 - diag_sum1
    diag_sum2 = tf.reduce_sum(irfft2*array_gaussian,axis=-1)
    ##(batch_size,Np*Ncells) part energy
    total2 = tf.reduce_sum(tf.reduce_sum(irfft2,axis=1,keepdims=True)*array_gaussian,axis=-1)
    ##(batch_size,Np*Ncells) 
    energy2 = total2 - diag_sum2
    
    energy = tf.concat([tf.expand_dims(energy1,axis=-1),tf.expand_dims(energy2,axis=-1)],axis=-1)
    return energy

