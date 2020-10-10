import tensorflow as tf
import numpy as np 


@tf.function 
def gaussianPer(x, tau, L = 2*np.pi):
  return tf.exp( -tf.square(x  )/(4*tau)) + \
         tf.exp( -tf.square(x-L)/(4*tau)) + \
         tf.exp( -tf.square(x+L)/(4*tau))

@tf.function 
def gaussianDeconv2D(kx, ky, tau):
  return (np.pi/tau)*tf.exp((tf.square(kx) + tf.square(ky))*tau)


class NUFFTLayerMultiChannel2Dmixed(tf.keras.layers.Layer):
  def __init__(self, nChannels, NpointsMesh, xLims, 
               mu0 = 1.0, mu1 = 1.0):
    super(NUFFTLayerMultiChannel2Dmixed, self).__init__()
    self.nChannels = nChannels
    self.NpointsMesh = NpointsMesh 

    # this is for the initial guess 
    self.mu0 = tf.constant(mu0, dtype=tf.float32)
    self.mu1 = tf.constant(mu1, dtype=tf.float32)
    
    # we need the number of points to be odd 
    assert NpointsMesh % 2 == 1

    self.xLims = xLims
    print(xLims)
    self.L = np.abs(self.xLims[1] - self.xLims[0])
    self.tau = tf.constant(12*(self.L/(2*np.pi*NpointsMesh))**2, 
                           dtype = tf.float32)# the size of the mollifications
    self.kGrid = tf.constant((2*np.pi/self.L)*\
                              np.linspace(-(NpointsMesh//2), 
                                            NpointsMesh//2, 
                                            NpointsMesh), 
                              dtype = tf.float32)
    self.ky_grid, self.kx_grid = tf.meshgrid(self.kGrid, 
                                             self.kGrid ) 

    # we need to define a mesh betwen xLims[0] and xLims[1]
    self.xGrid =  tf.constant(np.linspace(xLims[0], 
                                          xLims[1], 
                                          NpointsMesh+1)[:-1], 
                              dtype = tf.float32)

    self.y_grid, self.x_grid = tf.meshgrid(self.xGrid, 
                                           self.xGrid) 



  def build(self, input_shape):

    print("building the channels")
    # we initialize the channel multipliers
    # we need to add a parametrized family in here
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
    # this needs to be properly initialized it

  @tf.function
  def call(self, input):
    # we need to add an iterpolation step
    # this needs to be perodic distance!!!
    # (batch_size, Np*Ncells, 2)
    diffx  = tf.expand_dims(tf.expand_dims(input[:,:,0], -1), -1) \
           - tf.reshape(self.x_grid, (1,1, 
                                      self.NpointsMesh, 
                                      self.NpointsMesh))

    diffy  = tf.expand_dims(tf.expand_dims(input[:,:,1], -1), -1) \
           - tf.reshape(self.y_grid, (1,1, 
                                      self.NpointsMesh, 
                                      self.NpointsMesh))

    # 2 x (batch_size, Np*Ncells, NpointsMesh, NpointsMesh)
    # we compute all the gaussians
    array_gaussian_x = gaussianPer(diffx, self.tau, self.L)
    array_gaussian_y = gaussianPer(diffy, self.tau, self.L)

    # we multiply the components
    array_gaussian = array_gaussian_x*array_gaussian_y
    # (batch_size, Np*Ncells, NpointsMesh, NpointsMesh)
    arrayReducGaussian = tf.complex(array_gaussian, 0.0)
    # (batch_size, Npoints,NpointsMesh, NpointsMesh)
    # we apply the fft
    fftGauss = tf.signal.fftshift(tf.signal.fft2d(arrayReducGaussian))
    # (batch_size, Npoints,NpointsMesh, NpointsMesh)
    # compute the deconvolution kernel 
    gauss_deconv = gaussianDeconv2D(self.kx_grid, 
                                    self.ky_grid, 
                                    self.tau)
    Deconv = tf.complex(tf.expand_dims(tf.expand_dims(gauss_deconv, 0),0),0.0)
    # (1, 1, NpointsMesh, NpointsMesh)


    rfft = tf.multiply(fftGauss, Deconv)
    Rerfft = tf.math.real(rfft)
    Imrfft = tf.math.imag(rfft)
    # compute two multipliers
    multiplier1 = tf.expand_dims(tf.expand_dims(self.amplitud[0]*4*np.pi*\
                                tf.math.reciprocal( tf.square(self.kx_grid) + \
                                                    tf.square(self.ky_grid) + \
                                tf.square(self.mu0*self.shift[0])), 0),0)
    multiplierRe1 = tf.math.real(multiplier1)    
    multReRefft = tf.multiply(multiplierRe1,Rerfft)
    multImRefft = tf.multiply(multiplierRe1,Imrfft)
    multfft = tf.complex(multReRefft,multImRefft)


    multiplier2 = tf.expand_dims(tf.expand_dims(self.amplitud[1]*4*np.pi*\
                                tf.math.reciprocal( tf.square(self.kx_grid) + \
                                                    tf.square(self.ky_grid) + \
                                tf.square(self.mu1*self.shift[1])), 0),0)
    multiplierRe2 = tf.math.real(multiplier2)    
    multReRefft2 = tf.multiply(multiplierRe2,Rerfft)
    multImRefft2 = tf.multiply(multiplierRe2,Imrfft)
    multfft2 = tf.complex(multReRefft2, multImRefft2)
    #(batch_size, 1, NpointsMesh, NpointsMesh)

    multfftDeconv1 = tf.multiply(multfft, Deconv)
    irfft1 = tf.math.real(tf.signal.ifft2d(
                         tf.signal.ifftshift(multfftDeconv1)))/(2*np.pi*self.NpointsMesh/self.L)**2/(2*np.pi)**2/2
    ##(Nsamples,Npoints,NpointsMesh,NpointsMesh)

    diag_sum1 = tf.reduce_sum(tf.reduce_sum(irfft1*array_gaussian,axis=-1),axis=-1)
    ##(Nsamples,Np*Ncells) part energy
    total1 = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(irfft1,axis=1,keepdims=True)*array_gaussian,axis=-1),axis=-1)    
    ##(Nsamples,Np*Ncells)
    energy1 = total1 - diag_sum1

    
    multfftDeconv2 = tf.multiply(multfft2, Deconv)
    irfft2 = tf.math.real(tf.signal.ifft2d(
                         tf.signal.ifftshift(multfftDeconv2)))/(2*np.pi*self.NpointsMesh/self.L)**2/(2*np.pi)**2/2
    ##(Nsamples,Npoints,NpointsMesh,NpointsMesh)
    
    diag_sum2 = tf.reduce_sum(tf.reduce_sum(irfft2*array_gaussian,axis=-1),axis=-1)
    ##(Nsamples,Np*Ncells) part energy
    total2 = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(irfft2,axis=1,keepdims=True)*array_gaussian,axis=-1),axis=-1)    
    ##(Nsamples,Np*Ncells) 
    energy2 = total2 - diag_sum2

    energy = tf.concat([tf.expand_dims(energy1,axis=-1),tf.expand_dims(energy2,axis=-1)],axis=-1)
    ##(Nsamples,Np*Ncells,2)
    return energy

