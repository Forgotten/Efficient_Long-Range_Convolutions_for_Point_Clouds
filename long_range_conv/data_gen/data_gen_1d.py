import numpy as np

def potential_Per(x,y, mu,L):
    return np.exp( -mu*np.abs(y - x) )+\
           np.exp( -mu*np.abs(y - x - L) )+\
           np.exp( -mu*np.abs(y - x + L) )

def forces_Per(x,y, mu,L):
    return -mu*np.sign(y - x)*np.exp( -mu*np.abs(y - x) ) +\
           -mu*np.sign(y - x - L)*np.exp( -mu*np.abs(y - x - L) ) +\
           -mu*np.sign(y - x + L)*np.exp( -mu*np.abs(y - x + L) )

def gen_data_Per_Mixed(Ncells, Np, mu1, mu2, Nsamples, minDelta = 0.0, Lcell = 0.0 , weight1=0.5 , weight2=0.5):
    
    pointsArray = np.zeros((Nsamples, Np*Ncells))
    potentialArray = np.zeros((Nsamples,1))
    forcesArray = np.zeros((Nsamples, Np*Ncells))
    

    sizeCell = Lcell 
    L = sizeCell*Ncells
    
    for i in range(Nsamples):
        midPoints = np.linspace(sizeCell/2.0,Ncells*sizeCell-sizeCell/2.0, Ncells)

        points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
        points = np.sort(points.reshape((-1,1)), axis = 0)

        pointsExt = np.concatenate([points - L, points, points + L])
        # we want to check that the points are not too close
        while np.min(pointsExt[1:] - pointsExt[0:-1]) < minDelta:            
            points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
            points = np.sort(points.reshape((-1,1)), axis = 0)
            pointsExt = np.concatenate([points - L, points, points + L])
            
        pointsArray[i, :] = points.T

        R1 = potential_Per(points,points.T, mu1,L)
        RR1 = np.triu(R1, 1)
        potTotal1 = np.sum(RR1)
        R2 = potential_Per(points,points.T, mu2,L)
        RR2 = np.triu(R2, 1)
        potTotal2 = np.sum(RR2)
        potentialArray[i,:] = weight1*potTotal1+weight2*potTotal2

        F1 = forces_Per(points,points.T, mu1,L)
        Forces1 = np.sum(F1, axis = 1)
        F2 = forces_Per(points,points.T, mu2,L)
        Forces2 = np.sum(F2, axis = 1)
        forcesArray[i,:] = weight1*Forces1.T + weight2*Forces2.T

    return pointsArray, potentialArray, forcesArray



def genDataYukawaPerMixed(Ncells, Np, sigma1, sigma2, Nsamples, minDelta = 0.0, Lcell = 0.0,weight1=0.5,weight2=0.5):

    pointsArray = np.zeros((Nsamples, Np*Ncells))
    potentialArray = np.zeros((Nsamples,1))
    forcesArray = np.zeros((Nsamples, Np*Ncells))


    sizeCell = Lcell
    NpointsPerCell = int(1000*sizeCell)

    Nx = Ncells*NpointsPerCell + 1
    Ls = Ncells*sizeCell
    print('here I make an adjustment')
    xGrid, pot1, dpotdx1 = computeDerPotPer(Nx, sigma1, Ls)
    xGrid, pot2, dpotdx2 = computeDerPotPer(Nx, sigma2, Ls)

    idxCell = np.linspace(0,NpointsPerCell-1, NpointsPerCell).astype(int)
    idxStart = np.array([ii*NpointsPerCell for ii in range(Ncells)]).reshape(-1,1)

    for i in range(Nsamples):

        idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
        idxPointCell = np.sort(idxPointCell.reshape((-1,1)), axis = 0) ##position
        points = xGrid[idxPointCell]
        # this is to keep the periodicity
        pointsExt = np.concatenate([points - Ls, points, points + Ls])
        # we want to check that the points are not too close
        while np.min(pointsExt[1:] - pointsExt[0:-1]) < minDelta:
            idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
            idxPointCell = np.sort(idxPointCell.reshape((-1,1)), axis = 0)
            points = xGrid[idxPointCell]
            pointsExt = np.concatenate([points - Ls, points, points + Ls])

        pointsArray[i, :] = points.T

        R1 = pot1[idxPointCell - idxPointCell.T]##matrix interaction
        RR1 = np.triu(R1, 1) ###consider interaction effect
        potTotal1 = np.sum(RR1)
        R2 = pot2[idxPointCell - idxPointCell.T]##matrix interaction
        RR2 = np.triu(R2, 1) ###consider interaction effect
        potTotal2 = np.sum(RR2)
        potentialArray[i,:] = weight1*potTotal1 + weight2*potTotal2

        F1 = dpotdx1[idxPointCell - idxPointCell.T]
        F1 = np.triu(F1,1) + np.tril(F1,-1)
        Forces1 = -np.sum(F1, axis = 1)
        F2 = dpotdx2[idxPointCell - idxPointCell.T]
        F2 = np.triu(F2,1) + np.tril(F2,-1)
        Forces2 = -np.sum(F2, axis = 1)
        forcesArray[i,:] = weight1*Forces1.T + weight2*Forces2.T

    return pointsArray, potentialArray, forcesArray


def gaussian(x, xCenter, tau):
    return (1/np.sqrt(2*np.pi*tau**2))*\
           np.exp( -0.5*np.square(x - xCenter)/tau**2 )

def computeDerPotPer(Nx, mu, Ls, xCenter = 0, nPointSmear = 10):

    xGrid = np.linspace(0, Ls, Nx+1)[:-1] ##delete the last one
    kGrid = 2*np.pi*np.linspace(-(Nx//2), Nx//2, Nx)/Ls ##frequency domain

    filterM = 1#0.5 - 0.5*np.tanh(np.abs(3*kGrid/np.sqrt(Nx)) - np.sqrt(Nx))
    mult = 4*np.pi*filterM/(np.square(kGrid) + np.square(mu))
    ##yukuwa potential V = F(mult)

    # here we smear the dirac delta
    # we use the width of the smearing for
    tau = nPointSmear*Ls/Nx #standard deviation for guass, use guass to approach dirac

    x = gaussian(xGrid, xCenter, tau) + \
        gaussian(xGrid - Ls, xCenter, tau) +\
        gaussian(xGrid + Ls, xCenter, tau)

    xFFT = np.fft.fftshift(np.fft.fft(x)) #FFT of delta function, shift is just for convenience

    yFFT = xFFT*mult ### in frequency domain, multiply a certain factor
                     ###convolution
    y = np.real(np.fft.ifft(np.fft.ifftshift(yFFT))) ## y = dirac convolution V

    dydxFFT = 1.j*kGrid*yFFT #1.j=1j 1st derivative
    dydx = np.real(np.fft.ifft(np.fft.ifftshift(dydxFFT)))

    return xGrid, y, dydx ##y energy dydx force













