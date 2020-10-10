import numpy as np

def potential(x,y, mu):
    return -np.exp(-mu*np.sqrt(np.sum(np.square(y - x), axis = -1)))



def forces(x,y, mu):
    return -mu*(y - x)/(np.finfo(float).eps+np.sqrt(np.sum(np.square(y - x), \
            axis = -1, keepdims = True)))*np.exp(-mu*np.sqrt(np.sum(np.square(y - x), axis = -1, keepdims = True)))


# compute exponential potential 
def potentialPer(x,y, mu, L):
    shift_x = np.reshape(np.array([L, 0.]), (1,1,2))
    shift_y = np.reshape(np.array([0., L]), (1,1,2))
    return potential(x,y, mu) + potential(x+shift_x,y, mu) + potential(x-shift_x,y, mu)\
           +potential(x+shift_y,y, mu) + potential(x+shift_x+shift_y,y, mu) + potential(x-shift_x+shift_y,y, mu) \
           +potential(x-shift_y,y, mu) + potential(x+shift_x-shift_y,y, mu) + potential(x-shift_x-shift_y,y, mu)

# compute exponential force
def forcesPer(x,y, mu, L):
    shift_x = np.reshape(np.array([L, 0.]), (1,1,2))
    shift_y = np.reshape(np.array([0., L]), (1,1,2))
    return   forces(x,y, mu) + forces(x+shift_x,y, mu) + forces(x-shift_x,y, mu)\
           + forces(x+shift_y,y, mu) + forces(x+shift_x+shift_y,y, mu) + forces(x-shift_x+shift_y,y, mu)\
           + forces(x-shift_y,y, mu) + forces(x+shift_x-shift_y,y, mu) + forces(x-shift_x-shift_y,y, mu)


def gaussian2D(x,y, center, tau):
    return (1/(2*np.pi*(tau**2)))*\
           np.exp( -0.5*(  np.square(x - center[0])
                         + np.square(y - center[1]))/tau**2)





def computeDerPot2DPer(Nx, mu, Ls, x_center = [0.0, 0.0], nPointSmear = 5):   

    xGrid = np.linspace(0, Ls, Nx+1)[:-1] 
    kGrid = 2*np.pi*np.linspace(-(Nx//2), Nx//2, Nx)/Ls      
    # creating the 2D space and frequency grids
    y_grid, x_grid = np.meshgrid(xGrid, xGrid)
    ky_grid, kx_grid = np.meshgrid(kGrid, kGrid)
    mult = 4*np.pi/(  np.square(kx_grid) 
                      + np.square(ky_grid) 
                      + np.square(mu))
    # here we smear the dirac delta
    tau = nPointSmear*Ls/Nx
    # periodic distance 
    x_diff = x_grid - x_center[0]
    x_diff_per = x_diff - Ls*np.round(x_diff/Ls)

    y_diff = y_grid - x_center[0]
    y_diff_per = y_diff - Ls*np.round(y_diff/Ls)



    # define the periodic gaussian
    tau_gauss = gaussian2D(x_diff_per,y_diff_per, [0.0, 0.0], tau)

    # compute the fourier transform of the gaussian 
    xFFT = np.fft.fftshift(np.fft.fft2(tau_gauss))
    fFFT = xFFT*mult
    f = np.real(np.fft.ifft2(np.fft.ifftshift(fFFT)))

    # compute force
    dfdxFFT = 1.j*kx_grid*fFFT
    dfdyFFT = 1.j*ky_grid*fFFT

    dfdx = np.fft.ifft2(np.fft.ifftshift(dfdxFFT))
    dfdy = np.fft.ifft2(np.fft.ifftshift(dfdyFFT))
    return x_grid, y_grid, f, np.real(dfdx), np.real(dfdy)




# compute Yukawa data
def genDataYukawa2DPermixed(Ncells, Np, mu1, mu2, Nsamples, minDelta, Lcell,weight1,weight2): 

    points_array = np.zeros((Nsamples, Np*Ncells**2, 2))
    potential_array = np.zeros((Nsamples,1))
    forces_array = np.zeros((Nsamples, Np*Ncells**2, 2))
    sizeCell = Lcell

    # define a mesh
    NpointsPerCell = 1000
    Nx = Ncells*NpointsPerCell + 1
    Ls = Ncells*sizeCell
    
    # compute grid, potential, force for different mu
    x_grid, y_grid, pot1, dpotdx1, dpotdy1 = computeDerPot2DPer(Nx, mu1, Ls)
    x_grid, y_grid, pot2, dpotdx2, dpotdy2 = computeDerPot2DPer(Nx, mu2, Ls)


    # centering the points within each cell 
    idxCell = np.linspace(0,NpointsPerCell-1, NpointsPerCell).astype(int)
    idxStart = np.array([ii*NpointsPerCell for ii in range(Ncells)]).reshape(-1,1)
    idx_cell_y, idx_cell_x = np.meshgrid(idxCell, idxCell) 
    idx_start_y, idx_start_x = np.meshgrid(idxStart, idxStart) 

    for i in range(Nsamples):

        dist = 0.0
        # generate the index randomly
        idx_point_x = idx_start_x.reshape((Ncells, Ncells, 1)) \
                      + np.random.choice(idx_cell_x.reshape((-1,)), 
                                         [Ncells, Ncells, Np])
        idx_point_y = idx_start_y.reshape((Ncells, Ncells, 1)) \
                      + np.random.choice(idx_cell_y.reshape((-1,)), 
                                         [Ncells, Ncells, Np])
        
        # to avoid two points are too close: sigularity 
        while np.min(dist) < minDelta:
            idx_point_x = idx_start_x.reshape((Ncells, Ncells, 1)) \
                          + np.random.choice(idx_cell_x.reshape((-1,)), 
                                             [Ncells, Ncells, Np])
            idx_point_y = idx_start_y.reshape((Ncells, Ncells, 1)) \
                          + np.random.choice(idx_cell_y.reshape((-1,)), 
                                             [Ncells, Ncells, Np])

            points_x = x_grid[idx_point_x, 0]
            points_y = y_grid[0, idx_point_y]
            # compute the periodic distance

            diff_x = points_x.reshape((-1,1)) - points_x.reshape((-1,1)).T
            diff_y = points_y.reshape((-1,1)) - points_y.reshape((-1,1)).T
            diff_x -= Ls*np.round(diff_x/Ls)
            diff_y -= Ls*np.round(diff_y/Ls)
            dist = np.sqrt(np.square(diff_x) + np.square(diff_y))
            # the diagonal will be zero, so we add a diagonal to properly 
            # comput the minimal distance
            dist += 10*np.eye(Ncells**2*Np)

        # compute the points,weighted energy and weighte force
        points_array[i, :, 0] = x_grid[idx_point_x.reshape((-1,)), 0]
        points_array[i, :, 1] = y_grid[0, idx_point_y.reshape((-1,))]


        R1 = pot1[idx_point_x.reshape((-1,1)) - idx_point_x.reshape((-1,1)).T, 
                  idx_point_y.reshape((-1,1)) - idx_point_y.reshape((-1,1)).T]
        RR1 = np.triu(R1, 1)
        potTotal1 = np.sum(RR1)

        R2 = pot2[idx_point_x.reshape((-1,1)) - idx_point_x.reshape((-1,1)).T, 
                  idx_point_y.reshape((-1,1)) - idx_point_y.reshape((-1,1)).T]
        RR2 = np.triu(R2, 1)
        potTotal2 = np.sum(RR2)

        potential_array[i,:] = potTotal1*weight1 + potTotal2*weight2



        Fx1 = dpotdx1[idx_point_x.reshape((-1,1)) - idx_point_x.reshape((-1,1)).T, 
                      idx_point_y.reshape((-1,1)) - idx_point_y.reshape((-1,1)).T]

        Fy1 = dpotdy1[idx_point_x.reshape((-1,1)) - idx_point_x.reshape((-1,1)).T, 
                      idx_point_y.reshape((-1,1)) - idx_point_y.reshape((-1,1)).T]
                                         
        Fx1 = np.triu(Fx1,1) + np.tril(Fx1,-1)
        Fy1 = np.triu(Fy1,1) + np.tril(Fy1,-1)
        Forcesx1 = -np.sum(Fx1, axis = 1) 
        Forcesy1 = -np.sum(Fy1, axis = 1) 

        Fx2 = dpotdx2[idx_point_x.reshape((-1,1)) - idx_point_x.reshape((-1,1)).T, 
                      idx_point_y.reshape((-1,1)) - idx_point_y.reshape((-1,1)).T]

        Fy2 = dpotdy2[idx_point_x.reshape((-1,1)) - idx_point_x.reshape((-1,1)).T, 
                      idx_point_y.reshape((-1,1)) - idx_point_y.reshape((-1,1)).T]
                                         
        Fx2 = np.triu(Fx2,1) + np.tril(Fx2,-1)
        Fy2 = np.triu(Fy2,1) + np.tril(Fy2,-1)
        Forcesx2 = -np.sum(Fx2, axis = 1) 
        Forcesy2 = -np.sum(Fy2, axis = 1) 
        

        forces_array[i,:,0] = Forcesx1.T*weight1 + Forcesx2.T*weight2
        forces_array[i,:,1] = Forcesy1.T*weight1 + Forcesy2.T*weight2

    return points_array, potential_array, forces_array

# compute exponential data
def genDataPer2DMixed(Ncells, Np, mu1, mu2, Nsamples, minDelta = 0.0, Lcell = 0.0, weight1=0.5, weight2=0.5): 

	pointsArray = np.zeros((Nsamples, Np*Ncells**2, 2))
	potentialArray = np.zeros((Nsamples,1))
	forcesArray = np.zeros((Nsamples, Np*Ncells**2, 2))
	sizeCell = Lcell
	L = sizeCell*Ncells

    # define a mesh
	midPoints = np.linspace(sizeCell/2.0,Ncells*sizeCell-sizeCell/2.0, Ncells)
	xx, yy = np.meshgrid(midPoints, midPoints)
	midPoints = np.concatenate([np.reshape(xx, (Ncells,Ncells,1,1)), 
								np.reshape(yy, (Ncells,Ncells,1,1))], axis = -1) 

	for i in range(Nsamples):
        # generate the index randomly
		points = midPoints + sizeCell*(np.random.rand(Ncells, Ncells, Np,2) -0.5)
		relPoints = np.reshape(points, (-1,1,2)) -np.reshape(points, (1,-1,2))
		relPointsPer = relPoints - L*np.round(relPoints/L)
		distPoints = np.sqrt(np.sum(np.square(relPointsPer), axis=-1))

        # to avoid two points are too close: sigularity
		while np.min( distPoints[distPoints>0] ) < minDelta:
		    points = midPoints + sizeCell*(np.random.rand(Ncells, Ncells, Np,2)-0.5)
		    relPoints = np.reshape(points, (-1,1,2)) -np.reshape(points, (1,-1,2))  
		    relPointsPer = relPoints - L*np.round(relPoints/L)    
		    distPoints = np.sqrt(np.sum(np.square(relPointsPer), axis=-1))            
            
        # compute the points,weighted energy and weighte force    
		pointsArray[i, :, :] = np.reshape(points,(Np*Ncells**2, 2))
		points = np.reshape(points, (Np*Ncells**2,1,2))
		pointsT = np.reshape(points, (1,Np*Ncells**2,2))

		R1 = potentialPer(points, pointsT, mu1, L)
		RR1 = np.triu(R1, 1)
		potTotal1 = np.sum(RR1)

		R2 = potentialPer(points, pointsT, mu2, L)
		RR2 = np.triu(R2, 1)
		potTotal2 = np.sum(RR2)
        
		potentialArray[i,:] = potTotal1*weight1 + potTotal2*weight2
		F1 = forcesPer(points,pointsT, mu1, L)
		Forces1 = np.sum(F1, axis = 1)
		F2 = forcesPer(points,pointsT, mu2, L)
		Forces2 = np.sum(F2, axis = 1)
        
		forcesArray[i,:,:] = np.reshape(Forces1,(Np*Ncells**2, 2))*weight1 +\
                             np.reshape(Forces2,(Np*Ncells**2, 2))*weight2

	return pointsArray, potentialArray, forcesArray
