

import numpy as np



def potential(x,y, mu):

    return -np.exp(-mu*np.sqrt(np.sum(np.square(y - x), axis = -1)))



def potential_diff(diff, mu):

    return -np.exp(-mu*np.sqrt(np.sum(np.square(diff), axis = -1)))



def forces(x,y, mu):

    return -mu*(y - x)/(np.finfo(float).eps+np.sqrt(np.sum(np.square(y - x), \
                                                           axis = -1, keepdims = True)))\
           *np.exp(-mu*np.sqrt(np.sum(np.square(y - x), axis = -1, keepdims = True)))



def forces_diff(diff, mu):

    return -mu*(diff)/(np.finfo(float).eps+np.sqrt(np.sum(np.square(diff), \
                                        axis = -1, keepdims = True)))\
           *np.exp(-mu*np.sqrt(np.sum(np.square(diff), \
                                      axis = -1, keepdims = True)))





def potential_per3D(x,y, mu, L):
    diff = y - x
    diff_per = diff - L*np.round(diff/L)
    return    potential_diff(diff_per, mu)



def forces_per3D(x,y, mu, L):
    diff = y - x
    diff_per = diff - L*np.round(diff/L)
    return    forces_diff(diff_per, mu)



def gaussian3D(x, y, z, center, tau):

    return (1/np.sqrt(2*np.pi*tau)**3)*\
           np.exp( -0.5*(  np.square(x - center[0]) \
                         + np.square(y - center[1]) \
                         + np.square(z - center[2]))/tau**2 )



def genDataPer3DMixed(Ncells, Np, mu1, mu2, Nsamples, minDelta, Lcell,weight1,weight2): 



    pointsArray = np.zeros((Nsamples, Np*Ncells**3, 3))
    potentialArray = np.zeros((Nsamples,1))
    forcesArray = np.zeros((Nsamples, Np*Ncells**3, 3))


    sizeCell = Lcell
    L = sizeCell*Ncells

    # define a mesh
    midPoints = np.linspace(sizeCell/2.0,Ncells*sizeCell-sizeCell/2.0, Ncells)
    yy, xx, zz = np.meshgrid(midPoints, midPoints, midPoints)
    midPoints = np.concatenate([np.reshape(xx, (Ncells, Ncells, 
                                                Ncells, 1,1)), 
                                np.reshape(yy, (Ncells, Ncells, 
                                                Ncells, 1,1)),
                                np.reshape(zz, (Ncells, Ncells, 
                                                Ncells, 1,1))], axis = -1) 

    for i in range(Nsamples):

        # generate the index randomly
        points = midPoints + sizeCell*(np.random.rand(Ncells, Ncells, 
                                                      Ncells, Np, 3)-0.5)
        relPoints = np.reshape(points, (-1,1,3)) -np.reshape(points, (1,-1,3))
        relPointsPer = relPoints - L*np.round(relPoints/L)
        distPoints = np.sqrt(np.sum(np.square(relPointsPer), axis=-1))

        # to avoid two points are too close: sigularity
        while np.min( distPoints[distPoints>0] ) < minDelta:

            points = midPoints + sizeCell*(np.random.rand(Ncells, Ncells, 
                                                          Ncells, Np, 3)-0.5)
            relPoints = np.reshape(points, (-1,1,3)) -np.reshape(points, (1,-1,3))    
            relPointsPer = relPoints - L*np.round(relPoints/L)
            distPoints = np.sqrt(np.sum(np.square(relPointsPer), axis=-1))


        # compute the points,weighted energy and weighte force
        pointsArray[i, :, :] = np.reshape(points,(Np*Ncells**3, 3))
        points  = np.reshape(points, (Np*Ncells**3, 1, 3))
        pointsT = np.reshape(points, (1, Np*Ncells**3, 3))

        R1 = potential_per3D(points, pointsT, mu1, L)
        RR1 = np.triu(R1, 1)
        potTotal1 = np.sum(RR1)

        R2 = potential_per3D(points, pointsT, mu2, L)
        RR2 = np.triu(R2, 1)
        potTotal2 = np.sum(RR2)

        potentialArray[i,:] = potTotal1*weight1+potTotal2*weight2


        F1 = forces_per3D(points,pointsT, mu1, L)
        Forces1 = np.sum(F1, axis = 1) 

        F2 = forces_per3D(points,pointsT, mu2, L)
        Forces2 = np.sum(F2, axis = 1) 

        forcesArray[i,:,:] = np.reshape(Forces1,(Np*Ncells**3, 3))*weight1 +\
                             np.reshape(Forces2,(Np*Ncells**3, 3))*weight2

    return pointsArray, potentialArray, forcesArray

