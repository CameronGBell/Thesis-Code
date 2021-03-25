import numpy as np
import hyperspy.api as hs
import ParticleSpy.api as ps

#this code is used to assessing pixel wise accuracy and calculating dice coefficients

def check_ground(labels, truth):

    shape = labels.shape
    correcc = labels == truth
    num = np.count_nonzero(correcc)

    return (100*num/(shape[0]*shape[1]))


def dice(labels, truth):
    #die is [Pred,True]
    die = np.zeros((2,2))

    particles = truth
    particle_pixels = np.count_nonzero(particles)
    die[1,1] = np.count_nonzero((particles*labels)==1)/particle_pixels
    die[1,0] = 1-die[1,1]

    background = 1-truth
    background_pixels = np.count_nonzero(background)
    die[0,0] = np.count_nonzero((truth+labels)==0)/background_pixels
    die[0,1] = 1-die[0,0]
    
    return die

