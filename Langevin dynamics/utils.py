# Some util functions to use across tests

import pickle
import numpy as np
from matplotlib import pyplot as plt
import openmm as mm

def save_pickle(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)

def load_pickle(filename):
    with open(filename, 'rb') as infile:
        return pickle.load(infile)
    
def plot_potential(potential, xlim, ylim):
    '''
    Plots potential based on analytic function.
    '''
    x = np.arange(*xlim)
    y = np.arange(*ylim)
    X, Y = np.meshgrid(x, y) # grid of point
    Z = potential(X, Y) # evaluation of the function on the grid

    im = plt.imshow(Z, cmap=plt.cm.jet, extent=[xlim[0], xlim[1], ylim[0], ylim[1]]) # drawing the function
    # adding the Contour lines with labels
    # cset = plt.contour(Z, [-2, -1, 0, 1, 2, 3], linewidths=2, cmap=plt.cm.binary, extent=[xlim[0], xlim[1], ylim[1], ylim[0]])
    # plt.clabel(cset, inline=True,fmt='%1.1f', fontsize=10)
    plt.colorbar(im) # adding the colobar on the right
    plt.show()
    plt.close()

# Simulation run snippet --> For long unbiased simulation only

def run_simulation(n_steps=10000, potential='', platform='CPU'): # Base code I found online
    '''
    Simulate a single particle under Langevin dynamics.
    '''
    system = mm.System()
    system.addParticle(100) # Added particle with mass of 1 amu
    force = mm.CustomExternalForce(potential) # Defines the potential
    force.addParticle(0, [])
    system.addForce(force)
    integrator = mm.LangevinIntegrator(300, 1, 0.02) # Langevin integrator with 300K temperature, gamma=1, step size = 0.02
    platform = mm.Platform.getPlatformByName(platform)
    context = mm.Context(system, integrator, platform)
    context.setPositions([[0, 0, 0]])
    context.setVelocitiesToTemperature(300)
    x = np.zeros((n_steps, 3))
    for i in range(n_steps):
        x[i] = context.getState(getPositions=True).getPositions(asNumpy=True)._value
        integrator.step(1)
    return x

# Simulation run snippets for adaptive sampling

def setup_simulation(potential='', platform='CPU'):
    '''
    Setup Langevin dynamics simulation with custom potential.
    '''
    system = mm.System()
    system.addParticle(100) # Add particle with mass of 1 amu
    force = mm.CustomExternalForce(potential) # Defines the potential
    force.addParticle(0, [])
    system.addForce(force)
    integrator = mm.LangevinIntegrator(300, 1, 0.002) # Langevin integrator with 300 temperature, gamma=1, step size = 0.002
    platform = mm.Platform.getPlatformByName(platform)
    
    return system, integrator, platform

def run_trajectory(n_steps=0, potential='', initial_position=[0,0,0]):
    '''
    Run a simulation of a single particle under Langevin dynamics for n_steps.
    '''
    system, integrator, platform = setup_simulation(potential=potential)
    context = mm.Context(system, integrator, platform)
    context.setPositions([initial_position])
    context.setVelocitiesToTemperature(300)
    x = np.zeros((n_steps, 3))
    for i in range(n_steps):
        x[i] = context.getState(getPositions=True).getPositions(asNumpy=True)._value
        integrator.step(1)
    return x

def area_explored(potential, xlim, ylim, data, threshold):
    '''
    Computes the fraction of area discovered by the trajectories so far.
    
    Args
    -------------
    potential (callable): potential function (two-dimensional).
    xlim (tuple): start, end, and stride for grid definition (x-variable).
    ylim (tuple): start, end, and stride for grid definition (y-variable).
    data (np.ndarray): array of shape (n_samples, n_features) (here, n_features = 2).
        Trajectories computed so far.
    threshold (float): only count the area where the free energy is less than this threshold.
   
    Returns
    -------------
    fraction (float): fraction of area discovered by the trajectories. Varies from 0 to 1.
    '''
    x = np.arange(*xlim)
    y = np.arange(*ylim)
    
    x_width = xlim[-1]
    y_width = ylim[-1]
    
    X, Y = np.meshgrid(x, y) # grid of point
    Z = potential(X, Y) # evaluation of the function on the grid
    
    H = np.histogramdd(data, bins=(x,y))[0]
    H_masked = H[ (np.where(Z < threshold)) ]
    
    return np.count_nonzero(H_masked)/H_masked.shape[0]
