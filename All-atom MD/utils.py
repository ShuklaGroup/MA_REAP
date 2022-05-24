# Some util functions to use across notebooks for tests with toy potentials

import pickle

import openmm as mm
from simtk.unit import *


def save_pickle(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)


def load_pickle(filename):
    with open(filename, 'rb') as infile:
        return pickle.load(infile)


def frame_to_openmm(xyz_positions):
    '''
    Converts a numpy array containing (n_atoms, 3) positions (single frame) into an openmm Quantity[Vec3] object for a simulation.
    '''
    return Quantity(value=[mm.vec3.Vec3(*atom_pos) for atom_pos in xyz_positions],
                    unit=nanometer)
