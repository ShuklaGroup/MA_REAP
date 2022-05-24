import numpy as np


# Potential definitions

########################################## EXAMPLES (NOT USED) ##########################################
def two_wells_func(x, y):
    return 2 * (x - 1) ** 2 * (x + 1) ** 2 + y ** 2


two_wells = '2*(x-1)^2*(x+1)^2 + y^2'


def two_wells_complex_func(x, y):
    return -3 * np.exp(-(x - 1) ** 2 - y ** 2) - 3 * np.exp(-(x + 1) ** 2 - y ** 2) + 15 * np.exp(
        -0.32 * (x ** 2 + y ** 2 + 20 * (x + y) ** 2)) + 0.0512 * (x ** 4 + y ** 4) + 0.4 * np.exp(-2 - 4 * y)


two_wells_complex = '-3*exp(-(x-1)^2-y^2)-3*exp(-(x+1)^2-y^2)+15*exp(-0.32*(x^2+y^2+20*(x+y)^2))+0.0512*(x^4+y^4)+0.4*exp(-2-4*y)'


#########################################################################################################

# Potentials for MA REAP

def gaussian_bivariate(c, mean, std, x, y):
    return c * np.exp(-0.5 * ((x - mean[0]) ** 2 / std[0] + (y - mean[1]) ** 2 / std[1]))


def four_wells_asymmetric_func(x, y):
    # Four corners
    cs = [-80, -25, -25, -25]
    means = [(0.2, 1),
             (1, 0.2),
             (1.8, 1),
             (1, 1.8),
             ]

    stds = [(0.008, 0.008),
            (0.008, 0.008),
            (0.008, 0.008),
            (0.008, 0.008),
            ]

    # Bridges
    cs += [-50, -50, -50, -50, -50, -50, -50, -50]

    means += [(0.4, 1),
              (0.8, 1),
              (1.2, 1),
              (1.6, 1),
              (1, 0.4),
              (1, 0.8),
              (1, 1.2),
              (1, 1.6),
              ]

    stds += [(0.04, 0.02),
             (0.04, 0.02),
             (0.04, 0.02),
             (0.04, 0.02),
             (0.02, 0.04),
             (0.02, 0.04),
             (0.02, 0.04),
             (0.02, 0.04),
             ]

    # Center compensation
    cs += [53]

    means += [(1, 1),
              ]

    stds += [(0.02, 0.02),
             ]

    potential = 0
    for c, mean, std in zip(cs, means, stds):
        potential += gaussian_bivariate(c, mean, std, x, y)

    return potential


def gaussian_bivariate_string():
    # Four corners
    cs = [-80, -25, -25, -25]
    means = [(0.2, 1),
             (1, 0.2),
             (1.8, 1),
             (1, 1.8),
             ]

    stds = [(0.008, 0.008),
            (0.008, 0.008),
            (0.008, 0.008),
            (0.008, 0.008),
            ]

    # Bridges
    cs += [-50, -50, -50, -50, -50, -50, -50, -50]

    means += [(0.4, 1),
              (0.8, 1),
              (1.2, 1),
              (1.6, 1),
              (1, 0.4),
              (1, 0.8),
              (1, 1.2),
              (1, 1.6),
              ]

    stds += [(0.04, 0.02),
             (0.04, 0.02),
             (0.04, 0.02),
             (0.04, 0.02),
             (0.02, 0.04),
             (0.02, 0.04),
             (0.02, 0.04),
             (0.02, 0.04),
             ]

    # Center compensation
    cs += ['+53']

    means += [(1, 1),
              ]

    stds += [(0.02, 0.02),
             ]

    base_string = '{c}*exp(-0.5*((x-{mean_x})^2/{std_x}+(y-{mean_y})^2/{std_y}))'

    string = ''
    for c, mean, std in zip(cs, means, stds):
        string += base_string.format(c=str(c), mean_x=str(mean[0]), mean_y=str(mean[1]), std_x=str(std[0]),
                                     std_y=str(std[1]))

    return string


four_wells_asymmetric = gaussian_bivariate_string()


def four_wells_symmetric_func(x, y):
    # Four corners
    cs = [-25, -25, -25, -25]
    means = [(0.2, 1),
             (1, 0.2),
             (1.8, 1),
             (1, 1.8),
             ]

    stds = [(0.008, 0.008),
            (0.008, 0.008),
            (0.008, 0.008),
            (0.008, 0.008),
            ]

    # Bridges
    cs += [-50, -50, -50, -50, -50, -50, -50, -50]

    means += [(0.4, 1),
              (0.8, 1),
              (1.2, 1),
              (1.6, 1),
              (1, 0.4),
              (1, 0.8),
              (1, 1.2),
              (1, 1.6),
              ]

    stds += [(0.04, 0.02),
             (0.04, 0.02),
             (0.04, 0.02),
             (0.04, 0.02),
             (0.02, 0.04),
             (0.02, 0.04),
             (0.02, 0.04),
             (0.02, 0.04),
             ]

    # Center compensation
    cs += [60]

    means += [(1, 1),
              ]

    stds += [(0.02, 0.02),
             ]

    potential = 0
    for c, mean, std in zip(cs, means, stds):
        potential += gaussian_bivariate(c, mean, std, x, y)

    return potential


four_wells_symmetric = gaussian_bivariate_string().replace('-80', '-25', 1)


# Potential for TSLC

def circular_potential_func(v, r=2, c=-250, a=-10):
    '''
    Computes circular potential value at point v = (x, y ,z).
    It is assumed the circle is centered at the origin.
    
    Args
    -----------------
    v (array-like): 3d coordinates of point in cartesian coordinates. 
    r (float): circle radius.
    c (float): energy minimum at the circle. Should be negative for the circle to be a stable equilibrium line.
    a (float): exponential scaling factor (adjusts how quickly the energy increases far form the circle). Should be negative.
    
    Returns
    -----------------
    potential (float): potential value at v.
    '''
    x, y, z = v
    circle_radius = r
    d_squared = z ** 2 + np.square(np.sqrt(x ** 2 + y ** 2) - circle_radius)
    potential = c * np.exp(a * d_squared)

    return potential


def circular_potential_string(r=2, c=-250, a=-10):
    d_squared = "z^2 + (sqrt(x^2 + y^2) - {circle_radius})^2".format(circle_radius=r)
    circular_potential = "{c}*exp({a}*({d}))".format(c=c, a=a, d=d_squared)

    return circular_potential


circular_potential = circular_potential_string()
