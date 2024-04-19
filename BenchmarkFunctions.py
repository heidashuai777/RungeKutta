# Description: This file contains the benchmark functions used in the Runge-Kutta optimization algorithm.
# The functions are F1 to F14. The functions are taken from the following paper:
# https://doi.org/10.1016/j.advengsoft.2020.102905
#importing the necessary packages
import numpy as np
# This function initializes the population of the optimization algorithm
def Ufun(x, a, k, m):
    return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < -a)

def F1(x):
    D = len(x)
    return x[0]**2 + 10**6 * np.sum(x[1:D]**2)

def F2(x):
    D = len(x)
    return np.sum([abs(x[i])**(i+1) for i in range(D)])

def F3(x):
    return np.sum(x**2) + (np.sum(0.5 * x))**2 + (np.sum(0.5 * x))**4

def F4(x):
    D = len(x)
    return np.sum([100 * (x[i]**2 - x[i+1])**2 + (x[i] - 1)**2 for i in range(D-1)])

def F5(x):
    D = len(x)
    return 10**6 * x[0]**2 + np.sum(x[1:D]**2)

def F6(x):
    D = len(x)
    return np.sum([((10**6)**((i)/(D-1))) * x[i]**2 for i in range(D)])

def F7(x):
    D = len(x)
    return np.sum([0.5 + (np.sin(np.sqrt(x[i]**2 + x[(i+1)%D]**2))**2 - 0.5) / (1 + 0.001 * (x[i]**2 + x[(i+1)%D]**2))**2 for i in range(D)])

def F8(x):
    D = len(x)
    w = 1 + (x - 1) / 4
    return np.sin(np.pi * w[0])**2 + np.sum([(w[i] - 1)**2 * (1 + 10 * np.sin(np.pi * w[i] + 1)**2) for i in range(D-1)]) + (w[D-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[D-1])**2)

def F9(x):
    D = len(x)
    y = x + 4.209687462275036e+002
    f = np.where(abs(y) < 500, y * np.sin(np.sqrt(abs(y))), (np.mod(abs(y), 500) - 500) * np.sin(np.sqrt(abs(np.mod(abs(y), 500) - 500))) - (y + 500)**2 / (10000 * D))
    return 418.9829 * D - np.sum(f)

def F10(x):
    D = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / D)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / D) + 20 + np.exp(1)

def F11(x):
    D = len(x)
    x = x + 0.5
    a = 0.5
    b = 3
    kmax = 20
    c1 = a**np.arange(0, kmax+1)
    c2 = 2 * np.pi * b**np.arange(0, kmax+1)
    f = np.sum([c1[i] * np.cos(c2[i] * x) for i in range(len(c1))], axis=0)
    c = -np.sum([c1[i] * np.cos(c2[i] * 0.5) for i in range(len(c1))])
    return np.sum(f) + c * D

def F12(x):
    D = len(x)
    return (abs(np.sum(x**2) - D))**(1/4) + (0.5 * np.sum(x**2) + np.sum(x)) / D + 0.5

def F13(x):
    D = len(x)
    return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, D+1)))) + 1

def F14(x):
    D = len(x)
    return (np.pi / D) * (10 * (np.sin(np.pi * (1 + (x[0] + 1) / 4)))**2 + np.sum((((x[:D-1] + 1) / 4)**2 * (1 + 10 * (np.sin(np.pi * (1 + (x[1:D] + 1) / 4))))**2) + ((x[D-1] + 1) / 4)**2) + np.sum(Ufun(x, 10, 100, 4)))
# This function loads the details of the selected benchmark function
def BenchmarkFunctions(F):
    D = 30
    if F == 'F1':
        fobj = F1
        lb = -100
        ub = 100
        dim = D
    elif F == 'F2':
        fobj = F2
        lb = -100
        ub = 100
        dim = D
    elif F == 'F3':
        fobj = F3
        lb = -100
        ub = 100
        dim = D
    elif F == 'F4':
        fobj = F4
        lb = -100
        ub = 100
        dim = D
    elif F == 'F5':
        fobj = F5
        lb = -100
        ub = 100
        dim = D
    elif F == 'F6':
        fobj = F6
        lb = -100
        ub = 100
        dim = D
    elif F == 'F7':
        fobj = F7
        lb = -100
        ub = 100
        dim = D
    elif F == 'F8':
        fobj = F8
        lb = -100
        ub = 100
        dim = D
    elif F == 'F9':
        fobj = F9
        lb = -100
        ub = 100
        dim = D
    elif F == 'F10':
        fobj = F10
        lb = -32.768
        ub = 32.768
        dim = D
    elif F == 'F11':
        fobj = F11
        lb = -100
        ub = 100
        dim = D
    elif F == 'F12':
        fobj = F12
        lb = -100
        ub = 100
        dim = D
    elif F == 'F13':
        fobj = F13
        lb = -600
        ub = 600
        dim = D
    elif F == 'F14':
        fobj = F14
        lb = -50
        ub = 50
        dim = D
    else:
        raise ValueError(f"Unsupported function: {F}")
    return lb, ub, dim, fobj


