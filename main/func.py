import numpy as np 

def sphere(array):
    return np.sum(np.square(array))

def rastrign(array, N):
    return 10*N + np.sum(np.square(array)-10*np.cos(2*np.pi*array))

def rosenbrock(array):
    return np.sum(100*np.square(np.diff(array))+np.square(1-array))

def griewank(array, N):
    return 1+np.sum(np.square(array))/4000-np.prod(np.cos(x/np.sqrt(np.arange(1, N+1))))

def alpine(array):
    return np.sum(np.abs(x*np.sin(x)+0.1*x))

def minima(array):
    return np.sum(np.square(np.square(x))-16*np.square(x)+5*x)
