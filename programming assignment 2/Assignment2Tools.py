import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

def entropy_func(theta):
    return np.sum(theta*np.log(np.clip(theta+1e-12,1e-12, 1)))
    
def entropy_func_gradient(theta):
    return np.log(np.clip(theta+1e-12, 1e-12, 1))+1

def prob_vector_generator(N, mu, stddev):
    
    bounds = Bounds(np.zeros(N+1), np.ones(N+1))
    
    A = np.zeros((3,N+1))
    A[0,:] = 1
    A[1,:] = np.arange(N+1)
    A[2,:] = np.arange(N+1)**2
    
    b_lb = np.array([1,mu,mu**2+stddev**2])
    b_ub = np.array([1,mu,mu**2+stddev**2])
    
    linear_constraint = LinearConstraint(A, b_lb, b_ub)
    
    x_start = (1/(N+1))*np.ones(N+1)
    result = minimize(entropy_func, x_start, method='slsqp', jac=entropy_func_gradient, constraints=linear_constraint, bounds=bounds)
    
    theta = np.clip(result.x, 0, 1)
    theta = theta/np.sum(theta)
    
    return theta