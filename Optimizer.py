import sympy as sp
import numpy as np
from tabulate import tabulate

class Optimizer:
    def __init__(self, obj_fn: str):
        # Create the objective function and obtain its variables
        f: sp.Function = sp.parse_expr(obj_fn)
        self.variables = f.free_symbols

        # Create partial derivatives (gradients) based on the function
        self.gradient = [sp.lambdify(var, f.diff(var), modules='numpy') for var in self.variables]

        # Create lambda-fied version of f so it can be used as an function
        self.f = sp.lambdify(self.variables, f, modules='numpy')

        # Default to steepest descent strategy
        self.strategy = 'steepest'

    '''
    Perform the optimization
    
    On every iteration, the gradient at the current location is calculated and used to determine
    the search direction. Then, using backtracking line search, a new location is calculated
    which is strictly better than the current location. This process continues until the norm
    of the gradient is smaller or equal to the stop condition.
    
    If the objective is convex, this method guarantees to find a solution with error no greater 
    than the stopping condition. If the objective is non-convex, it may instead find local optima.
    
    Parameters:
        starting_point : [float]
            An vector that defines the starting position of the search
            Must be of the same length as the variable vector
        stop_condition : float 
            Once the norm of the gradient at current location is lesser or equal to this number, 
            the optimization/search will terminate
        alpha: float
            A value used to find step sizes. Larger alpha results in longer steps and vice versa
            Accepted values: (0, 1), recommended values: (0.2, 0.5)
        beta: float
            Another value used to find step sizes. Larger value results in a finer search, and 
            vice versa
            Accepted values: (0, 1), recommended values: ~0.5
        search_strategy: str
            Defines the strategy to search for descent direction. Valid options are:
                steepest: use the negative of gradient, -∇f(x) as the next descent direction (default)
                newton: use the newton's method, -H^(-1)∇f(x), where H is the Hessian of the function
                
    Return:
        Outputs information about the function and step-by-step result to console
    '''
    def optimize(self, starting_point, stop_condition, alpha, beta, search_strategy='steepest'):
        pass
