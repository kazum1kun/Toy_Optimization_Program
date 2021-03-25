import sympy as sp
import numpy as np
from tabulate import tabulate
from optimizer import Optimizer


def main():
    # Set up the problem
    problem = Optimizer('100 * x1**2 + x2**2')

    # Run optimization with the specified parameters
    result = problem.optimize([10, 1], 10**-8, 0.2, 0.5)

    print_output(result, problem.variables)


def print_output(result, variables):
    output_header = ['k', *variables, 'f(x)', 'norm', 't']
    print(tabulate(result, headers=output_header, tablefmt='fancy_grid'))


if __name__ == '__main__':
    main()
