import sympy as sp
import numpy as np
from tabulate import tabulate


def main():
    output_data = []
    starting_point = np.array([10, 1])
    stop_condition = 10 ** -6
    # Backtracking search params
    alpha = 0.5
    beta = 0.8

    # Create two symbols (variables)
    x1, x2 = sp.symbols('x1, x2', real=True)

    # Create objective function
    f = 100 * x1 ** 2 + x2 ** 2

    # Create partial derivatives (gradient)
    gradient = []
    for var in [x1, x2]:
        # Create a partial dir for each variable
        partial_dir = f.diff(var)
        partial_dir = sp.lambdify(var, partial_dir)
        gradient.append(partial_dir)

    f = sp.lambdify([x1, x2], f, modules="numpy")

    # Gradient that's evaluated at a particular (x1, x2)
    gradient_x = np.array([gradient[0](starting_point[0]), gradient[1](starting_point[1])])
    # Calculate the norm of the function
    nabla_norm = gradient_x.dot(gradient_x)
    x = starting_point

    itr = 0
    output_data.append([itr, x[0], x[1], f(*x), nabla_norm, np.inf])

    while nabla_norm > stop_condition:
        itr += 1
        # Update the search direction - steepest descent
        search_dir = -gradient_x
        # Calculate step size - Armijo rule
        t = 1
        # While the search function value lies above the tangent line (of factor alpha), make t smaller
        while f(*(x + t * search_dir)) > f(*x) + alpha * t * gradient_x.dot(search_dir):
            t = beta * t
        x = x + t * search_dir

        # Gradient that's evaluated at a particular (x1, x2)
        gradient_x = np.array([gradient[0](x[0]), gradient[1](x[1])])
        # Calculate the norm of the function
        nabla_norm = gradient_x.dot(gradient_x)

        output_data.append([itr, x[0], x[1], f(*x), nabla_norm, t])

    output_header = ['k', 'x1', 'x2', 'f(x)', 'norm', 't']

    print(tabulate(output_data, headers=output_header))


if __name__ == '__main__':
    main()
