import os

from tabulate import tabulate

from optimizer import Optimizer


def main():
    objective = '100 * x1**2 + x2**2'
    starting_points = [[10, 1], [1, 10], [5, 5], [10, 0], [0, 10]]
    strategies = ['steepest', 'newton']
    alpha = 0.2
    beta = 0.5

    # Set up the problem
    problem = Optimizer(objective)

    # Run optimization with the specified parameters
    for strategy in strategies:
        for starting_point in starting_points:
            result, log = problem.optimize(starting_point, 10 ** -8, alpha, beta, search_strategy=strategy)
            file_name = f'q1-({starting_point[0]}, {starting_point[1]})-{alpha}-{beta}-{strategy}'
            save_output(result, problem.variables, log, file_name)


def print_output(result, variables):
    output_header = ['k', *variables, 'f(x)', 'norm', 't']
    print(tabulate(result, headers=output_header, tablefmt='fancy_grid'))


def save_output(result, variables, log,  file_name):
    output_header = ['k', *variables, 'f(x)', 'norm', 't']
    output_table = tabulate(result, headers=output_header, tablefmt='fancy_grid')

    with open('out/' + file_name, 'w', encoding='utf-8') as file:
        for entry in log:
            file.write(entry)
            file.write('\n')
        file.write(output_table)


if __name__ == '__main__':
    main()
