from tabulate import tabulate

from optimizer import Optimizer


def main():
    # q1
    # objective = '100 * x1**2 + x2**2'
    # starting_points = [[10, 1], [1, 10], [5, 5], [10, 0], [0, 10]]
    # q2
    # objective = 'x1**2 + x2**2'
    # starting_points = [[10, 1], [1, 10], [5, 5], [10, 0], [0, 10]]
    # q3
    # objective = 'x**4 - 13*x**2 + 36'
    # starting_points = [[-2], [1.5], [0]]
    # q4
    # objective = 'x**3 - x**2'
    # starting_points = [[0.8], [0.1], [-2]]
    # q5
    objective = 'x1 ** 4 - 10*x1**2 + 5*x2**2 + x3**2 + x1 + x2 - x3'
    starting_points = [[10, 0, 0], [1, 2, 3], [5, 5, 5]]
    strategies = ['steepest', 'newton']
    alpha = 0.2
    beta = 0.5

    # Set up the problem
    problem = Optimizer(objective)

    # Run optimization with the specified parameters
    for strategy in strategies:
        for starting_point in starting_points:
            result, log = problem.optimize(starting_point, 10 ** -8, alpha, beta, search_strategy=strategy)
            file_name = f'q5-({starting_point[0]})-{alpha}-{beta}-{strategy}'
            save_output(result, problem.variables, log, file_name)


def print_output(result, variables):
    output_header = ['k', *variables, 'f(x)', 'norm', 't']
    print(tabulate(result, headers=output_header, tablefmt='fancy_grid'))


def save_output(result, variables, log,  file_name):
    output_header = ['k', *variables, 'f(x)', 'norm', 't']
    output_table = tabulate(result, headers=output_header, tablefmt='fancy_grid')

    with open('results/' + file_name, 'w', encoding='utf-8') as file:
        for entry in log:
            file.write(entry)
            file.write('\n')
        file.write(output_table)


if __name__ == '__main__':
    main()
