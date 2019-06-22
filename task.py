def sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))


def get_theta(operation):
    import random
    if operation == 'or':
        th0 = random.randint(-15, -9)
        th1 = random.randint(18, 22)
        th2 = random.randint(18, 22)
        theta = [th0, th1, th2]

    elif operation == 'and':
        th0 = random.randint(-33, -29)
        th1 = random.randint(18, 22)
        th2 = random.randint(18, 22)
        theta = [th0, th1, th2]

    elif operation == 'not':
        th0 = random.randint(1, 5)
        th1 = random.randint(-40, -25)
        theta = [th0, th1]
    return theta


def perceptron_or_and(input_X, theta, bias_X=1):
    results = []
    for i in range(len(input_X)):
        result = sigmoid(bias_X * theta[0] + input_X[i][0] * theta[1] + input_X[i][1] * theta[2])
        if (result >= 0.9):
            result = 1
        else:
            result = 0
        results.append(result)
    return results


def perceptron_not(input_X, theta, bias_X=1):
    results = []
    for i in range(len(input_X)):
        result = sigmoid(bias_X * theta[0] + input_X[i] * theta[1])
        if (result >= 0.5):
            result = 1
        else:
            result = 0
        results.append(result)
    return results


def perceptron_xnor(input_X, theta_or, theta_and, theta_not):
    """
    XNOR
    1) x1 & x2  (a1)
    2) ¬ x1 & ¬ x2  (a2)
    3) x1 || x2  (a1 || a2)
    """
    new_input_X = []
    a1 = perceptron_or_and(input_X, theta_and)
    for item in input_X:
        new_input_X.append(tuple(perceptron_not(list(item), theta_not)))
    a2 = perceptron_or_and(new_input_X, theta_and)
    a1a2 = list(zip(a1, a2))
    result = perceptron_or_and(a1a2, theta_or)
    return result, a1, a2


def perceptron_xor(input_X, theta_or, theta_and, theta_not):
    """
    XOR
    1) x1 || x2  (a1)
    2) ¬ x1 || ¬ x2  (a2)
    3) x1 & x2  (a1 & a2)
    """
    new_input_X = []
    a1 = perceptron_or_and(input_X, theta_or)
    for item in input_X:
        new_input_X.append(tuple(perceptron_not(list(item), theta_not)))
    a2 = perceptron_or_and(new_input_X, theta_or)
    a1a2 = list(zip(a1, a2))
    result = perceptron_or_and(a1a2, theta_and)
    return result, a1, a2


if __name__ == "__main__":
    from prettytable import PrettyTable

    input_X = [(0, 0), (0, 1), (1, 0), (1, 1)]
    theta_or = get_theta('or')
    theta_and = get_theta('and')
    theta_not = get_theta('not')

    z = PrettyTable()
    column_names = ["x", "h(x)"]
    z.add_column("x", ["0", "1"])
    z.add_column("h(x)", perceptron_not([0, 1], theta_not))
    print(f'\nNOT\nθ = {theta_not}')
    print(z)

    s = PrettyTable()
    column_names2 = ["x1", "x2", "h(x)"]
    s.add_column("x1", ["0", "0", "1", "1"])
    s.add_column("x2", ["0", "1", "0", "1"])
    s.add_column("h(x)", perceptron_or_and(input_X, theta_or))
    print(f'\nOR\nθ = {theta_or}')
    print(s)

    z = PrettyTable()
    column_names3 = ["x1", "x2", "h(x)"]
    z.add_column("x1", ["0", "0", "1", "1"])
    z.add_column("x2", ["0", "1", "0", "1"])
    z.add_column("h(x)", perceptron_or_and(input_X, theta_and))
    print(f'\nAND\nθ = {theta_and}')
    print(z)

    x = PrettyTable()
    column_names4 = ["x1", "x2", "a1", "a2", "h(x)"]
    x.add_column("x1", ["0", "0", "1", "1"])
    x.add_column("x2", ["0", "1", "0", "1"])
    x.add_column("a1", perceptron_xor(input_X, theta_or, theta_and, theta_not)[1])
    x.add_column("a2", perceptron_xor(input_X, theta_or, theta_and, theta_not)[2])
    x.add_column("h(x)", perceptron_xor(input_X, theta_or, theta_and, theta_not)[0])
    print('\nXOR')
    print(x)

    y = PrettyTable()
    column_names5 = ["x1", "x2", "a1", "a2", "h(x)"]
    y.add_column("x1", ["0", "0", "1", "1"])
    y.add_column("x2", ["0", "1", "0", "1"])
    y.add_column("a1", perceptron_xnor(input_X, theta_or, theta_and, theta_not)[1])
    y.add_column("a2", perceptron_xnor(input_X, theta_or, theta_and, theta_not)[2])
    y.add_column("h(x)", perceptron_xnor(input_X, theta_or, theta_and, theta_not)[0])
    print('\nXNOR')
    print(y)
