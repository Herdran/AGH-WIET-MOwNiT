import numpy as np
import matplotlib.pyplot as plt


accuracy_q = []
accuracy_c = []

def f(x):
    return np.exp(-3 * np.sin(x))

def quadratic_spline(X, Y, bct):
    h = X[1] - X[0]

    matrix = np.zeros(shape=(n, n))

    matrix[0][0] = 1

    for i in range(1, n):
        matrix[i][i - 1] = 1
        matrix[i][i] = 1

    matrix2 = np.zeros(shape=(n, 1))

    for i in range(1, n - 1):
        matrix2[i] = 2 * (Y[i] - Y[i - 1]) / h

    result = np.linalg.inv(matrix).dot(matrix2)

    coefficients = np.zeros(shape=(3, n - 1))

    for i in range(n - 1):
        coefficients[0][i] = (result[i + 1] - result[i]) / (2 * h)
        coefficients[1][i] = result[i]
        coefficients[2][i] = Y[i]

    x = np.linspace(interval[0], interval[1], num=1000)
    y = []

    for xx in x:
        i = int((xx - X[0]) // h)
        if i == len(coefficients[0]):
            i = len(coefficients[0]) - 1
        y.append(coefficients[0][i] * (xx - X[i]) ** 2 + coefficients[1][i] * (xx - X[i]) + coefficients[2][i])
    return y


def cubic_spline(X, Y, bct):
    h = X[1] - X[0]
    matrix = np.zeros(shape=(n, n))

    matrix[0][0] = 1
    matrix[n - 1][n - 1] = 1

    for i in range(1, n - 1):
        matrix[i][i - 1] = 1
        matrix[i][i] = 4
        matrix[i][i + 1] = 1

    matrix2 = np.zeros(shape=(n, 1))

    for i in range(1, n - 1):
        matrix2[i] = (6 / h ** 2) * (Y[i + 1] - 2 * Y[i] + Y[i - 1])

    result = np.linalg.inv(matrix).dot(matrix2)
    coefficients = np.zeros(shape=(4, n - 1))

    for i in range(n - 1):
        coefficients[0][i] = (result[i + 1] - result[i]) / (6 * h)
        coefficients[1][i] = result[i] / 2
        coefficients[2][i] = (Y[i + 1] - Y[i]) / h - (result[i + 1] * h / 6) - (result[i] * h / 3)
        coefficients[3][i] = Y[i]

    x = np.linspace(interval[0], interval[1], num=1000)
    y = []

    for xx in x:
        i = int((xx - X[0]) // h)
        if i == len(coefficients[0]):
            i = len(coefficients[0]) - 1
        y.append(coefficients[0][i] * (xx - X[i]) ** 3 + coefficients[1][i] * (xx - X[i]) ** 2 + coefficients[2][i] * (
                xx - X[i]) + coefficients[3][i])
    return y


def get_accuracy(y, interpolated_y):
    points = len(y)
    acc1 = 0
    acc2 = 0

    for i in range(points):
        if acc1 < abs(y[i] - interpolated_y[i]):
            acc1 = abs(y[i] - interpolated_y[i])

    for i in range(points):
        acc2 += (y[i] - interpolated_y[i]) ** 2
    return acc1, acc2 / points


def draw_graph(x_eq, y_eq, func, title):
    x = np.linspace(interval[0], interval[1], num=1000)
    y = f(x)

    plt.figure(figsize=(11, 9))
    plt.title(title)
    plt.scatter(x_eq, y_eq, label="knots", color="orange", s=100)

    plt.plot(x, y, label="f", color="blue")

    plt.plot(x, quadratic_spline(x_eq, y_eq, func), label="Quadratic spline", color="green")
    plt.plot(x, cubic_spline(x_eq, y_eq, func), label="Cubic spline", color="red")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()
    # accuracy_q.append(get_accuracy(y, quadratic_spline(x_eq, y_eq, func)))
    # accuracy_c.append(get_accuracy(y, cubic_spline(x_eq, y_eq, func)))


def visualize(func):
    x_eq = np.linspace(interval[0], interval[1], num=n)
    y_eq = [f(x) for x in x_eq]

    draw_graph(x_eq, y_eq, func, func + " polynomial interpolation on " + str(n) + " nodes")


def visualise_natural():
    visualize("Natural")


def visualise_periodic():
    visualize("Periodic")


interval = [-2 * np.pi, 4 * np.pi]

n = 15
visualise_natural()
# visualise_periodic()


# for n in range(5, 52, 2):
#     visualise_natural()
#     visualise_periodic()


# np.savetxt("accuracy_q.txt", accuracy_q, delimiter =", ", fmt='%f')
# np.savetxt("accuracy_c.txt", accuracy_c, delimiter =", ", fmt='%f')