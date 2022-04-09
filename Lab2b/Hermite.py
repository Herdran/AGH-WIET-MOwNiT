import matplotlib.pyplot as plt
import numpy as np


accuracy_h_e = []
accuracy_h_c = []


def f(x):
    return np.exp(-3 * np.sin(x))


def f_der(x):
    return -3 * np.exp(-3 * np.sin(x)) * np.cos(x)


def hermite_interpolation(X, x, y):
    length = len(x)
    points = [[0 for _ in range(2 * length + 1)] for __ in range(2 * length + 1)]
    res = []

    for i in range(0, 2 * length, 2):
        points[i][0] = x[i // 2]
        points[i + 1][0] = x[i // 2]
        points[i][1] = y[i // 2]
        points[i + 1][1] = y[i // 2]

    for i in range(2, 2 * length + 1):
        for j in range(i - 1, 2 * length):
            if i == 2 and j % 2 == 1:
                points[j][i] = f_der(x[j // 2])
            else:
                points[j][i] = (points[j][i - 1] - points[j - 1][i - 1]) / (points[j][0] - points[(j - 1) - (i - 2)][0])

    for point in X:
        val = 0
        for i in range(0, 2 * length):
            factor = 1.
            j = 0
            while j < i:
                factor *= (point - x[j // 2])
                if j + 1 != i:
                    factor *= (point - x[j // 2])
                    j += 1
                j += 1
            val += factor * points[i][i + 1]
        res.append(val)
    return res


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


def draw_graph(x_eq, x_ch, y_eq, y_ch, func, title):
    x = np.linspace(interval[0], interval[1], num=1000)
    y = f(x)

    plt.figure(figsize=(12, 10))
    plt.title(title)
    plt.scatter(x_eq, y_eq, label="Equidistant", color="red", s=100)
    plt.scatter(x_ch, y_ch, label="Chebyshev", color="green", s=100)

    plt.plot(x, func(x, x_eq, y_eq), label="Equidistant", color="red")
    plt.plot(x, func(x, x_ch, y_ch), label="Chebyshev", color="green")

    plt.plot(x, y, label="f", color="blue")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()
    # accuracy_h_e.append(get_accuracy(y, func(x, x_eq, y_eq)))
    # accuracy_h_c.append(get_accuracy(y, func(x, x_ch, y_ch)))


def visualize(func, method):
    x_eq = np.linspace(interval[0], interval[1], num=n)
    x_ch = [0.5 * (interval[0] + interval[1]) + 0.5 * (interval[1] - interval[0]) * np.cos((2 * i - 1) / (2 * n) * np.pi) for i in range(1, n + 1)]
    y_eq = [f(x) for x in x_eq]
    y_ch = [f(x) for x in x_ch]

    draw_graph(x_eq, x_ch, y_eq, y_ch, func, method + "'s interpolation on " + str(n) + " nodes")


def visualise_hermite():
    visualize(hermite_interpolation, "Hermite")



interval = [-2 * np.pi, 4 * np.pi]

n = 4
visualise_hermite()


# for n in range(2, 31, 2):
#     visualise_hermite()


# np.savetxt("accuracy_h_e.txt", accuracy_h_e, delimiter =", ", fmt='%f')
# np.savetxt("accuracy_h_c.txt", accuracy_h_c, delimiter =", ", fmt='%f')
