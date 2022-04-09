import matplotlib.pyplot as plt
import numpy as np


accuracy_n_e = []
accuracy_n_c = []
accuracy_l_e = []
accuracy_l_c = []


def f(x):
    return np.exp(-3 * np.sin(1 * x))


def lagrange_interpolation(X, x, y):
    res = 0
    for i in range(n):
        p = 1
        for j in range(n):
            if i != j:
                p = p * (X - x[j]) / (x[i] - x[j])
        res = res + p * y[i]
    return res


def newton_interpolation(X, x, y):
    x1 = np.copy(x)
    a = np.copy(y)

    for i in range(1, n):
        a[i:n] = (a[i:n] - a[i - 1]) / (x1[i:n] - x1[i - 1])

    res = a[n - 1]

    for k in range(1, n):
        res = a[n - 1 - k] + (X - x[n - 1 - k]) * res
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
    # if func == newton_interpolation:
    #     accuracy_n_e.append(get_accuracy(y, func(x, x_eq, y_eq)))
    #     accuracy_n_c.append(get_accuracy(y, func(x, x_ch, y_ch)))
    # else:
    #     accuracy_l_e.append(get_accuracy(y, func(x, x_eq, y_eq)))
    #     accuracy_l_c.append(get_accuracy(y, func(x, x_ch, y_ch)))


def visualize(func, method):
    x_eq = [interval[0] + (interval[1] - interval[0]) / (n - 1) * i for i in range(n)]
    x_ch = [0.5 * (interval[0] + interval[1]) + 0.5 * (interval[1] - interval[0]) * np.cos((2 * i - 1) / (2 * n) * np.pi) for
            i in range(1, n + 1)]
    y_eq = [f(x) for x in x_eq]
    y_ch = [f(x) for x in x_ch]

    draw_graph(x_eq, x_ch, y_eq, y_ch, func, method + "'s interpolation on " + str(n) + " nodes")


def visualise_newton():
    visualize(newton_interpolation, "Newton")


def visualise_lagrange():
    visualize(lagrange_interpolation, "Lagrange")


interval = [-2 * np.pi, 4 * np.pi]

# n = 50
# visualise_lagrange()
# visualise_newton()

for n in range(2, 51, 2):
    visualise_newton()
    visualise_lagrange()


# np.savetxt("accuracy_n_e.txt", accuracy_n_e, delimiter =", ", fmt='%f')
# np.savetxt("accuracy_n_c.txt", accuracy_n_c, delimiter =", ", fmt='%f')
# np.savetxt("accuracy_l_e.txt", accuracy_l_e, delimiter =", ", fmt='%f')
# np.savetxt("accuracy_l_c.txt", accuracy_l_c, delimiter =", ", fmt='%f')
