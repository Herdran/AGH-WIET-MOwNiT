from numpy import single, double, longdouble
import matplotlib.pyplot as plt


def func_float(k):
    k += 1
    print("Float")
    numbers = [single() for _ in range(k)]
    numbers[0] = single(11 / 2)
    numbers[1] = single(61 / 11)
    for i in range(2, k):
        numbers[i] = single(111 - single(1130 - single(3000 / numbers[i - 2])) / numbers[i - 1])
    print(numbers)
    print("x34 = ", numbers[34])
    return numbers


def func_double(k):
    k += 1
    print("Double")
    numbers = [double() for _ in range(k)]
    numbers[0] = double(11 / 2)
    numbers[1] = double(61 / 11)
    for i in range(2, k):
        numbers[i] = double(111 - double(1130 - double(3000 / numbers[i - 2])) / numbers[i - 1])
    print(numbers)
    print("x34 = ", numbers[34])
    return numbers


def func_longdouble(k):
    k += 1
    print("Long double")
    numbers = [longdouble() for _ in range(k)]
    numbers[0] = longdouble(11 / 2)
    numbers[1] = longdouble(61 / 11)
    for i in range(2, k):
        numbers[i] = longdouble(111 - longdouble(1130 - longdouble(3000 / numbers[i - 2])) / numbers[i - 1])
    print(numbers)
    print("x34 = ", numbers[34])
    return numbers


x = [_ for _ in range(35)]

plt.title('Float')
plt.scatter(x, func_float(34), s=10, c='red')
plt.grid()
plt.hlines(6, 0, 34, linestyles='dashed')
plt.hlines(100, 0, 34, linestyles='dashed')
plt.xlabel('k')
plt.ylabel('xk')
plt.show()
plt.clf()

plt.title('Double')
plt.scatter(x, func_double(34), s=10, c='red')
plt.grid()
plt.hlines(6, 0, 34, linestyles='dashed')
plt.hlines(100, 0, 34, linestyles='dashed')
plt.xlabel('k')
plt.ylabel('xk')
plt.show()
plt.clf()

plt.title('Longdouble')
plt.scatter(x, func_longdouble(34), s=10, c='red')
plt.grid()
plt.hlines(6, 0, 34, linestyles='dashed')
plt.hlines(100, 0, 34, linestyles='dashed')
plt.xlabel('k')
plt.ylabel('xk')
plt.show()
plt.clf()
