import matplotlib.pyplot as plt
from typing import Tuple, Union
import numpy as np
import random


class Regression:
    def __new__(cls, *args, **kwargs):
        raise RuntimeError("Regression class is static class")

    @staticmethod
    def rand_in_range(rand_range: Union[float, Tuple[float, float]] = 1.0) -> float:
        if isinstance(rand_range, float):
            return random.uniform(-0.5 * rand_range, 0.5 * rand_range)
        if isinstance(rand_range, tuple):
            return random.uniform(rand_range[0], rand_range[1])
        return random.uniform(-0.5, 0.5)

    @staticmethod
    def test_data_along_line(k: float = 1.0, b: float = 0.1, arg_range: float = 1.0,
                             rand_range: float = 0.05, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерирует линию вида y = k * x + b + dy, где dy - аддитивный шум с амплитудой half_disp
        :param k: наклон линии
        :param b: смещение по y
        :param arg_range: диапазон аргумента от 0 до arg_range
        :param rand_range: диапазон шума данных
        :param n_points: количество точек
        :return: кортеж значений по x и y
        """
        x_step = arg_range / (n_points - 1)
        return np.array([i * x_step for i in range(n_points)]), \
            np.array([i * x_step * k + b + Regression.rand_in_range(rand_range) for i in range(n_points)]) ** 2

    @staticmethod
    def second_order_surface_2d(surf_params:
    Tuple[float, float, float, float, float, float] = (1.0, -2.0, 3.0, 1.0, 2.0, -3.0),
                                args_range: float = 1.0, rand_range: float = .1, n_points: int = 1000) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Генерирует набор тестовых данных около поверхности второго порядка.
        Уравнение поверхности:
        z(x,y) = a * x^2 + x * y * b + c * y^2 + d * x + e * y + f
        :param surf_params: 
        :param surf_params [a, b, c, d, e, f]:
        :param args_range x in [x0, x1], y in [y0, y1]:
        :param rand_range:
        :param n_points:
        :return:
        """
        x = np.array([Regression.rand_in_range(args_range) for _ in range(n_points)])
        y = np.array([Regression.rand_in_range(args_range) for _ in range(n_points)])
        dz = np.array([surf_params[5] + Regression.rand_in_range(rand_range) for _ in range(n_points)])
        return x, y, surf_params[0] * x * x + surf_params[1] * y * x + surf_params[2] * y * y + \
                     surf_params[3] * x + surf_params[4] * y + dz

    @staticmethod
    def test_data_2d(kx: float = -2.0, ky: float = 2.0, b: float = 12.0, args_range: float = 1.0,
                     rand_range: float = 1.0, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Генерирует плоскость вида z = kx*x + ky*x + b + dz, где dz - аддитивный шум в диапазоне rand_range
        :param kx: наклон плоскости по x
        :param ky: наклон плоскости по y
        :param b: смещение по z
        :param args_range: диапазон аргументов по кажой из осей от 0 до args_range
        :param rand_range: диапазон шума данных
        :param n_points: количество точек
        :returns: кортеж значенией по x, y и z
        """
        x = np.array([Regression.rand_in_range(args_range) for _ in range(n_points)])
        y = np.array([Regression.rand_in_range(args_range) for _ in range(n_points)])
        dz = np.array([b + Regression.rand_in_range(rand_range) for _ in range(n_points)])
        return x, y, x * kx + y * ky + dz

    @staticmethod
    def test_data_nd(surf_settings: np.ndarray = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 12.0]), args_range: float = 1.0,
                     rand_range: float = 0.1, n_points: int = 125) -> np.ndarray:
        """
        Генерирует плоскость вида z = k_0*x_0 + k_1*x_1...,k_n*x_n + d + dz, где dz - аддитивный шум в диапазоне rand_range
        :param surf_settings: параметры плоскости в виде k_0,k_1,...,k_n,d
        :param args_range: диапазон аргументов по кажой из осей от 0 до args_range
        :param n_points: количество точек
        :param rand_range: диапазон шума данных
        :returns: массив из строк вида x_0, x_1,...,x_n, f(x_0, x_1,...,x_n)
        """
        n_dims = surf_settings.size - 1
        data = np.zeros((n_points, n_dims + 1,), dtype=float)
        for i in range(n_dims):
            data[:, i] = np.array([Regression.rand_in_range(args_range) for _ in range(n_points)])
            data[:, n_dims] += surf_settings[i] * data[:, i]
        dz = np.array([surf_settings[n_dims] + Regression.rand_in_range(rand_range) for _ in range(n_points)])
        data[:, n_dims] += dz
        return data

    @staticmethod
    def distance_sum(x: np.ndarray, y: np.ndarray, k: float, b: float) -> float:
        """
        Вычисляет сумму квадратов расстояний от набора точек до линии вида y = k*x + b при фиксированных k и b
        по формуле: F(k, b) = (Σ(yi -(k * xi + b))^2)^0.5 (суммирование по i)
        :param x: массив значений по x
        :param y: массив значений по y
        :param k: значение параметра k (наклон)
        :param b: значение параметра b (смещение)
        :returns: F(k, b) = (Σ(yi -(k * xi + b))^2)^0.5
        """
        return np.sqrt(np.power((y - x * k + b), 2.0).sum())

    @staticmethod
    def distance_field(x: np.ndarray, y: np.ndarray, k: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Вычисляет сумму квадратов расстояний от набора точек до линии вида y = k*x + b, где k и b являются диапазонами
        значений. Формула расстояния для j-ого значения из набора k и k-ого значения из набора b:
        F(k_j, b_k) = (Σ(yi -(k_j * xi + b_k))^2)^0.5 (суммирование по i)
        :param x: массив значений по x
        :param y: массив значений по y
        :param k: массив значений параметра k (наклоны)
        :param b: массив значений параметра b (смещения)
        :returns: поле расстояний вида F(k, b) = (Σ(yi -(k * xi + b))^2)^0.5 (суммирование по i)
        """
        return np.array([[Regression.distance_sum(x, y, k_i, b_i) for k_i in k.flat] for b_i in b.flat])

    @staticmethod
    def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        k = (Σxi*yi - Σxi*Σyi / n) / (Σxi^2 - (Σxi)^2 / n)\n
        b = (Σyi - k * Σxi) /n\n
        :param x: массив значений по x
        :param y: массив значений по y
        :returns: возвращает пару (k, b), которая является решением задачи (Σ(yi -(k * xi + b))^2)->min
        """
        n = 1.0 / x.size
        x_s = x.sum()
        y_s = y.sum()
        xy_s = np.dot(x, y)
        xx_s = np.dot(x, x)
        k = (xy_s - x_s * y_s * n) / (xx_s - x_s * x_s * n)
        b = (y_s - k * x_s) * n
        return k, b

    @staticmethod
    def bi_linear_regression(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float, float]:
        """
        Hesse matrix:\n
                       | Σ xi^2;  Σ xi*yi; Σ xi |\n
        H(kx, ky, b) = | Σ xi*yi; Σ yi^2;  Σ yi |\n
                       | Σ xi;    Σ yi;    n    |\n
        ====================================================================================================================\n
                          | Σ-zi*xi + ky*xi*yi + kx*xi^2 + xi*b |\n
        grad(kx, ky, b) = | Σ-zi*yi + ky*yi^2 + kx*xi*yi + b*yi |\n
                          | Σ-zi + yi*ky + xi*kx                |\n
        ====================================================================================================================\n
        Окончательно решение:\n
        |kx|   |1|\n
        |ky| = |1| -  H(1, 1, 0)^-1 * grad(1, 1, 0)\n
        | b|   |0|\n

        :param x: массив значений по x
        :param y: массив значений по y
        :param z: массив значений по z
        :returns: возвращает тройку (kx, ky, b), которая является решением задачи (Σ(zi - (yi * ky + xi * kx + b))^2)->min
        """
        n = x.size
        xx_s = np.dot(x, x)
        xy_s = np.dot(x, y)
        yy_s = np.dot(y, y)
        x_s = x.sum()
        y_s = y.sum()
        H = np.array(((xx_s, xy_s, x_s),
                      (xy_s, yy_s, y_s),
                      (x_s, y_s, n)))

        zx_s = np.dot(x, z)
        zy_s = np.dot(y, z)
        z_s = z.sum()

        grad = np.array((-zx_s + xy_s + xx_s,
                         -zy_s + yy_s + xy_s,
                         -z_s  + yy_s + x_s))
        vec = np.array([1.0, 1.0, 0.0]) - np.linalg.inv(H) @ grad
        return vec[0], vec[1], vec[2]

    @staticmethod
    def n_linear_regression(data_rows: np.ndarray) -> np.ndarray:
        """
        H_ij = Σx_i * x_j, i in [0, rows - 1] , j in [0, rows - 1]
        H_ij = Σx_i, j = rows i in [rows, :]
        H_ij = Σx_j, j in [:, rows], i = rows

               | Σkx * xi^2    + Σky * xi * yi + b * Σxi - Σzi * xi|\n
        grad = | Σkx * xi * yi + Σky * yi^2    + b * Σyi - Σzi * yi|\n
               | Σyi * ky      + Σxi * kx                - Σzi     |\n

        x_0 = [1,...1, 0] =>

               | Σ xi^2    + Σ xi * yi - Σzi * xi|\n
        grad = | Σ xi * yi + Σ yi^2    - Σzi * yi|\n
               | Σxi       + Σ yi      - Σzi     |\n

        :param data_rows:  состоит из строк вида: [x_0,x_1,...,x_n, f(x_0,x_1,...,x_n)]
        :return:
        """
        n_points, n_args = data_rows.shape
        x_0 = np.ones(shape=(n_args, 1))
        x_0[-1] = 0
        H = np.zeros(shape=(n_args, n_args))

        for row in range(n_args - 1):
            H[-1, row] = H[row, -1] = data_rows[:, row].sum()
            for col in range(row + 1):
                H[row, col] = H[col, row] = np.dot(data_rows[:, row], data_rows[:, col])
        H[-1, -1] = n_points

        grad = np.zeros(shape=(n_args, 1))
        for row in range(n_args - 1):
            grad[row] = H[row, :-1].sum() - np.dot(data_rows[:, -1], data_rows[:, row])
        grad[-1] = H[-1, :-1].sum() - data_rows[:, -1].sum()
        return x_0 - np.linalg.inv(H) @ grad

    @staticmethod
    def poly_regression(x: np.ndarray, y: np.ndarray, order: int = 5) -> np.ndarray:
        """
        Полином: y = Σ_j x^j * bj\n
        Отклонение: ei =  yi - Σ_j xi^j * bj\n
        Минимизируем: Σ_i(yi - Σ_j xi^j * bj)^2 -> min\n
        Σ_i(yi - Σ_j xi^j * bj)^2 = Σ_iyi^2 - 2 * yi * Σ_j xi^j * bj +(Σ_j xi^j * bj)^2\n
        условие минимума:\n d/dbj Σ_i ei = d/dbj (Σ_i yi^2 - 2 * yi * Σ_j xi^j * bj +(Σ_j xi^j * bj)^2) = 0\n
        :param x: массив значений по x
        :param y: массив значений по y
        :param order: порядок полинома
        :return: набор коэффициентов bi полинома y = Σx^i*bi
        """
        a_matrix = np.zeros((order, order), dtype=float)
        c_matrix = np.zeros(order, dtype=float)
        x_row = np.ones_like(x)
        for row in range(order):
            x_row = x_row if row == 0 else x_row * x
            c_matrix[row] = (x_row * y).sum()
            x_col = np.ones_like(x)
            for col in range(row + 1):
                x_col = x_col if col == 0 else x_col * x
                a_matrix[col][row] = a_matrix[row][col] = np.dot(x_col, x_row)

        return np.dot(np.linalg.inv(a_matrix), c_matrix)

    @staticmethod
    def polynom(x: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        :param x: массив значений по x\n
        :param b: массив коэффициентов полинома\n
        :returns: возвращает полином yi = Σxi^j*bj\n
        """
        result = b[0] + b[1] * x
        x1 = x.copy()
        for i in range(2, b.size):
            x1 *= x
            result += b[i] * x1
        return result

    @staticmethod
    def quadratic_regression_2d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        b = [x * x, x * y, y * y, x, y, np.array([1.0])]

        a_matrix = np.zeros((6, 6), dtype=float)

        b_matrix = np.zeros(6, dtype=float)

        for row in range(6):
            b_matrix[row] = np.sum(b[row] * z)
            for col in range(row + 1):
                a_matrix[col, row] = a_matrix[row][col] = np.sum(b[row] * b[col])

        a_matrix[-1, -1] = x.size

        return np.dot(np.linalg.inv(a_matrix), b_matrix)

    @staticmethod
    def distance_field_example():
        print("distance field test:")
        x, y = Regression.test_data_along_line()  # Посчитать тестовыe x и y используя функцию test_data
        k_, b_ = Regression.linear_regression(x, y)  # Задать интересующие нас диапазоны k и b
        print(f"y(x) = {k_:1.5} * x + {b_:1.5}\n")
        k = np.linspace(-2.0, 2.0, 128, dtype=float)
        b = np.linspace(-2.0, 2.0, 128, dtype=float)
        z = Regression.distance_field(x, y, k, b)
        plt.imshow(z, extent=[k.min(), k.max(), b.min(), b.max()])
        plt.plot(k_, b_, 'r*')
        plt.xlabel("k")
        plt.ylabel("b")
        plt.grid(True)
        plt.show()

    @staticmethod
    def linear_reg_example():
        print("linear reg test:")
        x, y = Regression.test_data_along_line()  # Посчитать тестовыe x и y используя функцию test_data
        k, b = Regression.linear_regression(x, y)  # Получить с помошью linear_regression значения k и b
        print(f"y(x) = {k:1.3} * x + {b}")
        xs = np.linspace(np.min(x), np.max(x), 128)  # Вывести на графике x и y в виде массива точек и построить
        ys = xs * k + b  # регрессионная прямая вида: y = k*x + b
        plt.plot(x, y, '.r')
        plt.plot(xs, ys, 'g')
        plt.show()

    @staticmethod
    def bi_linear_reg_example():
        x, y, z = Regression.test_data_2d(rand_range=10, args_range=10000)
        kx, ky, b = Regression.bi_linear_regression(x, y, z)
        print("\nbi linear regression test:")
        print(f"z(x, y) = {kx:1.3} * x + {ky:1.3} * x + {b}")

        z = kx * x + ky * y + b
        fig = plt.figure()
        x_, y_ = np.meshgrid(np.linspace(np.min(x), np.max(x), 100), np.linspace(np.min(y), np.max(y), 100))
        z_ = kx * x_ + ky * y_ + b
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(x_, y_, z_)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.scatter(x, y, z, color="red")
        plt.title("Bi linear regression")
        plt.show()

    @staticmethod
    def poly_reg_example():
        """
        Функция проверки работы метода полиномиальной регрессии:\n
        1) Посчитать тестовыe x, y используя функцию test_data\n
        2) Посчитать набор коэффициентов bi полинома y = Σx^i*bi используя функцию poly_regression\n
        3) Вывести на графике x и y в виде массива точек и построить\n
           регрессионную кривую. Для построения кривой использовать метод polynom\n
        :return:
        """
        x, y = Regression.test_data_along_line()
        coefficients = Regression.poly_regression(x, y)
        y_ = Regression.polynom(x, coefficients)
        print("\nn poly regression test:")
        print(f"y(x) = {' + '.join(f'{coefficients[i]:.4} * x^{i}' for i in range(coefficients.size))}\n")
        plt.scatter(x, y)
        plt.plot(x, y_, color="red")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Poly regression")
        plt.show()

    @staticmethod
    def n_linear_reg_example():
        print("\nn linear regression test:")
        data = Regression.test_data_nd()
        coefficients = Regression.n_linear_regression(data)
        print(' + '.join(f'X_{i} * {v:.4f}' for i, v in enumerate(coefficients.flat)))

    @staticmethod
    def quadratic_reg_example():
        print('2d quadratic regression test:')
        x, y, z = Regression.second_order_surface_2d()
        coefficients = Regression.quadratic_regression_2d(x, y, z)
        print(
            f"z(x, y) = {coefficients[0]:1.3} * x^2 + {coefficients[1]:1.3} * x * y + {coefficients[2]:1.3} * y^2 + {coefficients[3]:1.3} * x + {coefficients[4]:1.3} * y + {coefficients[5]:1.3}")
        z = coefficients[0] * x * x + coefficients[1] * x * y + coefficients[2] * y * y + coefficients[3] * x + \
            coefficients[4] * y + coefficients[5]
        bounds_x, bounds_y = (x.max(), x.min()), (y.max(), y.min())
        fig = plt.figure()
        xs, ys = np.meshgrid(np.linspace(*bounds_x, 100), np.linspace(*bounds_y, 100))
        zs = coefficients[0] * xs * xs + coefficients[1] * xs * ys + coefficients[2] * ys * ys + coefficients[3] * xs + \
             coefficients[4] * ys + coefficients[5]
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(xs, ys, zs)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.scatter(x, y, z, color="red")
        plt.title("2d quadratic regression")
        plt.show()


if __name__ == "__main__":
    Regression.distance_field_example()
    Regression.linear_reg_example()
    Regression.bi_linear_reg_example()
    Regression.poly_reg_example()
    Regression.n_linear_reg_example()
    Regression.quadratic_reg_example()
