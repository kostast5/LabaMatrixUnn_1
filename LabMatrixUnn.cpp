#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <locale>
#include <algorithm>


/* ----------Исключения---------- */
class MatrixError : public std::runtime_error {
public:
    explicit MatrixError(const std::string& message) : std::runtime_error(message) {}
};

class OutMatrixError : public MatrixError {
public:
    explicit OutMatrixError(const std::string& message) : MatrixError(message) {}
};

class MathMatrixError : public MatrixError {
public:
    explicit MathMatrixError(const std::string& message) : MatrixError(message) {}
};

/* ----------Класс матрицы---------- */
template <typename T>
class Matrix : public std::vector<std::vector<T>> {
private:
    size_t rows_cnt;
    size_t cols_cnt;
public:
    Matrix() : rows_cnt(0), cols_cnt(0) {}

    Matrix(size_t r, size_t c, const T& value = T()) {
        rows_cnt = r;
        cols_cnt = c;
        this->assign(r, std::vector<T>(c, value));
    }

    size_t get_rows() const { return rows_cnt; }
    size_t get_cols() const { return cols_cnt; }

    T& at(size_t r, size_t c) {
        if (r >= rows_cnt || c >= cols_cnt)
            throw OutMatrixError("Выход за границы матрицы!");
        return (*this)[r][c];
    }

    T& operator()(size_t r, size_t c) { return at(r, c); }

    std::vector<T>& operator[](size_t r) {
        return std::vector<std::vector<T>>::operator[](r);
    }

    void resize(size_t r, size_t c) {
        rows_cnt = r;
        cols_cnt = c;
        this->assign(r, std::vector<T>(c));
    }
};

/* ----------Решение методом Гаусса---------- */
template <typename T>
std::vector<T> solve_gauss(Matrix<T> A, std::vector<T> b) {
    size_t n = A.get_rows();

    /* Проверка на пере/недоопределённость*/
    if (n != A.get_cols())
        throw MathMatrixError("Система не является квадратной (переопределена или недоопределена)");
    if (b.size() != n)
        throw MathMatrixError("Размер вектора правой части не совпадает с размером матрицы");
    if (n == 0)
        return std::vector<T>();

    const T EPS = static_cast<T>(1e-9);

    std::vector<size_t> pivot_cols;   /* ведущие столбцы*/
    size_t row = 0;                   /* текущая строка для исключения*/

    for (size_t col = 0; col < n && row < n; ++col) {
        size_t pivot_row = row;
        T max_val = std::abs(A[row][col]);

        for (size_t i = row + 1; i < n; ++i) {
            T val = std::abs(A[i][col]);
            if (val > max_val) {
                max_val = val;
                pivot_row = i;
            }
        }

        if (max_val < EPS) {
            continue;   /* столбец col не ведущий*/
        }

        if (pivot_row != row) {
            std::swap(A[row], A[pivot_row]);
            std::swap(b[row], b[pivot_row]);
        }

        /* Запоминаем, что столбец col – ведущий на строке row*/
        pivot_cols.push_back(col);

        /* Исключаем переменную в нижележащих строках*/
        for (size_t i = row + 1; i < n; ++i) {
            T factor = A[i][col] / A[row][col];
            for (size_t j = col; j < n; ++j) {
                A[i][j] -= factor * A[row][j];
            }
            b[i] -= factor * b[row];
        }
        ++row;
    }

    size_t rank = pivot_cols.size();  /* ранг матрицы */

    /* Проверка на несовместность:*/
    for (size_t i = rank; i < n; ++i) {
        if (std::abs(b[i]) > EPS) {
            throw MathMatrixError("Система несовместна (нет решений).");
        }
    }

    if (rank < n) {
        std::cout << "\nСистема имеет бесконечно много решений (ранг " << rank
            << " < " << n << ").\n";

        std::vector<T> x_base(n, T(0));
        /* Обратный ход */
        for (size_t p = rank; p > 0; --p) {
            size_t k = p - 1;
            size_t col = pivot_cols[k]; /* ведущий столбец */
            size_t r = k;
            T sum = b[r];
            for (size_t j = col + 1; j < n; ++j) {
                sum -= A[r][j] * x_base[j];
            }
            x_base[col] = sum / A[r][col];
        }
        std::cout << "Базисное решение (свободные переменные = 0):\n";
        for (size_t i = 0; i < n; ++i)
            std::cout << "x" << i + 1 << " = " << x_base[i] << "\n";

        std::vector<size_t> free_cols;
        for (size_t col = 0; col < n; ++col) {
            if (std::find(pivot_cols.begin(), pivot_cols.end(), col) == pivot_cols.end())
                free_cols.push_back(col);
        }

        for (size_t f : free_cols) {
            std::vector<T> d(n, T(0));
            d[f] = T(1);
            for (size_t p = rank; p > 0; --p) {
                size_t k = p - 1;
                size_t col = pivot_cols[k];
                size_t r = k;
                T sum = T(0);
                for (size_t j = col + 1; j < n; ++j) {
                    sum -= A[r][j] * d[j];
                }
                d[col] = sum / A[r][col];
            }
            std::vector<T> x1(n);
            for (size_t i = 0; i < n; ++i)
                x1[i] = x_base[i] + d[i];
            std::cout << "x" << f + 1 << " = 1\n";
            for (size_t i = 0; i < n; ++i)
                std::cout << "x" << i + 1 << " = " << x1[i] << "\n";

            std::vector<T> x2(n);
            for (size_t i = 0; i < n; ++i)
                x2[i] = x_base[i] - d[i];
            std::cout << "x" << f + 1 << " = -1\n";
            for (size_t i = 0; i < n; ++i)
                std::cout << "x" << i + 1 << " = " << x2[i] << "\n";
        }

        return x_base;
    }

    std::vector<T> x(n);
    for (size_t i = n; i > 0; --i) {
        size_t k = i - 1;
        T sum = b[k];
        for (size_t j = k + 1; j < n; ++j) {
            sum -= A[k][j] * x[j];
        }
        if (std::abs(A[k][k]) < EPS) {
            throw MathMatrixError("Нулевой ведущий элемент на обратном ходе (не должно происходить).");
        }
        x[k] = sum / A[k][k];
    }
    return x;
}

/* ----------Главная программа---------- */
int main() {
    std::setlocale(LC_ALL, "Russian");

    try {
        size_t n;
        std::cout << "Введите размерность квадратной матрицы A|b: ";
        if (!(std::cin >> n)) return 0;

        Matrix<double> A(n, n);
        std::cout << "Введите элементы матрицы A построчно:\n";
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                std::cin >> A(i, j);

        std::vector<double> b(n);
        std::cout << "Введите элементы вектора b:\n";
        for (size_t i = 0; i < n; ++i)
            std::cin >> b[i];

        std::vector<double> x = solve_gauss(A, b);

        if (!x.empty() && A.get_rows() == A.get_cols()) {
            std::cout << "Единственное решение:" << std::endl;
            for (size_t i = 0; i < n; ++i)
                std::cout << "x" << i + 1 << " = " << x[i] << "\n";
        }
    }
    catch (const MatrixError& e) {
        std::cerr << "Ошибка матрицы: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Общая ошибка: " << e.what() << std::endl;
    }

    return 0;
}

