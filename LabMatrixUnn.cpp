#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <locale>

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

template <typename T>
class Matrix {
private:
    std::vector<std::vector<T>> grid;
    size_t rows_cnt;
    size_t cols_cnt;

public:
    Matrix() : rows_cnt(0), cols_cnt(0) {}

    Matrix(size_t r, size_t c, const T& value = T()) {
        rows_cnt = r;
        cols_cnt = c;
        grid.assign(r, std::vector<T>(c, value));
    }

    size_t get_rows() const { return rows_cnt; }
    size_t get_cols() const { return cols_cnt; }

    T& at(size_t r, size_t c) {
        if (r >= rows_cnt || c >= cols_cnt) {
            throw OutMatrixError("Выход за пределы матрицы!");
        }
        return grid[r][c];
    }

    T& operator()(size_t r, size_t c) { return at(r, c); }
    std::vector<T>& operator[](size_t r) { return grid[r]; }

    void resize(size_t r, size_t c) {
        rows_cnt = r;
        cols_cnt = c;
        grid.assign(r, std::vector<T>(c, T()));
    }
};

template <typename T>
std::vector<T> solve_gauss(Matrix<T> A, std::vector<T> b) {
    size_t n = A.get_rows();
    if (n == 0) {
        return std::vector<T>();
    }

    const T EPS = 1e-9;

    for (size_t col = 0; col < n; ++col) {
        size_t pivot_row = col;
        T max_val = std::fabs(A[col][col]);

        for (size_t row = col + 1; row < n; ++row) {
            T current_val = std::fabs(A[row][col]);
            if (current_val > max_val) {
                max_val = current_val;
                pivot_row = row;
            }
        }

        if (max_val < EPS) {
            throw MathMatrixError("Матрица вырождена или имеет бесконечно много решений");
        }

        if (pivot_row != col) {
            std::swap(A[col], A[pivot_row]);
            std::swap(b[col], b[pivot_row]);
        }

        for (size_t row = col + 1; row < n; ++row) {
            T factor = A[row][col] / A[col][col];
            for (size_t j = col; j < n; ++j) {
                A[row][j] -= factor * A[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    std::vector<T> x(n);

    for (size_t i = n; i > 0; --i) {
        size_t k = i - 1;
        T sum = b[k];

        for (size_t j = k + 1; j < n; ++j) {
            sum -= A[k][j] * x[j];
        }

        if (std::fabs(A[k][k]) < EPS) {
            throw MathMatrixError("Деление на ноль при обратном ходе.");
        }

        x[k] = sum / A[k][k];
    }
    return x;
}

int main() {
    std::setlocale(LC_ALL, "Russian");

    try {
        size_t n;
        std::cout << "Введите размер квадратной матрицы системы (N): ";
        if (!(std::cin >> n)) return 0;

        Matrix<double> A(n, n);
        std::cout << "Введите элементы матрицы A по строкам:\n";
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                std::cin >> A(i, j);
            }
        }

        std::vector<double> b(n);
        std::cout << "Введите элементы вектора b:\n";
        for (size_t i = 0; i < n; ++i) {
            std::cin >> b[i];
        }

        std::vector<double> x = solve_gauss(A, b);

        std::cout << "Результат:" << std::endl;
        for (size_t i = 0; i < n; ++i) {
            std::cout << "x" << i + 1 << " = " << x[i] << "\n";
        }

    }
    catch (const MatrixError& e) {
        std::cerr << "Матричная ошибка: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Общая ошибка: " << e.what() << std::endl;
    }

    return 0;
}