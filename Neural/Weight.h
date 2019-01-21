//
// Created by qawbecrdtey on 2019-01-17.
//

#ifndef NEURALNETWORK_WEIGHT_H
#define NEURALNETWORK_WEIGHT_H

#include <cstddef>
#include <cassert>
#include <algorithm>
#include <ostream>
#include <chrono>
#include <random>

namespace Neural {
    template<std::size_t _row, std::size_t _col>
    class Weight {
        static constexpr std::size_t row = _row;
        static constexpr std::size_t col = _col;
        double mat[_row * _col];

    public:
        explicit Weight() {
            auto seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine generator(seed);
            std::normal_distribution<double> distribution(0, 2.0 / row);
            for(std::size_t i = 0; i < row * col; i++) {
                mat[i] = distribution(generator);
            }
        }
        template<std::size_t __row, std::size_t __col>
        Weight(const Weight<__row, __col> &weight) {
            assert(row == weight.row && col == weight.col);
            std::copy(weight.mat, weight.mat + row * col, mat);
        }
        template<std::size_t __row, std::size_t __col>
        Weight(Weight<__row, __col> &&weight) noexcept {
            assert(row == weight.row && col == weight.col);
            std::move(weight.mat, weight.mat + row * col, mat);
        }
        template<std::size_t __row, std::size_t __col>
        Weight &operator=(const Weight<__row, __col> &weight) {
            assert(row == weight.row && col == weight.col);
            std::copy(mat, mat + row * col, weight.mat);
            return (*this);
        }
        template<std::size_t __row, std::size_t __col>
        Weight &operator=(Weight<__row, __col> &&weight) noexcept {
            assert(row == weight.row && col == weight.col);
            mat = std::move(weight.mat);
            return (*this);
        }

        virtual ~Weight() = default;

        template<typename in_RandomIt, typename out_RandomIt>
        void forward(in_RandomIt in_first, out_RandomIt out_first);

        double operator()(std::size_t r, std::size_t c) const;

        friend std::ostream &operator<<(std::ostream &os, Weight &weight);
    };

    template<std::size_t _col>
    class Weight<0, _col> {
        std::size_t row;
        static constexpr std::size_t col = _col;
        double *mat;

    public:
        explicit Weight(std::size_t _row)
                : row(_row), mat(new double[_row * _col]) {
            auto seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine generator(seed);
            std::normal_distribution<double> distribution(0, 2.0 / row);
            for(int i = 0; i < row * col; i++) {
                mat[i] = distribution(generator);
            }
        }

        template<std::size_t __row, std::size_t __col>
        Weight(const Weight<__row, __col> &weight) {
            assert(col == weight.col);
            delete[] mat;
            mat = new double[(row = weight.row) * col];
            std::copy(weight.mat, weight.mat + row * col, mat);
        }
        template<std::size_t __row, std::size_t __col>
        Weight(Weight<__row, __col> &&weight) noexcept {
            assert(col == weight.col);
            row = weight.row;
            mat = std::move(weight.mat);
        }
        template<std::size_t __row, std::size_t __col>
        Weight &operator=(const Weight<__row, __col> &weight) {
            assert(col == weight.col);
            delete[] mat;
            mat = new double[(row = weight.row) * col];
            std::copy(weight.mat, weight.mat + row * col, mat);
            return (*this);
        }
        template<std::size_t __row, std::size_t __col>
        Weight &operator=(Weight<__row, __col> &&weight) noexcept {
            assert(col == weight.col);
            row = weight.row;
            mat = std::move(weight.mat);
        }

        virtual ~Weight() {
            delete[] mat;
        }

        template<typename in_RandomIt, typename out_RandomIt>
        void forward(in_RandomIt in_first, out_RandomIt out_first);

        double operator()(std::size_t r, std::size_t c) const;

        friend std::ostream &operator<<(std::ostream &os, Weight &weight);
    };

    template<std::size_t _row>
    class Weight<_row, 0> {
        static constexpr std::size_t row = _row;
        std::size_t col;
        double *mat;

    public:
        explicit Weight(std::size_t _col)
                : col(_col), mat(new double[_row * _col]) {
            auto seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine generator(seed);
            std::normal_distribution<double> distribution(0, 2.0 / row);
            for(int i = 0; i < row * col; i++) {
                mat[i] = distribution(generator);
            }
        }

        template<std::size_t __row, std::size_t __col>
        Weight(const Weight<__row, __col> &weight) {
            assert(row == weight.row);
            delete[] mat;
            mat = new double[row * (col = weight.col)];
            std::copy(weight.mat, weight.mat + row * col, mat);
        }
        template<std::size_t __row, std::size_t __col>
        Weight(Weight<__row, __col> &&weight) noexcept {
            assert(row == weight.row);
            col = weight.col;
            mat = std::move(weight.mat);
        }
        template<std::size_t __row, std::size_t __col>
        Weight &operator=(const Weight<__row, __col> &weight) {
            assert(row == weight.row);
            delete[] mat;
            mat = new double[row * (col = weight.col)];
            std::copy(weight.mat, weight.mat + row * col, mat);
            return (*this);
        }
        template<std::size_t __row, std::size_t __col>
        Weight &operator=(Weight<__row, __col> &&weight) noexcept {
            assert(row == weight.row);
            col = weight.col;
            mat = std::move(weight.mat);
            return (*this);
        }

        virtual ~Weight() {
            delete[] mat;
        }

        template<typename in_RandomIt, typename out_RandomIt>
        void forward(in_RandomIt in_first, out_RandomIt out_first);

        double operator()(std::size_t r, std::size_t c) const;

        friend std::ostream &operator<<(std::ostream &os, Weight &weight);
    };

    template<>
    class Weight<0, 0> {
        std::size_t row;
        std::size_t col;
        double *mat;

    public:
        explicit Weight(std::size_t _row, std::size_t _col)
                : row(_row), col(_col), mat(new double[_row * _col]) {
            auto seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine generator(seed);
            std::normal_distribution<double> distribution(0, 2.0 / row);
            for(int i = 0; i < row * col; i++) {
                mat[i] = distribution(generator);
            }
        }

        template<std::size_t __row, std::size_t __col>
        Weight(const Weight<__row, __col> &weight) {
            row = weight.row;
            col = weight.col;
            delete[] mat;
            mat = new double[row * col];
            std::copy(weight.mat, weight.mat + row * col, mat);
        }
        template<std::size_t __row, std::size_t __col>
        Weight(Weight<__row, __col> &&weight) noexcept {
            row = weight.row;
            col = weight.col;
            mat = std::move(weight.mat);
        }
        template<std::size_t __row, std::size_t __col>
        Weight &operator=(const Weight<__row, __col> &weight) {
            row = weight.row;
            col = weight.col;
            delete[] mat;
            mat = new double[row * col];
            std::copy(weight.mat, weight.mat + row * col, mat);
            return (*this);
        }
        template<std::size_t __row, std::size_t __col>
        Weight &operator=(Weight<__row, __col> &&weight) noexcept {
            row = weight.row;
            col = weight.col;
            mat = std::move(weight.mat);
            return (*this);
        }

        virtual ~Weight() {
            delete[] mat;
        }

        template<typename in_RandomIt, typename out_RandomIt>
        void forward(in_RandomIt in_first, out_RandomIt out_first);

        double operator()(std::size_t r, std::size_t c) const;

        friend std::ostream &operator<<(std::ostream &os, Weight &weight);
    };

    template<std::size_t _row, std::size_t _col>
    template<typename in_RandomIt, typename out_RandomIt>
    void Weight<_row, _col>::forward(in_RandomIt in_first, out_RandomIt out_first) {
        auto out = out_first;
        for(std::size_t j = 0; j < col; j++, out++) {
            *out = 0;
            auto in = in_first;
            for(std::size_t i = 0; i < row; i++, in++) {
                *out += mat(i, j) * (*in);
            }
        }
    } // z_j1 += l_i1 * w_ij = (w^T)_ji * l_i1

    template<std::size_t _row, std::size_t _col>
    double Weight<_row, _col>::operator()(std::size_t r, std::size_t c) const {
        assert(r < row && c < col);
        return mat[r * col + c];
    }

    template<std::size_t _row, std::size_t _col>
    std::ostream &operator<<(std::ostream &os, Weight<_row, _col> &weight) {
        for(std::size_t i = 0; i < weight.row; i++) {
            for(std::size_t j = 0; j < weight.col; j++) {
                os << weight(i, j) << ' ';
            }
            os << '\n';
        }
        return os;
    }
}

#endif //NEURALNETWORK_WEIGHT_H
