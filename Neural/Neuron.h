//
// Created by qawbecrdtey on 2019-01-08.
//

#ifndef NEURALNETWORK_NEURON_H
#define NEURALNETWORK_NEURON_H


#include <cstddef>
#include <functional>
#include <cmath>
#include <ostream>
#include <cassert>

namespace Neural {

    enum class ActivationFunctionType {
        Dynamic,
        Zero,
        One,
        Identity,
        Sigmoid,
        Tanh,
        Arctan,
        Softsign,
        ReLU,
        SQNL,
        Softplus,
    };
    using AFT = ActivationFunctionType;

    template<AFT ActFuncType, std::size_t _input_size>
    class Neuron {
        static constexpr std::size_t input_size = _input_size;
        double bias;
        double sum;
        std::function<double(double)> activationFunction;
        std::function<double(double)> dactivationFunction;

    public:
        explicit Neuron(double bias = 0) : bias(bias), sum(0) {
            switch (ActFuncType) {
                case AFT::Zero:
                    activationFunction = [](double x) -> double { return 0; };
                    dactivationFunction = [](double x) -> double { return 0; };
                    break;
                case AFT::One:
                    activationFunction = [](double x) -> double { return 1; };
                    dactivationFunction = [](double x) -> double { return 1; };
                    break;
                case AFT::Identity:
                    activationFunction = [](double x) -> double { return x; };
                    dactivationFunction = [](double x) -> double { return x; };
                    break;
                case AFT::Sigmoid:
                    activationFunction = [](double x) -> double { return 1 / (1 + std::exp(-x)); };
                    dactivationFunction = [&](double x) -> double {
                        auto a = activationFunction(x);
                        return a * (1 - a);
                    };
                    break;
                case AFT::Tanh:
                    activationFunction = [](double x) -> double { return std::tanh(x); };
                    dactivationFunction = [](double x) -> double {
                        auto a = std::tanh(x);
                        return 1 - a * a;
                    };
                    break;
                case AFT::Arctan:
                    activationFunction = [](double x) -> double { return std::atan(x); };
                    dactivationFunction = [](double x) -> double { return 1 / (1 + x * x); };
                    break;
                case AFT::Softsign:
                    activationFunction = [](double x) -> double { return x / (1 + std::abs(x)); };
                    dactivationFunction = [](double x) -> double {
                        return 1 / ((1 + std::abs(x)) * (1 + std::abs(x)));
                    };
                    break;
                case AFT::ReLU:
                    activationFunction = [](double x) -> double { return x > 0 ? x : 0; };
                    dactivationFunction = [](double x) -> double { return x > 0 ? 1 : (x < 0 ? 0 : 0.5); };
                    break;
                case AFT::SQNL:
                    activationFunction = [](double x) -> double {
                        return x > 0 ? (x > 2 ? 1 : (x - x * x / 4)) : (x < -2 ? -1 : (x + x * x / 4));
                    };
                    dactivationFunction = [](double x) -> double {
                        return x > 0 ? (x > 2 ? 0 : (1 - x / 2)) : (x < -2 ? 0 : (1 + x / 2));
                    };
                    break;
                case AFT::Softplus:
                    activationFunction = [](double x) -> double { return std::log(1 + std::exp(x)); };
                    dactivationFunction = [](double x) -> double { return 1 / (1 + std::exp(-x)); };
                    break;
                default:
                    throw std::runtime_error("Unknown error!");
            }
        }

        Neuron(const Neuron &neuron) {
            assert(input_size == neuron.input_size &&
                   activationFunction == neuron.activationFunction &&
                   dactivationFunction == neuron.dactivationFunction);
            bias = neuron.bias;
            sum = neuron.sum;
        }
        Neuron(Neuron &&neuron) noexcept {
            assert(input_size == neuron.input_size &&
                   activationFunction == neuron.activationFunction &&
                   dactivationFunction == neuron.dactivationFunction);
            bias = neuron.bias;
            sum = neuron.sum;
        }
        Neuron &operator=(const Neuron &neuron) {
            assert(input_size == neuron.input_size &&
                   activationFunction == neuron.activationFunction &&
                   dactivationFunction == neuron.dactivationFunction);
            bias = neuron.bias;
            sum = neuron.sum;
        }
        Neuron &operator=(Neuron &&neuron) noexcept {
            assert(input_size == neuron.input_size &&
                   activationFunction == neuron.activationFunction &&
                   dactivationFunction == neuron.dactivationFunction);
            bias = neuron.bias;
            sum = neuron.sum;
        }
        virtual ~Neuron() = default;

        template<AFT _ActFuncType, std::size_t __input_size>
        bool operator==(const Neuron<_ActFuncType, __input_size> &rhs) const;

        template<AFT _ActFuncType, std::size_t __input_size>
        bool operator!=(const Neuron<_ActFuncType, __input_size> &rhs) const;

        template<typename RandomIt>
        double forward(RandomIt first, RandomIt last);
    };

    template<std::size_t _input_size>
    class Neuron<AFT::Dynamic, _input_size> {
        static constexpr std::size_t input_size = _input_size;
        double bias;
        double sum;
        std::function<double(double)> activationFunction;
        std::function<double(double)> dactivationFunction;

    public:
        explicit Neuron(std::function<double(double)> aF = [](double x) -> double { return 0; },
                        std::function<double(double)> daF = [](double x) -> double { return 0; },
                        double bias = 0)
                : bias(bias), sum(0), activationFunction(std::move(aF)), dactivationFunction(std::move(daF)) {}

        explicit Neuron(AFT ActFuncType, double bias = 0) : bias(bias), sum(0) {
            switch (ActFuncType) {
                case AFT::Zero:
                    activationFunction = [](double x) -> double { return 0; };
                    dactivationFunction = [](double x) -> double { return 0; };
                    break;
                case AFT::One:
                    activationFunction = [](double x) -> double { return 1; };
                    dactivationFunction = [](double x) -> double { return 1; };
                    break;
                case AFT::Identity:
                    activationFunction = [](double x) -> double { return x; };
                    dactivationFunction = [](double x) -> double { return x; };
                    break;
                case AFT::Sigmoid:
                    activationFunction = [](double x) -> double { return 1 / (1 + std::exp(-x)); };
                    dactivationFunction = [&](double x) -> double {
                        auto a = activationFunction(x);
                        return a * (1 - a);
                    };
                    break;
                case AFT::Tanh:
                    activationFunction = [](double x) -> double { return std::tanh(x); };
                    dactivationFunction = [](double x) -> double {
                        auto a = std::tanh(x);
                        return 1 - a * a;
                    };
                    break;
                case AFT::Arctan:
                    activationFunction = [](double x) -> double { return std::atan(x); };
                    dactivationFunction = [](double x) -> double { return 1 / (1 + x * x); };
                    break;
                case AFT::Softsign:
                    activationFunction = [](double x) -> double { return x / (1 + std::abs(x)); };
                    dactivationFunction = [](double x) -> double {
                        return 1 / ((1 + std::abs(x)) * (1 + std::abs(x)));
                    };
                    break;
                case AFT::ReLU:
                    activationFunction = [](double x) -> double { return x > 0 ? x : 0; };
                    dactivationFunction = [](double x) -> double { return x > 0 ? 1 : (x < 0 ? 0 : 0.5); };
                    break;
                case AFT::SQNL:
                    activationFunction = [](double x) -> double {
                        return x > 0 ? (x > 2 ? 1 : (x - x * x / 4)) : (x < -2 ? -1 : (x + x * x / 4));
                    };
                    dactivationFunction = [](double x) -> double {
                        return x > 0 ? (x > 2 ? 0 : (1 - x / 2)) : (x < -2 ? 0 : (1 + x / 2));
                    };
                    break;
                case AFT::Softplus:
                    activationFunction = [](double x) -> double { return std::log(1 + std::exp(x)); };
                    dactivationFunction = [](double x) -> double { return 1 / (1 + std::exp(-x)); };
                    break;
                default:
                    throw std::runtime_error("Unknown error!");
            }
        }

        Neuron(const Neuron &neuron) {
            assert(input_size == neuron.input_size);
            activationFunction = neuron.activationFunction;
            dactivationFunction = neuron.dactivationFunction;
            bias = neuron.bias;
            sum = neuron.sum;
        }
        Neuron(Neuron &&neuron) noexcept {
            assert(input_size == neuron.input_size);
            std::swap(activationFunction, neuron.activationFunction);
            std::swap(dactivationFunction, neuron.dactivationFunction);
            bias = neuron.bias;
            sum = neuron.sum;
        }
        Neuron &operator=(const Neuron &neuron) {
            assert(input_size == neuron.input_size);
            activationFunction = neuron.activationFunction;
            dactivationFunction = neuron.dactivationFunction;
            bias = neuron.bias;
            sum = neuron.sum;
        }
        Neuron &operator=(Neuron &&neuron) noexcept {
            assert(input_size == neuron.input_size);
            std::swap(activationFunction, neuron.activationFunction);
            std::swap(dactivationFunction, neuron.dactivationFunction);
            bias = neuron.bias;
            sum = neuron.sum;
        }
        virtual ~Neuron() = default;

        template<AFT _ActFuncType, std::size_t __input_size>
        bool operator==(const Neuron<_ActFuncType, __input_size> &rhs) const;

        template<AFT _ActFuncType, std::size_t __input_size>
        bool operator!=(const Neuron<_ActFuncType, __input_size> &rhs) const;

        template<typename RandomIt>
        double forward(RandomIt first, RandomIt last);
    };

    template<AFT ActFuncType>
    class Neuron<ActFuncType, 0> {
        std::size_t input_size;
        double bias;
        double sum;
        std::function<double(double)> activationFunction;
        std::function<double(double)> dactivationFunction;

    public:
        explicit Neuron<ActFuncType, 0>(std::size_t input_size, double bias = 0) : bias(bias), input_size(input_size), sum(0) {
            switch (ActFuncType) {
                case AFT::Zero:
                    activationFunction = [](double x) -> double { return 0; };
                    dactivationFunction = [](double x) -> double { return 0; };
                    break;
                case AFT::One:
                    activationFunction = [](double x) -> double { return 1; };
                    dactivationFunction = [](double x) -> double { return 1; };
                    break;
                case AFT::Identity:
                    activationFunction = [](double x) -> double { return x; };
                    dactivationFunction = [](double x) -> double { return x; };
                    break;
                case AFT::Sigmoid:
                    activationFunction = [](double x) -> double { return 1 / (1 + std::exp(-x)); };
                    dactivationFunction = [&](double x) -> double {
                        auto a = activationFunction(x);
                        return a * (1 - a);
                    };
                    break;
                case AFT::Tanh:
                    activationFunction = [](double x) -> double { return std::tanh(x); };
                    dactivationFunction = [](double x) -> double {
                        auto a = std::tanh(x);
                        return 1 - a * a;
                    };
                    break;
                case AFT::Arctan:
                    activationFunction = [](double x) -> double { return std::atan(x); };
                    dactivationFunction = [](double x) -> double { return 1 / (1 + x * x); };
                    break;
                case AFT::Softsign:
                    activationFunction = [](double x) -> double { return x / (1 + std::abs(x)); };
                    dactivationFunction = [](double x) -> double {
                        return 1 / ((1 + std::abs(x)) * (1 + std::abs(x)));
                    };
                    break;
                case AFT::ReLU:
                    activationFunction = [](double x) -> double { return x > 0 ? x : 0; };
                    dactivationFunction = [](double x) -> double { return x > 0 ? 1 : (x < 0 ? 0 : 0.5); };
                    break;
                case AFT::SQNL:
                    activationFunction = [](double x) -> double {
                        return x > 0 ? (x > 2 ? 1 : (x - x * x / 4)) : (x < -2 ? -1 : (x + x * x / 4));
                    };
                    dactivationFunction = [](double x) -> double {
                        return x > 0 ? (x > 2 ? 0 : (1 - x / 2)) : (x < -2 ? 0 : (1 + x / 2));
                    };
                    break;
                case AFT::Softplus:
                    activationFunction = [](double x) -> double { return std::log(1 + std::exp(x)); };
                    dactivationFunction = [](double x) -> double { return 1 / (1 + std::exp(-x)); };
                    break;
                default:
                    throw std::runtime_error("Unknown error!");
            }
        }

        Neuron(const Neuron &neuron) {
            assert(activationFunction == neuron.activationFunction &&
                   dactivationFunction == neuron.dactivationFunction);
            input_size = neuron.input_size;
            bias = neuron.bias;
            sum = neuron.sum;
        }
        Neuron(Neuron &&neuron) noexcept {
            assert(activationFunction == neuron.activationFunction &&
                   dactivationFunction == neuron.dactivationFunction);
            input_size = neuron.input_size;
            bias = neuron.bias;
            sum = neuron.sum;
        }
        Neuron &operator=(const Neuron &neuron) {
            assert(activationFunction == neuron.activationFunction &&
                   dactivationFunction == neuron.dactivationFunction);
            input_size = neuron.input_size;
            bias = neuron.bias;
            sum = neuron.sum;
        }
        Neuron &operator=(Neuron &&neuron) noexcept {
            assert(activationFunction == neuron.activationFunction &&
                   dactivationFunction == neuron.dactivationFunction);
            input_size = neuron.input_size;
            bias = neuron.bias;
            sum = neuron.sum;
        }
        virtual ~Neuron() = default;

        template<AFT _ActFuncType, std::size_t __input_size>
        bool operator==(const Neuron<_ActFuncType, __input_size> &rhs) const;

        template<AFT _ActFuncType, std::size_t __input_size>
        bool operator!=(const Neuron<_ActFuncType, __input_size> &rhs) const;

        template<typename RandomIt>
        double forward(RandomIt first, RandomIt last);
    };

    template<>
    class Neuron<AFT::Dynamic, 0> {
        std::size_t input_size;
        double bias;
        double sum;
        std::function<double(double)> activationFunction;
        std::function<double(double)> dactivationFunction;

    public:
        explicit Neuron(std::size_t input_size = 0,
                        std::function<double(double)> aF = [](double x) -> double { return 0; },
                        std::function<double(double)> daF = [](double x) -> double { return 0; },
                        double bias = 0)
                : bias(bias), input_size(input_size), sum(0), activationFunction(std::move(aF)), dactivationFunction(std::move(daF)) {}

        explicit Neuron(std::size_t input_size, AFT ActFuncType, double bias = 0) : bias(bias), input_size(input_size), sum(0) {
            switch (ActFuncType) {
                case AFT::Zero:
                    activationFunction = [](double x) -> double { return 0; };
                    dactivationFunction = [](double x) -> double { return 0; };
                    break;
                case AFT::One:
                    activationFunction = [](double x) -> double { return 1; };
                    dactivationFunction = [](double x) -> double { return 1; };
                    break;
                case AFT::Identity:
                    activationFunction = [](double x) -> double { return x; };
                    dactivationFunction = [](double x) -> double { return x; };
                    break;
                case AFT::Sigmoid:
                    activationFunction = [](double x) -> double { return 1 / (1 + std::exp(-x)); };
                    dactivationFunction = [&](double x) -> double {
                        auto a = activationFunction(x);
                        return a * (1 - a);
                    };
                    break;
                case AFT::Tanh:
                    activationFunction = [](double x) -> double { return std::tanh(x); };
                    dactivationFunction = [](double x) -> double {
                        auto a = std::tanh(x);
                        return 1 - a * a;
                    };
                    break;
                case AFT::Arctan:
                    activationFunction = [](double x) -> double { return std::atan(x); };
                    dactivationFunction = [](double x) -> double { return 1 / (1 + x * x); };
                    break;
                case AFT::Softsign:
                    activationFunction = [](double x) -> double { return x / (1 + std::abs(x)); };
                    dactivationFunction = [](double x) -> double {
                        return 1 / ((1 + std::abs(x)) * (1 + std::abs(x)));
                    };
                    break;
                case AFT::ReLU:
                    activationFunction = [](double x) -> double { return x > 0 ? x : 0; };
                    dactivationFunction = [](double x) -> double { return x > 0 ? 1 : (x < 0 ? 0 : 0.5); };
                    break;
                case AFT::SQNL:
                    activationFunction = [](double x) -> double {
                        return x > 0 ? (x > 2 ? 1 : (x - x * x / 4)) : (x < -2 ? -1 : (x + x * x / 4));
                    };
                    dactivationFunction = [](double x) -> double {
                        return x > 0 ? (x > 2 ? 0 : (1 - x / 2)) : (x < -2 ? 0 : (1 + x / 2));
                    };
                    break;
                case AFT::Softplus:
                    activationFunction = [](double x) -> double { return std::log(1 + std::exp(x)); };
                    dactivationFunction = [](double x) -> double { return 1 / (1 + std::exp(-x)); };
                    break;
                default:
                    throw std::runtime_error("Unknown error!");
            }
        }

        Neuron(const Neuron &neuron) {
            input_size = neuron.input_size;
            activationFunction = neuron.activationFunction;
            dactivationFunction = neuron.dactivationFunction;
            bias = neuron.bias;
            sum = neuron.sum;
        }
        Neuron(Neuron &&neuron) noexcept {
            input_size = neuron.input_size;
            std::swap(activationFunction, neuron.activationFunction);
            std::swap(dactivationFunction, neuron.dactivationFunction);
            bias = neuron.bias;
            sum = neuron.sum;
        }
        Neuron &operator=(const Neuron &neuron) {
            input_size = neuron.input_size;
            activationFunction = neuron.activationFunction;
            dactivationFunction = neuron.dactivationFunction;
            bias = neuron.bias;
            sum = neuron.sum;
        }
        Neuron &operator=(Neuron &&neuron) noexcept {
            input_size = neuron.input_size;
            std::swap(activationFunction, neuron.activationFunction);
            std::swap(dactivationFunction, neuron.dactivationFunction);
            bias = neuron.bias;
            sum = neuron.sum;
        }
        virtual ~Neuron() = default;

        template<AFT _ActFuncType, std::size_t __input_size>
        bool operator==(const Neuron<_ActFuncType, __input_size> &rhs) const;

        bool operator!=(const Neuron<AFT::Dynamic, 0> &rhs) const;

        template<typename RandomIt>
        double forward(RandomIt first, RandomIt last);
    };

    template<AFT ActFuncType, size_t _input_size>
    template<AFT _ActFuncType, size_t __input_size>
    bool Neuron<ActFuncType, _input_size>::operator==(const Neuron<_ActFuncType, __input_size> &rhs) const {
        return bias == rhs.bias &&
               activationFunction == rhs.activationFunction &&
               dactivationFunction == rhs.dactivationFunction &&
               input_size == rhs.input_size;
    }

    template<AFT ActFuncType, size_t _input_size>
    template<AFT _ActFuncType, size_t __input_size>
    bool Neuron<ActFuncType, _input_size>::operator!=(const Neuron<_ActFuncType, __input_size> &rhs) const {
        return !(rhs == *this);
    }

    template<AFT ActFuncType, size_t _input_size>
    template<typename RandomIt>
    double Neuron<ActFuncType, _input_size>::forward(RandomIt first, RandomIt last) {
        assert(last - first == input_size);
        sum = bias;
        for (auto it = first; it != last; it++) {
            sum += *it;
        }
        return activationFunction(sum);
    }
}

#endif //NEURALNETWORK_NEURON_H
