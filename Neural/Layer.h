//
// Created by qawbecrdtey on 2019-01-09.
//

#ifndef NEURALNETWORK_LAYER_H
#define NEURALNETWORK_LAYER_H

#include <random>
#include "Neuron.h"

namespace Neural {

    template<AFT ActFuncType, std::size_t _input_size>
    class Layer {
        Neuron<ActFuncType, _input_size> neurons[_input_size];
        static constexpr std::size_t input_size = _input_size;

    public:
        explicit Layer(double bias = 0) {
            for(std::size_t i = 0; i < _input_size; i++) {
                neurons[i] = Neuron<ActFuncType, _input_size>(bias);
            }
        }

        template<typename Generator, typename Distribution>
        explicit Layer(Generator generator, Distribution distribution) {
            for(std::size_t i = 0; i < _input_size; i++) {
                neurons[i] = Neuron<ActFuncType, _input_size>(distribution(generator));
            }
        }

        template<typename RandomIt>
        explicit Layer(RandomIt first) {
            for (std::size_t i = 0; i < _input_size; i++, first++) {
                neurons[i] = Neuron<ActFuncType, _input_size>(*first);
            }
        }

        template<typename RandomIt>
        explicit Layer(RandomIt first, RandomIt last) {
            auto it = first;
            for(std::size_t i = 0; i < _input_size; i++, it++) {
                neurons[i] = Neuron<ActFuncType, _input_size>(*(it++));
                if(it == last) it = first;
            }
        }

        Layer(const Layer &layer) {
            assert(input_size == layer.input_size);
            std::copy(layer.neurons, layer.neurons + input_size, neurons);
        }
        Layer(Layer &&layer) noexcept {
            assert(input_size == layer.input_size);
            std::swap(neurons, layer.neurons);
        } // swap might not work well
        Layer &operator=(const Layer &layer) {
            assert(input_size == layer.input_size);
            std::copy(layer.neurons, layer.neurons + input_size, neurons);
        }
        Layer &operator=(Layer &&layer) noexcept {
            assert(input_size == layer.input_size);
            std::swap(neurons, layer.neurons);
        } // swap might not work well
        virtual ~Layer() = default;

        template<typename in_RandomIt, typename out_RandomIt>
        void forward(in_RandomIt in_first, in_RandomIt in_last, out_RandomIt out);
    };

    template<std::size_t _input_size>
    class Layer<AFT::Dynamic, _input_size> {
        Neuron<AFT::Dynamic, _input_size> neurons[_input_size];
        static constexpr std::size_t input_size = _input_size;

    public:
        explicit Layer(AFT ActFuncType, double bias = 0) {
            for(std::size_t i = 0; i < _input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, _input_size>(ActFuncType, bias);
            }
        }

        explicit Layer(const std::function<double(double)> &aF = [](double x) -> double { return 0; },
                       const std::function<double(double)> &daF = [](double x) -> double { return 0; },
                       double bias = 0) {
            for(std::size_t i = 0; i < _input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, _input_size>(aF, daF, bias);
            }
        }

        template<typename actRandomIt, typename dactRandomIt>
        explicit Layer(actRandomIt act_first, dactRandomIt dact_first, double bias = 0) {
            for(std::size_t i = 0; i < _input_size; i++, act_first++, dact_first++) {
                neurons[i] = Neuron<AFT::Dynamic, _input_size>(*act_first, *dact_first, bias);
            }
        }

        template<typename actRandomIt, typename dactRandomIt>
        explicit Layer(actRandomIt act_first, actRandomIt act_end, dactRandomIt dact_first, dactRandomIt dact_end, double bias = 0) {
            auto act_it = act_first;
            auto dact_it = dact_first;
            for(std::size_t i = 0; i < _input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, _input_size>(*(act_it++), *(dact_it++), bias);
                if(act_it == act_end) act_it = act_first;
                if(dact_it == dact_end) dact_it = dact_first;
            }
        }

        template<typename Generator, typename Distribution>
        explicit Layer(Generator generator, Distribution distribution, AFT ActFuncType) {
            for(std::size_t i = 0; i < _input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, _input_size>(ActFuncType, generator(distribution));
            }
        }

        template<typename Generator, typename Distribution>
        explicit Layer(Generator generator, Distribution distribution,
                       const std::function<double(double)> &aF = [](double x) -> double { return 0; },
                       const std::function<double(double)> &daF = [](double x) -> double { return 0; }) {
            for(std::size_t i = 0; i < _input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, _input_size>(aF, daF, generator(distribution));
            }
        }

        template<typename Generator, typename Distribution, typename actRandomIt, typename dactRandomIt>
        explicit Layer(Generator generator, Distribution distribution, actRandomIt act_first, dactRandomIt dact_first) {
            for(std::size_t i = 0; i < _input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, _input_size>(*(act_first++), *(dact_first++), generator(distribution));
            }
        }

        template<typename Generator, typename Distribution, typename actRandomIt, typename dactRandomIt>
        explicit Layer(Generator generator, Distribution distribution, actRandomIt act_first, actRandomIt act_end, dactRandomIt dact_first, dactRandomIt dact_end) {
            auto act_it = act_first;
            auto dact_it = dact_first;
            for(std::size_t i = 0; i < _input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, _input_size>(*(act_it++), *(dact_it++), generator(distribution));
                if(act_it == act_end) act_it = act_first;
                if(dact_it == dact_end) dact_it = dact_first;
            }
        }

        template<typename RandomIt>
        explicit Layer(AFT ActFuncType, RandomIt first) {
            for(std::size_t i = 0; i < _input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, _input_size>(ActFuncType, *(first++));
            }
        }

        template<typename RandomIt>
        explicit Layer(RandomIt first,
                       const std::function<double(double)> &aF = [](double x) -> double { return 0; },
                       const std::function<double(double)> &daF = [](double x) -> double { return 0; }) {
            for(std::size_t i = 0; i < _input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, _input_size>(aF, daF, *(first++));
            }
        }

        template<typename actRandomIt, typename dactRandomIt, typename RandomIt>
        explicit Layer(actRandomIt act_first, dactRandomIt dact_first, RandomIt first) {
            for(std::size_t i = 0; i < _input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, _input_size>(*(act_first++), *(dact_first++), *(first++));
            }
        }

        template<typename actRandomIt, typename dactRandomIt, typename RandomIt>
        explicit Layer(actRandomIt act_first, actRandomIt act_end, dactRandomIt dact_first, dactRandomIt dact_end, RandomIt first) {
            auto act_it = act_first;
            auto dact_it = dact_first;
            for(std::size_t i = 0; i < _input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, _input_size>(*(act_it++), *(dact_it++), *(first++));
                if(act_it == act_end) act_it = act_first;
                if(dact_it == dact_end) dact_it == dact_first;
            }
        }

        template<typename RandomIt>
        explicit Layer(AFT ActFuncType, RandomIt first, RandomIt end) {
            auto it = first;
            for(std::size_t i = 0; i < _input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, _input_size>(ActFuncType, *(it++));
                if(it == end) it = first;
            }
        }

        template<typename RandomIt>
        explicit Layer(RandomIt first, RandomIt end,
                       const std::function<double(double)> &aF = [](double x) -> double { return 0; },
                       const std::function<double(double)> &daF = [](double x) -> double { return 0; }) {
            auto it = first;
            for(std::size_t i = 0; i < _input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, _input_size>(aF, daF, *(it++));
                if(it == end) it = first;
            }
        }

        template<typename actRandomIt, typename dactRandomIt, typename RandomIt>
        explicit Layer(actRandomIt act_first, dactRandomIt dact_first, RandomIt first, RandomIt end) {
            auto it = first;
            for(std::size_t i = 0; i < _input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, _input_size>(*(act_first++), *(dact_first++), *(it++));
                if(it == end) it = first;
            }
        }

        template<typename actRandomIt, typename dactRandomIt, typename RandomIt>
        explicit Layer(actRandomIt act_first, actRandomIt act_end, dactRandomIt dact_first, dactRandomIt dact_end, RandomIt first, RandomIt end) {
            auto it = first;
            auto act_it = act_first;
            auto dact_it = dact_first;
            for(std::size_t i = 0; i < _input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, _input_size>(*(act_it++), *(dact_it++), *(it++));
                if(it == end) it = first;
                if(act_it == act_end) act_it = act_first;
                if(dact_it == dact_end) dact_it == dact_first;
            }
        }

        Layer(const Layer &layer) {
            assert(input_size == layer.input_size);
            std::copy(layer.neurons, layer.neurons + input_size, neurons);
        }
        Layer(Layer &&layer) noexcept {
            assert(input_size == layer.input_size);
            std::swap(neurons, layer.neurons);
        } // swap might not work well
        Layer &operator=(const Layer &layer) {
            assert(input_size == layer.input_size);
            std::copy(layer.neurons, layer.neurons + input_size, neurons);
        }
        Layer &operator=(Layer &&layer) noexcept {
            assert(input_size == layer.input_size);
            std::swap(neurons, layer.neurons);
        } // swap might not work well
        virtual ~Layer() = default;

        template<typename in_RandomIt, typename out_RandomIt>
        void forward(in_RandomIt in_first, in_RandomIt in_last, out_RandomIt out);
    };

    template<AFT ActFuncType>
    class Layer<ActFuncType, 0> {
        Neuron<ActFuncType, 0> *neurons;
        std::size_t input_size;

    public:
        explicit Layer(std::size_t input_size, double bias = 0)
                : input_size(input_size),
                  neurons(new Neuron<ActFuncType, 0>[input_size]) {
            for(std::size_t i = 0; i < input_size; i++) {
                neurons[i] = Neuron<ActFuncType, 0>(input_size, bias);
            }
        }

        template<typename Generator, typename Distribution>
        explicit Layer(Generator generator, Distribution distribution, std::size_t input_size)
                : input_size(input_size),
                  neurons(new Neuron<ActFuncType, 0>[input_size]) {
            for(std::size_t i = 0; i < input_size; i++) {
                neurons[i] = Neuron<ActFuncType, 0>(input_size, distribution(generator));
            }
        }

        template<typename RandomIt>
        explicit Layer(std::size_t input_size, RandomIt first)
                : input_size(input_size),
                  neurons(new Neuron<ActFuncType, 0>[input_size]) {
            for(std::size_t i = 0; i < input_size; i++, first++) {
                neurons[i] = Neuron<ActFuncType, 0>(input_size, *first);
            }
        }

        template<typename RandomIt>
        explicit Layer(std::size_t input_size, RandomIt first, RandomIt last)
                : input_size(input_size),
                  neurons(new Neuron<ActFuncType, 0>[input_size]) {
            auto it = first;
            for(std::size_t i = 0; i < input_size; i++) {
                neurons[i] = Neuron<ActFuncType, 0>(input_size, *(it++));
                if(it == last) it = first;
            }
        }

        Layer(const Layer &layer) {
            if(input_size != layer.input_size) {
                input_size = layer.input_size;
                delete[] neurons;
                neurons = new Neuron<ActFuncType, 0>[input_size];
            }
            std::copy(layer.neurons, layer.neurons + input_size, neurons);
        }
        Layer(Layer &&layer) noexcept {
            input_size = layer.input_size;
            std::swap(neurons, layer.neurons);
        } // swap might not work well
        Layer &operator=(const Layer &layer) {
            if(input_size != layer.input_size) {
                input_size = layer.input_size;
                delete[] neurons;
                neurons = new Neuron<ActFuncType, 0>[input_size];
            }
            std::copy(layer.neurons, layer.neurons + input_size, neurons);
        }
        Layer &operator=(Layer &&layer) noexcept {
            std::swap(input_size, layer.input_size);
            std::swap(neurons, layer.neurons);
        } // swap might not work well
        virtual ~Layer() {
            delete[] neurons;
        }

        template<typename in_RandomIt, typename out_RandomIt>
        void forward(in_RandomIt in_first, in_RandomIt in_last, out_RandomIt out);
    };

    template<>
    class Layer<AFT::Dynamic, 0> {
        Neuron<AFT::Dynamic, 0> *neurons;
        std::size_t input_size;

    public:
        explicit Layer(std::size_t input_size, AFT ActFuncType, double bias = 0)
                : input_size(input_size),
                  neurons(new Neuron<AFT::Dynamic, 0>[input_size]) {
            for(std::size_t i = 0; i < input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, 0>(input_size, ActFuncType, bias);
            }
        }

        explicit Layer(std::size_t input_size,
                       const std::function<double(double)> &aF = [](double x) -> double { return 0; },
                       const std::function<double(double)> &daF = [](double x) -> double { return 0; },
                       double bias = 0)
                : input_size(input_size),
                  neurons(new Neuron<AFT::Dynamic, 0>[input_size]) {
            for(std::size_t i = 0; i < input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, 0>(input_size, aF, daF, bias);
            }
        }

        template<typename actRandomIt, typename dactRandomIt>
        explicit Layer(std::size_t input_size, actRandomIt act_first, dactRandomIt dact_first, double bias = 0)
                : input_size(input_size),
                  neurons(new Neuron<AFT::Dynamic, 0>[input_size]) {
            for(std::size_t i = 0; i < input_size; i++, act_first++, dact_first++) {
                neurons[i] = Neuron<AFT::Dynamic, 0>(input_size, *act_first, *dact_first, bias);
            }
        }

        template<typename actRandomIt, typename dactRandomIt>
        explicit Layer(std::size_t input_size, actRandomIt act_first, actRandomIt act_end, dactRandomIt dact_first, dactRandomIt dact_end, double bias = 0)
                : input_size(input_size),
                  neurons(new Neuron<AFT::Dynamic, 0>[input_size]) {
            auto act_it = act_first;
            auto dact_it = dact_first;
            for(std::size_t i = 0; i < input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, 0>(input_size, *(act_it++), *(dact_it++), bias);
                if(act_it == act_end) act_it = act_first;
                if(dact_it == dact_end) dact_it = dact_first;
            }
        }

        template<typename Generator, typename Distribution>
        explicit Layer(std::size_t input_size, Generator generator, Distribution distribution, AFT ActFuncType)
                : input_size(input_size),
                  neurons(new Neuron<AFT::Dynamic, 0>[input_size]) {
            for(std::size_t i = 0; i < input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, 0>(input_size, ActFuncType, generator(distribution));
            }
        }

        template<typename Generator, typename Distribution>
        explicit Layer(std::size_t input_size, Generator generator, Distribution distribution,
                       const std::function<double(double)> &aF = [](double x) -> double { return 0; },
                       const std::function<double(double)> &daF = [](double x) -> double { return 0; })
                : input_size(input_size),
                  neurons(new Neuron<AFT::Dynamic, 0>[input_size]) {
            for(std::size_t i = 0; i < input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, 0>(input_size, aF, daF, generator(distribution));
            }
        }

        template<typename Generator, typename Distribution, typename actRandomIt, typename dactRandomIt>
        explicit Layer(std::size_t input_size, Generator generator, Distribution distribution, actRandomIt act_first, dactRandomIt dact_first)
                : input_size(input_size),
                  neurons(new Neuron<AFT::Dynamic, 0>[input_size]) {
            for(std::size_t i = 0; i < input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, 0>(input_size, *(act_first++), *(dact_first++), generator(distribution));
            }
        }

        template<typename Generator, typename Distribution, typename actRandomIt, typename dactRandomIt>
        explicit Layer(std::size_t input_size, Generator generator, Distribution distribution, actRandomIt act_first, actRandomIt act_end, dactRandomIt dact_first, dactRandomIt dact_end)
                : input_size(input_size),
                  neurons(new Neuron<AFT::Dynamic, 0>[input_size]) {
            auto act_it = act_first;
            auto dact_it = dact_first;
            for(std::size_t i = 0; i < input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, 0>(input_size, *(act_it++), *(dact_it++), generator(distribution));
                if(act_it == act_end) act_it = act_first;
                if(dact_it == dact_end) dact_it = dact_first;
            }
        }

        template<typename RandomIt>
        explicit Layer(std::size_t input_size, AFT ActFuncType, RandomIt first)
                : input_size(input_size),
                  neurons(new Neuron<AFT::Dynamic, 0>[input_size]) {
            for(std::size_t i = 0; i < input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, 0>(input_size, ActFuncType, *(first++));
            }
        }

        template<typename RandomIt>
        explicit Layer(std::size_t input_size, RandomIt first,
                       const std::function<double(double)> &aF = [](double x) -> double { return 0; },
                       const std::function<double(double)> &daF = [](double x) -> double { return 0; })
                : input_size(input_size),
                  neurons(new Neuron<AFT::Dynamic, 0>[input_size]) {
            for(std::size_t i = 0; i < input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, 0>(input_size, aF, daF, *(first++));
            }
        }

        template<typename actRandomIt, typename dactRandomIt, typename RandomIt>
        explicit Layer(std::size_t input_size, actRandomIt act_first, dactRandomIt dact_first, RandomIt first)
                : input_size(input_size),
                  neurons(new Neuron<AFT::Dynamic, 0>[input_size]) {
            for(std::size_t i = 0; i < input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, 0>(input_size, *(act_first++), *(dact_first++), *(first++));
            }
        }

        template<typename actRandomIt, typename dactRandomIt, typename RandomIt>
        explicit Layer(std::size_t input_size, actRandomIt act_first, actRandomIt act_end, dactRandomIt dact_first, dactRandomIt dact_end, RandomIt first)
                : input_size(input_size),
                  neurons(new Neuron<AFT::Dynamic, 0>[input_size]) {
            auto act_it = act_first;
            auto dact_it = dact_first;
            for(std::size_t i = 0; i < input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, 0>(input_size, *(act_it++), *(dact_it++), *(first++));
                if(act_it == act_end) act_it = act_first;
                if(dact_it == dact_end) dact_it == dact_first;
            }
        }

        template<typename RandomIt>
        explicit Layer(std::size_t input_size, AFT ActFuncType, RandomIt first, RandomIt end)
                : input_size(input_size),
                  neurons(new Neuron<AFT::Dynamic, 0>[input_size]) {
            auto it = first;
            for(std::size_t i = 0; i < input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, 0>(input_size, ActFuncType, *(it++));
                if(it == end) it = first;
            }
        }

        template<typename RandomIt>
        explicit Layer(std::size_t input_size, RandomIt first, RandomIt end,
                       const std::function<double(double)> &aF = [](double x) -> double { return 0; },
                       const std::function<double(double)> &daF = [](double x) -> double { return 0; })
                : input_size(input_size),
                  neurons(new Neuron<AFT::Dynamic, 0>[input_size]) {
            auto it = first;
            for(std::size_t i = 0; i < input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, 0>(input_size, aF, daF, *(it++));
                if(it == end) it = first;
            }
        }

        template<typename actRandomIt, typename dactRandomIt, typename RandomIt>
        explicit Layer(std::size_t input_size, actRandomIt act_first, dactRandomIt dact_first, RandomIt first, RandomIt end)
                : input_size(input_size),
                  neurons(new Neuron<AFT::Dynamic, 0>[input_size]) {
            auto it = first;
            for(std::size_t i = 0; i < input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, 0>(input_size, *(act_first++), *(dact_first++), *(it++));
                if(it == end) it = first;
            }
        }

        template<typename actRandomIt, typename dactRandomIt, typename RandomIt>
        explicit Layer(std::size_t input_size, actRandomIt act_first, actRandomIt act_end, dactRandomIt dact_first, dactRandomIt dact_end, RandomIt first, RandomIt end)
                : input_size(input_size),
                  neurons(new Neuron<AFT::Dynamic, 0>[input_size]) {
            auto it = first;
            auto act_it = act_first;
            auto dact_it = dact_first;
            for(std::size_t i = 0; i < input_size; i++) {
                neurons[i] = Neuron<AFT::Dynamic, 0>(input_size, *(act_it++), *(dact_it++), *(it++));
                if(it == end) it = first;
                if(act_it == act_end) act_it = act_first;
                if(dact_it == dact_end) dact_it == dact_first;
            }
        }

        Layer(const Layer &layer) {
            if(input_size != layer.input_size) {
                input_size = layer.input_size;
                delete[] neurons;
                neurons = new Neuron<AFT::Dynamic, 0>[input_size];
            }
            std::copy(layer.neurons, layer.neurons + input_size, neurons);
        }
        Layer(Layer &&layer) noexcept {
            input_size = layer.input_size;
            std::swap(neurons, layer.neurons);
        } // swap might not work well
        Layer &operator=(const Layer &layer) {
            if(input_size != layer.input_size) {
                input_size = layer.input_size;
                delete[] neurons;
                neurons = new Neuron<AFT::Dynamic, 0>[input_size];
            }
            std::copy(layer.neurons, layer.neurons + input_size, neurons);
        }
        Layer &operator=(Layer &&layer) noexcept {
            std::swap(input_size, layer.input_size);
            std::swap(neurons, layer.neurons);
        } // swap might not work well
        virtual ~Layer() {
            delete[] neurons;
        }

        template<typename in_RandomIt, typename out_RandomIt>
        void forward(in_RandomIt in_first, in_RandomIt in_last, out_RandomIt out);
    };

    template<AFT ActFuncType, size_t _input_size>
    template<typename in_RandomIt, typename out_RandomIt>
    void Layer<ActFuncType, _input_size>::forward(in_RandomIt in_first, in_RandomIt in_last, out_RandomIt out) {
        assert(in_last - in_first == input_size);
        std::size_t i = 0;
        auto it = in_first;
        while(i < input_size) {
            *out = neurons[i].forward(in_first, in_last);
            i++; it++; out++;
        }
    }
}


#endif //NEURALNETWORK_LAYER_H
