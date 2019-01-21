//
// Created by qawbecrdtey on 2019-01-21.
//

#ifndef NEURALNETWORK_NETWORK_H
#define NEURALNETWORK_NETWORK_H

#include "Layer.h"
#include "Weight.h"

namespace Neural {

    template<std::size_t _hidden_size> // does not count output, counts input
    class Network {
        static constexpr std::size_t hidden_size = _hidden_size;

    };

    template<>
    class Network<0> {
        std::size_t hidden_size;
    };

}

#endif //NEURALNETWORK_NETWORK_H
