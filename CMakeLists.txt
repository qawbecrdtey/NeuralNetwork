cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)

add_executable(NeuralNetwork main.cpp Neural/Neuron.h Neural/Layer.h Neural/Weight.h Neural/Network.h)