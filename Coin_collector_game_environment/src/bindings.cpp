#include <pybind11/pybind11.h>

#include "GameEnvironment.hpp"

namespace py = pybind11;

PYBIND11_MODULE(game_env, m)
{
    m.doc() = "Python bindings for the C++ coin collection RL environment";

    py::enum_<Action>(m, "Action")
        .value("Up", Action::Up)
        .value("Down", Action::Down)
        .value("Left", Action::Left)
        .value("Right", Action::Right)
        .export_values();

    py::class_<State>(m, "State")
        .def_readonly("dx", &State::dx)
        .def_readonly("dy", &State::dy);

    py::class_<StepResult>(m, "StepResult")
        .def_readonly("nextState", &StepResult::nextState)
        .def_readonly("reward", &StepResult::reward)
        .def_readonly("done", &StepResult::done);

    py::class_<GameEnvironment>(m, "GameEnvironment")
        .def(py::init<int, int>())
        .def("reset", &GameEnvironment::reset)
        .def("getState", &GameEnvironment::getState)
        .def("step", &GameEnvironment::step)
        .def("enable_rendering", &GameEnvironment::enableRendering)
        .def("render_window", &GameEnvironment::renderWindow)
        .def("is_window_open", &GameEnvironment::isWindowOpen)
        .def("handle_window_events", &GameEnvironment::handleWindowEvents);
}