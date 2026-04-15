#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/beast/websocket.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "orbital/control/lqr_controller.hpp"
#include "orbital/control/reward_model.hpp"
#include "orbital/simulation/orbital_dynamics.hpp"
#include "orbital/simulation/orbital_state.hpp"

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
using tcp = boost::asio::ip::tcp;

namespace {

class TelemetrySimulator {
public:
    TelemetrySimulator()
        : dynamics_({0.14, 0.00045, 255.0}), reward_model_({0.0003, 0.04, 0.01, 12.0}) {
        state_.position_m = {6'805'000.0, -500.0, 250.0};
        state_.velocity_mps = {0.2, 7'680.0, 0.04};
        target_.target_position_m = {6'800'000.0, 0.0, 0.0};
        target_.target_velocity_mps = {0.0, 7'670.0, 0.0};
    }

    [[nodiscard]] std::string next_payload() {
        const std::array<double, 3> position_error{
            target_.target_position_m[0] - state_.position_m[0],
            target_.target_position_m[1] - state_.position_m[1],
            target_.target_position_m[2] - state_.position_m[2],
        };

        const std::array<double, 3> velocity_error{
            target_.target_velocity_mps[0] - state_.velocity_mps[0],
            target_.target_velocity_mps[1] - state_.velocity_mps[1],
            target_.target_velocity_mps[2] - state_.velocity_mps[2],
        };

        orbital::control::ThrusterCommand command;
        command.thrust_axis = controller_.compute(position_error, velocity_error);

        state_ = dynamics_.propagate(state_, command, kDtSeconds);

        const bool success =
            std::abs(position_error[0]) < 3.0 && std::abs(position_error[1]) < 3.0 && std::abs(position_error[2]) < 3.0;

        const double reward = reward_model_.compute_step_reward(state_, command, target_, success);
        const double policy_std = 0.22 + 0.08 * std::exp(-0.002 * static_cast<double>(step_));

        std::ostringstream payload;
        payload << '{';
        payload << "\"step\":" << step_ << ',';
        payload << "\"mission_time_s\":" << state_.mission_time_s << ',';
        payload << "\"position_m\":[" << state_.position_m[0] << ',' << state_.position_m[1] << ',' << state_.position_m[2] << "],";
        payload << "\"velocity_mps\":[" << state_.velocity_mps[0] << ',' << state_.velocity_mps[1] << ',' << state_.velocity_mps[2] << "],";
        payload << "\"policy_std\":" << policy_std << ',';
        payload << "\"reward\":" << reward;
        payload << '}';

        ++step_;
        return payload.str();
    }

private:
    static constexpr double kDtSeconds = 0.10;

    orbital::simulation::OrbitalState3DOF state_{};
    orbital::control::MissionTarget target_{};
    orbital::simulation::OrbitalDynamics3DOF dynamics_;
    orbital::control::RewardModel reward_model_;
    orbital::control::LqrBaselineController3DOF controller_;
    int step_ = 0;
};

void send_response(
    tcp::socket& socket,
    const http::status status,
    const std::string& body,
    const std::string& content_type = "application/json"
) {
    http::response<http::string_body> response{status, 11};
    response.set(http::field::server, "orbital-backend");
    response.set(http::field::content_type, content_type);
    response.keep_alive(false);
    response.body() = body;
    response.prepare_payload();
    http::write(socket, response);
}

void handle_websocket(tcp::socket socket, http::request<http::string_body> request) {
    websocket::stream<tcp::socket> ws(std::move(socket));
    ws.set_option(websocket::stream_base::timeout::suggested(beast::role_type::server));
    ws.accept(request);

    TelemetrySimulator simulator;
    for (int step = 0; step < 3000; ++step) {
        ws.text(true);
        const auto payload = simulator.next_payload();
        ws.write(boost::asio::buffer(payload));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    ws.close(websocket::close_code::normal);
}

void handle_session(tcp::socket socket) {
    beast::error_code error;
    beast::flat_buffer buffer;

    http::request<http::string_body> request;
    http::read(socket, buffer, request, error);
    if (error) {
        return;
    }

    if (websocket::is_upgrade(request) && request.target() == "/ws/telemetry") {
        handle_websocket(std::move(socket), std::move(request));
        return;
    }

    if (request.method() == http::verb::get && request.target() == "/health") {
        send_response(socket, http::status::ok, "{\"status\":\"ok\",\"service\":\"orbital-backend\"}");
    } else if (request.method() == http::verb::get && request.target() == "/api/telemetry/snapshot") {
        TelemetrySimulator simulator;
        send_response(socket, http::status::ok, simulator.next_payload());
    } else {
        send_response(socket, http::status::not_found, "{\"error\":\"not_found\"}");
    }

    socket.shutdown(tcp::socket::shutdown_send, error);
}

}  // namespace

int main() {
    const auto env_port = std::getenv("ORBITAL_BACKEND_PORT");
    const unsigned short port = static_cast<unsigned short>(env_port != nullptr ? std::atoi(env_port) : 8080);

    boost::asio::io_context io_context{1};
    tcp::acceptor acceptor(io_context, {tcp::v4(), port});

    std::cout << "orbital_backend listening on 0.0.0.0:" << port << std::endl;

    while (true) {
        beast::error_code error;
        tcp::socket socket(io_context);
        acceptor.accept(socket, error);
        if (error) {
            continue;
        }

        std::thread{handle_session, std::move(socket)}.detach();
    }

    return 0;
}
