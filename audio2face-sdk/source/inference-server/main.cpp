#include "inference_sessions.h"
#include "websocket_server.h"

#include <cxxopts.hpp>
#include <nlohmann/json.hpp>

#include <cctype>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>

#ifdef _WIN32
#    define NOMINMAX
#    include <winsock2.h>
#endif

namespace {

using json = nlohmann::json;

std::string LowerAscii(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

std::string CanonicalizePath(std::string s) {
    for (char& c : s) {
        if (c == '\\') {
            c = '/';
        }
#ifdef _WIN32
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
#endif
    }
    while (!s.empty() && (s.back() == '/' || std::isspace(static_cast<unsigned char>(s.back())))) {
        s.pop_back();
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
    }
    if (s.size() >= 2 && s[0] == '.' && s[1] == '/') {
        s.erase(0, 2);
    }
    return s;
}

std::string CanonicalizeExecutionOption(std::string s) {
    s = LowerAscii(std::move(s));
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            out.push_back(c);
        }
    }
    return out;
}

std::optional<std::int64_t> ReadI64LE(const std::uint8_t* data, std::size_t len) {
    if (len < 8) {
        return std::nullopt;
    }
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= (static_cast<std::uint64_t>(data[i]) << (8 * i));
    }
    return static_cast<std::int64_t>(v);
}

void SendJson(a2fws::Socket& sock, const json& msg) {
    const auto text = msg.dump();
    (void)a2fws::SendFrame(sock, a2fws::Opcode::Text, text.data(), text.size());
}

bool TryParseFrameRate(const json& v, std::size_t& outNum, std::size_t& outDen, std::string& outError) {
    if (v.is_number_integer() || v.is_number_unsigned()) {
        const auto fps = v.get<std::int64_t>();
        if (fps <= 0) {
            outError = "fps must be > 0";
            return false;
        }
        outNum = static_cast<std::size_t>(fps);
        outDen = 1;
        return true;
    }
    if (v.is_object()) {
        if (!v.contains("numerator") || !v.contains("denominator")) {
            outError = "frame_rate must contain numerator and denominator";
            return false;
        }
        if (!(v["numerator"].is_number_integer() || v["numerator"].is_number_unsigned()) ||
            !(v["denominator"].is_number_integer() || v["denominator"].is_number_unsigned())) {
            outError = "frame_rate numerator/denominator must be integers";
            return false;
        }
        const auto num = v["numerator"].get<std::int64_t>();
        const auto den = v["denominator"].get<std::int64_t>();
        if (num <= 0 || den <= 0) {
            outError = "frame_rate numerator/denominator must be > 0";
            return false;
        }
        outNum = static_cast<std::size_t>(num);
        outDen = static_cast<std::size_t>(den);
        return true;
    }
    outError = "fps must be an integer or an object {numerator,denominator}";
    return false;
}

bool ValidateStartSessionRequest(const json& request, const json& sessionStarted, std::string& outError) {
    if (request.contains("model")) {
        if (!request["model"].is_string()) {
            outError = "StartSession.model must be a string";
            return false;
        }
        const auto reqModel = CanonicalizePath(request["model"].get<std::string>());
        const auto actualModel = CanonicalizePath(sessionStarted.value("model", ""));
        if (!actualModel.empty() && reqModel != actualModel) {
            outError = "Requested model does not match server model";
            return false;
        }
    }

    if (request.contains("fps") || request.contains("frame_rate")) {
        std::size_t reqNum = 0;
        std::size_t reqDen = 0;
        std::string parseErr;
        const auto& fpsVal = request.contains("frame_rate") ? request["frame_rate"] : request["fps"];
        if (!TryParseFrameRate(fpsVal, reqNum, reqDen, parseErr)) {
            outError = parseErr;
            return false;
        }

        if (!sessionStarted.contains("frame_rate") || !sessionStarted["frame_rate"].is_object()) {
            outError = "Internal error: missing frame_rate in SessionStarted";
            return false;
        }
        const auto& fr = sessionStarted["frame_rate"];
        if (!fr.contains("numerator") || !fr.contains("denominator")) {
            outError = "Internal error: invalid frame_rate in SessionStarted";
            return false;
        }
        const auto actualNum = fr["numerator"].get<std::size_t>();
        const auto actualDen = fr["denominator"].get<std::size_t>();
        if (reqNum != actualNum || reqDen != actualDen) {
            std::ostringstream oss;
            oss << "Requested frame_rate " << reqNum << "/" << reqDen << " does not match server "
                << actualNum << "/" << actualDen;
            outError = oss.str();
            return false;
        }
    }

    if (request.contains("options")) {
        if (!request["options"].is_object()) {
            outError = "StartSession.options must be an object";
            return false;
        }
        if (!sessionStarted.contains("options") || !sessionStarted["options"].is_object()) {
            outError = "Internal error: missing options in SessionStarted";
            return false;
        }

        const auto& reqOpt = request["options"];
        const auto& actualOpt = sessionStarted["options"];

        if (reqOpt.contains("use_gpu_solver")) {
            if (!reqOpt["use_gpu_solver"].is_boolean()) {
                outError = "options.use_gpu_solver must be boolean";
                return false;
            }
            const auto reqGpu = reqOpt["use_gpu_solver"].get<bool>();
            const auto actualGpu = actualOpt.value("use_gpu_solver", true);
            if (reqGpu != actualGpu) {
                outError = "options.use_gpu_solver does not match server";
                return false;
            }
        }

        if (reqOpt.contains("execution_option")) {
            if (!reqOpt["execution_option"].is_string()) {
                outError = "options.execution_option must be a string";
                return false;
            }
            const auto reqExec = CanonicalizeExecutionOption(reqOpt["execution_option"].get<std::string>());
            const auto actualExec = CanonicalizeExecutionOption(actualOpt.value("execution_option", ""));
            if (!actualExec.empty() && reqExec != actualExec) {
                outError = "options.execution_option does not match server";
                return false;
            }
        }
    }

    return true;
}

void HandleClient(a2fws::Socket client, a2fserver::SessionPool& pool) {
    client.SetNoDelay(true);
    if (!a2fws::PerformServerHandshake(client)) {
        return;
    }

    std::optional<std::size_t> sessionIndex;
    constexpr std::size_t kMaxPayload = 4 * 1024 * 1024; // 4MB per message.

    while (true) {
        a2fws::Frame frame;
        if (!a2fws::ReadFrame(client, frame, kMaxPayload)) {
            break;
        }

        if (frame.opcode == a2fws::Opcode::Ping) {
            (void)a2fws::SendFrame(client, a2fws::Opcode::Pong, frame.payload.data(), frame.payload.size());
            continue;
        }
        if (frame.opcode == a2fws::Opcode::Close) {
            (void)a2fws::SendFrame(client, a2fws::Opcode::Close, nullptr, 0);
            break;
        }

        if (frame.opcode == a2fws::Opcode::Text) {
            json msg;
            try {
                msg = json::parse(std::string(frame.payload.begin(), frame.payload.end()));
            } catch (const std::exception& e) {
                SendJson(client, json{{"type", "Error"}, {"message", std::string("Invalid JSON: ") + e.what()}});
                continue;
            }

            const auto type = msg.value("type", "");
            if (type == "StartSession") {
                if (sessionIndex) {
                    SendJson(client, json{{"type", "Error"}, {"message", "Session already started for this connection"}});
                    continue;
                }

                auto acquired = pool.Acquire(&client);
                if (!acquired) {
                    SendJson(client, json{{"type", "Error"}, {"message", "Server busy (no free sessions)"}});
                    continue;
                }
                const auto idx = *acquired;
                const auto started = pool.Get(idx).DescribeSessionStarted();

                std::string validationError;
                if (!ValidateStartSessionRequest(msg, started, validationError)) {
                    pool.Release(idx);
                    SendJson(client, json{{"type", "Error"}, {"message", validationError}});
                    continue;
                }

                sessionIndex = idx;
                SendJson(client, started);
                continue;
            }

            if (type == "EndSession") {
                if (!sessionIndex) {
                    SendJson(client, json{{"type", "Error"}, {"message", "No active session for this connection"}});
                    continue;
                }
                const auto sid = pool.Get(*sessionIndex).SessionId();
                if (msg.contains("session_id")) {
                    if (!msg["session_id"].is_string()) {
                        SendJson(client, json{{"type", "Error"}, {"message", "EndSession.session_id must be a string"}});
                        continue;
                    }
                    if (msg["session_id"].get<std::string>() != sid) {
                        SendJson(client, json{{"type", "Error"}, {"message", "EndSession.session_id does not match active session"}});
                        continue;
                    }
                }
                pool.Release(*sessionIndex);
                sessionIndex.reset();

                SendJson(client, json{{"type", "SessionEnded"}, {"session_id", sid}});
                continue;
            }

            SendJson(client, json{{"type", "Error"}, {"message", "Unknown message type"}});
            continue;
        }

        if (frame.opcode == a2fws::Opcode::Binary) {
            if (!sessionIndex) {
                SendJson(client, json{{"type", "Error"}, {"message", "StartSession must be called before PushAudio"}});
                continue;
            }
            if (frame.payload.size() < 8 || ((frame.payload.size() - 8) % 2) != 0) {
                SendJson(client, json{{"type", "Error"}, {"message", "Invalid PushAudio binary payload"}});
                continue;
            }

            const auto startSample = ReadI64LE(frame.payload.data(), frame.payload.size());
            if (!startSample) {
                SendJson(client, json{{"type", "Error"}, {"message", "Invalid PushAudio header"}});
                continue;
            }

            const auto sampleCount = (frame.payload.size() - 8) / 2;
            const auto* pcm = reinterpret_cast<const std::int16_t*>(frame.payload.data() + 8);
            (void)pool.Get(*sessionIndex).PushAudio(*startSample, pcm, sampleCount);
            continue;
        }
    }

    if (sessionIndex) {
        pool.Release(*sessionIndex);
    }
}

std::optional<nva2f::IGeometryExecutor::ExecutionOption> ParseExecutionOption(const std::string& s) {
    std::string v;
    v.reserve(s.size());
    for (char c : s) {
        v.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (v == "skintongue") {
        return nva2f::IGeometryExecutor::ExecutionOption::SkinTongue;
    }
    if (v == "skin") {
        return nva2f::IGeometryExecutor::ExecutionOption::Skin;
    }
    if (v == "tongue") {
        return nva2f::IGeometryExecutor::ExecutionOption::Tongue;
    }
    if (v == "none") {
        return nva2f::IGeometryExecutor::ExecutionOption::None;
    }
    return std::nullopt;
}

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    WSADATA wsaData{};
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed\n";
        return 1;
    }
#endif

    try {
        a2fserver::ServerConfig config;
        config.modelJsonPath = "_data/generated/audio2face-sdk/samples/data/mark/model.json";

        cxxopts::Options options("audio2face-inference-server", "WebSocket Audio2Face blendshape inference server");
        options.add_options()
            ("host", "Bind host (IPv4)", cxxopts::value<std::string>()->default_value(config.host))
            ("port", "Bind port", cxxopts::value<int>()->default_value(std::to_string(config.port)))
            ("cuda_device", "CUDA device id", cxxopts::value<int>()->default_value(std::to_string(config.cudaDevice)))
            ("max_sessions", "Max concurrent sessions", cxxopts::value<std::size_t>()->default_value(std::to_string(config.maxSessions)))
            ("model", "Path to model.json", cxxopts::value<std::string>()->default_value(config.modelJsonPath))
            ("diffusion", "Use diffusion model", cxxopts::value<bool>()->default_value("false"))
            ("identity", "Diffusion identity index", cxxopts::value<std::size_t>()->default_value("0"))
            ("constant_noise", "Diffusion constant noise", cxxopts::value<bool>()->default_value("true"))
            ("execution_option", "Execution option: SkinTongue|Skin|Tongue|None", cxxopts::value<std::string>()->default_value("SkinTongue"))
            ("fps", "Frame rate numerator (denominator is 1)", cxxopts::value<std::size_t>()->default_value("60"))
            ("help", "Print help");

        const auto parsed = options.parse(argc, argv);
        if (parsed.count("help")) {
            std::cout << options.help() << "\n";
            return 0;
        }

        config.host = parsed["host"].as<std::string>();
        config.port = parsed["port"].as<int>();
        config.cudaDevice = parsed["cuda_device"].as<int>();
        config.maxSessions = parsed["max_sessions"].as<std::size_t>();
        config.modelJsonPath = parsed["model"].as<std::string>();
        config.diffusion = parsed["diffusion"].as<bool>();
        config.diffusionIdentity = parsed["identity"].as<std::size_t>();
        config.diffusionConstantNoise = parsed["constant_noise"].as<bool>();
        config.fpsNumerator = parsed["fps"].as<std::size_t>();
        config.fpsDenominator = 1;

        const auto optStr = parsed["execution_option"].as<std::string>();
        const auto opt = ParseExecutionOption(optStr);
        if (!opt) {
            std::cerr << "Unsupported execution option: " << optStr << "\n";
            return 1;
        }
        config.executionOption = *opt;

        std::cout << "Starting Audio2Face inference server on ws://" << config.host << ":" << config.port << "\n";
        std::cout << "Model: " << config.modelJsonPath << "\n";
        std::cout << "Max sessions: " << config.maxSessions << "\n";

        a2fserver::SessionPool pool;
        if (!pool.Init(config)) {
            return 1;
        }

        a2fws::Socket listener = a2fws::CreateListenSocket(config.host, config.port);
        if (!listener.Valid()) {
            std::cerr << "Failed to bind/listen on " << config.host << ":" << config.port << "\n";
            return 1;
        }

        while (true) {
            sockaddr_in clientAddr{};
#ifdef _WIN32
            int addrLen = sizeof(clientAddr);
            SOCKET clientSock = accept(listener.Native(), reinterpret_cast<sockaddr*>(&clientAddr), &addrLen);
#else
            socklen_t addrLen = sizeof(clientAddr);
            int clientSock = accept(listener.Native(), reinterpret_cast<sockaddr*>(&clientAddr), &addrLen);
#endif
            if (clientSock == a2fws::kInvalidSocket) {
                continue;
            }

            std::thread([client = a2fws::Socket(clientSock), &pool]() mutable {
                HandleClient(std::move(client), pool);
            }).detach();
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
