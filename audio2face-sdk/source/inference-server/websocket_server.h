#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#    define NOMINMAX
#    include <winsock2.h>
#    include <ws2tcpip.h>
#else
#    include <arpa/inet.h>
#    include <netinet/in.h>
#    include <sys/socket.h>
#    include <sys/types.h>
#    include <unistd.h>
#endif

namespace a2fws {

#ifdef _WIN32
using socket_t = SOCKET;
constexpr socket_t kInvalidSocket = INVALID_SOCKET;
#else
using socket_t = int;
constexpr socket_t kInvalidSocket = -1;
#endif

class Socket {
public:
    Socket() = default;
    explicit Socket(socket_t s);
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    Socket(Socket&& other) noexcept;
    Socket& operator=(Socket&& other) noexcept;
    ~Socket();

    bool Valid() const;
    socket_t Native() const;
    void Close();

    bool SetNoDelay(bool enabled);
    bool SendAll(const void* data, std::size_t len);
    bool RecvAll(void* data, std::size_t len);
    bool RecvUntil(std::string& out, std::string_view delimiter, std::size_t maxBytes);

private:
    socket_t _s{kInvalidSocket};
};

enum class Opcode : std::uint8_t {
    Continuation = 0x0,
    Text = 0x1,
    Binary = 0x2,
    Close = 0x8,
    Ping = 0x9,
    Pong = 0xA,
};

struct Frame {
    bool fin{true};
    Opcode opcode{Opcode::Binary};
    std::vector<std::uint8_t> payload;
};

Socket CreateListenSocket(const std::string& host, int port);

bool PerformServerHandshake(Socket& sock);
bool ReadFrame(Socket& sock, Frame& outFrame, std::size_t maxPayloadBytes);
bool SendFrame(Socket& sock, Opcode opcode, const void* payload, std::size_t payloadLen);

} // namespace a2fws

