#include "websocket_server.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifndef _WIN32
#    include <netinet/tcp.h>
#endif

namespace a2fws {
namespace {

std::string ToLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string Trim(std::string_view s) {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return std::string{s.substr(start, end - start)};
}

std::array<std::uint8_t, 20> Sha1(std::string_view input) {
    auto leftRotate = [](std::uint32_t value, std::uint32_t bits) -> std::uint32_t {
        return (value << bits) | (value >> (32 - bits));
    };

    std::uint64_t bitLen = static_cast<std::uint64_t>(input.size()) * 8;
    std::vector<std::uint8_t> msg(input.begin(), input.end());
    msg.push_back(0x80);
    while ((msg.size() % 64) != 56) {
        msg.push_back(0x00);
    }
    for (int i = 7; i >= 0; --i) {
        msg.push_back(static_cast<std::uint8_t>((bitLen >> (i * 8)) & 0xff));
    }

    std::uint32_t h0 = 0x67452301;
    std::uint32_t h1 = 0xEFCDAB89;
    std::uint32_t h2 = 0x98BADCFE;
    std::uint32_t h3 = 0x10325476;
    std::uint32_t h4 = 0xC3D2E1F0;

    std::array<std::uint32_t, 80> w{};
    for (std::size_t chunk = 0; chunk < msg.size(); chunk += 64) {
        for (int i = 0; i < 16; ++i) {
            const std::size_t idx = chunk + (i * 4);
            w[i] = (static_cast<std::uint32_t>(msg[idx]) << 24) |
                   (static_cast<std::uint32_t>(msg[idx + 1]) << 16) |
                   (static_cast<std::uint32_t>(msg[idx + 2]) << 8) |
                   (static_cast<std::uint32_t>(msg[idx + 3]));
        }
        for (int i = 16; i < 80; ++i) {
            w[i] = leftRotate(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        }

        std::uint32_t a = h0;
        std::uint32_t b = h1;
        std::uint32_t c = h2;
        std::uint32_t d = h3;
        std::uint32_t e = h4;

        for (int i = 0; i < 80; ++i) {
            std::uint32_t f = 0;
            std::uint32_t k = 0;
            if (i < 20) {
                f = (b & c) | ((~b) & d);
                k = 0x5A827999;
            } else if (i < 40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1;
            } else if (i < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDC;
            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6;
            }

            const std::uint32_t temp = leftRotate(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = leftRotate(b, 30);
            b = a;
            a = temp;
        }

        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
    }

    std::array<std::uint8_t, 20> digest{};
    auto storeBE32 = [&](std::uint32_t v, std::size_t offset) {
        digest[offset] = static_cast<std::uint8_t>((v >> 24) & 0xff);
        digest[offset + 1] = static_cast<std::uint8_t>((v >> 16) & 0xff);
        digest[offset + 2] = static_cast<std::uint8_t>((v >> 8) & 0xff);
        digest[offset + 3] = static_cast<std::uint8_t>((v)&0xff);
    };
    storeBE32(h0, 0);
    storeBE32(h1, 4);
    storeBE32(h2, 8);
    storeBE32(h3, 12);
    storeBE32(h4, 16);
    return digest;
}

std::string Base64Encode(const std::uint8_t* data, std::size_t len) {
    static constexpr char kChars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (std::size_t i = 0; i < len; i += 3) {
        const std::uint32_t octetA = i < len ? data[i] : 0;
        const std::uint32_t octetB = (i + 1) < len ? data[i + 1] : 0;
        const std::uint32_t octetC = (i + 2) < len ? data[i + 2] : 0;
        const std::uint32_t triple = (octetA << 16) | (octetB << 8) | octetC;
        out.push_back(kChars[(triple >> 18) & 0x3f]);
        out.push_back(kChars[(triple >> 12) & 0x3f]);
        out.push_back((i + 1) < len ? kChars[(triple >> 6) & 0x3f] : '=');
        out.push_back((i + 2) < len ? kChars[(triple)&0x3f] : '=');
    }
    return out;
}

std::string WebSocketAcceptKey(std::string_view secWebSocketKey) {
    static constexpr std::string_view kGuid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string combined;
    combined.reserve(secWebSocketKey.size() + kGuid.size());
    combined.append(secWebSocketKey);
    combined.append(kGuid);
    const auto digest = Sha1(combined);
    return Base64Encode(digest.data(), digest.size());
}

} // namespace

Socket::Socket(socket_t s) : _s(s) {}

Socket::Socket(Socket&& other) noexcept : _s(other._s) { other._s = kInvalidSocket; }

Socket& Socket::operator=(Socket&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    Close();
    _s = other._s;
    other._s = kInvalidSocket;
    return *this;
}

Socket::~Socket() { Close(); }

bool Socket::Valid() const { return _s != kInvalidSocket; }

socket_t Socket::Native() const { return _s; }

void Socket::Close() {
    if (!Valid()) {
        return;
    }
#ifdef _WIN32
    closesocket(_s);
#else
    close(_s);
#endif
    _s = kInvalidSocket;
}

bool Socket::SetNoDelay(bool enabled) {
    int flag = enabled ? 1 : 0;
    return setsockopt(_s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&flag), sizeof(flag)) == 0;
}

bool Socket::SendAll(const void* data, std::size_t len) {
    const std::uint8_t* ptr = static_cast<const std::uint8_t*>(data);
    std::size_t sentTotal = 0;
    while (sentTotal < len) {
#ifdef _WIN32
        const int sent = send(_s, reinterpret_cast<const char*>(ptr + sentTotal), static_cast<int>(len - sentTotal), 0);
        if (sent == SOCKET_ERROR || sent == 0) {
            return false;
        }
#else
        const ssize_t sent = send(_s, ptr + sentTotal, len - sentTotal, 0);
        if (sent <= 0) {
            return false;
        }
#endif
        sentTotal += static_cast<std::size_t>(sent);
    }
    return true;
}

bool Socket::RecvAll(void* data, std::size_t len) {
    std::uint8_t* ptr = static_cast<std::uint8_t*>(data);
    std::size_t receivedTotal = 0;
    while (receivedTotal < len) {
#ifdef _WIN32
        const int received = recv(_s, reinterpret_cast<char*>(ptr + receivedTotal), static_cast<int>(len - receivedTotal), 0);
        if (received == SOCKET_ERROR || received == 0) {
            return false;
        }
#else
        const ssize_t received = recv(_s, ptr + receivedTotal, len - receivedTotal, 0);
        if (received <= 0) {
            return false;
        }
#endif
        receivedTotal += static_cast<std::size_t>(received);
    }
    return true;
}

bool Socket::RecvUntil(std::string& out, std::string_view delimiter, std::size_t maxBytes) {
    out.clear();
    std::vector<char> buf(1024);
    while (out.size() < maxBytes) {
#ifdef _WIN32
        const int received = recv(_s, buf.data(), static_cast<int>(buf.size()), 0);
        if (received == SOCKET_ERROR || received == 0) {
            return false;
        }
#else
        const ssize_t received = recv(_s, buf.data(), buf.size(), 0);
        if (received <= 0) {
            return false;
        }
#endif
        out.append(buf.data(), buf.data() + received);
        if (out.find(delimiter) != std::string::npos) {
            return true;
        }
    }
    return false;
}

Socket CreateListenSocket(const std::string& host, int port) {
    socket_t s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == kInvalidSocket) {
        return {};
    }

    int opt = 1;
    (void)setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<std::uint16_t>(port));
    if (host == "0.0.0.0") {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else {
        if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
            Socket sock(s);
            sock.Close();
            return {};
        }
    }

    if (bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        Socket sock(s);
        sock.Close();
        return {};
    }
    if (listen(s, SOMAXCONN) != 0) {
        Socket sock(s);
        sock.Close();
        return {};
    }

    return Socket(s);
}

bool PerformServerHandshake(Socket& sock) {
    std::string request;
    if (!sock.RecvUntil(request, "\r\n\r\n", 16 * 1024)) {
        return false;
    }

    std::istringstream iss(request);
    std::string line;
    std::string secKey;
    bool isWebSocket = false;
    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            break;
        }
        const auto colonPos = line.find(':');
        if (colonPos == std::string::npos) {
            continue;
        }
        const auto key = ToLower(Trim(std::string_view(line).substr(0, colonPos)));
        const auto value = Trim(std::string_view(line).substr(colonPos + 1));
        if (key == "sec-websocket-key") {
            secKey = value;
        } else if (key == "upgrade" && ToLower(value) == "websocket") {
            isWebSocket = true;
        }
    }

    if (!isWebSocket || secKey.empty()) {
        return false;
    }

    const auto accept = WebSocketAcceptKey(secKey);
    std::ostringstream resp;
    resp << "HTTP/1.1 101 Switching Protocols\r\n"
         << "Upgrade: websocket\r\n"
         << "Connection: Upgrade\r\n"
         << "Sec-WebSocket-Accept: " << accept << "\r\n"
         << "\r\n";
    const auto respStr = resp.str();
    return sock.SendAll(respStr.data(), respStr.size());
}

bool ReadFrame(Socket& sock, Frame& outFrame, std::size_t maxPayloadBytes) {
    std::uint8_t header[2] = {};
    if (!sock.RecvAll(header, 2)) {
        return false;
    }

    const bool fin = (header[0] & 0x80) != 0;
    const auto opcode = static_cast<Opcode>(header[0] & 0x0f);
    const bool masked = (header[1] & 0x80) != 0;
    std::uint64_t payloadLen = (header[1] & 0x7f);

    if (!fin) {
        return false; // no fragmentation support
    }

    if (payloadLen == 126) {
        std::uint8_t ext[2] = {};
        if (!sock.RecvAll(ext, 2)) {
            return false;
        }
        payloadLen = (static_cast<std::uint64_t>(ext[0]) << 8) | static_cast<std::uint64_t>(ext[1]);
    } else if (payloadLen == 127) {
        std::uint8_t ext[8] = {};
        if (!sock.RecvAll(ext, 8)) {
            return false;
        }
        payloadLen = 0;
        for (int i = 0; i < 8; ++i) {
            payloadLen = (payloadLen << 8) | static_cast<std::uint64_t>(ext[i]);
        }
    }

    if (payloadLen > maxPayloadBytes) {
        return false;
    }

    std::array<std::uint8_t, 4> maskKey{};
    if (masked) {
        if (!sock.RecvAll(maskKey.data(), maskKey.size())) {
            return false;
        }
    }

    outFrame.fin = fin;
    outFrame.opcode = opcode;
    outFrame.payload.resize(static_cast<std::size_t>(payloadLen));
    if (payloadLen > 0) {
        if (!sock.RecvAll(outFrame.payload.data(), static_cast<std::size_t>(payloadLen))) {
            return false;
        }
    }

    if (masked) {
        for (std::size_t i = 0; i < outFrame.payload.size(); ++i) {
            outFrame.payload[i] ^= maskKey[i % 4];
        }
    }
    return true;
}

bool SendFrame(Socket& sock, Opcode opcode, const void* payload, std::size_t payloadLen) {
    std::vector<std::uint8_t> frame;
    frame.reserve(14 + payloadLen);
    frame.push_back(static_cast<std::uint8_t>(0x80 | (static_cast<std::uint8_t>(opcode) & 0x0f)));

    if (payloadLen <= 125) {
        frame.push_back(static_cast<std::uint8_t>(payloadLen));
    } else if (payloadLen <= 0xffff) {
        frame.push_back(126);
        frame.push_back(static_cast<std::uint8_t>((payloadLen >> 8) & 0xff));
        frame.push_back(static_cast<std::uint8_t>((payloadLen)&0xff));
    } else {
        frame.push_back(127);
        for (int i = 7; i >= 0; --i) {
            frame.push_back(static_cast<std::uint8_t>((static_cast<std::uint64_t>(payloadLen) >> (i * 8)) & 0xff));
        }
    }

    const std::size_t headerSize = frame.size();
    frame.resize(headerSize + payloadLen);
    if (payloadLen > 0) {
        std::memcpy(frame.data() + headerSize, payload, payloadLen);
    }

    return sock.SendAll(frame.data(), frame.size());
}

} // namespace a2fws

