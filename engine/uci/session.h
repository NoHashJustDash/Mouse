// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <functional>
#include <memory>
#include <stop_token>
#include <string_view>

#include "engine/uci/options.h"
namespace mouse::uci {
// One input owner calls handle/shutdown; the destructor drains all owned threads.
// Output may run on input/publication/worker threads and is serialized by Session.
// A single worker owns inference and a bounded pending turn. Deadline publication
// never waits for provider completion; late failures still fail the whole session.
// BackendFactory runs only on the persistent worker, starting with configured
// weights at construction or backend option changes. Superseded loads are retired
// there; readiness joins the latest configuration without blocking publication. All
// accepted calls and owned threads are joined before shutdown returns.
class Session {
    struct Impl;

   public:
    using Output = std::function<void(std::string_view)>;
    using BackendFactory =
        std::function<Result<std::unique_ptr<nn::Backend>>(const nn::BackendConfig&)>;
    Session(Options, Output, BackendFactory = nn::create_backend);
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&&) = delete;
    Session& operator=(Session&&) = delete;
    [[nodiscard]] bool handle(std::string_view);
    [[nodiscard]] bool report_input_error(std::string_view);
    void shutdown() noexcept;
    [[nodiscard]] int exit_code() const noexcept;
    [[nodiscard]] std::stop_token cancellation_token() const noexcept;

   private:
    std::unique_ptr<Impl> impl_;
};
}  // namespace mouse::uci
