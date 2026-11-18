// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/doctor.h"

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
using mouse::nn::doctor_report;
using mouse::nn::DoctorFacts;
using mouse::nn::missing_gpu_libraries;
using mouse::nn::required_gpu_libraries;
bool contains(const std::string& text, const char* piece) {
    return text.find(piece) != std::string::npos;
}
std::filesystem::path fresh_directory(const char* name) {
    const auto root = std::filesystem::temp_directory_path() / "mouse-doctor" / name;
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    return root;
}
DoctorFacts nvidia_ready() {
    DoctorFacts facts;
    facts.version = "2.0.0";
    facts.nvidia_driver = true;
    facts.plugin_build = true;
    facts.plugin_present = true;
    facts.gpu_directories = {std::filesystem::path{"C:/mouse/gpu"}};
    facts.gpu_name = "NVIDIA GeForce RTX 5070";
    return facts;
}
}  // namespace

TEST_CASE("Doctor lists the libraries that are in none of the GPU folders", "[nn][doctor]") {
    const auto first = fresh_directory("first");
    const auto second = fresh_directory("second");
    const auto required = required_gpu_libraries();
    REQUIRE(required.size() >= 3);
    // Everything but the first library, split over two folders.
    for (std::size_t index = 1; index < required.size(); ++index)
        std::ofstream{(index % 2 == 0 ? first : second) / required[index]} << "library";
    const std::vector<std::filesystem::path> directories{first, second};
    const auto missing = missing_gpu_libraries(directories);
    REQUIRE(missing.size() == 1);
    CHECK(missing.front() == required.front());
    CHECK(missing_gpu_libraries({}).size() == required.size());
}

TEST_CASE("Doctor reports a complete NVIDIA setup as ready", "[nn][doctor]") {
    const auto report = doctor_report(nvidia_ready());
    CHECK(report.ready);
    CHECK(contains(report.text, "NVIDIA GeForce RTX 5070"));
    CHECK(contains(report.text, "Result: ready"));
}

TEST_CASE("Doctor names each missing library and the way to get it", "[nn][doctor]") {
    auto facts = nvidia_ready();
    facts.missing_libraries = {"nvinfer_10.dll", "cudnn64_9.dll"};
    facts.gpu_name.clear();
    const auto report = doctor_report(facts);
    CHECK_FALSE(report.ready);
    CHECK(contains(report.text, "nvinfer_10.dll"));
    CHECK(contains(report.text, "cudnn64_9.dll"));
    CHECK(contains(report.text, "GpuDllDirectories"));
#ifdef _WIN32
    CHECK(contains(report.text, "setup-gpu.ps1"));
#else
    // The setup script is Windows PowerShell; no other system is told to run it.
    CHECK_FALSE(contains(report.text, "setup-gpu.ps1"));
#endif
    CHECK(contains(report.text, "Result: not ready"));
}

TEST_CASE("Doctor says so when no GPU library folder exists at all", "[nn][doctor]") {
    auto facts = nvidia_ready();
    facts.gpu_directories.clear();
    facts.missing_libraries = {"cudart64_12.dll"};
    const auto report = doctor_report(facts);
    CHECK_FALSE(report.ready);
    CHECK(contains(report.text, "GpuDllDirectories"));
    CHECK(contains(report.text, "no gpu folder"));
}

TEST_CASE("Doctor tells a computer without an NVIDIA driver what it can run instead",
          "[nn][doctor]") {
    auto facts = nvidia_ready();
    facts.nvidia_driver = false;
    const auto report = doctor_report(facts);
    CHECK_FALSE(report.ready);
    CHECK(contains(report.text, "NVIDIA driver"));
#ifdef _WIN32
    CHECK(contains(report.text, "DirectML"));
#else
    // DirectML is Windows only.
    CHECK_FALSE(contains(report.text, "DirectML"));
    CHECK(contains(report.text, "cpu"));
#endif
}

TEST_CASE("Doctor on a build with no GPU provider says it runs on the CPU only", "[nn][doctor]") {
    DoctorFacts facts;
    facts.version = "2.0.0";
    facts.cpu_only_build = true;
    const auto report = doctor_report(facts);
    CHECK(report.ready);
    CHECK(contains(report.text, "CPU-only build"));
    CHECK(contains(report.text, "Backend"));
    CHECK(contains(report.text, "cpu"));
    // It has no NVIDIA libraries to miss.
    CHECK_FALSE(contains(report.text, "Missing library"));
    CHECK_FALSE(contains(report.text, "NVIDIA driver"));
}

TEST_CASE("Doctor fails a plugin build whose plugin library is gone", "[nn][doctor]") {
    auto facts = nvidia_ready();
    facts.plugin_present = false;
    const auto report = doctor_report(facts);
    CHECK_FALSE(report.ready);
    CHECK(contains(report.text, "mouse_moe_trt"));
}

TEST_CASE("Doctor on a DirectML build needs only DirectML.dll and a GPU", "[nn][doctor]") {
    DoctorFacts facts;
    facts.version = "2.0.0";
    facts.directml_build = true;
    facts.directml_present = true;
    facts.gpu_name = "AMD Radeon RX 6800S";
    auto report = doctor_report(facts);
    CHECK(report.ready);
    CHECK(contains(report.text, "AMD Radeon RX 6800S"));
    CHECK_FALSE(contains(report.text, "setup-gpu.ps1"));

    facts.gpu_name.clear();
    facts.gpu_problem = "DirectML found no hardware GPU adapter";
    report = doctor_report(facts);
    CHECK_FALSE(report.ready);
    CHECK(contains(report.text, "DirectML found no hardware GPU adapter"));
    CHECK(contains(report.text, "Backend"));
}
