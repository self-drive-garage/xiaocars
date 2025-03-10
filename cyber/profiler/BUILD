#load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_cc_library", "apollo_package")

package(default_visibility = ["//visibility:public"])

apollo_cc_library(
    name = "cyber_profiler",
    srcs = [
        "block.cc",
        "block_manager.cc",
        "frame.cc",
    ],
    hdrs = [
        "block.h",
        "block_manager.h",
        "frame.h",
        "profiler.h",
    ],
    deps = [
        "//cyber/common:cyber_common",
        "//cyber/croutine:cyber_croutine",
    ],
)

cc_test(
    name = "profiler_test",
    size = "small",
    srcs = ["profiler_test.cc"],
    linkstatic = True,
    deps = [
        ":cyber_profiler",
        "@com_google_googletest//:gtest_main",
    ],
)

#apollo_package()
#cpplint()
