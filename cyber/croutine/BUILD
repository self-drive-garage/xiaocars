#load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_cc_library", "apollo_package")

package(default_visibility = ["//visibility:public"])

apollo_cc_library(
    name = "cyber_croutine",
    srcs = [
        "croutine.cc",
        "detail/routine_context.cc",
    ] + select(
        {
            "@platforms//cpu:x86_64": ["detail/swap_x86_64.S"],
            "@platforms//cpu:aarch64": ["detail/swap_aarch64.S"],
        },
        no_match_error = "Please Build with a Linux aarch64 or x86_64 platform",
    ),
    hdrs = [
        "croutine.h",
        "detail/routine_context.h",
        "routine_factory.h",
    ],
    linkopts = ["-latomic"],
    deps = [
        "//cyber/base:cyber_base",
        "//cyber/common:cyber_common",
        "//cyber/event:cyber_event",
        "//cyber/time:cyber_time",
    ],
)

cc_test(
    name = "croutine_test",
    size = "small",
    srcs = ["croutine_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

#apollo_package()
#cpplint()
