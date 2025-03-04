#load("//tools:cpplint.bzl", "cpplint")
#load("//tools:apollo_package.bzl", "apollo_package", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cyber_event",
    srcs = ["perf_event_cache.cc"],
    hdrs = ["perf_event_cache.h", "perf_event.h"],
    deps = [
        "//cyber:cyber_state",
        "//cyber/base:cyber_base",
        "//cyber/common:cyber_common",
        "//cyber/time:cyber_time",
    ],
)

#apollo_package()
#cpplint()
