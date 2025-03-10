#load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_cc_library", "apollo_package")

package(default_visibility = ["//visibility:public"])

apollo_cc_library(
    name = "cyber_event",
    srcs = ["perf_event_cache.cc"],
    hdrs = [
        "perf_event.h",
        "perf_event_cache.h",
    ],
    deps = [
        "//cyber:cyber_state",
        "//cyber/base:cyber_base",
        "//cyber/common:cyber_common",
        "//cyber/time:cyber_time",
    ],
)

#apollo_package()
#cpplint()
