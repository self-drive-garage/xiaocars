#load("//tools:cpplint.bzl", "cpplint")
##load("//tools:apollo_package.bzl", "cc_library", "apollo_package", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cyber_timer",
    srcs = [
        "timer.cc",
        "timing_wheel.cc",
    ],
    hdrs = [
        "timer.h",
        "timer_task.h",
        "timer_bucket.h",
        "timing_wheel.h"
    ],
    deps = [
        "//cyber/common:cyber_common",
        "//cyber/task:cyber_task",
        "//cyber/time:cyber_time",
    ],
)

cc_test(
    name = "timer_test",
    size = "small",
    srcs = ["timer_test.cc"],
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
    linkstatic = True,
)

#apollo_package()
#cpplint()
