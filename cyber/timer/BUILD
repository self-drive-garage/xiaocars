#load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_cc_library", "apollo_package")

package(default_visibility = ["//visibility:public"])

apollo_cc_library(
    name = "cyber_timer",
    srcs = [
        "timer.cc",
        "timing_wheel.cc",
    ],
    hdrs = [
        "timer.h",
        "timer_bucket.h",
        "timer_task.h",
        "timing_wheel.h",
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
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

#apollo_package()
#cpplint()
