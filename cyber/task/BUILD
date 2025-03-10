#load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_cc_library", "apollo_package")

package(default_visibility = ["//visibility:public"])

apollo_cc_library(
    name = "cyber_task",
    srcs = ["task_manager.cc"],
    hdrs = [
        "task.h",
        "task_manager.h",
    ],
    copts = ["-faligned-new"],
    deps = [
        "//cyber/scheduler:cyber_scheduler",
    ],
)

cc_test(
    name = "task_test",
    size = "small",
    srcs = ["task_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

#apollo_package()
#cpplint()
