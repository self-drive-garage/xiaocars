#load("//tools:cpplint.bzl", "cpplint")
#load("//tools:apollo_package.bzl", "apollo_package", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cyber_component",
    srcs = [
        "timer_component.cc",
    ],
    hdrs = [
        "component.h",
        "component_base.h",
        "timer_component.h",
    ],
    deps = [
        "//cyber/base:cyber_base",
        "//cyber/blocker:cyber_blocker",
        "//cyber/class_loader:cyber_class_loader",
        "//cyber/node:cyber_node",
        "//cyber/proto:component_conf_cc_proto",
        "//cyber/scheduler:cyber_scheduler",
        "//cyber/timer:cyber_timer",
        "//cyber/transport:cyber_transport",
        "@com_github_gflags_gflags//:gflags",
    ],
)

cc_test(
    name = "component_test",
    size = "small",
    srcs = ["component_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "timer_component_test",
    size = "small",
    srcs = ["timer_component_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

#apollo_package()
#cpplint()
