#load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_cc_library", "apollo_package")

package(default_visibility = ["//visibility:public"])

apollo_cc_library(
    name = "cyber_blocker",
    srcs = [
        "blocker_manager.cc",
    ],
    hdrs = [
        "blocker.h",
        "blocker_manager.h",
        "intra_reader.h",
        "intra_writer.h",
    ],
)

cc_test(
    name = "blocker_manager_test",
    size = "small",
    srcs = ["blocker_manager_test.cc"],
    deps = [
        "//cyber",
        "//cyber/proto:unit_test_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "blocker_test",
    size = "small",
    srcs = ["blocker_test.cc"],
    deps = [
        "//cyber",
        "//cyber/proto:unit_test_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

#apollo_package()
#cpplint()
