#load("//tools:cpplint.bzl", "cpplint")
#load("//tools:apollo_package.bzl", "apollo_package", "cc_library", "cc_test", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cyber_logger",
    srcs = [
        "async_logger.cc",
        "log_file_object.cc",
        "logger_util.cc",
        "logger.cc",
    ],
    hdrs = [
        "async_logger.h",
        "log_file_object.h",
        "logger.h",
        "logger_util.h",
    ],
    deps = [
        "//cyber:cyber_binary",
        "//cyber/common:cyber_common",
        "//cyber/base:cyber_base",
    ],
)

cc_test(
    name = "logger_test",
    size = "small",
    srcs = ["logger_test.cc"],
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
    linkstatic = True,
)

cc_test(
    name = "async_logger_test",
    size = "small",
    srcs = ["async_logger_test.cc"],
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
    linkstatic = True,
)

cc_test(
    name = "log_file_object_test",
    size = "small",
    srcs = ["log_file_object_test.cc"],
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
    linkstatic = True,
)

cc_test(
    name = "logger_util_test",
    size = "small",
    srcs = ["logger_util_test.cc"],
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
    linkstatic = True,
)

#apollo_package()
#cpplint()
