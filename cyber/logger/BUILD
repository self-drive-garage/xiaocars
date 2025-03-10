#load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_cc_binary", "apollo_cc_library", "apollo_package")

package(default_visibility = ["//visibility:public"])

apollo_cc_library(
    name = "cyber_logger",
    srcs = [
        "async_logger.cc",
        "log_file_object.cc",
        "logger.cc",
        "logger_util.cc",
    ],
    hdrs = [
        "async_logger.h",
        "log_file_object.h",
        "logger.h",
        "logger_util.h",
    ],
    deps = [
        "//cyber:cyber_binary",
        "//cyber/base:cyber_base",
        "//cyber/common:cyber_common",
    ],
)

cc_test(
    name = "logger_test",
    size = "small",
    srcs = ["logger_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "async_logger_test",
    size = "small",
    srcs = ["async_logger_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "log_file_object_test",
    size = "small",
    srcs = ["log_file_object_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "logger_util_test",
    size = "small",
    srcs = ["logger_util_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

#apollo_package()
#cpplint()
