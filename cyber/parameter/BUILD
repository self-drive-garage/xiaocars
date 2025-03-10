#load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_cc_binary", "apollo_cc_library", "apollo_package")

package(default_visibility = ["//visibility:public"])

apollo_cc_library(
    name = "cyber_parameter",
    srcs = [
        "parameter.cc",
        "parameter_client.cc",
        "parameter_server.cc",
    ],
    hdrs = [
        "parameter.h",
        "parameter_client.h",
        "parameter_server.h",
        "parameter_service_names.h",
    ],
    deps = [
        "//cyber/message:cyber_message",
        "//cyber/node:cyber_node",
        "//cyber/proto:parameter_cc_proto",
        "//cyber/service:cyber_service",
        "@fastdds",
    ],
)

cc_test(
    name = "parameter_test",
    size = "small",
    srcs = ["parameter_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "parameter_client_test",
    size = "small",
    srcs = ["parameter_client_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "parameter_server_test",
    size = "small",
    srcs = ["parameter_server_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

#apollo_package()
#cpplint()
