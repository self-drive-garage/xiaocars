#load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_cc_binary", "apollo_cc_library", "apollo_component")

package(default_visibility = ["//visibility:public"])

apollo_component(
    name = "libcommon_component_example.so",
    srcs = ["common_component_example.cc"],
    hdrs = ["common_component_example.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
    ],
)

#apollo_cc_binary(
#    name = "libcommon_component_example.so",
#    linkshared = True,
#    linkstatic = False,
#    deps = [":common_component_example_lib"],
#)
#
#apollo_cc_library(
#    name = "common_component_example_lib",
#    srcs = ["common_component_example.cc"],
#    hdrs = ["common_component_example.h"],
#    visibility = ["//visibility:private"],
#    deps = [
#        "//cyber",
#        "//cyber/examples/proto:examples_cc_proto",
#    ],
#)

apollo_cc_binary(
    name = "channel_test_writer",
    srcs = ["channel_test_writer.cc"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
    ],
)

apollo_cc_binary(
    name = "channel_prediction_writer",
    srcs = ["channel_prediction_writer.cc"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
    ],
)

filegroup(
    name = "conf",
    srcs = [
        ":common.dag",
        ":common.launch",
    ],
)

#apollo_package()

#cpplint()
