#load("//tools:cpplint.bzl", "cpplint")
#load("//tools:apollo_package.bzl", "apollo_package", "cc_library", "cc_binary", "apollo_component")

package(default_visibility = ["//visibility:public"])

# apollo_component(
#     name = "libcommon_component_example.so",
#     srcs = ["common_component_example.cc"],
#     hdrs = ["common_component_example.h"],
#     visibility = ["//visibility:private"],
#     deps = [
#         "//cyber",
#         "//cyber/examples/proto:examples_cc_proto",
#     ],
# )

cc_binary(
    name = "channel_test_writer",
    srcs = ["channel_test_writer.cc"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
    ],
)

cc_binary(
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
