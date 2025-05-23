#load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_cc_library", "apollo_package")

package(default_visibility = ["//visibility:public"])

apollo_cc_library(
    name = "cyber_record",
    srcs = [
        "file/record_file_base.cc",
        "file/record_file_reader.cc",
        "file/record_file_writer.cc",
        "header_builder.cc",
        "record_reader.cc",
        "record_viewer.cc",
        "record_writer.cc",
    ],
    hdrs = [
        "file/record_file_base.h",
        "file/record_file_reader.h",
        "file/record_file_writer.h",
        "file/section.h",
        "header_builder.h",
        "record_base.h",
        "record_message.h",
        "record_reader.h",
        "record_viewer.h",
        "record_writer.h",
    ],
    deps = [
        "//cyber/common:cyber_common",
        "//cyber/message:cyber_message",
        "//cyber/proto:record_cc_proto",
        "//cyber/time:cyber_time",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "record_file_test",
    size = "small",
    srcs = ["file/record_file_test.cc"],
    deps = [
        "//cyber",
        "//cyber/proto:record_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "record_file_integration_test",
    size = "small",
    srcs = ["file/record_file_integration_test.cc"],
    tags = [
        "cpu:3",
        "exclusive",
        "manual",
    ],
    deps = [
        "//cyber",
        "//cyber/proto:record_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "record_reader_test",
    size = "small",
    srcs = ["record_reader_test.cc"],
    deps = [
        "//cyber",
        "//cyber/proto:record_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "record_viewer_test",
    size = "small",
    srcs = ["record_viewer_test.cc"],
    deps = [
        "//cyber",
        "//cyber/proto:record_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

#apollo_package()
#cpplint()
