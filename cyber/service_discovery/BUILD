#load("//tools:cpplint.bzl", "cpplint")
##load("//tools:apollo_package.bzl", "cc_library", "apollo_package", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cyber_service_discovery",
    srcs = [
        "communication/participant_listener.cc",
        "communication/subscriber_listener.cc",
        "container/graph.cc",
        "container/multi_value_warehouse.cc",
        "container/single_value_warehouse.cc",
        "specific_manager/channel_manager.cc",
        "specific_manager/manager.cc",
        "specific_manager/node_manager.cc",
        "specific_manager/service_manager.cc",
        "topology_manager.cc",
    ],
    hdrs = [
        "communication/participant_listener.h",
        "communication/subscriber_listener.h",
        "container/graph.h",
        "container/multi_value_warehouse.h",
        "container/single_value_warehouse.h",
        "container/warehouse_base.h",
        "specific_manager/channel_manager.h",
        "specific_manager/manager.h",
        "specific_manager/node_manager.h",
        "specific_manager/service_manager.h",
        "topology_manager.h",
    ],
    deps = [
        "//cyber:cyber_binary",
        "//cyber:cyber_state",
        "//cyber/base:cyber_base",
        "//cyber/common:cyber_common",
        "//cyber/message:cyber_message",
        "//cyber/proto:proto_desc_cc_proto",
        "//cyber/proto:role_attributes_cc_proto",
        "//cyber/proto:topology_change_cc_proto",
        "//cyber/time:cyber_time",
        "//cyber/transport:cyber_transport",
    ],
)

cc_library(
    name = "subscriber_listener",
    hdrs = ["communication/subscriber_listener.h"],
)

cc_library(
    name = "cyber_service_discovery_role",
    srcs = ["role/role.cc"],
    hdrs = ["role/role.h"],
    deps = [
        "//cyber:cyber_binary",
        "//cyber/common:cyber_common",
        "//cyber/proto:role_attributes_cc_proto",
    ],
)

cc_test(
    name = "topology_manager_test",
    size = "small",
    srcs = ["topology_manager_test.cc"],
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "graph_test",
    size = "small",
    srcs = ["container/graph_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "multi_value_warehouse_test",
    size = "small",
    srcs = ["container/multi_value_warehouse_test.cc"],
    deps = [
        ":cyber_service_discovery",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "single_value_warehouse_test",
    size = "small",
    srcs = ["container/single_value_warehouse_test.cc"],
    deps = [
        ":cyber_service_discovery",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "role_test",
    size = "small",
    srcs = ["role/role_test.cc"],
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "channel_manager_test",
    size = "small",
    srcs = ["specific_manager/channel_manager_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "//cyber/proto:unit_test_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "node_manager_test",
    size = "small",
    srcs = ["specific_manager/node_manager_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "service_manager_test",
    size = "small",
    srcs = ["specific_manager/service_manager_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

#apollo_package()
#cpplint()
