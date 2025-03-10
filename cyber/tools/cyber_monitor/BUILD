#load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_cc_binary", "apollo_package")

package(default_visibility = ["//visibility:public"])

apollo_cc_binary(
    name = "cyber_monitor",
    srcs = [
        "cyber_topology_message.cc",
        "cyber_topology_message.h",
        "general_channel_message.cc",
        "general_channel_message.h",
        "general_message.cc",
        "general_message.h",
        "general_message_base.cc",
        "general_message_base.h",
        "main.cc",
        "renderable_message.cc",
        "renderable_message.h",
        "screen.cc",
        "screen.h",
    ],
    linkopts = ["-pthread"],
    deps = [
        "//cyber",
        "@ncurses",
    ],
)

#apollo_package()
#cpplint()
