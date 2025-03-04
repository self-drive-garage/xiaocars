##load("//tools:apollo_package.bzl", "cc_library", "apollo_package")
#load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cyber_plugin_manager",
    srcs = [
        "plugin_description.cc",
        "plugin_manager.cc",
    ],
    hdrs = [
        "plugin_description.h",
        "plugin_manager.h",
    ],
    deps = [
        "//cyber/class_loader:cyber_class_loader",
        "//cyber/common:cyber_common",
        "@tinyxml2",
    ],
)

#apollo_package()

#cpplint()
