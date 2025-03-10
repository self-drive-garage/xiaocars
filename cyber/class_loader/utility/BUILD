#load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_cc_library", "apollo_package")

package(default_visibility = ["//visibility:public"])

apollo_cc_library(
    name = "class_loader_utility",
    srcs = [
        "class_factory.cc",
        "class_loader_utility.cc",
    ],
    hdrs = [
        "class_factory.h",
        "class_loader_utility.h",
    ],
    deps = [
        "//cyber/class_loader/shared_library",
        "//cyber/common:cyber_common",
    ],
)

#apollo_package()
#cpplint()
