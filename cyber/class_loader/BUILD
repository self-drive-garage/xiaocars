#load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_cc_library", "apollo_package")

package(default_visibility = ["//visibility:public"])

apollo_cc_library(
    name = "cyber_class_loader",
    srcs = [
        "class_loader.cc",
        "class_loader_manager.cc",
    ],
    hdrs = [
        "class_loader.h",
        "class_loader_manager.h",
        "class_loader_register_macro.h",
    ],
    deps = [
        "//cyber/class_loader/shared_library",
        "//cyber/class_loader/utility:class_loader_utility",
        "//cyber/common:cyber_common",
    ],
)

# cc_test(
#     name = "class_loader_test",
#     size = "small",
#     srcs = ["class_loader_test.cc"],
#     data = [
#         "//cyber/class_loader/test:plugin1",
#         "//cyber/class_loader/test:plugin2",
#     ],
#     deps = [
#         "//cyber",
#         "//cyber/class_loader/test:base",
#         "//cyber/proto:unit_test_cc_proto",
#         "@com_google_googletest//:gtest",
#     ],
# )

#apollo_package()
#cpplint()
