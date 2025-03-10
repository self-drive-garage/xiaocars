#load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_cc_library", "apollo_package")

package(default_visibility = ["//visibility:public"])

apollo_cc_library(
    name = "cyber_context",
    srcs = [
    ],
    hdrs = [
        "context.h",
    ],
    deps = [
    ],
)

cc_test(
    name = "cyber_context_test",
    size = "small",
    srcs = ["context_test.cc"],
    linkopts = ["-lm"],
    linkstatic = True,
    deps = [
        ":cyber_context",
        "@com_google_googletest//:gtest_main",
    ],
)

#apollo_package()

#cpplint()
