#load("//tools:cpplint.bzl", "cpplint")
##load("//tools:apollo_package.bzl", "cc_library", "cc_test", "apollo_package")

package(default_visibility = ["//visibility:public"])

cc_library(
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
    linkstatic = True,
    linkopts = ["-lm"],
    deps = [
        ":cyber_context",
        "@com_google_googletest//:gtest_main",
    ],
)

#apollo_package()

#cpplint()
