#load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_cc_binary", "apollo_cc_library", "apollo_package")

package(default_visibility = ["//visibility:public"])

apollo_cc_library(
    name = "shared_library",
    srcs = ["shared_library.cc"],
    hdrs = [
        "exceptions.h",
        "shared_library.h",
    ],
    linkopts = ["-ldl"],
    deps = [
        "//cyber/common:cyber_common",
    ],
)

apollo_cc_binary(
    name = "libcyber_sample.so",
    testonly = True,
    srcs = [
        "sample.cc",
        "sample.h",
    ],
    linkopts = ["-lm"],
    linkshared = True,
    linkstatic = False,
    visibility = ["//visibility:private"],
)

cc_test(
    name = "shared_library_test",
    size = "small",
    srcs = ["shared_library_test.cc"],
    data = [
        ":libcyber_sample.so",
    ],
    deps = [
        ":shared_library",
        "@com_google_googletest//:gtest_main",
    ],
)

#apollo_package()
#cpplint()
