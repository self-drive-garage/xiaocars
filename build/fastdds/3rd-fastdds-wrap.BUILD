load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "fastdds",
    includes = [
        "include",
    ],
    hdrs = glob(["include/**/*"]),
    srcs = ["lib/libfastrtps.so", "lib/libfastcdr.so"],
    linkopts = [
        "-ltinyxml2",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)

