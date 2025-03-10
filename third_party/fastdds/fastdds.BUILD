load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "fastdds",
    includes = ["include"],
    hdrs = glob(["include/**/*"]),
    linkopts = [
        "-L/usr/local/fast-rtps-1.5.0-1/lib",
        "-lfastrtps",
        "-lfastcdr",
    ],
    visibility = ["//visibility:public"],
)