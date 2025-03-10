#load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_cc_binary", "apollo_package")

package(default_visibility = ["//visibility:public"])

apollo_cc_binary(
    name = "mainboard",
    srcs = [
        "mainboard.cc",
        "module_argument.cc",
        "module_argument.h",
        "module_controller.cc",
        "module_controller.h",
    ],
    linkopts = [
        "-pthread",
        #        "-lprofiler",
        # "-ltcmalloc",
    ],
    deps = [
        "//cyber",
        "//cyber/common:cyber_common",
        "//cyber/plugin_manager:cyber_plugin_manager",
        "//cyber/proto:dag_conf_cc_proto",
        # "//cyber/common/glog_init",
        #"@com_github_gperftools_gperftools//:tcmalloc_full_bench_with_profiler",
    ],
)

#apollo_package()
#cpplint()
