#load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_cc_library", "apollo_package")

package(default_visibility = ["//visibility:public"])

apollo_cc_library(
    name = "cyber_scheduler",
    srcs = [
        "common/pin_thread.cc",
        "policy/choreography_context.cc",
        "policy/classic_context.cc",
        "policy/scheduler_choreography.cc",
        "policy/scheduler_classic.cc",
        "processor.cc",
        "processor_context.cc",
        "scheduler.cc",
        "scheduler_factory.cc",
    ],
    hdrs = [
        "common/cv_wrapper.h",
        "common/mutex_wrapper.h",
        "common/pin_thread.h",
        "policy/choreography_context.h",
        "policy/classic_context.h",
        "policy/scheduler_choreography.h",
        "policy/scheduler_classic.h",
        "processor.h",
        "processor_context.h",
        "scheduler.h",
        "scheduler_factory.h",
    ],
    deps = [
        "//cyber/croutine:cyber_croutine",
        "//cyber/data:cyber_data",
        "//cyber/common:cyber_common",
        "//cyber/time:cyber_time",
        #"//cyber/proto:component_conf_cc_proto",
        "//cyber/proto:choreography_conf_cc_proto",
        "//cyber/proto:classic_conf_cc_proto",
    ],
)

cc_test(
    name = "scheduler_test",
    size = "small",
    srcs = ["scheduler_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "scheduler_classic_test",
    size = "small",
    srcs = ["scheduler_classic_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "scheduler_choreo_test",
    size = "small",
    srcs = ["scheduler_choreo_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "processor_test",
    size = "small",
    srcs = ["processor_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "pin_thread_test",
    size = "small",
    srcs = ["common/pin_thread_test.cc"],
    linkstatic = True,
    deps = [
        "//cyber",
        "@com_google_googletest//:gtest_main",
    ],
)

#apollo_package()
#cpplint()
