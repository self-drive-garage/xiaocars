load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")


package(default_visibility = ["//visibility:public"])

CAMERA_COPTS = ['-DMODULE_NAME=\\"camera\\"']

cc_binary(
    name = "libcamera_component.so",
    linkopts = [
        "-latomic",
    ],
    linkshared = True,
    linkstatic = False,
    deps = [
        ":camera_component_lib",
        ":compress_component_lib",
        "@ffmpeg//:avcodec",
        "@ffmpeg//:avformat",
        "@ffmpeg//:swscale",
    ],
)

filegroup(
    name = "runtime_data",
    srcs = glob([
        "conf/*.txt",
        "dag/*.dag",
        "launch/*.launch",
    ]),
)

cc_library(
    name = "camera_component_lib",
    srcs = ["camera_component.cc"],
    hdrs = ["camera_component.h"],
    copts = CAMERA_COPTS,
    deps = [
        ":camera",
        "//cyber",
        "//common_msgs/common/proto:error_code_cc_proto",
        "//common_msgs/common/proto:header_cc_proto",
        "//common_msgs/drivers/proto:sensor_image_cc_proto",
    ],
)

cc_library(
    name = "compress_component_lib",
    srcs = ["compress_component.cc"],
    hdrs = ["compress_component.h"],
    copts = CAMERA_COPTS,
    deps = [
        "//cyber",
        "//common_msgs/common/proto:error_code_cc_proto",
        "//common_msgs/common/proto:header_cc_proto",
        "//common_msgs/drivers/camera/proto:config_cc_proto",
        "//common_msgs/drivers/proto:sensor_image_cc_proto",
        "@opencv//:core",
        "@opencv//:highgui",
        "@opencv//:imgcodecs",
        "@opencv//:imgproc",
    ],
)

cc_library(
    name = "camera",
    srcs = select(
        {
            "@platforms//cpu:x86_64": [
                "usb_cam.cc",
                "util.cc",
            ],
            "@platforms//cpu:aarch64": [
                "usb_cam.cc",
            ],
        },
        no_match_error = "Please Build with an ARM or Linux x86_64 platform",
    ),
    hdrs = select(
        {
            "@platforms//cpu:x86_64": [
                "usb_cam.h",
                "util.h",
            ],
            "@platforms//cpu:aarch64": [
                "usb_cam.h",
            ],
        },
        no_match_error = "Please Build with an ARM or Linux x86_64 platform",
    ),
    deps = [
        "//cyber",
        "//common_msgs/drivers/camera/proto:config_cc_proto",
        "@ffmpeg//:avcodec",
    ] + select({
        # "@platforms//cpu:x86_64": ["@adv_plat"],
        "//conditions:default": [],
    }),
)


