# NOTE: as of this writing Bazel support is highly experimental. It is
# also not entirely complete. It lacks most tests, for example.

package(default_visibility = ["//visibility:public"])


config_setting(
    name = "is_gcc",
    flag_values = {"@bazel_tools//tools/cpp:compiler": "gcc"},
)



CFLAGS = [
    "-Wall",
    "-Wwrite-strings",
    "-Wno-sign-compare",
    "-DTCMALLOC_DISABLE_HIDDEN_VISIBILITY",
]

CXXFLAGS = CFLAGS + [
    "-Woverloaded-virtual",
    "-std=gnu++17",
    "-fsized-deallocation",
    #"-Wthread-safety"
]

cc_library(
    name = "trivialre",
    hdrs = ["benchmark/trivialre.h"],
    copts = CXXFLAGS,
)

cc_library(
    name = "all_headers",
    hdrs = glob([
        "src/*h",
        "src/base/*h",
        "generic-config/*h",
        "src/gperftools/*h",
    ]),
    copts = CXXFLAGS,
)

cc_library(
    name = "run_benchmark",
    srcs = ["benchmark/run_benchmark.cc"],
    hdrs = ["benchmark/run_benchmark.h"],
    copts = CXXFLAGS,
    includes = [
        "generic-config",
        "src",
    ],
    deps = [
        ":all_headers",
        ":trivialre",
    ],
)

cc_binary(
    name = "basic_benchmark",
    srcs = ["benchmark/malloc_bench.cc"],
    copts = CXXFLAGS,
    deps = [":run_benchmark"],
)

cc_library(
    name = "common",
    srcs = [
        "src/base/dynamic_annotations.cc",
        "src/base/generic_writer.cc",
        "src/base/logging.cc",
        "src/base/proc_maps_iterator.cc",
        "src/base/spinlock.cc",
        "src/base/spinlock_internal.cc",
        "src/base/sysinfo.cc",
        "src/safe_strerror.cc",
    ] + select({
        "@platforms//os:windows": [
            "src/windows/ia32_modrm_map.cc",
            "src/windows/ia32_opcode_map.cc",
            "src/windows/mini_disassembler.cc",
            "src/windows/port.cc",
            "src/windows/preamble_patcher.cc",
            "src/windows/preamble_patcher_with_stub.cc",
        ],
        "//conditions:default": [],
    }),
    copts = CXXFLAGS,
    includes = [
        "generic-config",
        "src",
        "src/base",
    ],
    linkopts = select({
        "@platforms//os:windows": [
            "psapi.lib",
            "synchronization.lib",
            "shlwapi.lib",
        ],
        "//conditions:default": [],
    }),
    deps = [":all_headers"],
)

cc_library(
    name = "tcmalloc_minimal",
    srcs = [
        "src/central_freelist.cc",
        "src/common.cc",
        "src/internal_logging.cc",
        "src/malloc_extension.cc",
        "src/malloc_hook.cc",
        "src/memfs_malloc.cc",
        "src/page_heap.cc",
        "src/sampler.cc",
        "src/span.cc",
        "src/stack_trace_table.cc",
        "src/static_vars.cc",
        "src/thread_cache.cc",
        "src/thread_cache_ptr.cc",
    ] + select({
        "@platforms//os:windows": [
            "src/windows/patch_functions.cc",
            "src/windows/system-alloc.cc",
        ],
        "//conditions:default": [
            "src/system-alloc.cc",
            "src/tcmalloc.cc",
        ],
    }),
    hdrs = [
        "src/gperftools/malloc_extension.h",
        "src/gperftools/malloc_extension_c.h",
        "src/gperftools/malloc_hook.h",
        "src/gperftools/malloc_hook_c.h",
        "src/gperftools/nallocx.h",
        "src/gperftools/tcmalloc.h",
    ],
    copts = CXXFLAGS,
    includes = [
        "generic-config",
        "src",
        "src/base",
    ],
    # note, bazel thingy is passing NDEBUG automagically in -c opt builds. So we're okay with that.
    local_defines = ["NO_TCMALLOC_SAMPLES"],
    visibility = ["//visibility:public"],
    deps = [
        ":all_headers",
        ":common",
    ],
    alwayslink = 1,
)

cc_library(
    name = "libbacktrace",
    srcs = [
        "vendor/libbacktrace-integration/file-format.c",
        "vendor/libbacktrace/dwarf.c",
        "vendor/libbacktrace/fileline.c",
        "vendor/libbacktrace/posix.c",
        "vendor/libbacktrace/read.c",
        "vendor/libbacktrace/sort.c",
        "vendor/libbacktrace/state.c",
    ] + glob([
        "vendor/libbacktrace-integration/*.h",
        "vendor/libbacktrace/*.h",
    ]),
    hdrs = [
        "vendor/libbacktrace/elf.c",
        "vendor/libbacktrace/macho.c",
    ],  # yes, elf.c is included by file-format.c below and bazel makes us do this
    copts = CFLAGS,
    includes = [
        "vendor/libbacktrace",
        "vendor/libbacktrace-integration",
    ],
    #target_compatible_with = NON_WINDOWS,
)

cc_library(
    name = "symbolize",
    srcs = [
        "src/symbolize.cc",
        "vendor/libbacktrace-integration/backtrace-alloc.cc",
    ],
    copts = CXXFLAGS,
    includes = [
        "generic-config",
        "src",
        "src/base",
    ],
    #target_compatible_with = NON_WINDOWS,
    deps = [
        ":all_headers",
        ":libbacktrace",
    ],
)

cc_library(
    name = "low_level_alloc",
    srcs = ["src/base/low_level_alloc.cc"],
    copts = CXXFLAGS,
    includes = [
        "generic-config",
        "src",
        "src/base",
    ],
    deps = [":all_headers"],
)

cc_library(
    name = "tcmalloc_minimal_debug",
    srcs = [
        "src/central_freelist.cc",
        "src/common.cc",
        "src/debugallocation.cc",
        "src/internal_logging.cc",
        "src/malloc_extension.cc",
        "src/malloc_hook.cc",
        "src/memfs_malloc.cc",
        "src/page_heap.cc",
        "src/sampler.cc",
        "src/span.cc",
        "src/stack_trace_table.cc",
        "src/static_vars.cc",
        "src/system-alloc.cc",
        "src/thread_cache.cc",
        "src/thread_cache_ptr.cc",
    ],
    hdrs = [
        "src/gperftools/malloc_extension.h",
        "src/gperftools/malloc_extension_c.h",
        "src/gperftools/malloc_hook.h",
        "src/gperftools/malloc_hook_c.h",
        "src/gperftools/nallocx.h",
        "src/gperftools/tcmalloc.h",
        "src/tcmalloc.cc",
    ],
    copts = CXXFLAGS,
    includes = [
        "generic-config",
        "src",
        "src/base",
    ],
    # note, bazel thingy is passing NDEBUG automagically in -c opt builds. So we're okay with that.
    local_defines = ["NO_TCMALLOC_SAMPLES"],
    #target_compatible_with = NON_WINDOWS,
    visibility = ["//visibility:public"],
    deps = [
        ":all_headers",
        ":common",
        ":low_level_alloc",
        ":symbolize",
    ],
    alwayslink = 1,
)

cc_library(
    name = "stacktrace",
    srcs = [
        "src/base/elf_mem_image.cc",
        "src/base/vdso_support.cc",
        "src/stacktrace.cc",
    ],
    hdrs = ["src/gperftools/stacktrace.h"],
    copts = CXXFLAGS,
    includes = [
        "generic-config",
        "src",
        "src/base",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":all_headers",
        ":common",
    ],
)

cc_binary(
    name = "tcmalloc_bench",
    srcs = ["benchmark/malloc_bench.cc"],
    copts = CXXFLAGS,
    deps = [
        ":run_benchmark",
        ":tcmalloc_minimal",
    ],
)

cc_binary(
    name = "tcmalloc_debug_bench",
    srcs = ["benchmark/malloc_bench.cc"],
    copts = CXXFLAGS,
    deps = [
        ":run_benchmark",
        ":tcmalloc_minimal_debug",
    ],
)

cc_library(
    name = "tcmalloc",
    srcs = [
        "src/central_freelist.cc",
        "src/common.cc",
        "src/emergency_malloc.cc",
        "src/heap-checker-stub.cc",
        "src/heap-profile-table.cc",
        "src/heap-profiler.cc",
        "src/internal_logging.cc",
        "src/malloc_backtrace.cc",
        "src/malloc_extension.cc",
        "src/malloc_hook.cc",
        "src/memfs_malloc.cc",
        "src/page_heap.cc",
        "src/sampler.cc",
        "src/span.cc",
        "src/stack_trace_table.cc",
        "src/static_vars.cc",
        "src/system-alloc.cc",
        "src/tcmalloc.cc",
        "src/thread_cache.cc",
        "src/thread_cache_ptr.cc",
    ],
    hdrs = [
        "src/gperftools/heap-profiler.h",
        "src/gperftools/malloc_extension.h",
        "src/gperftools/malloc_extension_c.h",
        "src/gperftools/malloc_hook.h",
        "src/gperftools/malloc_hook_c.h",
        "src/gperftools/nallocx.h",
        "src/gperftools/tcmalloc.h",
    ],
    copts = CXXFLAGS,
    includes = [
        "generic-config",
        "src",
        "src/base",
    ],
    # note, bazel thingy is passing NDEBUG automagically in -c opt builds. So we're okay with that.
    local_defines = ["ENABLE_EMERGENCY_MALLOC"],
    #target_compatible_with = NON_WINDOWS,
    visibility = ["//visibility:public"],
    deps = [
        ":all_headers",
        ":common",
        ":low_level_alloc",
        ":stacktrace",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "tcmalloc_full_bench",
    srcs = ["benchmark/malloc_bench.cc"],
    copts = CXXFLAGS,
    deps = [
        ":run_benchmark",
        ":tcmalloc",
    ],
)

cc_library(
    name = "tcmalloc_debug",
    srcs = [
        "src/central_freelist.cc",
        "src/common.cc",
        "src/debugallocation.cc",
        "src/emergency_malloc.cc",
        "src/heap-checker-stub.cc",
        "src/heap-profile-table.cc",
        "src/heap-profiler.cc",
        "src/internal_logging.cc",
        "src/malloc_backtrace.cc",
        "src/malloc_extension.cc",
        "src/malloc_hook.cc",
        "src/memfs_malloc.cc",
        "src/page_heap.cc",
        "src/sampler.cc",
        "src/span.cc",
        "src/stack_trace_table.cc",
        "src/static_vars.cc",
        "src/system-alloc.cc",
        "src/thread_cache.cc",
        "src/thread_cache_ptr.cc",
    ],
    hdrs = [
        "src/gperftools/heap-profiler.h",
        "src/gperftools/malloc_extension.h",
        "src/gperftools/malloc_extension_c.h",
        "src/gperftools/malloc_hook.h",
        "src/gperftools/malloc_hook_c.h",
        "src/gperftools/nallocx.h",
        "src/gperftools/tcmalloc.h",
        "src/tcmalloc.cc",  # tcmalloc.cc gets included by debugallocation.cc
    ],
    copts = CXXFLAGS,
    includes = [
        "generic-config",
        "src",
        "src/base",
    ],
    # note, bazel thingy is passing NDEBUG automagically in -c opt builds. So we're okay with that.
    local_defines = ["ENABLE_EMERGENCY_MALLOC"],
    #target_compatible_with = NON_WINDOWS,
    visibility = ["//visibility:public"],
    deps = [
        ":all_headers",
        ":common",
        ":low_level_alloc",
        ":stacktrace",
        ":symbolize",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "tcmalloc_full_debug_bench",
    srcs = ["benchmark/malloc_bench.cc"],
    copts = CXXFLAGS,
    deps = [
        ":run_benchmark",
        ":tcmalloc_debug",
    ],
)

cc_test(
    name = "tcmalloc_minimal_test",
    srcs = [
        "src/tests/tcmalloc_unittest.cc",
        "src/tests/testutil.h",
    ],
    copts = CXXFLAGS,
    deps = [
        ":all_headers",
        ":tcmalloc_minimal",
        "@com_google_googletest//:gtest_main", "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "tcmalloc_minimal_debug_test",
    srcs = [
        "src/tests/tcmalloc_unittest.cc",
        "src/tests/testutil.h",
    ],
    copts = CXXFLAGS,
    deps = [
        ":all_headers",
        ":tcmalloc_minimal_debug",
        "@com_google_googletest//:gtest_main", "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "tcmalloc_test",
    srcs = [
        "src/tests/tcmalloc_unittest.cc",
        "src/tests/testutil.h",
    ],
    copts = CXXFLAGS,
    deps = [
        ":all_headers",
        ":tcmalloc",
        "@com_google_googletest//:gtest_main", "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "tcmalloc_debug_test",
    srcs = [
        "src/tests/tcmalloc_unittest.cc",
        "src/tests/testutil.h",
    ],
    copts = CXXFLAGS,
    deps = [
        ":all_headers",
        ":tcmalloc_debug",
        "@com_google_googletest//:gtest_main", "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "debugallocation_test",
    srcs = [
        "src/tests/debugallocation_test.cc",
        "src/tests/testutil.h",
    ],
    copts = CXXFLAGS,
    deps = [
        ":all_headers",
        ":tcmalloc_debug",
        "@com_google_googletest//:gtest_main", "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "cpu_profiler",
    srcs = [
        "src/profile-handler.cc",
        "src/profiledata.cc",
        "src/profiler.cc",
    ],
    hdrs = ["src/gperftools/profiler.h"],
    copts = CXXFLAGS,
    #target_compatible_with = NON_WINDOWS,
    visibility = ["//visibility:public"],
    deps = [
        ":all_headers",
        ":common",
        ":stacktrace",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "tcmalloc_full_bench_with_profiler",
    srcs = ["benchmark/malloc_bench.cc"],
    copts = CXXFLAGS,
    deps = [
        ":cpu_profiler",
        ":run_benchmark",
        ":tcmalloc",
    ],
)
