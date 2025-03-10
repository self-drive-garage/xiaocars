#define MODULE_NAME "minimal_example" // Before *any* includes

#include <dlfcn.h>
#include <iostream>
#include <cstdlib>
#include <glog/logging.h> // Include glog
#include "cyber/common/log.h"

int main() {
    google::InitGoogleLogging("minimal_example"); // Initialize glog!

    // Test 1: RTLD_LOCAL
    void* handle_local = dlopen("/home/samehm/workspace/apollo-orion/bazel-bin/cyber/examples/common_component_example/libcommon_component_example.so", RTLD_LAZY | RTLD_LOCAL);
    if (!handle_local) {
        std::cerr << "dlopen (RTLD_LOCAL) failed: " << dlerror() << std::endl;
        return 1;
    }
    dlclose(handle_local);

#if 0 // Disable RTLD_GLOBAL test for now
    // Test 2: RTLD_GLOBAL (should also work now)
    void* handle_global = dlopen("/home/samehm/workspace/apollo-orion/bazel-bin/cyber/examples/common_component_example/libcommon_component_example.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!handle_global) {
        std::cerr << "dlopen (RTLD_GLOBAL) failed: " << dlerror() << std::endl;
        return 1;
    }
    dlclose(handle_global);
#endif

    return 0;
}