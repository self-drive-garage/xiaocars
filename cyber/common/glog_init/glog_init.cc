#include "cyber/common/glog_init/glog_init.h"
#include <glog/logging.h>

namespace apollo {
namespace cyber {
namespace common {

void InitGlog() {
  google::InitGoogleLogging("cyber_mainboard"); // Initialize glog!
}
// Create a static object whose constructor calls InitGlog.
// Key Change: Use a static initializer to force early glog init.
static struct GlogInitializer {
    GlogInitializer(){
        InitGlog();
    }
} glog_initializer;

}  // namespace common
}  // namespace cyber
}  // namespace apollo