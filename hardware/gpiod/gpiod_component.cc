#include "gpiod_component.h"
#include "cyber/common/log.h"

namespace apollo {
namespace cyber {
namespace examples {

std::shared_ptr<GPIODComponent> GPIODComponent::Instance() {
  static std::shared_ptr<GPIODComponent> instance(new GPIODComponent());
  return instance;
}

bool GPIODComponent::Initialize(const std::string& chip_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_) {
    AINFO << "Already initialized.";
    return true;
  }

  try {
    chip_ = std::make_unique<::gpiod::chip>(chip_name);
    initialized_ = true;
    AINFO << "GPIODComponent initialized with chip: " << chip_name;
  } catch (const std::exception& ex) {
    AERROR << "Failed to initialize GPIODComponent: " << ex.what();
    initialized_ = false;
  }

  return initialized_;
}

bool GPIODComponent::GetOrRequestLine(int line_offset,
                                      ::gpiod::line_request::direction direction,
                                      ::gpiod::line& line) {
  auto it = lines_.find(line_offset);
  if (it != lines_.end()) {
    line = it->second;
    return true;
  }

  try {
    line = chip_->get_line(line_offset);
    ::gpiod::line_request config{
        "apollo_cyber", direction, 0};

    line.request(config);
    lines_[line_offset] = line;
    return true;
  } catch (const std::exception& ex) {
    AERROR << "Failed to request GPIO line: " << ex.what();
    return false;
  }
}

bool GPIODComponent::WriteGPIO(int line_offset, bool value) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_) {
    AERROR << "Component not initialized.";
    return false;
  }

  ::gpiod::line line;
  if (!GetOrRequestLine(line_offset, ::gpiod::line_request::DIRECTION_OUTPUT, line)) {
    return false;
  }

  try {
    line.set_value(value);
    return true;
  } catch (const std::exception& ex) {
    AERROR << "Failed to write GPIO: " << ex.what();
    return false;
  }
}

bool GPIODComponent::ReadGPIO(int line_offset, bool& value) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_) {
    AERROR << "Component not initialized.";
    return false;
  }

  ::gpiod::line line;
  if (!GetOrRequestLine(line_offset, ::gpiod::line_request::DIRECTION_INPUT, line)) {
    return false;
  }

  try {
    value = line.get_value();
    return true;
  } catch (const std::exception& ex) {
    AERROR << "Failed to read GPIO: " << ex.what();
    return false;
  }
}

}  // namespace examples
}  // namespace cyber
}  // namespace apollo
