#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include "cyber/cyber.h"
#include "gpiodcxx/chip.hpp"
#include "gpiodcxx/line.hpp"

namespace apollo {
    namespace cyber {
        namespace examples {

            class GPIODComponent {
            public:
                static std::shared_ptr<GPIODComponent> Instance();

                bool Initialize(const std::string& chip_name);

                // GPIO Access APIs
                bool WriteGPIO(int line_offset, bool value);
                bool ReadGPIO(int line_offset, bool& value);

            private:
                GPIODComponent() = default;
                ~GPIODComponent() = default;

                bool initialized_{false};
                std::mutex mutex_;

                std::unique_ptr<::gpiod::chip> chip_;
                std::unordered_map<int, ::gpiod::line> lines_;

                bool GetOrRequestLine(int line_offset, ::gpiod::line_request::direction direction, ::gpiod::line& line);
            };

        }  // namespace examples
    }  // namespace cyber
}  // namespace apollo
