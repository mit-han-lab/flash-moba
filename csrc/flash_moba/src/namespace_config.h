/**
 * @file flash_moba_namespace_config.h
 * @brief Configuration file for Flash namespace management and isolation
 *
 * This header provides configuration macros for managing the Flash namespace
 * across a codebase. It allows for flexible namespace naming and provides
 * utilities for namespace declaration and scoping.
 *
 * Usage Examples:
 *
 * 1. Basic namespace wrapping:
 * @code
 *   BEGIN_FLASH_MOBA_NAMESPACE
 *   class FlashDevice {
 *     // Implementation
 *   };
 *   END_FLASH_MOBA_NAMESPACE
 * @endcode
 *
 * 2. Accessing types within the namespace:
 * @code
 *   FLASH_MOBA_NAMESPACE_ALIAS(FlashDevice) device;
 * @endcode
 *
 * 3. Defining content within namespace scope:
 * @code
 *   FLASH_MOBA_NAMESPACE_SCOPE(
 *     struct Configuration {
 *       uint32_t size;
 *       bool enabled;
 *     };
 *   )
 * @endcode
 *
 * 4. Custom namespace name:
 * @code
 *   #define FLASH_MOBA_NAMESPACE custom_flash
 *   #include "FLASH_MOBA_NAMESPACE_config.h"
 * @endcode
 *
 * Configuration:
 * - The default namespace is 'flash' if FLASH_MOBA_NAMESPACE is not defined
 * - Define FLASH_MOBA_NAMESPACE before including this header to customize the
 * namespace name
 *
 * Best Practices:
 * - Include this header in all files that need access to the Flash namespace
 *
 */
#pragma once

#ifndef FLASH_MOBA_NAMESPACE_CONFIG_H
#define FLASH_MOBA_NAMESPACE_CONFIG_H

// Set default namespace to flash
#ifndef FLASH_MOBA_NAMESPACE
#define FLASH_MOBA_NAMESPACE flash
#endif

#define FLASH_MOBA_NAMESPACE_ALIAS(name) FLASH_MOBA_NAMESPACE::name

#define FLASH_MOBA_NAMESPACE_SCOPE(content)                                         \
  namespace FLASH_MOBA_NAMESPACE {                                                  \
  content                                                                      \
  }

#endif // FLASH_MOBA_NAMESPACE_CONFIG_H
