/**
 * @file FLASH_TOPK_NAMESPACE_config.h
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
 *   BEGIN_FLASH_TOPK_NAMESPACE
 *   class FlashDevice {
 *     // Implementation
 *   };
 *   END_FLASH_TOPK_NAMESPACE
 * @endcode
 *
 * 2. Accessing types within the namespace:
 * @code
 *   FLASH_TOPK_NAMESPACE_ALIAS(FlashDevice) device;
 * @endcode
 *
 * 3. Defining content within namespace scope:
 * @code
 *   FLASH_TOPK_NAMESPACE_SCOPE(
 *     struct Configuration {
 *       uint32_t size;
 *       bool enabled;
 *     };
 *   )
 * @endcode
 *
 * 4. Custom namespace name:
 * @code
 *   #define FLASH_TOPK_NAMESPACE custom_flash
 *   #include "FLASH_TOPK_NAMESPACE_config.h"
 * @endcode
 *
 * Configuration:
 * - The default namespace is 'flash' if FLASH_TOPK_NAMESPACE is not defined
 * - Define FLASH_TOPK_NAMESPACE before including this header to customize the
 * namespace name
 *
 * Best Practices:
 * - Include this header in all files that need access to the Flash namespace
 *
 */
#pragma once

#ifndef FLASH_TOPK_NAMESPACE_CONFIG_H
#define FLASH_TOPK_NAMESPACE_CONFIG_H

// Set default namespace to flash
#ifndef FLASH_TOPK_NAMESPACE
#define FLASH_TOPK_NAMESPACE flash
#endif

#define FLASH_TOPK_NAMESPACE_ALIAS(name) FLASH_TOPK_NAMESPACE::name

#define FLASH_TOPK_NAMESPACE_SCOPE(content)                                         \
  namespace FLASH_TOPK_NAMESPACE {                                                  \
  content                                                                      \
  }

#endif // FLASH_TOPK_NAMESPACE_CONFIG_H
