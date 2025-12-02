/**
 * @file gsat_memory_agent.h
 * @brief Memory Agent C Library for CXL Memory Testing
 *
 * This library provides memory testing functionality for Phase 1 RL environment.
 * Executes memory operations via devdax and detects Correctable Errors via umxc.
 * Like stressapp test like implementation
 */

#ifndef GSAT_MEMORY_AGENT_H
#define GSAT_MEMORY_AGENT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========== Data Structures ========== */

/**
 * @brief Correctable Error Information
 */
typedef struct {
    int volatile_count;      /**< Volatile error count */
    int persistent_count;    /**< Persistent error count */
    int total_count;         /**< Total error count */
    int temperature;         /**< Device temperature in Celsius */
    int health_status;       /**< Health status code */
} CEInfo;

/**
 * @brief Memory test operation types
 *
 * Matches Python Phase1Environment operation types
 */
typedef enum {
    OP_FILL = 0,        /* simple write  */
    OP_INVERT = 1,      /* write after invert-write */ 
    OP_COPY = 2,        /* Memory Copy */
    OP_CHECK = 3        /* Read Verify */
} OperationType;

/**
 * @brief Action execution result
 */
typedef struct {
    int success;             /**< 1 if success, 0 if error */
    CEInfo ce_info;          /**< CE information after test */
    char error_message[256]; /**< Error message if failed */
} ActionResult;

/* ========== Main API Functions ========== */

/**
 * @brief Initialize Memory Agent
 *
 * @param devdax_path Path to devdax device (e.g., "/dev/dax0.0")
 * @param memory_size_mb Memory size to test in MB
 * @param sampling_rate Fraction of memory to test (0.0-1.0)
 * @return 0 on success, -1 on error
 */
int ma_init(const char* devdax_path, size_t memory_size_mb);

/**
 * @brief Execute memory test action
 *
 * @param action Action ID (0-1535)
 *               Encoding: operation_type * 256 + pattern_byte
 * @param result Pointer to store execution result
 * @return 0 on success, -1 on error
 */
int ma_execute_action(int action, ActionResult* result);

/**
 * @brief Get current CE information (without executing test)
 *
 * @param ce_info Pointer to store CE information
 * @return 0 on success, -1 on error
 */
int ma_get_ce_info(CEInfo* ce_info);

/**
 * @brief Reset CE baseline for new episode
 *
 * Call this at the start of each episode to track only new errors.
 *
 * @return 0 on success, -1 on error
 */
int ma_reset_baseline(void);

/**
 * @brief Cleanup and release resources
 */
void ma_cleanup(void);

/**
 * @brief Get last error message
 *
 * @return Pointer to error message string (valid until next call)
 */
const char* ma_get_error(void);

/* ========== Helper Functions (for testing/debugging) ========== */

/**
 * @brief Decode action into operation type and pattern
 *
 * @param action Action ID (0-1535)
 * @param operation Output: operation type
 * @param pattern Output: pattern byte
 * @return 0 on success, -1 on invalid action
 */
int ma_decode_action(int action, OperationType* operation, uint8_t* pattern);

/**
 * @brief Check if Memory Agent is initialized
 *
 * @return 1 if initialized, 0 otherwise
 */
int ma_is_initialized(void);

#ifdef __cplusplus
}
#endif

#endif /* MEMORY_AGENT_H */
