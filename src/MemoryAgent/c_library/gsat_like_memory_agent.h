#ifndef MEMORY_AGENT_H
#define MEMORY_AGENT_H

#include <stdint.h>
#include <stddef.h>

/**
 * GSAT-like Memory Agent (StressAppTest Style)
 *
 * Action Space: 64 actions (4 operations × 16 key patterns)
 *
 * Operation Types:
 * 0: FILL   - Thermal stress (Solid write)
 * 1: INVERT - Switching noise (Write -> Invert -> Write) [RECOMMENDED for CE]
 * 2: COPY   - Bandwidth saturation (Memcpy halves)
 * 3: CHECK  - Read disturb (Read only)
 *
 * Key Patterns (16 patterns from StressAppTest):
 * 0x00, 0xFF, 0x55, 0xAA, 0xF0, 0x0F, 0xCC, 0x33,
 * 0x01, 0x80, 0x16, 0xB5, 0x4A, 0x57, 0x02, 0xFD
 *
 * Action Encoding: action = operation_id * 16 + pattern_id
 * Example:
 *   action=0:  FILL with 0x00
 *   action=17: INVERT with 0x55 (highly recommended for CE detection)
 *   action=63: CHECK with 0xFD
 */
typedef enum {
    OP_FILL   = 0,
    OP_INVERT = 1,
    OP_COPY   = 2,
    OP_CHECK  = 3
} OperationType;

/* Structure to hold CE information */
typedef struct {
    int volatile_count;
    int persistent_count;
    int total_count;
    int temperature;
    int health_status;
} CEInfo;

/* Structure to hold action result */
typedef struct {
    int success;
    char error_message[256];
    CEInfo ce_info;
} ActionResult;

/* Function Prototypes */
int ma_init(const char* devdax_path, size_t memory_size_mb);
int ma_execute_action(int action, ActionResult* result);
int ma_get_ce_info(CEInfo* ce_info);
int ma_reset_baseline(void);
void ma_cleanup(void);
const char* ma_get_error(void);
int ma_is_initialized(void);

#endif // MEMORY_AGENT_H
