/**
 * @file memory_agent.c
 * @brief Memory Agent Implementation
 */

#include "memory_agent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>

/* ========== Global State ========== */

typedef struct {
    int initialized;
    int devdax_fd;
    char devdax_path[256];
    size_t memory_size;

    /* CE baseline for delta calculation */
    int baseline_volatile;
    int baseline_persistent;

    /* Error message buffer */
    char error_msg[512];
} MemoryAgentState;

static MemoryAgentState g_state = {0};

/* ========== Internal Helper Functions ========== */

/**
 * @brief Set error message
 */
static void set_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_state.error_msg, sizeof(g_state.error_msg), fmt, args);
    va_end(args);
}

/**
 * @brief Execute umxc command and parse CE info
 */
static int execute_umxc(CEInfo* ce_info) {
    FILE* fp;
    char buffer[256];
    int volatile_count = 0;
    int persistent_count = 0;
    int temperature = 0;
    int health_status = 0;

    /* Execute: umxc mbox -H */
    fp = popen("umxc mbox -H", "r");
    if (fp == NULL) {
        set_error("Failed to execute umxc: %s", strerror(errno));
        return -1;
    }

    /* Parse output */
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        /* Parse: [0Ah] Corrected Volatile Error Count: 0 */
        if (strstr(buffer, "[0Ah]") && strstr(buffer, "Corrected Volatile Error Count")) {
            sscanf(buffer, "%*[^:]: %d", &volatile_count);
        }

        /* Parse: [0Eh] Corrected Persistent Error Count: 0 */
        if (strstr(buffer, "[0Eh]") && strstr(buffer, "Corrected Persistent Error Count")) {
            sscanf(buffer, "%*[^:]: %d", &persistent_count);
        }

        /* Parse: [04h] Device Temperature: 42C */
        if (strstr(buffer, "[04h]") && strstr(buffer, "Device Temperature")) {
            sscanf(buffer, "%*[^:]: %dC", &temperature);
        }

        /* Parse: [00h] Health Status: 0x0 */
        if (strstr(buffer, "[00h]") && strstr(buffer, "Health Status")) {
            sscanf(buffer, "%*[^:]: 0x%x", &health_status);
        }
    }

    pclose(fp);

    /* Calculate delta from baseline */
    ce_info->volatile_count = (volatile_count > g_state.baseline_volatile) ?
                               (volatile_count - g_state.baseline_volatile) : 0;
    ce_info->persistent_count = (persistent_count > g_state.baseline_persistent) ?
                                 (persistent_count - g_state.baseline_persistent) : 0;
    ce_info->total_count = ce_info->volatile_count + ce_info->persistent_count;
    ce_info->temperature = temperature;
    ce_info->health_status = health_status;

    return 0;
}

/**
 * @brief Write pattern to devdax in ascending order
 */
static int write_ascending(uint64_t start, uint64_t end, uint8_t pattern) {
    /* devdax requires 2MB alignment - use mmap instead of write() */
    #define ALIGN_SIZE (2 * 1024 * 1024)  /* 2MB alignment */

    /* Align start and end to 2MB boundaries */
    uint64_t aligned_start = (start / ALIGN_SIZE) * ALIGN_SIZE;
    uint64_t aligned_end = ((end + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
    size_t map_size = aligned_end - aligned_start;

    if (map_size == 0) {
        map_size = ALIGN_SIZE;
    }

    /* mmap the region */
    void* mapped = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, g_state.devdax_fd, aligned_start);
    if (mapped == MAP_FAILED) {
        set_error("mmap failed at 0x%lx (size=%zu): %s",
                 aligned_start, map_size, strerror(errno));
        return -1;
    }

    /* Write pattern to the mapped region in ascending order */
    uint8_t* ptr = (uint8_t*)mapped;
    size_t offset_start = start - aligned_start;
    size_t offset_end = (end - aligned_start < map_size) ?
                        (end - aligned_start) : map_size;

    for (size_t i = offset_start; i < offset_end; i += 64) {
        memset(ptr + i, pattern, 64);
    }

    /* Ensure writes are flushed */
    msync(mapped, map_size, MS_SYNC);

    munmap(mapped, map_size);
    return 0;
}

/**
 * @brief Write pattern to devdax in descending order
 */
static int write_descending(uint64_t start, uint64_t end, uint8_t pattern) {
    #define ALIGN_SIZE (2 * 1024 * 1024)  /* 2MB alignment */

    /* Align start and end to 2MB boundaries */
    uint64_t aligned_start = (start / ALIGN_SIZE) * ALIGN_SIZE;
    uint64_t aligned_end = ((end + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
    size_t map_size = aligned_end - aligned_start;

    if (map_size == 0) {
        map_size = ALIGN_SIZE;
    }

    /* mmap the region */
    void* mapped = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, g_state.devdax_fd, aligned_start);
    if (mapped == MAP_FAILED) {
        set_error("mmap failed at 0x%lx (size=%zu): %s",
                 aligned_start, map_size, strerror(errno));
        return -1;
    }

    /* Write pattern to the mapped region in descending order */
    uint8_t* ptr = (uint8_t*)mapped;
    size_t offset_start = start - aligned_start;
    size_t offset_end = (end - aligned_start < map_size) ?
                        (end - aligned_start) : map_size;

    /* Descending: start from end */
    for (size_t i = offset_end - 64; i >= offset_start; i -= 64) {
        memset(ptr + i, pattern, 64);
        if (i == offset_start) break;  /* Prevent underflow */
    }

    msync(mapped, map_size, MS_SYNC);
    munmap(mapped, map_size);
    return 0;
}

/**
 * @brief Read and verify from devdax in ascending order
 */
static int read_ascending(uint64_t start, uint64_t end, uint8_t expected_pattern) {
    (void)expected_pattern;  /* Unused - CE detector handles verification */
    #define ALIGN_SIZE (2 * 1024 * 1024)  /* 2MB alignment */

    uint64_t aligned_start = (start / ALIGN_SIZE) * ALIGN_SIZE;
    uint64_t aligned_end = ((end + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
    size_t map_size = aligned_end - aligned_start;

    if (map_size == 0) {
        map_size = ALIGN_SIZE;
    }

    /* mmap the region */
    void* mapped = mmap(NULL, map_size, PROT_READ,
                        MAP_SHARED, g_state.devdax_fd, aligned_start);
    if (mapped == MAP_FAILED) {
        set_error("mmap failed at 0x%lx (size=%zu): %s",
                 aligned_start, map_size, strerror(errno));
        return -1;
    }

    /* Read pattern from the mapped region in ascending order */
    volatile uint8_t* ptr = (volatile uint8_t*)mapped;
    size_t offset_start = start - aligned_start;
    size_t offset_end = (end - aligned_start < map_size) ?
                        (end - aligned_start) : map_size;

    uint8_t dummy;
    for (size_t i = offset_start; i < offset_end; i += 64) {
        /* Force read from memory - CE detector will catch errors */
        dummy = ptr[i];
        (void)dummy;
    }

    munmap(mapped, map_size);
    return 0;
}

/**
 * @brief Read and verify from devdax in descending order
 */
static int read_descending(uint64_t start, uint64_t end, uint8_t expected_pattern) {
    (void)expected_pattern;  /* Unused - CE detector handles verification */
    #define ALIGN_SIZE (2 * 1024 * 1024)  /* 2MB alignment */

    /* Align start and end to 2MB boundaries */
    uint64_t aligned_start = (start / ALIGN_SIZE) * ALIGN_SIZE;
    uint64_t aligned_end = ((end + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
    size_t map_size = aligned_end - aligned_start;

    if (map_size == 0) {
        map_size = ALIGN_SIZE;
    }

    /* mmap the region */
    void* mapped = mmap(NULL, map_size, PROT_READ,
                        MAP_SHARED, g_state.devdax_fd, aligned_start);
    if (mapped == MAP_FAILED) {
        set_error("mmap failed at 0x%lx (size=%zu): %s",
                 aligned_start, map_size, strerror(errno));
        return -1;
    }

    /* Read pattern from the mapped region in descending order */
    volatile uint8_t* ptr = (volatile uint8_t*)mapped;
    size_t offset_start = start - aligned_start;
    size_t offset_end = (end - aligned_start < map_size) ?
                        (end - aligned_start) : map_size;

    uint8_t dummy;
    /* Descending order - need to handle underflow carefully */
    if (offset_end >= 64) {
        for (size_t i = offset_end - 64; ; i -= 64) {
            /* Force read from memory - CE detector will catch errors */
            dummy = ptr[i];
            (void)dummy;

            if (i <= offset_start || i < 64) break;  /* Prevent underflow */
        }
    }

    munmap(mapped, map_size);
    return 0;
}

/**
 * @brief Write and read immediately (ascending)
 */
static int write_read_ascending(uint64_t start, uint64_t end, uint8_t pattern) {
    #define ALIGN_SIZE (2 * 1024 * 1024)  /* 2MB alignment */

    /* Align start and end to 2MB boundaries */
    uint64_t aligned_start = (start / ALIGN_SIZE) * ALIGN_SIZE;
    uint64_t aligned_end = ((end + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
    size_t map_size = aligned_end - aligned_start;

    if (map_size == 0) {
        map_size = ALIGN_SIZE;
    }

    /* mmap the region */
    void* mapped = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, g_state.devdax_fd, aligned_start);
    if (mapped == MAP_FAILED) {
        set_error("mmap failed at 0x%lx (size=%zu): %s",
                 aligned_start, map_size, strerror(errno));
        return -1;
    }

    volatile uint8_t* ptr = (volatile uint8_t*)mapped;
    size_t offset_start = start - aligned_start;
    size_t offset_end = (end - aligned_start < map_size) ?
                        (end - aligned_start) : map_size;

    uint8_t dummy;
    /* Write and immediately read back in ascending order */
    for (size_t i = offset_start; i < offset_end; i += 64) {
        /* Write pattern */
        memset((void*)(ptr + i), pattern, 64);

        /* Read back immediately - CE detector will catch errors */
        dummy = ptr[i];
        (void)dummy;
    }

    /* Ensure writes are flushed */
    msync(mapped, map_size, MS_SYNC);

    munmap(mapped, map_size);
    return 0;
}

/**
 * @brief Write and read immediately (descending)
 */
static int write_read_descending(uint64_t start, uint64_t end, uint8_t pattern) {
    #define ALIGN_SIZE (2 * 1024 * 1024)  /* 2MB alignment */

    /* Align start and end to 2MB boundaries */
    uint64_t aligned_start = (start / ALIGN_SIZE) * ALIGN_SIZE;
    uint64_t aligned_end = ((end + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
    size_t map_size = aligned_end - aligned_start;

    if (map_size == 0) {
        map_size = ALIGN_SIZE;
    }

    /* mmap the region */
    void* mapped = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, g_state.devdax_fd, aligned_start);
    if (mapped == MAP_FAILED) {
        set_error("mmap failed at 0x%lx (size=%zu): %s",
                 aligned_start, map_size, strerror(errno));
        return -1;
    }

    volatile uint8_t* ptr = (volatile uint8_t*)mapped;
    size_t offset_start = start - aligned_start;
    size_t offset_end = (end - aligned_start < map_size) ?
                        (end - aligned_start) : map_size;

    uint8_t dummy;
    /* Write and immediately read back in descending order */
    if (offset_end >= 64) {
        for (size_t i = offset_end - 64; ; i -= 64) {
            /* Write pattern */
            memset((void*)(ptr + i), pattern, 64);

            /* Read back immediately - CE detector will catch errors */
            dummy = ptr[i];
            (void)dummy;

            if (i <= offset_start || i < 64) break;  /* Prevent underflow */
        }
    }

    /* Ensure writes are flushed */
    msync(mapped, map_size, MS_SYNC);

    munmap(mapped, map_size);
    return 0;
}

/**
 * @brief Execute memory operation
 */
static int execute_operation(OperationType operation, uint8_t pattern) {
    uint64_t start_dpa = 0;
    uint64_t end_dpa = g_state.memory_size;

    switch (operation) {
        case WR_ASC_ASC:
            /* [^(W pat), ^(R pat)] - Ascending write, ascending read */
            if (write_ascending(start_dpa, end_dpa, pattern) < 0) return -1;
            if (read_ascending(start_dpa, end_dpa, pattern) < 0) return -1;
            break;

        case WR_DESC_DESC:
            /* [v(W pat), v(R pat)] - Descending write, descending read */
            if (write_descending(start_dpa, end_dpa, pattern) < 0) return -1;
            if (read_descending(start_dpa, end_dpa, pattern) < 0) return -1;
            break;

        case WR_ASC_DESC:
            /* [^(W pat), v(R pat)] - Ascending write, descending read */
            if (write_ascending(start_dpa, end_dpa, pattern) < 0) return -1;
            if (read_descending(start_dpa, end_dpa, pattern) < 0) return -1;
            break;

        case WR_DESC_ASC:
            /* [v(W pat), ^(R pat)] - Descending write, ascending read */
            if (write_descending(start_dpa, end_dpa, pattern) < 0) return -1;
            if (read_ascending(start_dpa, end_dpa, pattern) < 0) return -1;
            break;

        case WR_DESC_SINGLE:
            /* [v(W pat, R pat)] - Descending single-pass */
            if (write_read_descending(start_dpa, end_dpa, pattern) < 0) return -1;
            break;

        case WR_ASC_SINGLE:
            /* [^(W pat, R pat)] - Ascending single-pass */
            if (write_read_ascending(start_dpa, end_dpa, pattern) < 0) return -1;
            break;

        default:
            set_error("Invalid operation type: %d", operation);
            return -1;
    }

    return 0;
}

/* ========== Public API Implementation ========== */

int ma_init(const char* devdax_path, size_t memory_size_mb) {
    if (g_state.initialized) {
        set_error("Memory Agent already initialized");
        return -1;
    }

    /* Open devdax device */
    /* Use O_RDWR only - O_SYNC/O_DIRECT may cause alignment issues */
    g_state.devdax_fd = open(devdax_path, O_RDWR);
    if (g_state.devdax_fd < 0) {
        set_error("Failed to open %s: %s", devdax_path, strerror(errno));
        return -1;
    }

    /* Save configuration */
    strncpy(g_state.devdax_path, devdax_path, sizeof(g_state.devdax_path) - 1);
    g_state.memory_size = memory_size_mb * 1024UL * 1024UL;
    g_state.baseline_volatile = 0;
    g_state.baseline_persistent = 0;
    g_state.initialized = 1;

    printf("Memory Agent initialized:\n");
    printf("  Device: %s\n", devdax_path);
    printf("  Memory size: %zu MB (%zu bytes)\n", memory_size_mb, g_state.memory_size);

    return 0;
}

int ma_execute_action(int action, ActionResult* result) {
    if (!g_state.initialized) {
        set_error("Memory Agent not initialized");
        return -1;
    }

    if (action < 0 || action >= 1536) {
        set_error("Invalid action: %d (must be 0-1535)", action);
        return -1;
    }

    /* Decode action */
    OperationType operation;
    uint8_t pattern;
    if (ma_decode_action(action, &operation, &pattern) < 0) {
        return -1;
    }

    /* Reset CE baseline before test */
    if (ma_reset_baseline() < 0) {
        /* Non-fatal - continue anyway */
    }

    /* Execute memory operation */
    if (execute_operation(operation, pattern) < 0) {
        result->success = 0;
        strncpy(result->error_message, g_state.error_msg, sizeof(result->error_message) - 1);
        return -1;
    }

    /* Get CE information after test */
    if (execute_umxc(&result->ce_info) < 0) {
        result->success = 0;
        strncpy(result->error_message, g_state.error_msg, sizeof(result->error_message) - 1);
        return -1;
    }

    result->success = 1;
    result->error_message[0] = '\0';

    return 0;
}

int ma_get_ce_info(CEInfo* ce_info) {
    if (!g_state.initialized) {
        set_error("Memory Agent not initialized");
        return -1;
    }

    return execute_umxc(ce_info);
}

int ma_reset_baseline(void) {
    if (!g_state.initialized) {
        set_error("Memory Agent not initialized");
        return -1;
    }

    CEInfo current;
    if (execute_umxc(&current) < 0) {
        return -1;
    }

    g_state.baseline_volatile += current.volatile_count;
    g_state.baseline_persistent += current.persistent_count;

    return 0;
}

void ma_cleanup(void) {
    if (g_state.devdax_fd >= 0) {
        close(g_state.devdax_fd);
        g_state.devdax_fd = -1;
    }

    g_state.initialized = 0;

    printf("Memory Agent cleaned up\n");
}

const char* ma_get_error(void) {
    return g_state.error_msg;
}

int ma_decode_action(int action, OperationType* operation, uint8_t* pattern) {
    if (action < 0 || action >= 1536) {
        set_error("Invalid action: %d", action);
        return -1;
    }

    *operation = (OperationType)(action / 256);
    *pattern = (uint8_t)(action % 256);

    return 0;
}

int ma_is_initialized(void) {
    return g_state.initialized;
}
