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
#include <pthread.h>
#include <sched.h>

/* ========== Multi-threading Configuration ========== */
#define NUM_THREADS 16
#define ALIGN_SIZE (2 * 1024 * 1024)  /* 2MB alignment */>

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

/* ========== Multi-threading Support ========== */

typedef struct {
    int thread_id;
    int core_id;
    uint64_t start_dpa;
    uint64_t end_dpa;
    OperationType operation;
    uint8_t pattern;
    int result;
    char error_msg[256];
} ThreadTask;

/**
 * @brief Set CPU affinity for current thread
 */
static int set_cpu_affinity(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        return -1;
    }
    return 0;
}

/**
 * @brief Worker thread function
 */
static void* thread_worker(void* arg) {
    ThreadTask* task = (ThreadTask*)arg;

    /* Set CPU affinity to designated core */
    if (set_cpu_affinity(task->core_id) < 0) {
        snprintf(task->error_msg, sizeof(task->error_msg),
                 "Failed to set CPU affinity to core %d", task->core_id);
        task->result = -1;
        return NULL;
    }

    /* Execute operation on assigned memory region */
    switch (task->operation) {
        case WR_ASC_ASC:
            if (write_ascending(task->start_dpa, task->end_dpa, task->pattern) < 0) {
                snprintf(task->error_msg, sizeof(task->error_msg), "write_ascending failed");
                task->result = -1;
                return NULL;
            }
            if (read_ascending(task->start_dpa, task->end_dpa, task->pattern) < 0) {
                snprintf(task->error_msg, sizeof(task->error_msg), "read_ascending failed");
                task->result = -1;
                return NULL;
            }
            break;

        case WR_DESC_DESC:
            if (write_descending(task->start_dpa, task->end_dpa, task->pattern) < 0) {
                snprintf(task->error_msg, sizeof(task->error_msg), "write_descending failed");
                task->result = -1;
                return NULL;
            }
            if (read_descending(task->start_dpa, task->end_dpa, task->pattern) < 0) {
                snprintf(task->error_msg, sizeof(task->error_msg), "read_descending failed");
                task->result = -1;
                return NULL;
            }
            break;

        case WR_ASC_DESC:
            if (write_ascending(task->start_dpa, task->end_dpa, task->pattern) < 0) {
                snprintf(task->error_msg, sizeof(task->error_msg), "write_ascending failed");
                task->result = -1;
                return NULL;
            }
            if (read_descending(task->start_dpa, task->end_dpa, task->pattern) < 0) {
                snprintf(task->error_msg, sizeof(task->error_msg), "read_descending failed");
                task->result = -1;
                return NULL;
            }
            break;

        case WR_DESC_ASC:
            if (write_descending(task->start_dpa, task->end_dpa, task->pattern) < 0) {
                snprintf(task->error_msg, sizeof(task->error_msg), "write_descending failed");
                task->result = -1;
                return NULL;
            }
            if (read_ascending(task->start_dpa, task->end_dpa, task->pattern) < 0) {
                snprintf(task->error_msg, sizeof(task->error_msg), "read_ascending failed");
                task->result = -1;
                return NULL;
            }
            break;

        case WR_DESC_SINGLE:
            if (write_read_descending(task->start_dpa, task->end_dpa, task->pattern) < 0) {
                snprintf(task->error_msg, sizeof(task->error_msg), "write_read_descending failed");
                task->result = -1;
                return NULL;
            }
            break;

        case WR_ASC_SINGLE:
            if (write_read_ascending(task->start_dpa, task->end_dpa, task->pattern) < 0) {
                snprintf(task->error_msg, sizeof(task->error_msg), "write_read_ascending failed");
                task->result = -1;
                return NULL;
            }
            break;

        default:
            snprintf(task->error_msg, sizeof(task->error_msg),
                     "Invalid operation type: %d", task->operation);
            task->result = -1;
            return NULL;
    }

    task->result = 0;
    return NULL;
}

/**
 * @brief Execute memory operation with multi-threading (16 threads, 16 cores)
 */
static int execute_operation(OperationType operation, uint8_t pattern) {
    pthread_t threads[NUM_THREADS];
    ThreadTask tasks[NUM_THREADS];

    uint64_t total_size = g_state.memory_size;
    uint64_t chunk_size = total_size / NUM_THREADS;

    /* Align chunk_size to 2MB boundary */
    chunk_size = (chunk_size / ALIGN_SIZE) * ALIGN_SIZE;
    if (chunk_size == 0) {
        chunk_size = ALIGN_SIZE;
    }

    /* Create threads and assign memory regions */
    for (int i = 0; i < NUM_THREADS; i++) {
        tasks[i].thread_id = i;
        tasks[i].core_id = i;  /* Core 0-15 */
        tasks[i].start_dpa = i * chunk_size;

        /* Last thread handles remaining memory */
        if (i == NUM_THREADS - 1) {
            tasks[i].end_dpa = total_size;
        } else {
            tasks[i].end_dpa = (i + 1) * chunk_size;
        }

        tasks[i].operation = operation;
        tasks[i].pattern = pattern;
        tasks[i].result = 0;
        tasks[i].error_msg[0] = '\0';

        if (pthread_create(&threads[i], NULL, thread_worker, &tasks[i]) != 0) {
            set_error("Failed to create thread %d: %s", i, strerror(errno));

            /* Wait for already created threads */
            for (int j = 0; j < i; j++) {
                pthread_join(threads[j], NULL);
            }
            return -1;
        }
    }

    /* Wait for all threads to complete */
    int has_error = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);

        if (tasks[i].result != 0) {
            set_error("Thread %d (core %d) failed: %s",
                     tasks[i].thread_id, tasks[i].core_id, tasks[i].error_msg);
            has_error = 1;
            /* Continue joining other threads */
        }
    }

    if (has_error) {
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
