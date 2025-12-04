/**
 * @file gsat_like_memory_agent.c
 * @brief GSAT-like Memory Agent Implementation (StressAppTest Style)
 */

#include "gsat_like_memory_agent.h"
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
#include <time.h>
#include <immintrin.h> /* For _mm_clflush, _mm_mfence */

/* ========== Tuning Parameters ========== */
#define NUM_THREADS 16              /* GNR-SP CPU 코어 수에 맞게 조정 */
#define ALIGN_SIZE (2 * 1024 * 1024) /* 2MB Hugepage Alignment */
#define STRESS_DURATION_SEC 5       /* Action 1회당 5초 지속 부하 */
#define NUM_KEY_PATTERNS 16         /* StressAppTest 기반 핵심 패턴 개수 */

/* ========== StressAppTest Key Patterns ========== */
/* Based on Google StressAppTest: https://github.com/stressapptest/stressapptest */
static const uint8_t KEY_PATTERNS[NUM_KEY_PATTERNS] = {
    0x00,  /* All zeros */
    0xFF,  /* All ones */
    0x55,  /* 01010101 - Checkerboard A */
    0xAA,  /* 10101010 - Checkerboard B */
    0xF0,  /* 11110000 */
    0x0F,  /* 00001111 */
    0xCC,  /* 11001100 */
    0x33,  /* 00110011 */
    0x01,  /* Walking ones start */
    0x80,  /* Walking ones end */
    0x16,  /* 8b10b low transition density */
    0xB5,  /* 8b10b high transition density */
    0x4A,  /* Checker pattern */
    0x57,  /* Edge case 1 */
    0x02,  /* Edge case 2 */
    0xFD   /* Edge case 3 */
};

/* ========== Global State ========== */
typedef struct {
    int initialized;
    int devdax_fd;
    char devdax_path[256];
    size_t memory_size;
    int baseline_volatile;
    int baseline_persistent;
    char error_msg[512];
} MemoryAgentState;

static MemoryAgentState g_state = {0};

typedef struct {
    int thread_id;
    int core_id;
    uint64_t start_dpa;
    uint64_t size;
    OperationType operation;
    uint8_t pattern;
    int result;
    char error_msg[256];
} ThreadTask;

/* ========== Helper Functions ========== */

static void set_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_state.error_msg, sizeof(g_state.error_msg), fmt, args);
    va_end(args);
}

/* Execute umxc command and parse stats */
static int execute_umxc(CEInfo* ce_info) {
    FILE* fp;
    char buffer[256];
    int volatile_count = 0, persistent_count = 0, temperature = 0, health_status = 0;

    fp = popen("umxc mbox -H", "r");
    if (fp == NULL) {
        set_error("Failed to execute umxc: %s", strerror(errno));
        return -1;
    }

    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        if (strstr(buffer, "[0Ah]") && strstr(buffer, "Corrected Volatile Error Count"))
            sscanf(buffer, "%*[^:]: %d", &volatile_count);
        if (strstr(buffer, "[0Eh]") && strstr(buffer, "Corrected Persistent Error Count"))
            sscanf(buffer, "%*[^:]: %d", &persistent_count);
        if (strstr(buffer, "[04h]") && strstr(buffer, "Device Temperature"))
            sscanf(buffer, "%*[^:]: %dC", &temperature);
        if (strstr(buffer, "[00h]") && strstr(buffer, "Health Status"))
            sscanf(buffer, "%*[^:]: 0x%x", &health_status);
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

/* ========== Core Stress Logic (StressAppTest Style) ========== */

static void stress_memory_chunk(uint8_t* ptr, size_t size, OperationType op, uint8_t pattern) {
    volatile uint8_t* vptr = (volatile uint8_t*)ptr;
    uint8_t dummy;
    size_t half_size = size / 2;
    uint8_t pat_a = pattern;
    uint8_t pat_b = ~pattern; /* Inverted pattern (Bit Flip) */

    switch (op) {
        case OP_FILL: 
            /* [FILL] Simple Write - Thermal Stress */
            for (size_t i = 0; i < size; i += 64) {
                memset(ptr + i, pat_a, 64);
                _mm_clflush(ptr + i);
            }
            break;

        case OP_INVERT:
            /* [INVERT] Write -> Invert -> Write (High Noise) */
            for (size_t i = 0; i < size; i += 64) {
                memset(ptr + i, pat_a, 64); 
                _mm_clflush(ptr + i); 
                memset(ptr + i, pat_b, 64); /* Immediate Inversion */
                _mm_clflush(ptr + i);
            }
            break;

        case OP_COPY:
            /* [COPY] Memcpy (Bandwidth Saturation) */
            /* 1. Init Source */
            for (size_t i = 0; i < half_size; i+=64) {
                memset(ptr + i, pat_a, 64);
                _mm_clflush(ptr + i);
            }
            _mm_mfence();
            /* 2. Copy Loop */
            for (size_t i = 0; i < half_size; i += 64) {
                memcpy(ptr + half_size + i, ptr + i, 64);
                _mm_clflush(ptr + i);
                _mm_clflush(ptr + half_size + i);
            }
            break;

        case OP_CHECK:
            /* [CHECK] Read Only (Read Disturb) */
            for (size_t i = 0; i < size; i += 64) {
                _mm_clflush(ptr + i); 
                _mm_mfence();
                dummy = vptr[i]; /* Force DRAM Read */
                (void)dummy;
            }
            break;
    }
    _mm_mfence();
}

static void* thread_worker(void* arg) {
    ThreadTask* task = (ThreadTask*)arg;
    
    /* 1. CPU Pinning */
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(task->core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    /* 2. mmap (Once per thread, aligned) */
    uint64_t aligned_start = (task->start_dpa / ALIGN_SIZE) * ALIGN_SIZE;
    uint64_t map_offset = task->start_dpa - aligned_start;
    size_t map_len = task->size + map_offset;
    if (map_len % ALIGN_SIZE != 0) map_len = ((map_len / ALIGN_SIZE) + 1) * ALIGN_SIZE;

    void* mapped = mmap(NULL, map_len, PROT_READ | PROT_WRITE, 
                        MAP_SHARED, g_state.devdax_fd, aligned_start);
    
    if (mapped == MAP_FAILED) {
        task->result = -1;
        snprintf(task->error_msg, 256, "mmap failed");
        return NULL;
    }

    uint8_t* target_ptr = (uint8_t*)mapped + map_offset;
    uint8_t current_pattern = task->pattern;

    /* 3. Time-based Stress Loop (The "Sustained" part) */
    time_t start_time = time(NULL);
    while (time(NULL) - start_time < STRESS_DURATION_SEC) {
        stress_memory_chunk(target_ptr, task->size, task->operation, current_pattern);
        current_pattern = ~current_pattern; /* Rotate Pattern */
    }

    munmap(mapped, map_len);
    task->result = 0;
    return NULL;
}

/* ========== Public API ========== */

int ma_init(const char* devdax_path, size_t memory_size_mb) {
    if (g_state.initialized) return -1;
    
    g_state.devdax_fd = open(devdax_path, O_RDWR);
    if (g_state.devdax_fd < 0) {
        set_error("Failed to open %s", devdax_path);
        return -1;
    }

    strncpy(g_state.devdax_path, devdax_path, 255);
    g_state.memory_size = memory_size_mb * 1024UL * 1024UL;
    g_state.initialized = 1;
    printf("MA Initialized: %s, %zu MB\n", devdax_path, memory_size_mb);
    return 0;
}

int ma_execute_action(int action, ActionResult* result) {
    if (!g_state.initialized) { set_error("Not initialized"); return -1; }

    /* Decode Action (0-63): 4 operations × 16 patterns = 64 actions */
    if (action < 0 || action >= 64) {
        set_error("Invalid action: %d (must be 0-63)", action);
        return -1;
    }

    OperationType operation = (OperationType)(action / NUM_KEY_PATTERNS);
    int pattern_idx = action % NUM_KEY_PATTERNS;
    uint8_t pattern = KEY_PATTERNS[pattern_idx];

    ma_reset_baseline();

    /* Thread Setup */
    pthread_t threads[NUM_THREADS];
    ThreadTask tasks[NUM_THREADS];
    uint64_t chunk_size = (g_state.memory_size / NUM_THREADS / ALIGN_SIZE) * ALIGN_SIZE;
    if (chunk_size == 0) chunk_size = ALIGN_SIZE;

    printf(">> Stress: Op=%d, Pat=0x%02X, Dur=%ds\n", operation, pattern, STRESS_DURATION_SEC);

    for (int i = 0; i < NUM_THREADS; i++) {
        tasks[i].thread_id = i;
        tasks[i].core_id = i;
        tasks[i].start_dpa = i * chunk_size;
        tasks[i].size = (i == NUM_THREADS-1) ? (g_state.memory_size - tasks[i].start_dpa) : chunk_size;
        tasks[i].operation = operation;
        tasks[i].pattern = pattern;
        tasks[i].result = 0;
        pthread_create(&threads[i], NULL, thread_worker, &tasks[i]);
    }

    /* Wait */
    int has_error = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        if (tasks[i].result != 0) has_error = 1;
    }

    if (has_error) {
        result->success = 0;
        strncpy(result->error_message, "Thread Error", 255);
        return -1;
    }

    /* Stats */
    execute_umxc(&result->ce_info);
    result->success = 1;
    return 0;
}

int ma_get_ce_info(CEInfo* info) { return execute_umxc(info); }
int ma_reset_baseline(void) { CEInfo c; execute_umxc(&c); g_state.baseline_volatile += c.volatile_count; g_state.baseline_persistent += c.persistent_count; return 0; }
void ma_cleanup(void) { if (g_state.devdax_fd >= 0) close(g_state.devdax_fd); g_state.initialized = 0; }
const char* ma_get_error(void) { return g_state.error_msg; }
int ma_is_initialized(void) { return g_state.initialized; }
