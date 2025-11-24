# devdax 2MB Alignment Issue - mmap Migration

## 문제

`/sys/bus/dax/devices/dax1.0/align` = **2097152 (2MB)**

devdax는 2MB 단위로만 정렬된 접근을 허용합니다.
- `write()` / `read()` 시스템 콜 사용 시 `EINVAL (errno=22)` 발생
- 64바이트 단위 접근 불가능

## 해결 방법

**mmap()을 사용하여 2MB 단위로 매핑 후, 매핑된 메모리에 직접 접근**

### 변경 전 (write/read 방식)
```c
int fd = open("/dev/dax1.0", O_RDWR);
lseek(fd, 0x1000, SEEK_SET);  // EINVAL!
write(fd, buffer, 64);         // EINVAL!
```

### 변경 후 (mmap 방식)
```c
#define ALIGN_SIZE (2 * 1024 * 1024)  // 2MB

int fd = open("/dev/dax1.0", O_RDWR);

// 2MB 단위로 정렬
uint64_t aligned_offset = (offset / ALIGN_SIZE) * ALIGN_SIZE;
size_t map_size = ALIGN_SIZE;

// mmap으로 매핑
void* mapped = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd, aligned_offset);

// 매핑된 메모리에 직접 접근
uint8_t* ptr = (uint8_t*)mapped;
size_t offset_in_page = offset - aligned_offset;

// Write
memset(ptr + offset_in_page, pattern, 64);

// Read
uint8_t value = ptr[offset_in_page];

// Cleanup
munmap(mapped, map_size);
```

## 구현 상태

### ✅ 완료
- `write_ascending()` - mmap 방식으로 변경 완료
- `write_descending()` - mmap 방식으로 변경 완료
- `read_ascending()` - mmap 방식으로 변경 완료

### 🔄 진행 중
- `read_descending()` - 변경 필요
- `write_read_ascending()` - 변경 필요
- `write_read_descending()` - 변경 필요

## 나머지 함수 구현 가이드

### read_descending 예시
```c
static int read_descending(uint64_t start, uint64_t end, uint8_t expected_pattern) {
    (void)expected_pattern;
    #define ALIGN_SIZE (2 * 1024 * 1024)

    uint64_t aligned_start = (start / ALIGN_SIZE) * ALIGN_SIZE;
    uint64_t aligned_end = ((end + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
    size_t map_size = aligned_end - aligned_start;

    if (map_size == 0) map_size = ALIGN_SIZE;

    void* mapped = mmap(NULL, map_size, PROT_READ,
                        MAP_SHARED, g_state.devdax_fd, aligned_start);
    if (mapped == MAP_FAILED) {
        set_error("mmap failed: %s", strerror(errno));
        return -1;
    }

    volatile uint8_t* ptr = (volatile uint8_t*)mapped;
    size_t offset_start = start - aligned_start;
    size_t offset_end = (end - aligned_start < map_size) ?
                        (end - aligned_start) : map_size;

    uint8_t dummy;
    // Descending order
    for (size_t i = offset_end - 64; i >= offset_start; i -= 64) {
        dummy = ptr[i];
        (void)dummy;
        if (i == offset_start) break;
    }

    munmap(mapped, map_size);
    return 0;
}
```

### write_read_ascending 예시
```c
static int write_read_ascending(uint64_t start, uint64_t end, uint8_t pattern) {
    #define ALIGN_SIZE (2 * 1024 * 1024)

    uint64_t aligned_start = (start / ALIGN_SIZE) * ALIGN_SIZE;
    uint64_t aligned_end = ((end + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
    size_t map_size = aligned_end - aligned_start;

    if (map_size == 0) map_size = ALIGN_SIZE;

    void* mapped = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, g_state.devdax_fd, aligned_start);
    if (mapped == MAP_FAILED) {
        set_error("mmap failed: %s", strerror(errno));
        return -1;
    }

    volatile uint8_t* ptr = (volatile uint8_t*)mapped;
    size_t offset_start = start - aligned_start;
    size_t offset_end = (end - aligned_start < map_size) ?
                        (end - aligned_start) : map_size;

    // Single pass: write then read immediately
    for (size_t i = offset_start; i < offset_end; i += 64) {
        // Write
        memset((void*)(ptr + i), pattern, 64);
        // Read immediately
        volatile uint8_t dummy = ptr[i];
        (void)dummy;
    }

    msync(mapped, map_size, MS_SYNC);
    munmap(mapped, map_size);
    return 0;
}
```

### write_read_descending 예시
```c
static int write_read_descending(uint64_t start, uint64_t end, uint8_t pattern) {
    // write_read_ascending과 유사하지만 descending order
    // for loop를 역순으로 실행

    #define ALIGN_SIZE (2 * 1024 * 1024)

    uint64_t aligned_start = (start / ALIGN_SIZE) * ALIGN_SIZE;
    uint64_t aligned_end = ((end + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
    size_t map_size = aligned_end - aligned_start;

    if (map_size == 0) map_size = ALIGN_SIZE;

    void* mapped = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, g_state.devdax_fd, aligned_start);
    if (mapped == MAP_FAILED) {
        set_error("mmap failed: %s", strerror(errno));
        return -1;
    }

    volatile uint8_t* ptr = (volatile uint8_t*)mapped;
    size_t offset_start = start - aligned_start;
    size_t offset_end = (end - aligned_start < map_size) ?
                        (end - aligned_start) : map_size;

    // Descending: start from end
    for (size_t i = offset_end - 64; i >= offset_start; i -= 64) {
        // Write
        memset((void*)(ptr + i), pattern, 64);
        // Read immediately
        volatile uint8_t dummy = ptr[i];
        (void)dummy;
        if (i == offset_start) break;
    }

    msync(mapped, map_size, MS_SYNC);
    munmap(mapped, map_size);
    return 0;
}
```

## 주의사항

1. **항상 2MB 정렬 확인**
   - `aligned_start = (start / 2MB) * 2MB`
   - `aligned_end = ((end + 2MB - 1) / 2MB) * 2MB`

2. **msync() 호출**
   - Write 후에는 `msync(mapped, size, MS_SYNC)` 호출
   - 메모리 변경사항을 확실히 flush

3. **volatile 사용**
   - Read 시 compiler optimization 방지
   - `volatile uint8_t* ptr`

4. **munmap() 필수**
   - 사용 후 반드시 매핑 해제
   - 메모리 누수 방지

## 테스트 방법

```bash
# 1. 컴파일
cd src/MemoryAgent/c_library
make clean && make

# 2. 서버 실행 (dax1.0 사용!)
cd src/MemoryAgent
sudo python3 memory_agent_server.py \
    --devdax /dev/dax1.0 \
    --memory-size 128000 \
    --sampling-rate 0.01 \
    --port 5000

# 3. 테스트
curl -X POST http://localhost:5000/execute_action \
  -H "Content-Type: application/json" \
  -d '{"action": 0}'
```

## 성능 고려사항

- **mmap 오버헤드**: 매번 2MB를 매핑하므로 약간의 오버헤드
- **해결책**: 향후 persistent mmap 유지 고려
- **현재**: 정확성 우선, 성능은 나중에 최적화

## 참고

- devdax alignment 확인: `cat /sys/bus/dax/devices/dax1.0/align`
- devdax size 확인: `cat /sys/bus/dax/devices/dax1.0/size`
- man pages: `man mmap`, `man msync`
