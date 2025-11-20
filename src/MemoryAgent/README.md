# Memory Agent

GNR-SP에서 실행되는 메모리 테스트 에이전트입니다.

## 구성 요소

- **c_library/** - C로 작성된 메모리 테스트 라이브러리
  - `memory_agent.c` - devdax 접근 및 umxc CE 감지
  - `libmemory_agent.so` - 컴파일된 공유 라이브러리

- **memory_agent_c_wrapper.py** - C 라이브러리 Python wrapper (ctypes)

- **memory_agent_server.py** - Flask REST API 서버
  - `/health` - 헬스 체크
  - `/execute_action` - 메모리 테스트 액션 실행
  - `/reset_baseline` - CE baseline 리셋
  - `/get_ce_info` - 현재 CE 정보 조회

- **ce_monitor.py** - CE 모니터링 유틸리티

- **devdax_interface.py** - devdax 직접 접근 인터페이스

- **dpa_translator.py** - DPA ↔ DRAM 주소 변환

## 실행 방법

```bash
# GNR-SP에서 실행
cd /home/dhkang/cxl_memory_rl_project
python3 src/MemoryAgent/memory_agent_server.py \
    --devdax /dev/dax0.0 \
    --memory-size 128000 \
    --port 5000
```

## 요구사항

- Linux (GNR-SP)
- Python 3.8+
- Flask, flask-cors
- root 권한 (devdax 접근용)
- umxc tool (CE 감지용)
