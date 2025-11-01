# Phase 1 구현 일정 (Pattern Discovery)

**프로젝트**: CXL Type3 메모리 불량 검출 RL 시스템
**Phase 1 목표**: MRAT/stressapptest 패턴 재발견
**기간**: 6주 (Week 1-6)
**시작일**: 2025-11-04 (예정)

---

## 📅 전체 타임라인

```
Week 1-2: 환경 구축 및 기반 작업
Week 3-4: RL Agent 구현 및 시뮬레이션 테스트
Week 5-6: 실제 하드웨어 검증 및 분석
```

---

## Week 1: 환경 구축 (11/04 - 11/08)

### 목표
✅ 개발 환경 완성
✅ 불량 device 확보
✅ 시뮬레이터 준비

### 상세 일정

#### Day 1-2 (월-화): 개발 환경 세팅
**담당**: 개발팀 전체

- [ ] **개발 환경 구성**
  - Python 가상환경 설정 (venv)
  - 필수 패키지 설치 (gymnasium, pytorch, numpy)
  - Git 저장소 클론 및 동기화
  - C 라이브러리 빌드 확인

- [ ] **코드 리뷰 세션**
  - `phase1_environment.py` 리뷰
  - `mbist_interface.py` 리뷰
  - Action space (1536 actions) 이해
  - 질문 사항 정리

**체크포인트**:
```bash
# 각자 환경에서 테스트 실행
cd cxl_memory_rl_project
source venv/bin/activate
python src/phase1_environment.py
# 출력: "Phase 1 Environment Test" 성공
```

#### Day 3-4 (수-목): 불량 Device 확보 및 분석
**담당**: 하드웨어팀 + 개발팀 리드

- [ ] **불량 Device 수집**
  - MRAT FAIL device 최소 5개 확보
  - stressapptest PASS but 불량 device 찾기
  - Device ID 라벨링 (CXL-001, CXL-002, ...)
  - 불량 타입 기록

- [ ] **MRAT 분석**
  - MRAT 소스코드 분석 (가능하면)
  - MRAT 사용 패턴 추정
  - MRAT 실행 시간 측정
  - MRAT 검출 결과 문서화

**산출물**:
```
docs/faulty_devices_list.md
- Device ID, 불량 타입, MRAT 결과, stressapptest 결과
```

#### Day 5 (금): 시뮬레이터 개선
**담당**: 개발팀

- [ ] **시뮬레이터 불량 주입**
  - Mock mode에서 특정 패턴 FAIL 시뮬레이션
  - 다양한 불량 시나리오 추가
  - 재현 가능한 불량 생성

- [ ] **성능 최적화**
  - 메모리 스캔 속도 개선
  - Sampling rate 조정
  - 로깅 최적화

**주간 마일스톤**:
```
✅ 팀원 전원 환경 구축 완료
✅ 불량 device 5개 이상 확보
✅ 시뮬레이터 동작 확인
```

---

## Week 2: DQN Agent 기초 구현 (11/11 - 11/15)

### 목표
✅ DQN Agent 구현 완료
✅ 시뮬레이터에서 학습 검증
✅ 기본 성능 확인

### 상세 일정

#### Day 1-2 (월-화): DQN 네트워크 구현
**담당**: RL 개발자

- [ ] **Q-Network 설계**
  ```python
  # src/dqn_agent_phase1.py
  class DQNNetwork:
      - Input: state (sequence_history, last_result, ...)
      - Hidden: 256 -> 512 -> 256
      - Output: 1536 Q-values (one per action)
  ```

- [ ] **Experience Replay Buffer**
  - Replay buffer 크기: 10,000
  - Batch size: 64
  - Priority replay (optional)

- [ ] **Hyperparameters 설정**
  ```python
  learning_rate = 0.001
  gamma = 0.99
  epsilon_start = 1.0
  epsilon_end = 0.01
  epsilon_decay = 0.995
  target_update_frequency = 10 episodes
  ```

**체크포인트**:
- Network forward pass 테스트
- Replay buffer 동작 확인

#### Day 3-4 (수-목): 훈련 루프 구현
**담당**: RL 개발자 + 백엔드 개발자

- [ ] **Training Loop**
  ```python
  # src/train_phase1.py
  for episode in range(max_episodes):
      state = env.reset()
      while not done:
          action = agent.select_action(state, epsilon)
          next_state, reward, done, info = env.step(action)
          agent.store_experience(...)
          agent.train_step()
  ```

- [ ] **로깅 시스템**
  - TensorBoard 연동
  - Episode 별 통계
  - Reward 그래프
  - Success rate 추적

- [ ] **체크포인트 저장**
  - 모델 자동 저장 (매 100 episode)
  - Best model 저장
  - 훈련 재개 기능

#### Day 5 (금): 시뮬레이터 테스트
**담당**: 전체 팀

- [ ] **시뮬레이션 훈련**
  - Mock device에서 1000 episode 훈련
  - 수렴 확인
  - 패턴 발견 검증

- [ ] **팀 리뷰**
  - 코드 리뷰
  - 학습 결과 공유
  - 다음 주 계획 수립

**주간 마일스톤**:
```
✅ DQN Agent 구현 완료
✅ 시뮬레이터에서 학습 성공
✅ 기본 패턴 발견 확인
```

---

## Week 3: 시뮬레이션 최적화 (11/18 - 11/22)

### 목표
✅ 학습 안정화
✅ Hyperparameter 튜닝
✅ 다양한 불량 시나리오 테스트

### 상세 일정

#### Day 1-3 (월-수): Hyperparameter 튜닝
**담당**: RL 개발자 (병렬 실험)

- [ ] **Grid Search / Random Search**
  - Learning rate: [0.0001, 0.001, 0.01]
  - Network size: [128-256-128, 256-512-256, 512-1024-512]
  - Epsilon decay: [0.99, 0.995, 0.999]
  - Replay buffer: [5000, 10000, 20000]

- [ ] **실험 추적**
  ```
  experiments/
  ├── exp001_lr0.001_eps0.995/
  ├── exp002_lr0.0001_eps0.99/
  └── ...
  ```

- [ ] **Best configuration 선정**

#### Day 4-5 (목-금): 다양한 시나리오 테스트
**담당**: 전체 팀

- [ ] **다양한 불량 시나리오**
  - Scenario 1: Simple pattern (0x00, 0xFF)
  - Scenario 2: Checkerboard (0x55, 0xAA)
  - Scenario 3: Complex pattern
  - Scenario 4: Direction-sensitive fault

- [ ] **성능 분석**
  - 각 시나리오별 검출 성공률
  - 평균 에피소드 길이
  - 학습 수렴 속도

**주간 마일스톤**:
```
✅ 최적 hyperparameter 확정
✅ 5가지 이상 시나리오 테스트 완료
✅ 시뮬레이션 검증 완료
```

---

## Week 4: 실제 하드웨어 준비 (11/25 - 11/29)

### 목표
✅ MBIST 하드웨어 연동
✅ 안전 모드 구현
✅ 초기 하드웨어 테스트

### 상세 일정

#### Day 1-2 (월-화): MBIST 라이브러리 통합
**담당**: 시스템 개발자 + 하드웨어팀

- [ ] **Shared Library 빌드**
  ```bash
  cd /home/dhkang/data3/mbist_sample_code-gen2_es
  # .so 파일 생성
  gcc -shared -fPIC -o libmbist_rl.so \
      target/smbus/mt_rl_primitives.o ...
  ```

- [ ] **Python 연동**
  - Mock mode → Real mode 전환
  - 하드웨어 초기화 테스트
  - 단순 명령 실행 확인

- [ ] **에러 처리**
  - 하드웨어 오류 감지
  - 타임아웃 처리
  - 복구 메커니즘

#### Day 3-4 (수-목): 안전 장치 구현
**담당**: 시스템 개발자

- [ ] **Safety Limits**
  ```python
  class SafetyConfig:
      max_consecutive_tests = 1000
      max_test_duration = 3600  # 1 hour
      cooldown_time = 5.0  # seconds
      temperature_threshold = 85  # °C
      max_error_rate = 0.5  # 50%
  ```

- [ ] **Emergency Stop**
  - Ctrl+C graceful shutdown
  - 하드웨어 상태 복구
  - 로그 저장

- [ ] **모니터링 시스템**
  - 온도 모니터링 (가능하면)
  - 에러율 실시간 추적
  - 비정상 동작 감지

#### Day 5 (금): 첫 하드웨어 테스트
**담당**: 전체 팀 (관찰)

- [ ] **Test Device: CXL-001**
  - 안전 모드 활성화
  - 간단한 패턴 테스트 (10 episodes)
  - 결과 분석

- [ ] **문제점 파악**
  - 하드웨어 이슈
  - 성능 이슈
  - 안전성 이슈

**주간 마일스톤**:
```
✅ MBIST 하드웨어 연동 완료
✅ 안전 장치 구현 완료
✅ 첫 하드웨어 테스트 성공
```

---

## Week 5: 실제 하드웨어 검증 (12/02 - 12/06)

### 목표
✅ 불량 device에서 패턴 발견
✅ MRAT 수준 검출 검증
✅ 재현성 확인

### 상세 일정

#### Day 1-3 (월-수): 본격 훈련
**담당**: RL 개발자 + 하드웨어팀

- [ ] **Device별 훈련**
  ```
  CXL-001: MRAT FAIL (unknown pattern)
  CXL-002: MRAT FAIL (different pattern)
  CXL-003: stressapptest PASS + MRAT FAIL
  CXL-004: ...
  CXL-005: ...
  ```

- [ ] **훈련 실행**
  - 각 device당 500-1000 episodes
  - 24시간 모니터링
  - 자동 체크포인트

- [ ] **실시간 분석**
  - 패턴 발견 시 즉시 알림
  - 학습 곡선 분석
  - 중간 결과 공유

#### Day 4-5 (목-금): 재현성 테스트
**담당**: 검증팀

- [ ] **발견된 패턴 검증**
  - 각 device에서 발견한 패턴
  - 5회 반복 테스트
  - 100% 재현 확인

- [ ] **MRAT 비교**
  - RL 검출 vs MRAT 검출
  - 시퀀스 길이 비교
  - 실행 시간 비교

**주간 마일스톤**:
```
✅ 5개 device 모두 패턴 발견
✅ 재현성 100% 확인
✅ MRAT 수준 검출 달성
```

---

## Week 6: 분석 및 문서화 (12/09 - 12/13)

### 목표
✅ 결과 분석 및 리포트
✅ Phase 2 준비
✅ 프레젠테이션

### 상세 일정

#### Day 1-2 (월-화): 데이터 분석
**담당**: 데이터 분석팀

- [ ] **통계 분석**
  - 검출 성공률
  - 평균 시퀀스 길이
  - 학습 소요 시간
  - 패턴 빈도 분석

- [ ] **패턴 해석**
  - 발견된 패턴의 의미
  - MRAT과의 유사성
  - 새로운 인사이트

#### Day 3-4 (수-목): 문서 작성
**담당**: 전체 팀 (분담)

- [ ] **Phase 1 완료 보고서**
  ```
  docs/PHASE1_FINAL_REPORT.md
  - Executive Summary
  - 성과 및 KPI 달성도
  - 발견된 패턴 상세
  - 기술적 도전과 해결
  - 교훈 및 개선사항
  - Phase 2 권고사항
  ```

- [ ] **코드 문서화**
  - Docstring 정리
  - README 업데이트
  - API 문서 생성

#### Day 5 (금): 팀 리뷰 및 Phase 2 계획
**담당**: 전체 팀

- [ ] **발표 준비**
  - Phase 1 결과 프레젠테이션
  - 데모 준비
  - Q&A 준비

- [ ] **Phase 2 계획**
  - Phase 1 교훈 반영
  - Phase 2 action space 설계
  - 일정 수립

**주간 마일스톤**:
```
✅ Phase 1 완료 보고서
✅ 팀 발표 완료
✅ Phase 2 계획 수립
```

---

## 📊 KPI 및 성공 기준

### 정량적 목표

| 지표 | 목표 | 측정 방법 | 담당 |
|------|------|----------|------|
| **검출 성공률** | 100% | Known fault detection | 검증팀 |
| **재현성** | 5/5회 | Repeated tests | 검증팀 |
| **시퀀스 길이** | ≤10 steps | Episode length | RL팀 |
| **학습 시간** | ≤24시간/device | Training time | RL팀 |
| **False Positive** | 0% | Expert review | 하드웨어팀 |

### 정성적 목표

- [ ] 발견된 패턴이 해석 가능
- [ ] MRAT과의 비교 분석 완료
- [ ] 팀원 전원 프로젝트 이해
- [ ] 재사용 가능한 코드베이스

---

## 🎯 마일스톤 체크리스트

### Week 1 Milestone ✓
- [ ] 개발 환경 구축 (전원)
- [ ] 불량 device 5개 확보
- [ ] 시뮬레이터 동작 확인

### Week 2 Milestone ✓
- [ ] DQN Agent 구현 완료
- [ ] 시뮬레이터 학습 성공
- [ ] 기본 패턴 발견

### Week 3 Milestone ✓
- [ ] Hyperparameter 최적화
- [ ] 5+ 시나리오 테스트
- [ ] 시뮬레이션 검증 완료

### Week 4 Milestone ✓
- [ ] MBIST 하드웨어 연동
- [ ] 안전 장치 구현
- [ ] 첫 하드웨어 테스트

### Week 5 Milestone ✓
- [ ] 5개 device 패턴 발견
- [ ] 재현성 100% 달성
- [ ] MRAT 수준 도달

### Week 6 Milestone ✓
- [ ] 최종 보고서 작성
- [ ] 팀 발표 완료
- [ ] Phase 2 계획 수립

---

## 👥 역할 및 책임

### RL 개발자
- DQN Agent 구현 및 최적화
- Hyperparameter 튜닝
- 학습 모니터링

### 시스템 개발자
- MBIST 라이브러리 통합
- 안전 시스템 구축
- 성능 최적화

### 하드웨어팀
- 불량 device 확보 및 관리
- MRAT 분석
- 결과 검증

### 데이터 분석팀
- 결과 분석 및 시각화
- 패턴 해석
- 보고서 작성

### 프로젝트 리드
- 전체 일정 관리
- 팀 간 조율
- 의사결정

---

## 🚨 리스크 관리

### High Risk

#### R1: 불량 device 확보 실패
- **확률**: 중
- **영향**: 높음 (프로젝트 중단 가능)
- **대응**:
  - Week 1에 우선 확보
  - 대안: 시뮬레이터 고도화
  - 양산팀과 사전 협의

#### R2: 하드웨어 손상
- **확률**: 낮음
- **영향**: 높음 (비용, 일정)
- **대응**:
  - 안전 장치 철저히 구현
  - 불량 device만 사용
  - 보험/예산 확보

### Medium Risk

#### R3: 학습 수렴 실패
- **확률**: 중
- **영향**: 중
- **대응**:
  - Hyperparameter 튜닝
  - 네트워크 구조 변경
  - Expert demonstration 활용

#### R4: 일정 지연
- **확률**: 중
- **영향**: 중
- **대응**:
  - 주간 체크포인트 철저
  - Buffer 시간 활용
  - 우선순위 조정

---

## 📞 의사소통

### 일일 스탠드업 (매일 10:00, 15분)
- 어제 한 일
- 오늘 할 일
- 블로커

### 주간 리뷰 (매주 금요일 17:00, 1시간)
- 주간 성과 공유
- 문제점 논의
- 다음 주 계획

### 문서 공유
- **GitHub**: 코드, 문서
- **Slack/Teams**: 실시간 소통
- **Notion/Wiki**: 회의록, 결정사항

### 긴급 연락
- 하드웨어 이슈: 하드웨어팀 리드
- 코드 이슈: 개발팀 리드
- 일정 이슈: 프로젝트 리드

---

## 📚 참고 문서

- `PHASE1_PHASE2_STRATEGY.md` - 전략 문서
- `PROJECT_PLAN.md` - 전체 계획
- `C_LIBRARY_BUILD.md` - C 라이브러리 빌드
- `REVISED_PROJECT_DESIGN.md` - 프로젝트 설계

---

## 📝 변경 이력

- **2025-11-01**: 초안 작성
  - 6주 일정 수립
  - 역할 및 책임 정의
  - 마일스톤 설정
  - 리스크 관리 계획

---

## ✅ 다음 액션 (즉시)

1. **팀 킥오프 미팅 일정 잡기** (Week 1 시작 전)
2. **불량 device 확보 시작** (최우선!)
3. **개발 환경 가이드 공유** (README 업데이트)
4. **Slack/Teams 채널 생성**
5. **GitHub 권한 설정** (팀원 추가)

---

**작성자**: AI Assistant
**검토자**: (팀 리드가 검토 후 승인)
**승인일**: (미정)
**다음 리뷰**: Week 3 (중간 점검)
