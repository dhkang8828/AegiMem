# Phase#1 개선 사항 제안

## 작성일: 2024-11-20

## 현재 문제점

### 1. 보상 체계 문제
```python
# 현재
CE detected: +1000
No CE: +1
```

**문제점:**
- CE가 매우 드물게 발생한다면 agent가 거의 항상 +1만 받음
- 1000:1 비율은 너무 극단적
- 탐색(exploration) 유도 메커니즘 부족
- CE 발견 전까지 의미있는 학습 신호 없음

### 2. Single Action 실행 성능 문제
- 매 step마다 REST API 1회 호출
- 네트워크 latency 누적
- Memory Agent도 매번 umxc 호출 (매우 느림)
- 1536 actions을 모두 시도하려면 너무 오래 걸림

### 3. 탐색 부족
- Random exploration만으로는 CE 발견 어려움
- 패턴 간 상관관계 학습 어려움
- Action space가 너무 큼 (1536 actions)

---

## 개선 방안

### 💡 제안 1: Shaped Reward + Curiosity-driven Exploration

#### 목표
- CE 발견 전에도 의미있는 학습 신호 제공
- 탐색 장려
- 패턴 다양성 유도

#### 구현 방안

```python
def _calculate_reward(self, result: Dict, step_num: int, history: List) -> float:
    """
    개선된 보상 함수

    1. CE Detection: +100 (기존 1000에서 줄임)
    2. Progressive reward: 이전 step과 비교
    3. Pattern diversity bonus: 새로운 패턴 시도 시
    4. Sequence coherence: 논리적 시퀀스 보너스
    """
    reward = 0.0

    if result['ce_detected']:
        # ===== CE 발견 =====
        # 큰 보상
        reward += 100.0

        # CE 개수에 비례 (많이 발견할수록 좋음)
        reward += result['ce_total'] * 5.0

        # 짧은 시퀀스 보너스 (빨리 찾을수록 좋음)
        reward += 50.0 / step_num

    else:
        # ===== CE 없음 - 하지만 유용한 정보 =====

        # 1. Novelty bonus: 새로운 action 시도
        if action not in history:
            reward += 5.0  # 탐색 장려
        else:
            reward -= 1.0  # 중복 패널티

        # 2. Pattern exploration bonus
        operation_type = action // 256
        pattern = action % 256

        # 다양한 operation 시도 장려
        unique_ops = len(set([a // 256 for a in history]))
        reward += unique_ops * 0.5

        # 다양한 pattern 시도 장려
        unique_patterns = len(set([a % 256 for a in history]))
        reward += unique_patterns * 0.1

        # 3. 특정 패턴 조합 보너스 (heuristic)
        if step_num > 1:
            prev_op = history[-1] // 256
            curr_op = operation_type

            # March-like sequence 보너스
            if prev_op == 0 and curr_op == 1:  # ASC→DESC
                reward += 2.0
            elif prev_op == 1 and curr_op == 0:  # DESC→ASC
                reward += 2.0

            # Cross pattern 보너스
            if prev_op in [0, 1] and curr_op in [2, 3]:
                reward += 1.5

    # 4. Time penalty (너무 길어지면 페널티)
    reward -= step_num * 0.1

    return reward
```

#### 수정 파일
- `src/RLAgent/phase1_environment_distributed.py`
  - `_calculate_reward()` 함수 교체
  - `step()` 함수에서 history 전달

#### 예상 효과
- CE 없어도 탐색 유도
- 다양한 패턴 시도
- 학습 속도 향상

---

### 💡 제안 2: Batch Execution으로 성능 개선 ⭐⭐⭐ (최우선)

#### 목표
- REST API 호출 횟수 대폭 감소
- umxc 호출 횟수 감소
- 실행 시간 10배 이상 단축

#### A. Environment 수정

```python
class Phase1EnvironmentDistributed(gym.Env):
    def __init__(
        self,
        memory_agent_url: str = "http://192.168.3.20:5000",
        batch_size: int = 5,  # 새로운 파라미터!
        ...
    ):
        self.batch_size = batch_size
        self.action_buffer = []

    def step(self, action: int):
        """Buffered step execution"""
        self.action_buffer.append(action)
        self.step_count += 1

        # Buffer가 찼거나 마지막 step이면 batch 실행
        should_execute = (
            len(self.action_buffer) >= self.batch_size or
            self.step_count >= self.max_seq_len
        )

        if should_execute:
            # Batch 실행
            results = self._execute_batch_remote(self.action_buffer)

            # CE 발견 여부 확인
            ce_detected_at = None
            for i, result in enumerate(results):
                if result['ce_detected']:
                    ce_detected_at = i
                    break

            # Buffer 초기화
            self.action_buffer = []

            # 결과 처리
            if ce_detected_at is not None:
                # CE 발견!
                final_result = results[ce_detected_at]
                terminated = True
                reward = self._calculate_reward(final_result, self.step_count)
            else:
                # 모두 PASS
                final_result = results[-1]
                terminated = False
                reward = sum([self._calculate_reward(r, i+1)
                             for i, r in enumerate(results)])

            return self._get_observation(), reward, terminated, False, {}
        else:
            # Buffer 중간 - 아무것도 반환 안함 (gym.Env 확장 필요)
            # 또는 intermediate reward 반환
            return self._get_observation(), 0.0, False, False, {}

    def _execute_batch_remote(self, actions: List[int]) -> List[Dict]:
        """
        Batch로 actions 실행

        Args:
            actions: List of action indices

        Returns:
            List of results (CE 발견 시 중간에 중단됨)
        """
        response = requests.post(
            f"{self.memory_agent_url}/execute_batch",
            json={'actions': actions},
            timeout=self.timeout * len(actions)
        )
        response.raise_for_status()
        return response.json()['results']
```

#### B. Memory Agent Server API 추가

```python
# src/MemoryAgent/memory_agent_server.py

@app.route('/execute_batch', methods=['POST'])
def execute_batch():
    """
    배치 실행 엔드포인트

    Request: {
        "actions": [0, 1, 2, 3, 4]  # 여러 action
    }

    Response: {
        "results": [
            {"ce_detected": False, "ce_total": 0, ...},
            {"ce_detected": False, "ce_total": 0, ...},
            {"ce_detected": True, "ce_total": 5, ...},  # 여기서 중단!
        ],
        "stopped_at": 2,  # CE 발견한 index
        "total_executed": 3
    }
    """
    try:
        data = request.get_json()
        actions = data.get('actions', [])

        if not actions:
            return jsonify({'error': 'No actions provided'}), 400

        results = []

        for i, action in enumerate(actions):
            logging.info(f"Batch action {i}/{len(actions)}: {action}")

            # Execute via C library
            ce_info, success = memory_agent.execute_action(action)

            # Decode action
            operation_type, pattern = memory_agent.decode_action(action)

            result = {
                'success': success,
                'ce_detected': ce_info.has_errors(),
                'ce_volatile': ce_info.volatile_count,
                'ce_persistent': ce_info.persistent_count,
                'ce_total': ce_info.total_count,
                'temperature': ce_info.temperature,
                'operation': OperationType.name(operation_type),
                'pattern': f'0x{pattern:02X}'
            }
            results.append(result)

            # CE 발견하면 즉시 중단
            if ce_info.has_errors():
                logging.info(f"CE detected at batch index {i}, stopping")
                return jsonify({
                    'results': results,
                    'stopped_at': i,
                    'total_executed': i + 1,
                    'ce_detected': True
                })

        # 모두 PASS
        return jsonify({
            'results': results,
            'stopped_at': len(actions) - 1,
            'total_executed': len(actions),
            'ce_detected': False
        })

    except Exception as e:
        logging.error(f"Batch execution error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

#### C. C Library 최적화 (선택적 - 추가 성능 향상)

```c
// include/memory_agent.h
int ma_execute_batch(int* actions, int count, ActionResult* results);

// src/MemoryAgent/c_library/memory_agent.c

int ma_execute_batch(int* actions, int count, ActionResult* results) {
    // Baseline 한 번만 기록
    CEInfo baseline;
    execute_umxc(&baseline);

    // 모든 actions 실행
    for (int i = 0; i < count; i++) {
        int operation_type = actions[i] / 256;
        unsigned char pattern = actions[i] % 256;

        // Execute operation
        switch (operation_type) {
            case WR_ASC_ASC:
                write_ascending(pattern);
                read_ascending(pattern);
                break;
            case WR_DESC_DESC:
                write_descending(pattern);
                read_descending(pattern);
                break;
            // ... other operations
        }

        // 중간 CE 체크 (선택적)
        if (i % 5 == 0) {  // 5개마다 체크
            CEInfo current;
            execute_umxc(&current);

            if (current.total_count > baseline.total_count) {
                // CE 발견! 중단
                results[i].ce_info = current;
                return i + 1;  // 실행한 개수 반환
            }
        }
    }

    // 마지막 umxc 한 번만 호출
    CEInfo final;
    execute_umxc(&final);

    // Delta 계산
    results[count-1].ce_info.total_count =
        final.total_count - baseline.total_count;

    return count;  // 모두 실행
}
```

#### 수정 파일
1. `src/RLAgent/phase1_environment_distributed.py`
   - `__init__()`: batch_size 파라미터 추가
   - `step()`: buffering 로직 추가
   - `_execute_batch_remote()`: 새 함수 추가

2. `src/MemoryAgent/memory_agent_server.py`
   - `/execute_batch` 엔드포인트 추가

3. (선택) `src/MemoryAgent/c_library/memory_agent.c`
   - `ma_execute_batch()` 함수 추가

4. (선택) `src/MemoryAgent/memory_agent_c_wrapper.py`
   - `execute_batch()` Python wrapper 추가

#### 예상 성능 개선
```
현재: 1 action = 1 REST call + 1 umxc call
     10 actions = 10 REST + 10 umxc ≈ 30초

개선: 10 actions = 1 REST call + 1 umxc call ≈ 3초
     → 10배 빠름!

C batch까지 적용:
     10 actions = 1 REST call + 1 umxc call ≈ 1초
     → 30배 빠름!
```

---

### 💡 제안 3: Curriculum Learning (단계적 난이도)

#### 목표
- 초기 학습 속도 향상
- Action space를 점진적으로 확장
- 쉬운 문제부터 어려운 문제로

#### 구현 방안

```python
class CurriculumPhase1Environment(Phase1EnvironmentDistributed):
    """단계적 난이도 증가"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.difficulty_level = 0
        self.success_count = 0
        self.total_episodes = 0

    def get_allowed_actions(self):
        """난이도에 따라 허용된 actions 반환"""
        if self.difficulty_level == 0:
            # Level 0: 2개 operation, 16개 pattern
            # Total: 32 actions
            ops = [0, 1]  # WR_ASC_ASC, WR_DESC_DESC
            patterns = list(range(16))  # 0x00-0x0F

        elif self.difficulty_level == 1:
            # Level 1: 4개 operation, 64개 pattern
            # Total: 256 actions
            ops = [0, 1, 2, 3]
            patterns = list(range(64))

        elif self.difficulty_level == 2:
            # Level 2: 6개 operation, 128개 pattern
            # Total: 768 actions
            ops = list(range(6))
            patterns = list(range(128))

        else:
            # Level 3: 전체
            # Total: 1536 actions
            ops = list(range(6))
            patterns = list(range(256))

        # Action list 생성
        actions = []
        for op in ops:
            for pat in patterns:
                actions.append(op * 256 + pat)

        return actions

    def step(self, action: int):
        """Step with curriculum"""
        # Allowed actions만 허용
        allowed = self.get_allowed_actions()
        if action not in allowed:
            # 허용 안된 action → 가장 가까운 것으로 매핑
            action = min(allowed, key=lambda x: abs(x - action))

        obs, reward, terminated, truncated, info = super().step(action)

        # Episode 종료 시 통계 업데이트
        if terminated or truncated:
            self.total_episodes += 1
            if terminated:
                self.success_count += 1

            # 난이도 조정
            self._update_difficulty()

        return obs, reward, terminated, truncated, info

    def _update_difficulty(self):
        """성공률에 따라 난이도 증가"""
        if self.total_episodes < 10:
            return  # 최소 10 episodes 필요

        success_rate = self.success_count / self.total_episodes

        # 70% 이상 성공하면 난이도 증가
        if success_rate > 0.7 and self.difficulty_level < 3:
            self.difficulty_level += 1
            logging.info(f"Difficulty increased to level {self.difficulty_level}")
            logging.info(f"New action space: {len(self.get_allowed_actions())} actions")

            # 통계 초기화
            self.success_count = 0
            self.total_episodes = 0
```

#### 수정 파일
- 새 파일: `src/RLAgent/phase1_curriculum_environment.py`
- 또는 `phase1_environment_distributed.py`에 통합

#### 예상 효과
- 초기 학습 30% 이상 빠름
- 수렴 안정성 향상

---

### 💡 제안 4: Prior Knowledge Injection (사전 지식 활용)

#### 목표
- 알려진 좋은 패턴으로 bootstrap
- 탐색 공간 효율적으로 줄임

#### 구현 방안

```python
def get_heuristic_action_sequences():
    """
    알려진 메모리 테스트 패턴들

    Returns:
        List of action sequences
    """
    sequences = []

    # 1. Walking 1/0 pattern
    sequences.append([
        0 * 256 + 0x00,  # ASC write all 0
        0 * 256 + 0xFF,  # ASC write all 1
    ])

    # 2. Checkerboard pattern
    sequences.append([
        0 * 256 + 0xAA,  # ASC write 0xAA
        0 * 256 + 0x55,  # ASC write 0x55
    ])

    # 3. March C- like
    sequences.append([
        0 * 256 + 0x00,  # ASC write 0
        0 * 256 + 0xFF,  # ASC read 0, write 1
        1 * 256 + 0x00,  # DESC read 1, write 0
        1 * 256 + 0x00,  # DESC read 0
    ])

    # 4. Row hammer like (cross pattern)
    sequences.append([
        2 * 256 + 0xAA,  # ASC write, DESC read 0xAA
        3 * 256 + 0x55,  # DESC write, ASC read 0x55
    ])

    # 5. Sliding diagonal
    sequences.append([
        4 * 256 + 0xF0,  # DESC single 0xF0
        5 * 256 + 0x0F,  # ASC single 0x0F
    ])

    # 6. Complementary patterns
    for pattern in [0x00, 0xFF, 0xAA, 0x55, 0xF0, 0x0F]:
        sequences.append([
            0 * 256 + pattern,
            0 * 256 + (pattern ^ 0xFF),  # Complement
        ])

    return sequences


# Training script에서 사용
def pretrain_with_heuristics(agent, env, sequences):
    """
    Heuristic sequences로 pre-training

    Args:
        agent: RL agent (DQN or PPO)
        env: Environment
        sequences: List of action sequences
    """
    logging.info("Pre-training with heuristic sequences...")

    for seq_idx, sequence in enumerate(sequences):
        obs, _ = env.reset()

        for step, action in enumerate(sequence):
            obs, reward, terminated, truncated, info = env.step(action)

            # Replay buffer에 추가
            if hasattr(agent, 'replay_buffer'):
                agent.replay_buffer.add(obs, action, reward, obs, terminated)

            if terminated or truncated:
                break

        logging.info(f"Heuristic sequence {seq_idx+1}/{len(sequences)}: "
                    f"{len(sequence)} steps, final reward={reward}")

    # Pre-training
    if hasattr(agent, 'train_step'):
        for _ in range(100):
            agent.train_step()

    logging.info("Pre-training complete")
```

#### 수정 파일
- 새 파일: `src/RLAgent/heuristics.py`
- Training script에서 import 후 사용

#### 예상 효과
- 초기 탐색 효율 향상
- 빠른 baseline 확보

---

## 구현 우선순위

### 🥇 1순위: Batch Execution (필수)
**이유:**
- 성능 10-30배 향상
- 구현 비교적 간단
- 즉시 효과 큼

**작업량:** 중간
**예상 시간:** 2-3시간

---

### 🥈 2순위: Shaped Reward (중요)
**이유:**
- CE 없어도 학습 가능
- 탐색 크게 개선
- 학습 속도 향상

**작업량:** 적음
**예상 시간:** 1시간

---

### 🥉 3순위: Curriculum Learning (선택)
**이유:**
- 초기 수렴 빠름
- 안정성 향상
- 하지만 필수는 아님

**작업량:** 중간
**예상 시간:** 2시간

---

### 4순위: Prior Knowledge (보조)
**이유:**
- Bootstrap 도움
- 하지만 없어도 됨
- Heuristic에 의존

**작업량:** 적음
**예상 시간:** 30분

---

## 구현 로드맵

### Phase 1: 성능 개선 (Week 1)
- [ ] Batch Execution API 추가
- [ ] Environment batch 지원
- [ ] 성능 테스트

### Phase 2: 학습 개선 (Week 1-2)
- [ ] Shaped Reward 구현
- [ ] 보상 함수 튜닝
- [ ] 학습 실험

### Phase 3: 고급 기능 (Week 2+)
- [ ] Curriculum Learning (선택)
- [ ] Prior Knowledge (선택)
- [ ] 최종 튜닝

---

## 참고사항

### Batch Size 선택
- **Small (2-3)**: 빠른 피드백, CE 조기 발견
- **Medium (5-10)**: 균형적
- **Large (20+)**: 최대 성능, CE 발견 늦을 수 있음

**권장:** 5-10 (균형)

### Reward Tuning
초기값으로 시작 후 실험적으로 조정:
- CE detected: 100
- Novelty: 5
- Diversity: 0.5
- Time penalty: 0.1

### Monitoring
다음 메트릭 추적:
- Success rate (CE 발견률)
- Average sequence length
- Unique actions tried
- Execution time per episode

---

## 다음 회의 시 논의 사항

1. **Batch size** 얼마로 시작할까?
2. **Reward weights** 초기값 검토
3. **Curriculum** 필요한가?
4. **Heuristics** 어떤 패턴 포함할까?

---

**작성자**: Claude Code
**검토 필요**: 사용자 검토 후 구현 시작
