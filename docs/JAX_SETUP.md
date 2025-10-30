# JAX 설치 가이드

이 문서는 CXL Memory RL 프로젝트에서 JAX를 사용하기 위한 설치 가이드입니다.

## 개요

JAX는 Google의 고성능 머신러닝 라이브러리로 다음과 같은 특징이 있습니다:
- **자동 미분**: `grad()` 함수로 자동 미분 지원
- **JIT 컴파일**: XLA를 통한 최적화
- **GPU/TPU 가속**: 투명한 가속 지원
- **벡터화**: `vmap()`으로 배치 처리
- **병렬화**: `pmap()`으로 멀티 디바이스 지원

## 설치 방법

### 1. 기본 패키지 설치 (먼저)

```bash
cd /home/dhkang/cxl_memory_rl_project
source venv/bin/activate
pip install -r requirements.txt
```

### 2. JAX 설치 (CUDA 12)

현재 시스템은 **CUDA 12.8**을 사용하므로 CUDA 12용 JAX를 설치해야 합니다.

```bash
# CUDA 12용 JAX 설치
pip install -r requirements-jax-cuda12.txt
```

**또는 직접 설치:**

```bash
pip install --upgrade "jax[cuda12]>=0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax optax chex
```

### 3. 설치 확인

```bash
python3 -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"
```

**예상 출력:**
```
JAX version: 0.4.20
Devices: [cuda(id=0)]
```

## JAX vs PyTorch

| 특징 | PyTorch | JAX |
|------|---------|-----|
| **스타일** | Object-oriented | Functional |
| **자동 미분** | Autograd | grad/value_and_grad |
| **JIT** | torch.jit | jax.jit |
| **병렬화** | DataParallel | pmap |
| **성능** | 빠름 | **매우 빠름** |
| **사용성** | 쉬움 | 중간 (함수형 스타일) |

## Flax (JAX의 Neural Network 라이브러리)

Flax는 JAX를 위한 신경망 라이브러리로 PyTorch의 `nn.Module`과 유사합니다.

### 간단한 예제

```python
import jax
import jax.numpy as jnp
from flax import linen as nn

class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=10)(x)
        return x

# 사용
model = CNN()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 28, 28, 1)))
output = model.apply(params, jnp.ones((1, 28, 28, 1)))
```

## Optax (JAX의 Optimizer 라이브러리)

Optax는 JAX를 위한 gradient processing 및 optimization 라이브러리입니다.

### 예제

```python
import optax

# Optimizer 생성
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

# 업데이트
@jax.jit
def update(params, opt_state, grads):
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state
```

## 향후 구현 예정

다음과 같은 JAX 기반 구현이 계획되어 있습니다:

1. **JAX-based DQN**: Flax로 구현된 DQN
2. **JAX-based PPO**: Flax로 구현된 PPO
3. **성능 비교**: PyTorch vs JAX

### 예상 장점

- **훈련 속도**: XLA JIT 컴파일로 20-30% 향상
- **메모리 효율**: 더 나은 메모리 관리
- **병렬화**: 멀티 GPU 쉬운 확장

## 문제 해결

### 1. CUDA 버전 불일치

```bash
# CUDA 버전 확인
nvcc --version

# CUDA 12가 아닌 경우
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 2. JAX가 GPU를 인식하지 못함

```bash
# CUDA 경로 확인
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 재설치
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 3. 메모리 부족

```python
# JAX 메모리 프리알로케이션 비활성화
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
```

## 참고 자료

- [JAX 공식 문서](https://jax.readthedocs.io/)
- [Flax 문서](https://flax.readthedocs.io/)
- [Optax 문서](https://optax.readthedocs.io/)
- [JAX GitHub](https://github.com/google/jax)
- [JAX Tutorial](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)

## 다음 단계

JAX가 설치되면:
1. 간단한 JAX 예제로 테스트
2. Flax로 신경망 구현 연습
3. 기존 PyTorch 모델을 JAX로 포팅 시도

나중에 JAX 기반 에이전트 구현 시 이 가이드를 업데이트할 예정입니다.
