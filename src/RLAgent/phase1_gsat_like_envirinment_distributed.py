"""
CXL Memory RL Environment (Distributed)
Phase 1: StressAppTest-like Action Space
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import time
import json

class CXLMemoryEnv(gym.Env):
    """
    Custom Environment for CXL Memory Testing
    Interacts with Remote Memory Agent (GNR-SP Linux)
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, agent_url="http://192.168.3.20:5000", max_steps=100):
        super(CXLMemoryEnv, self).__init__()
        
        self.agent_url = agent_url
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action Space: 4 Operations * 256 Patterns = 1024 Discrete Actions
        # Op 0: FILL   (Solid)
        # Op 1: INVERT (Toggle - High Noise)
        # Op 2: COPY   (Bandwidth)
        # Op 3: CHECK  (Read)
        self.action_space = spaces.Discrete(1024)
        
        # Observation Space: 
        # [0] CE Volatile Count (Delta)
        # [1] CE Persistent Count (Delta)
        # [2] Device Temperature
        # [3] Steps Done
        # [4] Unique Actions Tried
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(5,), 
            dtype=np.float32
        )
        
        self.used_actions = set()
        self.session = requests.Session()
        self.session.trust_env = False  # Proxy bypass
        
        # Initial Connection Check
        self._check_connection()

    def _check_connection(self):
        try:
            print(f"Connecting to Memory Agent at {self.agent_url}...")
            resp = self.session.get(f"{self.agent_url}/health", timeout=5)
            if resp.status_code == 200:
                print(f"Connected: {resp.json()}")
            else:
                print(f"Warning: Connection status {resp.status_code}")
        except Exception as e:
            print(f"Error: Could not connect to Memory Agent. {e}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.used_actions.clear()
        
        # Reset hardware baseline
        try:
            self.session.post(f"{self.agent_url}/reset_baseline", timeout=10)
            ce_info = self._get_ce_info()
            info = {}
            return self._get_obs(ce_info), info
        except Exception as e:
            print(f"Reset failed: {e}")
            # Return zero state on failure to prevent crash
            return np.zeros(5, dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1
        terminated = False
        truncated = False
        
        # Execute Action on Remote Hardware
        try:
            # Note: C agent runs for 5 seconds (STRESS_DURATION_SEC)
            # Timeout must be significantly larger (e.g., 30s)
            response = self.session.post(
                f"{self.agent_url}/execute_action",
                json={'action': int(action)},
                timeout=30 
            )
            result = response.json()
            
            ce_delta = result.get('ce_volatile', 0)
            temp = result.get('temperature', 0)
            
        except Exception as e:
            print(f"Action Execution Failed: {e}")
            ce_delta = 0
            temp = 0
            result = {}

        # Update State
        self.used_actions.add(action)
        
        # --- Reward Engineering (Phase 1) ---
        reward = 0.0
        
        # 1. Primary Goal: Detecting CE
        if ce_delta > 0:
            reward += 100.0 * ce_delta
            print(f"\n[!!!] CRITICAL: CE DETECTED! Delta={ce_delta}, Action={action}\n")
            
        # 2. Exploration Bonus (Try new patterns)
        if action not in self.used_actions:
            reward += 2.0
            
        # 3. Logic Preference (Heuristic)
        # Prefer OP_INVERT (Op=1) because it generates the most noise
        op_code = action // 256
        if op_code == 1: 
            reward += 0.5
            
        # 4. Step Cost (Time Penalty)
        reward -= 0.1

        # Check Termination
        if self.current_step >= self.max_steps:
            truncated = True
            
        obs = np.array([
            float(result.get('ce_volatile', 0)),
            float(result.get('ce_persistent', 0)),
            float(temp),
            float(self.current_step),
            float(len(self.used_actions))
        ], dtype=np.float32)

        return obs, reward, terminated, truncated, result

    def _get_obs(self, ce_info):
        return np.array([
            float(ce_info.get('volatile_count', 0)),
            float(ce_info.get('persistent_count', 0)),
            float(ce_info.get('temperature', 0)),
            float(self.current_step),
            float(len(self.used_actions))
        ], dtype=np.float32)

    def _get_ce_info(self):
        try:
            resp = self.session.get(f"{self.agent_url}/get_ce_info", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return {}

    def render(self):
        print(f"Step: {self.current_step}/{self.max_steps}, Actions: {len(self.used_actions)}")
