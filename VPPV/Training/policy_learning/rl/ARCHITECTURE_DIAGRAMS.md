# Architecture Diagrams: Double DDPG and Dual DDPG

## Overview

Both Double DDPG and Dual DDPG extend the standard DDPG algorithm by using two Q-networks (critics) instead of one. The key difference is how they combine the Q-values:
- **Double DDPG**: Uses **minimum** of Q1 and Q2 (reduces overestimation bias)
- **Dual DDPG**: Uses **average** of Q1 and Q2 (ensemble averaging)

---

## Network Architecture

### Actor Network (Same for Both Algorithms)

```
Input: [Observation (o) + Goal (g)]
  ↓
DeterministicActor
  ↓
MLP (4 layers):
  - Linear(o+g → 256) + ReLU
  - Linear(256 → 256) + ReLU
  - Linear(256 → 256) + ReLU
  - Linear(256 → action_dim)
  ↓
tanh(·) × max_action
  ↓
Output: Action (a) ∈ [-max_action, max_action]
```

### Critic Network Architecture

#### Standard DDPG (Single Critic)
```
Input: [State (s) + Action (a)]
  ↓
Critic
  ↓
MLP (4 layers):
  - Linear(s+a → 256) + ReLU
  - Linear(256 → 256) + ReLU
  - Linear(256 → 256) + ReLU
  - Linear(256 → 1)
  ↓
Output: Q(s, a) (single scalar)
```

#### Double/Dual DDPG (Double Critic)
```
Input: [State (s) + Action (a)]
  ↓
DoubleCritic
  ├─→ Q1 Network (MLP)
  │     - Linear(s+a → 256) + ReLU
  │     - Linear(256 → 256) + ReLU
  │     - Linear(256 → 256) + ReLU
  │     - Linear(256 → 1)
  │     └─→ Q1(s, a)
  │
  └─→ Q2 Network (MLP)
        - Linear(s+a → 256) + ReLU
        - Linear(256 → 256) + ReLU
        - Linear(256 → 256) + ReLU
        - Linear(256 → 1)
        └─→ Q2(s, a)
```

---

## Algorithm Flow Diagrams

### Double DDPG Training Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    DOUBLE DDPG TRAINING                     │
└─────────────────────────────────────────────────────────────┘

1. SAMPLE TRANSITION FROM REPLAY BUFFER
   (s, a, r, s', done)
   
2. COMPUTE TARGET Q-VALUE
   ┌─────────────────────────────────────┐
   │ a' = Actor_target(s')                │
   │ Q1_target = Critic1_target(s', a')   │
   │ Q2_target = Critic2_target(s', a')   │
   │ target_V = min(Q1_target, Q2_target) │ ← KEY: MINIMUM
   │ target_Q = r + γ × target_V          │
   └─────────────────────────────────────┘

3. UPDATE CRITIC NETWORKS
   ┌─────────────────────────────────────┐
   │ Q1_pred = Critic1(s, a)             │
   │ Q2_pred = Critic2(s, a)             │
   │ loss1 = MSE(Q1_pred, target_Q)      │
   │ loss2 = MSE(Q2_pred, target_Q)      │
   │ total_loss = loss1 + loss2          │
   │ Update Critic1 and Critic2          │
   └─────────────────────────────────────┘

4. UPDATE ACTOR NETWORK
   ┌─────────────────────────────────────┐
   │ a_new = Actor(s)                   │
   │ Q1 = Critic1(s, a_new)             │
   │ Q2 = Critic2(s, a_new)             │
   │ Q_min = min(Q1, Q2)                │ ← KEY: MINIMUM
   │ actor_loss = -mean(Q_min)          │
   │ Update Actor                       │
   └─────────────────────────────────────┘

5. SOFT UPDATE TARGET NETWORKS
   θ_target ← τ·θ + (1-τ)·θ_target
   (for both Actor and Critic)
```

### Dual DDPG Training Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     DUAL DDPG TRAINING                       │
└─────────────────────────────────────────────────────────────┘

1. SAMPLE TRANSITION FROM REPLAY BUFFER
   (s, a, r, s', done)
   
2. COMPUTE TARGET Q-VALUE
   ┌─────────────────────────────────────┐
   │ a' = Actor_target(s')                │
   │ Q1_target = Critic1_target(s', a')  │
   │ Q2_target = Critic2_target(s', a')   │
   │ target_V = (Q1_target + Q2_target)/2 │ ← KEY: AVERAGE
   │ target_Q = r + γ × target_V          │
   └─────────────────────────────────────┘

3. UPDATE CRITIC NETWORKS
   ┌─────────────────────────────────────┐
   │ Q1_pred = Critic1(s, a)             │
   │ Q2_pred = Critic2(s, a)             │
   │ loss1 = MSE(Q1_pred, target_Q)      │
   │ loss2 = MSE(Q2_pred, target_Q)      │
   │ total_loss = loss1 + loss2          │
   │ Update Critic1 and Critic2          │
   └─────────────────────────────────────┘

4. UPDATE ACTOR NETWORK
   ┌─────────────────────────────────────┐
   │ a_new = Actor(s)                    │
   │ Q1 = Critic1(s, a_new)              │
   │ Q2 = Critic2(s, a_new)              │
   │ Q_avg = (Q1 + Q2) / 2               │ ← KEY: AVERAGE
   │ actor_loss = -mean(Q_avg)           │
   │ Update Actor                        │
   └─────────────────────────────────────┘

5. SOFT UPDATE TARGET NETWORKS
   θ_target ← τ·θ + (1-τ)·θ_target
   (for both Actor and Critic)
```

---

## Complete System Architecture

### Double DDPG System

```
┌──────────────────────────────────────────────────────────────┐
│                      ENVIRONMENT                              │
│  Observation (o) + Goal (g) → State (s)                      │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────┐
        │      ACTOR NETWORK            │
        │  Input: s = [o, g]            │
        │  Output: Action (a)           │
        └──────────────┬────────────────┘
                       │
                       ↓
        ┌──────────────────────────────┐
        │    ENVIRONMENT STEP           │
        │  (s, a) → (s', r, done)      │
        └──────────────┬────────────────┘
                       │
                       ↓
        ┌──────────────────────────────┐
        │   REPLAY BUFFER              │
        │  Store: (s, a, r, s', done)  │
        └──────────────┬────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────────┐
        │         TRAINING UPDATE                       │
        │                                                │
        │  ┌──────────────────────────────────────┐    │
        │  │   DOUBLE CRITIC NETWORK               │    │
        │  │   ┌──────────┐      ┌──────────┐     │    │
        │  │   │  Q1(s,a) │      │  Q2(s,a) │     │    │
        │  │   └────┬─────┘      └────┬─────┘     │    │
        │  │        │                  │           │    │
        │  │        └──────┬───────────┘           │    │
        │  │               ↓                       │    │
        │  │        min(Q1, Q2)                    │    │
        │  └──────────────────────────────────────┘    │
        │                                                │
        │  ┌──────────────────────────────────────┐    │
        │  │   TARGET COMPUTATION                  │    │
        │  │   a' = Actor_target(s')               │    │
        │  │   Q1' = Critic1_target(s', a')       │    │
        │  │   Q2' = Critic2_target(s', a')       │    │
        │  │   target = r + γ·min(Q1', Q2')       │    │
        │  └──────────────────────────────────────┘    │
        │                                                │
        │  ┌──────────────────────────────────────┐    │
        │  │   LOSS COMPUTATION                    │    │
        │  │   loss_critic = MSE(Q1, target) +      │    │
        │  │                  MSE(Q2, target)      │    │
        │  │   loss_actor = -mean(min(Q1, Q2))     │    │
        │  └──────────────────────────────────────┘    │
        └────────────────────────────────────────────────┘
```

### Dual DDPG System

```
┌──────────────────────────────────────────────────────────────┐
│                      ENVIRONMENT                              │
│  Observation (o) + Goal (g) → State (s)                      │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────┐
        │      ACTOR NETWORK            │
        │  Input: s = [o, g]            │
        │  Output: Action (a)           │
        └──────────────┬────────────────┘
                       │
                       ↓
        ┌──────────────────────────────┐
        │    ENVIRONMENT STEP           │
        │  (s, a) → (s', r, done)      │
        └──────────────┬────────────────┘
                       │
                       ↓
        ┌──────────────────────────────┐
        │   REPLAY BUFFER              │
        │  Store: (s, a, r, s', done)  │
        └──────────────┬────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────────┐
        │         TRAINING UPDATE                       │
        │                                                │
        │  ┌──────────────────────────────────────┐    │
        │  │   DOUBLE CRITIC NETWORK               │    │
        │  │   ┌──────────┐      ┌──────────┐     │    │
        │  │   │  Q1(s,a) │      │  Q2(s,a) │     │    │
        │  │   └────┬─────┘      └────┬─────┘     │    │
        │  │        │                  │           │    │
        │  │        └──────┬───────────┘           │    │
        │  │               ↓                       │    │
        │  │        (Q1 + Q2) / 2                  │    │
        │  └──────────────────────────────────────┘    │
        │                                                │
        │  ┌──────────────────────────────────────┐    │
        │  │   TARGET COMPUTATION                  │    │
        │  │   a' = Actor_target(s')               │    │
        │  │   Q1' = Critic1_target(s', a')        │    │
        │  │   Q2' = Critic2_target(s', a')        │    │
        │  │   target = r + γ·(Q1' + Q2')/2       │    │
        │  └──────────────────────────────────────┘    │
        │                                                │
        │  ┌──────────────────────────────────────┐    │
        │  │   LOSS COMPUTATION                    │    │
        │  │   loss_critic = MSE(Q1, target) +     │    │
        │  │                  MSE(Q2, target)      │    │
        │  │   loss_actor = -mean((Q1+Q2)/2)      │    │
        │  └──────────────────────────────────────┘    │
        └────────────────────────────────────────────────┘
```

---

## Key Differences Summary

| Aspect | DDPG | Double DDPG | Dual DDPG |
|--------|------|-------------|-----------|
| **Number of Critics** | 1 | 2 | 2 |
| **Target Q-value** | Q_target(s', a') | min(Q1_target, Q2_target) | (Q1_target + Q2_target) / 2 |
| **Actor Update** | Q(s, a_new) | min(Q1, Q2) | (Q1 + Q2) / 2 |
| **Purpose** | Baseline | Reduce overestimation | Ensemble averaging |
| **Bias** | Overestimation bias | Reduced bias | Balanced estimate |

---

## Network Dimensions (SurRoL Context)

- **Observation (o)**: Variable (depends on task)
- **Goal (g)**: Variable (depends on task)
- **State (s)**: `[o, g]` concatenated
- **Action (a)**: Variable (depends on task, typically 4-7 DOF)
- **Hidden Dimension**: 256 (default)
- **MLP Layers**: 4 layers (3 hidden + 1 output)
- **Activation**: ReLU (hidden layers), tanh (actor output)

---

## Code Structure

```
agents/
├── ddpg.py          # Standard DDPG (single Critic)
├── double_ddpg.py  # Double DDPG (min(Q1, Q2))
└── dual_ddpg.py     # Dual DDPG (avg(Q1, Q2))

modules/
├── critics.py       # Critic, DoubleCritic
├── policies.py      # DeterministicActor
└── subnetworks.py   # MLP (4-layer network)
```

---

## Mathematical Formulation

### Double DDPG

**Target Q-value:**
```
target_Q = r + γ · min(Q1_target(s', a'), Q2_target(s', a'))
```

**Critic Loss:**
```
L_critic = MSE(Q1(s, a), target_Q) + MSE(Q2(s, a), target_Q)
```

**Actor Loss:**
```
L_actor = -E[min(Q1(s, π(s)), Q2(s, π(s)))]
```

### Dual DDPG

**Target Q-value:**
```
target_Q = r + γ · (Q1_target(s', a') + Q2_target(s', a')) / 2
```

**Critic Loss:**
```
L_critic = MSE(Q1(s, a), target_Q) + MSE(Q2(s, a), target_Q)
```

**Actor Loss:**
```
L_actor = -E[(Q1(s, π(s)) + Q2(s, π(s))) / 2]
```

---

## Implementation Details

1. **DoubleCritic Class**: Contains two independent MLP networks (Q1 and Q2)
2. **Target Networks**: Both algorithms maintain target copies of Actor and Critic networks
3. **Soft Updates**: Target networks updated using polyak averaging: `θ_target = τ·θ + (1-τ)·θ_target`
4. **Replay Buffer**: Uses Hindsight Experience Replay (HER) for goal-conditioned RL
5. **Normalization**: Observation and goal normalization for stable training



