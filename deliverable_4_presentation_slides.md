# Deliverable 4: 5-Slide Presentation Deck

## Title: Autonomous Base Logistics & Security Rover: Hierarchical Skill Sequencing
**Course:** Unit 5: Real-World Reinforcement Learning  
**Project Topic 15:** Game-Level Skill Hierarchy  

---

### Slide 1: Motivation and Problem Statement
* **Operational Rationale:**
  * Military base logistics, toxic hazard inspection, and forward checkpoint maintenance require autonomous ground vehicles that operate reliably without continuous radio link teleoperation.
  * Multi-stage compound traversal requires executing strict causal protocols: obtaining security clearance tokens, unlocking checkpoint blast gates, and accessing central control mainframes.
* **The Sparse-Reward Bottleneck in Flat RL:**
  * Flat reinforcement learning algorithms (standard PPO, DQN, Q-learning) operate over primitive motor ticks (turn, forward, toggle).
  * In a multi-stage compound, the chance of a random agent executing the exact sequence of picking up a keycard, facing a locked gate, toggling it, and driving to the goal terminal within hundreds of steps is vanishingly small ($\approx 10^{-6}$).
  * Flat agents suffer catastrophic sample inefficiency and fail completely without dense reward engineering.

---

### Slide 2: Hierarchical RL & The Options Framework
* **Theoretical Framework: Semi-Markov Decision Process (SMDP):**
  * We decompose the mission into a two-level hierarchy using Sutton, Precup, and Singh's Options framework.
  * **Low-Level Tactical Skills (Options $\omega \in \Omega$):** Certified, reusable subroutines running for variable duration $\tau$.
  * **High-Level Mission Manager:** A meta-controller selecting which skill to deploy based on compound status.
* **Temporal Discounting in SMDPs:**
  * Unlike standard MDPs where transitions take 1 step, an option executes over $\tau$ time steps.
  * Value update rule:
    $$Q(S_t, \omega_t) \leftarrow Q(S_t, \omega_t) + \alpha \left[ R_{cumulative} + \gamma^\tau \max_{\omega'} Q(S_{t+\tau}, \omega') - Q(S_t, \omega_t) \right]$$
  * Multi-step discount factor $\gamma^\tau$ accurately preserves the temporal cost of longer physical operations.

---

### Slide 3: Modular System Architecture
* **Modular Codebase Components:**
  * `MilitaryBaseRoverEnv`: Grid environment abstraction simulating secure sectors, blast gates, and mission terminals.
  * `SkillLibrary`: Standalone reusable motor controllers:
    1. *Skill 0 (Retrieve Keycard):* Local pathfinding and physical pickup.
    2. *Skill 1 (Unlock Blast Gate):* Approach locked checkpoint and engage token reader.
    3. *Skill 2 (Access Terminal):* Sector 2 ingress and terminal communication.
  * `SMDPManagerAgent`: Compact 3-bit abstract state space representing keycard possession, gate status, and current sector.
* **Extensibility and Modularity:**
  * New skills (such as fire suppression or perimeter patrol) can be plugged directly into the skill library without altering the manager's core learning logic.

---

### Slide 4: Experimental Results & Baseline Comparison
* **Quantitative Evaluation (200 Episodes):**
  * **Hierarchical SMDP Manager (Proposed):** Reaches 100.0% mission success rate within 40 episodes. Learns the correct causal sequence: Retrieve Keycard -> Unlock Blast Gate -> Access Terminal.
  * **Random Manager Baseline:** Achieves only 50.0% success rate due to frequent invalid skill invocations (attempting terminal access while the gate remains closed).
  * **Flat Primitive Q-Learning Baseline:** 0.0% mission success rate over 200 episodes. Incapable of solving the compound due to sparse reward horizon.
* **Key Finding:**
  * Temporal abstraction reduces policy search complexity by orders of magnitude, turning intractable sparse exploration into a deterministic high-level sequence.

---

### Slide 5: Limitations, Safety & Military Robotics Ethics
* **Technical Limitations:**
  * Hardcoded low-level skill termination conditions: In unstructured battlefield terrain, skill boundaries may become noisy.
  * Fixed sub-goal representations require prior knowledge of compound geometry or local SLAM waypoint mapping.
* **Safety Verification in Defense Deployments:**
  * Modular skill verification: Low-level controllers can be formally certified (formal methods / control barriers) for collision prevention before field deployment.
* **Ethics and Autonomous Systems Governance:**
  * Non-lethal scope: The rover is designed strictly for logistical transit, perimeter inspection, and secured terminal access.
  * Human-in-the-loop doctrine: High-level manager action overrides must remain available to human base operators at all times to prevent rogue autonomy during communication blackouts.

