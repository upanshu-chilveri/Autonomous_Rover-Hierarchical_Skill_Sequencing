# Deliverable 1: One-Page Problem Formulation

## Project Title: Autonomous Base Logistics & Security Rover: Hierarchical Skill Sequencing

### 1. Mission Context and Environment Specification
Autonomous Uncrewed Ground Vehicles (UGVs) deployed at forward military bases and secure compounds must execute compound logistics, hazardous reconnaissance, and emergency terminal access without constant human teleoperation. 

The environment is modeled as a partitioned compound (MiniGrid abstraction) consisting of:
* Sector 1 (Staging Zone): Initial rover deployment location containing an encrypted security clearance keycard.
* Security Checkpoint: A reinforced blast gate dividing Sector 1 from Sector 2. The gate remains locked until the rover physically possesses and toggles the clearance keycard.
* Sector 2 (Command Bunker): Secure inner perimeter containing the mission-critical mainframe terminal.

### 2. State Representations

#### Primitive Observation Space (Low-Level MDP)
At each discrete time step $t$, the low-level observation vector $s_t \in \mathcal{S}$ is:
$$s_t = (x_t, y_t, \theta_t, c_t, g_{open})$$
Where:
* $(x_t, y_t) \in \{0, \dots, W-1\} \times \{0, \dots, H-1\}$: Grid coordinates of the rover.
* $\theta_t \in \{0, 1, 2, 3\}$: Heading direction of the rover (East, South, West, North).
* $c_t \in \{0, 1\}$: Keycard possession flag (1 if carrying clearance token, 0 otherwise).
* $g_{open} \in \{0, 1\}$: Gate status (1 if blast gate is unlocked and opened, 0 if closed/locked).

#### Abstract State Space (High-Level SMDP)
To avoid the curse of dimensionality and temporal horizon bottlenecks, the Manager operates over a compressed 3-bit tactical state $S \in \{0, \dots, 7\}$:
$$S = (c_t \ll 2) \mid (g_{open} \ll 1) \mid r_{sec2}$$
Where $r_{sec2} = 1$ if $x_t > x_{gate}$ (rover has entered Sector 2), and 0 otherwise.

### 3. Action Spaces and Temporal Abstraction

#### Primitive Actions $\mathcal{A}$
The rover chassis supports 7 low-level discrete commands:
$$\mathcal{A} = \{\text{Turn Left}, \text{Turn Right}, \text{Forward}, \text{Pick Up}, \text{Drop}, \text{Toggle / Unlock}, \text{Standby}\}$$

#### High-Level Macro-Actions (Options $\Omega$)
The High-Level Manager selects from three reusable, certified tactical skills $\omega \in \Omega$:
1. $\omega_0$ (Retrieve Keycard): Navigates to the keycard coordinate and executes pickup.
   * Initiation set $\mathcal{I}_0$: Valid when keycard is uncollected ($c_t = 0$).
   * Termination condition $\beta_0(s)$: Terminates when keycard is acquired ($c_t = 1$) or timeout $\tau_{max}$.
2. $\omega_1$ (Unlock Blast Gate): Navigates to the checkpoint blast gate and unlocks it.
   * Initiation set $\mathcal{I}_1$: Valid when carrying keycard ($c_t = 1$) and gate is closed ($g_{open} = 0$).
   * Termination condition $\beta_1(s)$: Terminates when blast gate is open ($g_{open} = 1$) or timeout $\tau_{max}$.
3. $\omega_2$ (Access Terminal): Traverses through the opened checkpoint to the command terminal.
   * Initiation set $\mathcal{I}_2$: Valid when gate is open ($g_{open} = 1$).
   * Termination condition $\beta_2(s)$: Terminates upon reaching the terminal or timeout $\tau_{max}$.

### 4. Reward Formulation and SMDP Bellman Update

#### Low-Level Intrinsic Rewards
Skills receive sub-goal completion bonuses ($+0.1$ for keycard acquisition, $+0.2$ for gate breach) and small step penalties ($-0.01$) to encourage time-efficient traversal.

#### High-Level Extrinsic Reward
The primary mission objective yields a terminal success reward upon accessing the command terminal:
$$R_{terminal} = 1.0 - 0.9 \cdot \left(\frac{\text{Total Steps}}{\text{Max Allowed Steps}}\right)$$
Invalid option invocations incur an immediate penalty of $-0.05$.

#### SMDP Value Update
Let an option $\omega$ execute for $\tau$ primitive steps. The Manager updates option-values via the Semi-Markov Decision Process Bellman equation:
$$Q(S_t, \omega_t) \leftarrow Q(S_t, \omega_t) + \alpha \left[ R_{cumulative} + \gamma^\tau \max_{\omega' \in \Omega} Q(S_{t+\tau}, \omega') - Q(S_t, \omega_t) \right]$$
Where $\gamma^\tau$ applies exponential temporal discounting based on the actual duration $\tau$ spent executing skill $\omega_t$.

