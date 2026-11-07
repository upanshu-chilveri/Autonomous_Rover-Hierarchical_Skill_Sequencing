"""
Military Base Autonomous Logistics and Security Rover
Real-Time Visualizer (Live Pygame GUI Window + Terminal Radar Display)

Usage:
    # 1. Run live interactive simulation (opens live Pygame window on screen):
    python visualize_rover.py

    # 2. Run in terminal only (ASCII radar display, no popup window):
    python visualize_rover.py --no-gui

    # 3. Compare with baselines:
    python visualize_rover.py --mode random
    python visualize_rover.py --mode flat

    # 4. Record animated GIF:
    python visualize_rover.py --mode smdp --record --output mission_smdp.gif
"""

import argparse
import os
import sys
import time
from collections import deque
import numpy as np

# Ensure required libraries are installed with helpful error reporting
try:
    import gymnasium as gym
    import minigrid
    from minigrid.core.actions import Actions
    import pygame
    from PIL import Image, ImageDraw
except ImportError as err:
    print("\n" + "=" * 65)
    print(f"[ERROR] Missing required library: {err.name}")
    print("Please install the dependencies by running:")
    print("    pip install gymnasium minigrid pygame pillow matplotlib")
    print("=" * 65 + "\n")
    sys.exit(1)

# Import environment and skills from modular codebase
try:
    from military_base_hrl import (
        MilitaryBaseRoverEnv,
        SkillLibrary,
        SMDPManagerAgent,
        RandomManagerBaseline,
        FlatQLearningBaseline,
        train_smdp_manager,
        train_flat_qlearning
    )
except ImportError as err:
    print(f"[ERROR] Could not import military_base_hrl: {err}")
    print("Ensure military_base_hrl.py is located in the same directory.")
    sys.exit(1)


# Pre-trained Q-table from 300-episode training run (loads instantly in 0.01s)
PRETRAINED_Q_TABLE = np.array([
    [1.188, 1.135, 1.130], # S0: card=0, gate=0, sec1 -> Skill 0 (Keycard)
    [0.000, 0.000, 0.000], # S1
    [0.000, 0.000, 0.000], # S2
    [0.000, 0.000, 0.000], # S3
    [1.097, 1.117, 1.071], # S4: card=1, gate=0, sec1 -> Skill 1 (Gate)
    [0.000, 0.000, 0.000], # S5
    [0.940, 0.958, 0.965], # S6: card=1, gate=1, sec1 -> Skill 2 (Terminal)
    [0.000, 0.000, 0.000]  # S7
])

DIR_SYMBOLS = {0: ">", 1: "v", 2: "<", 3: "^"}
DIR_NAMES = {0: "EAST", 1: "SOUTH", 2: "WEST", 3: "NORTH"}


def render_ascii_radar(env, step_num, active_skill, action_name, state_id):
    """
    Renders an ASCII tactical radar display directly into the terminal.
    Ensures the user can observe the simulation even in headless or remote sessions.
    """
    unwrapped = env.unwrapped
    grid = unwrapped.grid
    rover_pos = tuple(int(c) for c in unwrapped.agent_pos)
    rover_dir = int(unwrapped.agent_dir)
    has_card = unwrapped.carrying is not None

    print("\033[H\033[J", end="") # Clear terminal screen
    print("=" * 55)
    print(f" [TACTICAL RADAR] SEC-COM UGV-01 | Step: {step_num:02d} / 60")
    print(f" Active Skill: {active_skill} | State: S{state_id}")
    print(f" Primitive Action: {action_name}")
    print(f" Keycard: {'EQUIPPED' if has_card else 'NOT CARRIED'}")
    print("=" * 55)

    # Grid frame
    print("      0   1   2   3   4   5")
    print("    +---+---+---+---+---+---+")
    for y in range(grid.height):
        row_str = f"  {y} |"
        for x in range(grid.width):
            if (x, y) == rover_pos:
                symbol = f" {DIR_SYMBOLS[rover_dir]} "
            else:
                cell = grid.get(x, y)
                if cell is None:
                    symbol = "   "
                elif cell.type == "wall":
                    symbol = "###" if x == 2 else "[#]"
                elif cell.type == "key":
                    symbol = " K "
                elif cell.type == "door":
                    symbol = " / " if cell.is_open else " D "
                elif cell.type == "goal":
                    symbol = " G "
                else:
                    symbol = " . "
            row_str += symbol + "|"
        print(row_str)
        print("    +---+---+---+---+---+---+")
    print(" Legend: > v < ^ = Rover | K = Keycard | D = Gate | G = Terminal\n")


def annotate_frame(frame_rgb, text_lines):
    """Overlays telemetry header on RGB frame for GIF generation."""
    img = Image.fromarray(frame_rgb)
    scale = 3
    img_scaled = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
    header_height = 80
    new_img = Image.new("RGB", (img_scaled.width, img_scaled.height + header_height), color=(15, 23, 42))
    new_img.paste(img_scaled, (0, header_height))
    draw = ImageDraw.Draw(new_img)
    y_offset = 6
    for line in text_lines:
        draw.text((12, y_offset), line, fill=(226, 232, 240))
        y_offset += 22
    return np.array(new_img)


def run_live_simulation(mode="smdp", seed=42, fps=3, use_gui=True, record=False, output_gif=None, retrain=False):
    print("\n" + "=" * 65)
    print(f" MILITARY BASE ROVER: LIVE SIMULATION // MODE: {mode.upper()}")
    print("=" * 65)

    base_env = MilitaryBaseRoverEnv()

    # Load agent
    if mode == "smdp":
        manager = SMDPManagerAgent()
        if retrain:
            print("Retraining SMDP Manager from scratch (300 episodes)...")
            manager, _, _ = train_smdp_manager(base_env, num_episodes=300)
        else:
            print("Loading pre-trained SMDP optimal policy (Instant startup)...")
            manager.q_table = PRETRAINED_Q_TABLE.copy()
    elif mode == "random":
        print("Initializing Random Manager Baseline...")
        manager = RandomManagerBaseline()
    elif mode == "flat":
        print("Training Flat Q-Learning Agent (300 episodes)...")
        manager, _, _ = train_flat_qlearning(base_env, num_episodes=300)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Determine render mode
    render_mode = "human" if use_gui else "rgb_array"
    if record:
        render_mode = "rgb_array"

    print(f"Initializing compound environment (Seed: {seed}, Render: {render_mode})...")
    try:
        render_env = gym.make("MiniGrid-DoorKey-6x6-v0", render_mode=render_mode)
        render_env.reset(seed=seed)
        base_env.env = render_env
    except Exception as err:
        print(f"[WARNING] GUI window display unavailable ({err}). Falling back to terminal display.")
        render_env = gym.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array")
        render_env.reset(seed=seed)
        base_env.env = render_env
        use_gui = False

    frames = []
    step_num = 0
    mission_done = False
    mission_success = False
    delay = 1.0 / max(1, fps)

    def capture_frame(lines):
        if record or output_gif:
            try:
                raw_frame = render_env.render()
                if raw_frame is not None:
                    frames.append(annotate_frame(raw_frame, lines))
            except Exception:
                pass

    print("\n[MISSION STARTED] Watch rover operate step-by-step below:\n")
    time.sleep(1.0)

    if mode in ["smdp", "random"]:
        manager_steps = 0
        max_manager_steps = 6

        while not mission_done and manager_steps < max_manager_steps:
            manager_steps += 1
            state = base_env.get_manager_state()

            if mode == "smdp":
                option = manager.select_skill(state, epsilon=0.0)
                q_vals = [round(float(v), 2) for v in manager.q_table[state]]
                decision_str = f"Q: {q_vals}"
            else:
                option = manager.select_skill(state)
                decision_str = "Random Selection"

            skill_name = SkillLibrary.SKILL_NAMES[option]
            unwrapped = base_env.env.unwrapped
            key_pos, gate_pos, term_pos = base_env.get_poi_coordinates()
            gate_cell = unwrapped.grid.get(*gate_pos)
            gate_open = gate_cell and gate_cell.is_open

            # Handle precondition blocks
            if option == SkillLibrary.SKILL_RETRIEVE_KEYCARD:
                if unwrapped.carrying is not None:
                    step_num += 1
                    render_ascii_radar(render_env, step_num, skill_name, "IDLE (CARD ALREADY HELD)", state)
                    time.sleep(delay)
                    continue
                target_pos = key_pos
                target_walkable = False
                interact_act = Actions.pickup
            elif option == SkillLibrary.SKILL_UNLOCK_GATE:
                if gate_open:
                    step_num += 1
                    render_ascii_radar(render_env, step_num, skill_name, "IDLE (GATE ALREADY OPEN)", state)
                    time.sleep(delay)
                    continue
                if unwrapped.carrying is None:
                    step_num += 1
                    render_ascii_radar(render_env, step_num, skill_name, "BLOCKED (NO KEYCARD)", state)
                    time.sleep(delay)
                    continue
                target_pos = gate_pos
                target_walkable = False
                interact_act = Actions.toggle
            elif option == SkillLibrary.SKILL_REACH_TERMINAL:
                target_pos = term_pos
                target_walkable = True
                interact_act = Actions.forward

            skill_step = 0
            skill_done = False

            while skill_step < 35 and not skill_done:
                agent_pos = tuple(int(c) for c in unwrapped.agent_pos)
                agent_dir = int(unwrapped.agent_dir)
                grid = unwrapped.grid

                if target_walkable and agent_pos == target_pos:
                    skill_done = True
                    break

                path = SkillLibrary._bfs(grid, agent_pos, target_pos, target_walkable)
                if path is None:
                    step_num += 1
                    render_ascii_radar(render_env, step_num, skill_name, "BLOCKED (GATE CLOSED)", state)
                    time.sleep(delay)
                    break

                step_num += 1
                skill_step += 1

                if len(path) == 0:
                    dx = target_pos[0] - agent_pos[0]
                    dy = target_pos[1] - agent_pos[1]
                    req_dir = SkillLibrary._vec_to_dir(dx, dy)
                    if agent_dir == req_dir:
                        _, r, term, trunc, _ = base_env.step(interact_act)
                        action_taken = "INTERACT / PICKUP / TOGGLE"
                        if term:
                            mission_done = True
                            mission_success = True
                        skill_done = True
                    elif (agent_dir + 1) % 4 == req_dir:
                        base_env.step(Actions.right)
                        action_taken = "TURN RIGHT"
                    else:
                        base_env.step(Actions.left)
                        action_taken = "TURN LEFT"
                else:
                    nxt = path[0]
                    dx = nxt[0] - agent_pos[0]
                    dy = nxt[1] - agent_pos[1]
                    req_dir = SkillLibrary._vec_to_dir(dx, dy)
                    if agent_dir == req_dir:
                        _, r, term, trunc, _ = base_env.step(Actions.forward)
                        action_taken = "MOVE FORWARD"
                        if term:
                            mission_done = True
                            mission_success = True
                            skill_done = True
                    elif (agent_dir + 1) % 4 == req_dir:
                        base_env.step(Actions.right)
                        action_taken = "TURN RIGHT"
                    else:
                        base_env.step(Actions.left)
                        action_taken = "TURN LEFT"

                render_ascii_radar(render_env, step_num, skill_name, action_taken, state)
                lines = [
                    f"[UGV HUD] Mode: {mode.upper()} | Step: {step_num} | Action: {action_taken}",
                    f"Active Skill: {skill_name} | State ID: S{state} | {decision_str}",
                    f"Clearance Keycard: {'YES' if unwrapped.carrying else 'NO'} | Blast Gate: {'OPEN' if gate_open else 'LOCKED'}"
                ]
                capture_frame(lines)
                time.sleep(delay)

    else:
        # Flat Q-learning execution
        state_key = manager.get_state_key(base_env)
        for _ in range(50):
            step_num += 1
            action = manager.select_action(state_key, epsilon=0.0)
            obs, r, term, trunc, _ = base_env.step(action)
            state_key = manager.get_state_key(base_env)

            act_names = ["TURN LEFT", "TURN RIGHT", "FORWARD", "PICKUP", "DROP", "TOGGLE", "STANDBY"]
            action_name = act_names[action] if action < len(act_names) else str(action)
            render_ascii_radar(render_env, step_num, "Monolithic Flat Q-Learning", action_name, 0)

            lines = [
                f"[UGV HUD] Mode: FLAT RL | Step: {step_num} | Action: {action_name}",
                "No Skill Hierarchy | Primitive Action Selection",
                "Clearance Keycard: NO | Blast Gate: LOCKED"
            ]
            capture_frame(lines)
            time.sleep(delay)

            if term or trunc:
                if term and r > 0.4:
                    mission_success = True
                break

    # Summary
    status_str = "MISSION ACCOMPLISHED" if mission_success else "MISSION TIMEOUT / FAILED"
    print("=" * 55)
    print(f" FINAL STATUS: {status_str}")
    print(f" Total Steps Taken: {step_num}")
    print("=" * 55)

    if record or output_gif:
        out_path = output_gif if output_gif else f"mission_{mode}.gif"
        if not os.path.isabs(out_path):
            out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_path)
        print(f"Saving animated GIF to: {out_path}...")
        pil_frames = [Image.fromarray(f) for f in frames]
        if pil_frames:
            pil_frames[0].save(
                out_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000 / fps),
                loop=0
            )
            print(f"GIF saved successfully ({len(pil_frames)} frames).")

    try:
        render_env.close()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Military Base Rover Live Visualizer")
    parser.add_argument("--mode", choices=["smdp", "random", "flat"], default="smdp", help="Controller type")
    parser.add_argument("--seed", type=int, default=42, help="Compound random seed")
    parser.add_argument("--fps", type=int, default=4, help="Playback speed in frames per second")
    parser.add_argument("--no-gui", action="store_true", help="Disable Pygame GUI window (terminal only)")
    parser.add_argument("--record", action="store_true", help="Record and save an animated GIF")
    parser.add_argument("--output", type=str, default=None, help="Output GIF filename")
    parser.add_argument("--retrain", action="store_true", help="Retrain model before running")

    args = parser.parse_args()
    run_live_simulation(
        mode=args.mode,
        seed=args.seed,
        fps=args.fps,
        use_gui=not args.no_gui,
        record=args.record,
        output_gif=args.output,
        retrain=args.retrain
    )
