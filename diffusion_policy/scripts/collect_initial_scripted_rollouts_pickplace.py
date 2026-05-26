"""
Collect scripted rollouts for the 4-object PickPlace task (state-space / lowdim).

Variations per demo (the three preference axes):
  - order_mode: 'canonical' (milk->bread->cereal->can) or 'reversed' (can->cereal->bread->milk).
    The pipeline can pick a canonical order via env vars; per-demo we just record which.
  - n_objects: how many of the 4 objects the policy actually places, sampled from
    {1,2,3,4}. The policy stops after placing n_objects then hovers.
  - drop_mode: 'careful' (release ~3 cm above bin floor) or 'drop' (release from a
    higher, variable drop_height).

Per-episode metadata is written to HDF5 attrs and aggregated into a rollouts.npz
with the per-axis reward fields the reward-learning pipeline consumes.

Usage:
  python scripts/collect_initial_scripted_rollouts_pickplace.py \
      -o shared_data_pickplace/scripted_data -n 200
"""

import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import click
import torch
import json
import numpy as np
import random
import h5py

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.env.robomimic.robomimic_pickplace_lowdim_wrapper import RobomimicPickPlaceLowdimWrapper

import gym


OBS_KEYS = ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']

# Canonical placement order is right-column-first (x-high), then left-column
# (x-low). The robot base is at x=-0.5; reaching x-high (Bread/Can) sweeps
# the arm over x-low items, so picking the right column first means we never
# disturb in-bin items by reaching over them.
#
# Object ids: 0=Milk (x-low, y-low), 1=Bread (x-high, y-low),
#             2=Cereal (x-low, y-high), 3=Can (x-high, y-high)
OBJ_NAMES = ['Milk', 'Bread', 'Cereal', 'Can']
CANONICAL_ORDER = [1, 3, 0, 2]   # Bread → Can → Milk → Cereal
REVERSED_ORDER  = [2, 0, 3, 1]   # Cereal → Milk → Can → Bread


def build_env_meta():
    """env_meta for the PickPlace 4-object task with absolute pose control."""
    return {
        'env_name': 'PickPlace',
        'type': 1,
        'env_kwargs': {
            'has_renderer': False,
            'has_offscreen_renderer': False,
            'ignore_done': True,
            'use_object_obs': True,
            'use_camera_obs': False,
            'control_freq': 20,
            'controller_configs': {
                'type': 'OSC_POSE',
                'input_max': 1, 'input_min': -1,
                'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                'kp': 150, 'damping': 1, 'impedance_mode': 'fixed',
                'kp_limits': [0, 300], 'damping_limits': [0, 10],
                'position_limits': None, 'orientation_limits': None,
                'uncouple_pos_ori': True, 'control_delta': False,
                'interpolation': None, 'ramp_ratio': 0.2,
            },
            'robots': ['Panda'],
            'camera_depths': False, 'camera_heights': 84, 'camera_widths': 84,
            'reward_shaping': False,
            'single_object_mode': 0,
        }
    }


class PickPlaceScriptedPolicy:
    """
    State-machine scripted picker-placer with re-grasp on misgrasp.

    Phases per object: APPROACH -> DESCEND -> CLOSE -> LIFT -> CHECK_GRASP ->
    (REGRASP_OPEN if misgrasped, else TRANSIT) -> LOWER -> RELEASE -> RETREAT.

    A misgrasp is detected by the Panda gripper width after the lift settles:
    fully closed (empty) ~ 0.001 m, grasping ~ 0.01–0.04 m. We retry up to
    max_grasp_attempts times before giving up on the object.
    """

    # Per-object grasp z offset (added to body xpos[2]). Calibrated so the
    # gripper closes around solid geometry for each object's typical height.
    # Cereal is ~13 cm tall — if the grasp z drops below the top, the gripper's
    # wrist collides with the box and tips it over before closing.
    GRASP_Z_OFFSET = {
        'Milk':   0.005,
        'Bread':  0.000,
        'Cereal': 0.045,
        'Can':    0.010,
    }
    # Per-object gripper yaw (Z-axis Euler) for the final grasp orientation.
    # Most objects work with the default pi/2. Cereal is too wide along the
    # default closing direction, so we rotate the gripper 90° around z so it
    # closes along the cereal's narrow body-x axis instead.
    DEFAULT_GRIPPER_YAW = np.pi / 2
    GRIPPER_YAW = {
        'Cereal': 0.0,   # = default - pi/2; rotates closing direction 90°
    }
    # Per-object xy shift applied to the bin2 target before releasing. The
    # default robosuite quadrant for the Can is at the far +y corner of bin2,
    # which is awkwardly far from the robot base; pulling it back ~5–8 cm in
    # both axes keeps it inside the Can quadrant of bin2 but easier to reach.
    PLACEMENT_OFFSET_XY = {
        'Can': (-0.05, -0.08),
    }
    # Width (in m) below which we treat the gripper as having grasped nothing.
    # Empty closed ~ 0.001; thinnest objects compress to ~ 0.015.
    GRASP_WIDTH_THRESHOLD = 0.008
    # Object must have lifted at least this much (m) above its initial z to
    # also count as grasped — guards against false positives where the gripper
    # caught a flap of cloth/mesh without moving the body.
    GRASP_LIFT_MIN = 0.03

    def __init__(self, env, order='canonical', n_objects=4,
                 drop_modes=None, drop_heights=None,
                 careful_height=0.04, noise_level=0.0, speed_factor=1.0,
                 max_grasp_attempts=3, active_order=None, path_jitter=0.05,
                 release_xy_noise=0.0):
        self.env = env
        self.order = order
        self.n_objects = int(n_objects)
        if drop_modes is None:
            drop_modes = ['careful'] * self.n_objects
        if drop_heights is None:
            drop_heights = [careful_height] * self.n_objects
        assert len(drop_modes) == self.n_objects
        assert len(drop_heights) == self.n_objects
        self.drop_modes = list(drop_modes)
        self.drop_heights = [float(h) for h in drop_heights]
        self.careful_height = float(careful_height)
        self.noise_level = float(noise_level)
        self.speed_factor = float(speed_factor)
        self.max_grasp_attempts = int(max_grasp_attempts)
        # `active_order` is the canonical right-first order restricted to the
        # active object subset (e.g. [1, 3] for pickplace_2). Defaults to the
        # full 4-object order.
        self.active_canonical = list(active_order) if active_order else list(CANONICAL_ORDER)
        self.active_reversed = list(reversed(self.active_canonical))
        # Per-step xy jitter (m) applied to TRANSIT phase waypoints — never to
        # CLOSE/RELEASE/regrasp moments, never to the final waypoint of a
        # motion. Adds path-level diversity without breaking precise grasping.
        self.path_jitter = float(path_jitter)
        # Per-object xy offset (m) added to the release position (bin_target).
        # Spreads where placed objects land within / near their target bin,
        # which makes the *_placed_raw reward more continuous instead of
        # clustering near 0 for every successful placement.
        self.release_xy_noise = float(release_xy_noise)
        # Per-object random xy offset sampled once per object at reset time
        # (filled in by reset()).
        self._release_offsets = []
        self.rotation_transformer = RotationTransformer('euler_angles', 'axis_angle', from_convention='XYZ')
        # Default gripper orientation: pitch=pi (point down), yaw=pi/2.
        # Per-object yaw overrides applied in _plan_phase.
        self.rot_euler = np.array([0., np.pi, self.DEFAULT_GRIPPER_YAW])

        # State-machine bookkeeping
        self.action_queue = []          # deque of euler-form 7D actions to emit
        self.queue = []                 # [(obj_idx, drop_mode, drop_height), ...]
        self.q_idx = 0
        self.phase = 'INIT'
        self.grasp_attempts = 0
        self.object_initial_z = None
        self.last_euler = None
        # Per-object outcome (for metadata)
        self.grasp_results = []         # list of {'obj', 'attempts', 'grasped'}

    # ----- env access helpers -----
    def _wrapper(self):
        # MultiStepWrapper -> VideoRecordingWrapper -> RobomimicPickPlaceLowdimWrapper
        return self.env.env.env

    def _robomimic_env(self):
        return self._wrapper().env

    def _robosuite_env(self):
        return self._robomimic_env().env

    # ----- trajectory generation -----
    def _bezier(self, start, end, num=20, vmax=0.0):
        ctrl = (start + end) / 2
        offset = np.random.uniform(low=-vmax, high=vmax, size=3)
        offset[2] = 0.0
        ctrl = ctrl + offset
        ts = np.linspace(0, 1, num=num)
        return np.array([
            (1 - t) ** 2 * start + 2 * (1 - t) * t * ctrl + t ** 2 * end
            for t in ts
        ])

    def _obj_pos(self, obj_name):
        rs = self._robosuite_env()
        return np.array(rs.sim.data.body_xpos[rs.obj_body_id[obj_name]])

    def _bin_target(self, obj_idx):
        rs = self._robosuite_env()
        return np.array(rs.target_bin_placements[obj_idx])

    # ----- state-machine helpers -----
    # Hold frames AFTER each motion so the OSC controller has time to reach
    # the commanded target before the next motion fires. Kept minimal because
    # static "hover" frames create ambiguous BC labels — the policy sees many
    # obs that look the same with conflicting actions.
    HOLD_STEPS_DEFAULT = 1

    def _xy_jitter(self):
        """Random per-step xy offset for path diversity. Returns (dx, dy)."""
        if self.path_jitter <= 0:
            return 0.0, 0.0
        return (float(np.random.uniform(-self.path_jitter, self.path_jitter)),
                float(np.random.uniform(-self.path_jitter, self.path_jitter)))

    def _push_step(self, target_euler, gripper=None):
        a = target_euler.copy()
        if gripper is not None:
            a[6] = gripper
        self.action_queue.append(a.copy())
        self.last_euler = a.copy()

    def _push_interp(self, target_euler, n_steps, gripper=None,
                     hold_steps=None, jitter=False):
        target = target_euler.copy()
        if gripper is not None:
            target[6] = gripper
        n_steps = max(int(n_steps), 1)
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            a = self.last_euler + (target - self.last_euler) * alpha
            # Apply jitter only to interior waypoints — the final waypoint
            # must land on `target` exactly so the gripper grasps cleanly.
            if jitter and i < n_steps - 1:
                dx, dy = self._xy_jitter()
                a[0] += dx; a[1] += dy
            self.action_queue.append(a.copy())
        hs = self.HOLD_STEPS_DEFAULT if hold_steps is None else hold_steps
        for _ in range(hs):
            self.action_queue.append(target.copy())
        self.last_euler = target.copy()

    def _push_bezier(self, start, end, num, gripper, vmax,
                     hold_steps=None, jitter=False):
        pts = self._bezier(start, end, num=num, vmax=vmax)
        for i, p in enumerate(pts):
            a = np.concatenate([p, self.rot_euler, [gripper]])
            if jitter and i < len(pts) - 1:
                dx, dy = self._xy_jitter()
                a[0] += dx; a[1] += dy
            self.action_queue.append(a.copy())
        end_action = np.concatenate([end, self.rot_euler, [gripper]])
        hs = self.HOLD_STEPS_DEFAULT if hold_steps is None else hold_steps
        for _ in range(hs):
            self.action_queue.append(end_action.copy())
        self.last_euler = end_action.copy()

    def _gripper_width(self):
        obs = self._robomimic_env().get_observation()
        q = obs['robot0_gripper_qpos']
        return float(q[0] - q[1])

    def _check_grasped(self, obj_name):
        """True iff the gripper closed on something and the object actually rose."""
        if self.object_initial_z is None:
            return False
        cur_z = self._obj_pos(obj_name)[2]
        lifted = (cur_z - self.object_initial_z) >= self.GRASP_LIFT_MIN
        wide = self._gripper_width() >= self.GRASP_WIDTH_THRESHOLD
        return lifted and wide

    def _current_obj(self):
        if self.q_idx >= len(self.queue):
            return None
        return self.queue[self.q_idx]

    def _plan_phase(self):
        sf = self.speed_factor
        v = self.noise_level

        if self.phase == 'INIT':
            self._push_interp(self.last_euler.copy(), int(20 * sf))
            self.phase = 'APPROACH' if self.queue else 'DONE'
            return

        if self.phase == 'DONE':
            # hold at last pose
            self._push_interp(self.last_euler.copy(), int(20 * sf))
            return

        cur = self._current_obj()
        if cur is None:
            self.phase = 'DONE'
            return
        obj_idx, _dmode, dheight = cur
        obj_name = OBJ_NAMES[obj_idx]
        obj_pos = self._obj_pos(obj_name)

        grasp_pos = obj_pos.copy()
        grasp_pos[2] = obj_pos[2] + self.GRASP_Z_OFFSET.get(obj_name, 0.005)
        above_obj = grasp_pos.copy()
        above_obj[2] = obj_pos[2] + 0.20  # 20 cm clearance

        bin_target = self._bin_target(obj_idx)
        dx_obj, dy_obj = self.PLACEMENT_OFFSET_XY.get(obj_name, (0.0, 0.0))
        bin_target = bin_target.copy()
        bin_target[0] += dx_obj
        bin_target[1] += dy_obj
        above_bin = bin_target.copy()
        above_bin[2] = bin_target[2] + 0.25
        # Per-object random xy offset sampled at reset (release_xy_noise).
        if self.q_idx < len(self._release_offsets):
            rx, ry = self._release_offsets[self.q_idx]
        else:
            rx, ry = 0.0, 0.0
        release_pos = bin_target.copy()
        release_pos[0] += rx
        release_pos[1] += ry
        release_pos[2] = bin_target[2] + dheight

        if self.phase == 'APPROACH':
            # On first attempt for this object, record its initial z for the
            # later grasp-success check.
            if self.grasp_attempts == 0:
                self.object_initial_z = obj_pos[2]
            tgt = np.concatenate([above_obj, self.rot_euler, [-1.]])
            # jitter on — long traverse, lots of room for path diversity.
            self._push_interp(tgt, int(20 * sf), jitter=True)
            self.phase = 'DESCEND'
            return

        if self.phase == 'DESCEND':
            # NO jitter on descent — ±5 cm xy waypoints near the target can
            # bump adjacent objects (cylinders like Can roll), shifting them
            # before the gripper closes. Path diversity is still added via
            # APPROACH (long traverse to above_obj) and TRANSIT/RETREAT.
            self._push_bezier(self.last_euler[:3], grasp_pos, num=20,
                              gripper=-1., vmax=v, jitter=False)
            self.phase = 'CLOSE'
            return

        if self.phase == 'CLOSE':
            # Two stages so the gripper closes AT the object's current xy,
            # not while still translating from DESCEND's stale target.
            #   ALIGN_OPEN: re-aim to fresh obj_pos with gripper OPEN — catches
            #               any object drift between DESCEND-queue-time and now.
            #   GRIP:       close gripper in place, no further translation.
            align_target = np.concatenate([grasp_pos, self.rot_euler, [-1.]])
            self._push_interp(align_target, int(6 * sf), gripper=-1., jitter=False)
            grip_target = np.concatenate([grasp_pos, self.rot_euler, [1.]])
            self._push_interp(grip_target, int(6 * sf), gripper=1., jitter=False)
            self.phase = 'LIFT'
            return

        if self.phase == 'LIFT':
            lift_pos = grasp_pos.copy()
            lift_pos[2] = obj_pos[2] + 0.20
            # jitter on for the lift bezier.
            self._push_bezier(self.last_euler[:3], lift_pos, num=20,
                              gripper=1., vmax=v, jitter=True)
            # Settle a few steps so the gripper-width / object-z readings stabilize.
            self._push_interp(self.last_euler.copy(), int(6 * sf), gripper=1.,
                              jitter=False)
            self.phase = 'CHECK_GRASP'
            return

        if self.phase == 'CHECK_GRASP':
            self.grasp_attempts += 1
            grasped = self._check_grasped(obj_name)
            width = self._gripper_width()
            cur_z = self._obj_pos(obj_name)[2]
            print(f"  [grasp_check] obj={obj_name} attempt={self.grasp_attempts} "
                  f"width={width:.4f} dz={(cur_z - (self.object_initial_z or 0.0)):.3f} "
                  f"-> {'OK' if grasped else 'MISS'}")
            if grasped:
                self.phase = 'TRANSIT'
            elif self.grasp_attempts < self.max_grasp_attempts:
                self.phase = 'REGRASP_OPEN'
            else:
                # Record failure and skip this object.
                self.grasp_results.append({
                    'object': obj_name, 'attempts': self.grasp_attempts, 'grasped': False
                })
                self.phase = 'NEXT'
            return  # caller will re-enter _plan_phase since action_queue is empty

        if self.phase == 'REGRASP_OPEN':
            # Precise re-positioning before next grasp attempt — no jitter.
            open_target = self.last_euler.copy()
            open_target[6] = -1.
            self._push_interp(open_target, int(6 * sf), gripper=-1., jitter=False)
            self.phase = 'APPROACH'
            return

        if self.phase == 'TRANSIT':
            self.grasp_results.append({
                'object': obj_name, 'attempts': self.grasp_attempts, 'grasped': True
            })
            # Long traverse — jitter on.
            self._push_bezier(self.last_euler[:3], above_bin, num=25,
                              gripper=1., vmax=v, jitter=True)
            self.phase = 'LOWER'
            return

        if self.phase == 'LOWER':
            # Precision drop into bin — no jitter so the object lands where intended.
            self._push_bezier(self.last_euler[:3], release_pos, num=15,
                              gripper=1., vmax=0.0, jitter=False)
            self.phase = 'RELEASE'
            return

        if self.phase == 'RELEASE':
            # Critical release moment — no jitter.
            release_target = np.concatenate([release_pos, self.rot_euler, [-1.]])
            self._push_interp(release_target, int(6 * sf), gripper=-1., jitter=False)
            self.phase = 'RETREAT'
            return

        if self.phase == 'RETREAT':
            # Long traverse upward — jitter on.
            retreat_target = np.concatenate([above_bin, self.rot_euler, [-1.]])
            self._push_interp(retreat_target, int(10 * sf), gripper=-1., jitter=True)
            self.phase = 'NEXT'
            return

        if self.phase == 'NEXT':
            self.q_idx += 1
            self.grasp_attempts = 0
            self.object_initial_z = None
            self.phase = 'APPROACH' if self.q_idx < len(self.queue) else 'DONE'
            return

    # ----- per-step interface -----
    def reset(self):
        # Build placement queue, restricted to the active subset of objects.
        if self.order == 'canonical':
            idxs = self.active_canonical[:self.n_objects]
        elif self.order == 'reversed':
            idxs = self.active_reversed[:self.n_objects]
        else:
            raise ValueError(f"Unknown order: {self.order}")
        self.queue = [(idx, self.drop_modes[k], self.drop_heights[k])
                      for k, idx in enumerate(idxs)]
        # Sample one xy release-offset per object so each placement lands at
        # a slightly different point inside (or near) the target bin. Same
        # offset is reused across grasp retries for the same object so the
        # release target stays consistent within one placement attempt.
        if self.release_xy_noise > 0.0:
            self._release_offsets = [
                (float(np.random.uniform(-self.release_xy_noise, self.release_xy_noise)),
                 float(np.random.uniform(-self.release_xy_noise, self.release_xy_noise)))
                for _ in idxs
            ]
        else:
            self._release_offsets = [(0.0, 0.0) for _ in idxs]
        self.q_idx = 0
        self.phase = 'INIT'
        self.grasp_attempts = 0
        self.object_initial_z = None
        self.action_queue = []
        self.grasp_results = []

        cur_xyz = self._robomimic_env().get_observation()['robot0_eef_pos']
        self.last_euler = np.concatenate([cur_xyz, self.rot_euler, [-1.]])

    def predict_action(self, obs):
        # Keep planning phases until either we have actions or we're DONE
        # and have generated some hold steps.
        guard = 0
        while not self.action_queue and guard < 20:
            self._plan_phase()
            guard += 1
        if self.action_queue:
            action_euler = self.action_queue.pop(0)
        else:
            action_euler = self.last_euler.copy()

        action = action_euler.reshape((1, 7)).copy()
        action[0, 3:6] = self.rotation_transformer.forward(action[:, 3:6].reshape((1, 3)))
        return action


def create_robomimic_env(quadrant_placement=True, quadrant_noise=0.03, settle_steps=40,
                         n_active_objects=4):
    env_meta = build_env_meta()
    ObsUtils.initialize_obs_modality_mapping_from_dict({'low_dim': OBS_KEYS})
    robomimic_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False,
    )

    fps = 10
    crf = 22
    robosuite_fps = 20
    steps_per_render = max(robosuite_fps // fps, 1)

    env = MultiStepWrapper(
        VideoRecordingWrapper(
            RobomimicPickPlaceLowdimWrapper(
                env=robomimic_env,
                obs_keys=OBS_KEYS,
                init_state=None,
                quadrant_placement=quadrant_placement,
                quadrant_noise=quadrant_noise,
                settle_steps=settle_steps,
                n_active_objects=n_active_objects,
            ),
            video_recoder=VideoRecorder.create_h264(
                fps=fps, codec='h264', input_pix_fmt='rgb24',
                crf=crf, thread_type='FRAME', thread_count=1
            ),
            file_path=None,
            steps_per_render=steps_per_render
        ),
        n_obs_steps=1,
        n_action_steps=1,
        max_episode_steps=2000,
    )

    return env, env_meta


def compute_smoothness(actions):
    if len(actions) < 4:
        return 1.0
    actions = np.array(actions)
    jerk = np.diff(actions, n=3, axis=0)
    jerk_mag = np.linalg.norm(jerk, axis=-1)
    return float(np.exp(-10.0 * float(np.mean(jerk_mag))))


def compute_speed_reward(steps_taken, max_steps=2000):
    return 1.0 - 0.9 * (steps_taken / max_steps)


@click.command()
@click.option('-o', '--output_dir', default='shared_data_pickplace/scripted_data')
@click.option('-n', '--num_episodes', type=int, default=200)
@click.option('--seed', type=int, default=0)
@click.option('--order_mode', type=click.Choice(['canonical', 'reversed', 'random']),
              default='random', help="'random' samples canonical-vs-reversed per demo (50/50).")
@click.option('--n_objects_min', type=int, default=1)
@click.option('--n_objects_max', type=int, default=4)
@click.option('--drop_mode', type=click.Choice(['careful', 'drop', 'random']),
              default='random', help="'random' samples careful-vs-drop per demo (50/50).")
@click.option('--drop_height_min', type=float, default=0.15)
@click.option('--drop_height_max', type=float, default=0.20)
@click.option('--careful_height', type=float, default=0.04)
@click.option('--noise_min', type=float, default=0.0)
@click.option('--noise_max', type=float, default=0.05)
@click.option('--speed_factor', type=float, default=1.0)
@click.option('--max_steps', type=int, default=2000)
@click.option('--save_all_videos/--save_some_videos', default=False,
              help='Record an MP4 for every episode vs. just the first 3 (default). '
                   '"Some" disables rendering entirely after the first 3, which '
                   'meaningfully speeds up bulk demo collection.')
@click.option('--quadrant_noise', type=float, default=0.03,
              help='Per-object xy jitter (m) around the quadrant center in bin1.')
@click.option('--settle_steps', type=int, default=40,
              help='sim.step() count after placement so objects can settle under gravity.')
@click.option('--quadrant_placement/--random_placement', default=True,
              help='Place objects in fixed bin1 quadrants (default) or use the env default random placement.')
@click.option('--max_grasp_attempts', type=int, default=3,
              help='Max times the scripted policy will retry a failed grasp before skipping that object.')
@click.option('--n_active_objects', type=int, default=4,
              help='Number of objects active in the scene (1..4). The first N in the right-first canonical order are kept; the rest are cleared from the bin.')
@click.option('--path_jitter', type=float, default=0.05,
              help='±xy meters of per-step jitter applied to transit waypoints (APPROACH/DESCEND/LIFT/TRANSIT/RETREAT). 0 disables. CLOSE/RELEASE/LOWER/regrasp are never jittered.')
@click.option('--release_xy_noise', type=float, default=0.0,
              help='±xy meters of random offset applied to the release position '
                   '(per object, sampled once at reset). Spreads where placed '
                   'objects land in / near their target bin so the *_placed_raw '
                   'reward varies continuously instead of clustering at 0. '
                   'Bin half-width is ~0.10m; values up to ~0.08 keep most '
                   'objects in-bin while still giving useful spread.')
def main(path_jitter, release_xy_noise, output_dir, num_episodes, seed,
         order_mode, n_objects_min, n_objects_max,
         drop_mode, drop_height_min, drop_height_max, careful_height,
         noise_min, noise_max, speed_factor, max_steps, save_all_videos,
         quadrant_noise, settle_steps, quadrant_placement, max_grasp_attempts,
         n_active_objects):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Seed: {seed}")
    print(f"Collecting {num_episodes} episodes — order={order_mode}, "
          f"n_objects=[{n_objects_min},{n_objects_max}], drop={drop_mode}, "
          f"drop_h=[{drop_height_min},{drop_height_max}], careful_h={careful_height}")

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_active_objects = max(1, min(4, int(n_active_objects)))
    active_canonical = CANONICAL_ORDER[:n_active_objects]
    active_reversed = list(reversed(active_canonical))
    env, env_cfg = create_robomimic_env(
        quadrant_placement=quadrant_placement,
        quadrant_noise=quadrant_noise,
        settle_steps=settle_steps,
        n_active_objects=n_active_objects,
    )

    dataset_file = h5py.File(pathlib.Path(output_dir).joinpath("demos.hdf5"), 'w')
    dataset_data_group = dataset_file.create_group('data')
    # robomimic dataset loaders read env_args from this attr.
    dataset_data_group.attrs['env_args'] = json.dumps(env_cfg.copy())

    total_rollouts = 0
    successful_rollouts = 0
    per_episode_metadata = []

    # Per-episode noise sampled uniformly.
    noise_levels = np.linspace(noise_min, noise_max, num_episodes)
    np.random.shuffle(noise_levels)

    for ep_num in range(num_episodes):
        noise = float(noise_levels[ep_num])

        # Sample per-axis values
        if order_mode == 'random':
            order = random.choice(['canonical', 'reversed'])
        else:
            order = order_mode

        n_objs_cap = min(n_objects_max, n_active_objects)
        n_objs = int(np.random.randint(min(n_objects_min, n_objs_cap), n_objs_cap + 1))

        # Per-object drop mode + height (in placement order, length = n_objs).
        # 'random' samples drop_height uniformly across the FULL range
        # [careful_height, drop_height_max] so the resulting drop reward is
        # continuous (no gap between the careful and drop modes). The
        # 'careful'/'drop' categorical mode label is then derived from the
        # sampled height (`careful` if height < drop_height_min else `drop`).
        per_obj_dmodes = []
        per_obj_dheights = []
        for _ in range(n_objs):
            if drop_mode == 'random':
                h = float(np.random.uniform(careful_height, drop_height_max))
                m = 'careful' if h < drop_height_min else 'drop'
            elif drop_mode == 'careful':
                h = float(careful_height)
                m = 'careful'
            else:  # drop_mode == 'drop'
                h = float(np.random.uniform(drop_height_min, drop_height_max))
                m = 'drop'
            per_obj_dmodes.append(m)
            per_obj_dheights.append(h)

        n_careful = sum(1 for m in per_obj_dmodes if m == 'careful')
        n_dropped = n_objs - n_careful

        print(f"\nEpisode {ep_num}: order={order}, n_objects={n_objs}, "
              f"drop_modes={per_obj_dmodes}, drop_heights={[f'{h:.3f}' for h in per_obj_dheights]}, "
              f"noise={noise:.4f}")

        # Reset env + policy
        assert isinstance(env.env, VideoRecordingWrapper)
        env.env.video_recoder.stop()
        if save_all_videos or ep_num < 3:
            vid_path = pathlib.Path(output_dir).joinpath(f"vids/ep{ep_num}.mp4")
            vid_path.parent.mkdir(parents=False, exist_ok=True)
            env.env.file_path = str(vid_path)
            current_video_path = str(vid_path)
        else:
            env.env.file_path = None
            current_video_path = None

        env.env.env.init_state = None
        env.seed(np.random.randint(0, 10_000_000))
        obs = env.reset()

        policy = PickPlaceScriptedPolicy(
            env=env,
            order=order,
            n_objects=n_objs,
            drop_modes=per_obj_dmodes,
            drop_heights=per_obj_dheights,
            careful_height=careful_height,
            noise_level=noise,
            speed_factor=speed_factor,
            max_grasp_attempts=max_grasp_attempts,
            active_order=active_canonical,
            path_jitter=path_jitter,
            release_xy_noise=release_xy_noise,
        )
        policy.reset()

        # Capture the initial sim state so the env_runner can replay this episode.
        init_state = policy._robomimic_env().get_state()['states'].copy()

        obs_list = []
        action_list = []
        rewards = []
        objects_in_bins_final = np.zeros(4)

        def _extract_oib(info):
            if isinstance(info, dict) and 'objects_in_bins' in info:
                return np.array(info['objects_in_bins'])
            if isinstance(info, list) and info and 'objects_in_bins' in info[-1]:
                return np.array(info[-1]['objects_in_bins'])
            return None

        def _settle(n_settle_steps, obs, objects_in_bins_final):
            for _ in range(n_settle_steps):
                act = policy.predict_action(obs)
                obs_list.append(obs[-1].copy())
                action_list.append(act[0].copy())
                obs, rew, _d, info = env.step(act)
                rewards.append(float(rew[-1]) if hasattr(rew, '__len__') else float(rew))
                new_oib = _extract_oib(info)
                if new_oib is not None:
                    objects_in_bins_final = new_oib
            return obs, objects_in_bins_final

        for step in range(max_steps):
            action = policy.predict_action(obs)

            obs_list.append(obs[-1].copy())
            action_list.append(action[0].copy())

            obs, reward, done, info = env.step(action)
            rewards.append(float(reward[-1]) if hasattr(reward, '__len__') else float(reward))
            new_oib = _extract_oib(info)
            if new_oib is not None:
                objects_in_bins_final = new_oib

            # Terminate early when:
            #   - we've placed the target number, OR
            #   - the policy has finished its plan (all objects attempted —
            #     either placed or skipped after max_grasp_attempts).
            n_placed_now = int(np.sum(objects_in_bins_final))
            policy_done = (policy.phase == 'DONE')
            if n_placed_now >= n_objs or policy_done:
                obs, objects_in_bins_final = _settle(5, obs, objects_in_bins_final)
                break

        n_placed = int(np.sum(objects_in_bins_final))
        success = n_placed >= n_objs  # placed the number the policy intended
        if success:
            successful_rollouts += 1
        total_rollouts += 1

        ep_steps = len(action_list)
        ep_smoothness = compute_smoothness(np.stack(action_list, axis=0))
        # Gate by success — 0 if no objects placed, real smoothness otherwise.
        if n_placed <= 0:
            ep_smoothness = 0.0
        ep_speed = compute_speed_reward(ep_steps, max_steps=max_steps)
        # Signed reward axes (mirror the existing peg axis convention):
        order_reward = 1.0 if order == 'canonical' else -1.0
        # drop_reward: signed fraction-careful in [-1, +1].
        # all-careful = +1, all-drop = -1, mixed = (n_careful - n_dropped) / n_objs.
        drop_reward = float(n_careful - n_dropped) / max(n_objs, 1)

        outcome = "SUCCESS" if success else "FAIL"
        print(f"  [{outcome}] steps={ep_steps}, placed={n_placed}/{n_objs} (target), "
              f"speed={ep_speed:.3f}, smoothness={ep_smoothness:.3f}, "
              f"order={order}({order_reward:+.0f}), "
              f"careful={n_careful}/{n_objs} drop_r={drop_reward:+.2f}")

        # Save to HDF5 (split obs back into per-key arrays for dataset compat).
        ep_group = dataset_data_group.create_group(f'demo_{ep_num}')
        obs_group = ep_group.create_group('obs')
        obs_array = np.stack(obs_list, axis=0)
        raw_obs = policy._robomimic_env().get_observation()
        idx = 0
        for key in OBS_KEYS:
            dim = raw_obs[key].shape[0]
            obs_group.create_dataset(key, data=obs_array[:, idx:idx + dim])
            idx += dim

        action_array = np.stack(action_list, axis=0)
        ep_group.create_dataset('actions', data=action_array)

        # Save full init state across all timesteps so env_runner can reset to it.
        states = np.broadcast_to(init_state, (len(obs_list), len(init_state))).copy()
        ep_group.create_dataset('states', data=states)

        # Map per-placement drop info back to per-object-id arrays of length 4
        # (NaN for objects this demo never tried to place).
        if order == 'canonical':
            order_idxs = active_canonical[:n_objs]
        else:
            order_idxs = active_reversed[:n_objs]
        per_obj_dmodes_by_id = ['none'] * 4
        per_obj_dheights_by_id = [float('nan')] * 4
        for place_idx, obj_id in enumerate(order_idxs):
            per_obj_dmodes_by_id[obj_id] = per_obj_dmodes[place_idx]
            per_obj_dheights_by_id[obj_id] = per_obj_dheights[place_idx]

        grasp_results = list(getattr(policy, 'grasp_results', []))
        ep_group.attrs['order'] = order
        ep_group.attrs['n_objects_target'] = n_objs
        ep_group.attrs['n_objects_placed'] = n_placed
        # Per-object drop info, both in placement order and indexed by object id.
        ep_group.attrs['drop_modes_in_order'] = json.dumps(per_obj_dmodes)
        ep_group.attrs['drop_heights_in_order'] = np.array(per_obj_dheights, dtype=np.float32)
        ep_group.attrs['drop_modes_by_object_id'] = json.dumps(per_obj_dmodes_by_id)
        ep_group.attrs['drop_heights_by_object_id'] = np.array(per_obj_dheights_by_id, dtype=np.float32)
        ep_group.attrs['n_careful'] = n_careful
        ep_group.attrs['n_dropped'] = n_dropped
        ep_group.attrs['drop_reward'] = float(drop_reward)
        ep_group.attrs['careful_height'] = careful_height
        ep_group.attrs['noise_level'] = noise
        ep_group.attrs['speed_factor'] = speed_factor
        ep_group.attrs['objects_in_bins'] = objects_in_bins_final
        ep_group.attrs['grasp_results'] = json.dumps(grasp_results)

        # Per-object placement axes (replaces the lumped `partial_reward`).
        oib = np.array(objects_in_bins_final, dtype=np.float32).flatten()
        milk_placed = float(oib[0]) if oib.size > 0 else 0.0
        bread_placed = float(oib[1]) if oib.size > 1 else 0.0
        cereal_placed = float(oib[2]) if oib.size > 2 else 0.0
        can_placed = float(oib[3]) if oib.size > 3 else 0.0

        per_episode_metadata.append({
            'demo_idx': ep_num,
            'video_path': current_video_path,
            'order': order,
            'n_objects_target': n_objs,
            'n_objects_placed': n_placed,
            'placement_order': [OBJ_NAMES[i] for i in order_idxs],
            'drop_modes_in_order': per_obj_dmodes,
            'drop_heights_in_order': per_obj_dheights,
            'drop_modes_by_object': {OBJ_NAMES[i]: per_obj_dmodes_by_id[i] for i in range(4)},
            'drop_heights_by_object': {OBJ_NAMES[i]: per_obj_dheights_by_id[i] for i in range(4)},
            'grasp_results': grasp_results,
            'n_careful': int(n_careful),
            'n_dropped': int(n_dropped),
            'careful_height': careful_height,
            'noise_level': float(noise),
            'speed_factor': float(speed_factor),
            'steps_taken': int(ep_steps),
            'objects_in_bins': [float(x) for x in oib.tolist()],
            'success': bool(success),
            'rewards': {
                'success': float(n_placed) / 4.0,
                'order_reward': float(order_reward),
                'milk_placed': milk_placed,
                'bread_placed': bread_placed,
                'cereal_placed': cereal_placed,
                'can_placed': can_placed,
                'drop_reward': float(drop_reward),
                'speed_reward': float(ep_speed),
                'smoothness': float(ep_smoothness),
            },
        })

    dataset_data_group.attrs['data collection'] = f"{successful_rollouts} of {total_rollouts} total rollouts successful"
    dataset_file.close()

    # Per-trajectory metadata (rewards, video paths) for inspection.
    metadata_path = pathlib.Path(output_dir) / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            'config': {
                'num_episodes': num_episodes,
                'seed': seed,
                'order_mode': order_mode,
                'n_objects_min': n_objects_min,
                'n_objects_max': n_objects_max,
                'drop_mode': drop_mode,
                'drop_height_min': drop_height_min,
                'drop_height_max': drop_height_max,
                'careful_height': careful_height,
                'noise_min': noise_min,
                'noise_max': noise_max,
                'speed_factor': speed_factor,
                'max_steps': max_steps,
            },
            'episodes': per_episode_metadata,
        }, f, indent=2)
    print(f"Wrote per-trajectory metadata to {metadata_path}")

    print("\nSaving .npz format...")
    save_npz_from_hdf5(output_dir, max_steps=max_steps)
    print(f"{successful_rollouts} of {total_rollouts} total rollouts successful")


def save_npz_from_hdf5(output_dir, max_steps=2000):
    """Convert HDF5 demos to .npz with the per-axis reward fields the pipeline expects."""
    hdf5_path = pathlib.Path(output_dir) / "demos.hdf5"
    npz_path = pathlib.Path(output_dir) / "rollouts.npz"

    rot_transformer = RotationTransformer(from_rep='axis_angle', to_rep='rotation_6d')

    with h5py.File(hdf5_path, 'r') as f:
        demos = f['data']
        n_episodes = len([k for k in demos.keys() if k.startswith('demo_')])

        all_obs, all_actions, all_lengths = [], [], []
        all_success, all_speed, all_smoothness = [], [], []
        all_order_reward, all_drop_reward = [], []
        # Per-object placement axes (replaces lumped partial_reward).
        all_milk_placed, all_bread_placed, all_cereal_placed, all_can_placed = [], [], [], []
        # Per-object drop axes (+1 careful, -1 drop, 0 not attempted).
        all_milk_drop, all_bread_drop, all_cereal_drop, all_can_drop = [], [], [], []
        all_n_placed, all_n_careful, all_n_dropped = [], [], []
        all_mean_drop_h_dropped = []
        # Per-object-id arrays of length 4 (NaN if not placed by this demo)
        all_drop_heights_by_id = []
        all_drop_modes_by_id = []

        max_len = 0
        for i in range(n_episodes):
            demo = demos[f'demo_{i}']
            obs_parts = [demo['obs'][key][:].astype(np.float32) for key in OBS_KEYS]
            obs = np.concatenate(obs_parts, axis=-1)
            actions = demo['actions'][:].astype(np.float32)

            if actions.shape[-1] == 7:
                pos = actions[:, :3]
                rot = actions[:, 3:6]
                gripper = actions[:, 6:]
                rot6d = rot_transformer.forward(rot)
                actions = np.concatenate([pos, rot6d, gripper], axis=-1).astype(np.float32)

            L = min(len(obs), len(actions))
            all_obs.append(obs[:L])
            all_actions.append(actions[:L])
            all_lengths.append(L)
            max_len = max(max_len, L)

            n_placed = int(demo.attrs.get('n_objects_placed', 0))
            all_n_placed.append(n_placed)
            all_success.append(float(n_placed) / 4.0)  # fraction placed
            all_speed.append(compute_speed_reward(L, max_steps=max_steps))
            # Gate smoothness by success: 0 if no objects placed.
            _smooth = compute_smoothness(actions[:L])
            if n_placed <= 0:
                _smooth = 0.0
            all_smoothness.append(_smooth)

            order = str(demo.attrs.get('order', 'canonical'))
            all_order_reward.append(1.0 if order == 'canonical' else -1.0)

            oib = np.array(demo.attrs.get('objects_in_bins', np.zeros(4)), dtype=np.float32).flatten()
            all_milk_placed.append(float(oib[0]) if oib.size > 0 else 0.0)
            all_bread_placed.append(float(oib[1]) if oib.size > 1 else 0.0)
            all_cereal_placed.append(float(oib[2]) if oib.size > 2 else 0.0)
            all_can_placed.append(float(oib[3]) if oib.size > 3 else 0.0)

            # Per-object drop info written by the new collector.
            drop_reward = float(demo.attrs.get('drop_reward', 0.0))
            all_drop_reward.append(drop_reward)
            n_careful = int(demo.attrs.get('n_careful', 0))
            n_dropped = int(demo.attrs.get('n_dropped', 0))
            all_n_careful.append(n_careful)
            all_n_dropped.append(n_dropped)

            heights_by_id = np.array(demo.attrs.get('drop_heights_by_object_id', np.full(4, np.nan)), dtype=np.float32)
            all_drop_heights_by_id.append(heights_by_id)
            try:
                modes_by_id = json.loads(str(demo.attrs.get('drop_modes_by_object_id', '["none","none","none","none"]')))
            except Exception:
                modes_by_id = ['none'] * 4
            all_drop_modes_by_id.append(modes_by_id)

            # Per-object drop reward (+1 careful, -1 drop, 0 if not attempted).
            def _drop_val(mode):
                if mode == 'careful':
                    return 1.0
                if mode == 'drop':
                    return -1.0
                return 0.0
            all_milk_drop.append(_drop_val(modes_by_id[0]))
            all_bread_drop.append(_drop_val(modes_by_id[1]))
            all_cereal_drop.append(_drop_val(modes_by_id[2]))
            all_can_drop.append(_drop_val(modes_by_id[3]))

            heights_in_order = np.array(demo.attrs.get('drop_heights_in_order', []), dtype=np.float32)
            try:
                modes_in_order = json.loads(str(demo.attrs.get('drop_modes_in_order', '[]')))
            except Exception:
                modes_in_order = []
            dropped_heights = [h for h, m in zip(heights_in_order.tolist(), modes_in_order) if m == 'drop']
            all_mean_drop_h_dropped.append(float(np.mean(dropped_heights)) if dropped_heights else 0.0)

        obs_dim = all_obs[0].shape[-1]
        act_dim = all_actions[0].shape[-1]
        obs_padded = np.zeros((n_episodes, max_len, obs_dim), dtype=np.float32)
        act_padded = np.zeros((n_episodes, max_len, act_dim), dtype=np.float32)
        for i in range(n_episodes):
            L = all_lengths[i]
            obs_padded[i, :L] = all_obs[i]
            act_padded[i, :L] = all_actions[i]

        np.savez(npz_path,
            obs=obs_padded,
            actions=act_padded,
            episode_lengths=np.array(all_lengths, dtype=np.int32),
            success=np.array(all_success, dtype=np.float32),
            speed_reward=np.array(all_speed, dtype=np.float32),
            smoothness=np.array(all_smoothness, dtype=np.float32),
            order_reward=np.array(all_order_reward, dtype=np.float32),
            # Per-object placement axes (each 0/1) — replace `partial_reward`.
            milk_placed=np.array(all_milk_placed, dtype=np.float32),
            bread_placed=np.array(all_bread_placed, dtype=np.float32),
            cereal_placed=np.array(all_cereal_placed, dtype=np.float32),
            can_placed=np.array(all_can_placed, dtype=np.float32),
            # Per-object drop axes (+1 careful, -1 drop, 0 not attempted) —
            # finer-grained version of the aggregated `drop_reward` axis.
            milk_drop=np.array(all_milk_drop, dtype=np.float32),
            bread_drop=np.array(all_bread_drop, dtype=np.float32),
            cereal_drop=np.array(all_cereal_drop, dtype=np.float32),
            can_drop=np.array(all_can_drop, dtype=np.float32),
            drop_reward=np.array(all_drop_reward, dtype=np.float32),
            n_placed=np.array(all_n_placed, dtype=np.int32),
            n_careful=np.array(all_n_careful, dtype=np.int32),
            n_dropped=np.array(all_n_dropped, dtype=np.int32),
            mean_drop_height_dropped=np.array(all_mean_drop_h_dropped, dtype=np.float32),
            drop_heights_by_object_id=np.stack(all_drop_heights_by_id, axis=0),
        )
        print(f"Saved {n_episodes} episodes to {npz_path}")
        print(f"  obs shape: {obs_padded.shape}, action shape: {act_padded.shape}")
        print(f"  Canonical: {sum(1 for r in all_order_reward if r > 0)}, "
              f"Reversed: {sum(1 for r in all_order_reward if r < 0)}")
        print(f"  Total careful placements: {sum(all_n_careful)}, "
              f"total drop placements: {sum(all_n_dropped)}")
        print(f"  Mean n_placed: {np.mean(all_n_placed):.2f}, "
              f"mean drop_reward: {np.mean(all_drop_reward):+.2f}")
        print(f"  Per-object placement rates: "
              f"milk={np.mean(all_milk_placed):.2f}, "
              f"bread={np.mean(all_bread_placed):.2f}, "
              f"cereal={np.mean(all_cereal_placed):.2f}, "
              f"can={np.mean(all_can_placed):.2f}")


if __name__ == '__main__':
    main()
