from oculus_reader.oculus_controller import VRPolicy
from simple_droid_env import DroidEnv
import time
import numpy as np
from data_utils import StateEncoding, convert_state_to_array
import pathlib
import h5py
import json 
from datetime import datetime
import cv2

def time_ms():
    return int(time.time() * 1000)

args = {
    'robot_action_space':"cartesian_velocity",
    'gripper_action_space':"position",
    'cameras':{"agent_view": "23404442", "wrist": "17471093"},
    'ip_address':"172.16.0.7"}
env = DroidEnv(**args)

controller = VRPolicy()

num_episodes = 141
episode_idx = 127
current_time = datetime.now().strftime("%H%M%S")
dataset_dir = "demos/test"
wait_for_controller = True
policy = None
horizon = None
image_keys = ["agent_view", "wrist"]

print(episode_idx)
while episode_idx < num_episodes:

    pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)

    writing_mode = "w"
    dataset_file = h5py.File(pathlib.Path(dataset_dir).joinpath(f"demos_{episode_idx}.hdf5"), writing_mode)
    dataset_data_group = dataset_file.create_group('data')
    dataset_data_group.attrs['env_args'] = json.dumps(args.copy())

    controller.reset_state()
    env.reset(randomize=False)

    num_steps = 0
    trajectory = []
    print("reset and start")

    while True:
        controller_info = {} if (controller is None) else controller.get_info()
        skip_action = wait_for_controller and (not controller_info["movement_enabled"])
        control_timestamps = {"step_start": time_ms()}

        obs = env._get_observation()
        obs["controller_info"] = controller_info
        obs["timestamp"] = dict()
        obs["timestamp"]["skip_action"] = skip_action

        control_timestamps["policy_start"] = time_ms()
        if policy is None:
            action, controller_action_info = controller.forward(obs, include_info=True)
            assert len(action)==7 
            action_dict = {
                StateEncoding.EE_POS: action[:3],
                StateEncoding.EE_EULER: action[3:6],
                StateEncoding.GRIPPER: np.array([action[-1]]),
            }
            action = {"desired_delta": action_dict, "desired_absolute": action_dict}
        else:
            action = policy.predict_action(obs)
            controller_action_info = {}

        control_timestamps["control_start"] = time_ms()
        if skip_action:
            action_info = env.create_action_dict(np.zeros(7))
        else:
            _, _, _, _, action_dict_returned = env.step(action)
            action_info = {}
            print(action)
            action_info['cartesian_velocity'] = np.concatenate([action["desired_delta"][StateEncoding.EE_POS], action["desired_delta"][StateEncoding.EE_EULER]])
            action_info['gripper_position'] = action["desired_delta"][StateEncoding.GRIPPER]
            action_info['joint_velocity'] = action_dict_returned['joint_velocity']
            action_info['joint_position'] = action_dict_returned['joint_position']
            action_info['cartesian_position'] = action_dict_returned['cartesian_position']

        action_info.update(controller_action_info)

        control_timestamps["step_end"] = time_ms()
        obs["timestamp"]["control"] = control_timestamps
        timestep = {"observations": obs, "action": action_info}
        trajectory.append(timestep)

        num_steps += 1
        if horizon is not None:
            end_traj = horizon == num_steps
        else:
            end_traj = controller_info["success"] or controller_info["failure"]

        comp_time = time_ms() - control_timestamps["step_start"]
        sleep_left = (1 / env.control_hz) - (comp_time / 1000)
        if sleep_left > 0:
            time.sleep(sleep_left)

        if end_traj:
            break

    if controller_info["success"]:
        t0 = time.time()

        # Initialize video writer before processing loop
        sample_frame = trajectory[0]["observations"]["image"][image_keys[0]]
        h, w, _ = sample_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(dataset_dir + f"/video_{episode_idx}.mp4", fourcc, 15, (w * len(image_keys), h))

        trajectory_treated = {
            "obs": [],
            "actions": [],
            "actions_abs": [],
        }

        while trajectory:
            ts = trajectory.pop(0)
            skip_action = ts["observations"]["timestamp"]["skip_action"]
            if skip_action:
                continue

            obs_dict = ts["observations"]
            action = ts["action"]

            # Write video frame inline
            frame_list = [cv2.cvtColor(obs_dict["image"][cam], cv2.COLOR_RGB2BGR) for cam in image_keys]
            video_writer.write(cv2.hconcat(frame_list))

            treated_obs_dict = obs_dict["state"]
            for cam_name in obs_dict["image"]:
                treated_obs_dict[cam_name] = obs_dict["image"][cam_name]
 
            trajectory_treated["obs"].append(treated_obs_dict)
            trajectory_treated["actions"].append(action)
            trajectory_treated["actions_abs"].append(np.concatenate([action["target_cartesian_position"], [action["target_gripper_position"]]]))

        video_writer.release()

        ep_group = dataset_data_group.create_group(f'demo_{episode_idx}')
        obs_group = ep_group.create_group('obs')
        for obs_key in trajectory_treated['obs'][0].keys():
            dataset_name = obs_key.value if isinstance(obs_key, StateEncoding) else obs_key
            obs_array = np.stack([od[obs_key] for od in trajectory_treated['obs']], axis=0)
            obs_group.create_dataset(dataset_name, data=obs_array)

        action_group = ep_group.create_group('actions')
        print(trajectory_treated['actions'][0].keys())
        for act_key in ['cartesian_velocity', 'gripper_position', 'joint_velocity', 'joint_position', 'cartesian_position']:
            act_data_array = np.stack([a[act_key] for a in trajectory_treated['actions']], axis=0)
            action_group.create_dataset(act_key, data=act_data_array)

        action_abs_array = np.stack([a for a in trajectory_treated['actions_abs']], axis=0)
        ep_group.create_dataset('actions_abs', data=action_abs_array)
        ep_group.attrs['scripted_policy_type'] = "human_teleop"
        print(f'Saving: {episode_idx} {time.time() - t0:.1f} secs')

        episode_idx += 1
    else:
        print("deleting traj")

    dataset_file.close()

print("Congrats! You're done collecting demos!")