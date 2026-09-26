import os
import h5py
import torch
import torch.nn.functional as F
import numpy as np
from icil.config.dataset_config import DatasetConfig
from icil.config.shared_config import SharedConfig
from typing import Tuple

class PseudoDemoDataset(torch.utils.data.Dataset):
    minimum_length : int = 20
    maximum_length : int = 250

    def __init__(
        self,
        dataset_config : DatasetConfig,
        shared_config : SharedConfig,
        split : str = "train",
    ): 
        # dataset_path: List of hdf5 paths
        self.dataset_path = dataset_config.hdf5_path
        self.num_traj_per_task = shared_config.num_traj_per_task
        self.num_prompt_traj = dataset_config.num_prompt_traj 
        self.enable_debug = dataset_config.enable_debug   
        self.obs_keys = shared_config.obs_keys
        self.label_keys = dataset_config.label_keys
        self.img_keys = shared_config.image_keys
        self.image_size = shared_config.image_size
        self.action_key = shared_config.action_key
        self.auxiliary_key = shared_config.auxiliary_key
        self.waypoint_key = dataset_config.waypoint_key
        self.bg_key = shared_config.bg_key
        self.goal_key = dataset_config.goal_key
        self.action_scale = dataset_config.action_scale
        self.action_min = dataset_config.action_min
        self.action_max = dataset_config.action_max
        self.auxiliary_scale = dataset_config.auxiliary_scale
        self.auxiliary_min = dataset_config.auxiliary_min
        self.auxiliary_max = dataset_config.auxiliary_max
        self.obs_scale = dataset_config.obs_scale
        self.obs_min = dataset_config.obs_min
        self.obs_max = dataset_config.obs_max
        self.with_respect_to_first_action = dataset_config.with_respect_to_first_action
        self.has_obs = len(self.obs_keys) > 0
        self.has_img = len(self.img_keys) > 0
        self.max_n_prompts = dataset_config.max_n_prompts
        self.split = split
        self.task_keys = []
        # for i in range(dataset_config.num_tasks):
        #     key = f"episode_{i}"
        #     self.task_keys.append(key)
        # print("Number of tasks: ", len(self.task_keys))
        # Get all the hdf5 files in the subfolders and dataset_path
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(".h5"):
                    file_path = os.path.join(root, file)
                    self.task_keys.append(file_path)
        print("Number of tasks: ", len(self.task_keys))

        # define train test split 
        self.split = self.split
        self.train_split = dataset_config.train_val_split
        
        # set seed and shuffle the hdf5 keys
        self.rng = np.random.RandomState(seed=shared_config.seed)
        self.rng.shuffle(self.task_keys)
        num_train = int(len(self.task_keys) * self.train_split)
        if self.split == "train": 
            self.task_keys = self.task_keys[:num_train]
        else:
            self.task_keys = self.task_keys[num_train:]

        # define sequence length 
        self.prompt_length = shared_config.prompt_length
        self.task_length = shared_config.task_length
        
        # change prediction to be k steps 
        self.num_pred_steps = shared_config.n_pred_steps
        self.num_history_steps = shared_config.n_hist_steps
        assert self.num_pred_steps >= 1, "Number of prediction steps must be at least 1"
        print("Number of prediction steps: ", self.num_pred_steps)

        # load the dataset 
        self.traj_key_to_index = {}

        # calculate normalization statistics
        self.calculate_norm()
    
    def calculate_norm(self,):
        # Calculate the normalization statistics for the dataset
        global_min_action = None
        global_max_action = None
        global_min_robot_state = None
        global_max_robot_state = None
        for file in self.task_keys:
            if file.endswith(".h5"):
                try:
                    with h5py.File(file, "r") as f:
                        # Iterate each group in the HDF5 file
                        for group_name in f.keys():
                            group = f[group_name]
                            robot_states = group['robot_states'][:]
                            if global_min_robot_state is None:
                                global_min_robot_state = robot_states.min(axis=0)
                                global_max_robot_state = robot_states.max(axis=0)
                            else:
                                global_min_robot_state = np.minimum(global_min_robot_state, robot_states.min(axis=0))
                                global_max_robot_state = np.maximum(global_max_robot_state, robot_states.max(axis=0))
                            actions = group['actions'][:]
                            if global_min_action is None:
                                global_min_action = actions.min(axis=0)
                                global_max_action = actions.max(axis=0)
                            else:
                                global_min_action = np.minimum(global_min_action, actions.min(axis=0))
                                global_max_action = np.maximum(global_max_action, actions.max(axis=0))
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                    continue
        self.action_min = global_min_action
        self.action_max = global_max_action
        self.obs_min = global_min_robot_state
        self.obs_max = global_max_robot_state
        # print("Calculated action min: ", self.action_min)
        # print("Calculated action max: ", self.action_max)
        # print("Calculated obs min: ", self.obs_min)
        # print("Calculated obs max: ", self.obs_max)

    def save_norm_stats(self, path: str) -> None:
        """
        Save stats to JSON/NPZ/HDF5 based on file extension or `fmt` ('json'|'npz'|'h5').
        """
        import json
        payload = dict(
            obs_min=self.obs_min.tolist(),
            obs_max=self.obs_max.tolist(),
            action_min=self.action_min.tolist(),
            action_max=self.action_max.tolist(),
        )
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def load_norm_stats(self, path: str,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load stats from JSON/NPZ/HDF5; returns (obs_min, obs_max, action_min, action_max).
        """
        import json
        with open(path, "r") as f:
            d = json.load(f)
        self.obs_min = np.asarray(d["obs_min"], dtype=np.float32)
        self.obs_max = np.asarray(d["obs_max"], dtype=np.float32)
        self.action_min = np.asarray(d["action_min"], dtype=np.float32)
        self.action_max = np.asarray(d["action_max"], dtype=np.float32)
        return (self.obs_min, self.obs_max, self.action_min, self.action_max)

    def hdf5_to_dict(self, hdf5_file):
        """
        Convert hdf5 file to dictionary
        """
        try:
            data = {}
            for key in hdf5_file.keys():
                if isinstance(hdf5_file[key], h5py.Group):
                    sub_data = {}
                    for sub_key in hdf5_file[key].keys():
                        sub_data[sub_key] = np.array(hdf5_file[key][sub_key])
                    data[key] = sub_data
            return data
        except Exception as e:
            raise e

    def process_h5(self, hdf5: h5py.File | dict, traj_key: str):
        """
        Update the dataset
        """
        data = hdf5[traj_key]
        actions = data[self.action_key]
        pred_actions = self.get_pred_action_seq(actions, self.num_pred_steps)
        # if self.auxiliary_key != "":
        #     auxiliary = data[self.auxiliary_key]
        #     pred_auxiliary = self.get_pred_action_seq(auxiliary, self.num_pred_steps)
        multi_history_obs = []
        multi_history_img = []
        if self.has_obs:
            for key in self.obs_keys:
                history_obs = self.get_history_obs_seq(data[key], self.num_history_steps)
                if len(history_obs.shape) == 2:
                    history_obs = np.expand_dims(history_obs, axis=-1)
                multi_history_obs.append(history_obs)
            multi_history_obs = np.concatenate(multi_history_obs, axis=-1)
        if self.has_img:
            for key in self.img_keys:
                if len(data[key].shape) < 3:
                    raise ValueError(f"Image data dimension error")
                images = self.resize_images(data[key])
                history_img = self.get_history_obs_seq(images, self.num_history_steps)
                multi_history_img.append(history_img)
        proc_data = {
            "action": pred_actions,
            # "task_id": ,
            # "trajectory_id": trajectory_id,
            # "label": data["trajectory_label"],
        }
        if len(self.label_keys) > 0:
            for key in self.label_keys:
                proc_data[key] = data[key]
        if len(self.obs_keys) > 0:
            proc_data["observation"] = multi_history_obs
        if len(self.img_keys) > 0:
            for i, key in enumerate(self.img_keys):
                proc_data[key] = multi_history_img[i]
        if "masks" in data:
            proc_data["masks"] = self.get_pred_mask_seq(data["masks"], self.num_pred_steps)
        if self.auxiliary_key != "":
            proc_data[self.auxiliary_key] = data[self.auxiliary_key]
        if self.bg_key != "":
            proc_data["background"] = data[self.bg_key]
        return proc_data

    def resize_images(self, images: np.ndarray) -> np.ndarray:
        """Resize an RGB trajectory before batching to keep host memory bounded."""
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        if tensor.max() > 1.5:
            tensor = tensor / 255.0
        if tuple(tensor.shape[-2:]) != tuple(self.image_size):
            tensor = F.interpolate(
                tensor,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return tensor.permute(0, 2, 3, 1).numpy()
    
    def get_pred_mask_seq(self, mask: np.array, num_pred_steps: int) -> torch.Tensor:
        """
        Get the prediction mask sequence
        """
        # Pad masks to len + num_pred_steps - 1 with the last mask
        pad_len = num_pred_steps
        pad_mask = np.repeat(mask[-1:], pad_len, axis=0)
        mask = np.concatenate([mask, pad_mask], axis=0)
        pred_masks = []
        for i in range(len(mask) - num_pred_steps):
            pred_masks.append(mask[i:i+num_pred_steps])
        return np.stack(pred_masks, axis=0)

    def get_pred_action_seq(self, actions: np.array, num_pred_steps: int) -> torch.Tensor:
        """
        Get the prediction action sequence
        """
        # Pad actions to len + num_pred_steps - 1 with the last action
        pad_len = num_pred_steps - 1
        pad_action = np.repeat(actions[-1:], pad_len, axis=0)
        actions = np.concatenate([actions, pad_action], axis=0)
        pred_actions = []
        for i in range(len(actions) - pad_len):
            pred_actions.append(actions[i:i+num_pred_steps])
        return np.stack(pred_actions, axis=0)

    def get_history_obs_seq(self, observation: np.array, num_hist_steps: int) -> torch.Tensor:
        """
        Get the history observation sequence
        """
        # Pad observations to len + num_pred_steps - 1 with the first observation
        pad_len = num_hist_steps - 1
        pad_observation = np.repeat(observation[:1], pad_len, axis=0)
        observation = np.concatenate([pad_observation, observation], axis=0)
        hist_observation = []
        for i in range(len(observation) - pad_len):
            hist_observation.append(observation[i:i+num_hist_steps])
        return np.stack(hist_observation, axis=0)

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.task_keys)
    
    def __getitem__(self, index):
        """
        Get the subsequence of the dataset starting from index to index + sequence_length
        return a diction of shape 
        {
            "observation": torch.Tensor, shape (seq_length, 2, 224, 224)
            "action": torch.Tensor, shape (seq_length, num_pred_steps, 2)
        }
        """
        task_key = self.task_keys[index]
        # print(f"Loading task {task_key}")
        while True:
            try:
                hdf5_file = h5py.File(f"{task_key}", "r")
                hdf5_file = self.hdf5_to_dict(hdf5_file)
                break
            except:
                # print(f"Failed to load {task_key}.h5")
                # index = random.randint(0, len(self.task_keys)-1)
                # task_key = self.task_keys[index]
                # print(f"Trying to load {task_key}.h5")
                return None
        
        # Randomly select prompt trajectories
        traj_keys = [f"trajectory_{i}" for i in range(self.num_traj_per_task)]
        task_traj_key = self.rng.choice(traj_keys)
        selectable_prompt_keys = [key for key in traj_keys if key != task_traj_key]
        self.rng.shuffle(selectable_prompt_keys)
        if self.num_prompt_traj is None:
            num_prompt_traj = self.rng.randint(1, min(len(selectable_prompt_keys), self.max_n_prompts)+1)
        else:
            num_prompt_traj = min(self.num_prompt_traj, len(selectable_prompt_keys))
        prompt_traj_keys = self.rng.choice(selectable_prompt_keys, num_prompt_traj, replace=False)

        # Process trajectories
        steps = {}
        for traj_key in traj_keys:
            try:
                steps[traj_key] = self.process_h5(hdf5_file, traj_key)
            except:
                print(f"Failed to process {traj_key} in {task_key}.h5")
                return None
        has_mask = "masks" in steps[task_traj_key]
        
        # Calculate the length of the prompt trajectories
        len_of_prompt = 0
        prompt_eos_idx = [0]*self.num_traj_per_task
        # prompt_init_obs = [[0, 0]]*self.num_traj_per_task
        waypoints_idx = []
        for i, traj_key in enumerate(prompt_traj_keys):
            prompt_eos_idx[i] = len_of_prompt
            if self.waypoint_key != "":
                waypoints_idx.extend(hdf5_file[traj_key][self.waypoint_key].tolist())
                len_of_prompt += len(waypoints_idx)
            else:
                len_of_prompt += len(hdf5_file[traj_key][self.action_key])
            # prompt_init_obs[i] = hdf5_file[traj_key]["trajectory_label"][3:5]
        # Load the prompt trajectories
        prompt_observation = []
        multi_prompt_imgs = [[] for _ in range(len(self.img_keys))]
        prompt_action = []
        prompt_auxiliary = []
        for i, traj in enumerate(prompt_traj_keys):
            if self.has_obs:
                if self.waypoint_key != "":
                    observation = steps[traj]["observation"][waypoints_idx, -1]
                else:
                    observation = steps[traj]["observation"][:, -1]
                prompt_observation.append(observation)
            if self.has_img:
                for i, img_key in enumerate(self.img_keys):
                    if self.waypoint_key != "":
                        imgs = steps[traj][img_key][waypoints_idx, -1]
                    else:
                        imgs = steps[traj][img_key][:, -1]
                    multi_prompt_imgs[i].append(imgs)
            if self.waypoint_key != "":
                action = hdf5_file[traj][self.action_key][waypoints_idx]
            else:
                action = hdf5_file[traj][self.action_key]
            prompt_action.append(action)
            if self.auxiliary_key != "":
                auxiliary = hdf5_file[traj][self.auxiliary_key]
                if auxiliary.ndim == 1:
                    auxiliary = np.expand_dims(auxiliary, axis=0)
                    auxiliary = np.tile(auxiliary, (self.task_length, 1))
                prompt_auxiliary.append(auxiliary)
        
        if self.has_obs:
            prompt_observation = np.concatenate(prompt_observation, axis=0)
        prompt_action = np.concatenate(prompt_action, axis=0)
        if self.auxiliary_key != "":
            prompt_auxiliary = np.concatenate(prompt_auxiliary, axis=0)
        if self.has_img:
            for i, img_key in enumerate(self.img_keys):
                multi_prompt_imgs[i] = np.concatenate(multi_prompt_imgs[i], axis=0)
        prompt_padding_mask = torch.full((self.prompt_length,), False)
        # Pad the prediction trajectories to the same length
        if len_of_prompt < self.prompt_length:
            pad_len = self.prompt_length - len_of_prompt
            if self.has_obs:
                pad_observation = np.repeat(np.zeros_like(prompt_observation[-1:]), pad_len, axis=0)
                prompt_observation = np.concatenate([prompt_observation, pad_observation], axis=0)
            pad_action = np.repeat(np.zeros_like(prompt_action[-1:]), pad_len, axis=0)
            prompt_action = np.concatenate([prompt_action, pad_action], axis=0)
            if self.has_img:
                for i, img_key in enumerate(self.img_keys):
                    pad_img = np.repeat(np.zeros_like(multi_prompt_imgs[i][-1:]), pad_len, axis=0)
                    multi_prompt_imgs[i] = np.concatenate([multi_prompt_imgs[i], pad_img], axis=0)
            prompt_padding_mask[len_of_prompt:] = True
        elif len_of_prompt > self.prompt_length:
            trunc_len = len_of_prompt - self.prompt_length
            if self.has_obs:
                prompt_observation = prompt_observation[:-trunc_len]
            prompt_action = prompt_action[:-trunc_len]
            if self.has_img:
                for i, img_key in enumerate(self.img_keys):
                    multi_prompt_imgs[i] = multi_prompt_imgs[i][:-trunc_len]

        # Double check the length of the observation and action
        if self.has_obs:
            prompt_observation = prompt_observation[:self.prompt_length]
            prompt_observation = torch.from_numpy(prompt_observation)
        prompt_action = prompt_action[:self.prompt_length]
        prompt_action = torch.from_numpy(prompt_action)
        if self.has_img:
            for i, img_key in enumerate(self.img_keys):
                multi_prompt_imgs[i] = multi_prompt_imgs[i][:self.prompt_length]
                multi_prompt_imgs[i] = torch.from_numpy(multi_prompt_imgs[i])
        if self.auxiliary_key != "":
            prompt_auxiliary = prompt_auxiliary[:self.task_length]
            prompt_auxiliary = torch.from_numpy(prompt_auxiliary)

        # Pad the observation and action to the same length
        if self.has_obs:
            observation = steps[task_traj_key]["observation"]
        if self.has_img:
            multi_imgs = []
            for img_key in self.img_keys:
                multi_imgs.append(steps[task_traj_key][img_key])
        if has_mask:
            masks = steps[task_traj_key]["masks"]
        action = steps[task_traj_key]["action"]
        if self.auxiliary_key != "":
            auxiliary = steps[task_traj_key][self.auxiliary_key]
        if self.bg_key != "":
            backgrounds = steps[task_traj_key]["background"]
        task_padding_mask = torch.full((self.task_length,), False)
        len_of_task = len(action)
        if len_of_task < self.task_length:
            task_padding_mask[len_of_task:] = True
            pad_len = self.task_length - len_of_task
            if self.has_obs:
                pad_observation = np.repeat(np.zeros_like(observation[-1:]), pad_len, axis=0)
                observation = np.concatenate([observation, pad_observation], axis=0)
            if self.has_img:
                for i, img_key in enumerate(self.img_keys):
                    pad_img = np.repeat(np.zeros_like(multi_imgs[i][-1:]), pad_len, axis=0)
                    multi_imgs[i] = np.concatenate([multi_imgs[i], pad_img], axis=0)
            if has_mask:
                pad_mask = np.repeat(np.zeros_like(masks[-1:]), pad_len, axis=0)
                masks = np.concatenate([masks, pad_mask], axis=0)
            pad_action = np.repeat(np.zeros_like(action[-1:]), pad_len, axis=0)
            action = np.concatenate([action, pad_action], axis=0)
            if self.auxiliary_key != "":
                pad_auxiliary = np.repeat(np.zeros_like(auxiliary[-1:]), pad_len, axis=0)
                auxiliary = np.concatenate([auxiliary, pad_auxiliary], axis=0)
            if self.bg_key != "":
                pad_backgrounds = np.repeat(np.zeros_like(backgrounds[-1:]), pad_len, axis=0)
                backgrounds = np.concatenate([backgrounds, pad_backgrounds], axis=0)
        elif len_of_task > self.task_length:
            trunc_len = len_of_task - self.task_length
            if self.has_obs:
                observation = observation[:-trunc_len]
            if self.has_img:
                for i, img_key in enumerate(self.img_keys):
                    multi_imgs[i] = multi_imgs[i][:-trunc_len]
            if has_mask:
                masks = masks[:-trunc_len]
            action = action[:-trunc_len]
            if self.auxiliary_key != "":
                auxiliary = auxiliary[:-trunc_len]
            if self.bg_key != "":
                backgrounds = backgrounds[:-trunc_len]
        if self.has_obs:
            observation = observation[:self.task_length]
            observation = torch.from_numpy(observation)
        if self.has_img:
            for i, img_key in enumerate(self.img_keys):
                multi_imgs[i] = multi_imgs[i][:self.task_length]
                multi_imgs[i] = torch.from_numpy(multi_imgs[i])
        action = action[:self.task_length]
        if self.auxiliary_key != "":
            auxiliary = auxiliary[:self.task_length]
            auxiliary = torch.from_numpy(auxiliary)
        if self.bg_key != "":
            backgrounds = backgrounds[:self.task_length]
            backgrounds = torch.from_numpy(backgrounds)
        action = torch.from_numpy(action)
        if not self.enable_debug:
            data = {
                # "prompt_action_seq": prompt_action/torch.tensor(self.action_scale),
                "prompt_action_seq": (prompt_action - torch.tensor(self.action_min)) / (torch.tensor(self.action_max) - torch.tensor(self.action_min) + 1e-6),
                "prompt_eos_idx": torch.tensor(prompt_eos_idx),
                # "action": action/torch.tensor(self.action_scale),
                "action": (action - torch.tensor(self.action_min)) / (torch.tensor(self.action_max) - torch.tensor(self.action_min) + 1e-6),
                "prompt_padding_mask": prompt_padding_mask,
                "task_padding_mask": task_padding_mask,
            }
            if self.has_obs:
                # data["prompt_observation_seq"] = prompt_observation/torch.tensor(self.obs_scale)
                data["prompt_observation_seq"] = (prompt_observation - torch.tensor(self.obs_min)) / (torch.tensor(self.obs_max) - torch.tensor(self.obs_min) + 1e-6)
                # data["observation"] = observation/torch.tensor(self.obs_scale)
                data["observation"] = (observation - torch.tensor(self.obs_min)) / (torch.tensor(self.obs_max) - torch.tensor(self.obs_min) + 1e-6)
            if self.has_img:
                for i, img_key in enumerate(self.img_keys):
                    data[f"prompt_{img_key}"] = multi_prompt_imgs[i]
                    data[img_key] = multi_imgs[i]
            if has_mask:
                data["task_padding_mask"] = torch.tensor(masks)
            if self.auxiliary_key != "":
                # data["prompt_auxiliary"] = (prompt_auxiliary - torch.tensor(self.auxiliary_min)) / (torch.tensor(self.auxiliary_max) - torch.tensor(self.auxiliary_min))
                # data["auxiliary"] = (auxiliary - torch.tensor(self.auxiliary_min)) / (torch.tensor(self.auxiliary_max) - torch.tensor(self.auxiliary_min))
                data["auxiliary"] = auxiliary
            if self.bg_key != "":
                data["backgrounds"] = backgrounds/255
        else:
            data = {
                # "prompt_action_seq": prompt_action/torch.tensor(self.action_scale),
                "prompt_action_seq": (prompt_action - torch.tensor(self.action_min)) / (torch.tensor(self.action_max) - torch.tensor(self.action_min) + 1e-6),
                "prompt_eos_idx": torch.tensor(prompt_eos_idx),
                # "action": action/torch.tensor(self.action_scale),
                "action": (action - torch.tensor(self.action_min)) / (torch.tensor(self.action_max) - torch.tensor(self.action_min) + 1e-6),
                "prompt_padding_mask": prompt_padding_mask,
                "task_padding_mask": task_padding_mask,
                # debug info
                "num_prompt_traj": num_prompt_traj,
                "task_key": task_key,
                # "prompt_init_obs": torch.tensor(prompt_init_obs),
                # "label": torch.tensor(steps[task_traj_key]["label"]),
            }
            if len(self.label_keys) > 0:
                for key in self.label_keys:
                    data[key] = steps[task_traj_key][key]
            if self.has_obs:
                # data["prompt_observation_seq"] = prompt_observation/torch.tensor(self.obs_scale)
                data["prompt_observation_seq"] = (prompt_observation - torch.tensor(self.obs_min)) / (torch.tensor(self.obs_max) - torch.tensor(self.obs_min) + 1e-6)
                # data["observation"] = observation/torch.tensor(self.obs_scale)
                data["observation"] = (observation - torch.tensor(self.obs_min)) / (torch.tensor(self.obs_max) - torch.tensor(self.obs_min) + 1e-6)
            if self.has_img:
                for i, img_key in enumerate(self.img_keys):
                    data[f"prompt_{img_key}"] = multi_prompt_imgs[i]
                    data[img_key] = multi_imgs[i]
            if has_mask:
                data["task_padding_mask"] = torch.tensor(masks)
            if self.auxiliary_key != "":
                # data["prompt_auxiliary"] = (prompt_auxiliary - torch.tensor(self.auxiliary_min)) / (torch.tensor(self.auxiliary_max) - torch.tensor(self.auxiliary_min))
                # data["auxiliary"] = (auxiliary - torch.tensor(self.auxiliary_min)) / (torch.tensor(self.auxiliary_max) - torch.tensor(self.auxiliary_min))
                data["auxiliary"] = auxiliary
            if self.bg_key != "":
                data["backgrounds"] = backgrounds/255
        return data
