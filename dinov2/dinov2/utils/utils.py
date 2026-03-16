# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import random
import subprocess
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn


logger = logging.getLogger("dinov2")


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message

# class CosineScheduler(object):
#     def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
#         super().__init__()
#         self.base_value = base_value # Store base_value for state_dict
#         self.final_value = final_value
#         self.total_iters = total_iters
#         self.warmup_iters = warmup_iters # Store warmup_iters for state_dict
#         self.start_warmup_value = start_warmup_value # Store start_warmup_value for state_dict
#         self.freeze_iters = freeze_iters # Store freeze_iters for state_dict

#         self._build_schedule() # Call a helper to build the schedule

#     def _build_schedule(self):
#         freeze_schedule = np.zeros((self.freeze_iters))

#         warmup_schedule = np.linspace(self.start_warmup_value, self.base_value, self.warmup_iters)

#         # Ensure that iters is calculated correctly if total_iters - warmup_iters - freeze_iters is 0 or negative
#         cosine_iters_length = self.total_iters - self.warmup_iters - self.freeze_iters
#         if cosine_iters_length < 0:
#             cosine_iters_length = 0 # Prevent negative length for np.arange

#         iters = np.arange(cosine_iters_length)
        
#         # Avoid division by zero if cosine_iters_length is 0
#         if cosine_iters_length > 0:
#             schedule = self.final_value + 0.5 * (self.base_value - self.final_value) * (1 + np.cos(np.pi * iters / cosine_iters_length))
#         else:
#             schedule = np.array([]) # Empty array if no cosine phase

#         self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

#         # The assertion might fail if total_iters - warmup_iters - freeze_iters resulted in a negative length.
#         # Adjusted for robust schedule building, so len(self.schedule) might be less than total_iters if inputs are off.
#         # It's better to ensure `total_iters` is correctly defined by the user to avoid truncated schedules.
#         # assert len(self.schedule) == self.total_iters # Re-enable if you want strict adherence to total_iters for schedule length
#         if len(self.schedule) != self.total_iters:
#              print(f"Warning: Built schedule length ({len(self.schedule)}) does not match total_iters ({self.total_iters}). This might indicate incorrect input parameters.")


#     def __getitem__(self, it):
#         if it >= self.total_iters:
#             return self.final_value
#         else:
#             return self.schedule[it]

#     def state_dict(self):
#         """Returns the state of the scheduler as a dict."""
#         return {
#             "base_value": self.base_value,
#             "final_value": self.final_value,
#             "total_iters": self.total_iters,
#             "warmup_iters": self.warmup_iters,
#             "start_warmup_value": self.start_warmup_value,
#             "freeze_iters": self.freeze_iters,
#             # We don't save self.schedule directly as it can be rebuilt from parameters
#             # and may be very large.
#         }

#     def load_state_dict(self, state_dict):
#         """Loads the scheduler's state."""
#         self.base_value = state_dict["base_value"]
#         self.final_value = state_dict["final_value"]
#         self.total_iters = state_dict["total_iters"]
#         self.warmup_iters = state_dict["warmup_iters"]
#         self.start_warmup_value = state_dict["start_warmup_value"]
#         self.freeze_iters = state_dict["freeze_iters"]
        
#         # Rebuild the schedule with the loaded parameters
#         self._build_schedule()

class CosineScheduler(object):
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]



def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
