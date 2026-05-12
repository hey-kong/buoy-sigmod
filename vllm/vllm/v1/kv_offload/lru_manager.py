# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.kv_offload.backend import Backend
from vllm.v1.kv_offload.policy_manager import PolicyOffloadingManager


class LRUOffloadingManager(PolicyOffloadingManager):
    """
    An OffloadingManager with a pluggable backend, which evicts blocks by LRU.
    """

    def __init__(
        self,
        backend: Backend,
        enable_events: bool = False,
        enable_hbm_residency_tracking: bool = False,
    ):
        super().__init__(
            backend=backend,
            enable_events=enable_events,
            eviction_policy="lru",
            enable_hbm_residency_tracking=enable_hbm_residency_tracking,
        )
