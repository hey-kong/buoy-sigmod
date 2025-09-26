# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import List

from transformers import AutoConfig


@dataclass
class QuantizationSpec:
    start_layer: int
    end_layer: int
    bins: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)


@dataclass
class CacheGenConfig:
    nlayers: int
    kspecs: List[QuantizationSpec]
    vspecs: List[QuantizationSpec]

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    @staticmethod
    def from_model_name(model_name: str) -> "CacheGenConfig":
        try:
            config = AutoConfig.from_pretrained(model_name)
            # Default name caught by num_hidden_layers
            if config.num_hidden_layers is None:
                raise ValueError(
                    f"num_hidden_layers is None for model {model_name}"
                )
            if config.num_hidden_layers < 10:
                return CacheGenConfig(
                    nlayers=config.num_hidden_layers,
                    kspecs=[
                        QuantizationSpec(
                            start_layer=0,
                            end_layer=config.num_hidden_layers,
                            bins=65536,  # float16
                        ),
                    ],
                    vspecs=[
                        QuantizationSpec(
                            start_layer=0,
                            end_layer=config.num_hidden_layers,
                            bins=65536,  # float16
                        ),
                    ],
                )
            else:
                return CacheGenConfig(
                    nlayers=config.num_hidden_layers,
                    kspecs=[
                        QuantizationSpec(start_layer=0, end_layer=10, bins=65536),  # float16
                        QuantizationSpec(
                            start_layer=10,
                            end_layer=config.num_hidden_layers,
                            bins=256,  # int8
                        ),
                    ],
                    vspecs=[
                        QuantizationSpec(start_layer=0, end_layer=2, bins=65536),  # float16
                        QuantizationSpec(
                            start_layer=2,
                            end_layer=config.num_hidden_layers,
                            bins=256,  # int8
                        ),
                    ],
                )
        except Exception as e:
            raise ValueError(
                f"Model {model_name} not supported by CacheGenConfig"
            ) from e
