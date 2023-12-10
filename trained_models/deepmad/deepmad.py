# Copyright (c) Alibaba, Inc. and its affiliates.

import ast
import os
import argparse
import torch
from .cnnnet import CnnNet


def get_backbone(
    output_structures,
    weights_path,
    network_id=0,
    classification=True,  # False for detetion
    descriptors_dimension=1_000,
):
    network_arch = output_structures["space_arch"]
    best_structures = output_structures["best_structures"]

    # If task type is classification, param num_classes is required
    out_indices = (1, 2, 3, 4) if not classification else (4,)
    backbone = CnnNet(
        structure_info=best_structures[network_id],
        out_indices=out_indices,
        num_classes=descriptors_dimension,
        classification=classification,
    )
    backbone.init_weights(weights_path)

    return backbone, network_arch
