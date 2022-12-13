#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
Remove useless operators from a sorted_graph.
"""
from typing import List

from aitemplate.utils import graph_utils
from aitemplate.compiler.transform import transform_utils

from ..base import Tensor


def remove_reshape_ops(sorted_graph: List[Tensor], workdir=None) -> None:
    """Remove ops which are not src operators of tensors in the input sorted_graph."""

    ops = graph_utils.get_sorted_ops(sorted_graph)
    for op in ops:
        if op._attrs["op"] not in ["reshape", "flatten"]:
            continue

        if op._attrs["unknown_idx"] != -1:
            continue
        
        assert len(op._attrs["inputs"]) >= 1, "reshape must have at least 1 input"
        reshape_input = op._attrs["inputs"][0]
        if reshape_input._attrs["is_input"]:
            continue
        assert len(op._attrs["outputs"]) == 1, "reshape must only have 1 output"
        reshape_output = op._attrs["outputs"][0]

        is_supported = True
        for dst in reshape_output.dst_ops():
            if not dst._attrs.get("input_accessors"):
                is_supported = False
                break
            
        if not is_supported:
            continue

        for dst in reshape_output.dst_ops():
            transform_utils.replace_tensor_for_op(dst, reshape_output, reshape_input)

        transform_utils.remove_tensor_from_sorted_graph(reshape_output)

    return transform_utils.sanitize_sorted_graph(sorted_graph)
