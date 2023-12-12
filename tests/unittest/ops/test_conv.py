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
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target


class ConvTestCase(unittest.TestCase):
    def _test_fp16(self, batch=1, copy_op=False):
        target = detect_target()
        X = Tensor(
            shape=[IntImm(batch), 224, 224, 3],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[256, 3, 3, 3], dtype="float16", name="input_1", is_input=True
        )
        OP = ops.conv2d(stride=1, pad=1, dilate=1)
        if copy_op:
            OP = ops.conv2d(**OP._get_op_attributes())
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", f"conv2d_{copy_op}")

        X_pt = torch.randn(batch, 3, 224, 224).cuda().half()
        W_pt = torch.randn(256, 3, 3, 3).cuda().half()
        Y_pt = torch.nn.functional.conv2d(X_pt, W_pt, padding=1)
        x = X_pt.permute((0, 2, 3, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 1)).contiguous()
        y = torch.empty_like(Y_pt).permute((0, 2, 3, 1)).contiguous()
        module.run_with_tensors({"input_0": x, "input_1": w}, [y])
        y_transpose = y.permute((0, 3, 1, 2))
        if target.name() == "cuda":
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))
        else:
<<<<<<< HEAD
            torch.testing.assert_close(Y_pt, y_transpose, atol=1.25e-1, rtol=1e-1)

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float32"), ("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_conv2d(self, dtype):
        self._test_conv(
            test_name=f"conv2d_{dtype}",
            dtype=dtype,
        )
        self._test_conv(
            copy_op=True,
            test_name=f"conv2d_{dtype}_copy_op",
            dtype=dtype,
        )

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float32"), ("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_conv1d(self, dtype):
        self._test_conv1d(dtype=dtype, bias=False)

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float32"), ("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_conv1d_bias(self, dtype):
        self._test_conv1d(dtype=dtype, bias=True)

    def _test_conv1d(self, dtype, bias):
        target = detect_target()
        batch = 4
        C_in = 80
        C_out = 512
        K = 3
        L = 28
        stride = 1
        padding = 1
        dilation = 1
        test_name = "test_conv1d"

        X_pt = get_random_torch_tensor([batch, C_in, L], dtype=dtype)
        W_pt = get_random_torch_tensor([C_out, C_in, K], dtype=dtype)
        bias_pt = get_random_torch_tensor([C_out], dtype=dtype) if bias else None
=======
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1.25e-1, rtol=1e-1))
>>>>>>> origin/navi3_rel_ver_1.0

    def test_fp16(self):
        self._test_fp16()
        self._test_fp16(copy_op=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()