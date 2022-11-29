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
Batched Gemm ROCM backend A[RowMajor], B[RowMajor], C[RowMajor], i.e.
c[b, m, n] = a[b, m, k] * b[b, k, n]
This is used for `ops.bmm_rrr_add`.
"""
import jinja2

from ... import registry
from . import bmm_common, common
from .layout import RRR

EXTRA_HEADER_TEMPLATE = jinja2.Template(
    """
#include "ck/tensor_operation/gpu/device/device_batched_gemm_multi_d_xdl.hpp"
"""
)


PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
{{indent}}                                static_cast<ck::half_t *>(in_ptr),
{{indent}}                                static_cast<ck::half_t *>(weight_ptr),
{% if "bias" in gemm_flag %}
{{indent}}                                std::array<const void*, 1>{static_cast<ck::half_t *>(bias_ptr)},
{% endif %}
{{indent}}                                static_cast<ck::half_t *>(out_ptr),
{{indent}}                                M,
{{indent}}                                N,
{{indent}}                                K,
{{indent}}                                B,
{{indent}}                                stride_a,
{{indent}}                                stride_b,
{% if "bias" in gemm_flag %}
{{indent}}                                std::array<ck::index_t, 1>{stride_c},
{% endif %}
{{indent}}                                stride_c,
{{indent}}                                M*K,
{{indent}}                                N*K,
{% if "bias" in gemm_flag %}
{{indent}}                                std::array<ck::index_t, 1>{M*N},
{% endif %}
{{indent}}                                M*N,
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{% if gemm_flag == "" %}
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{}
{% elif gemm_flag == "bias" %}
{{indent}}                                ck::tensor_operation::element_wise::Add{}
{% elif gemm_flag == "bias_relu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddRelu{}
{% elif gemm_flag == "bias_sigmoid" %}
{{indent}}                                ck::tensor_operation::element_wise::AddSigmoid{}
{% endif %}
"""
)

ARGS_PARSER_TEMPLATE = jinja2.Template(
    """
  int64_t B = std::atoi(argv[1]);
  int64_t M = std::atoi(argv[2]);
  int64_t N = std::atoi(argv[3]);
  int64_t K = std::atoi(argv[4]);

  int64_t a_dim0 = B;
  int64_t a_dim1 = M;
  int64_t a_dim2 = K;
  int64_t b_dim0 = B;
  int64_t b_dim1 = K;
  int64_t b_dim2 = N;
  int64_t c_dim0 = B;
  int64_t c_dim1 = M;
  int64_t c_dim2 = N;
"""
)


@registry.reg("rocm.bmm_rrr_add.config")
def bmm_config(func_attrs, dtype="float16"):
    """Extract (operation name, operation instance) pair from
    all operation candidates.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.

    Returns
    -------
    Dict
        Extracted (operation name, operation instance) pair
        from all operation candidates.
    """
    import ck_lib

    op_kind = ck_lib.library.GemmKind.BatchGemm
    extra_kind = ck_lib.library.TensorOperation.Add
    common.make_fproc_f16(func_attrs, RRR, op_kind, extra_kind)


@registry.reg("rocm.bmm_rrr_add.gen_profiler")
def bmm_gen_profiler(func_attrs, workdir, dim_info_dict):
    """Generates standalone executables for profiler.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    workdir : str
        Directory to store the generated outputs.
    dim_info_dict: Dict[str, DimInfo]
        Generated from bmm._extract_dims().
        Used to store mapping between dim_names to input / output tensor dims.
    """
    return bmm_common.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        args_parse=ARGS_PARSER_TEMPLATE.render(),
        dim_info_dict=dim_info_dict,
        gemm_flag="bias",
        problem_args_template=PROBLEM_ARGS_TEMPLATE,
        extra_header_template=EXTRA_HEADER_TEMPLATE,
    )


@registry.reg("rocm.bmm_rrr_add.gen_function")
def bmm_gen_function(func_attrs, exec_cond_template, dim_info_dict):
    """Generates function body.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    exec_cond_template : jinja2.Template
        Generates if statement to execute kernel.
    dim_info_dict: Dict[str, DimInfo]
        Generated from bmm._extract_dims().
        Used to store mapping between dim_names to input / output tensor dims.

    Returns
    -------
    str
        The rendered template of generated function body.
    """
    return bmm_common.gen_function(
        func_attrs,
        exec_cond_template,
        dim_info_dict,
        "bias",
        PROBLEM_ARGS_TEMPLATE,
        EXTRA_HEADER_TEMPLATE,
    )


@registry.reg("rocm.bmm_rrr_add.func_decl")
def bmm_gen_function_decl(func_attrs):
    """Generates function declarations.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.

    Returns
    -------
    str
        The rentered template of function declaration.
    """
    func_name = func_attrs["name"]
    return bmm_common.gen_function_decl(func_name=func_name, gemm_flag="bias")


@registry.reg("rocm.bmm_rrr_add.func_call")
def bmm_gen_function_call(func_attrs, indent="  "):
    """Generates function call.

    Parameters
    ----------
    func_attrs : Dict
        Stores the operation attributes.
    indent : str, optional
        Indent for codegen, target dependent e.g. C++, python, etc., by default "  ".

    Returns
    -------
    str
        The rendered template of generated function call.
    """
    return bmm_common.gen_function_call(func_attrs, indent, gemm_flag="bias")


@registry.reg("rocm.bmm_rrr_add.filter")
def bmm_function_filter(cfg, func_attrs, x_shape):
    """Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    ab_alignment:
        Input alignments.

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    return True
