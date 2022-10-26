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
GEMM ROCM backend for A[RowMajor], B[ColumnMajor], C[RowMajor], i.e.
c[m, n] = fast_gelu(a[m, k] * b[n, k] + bias[n])
This is used for `torch.nn.functional.linear + swish`
When used for `linear`, need to set A->Data, B->Weight, C->Bias
"""
from ... import registry
from . import common
from .layout import RCR
import jinja2

# pylint: disable=C0415,W0613

PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
{{indent}}                                static_cast<ck::half_t *>(in_ptr),
{{indent}}                                static_cast<ck::half_t *>(weight_ptr),

{% if gemm_flag == "bias_permute" %}
{{indent}}                                static_cast<ck::half_t *>(bias_ptr),
{% elif gemm_flag == "bias_permute_m2n3" %}
{{indent}}                                std::array<const void*, 1>{static_cast<ck::half_t *>(bias_ptr)},
{% elif gemm_flag == "permute_m2n3" %}
{{indent}}                                {},
{% else %}
{% if "bias" in gemm_flag and not has_d0 %}
{{indent}}                                std::array<const void*, 1>{static_cast<ck::half_t *>(bias_ptr)},
{% elif has_d0 and not has_d1 %}
{{indent}}                                std::array<const void*, 2>{static_cast<ck::half_t *>(bias_ptr),
                                                                    static_cast<ck::half_t *>(d0_ptr)},
{% elif has_d1 %}
{{indent}}                                std::array<const void*, 3>{static_cast<ck::half_t *>(bias_ptr),
                                                                    static_cast<ck::half_t *>(d0_ptr),
                                                                    static_cast<ck::half_t *>(d1_ptr)},
{% endif %}
{% endif %}
{{indent}}                                static_cast<ck::half_t *>(out_ptr) + output_offset,
{% if gemm_flag not in ["permute_m2n3", "bias_permute_m2n3", "bias_permute_m3n2"]  %}
{{indent}}                                M,
{{indent}}                                N,
{{indent}}                                K,
{{indent}}                                stride_a,
{{indent}}                                stride_b,
{% endif %}
{% if gemm_flag == "bias_permute" %}
{{indent}}                                {M0, M1, M2, N0, N1, stride_D_M0, stride_D_M1, stride_D_M2, stride_D_N0, stride_D_N1},
{{indent}}                                {M0, M1, M2, N0, N1, stride_E_M0, stride_E_M1, stride_E_M2, stride_E_N0, stride_E_N1},
{% elif gemm_flag in ["permute_m2n3", "bias_permute_m2n3", "bias_permute_m3n2"]  %}
{{indent}}                                a_ms_ks_lengths,
{{indent}}                                a_ms_ks_strides,
{{indent}}                                b_ns_ks_lengths,
{{indent}}                                b_ns_ks_strides,
    {% if gemm_flag == "permute_m2n3"  %}
{{indent}}                                {},
{{indent}}                                {},
    {% else %}
{{indent}}                                std::array<std::vector<ck::index_t>, 1>{d_ms_ns_lengths},
{{indent}}                                std::array<std::vector<ck::index_t>, 1>{d_ms_ns_strides},
    {% endif %}
{{indent}}                                e_ms_ns_lengths,
{{indent}}                                e_ms_ns_strides,
{% else %}
{% if "bias" in gemm_flag and not has_d0 %}
{{indent}}                                std::array<ck::index_t, 1>{0},
{% elif has_d0 and not has_d1 %}
{{indent}}                                std::array<ck::index_t, 2>{0, static_cast<int>(stride_c)},
{% elif has_d1 %}
{{indent}}                                std::array<ck::index_t, 3>{0, static_cast<int>(stride_c), static_cast<int>(stride_c)},
{% endif %}
{{indent}}                                output_stride,
{% endif %}
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{% if gemm_flag == "" %}
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{}
{% elif gemm_flag == "permute_m2n3" %}
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{}
{% elif gemm_flag == "bias" or "bias_permute" in gemm_flag %}
{{indent}}                                ck::tensor_operation::element_wise::Add{}
{% elif gemm_flag == "bias_relu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddRelu{}
{% elif gemm_flag == "bias_fast_gelu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddFastGelu{}
{% elif gemm_flag == "bias_swish" %}
{{indent}}                                ck::tensor_operation::element_wise::AddHardswish{}
{% elif gemm_flag == "bias_tanh" %}
{{indent}}                                ck::tensor_operation::element_wise::AddTanh{}
{% elif gemm_flag == "bias_sigmoid" %}
{{indent}}                                ck::tensor_operation::element_wise::AddSigmoid{}
{% elif gemm_flag == "bias_add" %}
{{indent}}                                ck::tensor_operation::element_wise::AddAdd{}
{% elif gemm_flag == "bias_mul" %}
{{indent}}                                ck::tensor_operation::element_wise::AddMul{}
{% elif gemm_flag == "bias_mul_tanh" %}
{{indent}}                                ck::tensor_operation::element_wise::AddMulTanh{}
{% elif gemm_flag == "bias_add_relu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddAddRelu{}
{% elif gemm_flag == "bias_add_fast_gelu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddAddFastGelu{}
{% elif gemm_flag == "bias_sigmoid_mul" %}
{{indent}}                                ck::tensor_operation::element_wise::AddSigmoidMul{}
{% elif gemm_flag == "bias_sigmoid_mul_tanh" %}
{{indent}}                                ck::tensor_operation::element_wise::AddSigmoidMulTanh{}
{% elif gemm_flag == "bias_mul_add" %}
{{indent}}                                ck::tensor_operation::element_wise::AddMulAdd{}
{% elif gemm_flag == "bias_add_add" %}
{{indent}}                                ck::tensor_operation::element_wise::AddAddAdd{}
{% elif gemm_flag == "bias_add_add_relu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddAddAddRelu{}
{% endif %}
"""
)


OUTPUT_ADDR_CALCULATOR = jinja2.Template(
    """
  {% if not output_accessor.is_from_strided_tensor %}
  int64_t output_stride = {{stride_dim}};
  int64_t output_offset = 0;
  {% else %}
  int64_t output_stride = {{output_accessor.actual_total_elements_from_stride_dim}};
  int64_t output_offset = {{output_accessor.offset}};
  {% endif %}
    """
)




@registry.reg("rocm.gemm_rcr_bias_fast_gelu.config")
def gemm_config(func_attrs, dtype="float16"):
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
    import ck_lib  # noqa: F401

    op_kind = ck_lib.library.GemmKind.Gemm
    extra_kind = ck_lib.library.TensorOperation.AddFastGelu
    common.make_fproc_f16(func_attrs, RCR, op_kind, extra_kind)


@registry.reg("rocm.gemm_rcr_bias_fast_gelu.gen_profiler")
def gemm_gen_profiler(func_attrs, workdir, dim_info_dict):
    """Generates standalone executables for profiler.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    workdir : str
        Directory to store the generated outputs.
    dim_info_dict: Dict[str, DimInfo]
        Generated from gemm._extract_dims().
        Used to store mapping between dim_names to input / output tensor dims.
    """
    common.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        dim_info_dict=dim_info_dict,
        args_parse=RCR.args_parse,
        gemm_flag="bias_fast_gelu",
    )


@registry.reg("rocm.gemm_rcr_bias_fast_gelu.gen_function")
def gemm_gen_function(func_attrs, exec_cond_template, dim_info_dict):
    """Generates function body.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    exec_cond_template : jinja2.Template
        Generates if statement to execute kernel.
    dim_info_dict: Dict[str, DimInfo]
        Generated from gemm._extract_dims().
        Used to store mapping between dim_names to input / output tensor dims.

    Returns
    -------
    str
        The rendered template of generated function body.
    """
    return common.gen_function(
        func_attrs,
        exec_cond_template,
        dim_info_dict,
        "bias_fast_gelu",
        problem_args_template = PROBLEM_ARGS_TEMPLATE,
        output_addr_calculator=OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="N", output_accessor=func_attrs["output_accessors"][0]
        ),
    )


@registry.reg("rocm.gemm_rcr_bias_fast_gelu.func_decl")
def gemm_gen_function_decl(func_attrs):
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
    return common.gen_function_decl(func_name=func_name, gemm_flag="bias_fast_gelu")


@registry.reg("rocm.gemm_rcr_bias_fast_gelu.func_call")
def gemm_gen_function_call(func_attrs, indent="  "):
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
    return common.gen_function_call(func_attrs, indent, gemm_flag="bias_fast_gelu")


@registry.reg("rocm.gemm_rcr_bias_fast_gelu.filter")
def gemm_function_filter(cfg, func_attrs, x_shape):
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
