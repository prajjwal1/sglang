# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark the dsv3 fused-A GEMM: CuTe DSL (dsl), CUDA JIT (cudajit),
sgl_kernel AOT (aot), and torch.matmul.
"""

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(est_time=8, suite="base-b-kernel-benchmark-1-gpu-large")

GEMM_M = 2112
GEMM_K_LIST = [6144, 7168]

from sgl_kernel import dsv3_fused_a_gemm as aot_fn

from sglang.jit_kernel.cutedsl_dsv3_fused_a_gemm import dsv3_fused_a_gemm as dsl_fn
from sglang.jit_kernel.dsv3_fused_a_gemm import dsv3_fused_a_gemm as cudajit_fn


def _median_us(fn, *args) -> float:
    result = marker.do_bench(
        fn,
        input_args=args,
        use_cuda_graph=True,
        metrics=(0.5,),
        disable_log_bandwidth=True,
    )
    return result.times[0] * 1e6


def benchmark():
    torch.manual_seed(0)
    num_tokens = [1] if is_in_ci() else list(range(1, 17))

    for gemm_k in GEMM_K_LIST:
        weight = torch.randn(GEMM_M, gemm_k, dtype=torch.bfloat16, device="cuda")
        mat_b = weight.t()

        has_aot = False

        print(f"dsv3 fused-A GEMM  K={gemm_k} N={GEMM_M}  (us)")
        print(
            f"{'M':>4} {'aot':>9} {'dsl':>9} {'cudajit':>9} {'torch':>9} "
            f"{'aot/dsl':>9} {'cudajit/dsl':>11} {'torch/dsl':>9}"
        )
        for m in num_tokens:
            a = torch.randn(m, gemm_k, dtype=torch.bfloat16, device="cuda")
            dsl_us = _median_us(dsl_fn, a, mat_b)
            cudajit_us = _median_us(cudajit_fn, a, mat_b)
            torch_us = _median_us(torch.matmul, a, mat_b)
            aot_str = f"{'-':>9}"
            aot_ratio_str = f"{'-':>9}"
            if has_aot:
                aot_us = _median_us(aot_fn, a, mat_b)
                aot_str = f"{aot_us:>9.2f}"
                aot_ratio_str = f"{aot_us / dsl_us:>9.2f}"
            print(
                f"{m:>4} {aot_str} {dsl_us:>9.2f} {cudajit_us:>9.2f} {torch_us:>9.2f} "
                f"{aot_ratio_str} {cudajit_us / dsl_us:>11.2f} {torch_us / dsl_us:>9.2f}"
            )
        print()


if __name__ == "__main__":
    benchmark()
