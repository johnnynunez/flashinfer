"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Global compilation context management for FlashInfer.
"""

import os
import torch
import logging

logger = logging.getLogger(__name__)


class CompilationContext:
    COMMON_NVCC_FLAGS = [
        "-DFLASHINFER_ENABLE_FP8_E8M0",
        "-DFLASHINFER_ENABLE_FP4_E2M1",
    ]

    def __init__(self):
        self.TARGET_CUDA_ARCHS = set()
        if "FLASHINFER_CUDA_ARCH_LIST" in os.environ:
            for arch in os.environ["FLASHINFER_CUDA_ARCH_LIST"].split(" "):
                major, minor = arch.split(".")
                major = int(major)
                self.TARGET_CUDA_ARCHS.add((int(major), str(minor)))
        else:
            try:
                for device in range(torch.cuda.device_count()):
                    major, minor = torch.cuda.get_device_capability(device)
                    if major >= 10:
                        # Since Blackwell (SM10x) NVCC has two ISA extension
                        # suffixes (see NVIDIA blog on CUDA 12.9 family-specific
                        # architecture features):
                        #
                        #   "a" = architecture-specific (accelerator): cubin runs
                        #         ONLY on that exact compute capability.
                        #   "f" = family-specific (full-chip): cubin runs on all
                        #         GPUs with the same major CC (>= minor).
                        #
                        # torch.cuda.get_device_capability() can't distinguish
                        # a vs f, so we emit both gencode targets to produce a
                        # fat binary that loads on either variant
                        self.TARGET_CUDA_ARCHS.add((int(major), str(minor) + "a"))
                        if minor <= 1:
                            self.TARGET_CUDA_ARCHS.add(
                                (int(major), str(minor) + "f")
                            )
                    elif major >= 9:
                        # SM90 (Hopper): only "a" (architecture-specific); the
                        # "f" family-specific suffix was not introduced until
                        # Blackwell / CUDA 12.9.
                        self.TARGET_CUDA_ARCHS.add((int(major), str(minor) + "a"))
                    else:
                        self.TARGET_CUDA_ARCHS.add((int(major), str(minor)))
            except Exception as e:
                logger.warning(f"Failed to get device capability: {e}.")

    def get_nvcc_flags_list(
        self, supported_major_versions: list[int] = None
    ) -> list[str]:
        if supported_major_versions:
            supported_cuda_archs = [
                major_minor_tuple
                for major_minor_tuple in self.TARGET_CUDA_ARCHS
                if major_minor_tuple[0] in supported_major_versions
            ]
        else:
            supported_cuda_archs = self.TARGET_CUDA_ARCHS
        if len(supported_cuda_archs) == 0:
            raise RuntimeError(
                f"No supported CUDA architectures found for major versions {supported_major_versions}."
            )
        return [
            f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
            for major, minor in supported_cuda_archs
        ] + self.COMMON_NVCC_FLAGS
