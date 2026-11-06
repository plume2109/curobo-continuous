# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cuda.bindings.runtime as cudart
import numpy as np
from cuda.core import launch

import curobo._src.runtime as runtime
from curobo._src.util.logging import log_and_raise


def launch_kernel(kernel_name, stream, config, kernel, *kernel_args):
    """Launch a compiled kernel and check for launch errors.

    cuda.core passes a Python ``float`` as a C ``double``, while cuRobo kernels take ``float``
    scalars and would read only the low 4 bytes of the double. Python floats are converted to
    ``numpy.float32`` here, so callers can pass plain Python floats.
    """
    kernel_args = tuple(np.float32(a) if type(a) is float else a for a in kernel_args)
    launch(stream, config, kernel, *kernel_args)

    if runtime.debug:
        stream_handle = stream.handle
        cudart.cudaStreamSynchronize(stream_handle)

    error = cudart.cudaGetLastError()
    if error[0] != cudart.cudaError_t.cudaSuccess:
        log_and_raise(f"Failed to launch kernel: {kernel_name}. CUDA error: {error}")
