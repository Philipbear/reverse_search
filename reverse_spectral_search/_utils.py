"""
This file contains code modified from the MSEntropy project
(https://github.com/YuanyueLi/MSEntropy)
Copyright Yuanyue Li 2023

Modified by Shipei Xing in 2024

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from numba import njit


@njit
def clean_peaks(peaks: np.ndarray,
                prec_mz: float,
                rel_int_threshold: float = 0.0,
                prec_mz_removal_da: float = 1.5,
                peak_transformation: str = 'sqrt',
                max_peak_num: int = 50):
    """
    Clean MS/MS peaks
    """

    peaks = peaks[np.bitwise_and(peaks[:, 0] > 0, peaks[:, 1] > 0)]

    if peaks.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Remove low intensity peaks
    peaks = peaks[peaks[:, 1] > rel_int_threshold * np.max(peaks[:, 1])]

    if peaks.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Remove peaks with mz > prec_mz - prec_mz_removal_da
    max_allowed_mz = prec_mz - prec_mz_removal_da
    peaks = peaks[peaks[:, 0] < max_allowed_mz]

    if peaks.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Maximum number of peaks
    if max_peak_num > 0 and len(peaks) > max_peak_num:
        # Sort the spectrum by intensity.
        peaks = peaks[np.argsort(peaks[:, 1])[-max_peak_num:]]

    # Sort peaks by m/z
    peaks = peaks[np.argsort(peaks[:, 0])]

    # Transform peak intensities
    if peak_transformation == 'sqrt':
        peaks[:, 1] = np.sqrt(peaks[:, 1])

    return np.asarray(peaks, dtype=np.float32)
