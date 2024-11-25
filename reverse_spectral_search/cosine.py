"""
This file contains code modified from the matchms project
(https://github.com/matchms/matchms)
Copyright matchms Team 2020

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

from typing import Tuple

import numba as nb
import numpy as np


@nb.njit
def find_matches(ref_spec_mz: np.ndarray, qry_spec_mz: np.ndarray,
                 tolerance: float, shift: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Find matching peaks between two spectra."""
    matches_idx1 = np.empty(len(ref_spec_mz) * len(qry_spec_mz), dtype=np.int64)
    matches_idx2 = np.empty_like(matches_idx1)
    match_count = 0
    lowest_idx = 0

    for peak1_idx in range(len(ref_spec_mz)):
        mz = ref_spec_mz[peak1_idx]
        low_bound = mz - tolerance
        high_bound = mz + tolerance

        for peak2_idx in range(lowest_idx, len(qry_spec_mz)):
            mz2 = qry_spec_mz[peak2_idx] + shift
            if mz2 > high_bound:
                break
            if mz2 < low_bound:
                lowest_idx = peak2_idx
            else:
                matches_idx1[match_count] = peak1_idx
                matches_idx2[match_count] = peak2_idx
                match_count += 1

    return matches_idx1[:match_count], matches_idx2[:match_count]


@nb.njit
def collect_peak_pairs(ref_spec: np.ndarray, qry_spec: np.ndarray, min_matched_peak: int, sqrt_transform: bool,
                       tolerance: float, shift: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find and score matching peak pairs between spectra."""

    if len(ref_spec) == 0 or len(qry_spec) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

    # Extract m/z values
    matches_idx1, matches_idx2 = find_matches(ref_spec[:, 0], qry_spec[:, 0], tolerance, shift)

    if len(matches_idx1) < min_matched_peak:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

    # Calculate scores for matches
    if sqrt_transform:
        scores = np.sqrt(ref_spec[matches_idx1, 1] * qry_spec[matches_idx2, 1]).astype(np.float32)
    else:
        scores = (ref_spec[matches_idx1, 1] * qry_spec[matches_idx2, 1]).astype(np.float32)

    # Sort by score descending
    sort_idx = np.argsort(-scores)
    return matches_idx1[sort_idx], matches_idx2[sort_idx], scores[sort_idx]


@nb.njit
def score_matches(matches_idx1: np.ndarray, matches_idx2: np.ndarray,
                  scores: np.ndarray, ref_spec: np.ndarray, qry_spec: np.ndarray,
                  sqrt_transform: bool, reverse: bool):
    """Calculate final similarity score from matching peaks."""

    # Use boolean arrays for tracking used peaks - initialized to False
    used1 = np.zeros(len(ref_spec), dtype=nb.boolean)
    used2 = np.zeros(len(qry_spec), dtype=nb.boolean)

    total_score = 0.0
    used_matches = 0

    # Find best non-overlapping matches
    for i in range(len(matches_idx1)):
        idx1 = matches_idx1[i]
        idx2 = matches_idx2[i]
        if not used1[idx1] and not used2[idx2]:
            total_score += scores[i]
            used1[idx1] = True
            used2[idx2] = True
            used_matches += 1

    if used_matches == 0:
        return 0.0, 0, 0.0

    # Calculate normalization factors
    if sqrt_transform:
        norm1 = np.sqrt(np.sum(np.sqrt(ref_spec[:, 1] * ref_spec[:, 1])))
    else:
        norm1 = np.sqrt(np.sum(ref_spec[:, 1] * ref_spec[:, 1]))

    # Sum intensities of matched peaks
    matched_intensities = np.zeros(used_matches, dtype=np.float32)
    match_idx = 0
    for i in range(len(qry_spec)):
        if used2[i]:
            matched_intensities[match_idx] = qry_spec[i, 1]
            match_idx += 1

    if reverse:
        if sqrt_transform:
            norm2 = np.sqrt(np.sum(np.sqrt(matched_intensities * matched_intensities)))
        else:
            norm2 = np.sqrt(np.sum(matched_intensities * matched_intensities))
    else:
        if sqrt_transform:
            norm2 = np.sqrt(np.sum(np.sqrt(qry_spec[:, 1] * qry_spec[:, 1])))
        else:
            norm2 = np.sqrt(np.sum(qry_spec[:, 1] * qry_spec[:, 1]))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0, used_matches, 0.0

    score = total_score / (norm1 * norm2)

    spec_usage = np.sum(matched_intensities) / np.sum(qry_spec[:, 1])

    return min(float(score), 1.0), used_matches, spec_usage


class CosineGreedy:
    """Calculate cosine similarity between mass spectra."""

    def __init__(self, tolerance: float = 0.1, reverse: bool = False):
        """Initialize with given parameters."""
        self.tolerance = np.float32(tolerance)
        self.reverse = reverse

    def pair(self, qry_spec, ref_spec,
             min_matched_peak: int = 1,
             analog_search: bool = False,
             sqrt_transform: bool = False,
             shift: float = 0.0):
        """
        Calculate similarity between two spectra.

        min_matched_peak: int, help for early stopping
        """

        if qry_spec.size == 0 or ref_spec.size == 0:
            return 0.0, 0, 0.0

        # normalize the intensity
        ref_spec[:, 1] /= np.max(ref_spec[:, 1])
        qry_spec[:, 1] /= np.max(qry_spec[:, 1])

        matches_idx1, matches_idx2, scores = collect_peak_pairs(
            ref_spec, qry_spec, min_matched_peak, sqrt_transform,
            self.tolerance, shift
        )

        if len(matches_idx1) == 0:
            return 0.0, 0, 0.0

        return score_matches(
            matches_idx1, matches_idx2, scores,
            ref_spec, qry_spec, sqrt_transform, self.reverse
        )


if __name__ == "__main__":
    peaks1 = np.array([
        [100., 0.7],
        [150., 0.2],
        [200., 0.1],
        [201., 0.2]
    ], dtype=np.float32)

    peaks2 = np.array([
        [105., 0.4],
        [150., 0.2],
        [190., 0.1],
        [200., 0.5]
    ], dtype=np.float32)

    # Example with reverse=False (cosine)
    cosine_standard = CosineGreedy(tolerance=0.05, reverse=False)
    score, n_matches, spec_usage = cosine_standard.pair(peaks1, peaks2)
    print(f"Standard Score: {score:.3f}, Matches: {n_matches}, Spec Usage: {spec_usage}")

    # Example with reverse=True (reverse cosine)
    cosine_reverse = CosineGreedy(tolerance=0.05, reverse=True)
    score, n_matches, spec_usage = cosine_reverse.pair(peaks1, peaks2)
    print(f"Reverse Score: {score:.3f}, Matches: {n_matches}, Spec Usage: {spec_usage}")
