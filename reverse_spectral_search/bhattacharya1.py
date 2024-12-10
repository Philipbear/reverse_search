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
                 tolerance: float) -> Tuple[np.ndarray, np.ndarray]:
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
            mz2 = qry_spec_mz[peak2_idx]
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
def collect_peak_pairs(ref_spec: np.ndarray, qry_spec: np.ndarray, min_matched_peak: int,
                       tolerance: float) -> Tuple[np.ndarray, np.ndarray]:
    """Find and score matching peak pairs between spectra."""
    if len(ref_spec) == 0 or len(qry_spec) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    # No need to copy arrays since we're only reading values
    matches_idx1, matches_idx2 = find_matches(ref_spec[:, 0], qry_spec[:, 0], tolerance)

    if len(matches_idx1) < min_matched_peak:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    # Calculate scores for matches
    scores = np.sqrt(ref_spec[matches_idx1, 1] * qry_spec[matches_idx2, 1]).astype(np.float32)

    # Sort by score descending
    sort_idx = np.argsort(-scores)
    return matches_idx1[sort_idx], matches_idx2[sort_idx]


@nb.njit
def score_matches(matches_idx1: np.ndarray, matches_idx2: np.ndarray,
                  ref_spec: np.ndarray, qry_spec: np.ndarray, penalty: float):
    """Calculate entropy similarity score from matching peaks."""

    # Use boolean arrays for tracking used peaks
    used1 = np.zeros(len(ref_spec), dtype=nb.boolean)
    used2 = np.zeros(len(qry_spec), dtype=nb.boolean)

    # Initialize arrays for matched intensities
    matched_a_intensities = np.empty(len(matches_idx1), dtype=np.float32)
    matched_b_intensities = np.empty(len(matches_idx1), dtype=np.float32)

    used_matches = 0

    # Find best non-overlapping matches
    for i in range(len(matches_idx1)):
        idx1 = matches_idx1[i]
        idx2 = matches_idx2[i]
        if not used1[idx1] and not used2[idx2]:
            matched_a_intensities[used_matches] = ref_spec[idx1, 1]
            matched_b_intensities[used_matches] = qry_spec[idx2, 1]
            used1[idx1] = True
            used2[idx2] = True
            used_matches += 1

    if used_matches == 0:
        return 0.0, 0

    # Trim arrays to used matches only
    matched_a_intensities = matched_a_intensities[:used_matches]
    matched_b_intensities = matched_b_intensities[:used_matches]

    # Normalize intensities
    sum_a = np.sum(ref_spec[:, 1])

    # For qry, add penalty to unmatched peaks
    sum_matched_qry = np.sum(matched_b_intensities)
    sum_b = sum_matched_qry + (np.sum(qry_spec[:, 1]) - sum_matched_qry) * (1 - penalty)

    if sum_a <= 0 or sum_b <= 0:
        return 0.0, used_matches

    matched_a_intensities /= sum_a
    matched_b_intensities /= sum_b

    # Calculate similarity
    score = np.sum(np.sqrt(matched_a_intensities * matched_b_intensities))
    if score > 1.0:
        score = 1.0
    if score < 0.0:
        score = 0.0
    score = 1 - np.arccos(score) / (np.pi / 2)

    return min(float(score), 1.0), used_matches


def bhattacharya1_similarity(qry_spec: np.ndarray, ref_spec: np.ndarray,
                             tolerance: float = 0.1,
                             min_matched_peak: int = 1,
                             penalty: float = 0.):
    """
    Calculate similarity between two spectra.

    Parameters
    ----------
    qry_spec: np.ndarray
        Query spectrum.
    ref_spec: np.ndarray
        Reference spectrum.
    tolerance: float
        Tolerance for m/z matching.
    min_matched_peak: int
        Minimum number of matched peaks.
    penalty: float
        Penalty for unmatched peaks. If set to 0, traditional cosine score; if set to 1, traditional reverse cosine score.
    """
    tolerance = np.float32(tolerance)
    penalty = np.float32(penalty)

    if qry_spec.size == 0 or ref_spec.size == 0:
        return 0.0, 0

    # normalize the intensity
    ref_spec[:, 1] /= np.sum(ref_spec[:, 1])
    qry_spec[:, 1] /= np.sum(qry_spec[:, 1])

    matches_idx1, matches_idx2 = collect_peak_pairs(
        ref_spec, qry_spec, min_matched_peak, tolerance
    )

    if len(matches_idx1) == 0:
        return 0.0, 0

    return score_matches(
        matches_idx1, matches_idx2, ref_spec, qry_spec, penalty
    )


if __name__ == "__main__":
    peaks1 = np.array([[69, 8.0], [86, 100.0], [99, 50.0]], dtype=np.float32)

    peaks2 = np.array([[41, 38.0], [69, 66.0], [86, 999.0]], dtype=np.float32)

    # Example with standard bhattacharya1
    score, n_matches = bhattacharya1_similarity(peaks1, peaks2, tolerance=0.05, penalty=0)
    print(f"Standard Score: {score:.3f}, Matches: {n_matches}")

    # Example with enhanced reverse bhattacharya1
    score, n_matches = bhattacharya1_similarity(peaks1, peaks2, tolerance=0.05, penalty=0.6)
    print(f"Reverse Score: {score:.3f}, Matches: {n_matches}")

    # Example with traditional reverse bhattacharya1
    score, n_matches = bhattacharya1_similarity(peaks1, peaks2, tolerance=0.05, penalty=1)
    print(f"Reverse Score: {score:.3f}, Matches: {n_matches}")
