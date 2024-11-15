import logging
from typing import Tuple

import numpy as np
from matchms.similarity.BaseSimilarity import BaseSimilarity
from matchms.similarity.spectrum_similarity_functions import find_matches
from matchms.typing import SpectrumType

logger = logging.getLogger("matchms")


class ReverseCosine(BaseSimilarity):
    """
    Calculate 'reverse cosine score' between mass spectra.
    query spectrum goes first, reference spectrum goes second.
    """
    # Set key characteristics as class attributes
    is_commutative = False

    # Set output data type, e.g. ("score", "float") or [("score", "float"), ("matches", "int")]
    # score_datatype = [("score", np.float64), ("matches", "int")]

    def __init__(self, tolerance: float = 0.1, mz_power: float = 0.0,
                 intensity_power: float = 1.0):
        """
        Parameters
        ----------
        tolerance:
            Peaks will be considered a match when <= tolerance apart. Default is 0.1.
        mz_power:
            The power to raise mz to in the cosine function. The default is 0, in which
            case the peak intensity products will not depend on the m/z ratios.
        intensity_power:
            The power to raise intensity to in the cosine function. The default is 1.
        """
        self.tolerance = tolerance
        self.mz_power = mz_power
        self.intensity_power = intensity_power

    def pair(self, query: SpectrumType, reference: SpectrumType):
        """Calculate reverse cosine score between two spectra.

        Parameters
        ----------
        query
            Single query spectrum.
        reference
            Single reference spectrum.

        Returns
        -------

        Tuple with cosine score and number of matched peaks.
        """

        def get_matching_pairs():
            """Find all pairs of peaks that match within the given tolerance."""
            matching_pairs = collect_peak_pairs_reverse(spec1, spec2, self.tolerance, shift=0.0,
                                                        mz_power=self.mz_power,
                                                        intensity_power=self.intensity_power)
            if matching_pairs is None:
                matching_pairs = np.zeros((0, 4))
            if matching_pairs.shape[0] > 0:
                matching_pairs = matching_pairs[np.argsort(matching_pairs[:, 2])[::-1], :]
            return matching_pairs

        spec1 = reference.peaks.to_numpy
        spec2 = query.peaks.to_numpy

        matching_pairs = get_matching_pairs()

        if matching_pairs.shape[0] == 0:
            return 0., 0

        return score_best_matches_reverse(matching_pairs, spec1, spec2,
                                          self.mz_power, self.intensity_power)


def collect_peak_pairs_reverse(spec1: np.ndarray, spec2: np.ndarray,
                               tolerance: float, shift: float = 0, mz_power: float = 0.0,
                               intensity_power: float = 1.0):
    """Find matching pairs between two spectra.

    Args
    ----
    spec1:
        Spectrum peaks and intensities as numpy array.
    spec2:
        Spectrum peaks and intensities as numpy array.
    tolerance
        Peaks will be considered a match when <= tolerance appart.
    shift
        Shift spectra peaks by shift. The default is 0.
    mz_power:
        The power to raise mz to in the cosine function. The default is 0, in which
        case the peak intensity products will not depend on the m/z ratios.
    intensity_power:
        The power to raise intensity to in the cosine function. The default is 1.

    Returns
    -------
    matching_pairs : numpy array
        Array of found matching peaks.
    """
    matches = find_matches(spec1[:, 0], spec2[:, 0], tolerance, shift)
    idx1 = [x[0] for x in matches]
    idx2 = [x[1] for x in matches]
    if len(idx1) == 0:
        return None
    matching_pairs = []
    for i, idx in enumerate(idx1):
        power_prod_spec1 = (spec1[idx, 0] ** mz_power) * (spec1[idx, 1] ** intensity_power)
        power_prod_spec2 = (spec2[idx2[i], 0] ** mz_power) * (spec2[idx2[i], 1] ** intensity_power)
        matching_pairs.append([idx, idx2[i], power_prod_spec1 * power_prod_spec2, power_prod_spec2])

    return np.array(matching_pairs.copy())  # idx1, idx2, int_spec1 * int_spec2, int_spec2


def score_best_matches_reverse(matching_pairs: np.ndarray, spec1: np.ndarray,
                               spec2: np.ndarray, mz_power: float = 0.0,
                               intensity_power: float = 1.0):
    """Calculate cosine-like score by multiplying matches. Does require a sorted
    list of matching peaks (sorted by intensity product)."""
    score = float(0.0)
    used_matches = int(0)
    used1 = set()
    used2 = set()
    spec2_matched_mz = np.array([], dtype=np.float64)
    spec2_matched_intensity = np.array([], dtype=np.float64)
    for i in range(matching_pairs.shape[0]):
        if not int(matching_pairs[i, 0]) in used1 and not int(matching_pairs[i, 1]) in used2:
            score += matching_pairs[i, 2]
            used1.add(int(matching_pairs[i, 0]))  # Every peak can only be paired once
            used2.add(int(matching_pairs[i, 1]))  # Every peak can only be paired once
            used_matches += 1

            spec2_idx = int(matching_pairs[i, 1])
            spec2_matched_mz = np.append(spec2_matched_mz, spec2[spec2_idx, 0])
            spec2_matched_intensity = np.append(spec2_matched_intensity, spec2[spec2_idx, 1])

    # Normalize score:
    spec1_power = np.power(spec1[:, 0], mz_power) * np.power(spec1[:, 1], intensity_power)
    spec2_power = np.power(spec2_matched_mz, mz_power) * np.power(spec2_matched_intensity, intensity_power)

    spec1_norm = np.sqrt(np.sum(np.power(spec1_power, 2)))
    spec2_norm = np.sqrt(np.sum(np.power(spec2_power, 2)))
    norm_product = spec1_norm * spec2_norm
    score /= norm_product

    return score, used_matches


if __name__ == "__main__":
    import numpy as np
    from matchms import Spectrum

    spectrum_1 = Spectrum(mz=np.array([100., 150, 200, 201]),
                          intensities=np.array([0.7, 0.2, 0.1, 0.2]),
                          metadata={"precursor_mz": 200.0})
    spectrum_2 = Spectrum(mz=np.array([105., 150, 190, 200]),
                          intensities=np.array([0.4, 0.2, 0.1, 0.5]),
                          metadata={"precursor_mz": 205.0})

    reverse_cosine = ReverseCosine(tolerance=0.05, mz_power=0.0, intensity_power=1.0)
    score, matched_peak = reverse_cosine.pair(spectrum_1, spectrum_2)  # query, reference

    print(score, matched_peak)
