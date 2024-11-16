import numpy as np
from ms_entropy.spectra import tools
from ms_entropy.spectra.entropy import apply_weight_to_intensity


def calculate_reverse_entropy_similarity(
    peaks_a,
    peaks_b,
    ms2_tolerance_in_da: float = 0.02,
    ms2_tolerance_in_ppm: float = -1,
    clean_spectra: bool = True,
    **kwargs
):
    """Calculate the reverse entropy similarity between two spectra.
    peaks_a is the query spectrum, and peaks_b is the reference spectrum.

    Parameters
    ----------
    peaks_a : np.ndarray in shape (n_peaks, 2), np.float32 or list[list[float, float]]
        The first spectrum to calculate entropy similarity for. The first column is m/z, and the second column is intensity.

    peaks_b : np.ndarray in shape (n_peaks, 2), np.float32 or list[list[float, float]]
        The second spectrum to calculate entropy similarity for. The first column is m/z, and the second column is intensity.

    ms2_tolerance_in_da : float, optional
        The MS2 tolerance in Da. Defaults to 0.02. If this is set to a negative value, ms2_tolerance_in_ppm will be used instead.

    ms2_tolerance_in_ppm : float, optional
        The MS2 tolerance in ppm. Defaults to -1. If this is set to a negative value, ms2_tolerance_in_da will be used instead.

        **Note:** Either `ms2_tolerance_in_da` or `ms2_tolerance_in_ppm` must be positive. If both `ms2_tolerance_in_da` and `ms2_tolerance_in_ppm` are positive, `ms2_tolerance_in_ppm` will be used.

    clean_spectra : bool, optional
        Whether to clean the spectra before calculating entropy similarity. Defaults to True. **Only set this to False if the spectra have been preprocessed by the `clean_spectrum()` function!** Otherwise, the results will be incorrect. If the spectra are already cleaned, set this to False to save time.

    **kwargs : optional
        The arguments and keyword arguments to pass to function ``clean_spectrum()``.

        _

    Returns
    -------
    float
        The entropy similarity between the two spectra.
    """
    if clean_spectra:
        kwargs.update(
            {
                "min_ms2_difference_in_da": max(2 * ms2_tolerance_in_da, kwargs.get("min_ms2_difference_in_da", -1)),
                "min_ms2_difference_in_ppm": max(2 * ms2_tolerance_in_ppm, kwargs.get("min_ms2_difference_in_ppm", -1)),
            }
        )
        peaks_a = tools.clean_spectrum(peaks_a, **kwargs)
        peaks_b = tools.clean_spectrum(peaks_b, **kwargs)
    else:
        peaks_a = np.asarray(peaks_a, dtype=np.float32, order="C").reshape(-1, 2)
        peaks_b = np.asarray(peaks_b, dtype=np.float32, order="C").reshape(-1, 2)

    # Apply the weights to the peaks.
    peaks_a = apply_weight_to_intensity(peaks_a)
    peaks_b = apply_weight_to_intensity(peaks_b)

    return calculate_unweighted_reverse_entropy_similarity(
        peaks_a, peaks_b, ms2_tolerance_in_da=ms2_tolerance_in_da, ms2_tolerance_in_ppm=ms2_tolerance_in_ppm, clean_spectra=False
    )


def calculate_unweighted_reverse_entropy_similarity(
    peaks_a,
    peaks_b,
    ms2_tolerance_in_da: float = 0.02,
    ms2_tolerance_in_ppm: float = -1,
    clean_spectra: bool = True,
    **kwargs
):
    """Calculate the unweighted reverse entropy similarity between two spectra.
    peaks_a is the query spectrum, and peaks_b is the reference spectrum.

    Parameters
    ----------
    peaks_a : np.ndarray in shape (n_peaks, 2), np.float32 or list[list[float, float]]
        The query spectrum to calculate unweighted entropy similarity for. The first column is m/z, and the second column is intensity.

    peaks_b : np.ndarray in shape (n_peaks, 2), np.float32 or list[list[float, float]]
        The reference spectrum to calculate unweighted entropy similarity for. The first column is m/z, and the second column is intensity.

    ms2_tolerance_in_da : float, optional
        The MS2 tolerance in Da. Defaults to 0.02. If this is set to a negative value, ms2_tolerance_in_ppm will be used instead.

    ms2_tolerance_in_ppm : float, optional
        The MS2 tolerance in ppm. Defaults to -1. If this is set to a negative value, ms2_tolerance_in_da will be used instead.

        **Note:** Either `ms2_tolerance_in_da` or `ms2_tolerance_in_ppm` must be positive. If both `ms2_tolerance_in_da` and `ms2_tolerance_in_ppm` are positive, `ms2_tolerance_in_ppm` will be used.

    clean_spectra : bool, optional
        Whether to clean the spectra before calculating unweighted entropy similarity. Defaults to True. Only set this to False if the spectra have been preprocessed by the clean_spectrum() function! Otherwise, the results will be incorrect. If the spectra are already cleaned, set this to False to save time. If the spectra are in the list format, always set this to True or an error will be raised.

    **kwargs : optional
        The arguments and keyword arguments to pass to function ``clean_spectrum()``.

        _

    Returns
    -------
    float
        The unweighted entropy similarity between the two spectra.
    """
    if clean_spectra:
        kwargs.update(
            {
                "min_ms2_difference_in_da": max(2 * ms2_tolerance_in_da, kwargs.get("min_ms2_difference_in_da", -1)),
                "min_ms2_difference_in_ppm": max(2 * ms2_tolerance_in_ppm, kwargs.get("min_ms2_difference_in_ppm", -1)),
            }
        )
        peaks_a = tools.clean_spectrum(peaks_a, **kwargs)
        peaks_b = tools.clean_spectrum(peaks_b, **kwargs)
    else:
        peaks_a = np.asarray(peaks_a, dtype=np.float32, order="C").reshape(-1, 2)
        peaks_b = np.asarray(peaks_b, dtype=np.float32, order="C").reshape(-1, 2)

    if peaks_a.shape[0] == 0 or peaks_b.shape[0] == 0:
        return 0.0

    # Calculate the entropy similarity of the two spectra.
    a: int = 0
    b: int = 0
    peak_a_intensity: float = 0.0
    peak_b_intensity: float = 0.0
    peak_ab_intensity: float = 0.0
    entropy_similarity: float = 0.0

    max_allowed_mass_difference: float = ms2_tolerance_in_da

    matched_peak_a_intensity_array = []
    matched_peak_b_intensity_array = []
    while a < peaks_a.shape[0] and b < peaks_b.shape[0]:
        mass_difference: float = peaks_a[a, 0] - peaks_b[b, 0]
        if ms2_tolerance_in_ppm > 0:
            max_allowed_mass_difference = peaks_a[a, 0] * ms2_tolerance_in_ppm * 1e-6
        if mass_difference < -max_allowed_mass_difference:
            # This peak only exists in peaks_a.
            a += 1
        elif mass_difference > max_allowed_mass_difference:
            # This peak only exists in peaks_b.
            b += 1
        else:
            # This peak exists in both peaks_a and peaks_b.
            matched_peak_a_intensity_array.append(peaks_a[a, 1])
            matched_peak_b_intensity_array.append(peaks_b[b, 1])
            a += 1
            b += 1

    matched_peak_a_intensity_array = np.array(matched_peak_a_intensity_array)
    matched_peak_b_intensity_array = np.array(matched_peak_b_intensity_array)

    if len(matched_peak_a_intensity_array) == 0 or np.sum(matched_peak_a_intensity_array) <= 0:
        return 0.0

    # renormalize the intensities of matched peak_a
    matched_peak_a_intensity_array /= np.sum(matched_peak_a_intensity_array)

    # calculate the entropy similarity
    peak_ab_intensity_array = matched_peak_a_intensity_array + matched_peak_b_intensity_array

    entropy_similarity = peak_ab_intensity_array * np.log2(peak_ab_intensity_array) - \
                         matched_peak_a_intensity_array * np.log2(matched_peak_a_intensity_array) - \
                         matched_peak_b_intensity_array * np.log2(matched_peak_b_intensity_array)
    entropy_similarity = np.sum(entropy_similarity) / 2

    return entropy_similarity


if __name__ == "__main__":
    import ms_entropy as me

    peaks_query = np.array([[69, 8.0], [86, 100.0], [99, 50.0]], dtype=np.float32)
    peaks_reference = np.array([[41, 38.0], [69, 66.0], [86, 999.0]], dtype=np.float32)

    # Calculate unweighted entropy similarity.
    unweighted_similarity = me.calculate_unweighted_entropy_similarity(peaks_query, peaks_reference, ms2_tolerance_in_da=0.05)
    print(f"Unweighted entropy similarity: {unweighted_similarity}.")

    # Calculate entropy similarity.
    similarity = me.calculate_entropy_similarity(peaks_query, peaks_reference, ms2_tolerance_in_da=0.05)
    print(f"Entropy similarity: {similarity}.")

    # Calculate reverse unweighted entropy similarity.
    rev_unweighted_similarity = calculate_unweighted_reverse_entropy_similarity(peaks_query, peaks_reference, ms2_tolerance_in_da=0.05)
    print(f"Reverse unweighted entropy similarity: {rev_unweighted_similarity}.")

    # Calculate reverse entropy similarity.
    rev_similarity = calculate_reverse_entropy_similarity(peaks_query, peaks_reference, ms2_tolerance_in_da=0.05)
    print(f"Reverse entropy similarity: {rev_similarity}.")
