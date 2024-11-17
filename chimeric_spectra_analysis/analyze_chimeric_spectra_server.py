import pickle
import numpy as np
from numba import njit
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


@njit
def _get_chimeric_metrics(peaks, prec_mz, original_total_int, include_isotopes=False, isotope_indices=None):
    """Calculate chimeric spectrum metrics for ±2 Da window.

    Args:
        peaks: peaks array
        prec_mz: precursor m/z value
        original_total_int: total intensity of spectrum
        include_isotopes: whether to include precursor isotopes in calculation
        isotope_indices: indices of precursor and its isotope peaks

    Returns:
        tuple: (count, purity) for ±2 Da window
    """
    if not include_isotopes and isotope_indices is not None:
        # Create mask array without specifying dtype
        mask = np.ones(peaks.shape[0])
        # Set mask to 0 for isotope peaks
        for idx in isotope_indices:
            mask[idx] = 0
        # Use the mask to filter peaks
        peaks_to_analyze = peaks[mask > 0.5]
    else:
        peaks_to_analyze = peaks

    # Window is ±2 Da
    window_mask = np.abs(peaks_to_analyze[:, 0] - prec_mz) <= 2.0

    count = np.sum(window_mask)
    window_int = 0.0
    # Manual sum for peaks in window
    for i in range(len(peaks_to_analyze)):
        if window_mask[i]:
            window_int += peaks_to_analyze[i, 1]

    purity = 1.0 - window_int / original_total_int if original_total_int > 0 else 0.0

    return np.array([count, purity])


def process_mgf(mgf, mz_tol=0.05):
    try:
        metrics = {
            'total_spectra_count': 0,
            'empty_peak_count': 0,
            'prec_mz_unfound': 0,
            'nonprec_ion_count_4da_with_isotopes': [],  # Including isotopes
            'prec_mz_purity_4da_with_isotopes': [],  # Including isotopes
            'nonprec_ion_count_4da_no_isotopes': [],  # Excluding isotopes
            'prec_mz_purity_4da_no_isotopes': [],  # Excluding isotopes
        }

        for spec in iterate_mgf(mgf):
            metrics['total_spectra_count'] += 1
            prec_mz = float(spec['pepmass'])

            if len(spec['peaks']) == 0:
                metrics['empty_peak_count'] += 1
                metrics['prec_mz_unfound'] += 1
                continue

            peaks = np.array(spec['peaks'])
            original_total_int = np.sum(peaks[:, 1])

            prec_mz_idx = _get_precmz_idx(peaks[:, 0], prec_mz, mz_tol)

            if prec_mz_idx is None:
                metrics['prec_mz_unfound'] += 1
            else:
                # Find isotope peaks
                isotope_indices = _find_isotope_peaks(peaks, prec_mz_idx, mz_tol)

                # Calculate metrics with isotopes included
                window_metrics_with_isotopes = _get_chimeric_metrics(
                    peaks, prec_mz, original_total_int,
                    include_isotopes=True
                )

                # Calculate metrics with isotopes excluded
                window_metrics_no_isotopes = _get_chimeric_metrics(
                    peaks, prec_mz, original_total_int,
                    include_isotopes=False,
                    isotope_indices=isotope_indices
                )

                # Store metrics with isotopes
                metrics['nonprec_ion_count_4da_with_isotopes'].append(window_metrics_with_isotopes[0])
                metrics['prec_mz_purity_4da_with_isotopes'].append(window_metrics_with_isotopes[1])

                # Store metrics without isotopes
                metrics['nonprec_ion_count_4da_no_isotopes'].append(window_metrics_no_isotopes[0])
                metrics['prec_mz_purity_4da_no_isotopes'].append(window_metrics_no_isotopes[1])

        # purity, save as float16
        metrics['prec_mz_purity_4da_with_isotopes'] = np.array(metrics['prec_mz_purity_4da_with_isotopes'], dtype=np.float16)
        metrics['prec_mz_purity_4da_no_isotopes'] = np.array(metrics['prec_mz_purity_4da_no_isotopes'], dtype=np.float16)
        metrics['nonprec_ion_count_4da_with_isotopes'] = np.array(metrics['nonprec_ion_count_4da_with_isotopes'], dtype=np.int64)
        metrics['nonprec_ion_count_4da_no_isotopes'] = np.array(metrics['nonprec_ion_count_4da_no_isotopes'], dtype=np.int64)

        return metrics, os.path.basename(mgf)

    except Exception as e:
        print(f"Error processing {mgf}: {str(e)}")
        return None, os.path.basename(mgf)


@njit
def _find_isotope_peaks(peaks, prec_idx, mz_tol):
    """Find isotope peaks of precursor ion.

    Args:
        peaks: numpy array of peaks (m/z, intensity)
        prec_idx: index of precursor peak
        mz_tol: m/z tolerance for matching

    Returns:
        numpy array of indices including precursor and its isotope peaks
    """
    prec_mz = peaks[prec_idx, 0]
    isotope_indices = [prec_idx]

    # Look for up to 3 isotope peaks (M+1, M+2, M+3)
    for i in range(1, 4):
        expected_mz = prec_mz + i * 1.003355  # Isotope mass difference
        diffs = np.abs(peaks[:, 0] - expected_mz)
        min_diff_idx = np.argmin(diffs)

        # Check if within tolerance and has reasonable intensity
        if diffs[min_diff_idx] <= mz_tol:
            # Check if intensity is reasonable (typically decreasing pattern)
            if peaks[min_diff_idx, 1] <= peaks[prec_idx, 1]:
                isotope_indices.append(min_diff_idx)
            else:
                break  # Stop if intensity pattern is violated
        else:
            break  # Stop if no matching peak found

    return np.array(isotope_indices)


def process_mgf_wrapper(args):
    """Wrapper function to unpack arguments and save individual results"""
    mgf, mz_tol, output_path = args
    metrics, filename = process_mgf(mgf, mz_tol)

    if metrics is not None:
        # Create output filename from input filename
        output_filename = os.path.splitext(filename)[0] + '_metrics.pkl'
        output_filepath = os.path.join(output_path, output_filename)

        # Save individual metrics file
        with open(output_filepath, 'wb') as f:
            pickle.dump(metrics, f)

    return None


@njit
def _get_precmz_idx(mz_array, mz, tol):
    """Find the index of the peak closest to mz within tolerance range.

    Args:
        mz_array: 1D array of peak m/z values
        mz: target m/z value to search for
        tol: tolerance window (absolute difference)

    Returns:
        int or None: Index of closest matching peak or None if no peak within tolerance
    """
    diffs = np.abs(mz_array - mz)
    within_tol = diffs < tol

    if not np.any(within_tol):
        return None

    # Get the index of minimum difference among peaks within tolerance
    candidates = np.where(within_tol)[0]
    min_idx = candidates[np.argmin(diffs[candidates])]

    return min_idx


def read_mgf_spectrum(file_obj):
    """Read a single spectrum block from an open MGF file.

    Args:
        file_obj: An opened file object positioned at the start of a spectrum

    Returns:
        dict: Spectrum information containing title, pepmass, charge and peaks
        or None if end of file is reached. Empty peak lists are allowed.
    """
    spectrum = {
        'title': '',
        'pepmass': 0.0,
        'charge': '',
        'peaks': []
    }

    # Skip any empty lines before BEGIN IONS
    for line in file_obj:
        if line.strip() == 'BEGIN IONS':
            break
    else:  # EOF reached
        return None

    # Read spectrum metadata and peaks
    for line in file_obj:
        line = line.strip()

        if not line:  # Skip empty lines
            continue

        if line == 'END IONS':
            return spectrum  # Return spectrum regardless of peaks presence

        if line.startswith('TITLE='):
            spectrum['title'] = line[6:]
        elif line.startswith('PEPMASS='):
            spectrum['pepmass'] = float(line[8:].split()[0])  # Handle additional intensity value
        elif line.startswith('CHARGE='):
            spectrum['charge'] = line[7:]
        elif line and not line.startswith(('BEGIN', 'END')):  # Should be a peak line
            mz, intensity = line.split()
            spectrum['peaks'].append((float(mz), float(intensity)))

    return None


def iterate_mgf(mgf_path, buffer_size=8192):
    """Iterate through spectra in an MGF file efficiently using buffered reading.

    Args:
        mgf_path: Path to the MGF file
        buffer_size: Read buffer size in bytes

    Yields:
        dict: Spectrum information containing title, pepmass, charge and peaks
    """
    with open(mgf_path, 'r', buffering=buffer_size) as f:
        while True:
            spectrum = read_mgf_spectrum(f)
            if spectrum is None:
                break
            yield spectrum


def main(folder='/home/shipei/projects/cluster/unclustered_all',
         out_folder='/home/shipei/projects/revcos/search/search_results',
         mz_tol=0.05, n_jobs=None):
    """Main function to process all MGF files and save individual results."""
    os.makedirs(out_folder, exist_ok=True)

    all_mgfs = [x for x in os.listdir(folder) if x.endswith('.mgf')]

    # read DIA datasets list
    with open('dia_datasets.pkl', 'rb') as f:
        dia_datasets = pickle.load(f)

    all_mgfs = [x for x in all_mgfs if x.split('_')[0] not in dia_datasets]
    all_mgfs = [os.path.join(folder, x) for x in all_mgfs]

    # Determine number of processes
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    print(f"Processing {len(all_mgfs)} files using {n_jobs} processes...")

    # Process files in parallel
    with Pool(n_jobs) as pool:
        args = [(mgf, mz_tol, out_folder) for mgf in all_mgfs]
        results = list(tqdm(pool.imap_unordered(process_mgf_wrapper, args),
                            total=len(all_mgfs),
                            desc="Processing MGF files"))

    print(f"Results saved to {out_folder}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process MGF files and analyze metrics')
    parser.add_argument('--input_folder', '-i', type=str,
                        default='raw_data/',
                        help='Input folder containing MGF files')
    parser.add_argument('--out_folder', '-o', type=str,
                        default='results/',
                        help='Output folder for results')
    parser.add_argument('--mz_tol', type=float, default=0.05,
                        help='m/z tolerance')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='Number of parallel jobs')

    args = parser.parse_args()

    main(folder=args.input_folder,
         out_folder=args.out_folder,
         mz_tol=args.mz_tol,
         n_jobs=args.n_jobs)