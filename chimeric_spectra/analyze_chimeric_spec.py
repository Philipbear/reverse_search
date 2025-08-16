import os
import numpy as np
from pyteomics import mzml
from numba import njit
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


@njit
def _analyze_ms1(ms1_peaks, precursor_mz):
    """
    Analyze MS1 spectrum to find precursor peak and calculate purity metrics

    Args:
        ms1_peaks (np.ndarray): 2D array of [mz, intensity] pairs
        precursor_mz (float): Target precursor m/z value

    Returns:
        tuple: (ion_count, precursor_purity)
            - ion_count: Number of peaks within ±0.5 Da window
            - precursor_purity: Intensity ratio of precursor peak to total in window
    """
    # Initialize variables
    ion_count = 0
    total_intensity = 0.0
    closest_intensity = 0.0
    min_diff = 0.05

    # First pass: find the closest peak within 0.05 Da
    for mz, intensity in ms1_peaks:
        diff = abs(mz - precursor_mz)
        if diff <= 0.05 and diff < min_diff:
            min_diff = diff
            closest_intensity = intensity

    # Second pass: calculate ion count and total intensity in 0.5 Da window
    for mz, intensity in ms1_peaks:
        if abs(mz - precursor_mz) <= 0.5:
            ion_count += 1
            total_intensity += intensity

    # Calculate precursor purity
    prec_purity = closest_intensity / total_intensity if total_intensity > 0 else None

    return ion_count, prec_purity


def process_mzml(mzml_file):
    """Process an mzML file and return ion count and precursor purity metrics"""
    try:
        reader = mzml.MzML(mzml_file)

        ms1_peaks = None
        ion_count_ls = []
        prec_purity_ls = []

        for spectrum in reader:
            if spectrum['ms level'] == 1:
                mz_array = np.array(spectrum['m/z array'])
                intensity_array = np.array(spectrum['intensity array'])
                ms1_peaks = np.column_stack((mz_array, intensity_array))

            if spectrum['ms level'] == 2:
                precursor_mz = float(
                    spectrum['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z'])

                if ms1_peaks is not None and precursor_mz is not None:
                    ion_count, prec_purity = _analyze_ms1(ms1_peaks, precursor_mz)
                    ion_count_ls.append(ion_count)
                    prec_purity_ls.append(prec_purity)

        return mzml_file, ion_count_ls, prec_purity_ls

    except Exception as e:
        print(f"Error processing {mzml_file}: {str(e)}")
        return mzml_file, [], []


def main(input_folder, output_folder, name, n_processes=None):
    """
    Process all mzML files in the input folder and save results to output folder

    Args:
        input_folder (str): Path to folder containing mzML files
        output_folder (str): Path to folder where results will be saved
        n_processes (int): Number of processes to use (default: CPU count - 1)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Find all mzML files
    mzml_files = os.listdir(input_folder)
    mzml_files = [os.path.join(input_folder, f) for f in mzml_files if f.endswith('.mzML')]

    # Set up multiprocessing
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    print(f"Processing {len(mzml_files)} files using {n_processes} processes...")

    # Process files in parallel
    all_ion_counts = []
    all_purities = []
    file_names = []

    with Pool(n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_mzml, mzml_files),
            total=len(mzml_files),
            desc="Processing mzML files"
        ))

    # Collect results
    for file_path, ion_counts, purities in results:
        if ion_counts:  # Only include if results were generated
            file_names.append(str(file_path))
            all_ion_counts.extend(ion_counts)
            all_purities.extend(purities)

    # Convert to numpy arrays
    all_ion_counts = np.array(all_ion_counts)
    all_purities = np.array(all_purities)

    # Save results
    np.save(os.path.join(output_folder, f"{name}_ion_counts.npy"), all_ion_counts)
    np.save(os.path.join(output_folder, f"{name}_precursor_purities.npy"), all_purities)

    print(f"\nResults saved to {output_folder}")
    print(f"Processed {len(file_names)} files")
    print(f"Total spectra analyzed: {len(all_ion_counts)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process mzML files to analyze MS1 spectra")
    parser.add_argument("--input_folder", '-i', help="Folder containing mzML files")
    parser.add_argument("--output_folder", '-o', help="Folder to save results")
    parser.add_argument("--name", '-n', type=str, help="Name of the output file")
    parser.add_argument("--processes", type=int, help="Number of processes to use")

    args = parser.parse_args()

    main(args.input_folder, args.output_folder, args.name, args.processes)