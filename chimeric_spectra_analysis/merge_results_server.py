import os
import pickle
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def process_file(pkl_file):
    """Process a single pickle file and return its metrics"""
    try:
        with open(pkl_file, 'rb') as f:
            metrics = pickle.load(f)
        return metrics
    except Exception as e:
        print(f"Error processing {pkl_file}: {str(e)}")
        return None


def summarize_results_parallel(metrics_folder, output_folder, n_jobs=None):
    """Summarize all pickle files containing metrics in parallel.

    Args:
        metrics_folder: Folder containing individual *_metrics.pkl files
        output_folder: Folder to save summarized results
        n_jobs: Number of parallel jobs (defaults to cpu_count-1)
    """
    os.makedirs(output_folder, exist_ok=True)

    # Get all metrics files
    pkl_files = [f for f in os.listdir(metrics_folder) if f.endswith('_metrics.pkl')]
    pkl_files = [os.path.join(metrics_folder, f) for f in pkl_files]

    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    # Process all files in parallel
    print(f"Processing {len(pkl_files)} files using {n_jobs} processes...")
    with Pool(n_jobs) as pool:
        all_metrics = list(tqdm(
            pool.imap_unordered(process_file, pkl_files),
            total=len(pkl_files),
            desc="Loading metrics files"
        ))

    # Remove None values from failed processes
    all_metrics = [m for m in all_metrics if m is not None]

    if not all_metrics:
        print("No valid metrics files found!")
        return

    # Initialize containers for different types of metrics
    numeric_totals = {}
    array_metrics = {}

    # First pass: identify metric types and initialize containers
    for metric_name, value in all_metrics[0].items():
        if isinstance(value, (int, float)):
            numeric_totals[metric_name] = 0
        elif isinstance(value, np.ndarray):
            array_metrics[metric_name] = []

    # Second pass: accumulate metrics
    print("Aggregating metrics...")
    for metrics in all_metrics:
        # Sum numeric metrics
        for metric_name in numeric_totals:
            numeric_totals[metric_name] += metrics[metric_name]

        # Collect arrays
        for metric_name in array_metrics:
            array_metrics[metric_name].append(metrics[metric_name])

    # Save numeric totals as CSV
    print("Saving numeric totals...")
    pd.DataFrame([numeric_totals]).to_csv(
        os.path.join(output_folder, 'numeric_totals.csv'),
        index=False
    )

    # Save array metrics
    print("Saving array metrics...")
    for metric_name, arrays in array_metrics.items():
        # Concatenate all arrays
        combined_array = np.concatenate(arrays)

        # Save to compressed npz file
        output_path = os.path.join(output_folder, f'{metric_name}.npz')
        np.savez_compressed(output_path, data=combined_array)

    print(f"Results saved to {output_folder}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Summarize MGF metrics results')
    parser.add_argument('--metrics_folder', '-i', type=str,
                        default='results/',
                        help='Input folder containing metrics pickle files')
    parser.add_argument('--output_folder', '-o', type=str,
                        default='summary/',
                        help='Output folder for summarized results')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='Number of parallel jobs')

    args = parser.parse_args()

    summarize_results_parallel(
        metrics_folder=args.metrics_folder,
        output_folder=args.output_folder,
        n_jobs=args.n_jobs
    )