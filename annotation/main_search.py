import os

import numpy as np
import pandas as pd

from _utils import clean_peaks
from cosine import cosine_similarity
from entropy import entropy_similarity
from file_io import batch_process_queries, iterate_gnps_lib_mgf

QRY_BATCH_SIZE = 100000  # number of query spectra to process at once


def main_batch(gnps_lib_mgf, qry_file,
               algorithm='cos', analog_search=False, analog_max_shift=200.,
               pm_tol=0.02, frag_tol=0.05,
               min_score=0.7, min_matched_peak=3,
               rel_int_threshold=0.0, prec_mz_removal_da=1.5,
               peak_transformation='sqrt', max_peak_num=100,
               unmatched_penalty_factor=0.6,
               qry_batch_size=QRY_BATCH_SIZE):
    """
    Main function to search GNPS library

    algorithm: str. cos, entropy, rev_cos, rev_entropy
    peak_transformation: str. 'sqrt' or 'none', only applied on cosine similarity
    """

    print(f"Searching {qry_file} with {algorithm} algorithm...")

    if algorithm in ['cos', 'rev_cos']:
        search_eng = cosine_similarity
    elif algorithm in ['entropy', 'rev_entropy']:
        search_eng = entropy_similarity
    else:
        raise ValueError(f"Invalid algorithm: {algorithm}")

    search_kwargs = {
        'tolerance': frag_tol,
        'min_matched_peak': min_matched_peak,
        'sqrt_transform': True if peak_transformation == 'sqrt' else False,
        'penalty': unmatched_penalty_factor if 'rev' in algorithm else 0.0  # penalty for unmatched peaks, only applied on reverse search
    }

    # Some preprocessing
    qry_file_name = os.path.basename(qry_file)
    qry_basename = os.path.basename(qry_file).split('_iimn')[0]
    
    if algorithm.startswith('rev_') and unmatched_penalty_factor == 1.0:
        # traditional reverse cosine similarity
        out_folder = 'traditional_' + algorithm
        out_path = os.path.join(out_folder, f"{qry_basename}_matches.tsv")
    else:
        out_path = os.path.join(algorithm, f"{qry_basename}_matches.tsv")

    min_matched_peak = max(min_matched_peak, 1)

    # Initialize list for batch writing
    matches = []

    bad_ref_indices = set()  # ref spectra that has peaks less than min_matched_peak
    for batch_specs, batch_prec_mzs in batch_process_queries(qry_file, min_matched_peak, qry_batch_size):

        # iterate GNPS library
        for gnps_idx, spec in enumerate(iterate_gnps_lib_mgf(gnps_lib_mgf)):
            if gnps_idx in bad_ref_indices:
                continue

            # ref peaks, number of peaks check
            if len(spec['peaks']) < min_matched_peak:
                bad_ref_indices.add(gnps_idx)
                continue

            # precursor mz check
            if analog_search:
                v = np.where(np.abs(batch_prec_mzs - spec['PEPMASS']) <= analog_max_shift)[0]
            else:
                # exact search
                v = np.where(np.abs(batch_prec_mzs - spec['PEPMASS']) <= pm_tol)[0]

            if len(v) == 0:
                continue

            # clean ref peaks
            ref_peaks = clean_peaks(spec['peaks'],
                                    spec['PEPMASS'],
                                    rel_int_threshold=rel_int_threshold,
                                    prec_mz_removal_da=prec_mz_removal_da,
                                    max_peak_num=max_peak_num)
            if len(ref_peaks) < min_matched_peak:
                bad_ref_indices.add(gnps_idx)
                continue

            for i in v:
                qry_spec = batch_specs[i]

                # clean peaks
                if not qry_spec.peaks_cleaned:
                    qry_spec.peaks = clean_peaks(qry_spec.peaks,
                                                 qry_spec.precursor_mz,
                                                 rel_int_threshold=rel_int_threshold,
                                                 prec_mz_removal_da=prec_mz_removal_da,
                                                 max_peak_num=max_peak_num)
                    qry_spec.peaks_cleaned = True
                if len(qry_spec.peaks) < min_matched_peak:
                    continue

                # calculate similarity score
                if analog_search:
                    score, n_matches = search_eng(qry_spec.peaks, ref_peaks, **search_kwargs, shift=qry_spec.precursor_mz - spec['PEPMASS'])
                else:
                    score, n_matches = search_eng(qry_spec.peaks, ref_peaks, **search_kwargs, shift=0.0)

                # filter by minimum score and minimum matched peaks
                if score < min_score or n_matches < min_matched_peak:
                    continue

                # store matched rows
                matches.append({
                    'qry_file': qry_file_name,
                    'qry_scan': qry_spec.scan,
                    'qry_mz': qry_spec.precursor_mz,
                    'qry_rt': qry_spec.rt,
                    'score': round(score, 4),
                    'peaks': n_matches,
                    'mass_diff': qry_spec.precursor_mz - spec['PEPMASS'],
                    'ref_mz': spec['PEPMASS'],
                    'ref_id': spec['SPECTRUMID'] if spec['SPECTRUMID'] != '' else f'scans_{spec["SCANS"]}',
                })

    # write results
    df = pd.DataFrame(matches)
    df.to_csv(out_path, sep='\t', index=False)

    return


if __name__ == "__main__":
    import os
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    os.makedirs('cos', exist_ok=True)
    os.makedirs('rev_cos', exist_ok=True)
    os.makedirs('traditional_rev_cos', exist_ok=True)
    os.makedirs('entropy', exist_ok=True)
    os.makedirs('rev_entropy', exist_ok=True)
    os.makedirs('traditional_rev_entropy', exist_ok=True)

    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/marine_DOM_iimn_fbmn.mgf',
    #            algorithm='cos', peak_transformation='sqrt', unmatched_penalty_factor=0.6)
    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/marine_DOM_iimn_fbmn.mgf',
    #            algorithm='rev_cos', peak_transformation='sqrt', unmatched_penalty_factor=0.6)
    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/marine_DOM_iimn_fbmn.mgf',
    #            algorithm='rev_cos', peak_transformation='sqrt', unmatched_penalty_factor=1.0)

    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/soil_DOM_iimn_fbmn.mgf',
    #            algorithm='cos', peak_transformation='sqrt', unmatched_penalty_factor=0.6)
    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/soil_DOM_iimn_fbmn.mgf',
    #            algorithm='rev_cos', peak_transformation='sqrt', unmatched_penalty_factor=0.6)
    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/soil_DOM_iimn_fbmn.mgf',
    #            algorithm='rev_cos', peak_transformation='sqrt', unmatched_penalty_factor=1.0)

    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_urine_iimn_fbmn.mgf',
    #            algorithm='cos', peak_transformation='sqrt', unmatched_penalty_factor=0.6)
    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_urine_iimn_fbmn.mgf',
    #            algorithm='rev_cos', peak_transformation='sqrt', unmatched_penalty_factor=0.6)
    main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
               'mzmine_output/mouse_urine_iimn_fbmn.mgf',
               algorithm='rev_cos', peak_transformation='sqrt', unmatched_penalty_factor=1.0)

    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_feces_iimn_fbmn.mgf',
    #            algorithm='cos', peak_transformation='sqrt', unmatched_penalty_factor=0.6)
    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_feces_iimn_fbmn.mgf',
    #            algorithm='rev_cos', peak_transformation='sqrt', unmatched_penalty_factor=0.6)
    main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
               'mzmine_output/mouse_feces_iimn_fbmn.mgf',
               algorithm='rev_cos', peak_transformation='sqrt', unmatched_penalty_factor=1.0)

    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_liver_iimn_fbmn.mgf',
    #            algorithm='cos', peak_transformation='sqrt', unmatched_penalty_factor=0.6)
    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_liver_iimn_fbmn.mgf',
    #            algorithm='rev_cos', peak_transformation='sqrt', unmatched_penalty_factor=0.6)
    main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
               'mzmine_output/mouse_liver_iimn_fbmn.mgf',
               algorithm='rev_cos', peak_transformation='sqrt', unmatched_penalty_factor=1.0)

    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_embryo_iimn_fbmn.mgf',
    #            algorithm='cos', peak_transformation='sqrt', unmatched_penalty_factor=0.6)
    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_embryo_iimn_fbmn.mgf',
    #            algorithm='rev_cos', peak_transformation='sqrt', unmatched_penalty_factor=0.6)
    main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
               'mzmine_output/mouse_embryo_iimn_fbmn.mgf',
               algorithm='rev_cos', peak_transformation='sqrt', unmatched_penalty_factor=1.0)

    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_placenta_iimn_fbmn.mgf',
    #            algorithm='cos', peak_transformation='sqrt', unmatched_penalty_factor=0.6)
    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_placenta_iimn_fbmn.mgf',
    #            algorithm='rev_cos', peak_transformation='sqrt', unmatched_penalty_factor=0.6)
    main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
               'mzmine_output/mouse_placenta_iimn_fbmn.mgf',
               algorithm='rev_cos', peak_transformation='sqrt', unmatched_penalty_factor=1.0)

    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_brain_iimn_fbmn.mgf',
    #            algorithm='cos', peak_transformation='sqrt', unmatched_penalty_factor=0.6)
    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_brain_iimn_fbmn.mgf',
    #            algorithm='rev_cos', peak_transformation='sqrt', unmatched_penalty_factor=0.6)
    main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
               'mzmine_output/mouse_brain_iimn_fbmn.mgf',
               algorithm='rev_cos', peak_transformation='sqrt', unmatched_penalty_factor=1.0)

    ##############################################################################################################
    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/marine_DOM_iimn_fbmn.mgf',
    #            algorithm='entropy', peak_transformation='none', unmatched_penalty_factor=0.6)
    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/marine_DOM_iimn_fbmn.mgf',
    #            algorithm='rev_entropy', peak_transformation='none', unmatched_penalty_factor=0.6)
    main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
               'mzmine_output/marine_DOM_iimn_fbmn.mgf',
               algorithm='rev_entropy', peak_transformation='none', unmatched_penalty_factor=1.0)

    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/soil_DOM_iimn_fbmn.mgf',
    #            algorithm='entropy', peak_transformation='none', unmatched_penalty_factor=0.6)
    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/soil_DOM_iimn_fbmn.mgf',
    #            algorithm='rev_entropy', peak_transformation='none', unmatched_penalty_factor=0.6)
    main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
               'mzmine_output/soil_DOM_iimn_fbmn.mgf',
               algorithm='rev_entropy', peak_transformation='none', unmatched_penalty_factor=1.0)

    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_urine_iimn_fbmn.mgf',
    #            algorithm='entropy', peak_transformation='none', unmatched_penalty_factor=0.6)
    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_urine_iimn_fbmn.mgf',
    #            algorithm='rev_entropy', peak_transformation='none', unmatched_penalty_factor=0.6)
    main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
               'mzmine_output/mouse_urine_iimn_fbmn.mgf',
               algorithm='rev_entropy', peak_transformation='none', unmatched_penalty_factor=1.0)

    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_feces_iimn_fbmn.mgf',
    #            algorithm='entropy', peak_transformation='none', unmatched_penalty_factor=0.6)
    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_feces_iimn_fbmn.mgf',
    #            algorithm='rev_entropy', peak_transformation='none', unmatched_penalty_factor=0.6)
    main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
               'mzmine_output/mouse_feces_iimn_fbmn.mgf',
               algorithm='rev_entropy', peak_transformation='none', unmatched_penalty_factor=1.0)

    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_liver_iimn_fbmn.mgf',
    #            algorithm='entropy', peak_transformation='none', unmatched_penalty_factor=0.6)
    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_liver_iimn_fbmn.mgf',
    #            algorithm='rev_entropy', peak_transformation='none', unmatched_penalty_factor=0.6)
    main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
               'mzmine_output/mouse_liver_iimn_fbmn.mgf',
               algorithm='rev_entropy', peak_transformation='none', unmatched_penalty_factor=1.0)

    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_embryo_iimn_fbmn.mgf',
    #            algorithm='entropy', peak_transformation='none', unmatched_penalty_factor=0.6)
    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_embryo_iimn_fbmn.mgf',
    #            algorithm='rev_entropy', peak_transformation='none', unmatched_penalty_factor=0.6)
    main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
               'mzmine_output/mouse_embryo_iimn_fbmn.mgf',
               algorithm='rev_entropy', peak_transformation='none', unmatched_penalty_factor=1.0)

    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_placenta_iimn_fbmn.mgf',
    #            algorithm='entropy', peak_transformation='none', unmatched_penalty_factor=0.6)
    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_placenta_iimn_fbmn.mgf',
    #            algorithm='rev_entropy', peak_transformation='none', unmatched_penalty_factor=0.6)
    main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
               'mzmine_output/mouse_placenta_iimn_fbmn.mgf',
               algorithm='rev_entropy', peak_transformation='none', unmatched_penalty_factor=1.0)

    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_brain_iimn_fbmn.mgf',
    #            algorithm='entropy', peak_transformation='none', unmatched_penalty_factor=0.6)
    # main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
    #            'mzmine_output/mouse_brain_iimn_fbmn.mgf',
    #            algorithm='rev_entropy', peak_transformation='none', unmatched_penalty_factor=0.6)
    main_batch('ALL_GNPS_NO_PROPOGATED.mgf',
               'mzmine_output/mouse_brain_iimn_fbmn.mgf',
               algorithm='rev_entropy', peak_transformation='none', unmatched_penalty_factor=1.0)
