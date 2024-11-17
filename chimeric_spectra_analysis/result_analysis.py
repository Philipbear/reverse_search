import pickle
import numpy as np


def load_results():
    with open('/Users/shipei/Downloads/MSV000085891_chunk2_metrics.pkl', 'rb') as f:
        results = pickle.load(f)

    # print the results
    print('total spectra:', results['total_spectra_count'])
    print('empty spectra:', results['empty_peak_count'])
    print('precursor not found:', results['prec_mz_unfound'])

    print('summary of nonprec ion count (with isotopes):')
    print('mean:', np.mean(results['nonprec_ion_count_4da_with_isotopes']))
    print('median:', np.median(results['nonprec_ion_count_4da_with_isotopes']))

    print('summary of nonprec ion count (without isotopes):')
    print('mean:', np.mean(results['nonprec_ion_count_4da_no_isotopes']))
    print('median:', np.median(results['nonprec_ion_count_4da_no_isotopes']))

    print('summary of prec purity (with isotopes):')
    print('mean:', np.mean(results['prec_mz_purity_4da_with_isotopes']))
    print('median:', np.median(results['prec_mz_purity_4da_with_isotopes']))

    print('summary of prec purity (without isotopes):')
    print('mean:', np.mean(results['prec_mz_purity_4da_no_isotopes']))
    print('median:', np.median(results['prec_mz_purity_4da_no_isotopes']))


    return results


if __name__ == '__main__':
    results = load_results()

    # print the results
    for result in results:
        print(result)
        print()
