# Reverse MS/MS spectral search

This is the repository for the reverse spectral search project.

<div style="display: flex; gap: 20px; justify-content: center;">
  <!-- Left column -->
  <div style="flex: 1; display: flex; justify-content: center;">
    <img src="fig/workflow.svg" width="300"/>
  </div>
  
  <!-- Right column -->
  <div style="flex: 1; display: flex; flex-direction: column; gap: 20px; align-items: center;">
    <img src="fig/fdr_cos_80_purity.svg" width="350"/>
    <img src="fig/annotation_cos.svg" width="400"/>
  </div>
</div>


## Source codes for symmetric & reverse MS/MS matching
- Cosine similarity: [`cosine.py`](https://github.com/Philipbear/reverse_MSMS_matching/blob/main/reverse_spectral_search/cosine.py)
- Entropy similarity: [`entropy.py`](https://github.com/Philipbear/reverse_MSMS_matching/blob/main/reverse_spectral_search/entropy.py)
- Bhattacharyya angle similarity: [`bhattacharya1.py`](https://github.com/Philipbear/reverse_MSMS_matching/blob/main/reverse_spectral_search/bhattacharya1.py)

## Chimeric spectra analysis
- [`analyze_chimeric_spec.py`](https://github.com/Philipbear/reverse_MSMS_matching/blob/main/chimeric_spectra/analyze_chimeric_spec.py)


## Citation
> Shipei Xing, Yasin El Abiead, Haoqi Nina Zhao, Vincent Charron-Lamoureux, Mingxun Wang, Pieter C. Dorrestein. Reverse spectral search: a simple but overlooked solution for chimeric spectra in metabolite annotation. To be preprinted.


## License
This project is licensed under the Apache 2.0 License (Copyright 2024 Shipei Xing).

