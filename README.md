# Reverse MS/MS spectral search

Chimeric spectra are ubiquitous in MS/MS data, which compromises the quality and reliability of MS/MS matching-based metabolite annotation.
Reverse spectral search is a simple yet overlooked solution to chimeric spectra. 

Here, we enhanced the reverse search by introducing a penalty factor to unmatched peaks, which substantially increases the number of spectral matches while maintaining rigorous quality control.

<table>
<tr>
  <td width="40%" align="center" valign="center">
    <img src="fig/workflow.svg" width="300"/>
  </td>
  <td width="60%" align="center">
    <img src="fig/fdr_cos_80_purity.svg" width="350"/>
    <br><br>
    <img src="fig/annotation_cos.svg" width="400"/>
  </td>
</tr>
</table>


## Symmetric & reverse spectral search
- Cosine similarity: [`cosine.py`](https://github.com/Philipbear/reverse_MSMS_matching/blob/main/reverse_spectral_search/cosine.py)
- Entropy similarity: [`entropy.py`](https://github.com/Philipbear/reverse_MSMS_matching/blob/main/reverse_spectral_search/entropy.py)
- Bhattacharyya angle: [`bhattacharya1.py`](https://github.com/Philipbear/reverse_MSMS_matching/blob/main/reverse_spectral_search/bhattacharya1.py)

## Chimeric spectra analysis
- [`analyze_chimeric_spec.py`](https://github.com/Philipbear/reverse_MSMS_matching/blob/main/chimeric_spectra/analyze_chimeric_spec.py)


## Citation
> Shipei Xing, Yasin El Abiead, Haoqi Nina Zhao, Vincent Charron-Lamoureux, Mingxun Wang, Pieter C. Dorrestein. Reverse spectral search: a simple but overlooked solution for chimeric spectra in metabolite annotation. To be preprinted.


## License
This project is licensed under the Apache 2.0 License (Copyright 2024 Shipei Xing).

