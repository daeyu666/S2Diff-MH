# Spectral-response and wavelength provenance

These frozen resources are shared byte-for-byte by `S2Diff-MH` and
`comparison_experiments`.  Do not silently change them during a benchmark
series.

## CAVE / Nikon D700

- HSI wavelengths: 31 bands, 400--700 nm at 10 nm steps, per the official CAVE database.
- Frozen file: `nikon_d700_relative_spectral_response.csv`.
- Upstream public Nikon D700 R/G/B curves:
  - https://github.com/Leyangf/ChromFringe/blob/master/data/raw/sensor_nikond700_red.csv
  - https://github.com/Leyangf/ChromFringe/blob/master/data/raw/sensor_nikond700_green.csv
  - https://github.com/Leyangf/ChromFringe/blob/master/data/raw/sensor_nikond700_blue.csv
- Dataset: https://cave.cs.columbia.edu/repository/Multispectral

## Botswana / EO-1 ALI

- Hyperion centres: former USGS EO-1 spectral-coverage table mirrored by
  `chryss/pygaarst`.
- Retained IDs: B10--B55, B82--B97, B102--B119, B134--B164, B187--B220
  (145 bands).
- Frozen HSI wavelength files:
  `../wavelengths/Botswana_Hyperion_145.txt` and
  `../wavelengths/Botswana_Hyperion_band_ids.txt`.
- Frozen ALI file: `eo1_ali_8band_relative_spectral_response.csv`.
- Upstream ALI RSR table:
  https://github.com/acolite/acolite/blob/main/acolite/data/RSR/EO1_ALI.txt
  (its header cites the CSIRO EO-1 response archive).
- Retained MSI bands: MS-1, MS-2, MS-3, MS-4, MS-4p, MS-5p, MS-5, MS-7.
  PAN and coastal MS-1p are excluded; MS-1p has substantial response support
  below the first retained Botswana Hyperion band B10=447.17 nm.
- Tiny negative calibration-tail values are clamped to zero.
- Dataset:
  https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes

## Augsburg / Sentinel-2A

- EnMAP 242 wavelengths are read directly from MDAS
  `band_242_meta_info.hdr`; no synthetic linear wavelength grid is used.
- Fixed synthetic MSI: native-10m Sentinel-2A B2/B3/B4/B8, SRF V4.0.
- Frozen file: `sentinel2a_srf_v4_B2_B3_B4_B8.csv`.
- Official workbook:
  `COPE-GSEG-EOPG-TN-15-0007 - Sentinel-2 Spectral Response Functions 2024 - 4.0.xlsx`.
- Official workbook URL:
  https://sentiwiki.copernicus.eu/__attachments/1692737/COPE-GSEG-EOPG-TN-15-0007%20-%20Sentinel-2%20Spectral%20Response%20Functions%202024%20-%204.0.xlsx
- Workbook SHA256:
  `1a9edc27d692a570911a460d589f188da0fc3e27f0b0bd1ad322059c380519b0`.
- Frozen CSV values are extracted from the public
  `MarcYin/spectral_library` Sentinel-2A MSI SRF representation, whose
  provenance manifest points to the official workbook.
- Dataset DOI: https://doi.org/10.14459/2022mp1657312


### Augsburg-Real platform rule

The Augsburg-Real branch does not assume that the real L2A stack is Sentinel-2A.
`prepare_augsburg_real.py` first inspects the GeoTIFF metadata and otherwise
requires an explicit `--s2_platform S2A|S2B`.  B2/B3/B4/B8 must use the
official SRF of that platform.  The repository freezes S2A V4.0 for the synthetic Augsburg benchmark and
S2B V3.0 for the fixed 7 May 2018 Augsburg-Real acquisition. The two resources
are never interchanged silently.


## Augsburg-Real / Sentinel-2B

- The real MDAS Sentinel-2 acquisition on 7 May 2018 is treated as Sentinel-2B.
  This is consistent with the R065 acquisition phase: a documented S2B R065
  acquisition occurs on 16 June 2018 (40 days later), while a documented S2A
  R065 acquisition occurs on 26 June 2017 (315 days earlier, i.e. the opposite
  5-day A/B phase).
- Frozen file: `sentinel2b_srf_v3_B2_B3_B4_B8.csv`.
- Source representation:
  `tnigon/hs_process/hs_process/data/sentinel-2b_band_response.csv`,
  containing ESA Sentinel-2B average SRF columns
  `S2B_SR_AV_B2/B3/B4/B8`.
- That representation traces to the official ESA
  `S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0.xlsx`.
- Only native-10m B2/B3/B4/B8 are retained for Augsburg-Real.
