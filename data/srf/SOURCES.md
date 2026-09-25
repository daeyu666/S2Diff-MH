# Spectral-response and wavelength provenance

These frozen resources are shared byte-for-byte by `S2Diff-MH` and `comparison_experiments`.

## CAVE / Nikon D700
- HSI wavelengths: 31 bands, 400--700 nm at 10 nm steps, per CAVE.
- `nikon_d700_relative_spectral_response.csv`: public Nikon D700 R/G/B curves from `Leyangf/ChromFringe/data/raw/sensor_nikond700_{red,green,blue}.csv`.

## Botswana / EO-1 ALI
- Hyperion centres: former USGS EO-1 spectral-coverage table mirrored by `chryss/pygaarst`.
- Retained IDs: B10--B55, B82--B97, B102--B119, B134--B164, B187--B220 (145 bands).
- ALI RSR: `acolite/acolite/data/RSR/EO1_ALI.txt` (header cites the CSIRO EO-1 response archive).
- Retained MSI bands: MS-1, MS-2, MS-3, MS-4, MS-4p, MS-5p, MS-5, MS-7. PAN and coastal MS-1p are excluded; MS-1p has substantial support below B10=447.17 nm.
- Tiny negative calibration-tail values are clamped to zero.

## Augsburg / Sentinel-2A
- EnMAP 242 wavelengths are read from MDAS `band_242_meta_info.hdr`.
- Fixed MSI: native-10m B2/B3/B4/B8 from Sentinel-2A SRF V4.0.
- Official workbook: `COPE-GSEG-EOPG-TN-15-0007 - Sentinel-2 Spectral Response Functions 2024 - 4.0.xlsx`.
- URL: https://sentiwiki.copernicus.eu/__attachments/1692737/COPE-GSEG-EOPG-TN-15-0007%20-%20Sentinel-2%20Spectral%20Response%20Functions%202024%20-%204.0.xlsx
- Workbook SHA256: `1a9edc27d692a570911a460d589f188da0fc3e27f0b0bd1ad322059c380519b0`.
- CSV values are extracted from `MarcYin/spectral_library/examples/official_mapping/srfs/sentinel-2a_msi.json`, whose provenance manifest points to that workbook.

Do not silently change these resources during a benchmark series.
