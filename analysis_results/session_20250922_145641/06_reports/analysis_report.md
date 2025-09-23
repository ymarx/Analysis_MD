# Independent Module Analysis Report
Generated: 2025-09-22 14:56:46
Session: 20250922_145641

## Module 1: Intensity Data Extraction
- Data shape: (200, 6832)
- Navigation points: 200
- Output file: analysis_results/session_20250922_145641/01_intensity_extraction/intensity_data.npz

## Module 2: Coordinate Mapping & Pixel Labels
### Mapping Statistics:
- Total mines: 25
- In bounds: 0 (0.0%)
- Out of bounds: 25

### XTF Data Bounds:
- Latitude range: [36.098730, 36.098743]
- Longitude range: [129.515067, 129.515293]

### Detailed Mapping Results:
```
Mine ID   GPS Lat    GPS Lon          Status  Pixel X  Pixel Y
 mine_1 36.593398 129.509296 ❌ OUT OF BOUNDS       -1       -1
 mine_2 36.593362 129.509493 ❌ OUT OF BOUNDS       -1       -1
 mine_3 36.593326 129.509693 ❌ OUT OF BOUNDS       -1       -1
 mine_4 36.593331 129.509894 ❌ OUT OF BOUNDS       -1       -1
 mine_5 36.593305 129.510092 ❌ OUT OF BOUNDS       -1       -1
 mine_6 36.593300 129.510295 ❌ OUT OF BOUNDS       -1       -1
 mine_7 36.593326 129.510497 ❌ OUT OF BOUNDS       -1       -1
 mine_8 36.593300 129.510692 ❌ OUT OF BOUNDS       -1       -1
 mine_9 36.593305 129.510896 ❌ OUT OF BOUNDS       -1       -1
mine_10 36.593279 129.511095 ❌ OUT OF BOUNDS       -1       -1
mine_11 36.593285 129.511297 ❌ OUT OF BOUNDS       -1       -1
mine_12 36.593269 129.511492 ❌ OUT OF BOUNDS       -1       -1
mine_13 36.593264 129.511693 ❌ OUT OF BOUNDS       -1       -1
mine_14 36.593248 129.511895 ❌ OUT OF BOUNDS       -1       -1
mine_15 36.593264 129.512088 ❌ OUT OF BOUNDS       -1       -1
mine_16 36.593249 129.512294 ❌ OUT OF BOUNDS       -1       -1
mine_17 36.593254 129.512493 ❌ OUT OF BOUNDS       -1       -1
mine_18 36.593228 129.512695 ❌ OUT OF BOUNDS       -1       -1
mine_19 36.593233 129.512896 ❌ OUT OF BOUNDS       -1       -1
mine_20 36.593249 129.513089 ❌ OUT OF BOUNDS       -1       -1
mine_21 36.593223 129.513295 ❌ OUT OF BOUNDS       -1       -1
mine_22 36.593187 129.513492 ❌ OUT OF BOUNDS       -1       -1
mine_23 36.593223 129.513694 ❌ OUT OF BOUNDS       -1       -1
mine_24 36.593197 129.513893 ❌ OUT OF BOUNDS       -1       -1
mine_25 36.593171 129.514092 ❌ OUT OF BOUNDS       -1       -1
```

## Module 3: Data Augmentation
- Positive samples (mines): 20
- Negative samples (background): 33
- Total samples: 53
- Class balance: 37.7% positive

## Module 4: Feature Extraction
- Feature matrix shape: (53, 8)
- Feature names: mean, std, min, max, median, variance, h_gradient, v_gradient
- Output file: analysis_results/session_20250922_145641/04_feature_extraction/features.npz

## Analysis Summary
⚠️ **WARNING**: Very low mapping rate detected!
   - Most mine locations are outside XTF data bounds
   - This suggests coordinate system mismatch or different survey areas
   - Synthetic data was likely used for testing

## Output Directory Structure
```
analysis_results/session_20250922_145641/
├── 01_intensity_extraction/
│   └── intensity_data.npz
├── 02_coordinate_mapping/
│   ├── pixel_labels.json
│   └── mapping_report.csv
├── 03_data_augmentation/
│   └── augmented_samples.json
├── 04_feature_extraction/
│   ├── features.npz
│   └── feature_statistics.csv
├── 05_visualizations/
│   ├── intensity_data.png
│   ├── coordinate_mapping.png
│   ├── augmented_samples.png
│   └── feature_distributions.png
└── 06_reports/
    └── analysis_report.md
```