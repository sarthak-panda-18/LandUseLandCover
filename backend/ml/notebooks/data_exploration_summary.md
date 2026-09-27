# Phase 2: LULC Dataset Exploration & Validation Summary

**Dataset Overview:** Sentinel-2 multispectral satellite imagery tiles and aligned Land Use / Land Cover (LULC) ground truth masks for Vijayawada.

---

## 1. Dataset & Spatial Summary

| Metric | Details | Status |
| :--- | :--- | :--- |
| **Total Tile Pairs** | 6 input tiles + 6 matching label masks |  Confirmed |
| **Input Tile Dimensions** | 4 tiles @ `2048 x 2048`, 2 tiles @ `915 x 2048`, 2 tiles @ `2048 x 1294`, 2 tiles @ `915 x 1294` |  Spatially Aligned |
| **Total Dataset Pixels** | 16,746,762 pixels |  Validated |
| **Coordinate Reference System (CRS)** | `EPSG:4326` (WGS84 Geographic 2D) |  Consistent across all 12 files |
| **Spatial Resolution** | `8.983152841195215e-05` degrees (~10.0 meters/pixel at equator) |  Consistent across all 12 files |
| **Input Band Count** | Exactly **5 bands** per tile |  Confirmed |
| **Label Band Count** | Exactly **1 band** per tile |  Confirmed |
| **Data Types** | Input: `float32` (all 5 bands) \| Labels: `uint8` |  Consistent |

---

## 2. Input Band Identification & Spectral Value Ranges

The embedded GeoTIFF metadata explicitly confirms band identities and designations (`src.descriptions` = `('B2', 'B3', 'B4', 'B8', 'NDVI')`):

| Band Index | Band Tag | Spectral Band / Feature | Value Range (Min, Max) | Global Mean | Scaling / Data Interpretation |
| :---: | :---: | :--- | :---: | :---: | :--- |
| **Band 1** | `B2` | Blue (~490 nm) | `[0.0254, 0.4962]` | `0.0715` | Float32 Surface Reflectance (0 to 1 scale) |
| **Band 2** | `B3` | Green (~560 nm) | `[0.0291, 0.5318]` | `0.0965` | Float32 Surface Reflectance (0 to 1 scale) |
| **Band 3** | `B4` | Red (~665 nm) | `[0.0208, 0.5932]` | `0.1011` | Float32 Surface Reflectance (0 to 1 scale) |
| **Band 4** | `B8` | Near-Infrared / NIR (~842 nm) | `[0.0018, 0.7160]` | `0.2616` | Float32 Surface Reflectance (0 to 1 scale) |
| **Band 5** | `NDVI` | Normalized Difference Vegetation Index | `[-0.8537, 0.8667]` | `0.4286` | Standard Index formula $\frac{B8 - B4}{B8 + B4} \in [-1, 1]$ |

> [!NOTE]
> **Scaling Clarification:** The raw Sentinel-2 reflectance values are already normalized to standard floating-point reflectance (`[0.0, 1.0]`) and index values (`[-1.0, 1.0]`). There is **no need to divide by 10,000** during preprocessing.

---

## 3. Label Class Distribution (Combined across all 6 tiles)

| Class ID | Class Name | Combined Pixel Count | % of Valid Pixels | % of Total Pixels | Sample Representation |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | **Water** | 940,630 | **7.33%** | 5.62% | Rivers, water bodies, reservoirs |
| **1** | **Trees / Forest** | 2,179,699 | **16.99%** | 13.02% | Canopy cover, dense forestry |
| **2** | **Crops** | 6,628,013 | **51.65%** | 39.58% | Agricultural fields (Majority class) |
| **3** | **Grassland** | 70,878 | **0.55%** | 0.42% | Pastures, scrub vegetation (Minority class) |
| **4** | **Built-up** | 2,673,103 | **20.83%** | 15.96% | Urban areas, roads, infrastructure |
| **5** | **Bare land** | 338,987 | **2.64%** | 2.02% | Exposed soil, sandbars, cleared ground |
| **255** | **Invalid / Masked** | 3,915,452 | — | **23.38%** | Background, nodata boundary padding |
| **TOTAL** | *Valid (0–5)* | **12,831,310** | **100.00%** | **76.62%** | *Trainable dataset pool* |
| **TOTAL** | *All Pixels* | **16,746,762** | — | **100.00%** | *Full 6-tile grid extent* |

---

## 4. Per-Tile Detailed Breakdown

| Tile Coordinate Offset | Total Pixels | Class 0 (Water) | Class 1 (Trees) | Class 2 (Crops) | Class 3 (Grass) | Class 4 (Built-up) | Class 5 (Bare) | Class 255 (Masked) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0000000000-0000000000` | 4,194,304 | 450,604 | 588,862 | 1,342,424 | 32,864 | 476,696 | 107,364 | 1,195,490 (28.50%) |
| `0000000000-0000002048` | 4,194,304 | 208,364 | 841,974 | 1,278,600 | 11,916 | 1,136,411 | 75,661 | 641,378 (15.29%) |
| `0000000000-0000004096` | 1,873,920 | 78,451 | 348,761 | 638,373 | 18,260 | 207,322 | 2,586 | 580,167 (30.96%) |
| `0000002048-0000000000` | 2,650,112 | 36,178 | 190,542 | 1,392,419 | 6,798 | 385,305 | 1,110 | 637,760 (24.07%) |
| `0000002048-0000002048` | 2,650,112 | 142,599 | 139,951 | 1,502,097 | 807 | 316,979 | 150,799 | 396,880 (14.98%) |
| `0000002048-0000004096` | 1,184,010 | 24,434 | 69,609 | 474,100 | 233 | 150,390 | 1,467 | 463,777 (39.17%) |

---

## 5. Flagged Issues & Key Observations for Phase 3

> [!WARNING]
> **Severe Class Imbalance (93.5x Ratio)**
> - **Crops (Class 2)** dominates with **51.65%** (6.63M pixels).
> - **Grassland (Class 3)** is severely underrepresented at **0.55%** (70.88k pixels).
> - **Bare land (Class 5)** is also sparse at **2.64%** (338.99k pixels).
> - *Recommendation for Phase 3:* Stratified sampling or class-weighted training (`class_weight='balanced'`) will be essential to prevent the Random Forest classifier from neglecting Grassland and Bare land.

> [!IMPORTANT]
> **Input NaN Values (7,389 Pixels in Class 0 - Water)**
> - Exactly **7,389 pixels** across 5 of the 6 tiles contain `NaN` in all 5 bands.
> - All 7,389 NaN pixels fall under **Class 0 (Water)** in the ground-truth masks (deep water absorption / cloud-shadow nodata artifacts).
> - *Recommendation for Phase 3:* When extracting feature vectors for training, remove rows where `np.isnan(X).any(axis=1)` in addition to filtering out `y == 255`.

> [!TIP]
> **Spatial Alignment & Boundary Padding**
> - All 6 tile pairs have **100% matching dimensions, bounding boxes, and affine geotransforms**.
> - Value `255` accounts for **23.38%** (3.92M pixels) of the dataset representing spatial tile borders and non-study-area padding. Masking `y != 255` is clean and straightforward.
