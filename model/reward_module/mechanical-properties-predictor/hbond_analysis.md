# Hydrogen-Bond Analysis for Mechanical Protein Properties

## Scope

This analysis links hydrogen-bond geometry in the structural set to two mechanical-property labels:

- `v127`: toughness averaged per amino acid.
- `v128`: maximum tensile strength.

Input structures: `/home/jianquanzhao/data/tsinghua/mpprd/pdbs/pdbs`  
Input mechanical-property table: `/mnt/data1/home/jianquanzhao/data/cath/filtered_All_Mechanical_Vectors_cath_all_fasta_results.csv`

Matched protein structures: **7041**.

## Hydrogen-Bond Definition

Because these PDB files contain explicit hydrogen atoms, hydrogen bonds were detected with a geometric donor-hydrogen-acceptor definition:

- donor heavy atom: N/O/S with an attached hydrogen within 1.35 Å;
- acceptor atom: protein O, weak S acceptor, or unprotonated histidine ND1/NE2;
- D-A distance <= 3.50 Å;
- H-A distance <= 2.70 Å;
- D-H-A angle >= 120.0 degrees;
- same-residue donor/acceptor pairs were excluded.

Each hydrogen bond was classified by backbone/sidechain role, donor/acceptor chemistry, and sequence separation (`local`, `nonlocal`, `interchain`).

## Dataset-Level Hydrogen-Bond Statistics

| feature               |   count |     mean |      std |      min |      25% |      50% |      75% |        max |
|:----------------------|--------:|---------:|---------:|---------:|---------:|---------:|---------:|-----------:|
| v127                  |    7041 | 351.218  | 102.333  | 122.97   | 275.46   | 348.06   | 413.16   |  1499.5    |
| v128                  |    7041 | 511.257  | 265.883  | 266.214  | 425.224  | 476.194  | 529.288  | 11108.1    |
| sequence_length       |    7041 | 105.391  |  71.7181 |  27      |  58      |  89      | 125      |   586      |
| hbond_count           |    7041 |  70.6182 |  62.4677 |   4      |  30      |  54      |  87      |   527      |
| hbond_per_residue     |    7041 |   0.6119 |   0.1601 |   0.0632 |   0.5    |   0.6344 |   0.7302 |     1.0693 |
| d_a_distance_mean     |    7041 |   2.9628 |   0.0427 |   2.7253 |   2.9367 |   2.9572 |   2.9821 |     3.289  |
| h_a_distance_mean     |    7041 |   2.0362 |   0.0563 |   1.828  |   1.9998 |   2.0266 |   2.0626 |     2.3988 |
| dha_angle_mean        |    7041 | 155.723  |   3.4124 | 138.317  | 154.188  | 156.335  | 157.949  |   168.837  |
| strong_hbond_fraction |    7041 |   0.5114 |   0.1321 |   0      |   0.4386 |   0.5339 |   0.6    |     1      |

## Correlation With Mechanical Properties

The tables below rank hydrogen-bond features by absolute Spearman correlation against log-transformed targets. Spearman is emphasized because mechanical labels are long-tailed and monotonic structure-property trends are more relevant than strictly linear trends.

### Top Correlations With log1p(v127), Toughness

| feature                              |   pearson |   spearman |   spearman_p |
|:-------------------------------------|----------:|-----------:|-------------:|
| chem_type_N_to_O_count               |   0.68563 |    0.85695 |            0 |
| hbond_count                          |   0.68284 |    0.85618 |            0 |
| seq_class_nonlocal_count             |   0.67948 |    0.85423 |            0 |
| role_type_backbone_to_backbone_count |   0.69041 |    0.84269 |            0 |
| atom_count                           |   0.67094 |    0.80792 |            0 |
| residue_count                        |   0.66509 |    0.8035  |            0 |
| sequence_length                      |   0.66502 |    0.8034  |            0 |
| seq_class_local_count                |   0.62277 |    0.7393  |            0 |

### Top Correlations With log1p(v128), Strength

| feature                              |   pearson |   spearman |   spearman_p |
|:-------------------------------------|----------:|-----------:|-------------:|
| residue_count                        |   0.60506 |    0.82791 |            0 |
| sequence_length                      |   0.605   |    0.82766 |            0 |
| atom_count                           |   0.60463 |    0.82703 |            0 |
| role_type_backbone_to_backbone_count |   0.57483 |    0.77319 |            0 |
| hbond_count                          |   0.56977 |    0.77262 |            0 |
| chem_type_N_to_O_count               |   0.57037 |    0.77166 |            0 |
| seq_class_nonlocal_count             |   0.55847 |    0.75551 |            0 |
| seq_class_local_count                |   0.54093 |    0.70374 |            0 |

Full correlation table: `outputs/hbond_analysis/hbond_property_correlations.csv`.

Main visualizations:

- `outputs/hbond_analysis/plots/hbond_per_residue_distribution.png`
- `outputs/hbond_analysis/plots/hbond_geometry_vs_v127.png`
- `outputs/hbond_analysis/plots/hbond_geometry_vs_v128.png`
- `outputs/hbond_analysis/plots/hbond_property_correlation_heatmap.png`

## Statistical Machine-Learning Model

Chosen method: **random forest regression on hydrogen-bond summary features**.

Rationale:

1. Hydrogen-bond effects are likely nonlinear and threshold-like: one extra nonlocal hydrogen bond may matter differently in a short beta-rich protein than in a long mixed fold.
2. Random forests handle correlated tabular descriptors reasonably well and do not assume a linear relationship.
3. Feature importance gives a first-pass interpretable ranking, useful before moving to deeper geometric or graph models.

The model was trained on `log1p(v127), log1p(v128)` and evaluated after inverse transform to the original mechanical-property scale.

### Metrics

| split   |    n |   toughness_v127/r2 |   toughness_v127/mae |   toughness_v127/rmse |   toughness_v127/spearman |   strength_v128/r2 |   strength_v128/mae |   strength_v128/rmse |   strength_v128/spearman |
|:--------|-----:|--------------------:|---------------------:|----------------------:|--------------------------:|-------------------:|--------------------:|---------------------:|-------------------------:|
| train   | 5632 |              0.8822 |              20.3123 |               35.1347 |                    0.9676 |             0.4409 |             45.3208 |              206.756 |                   0.9467 |
| val     |  704 |              0.709  |              37.2323 |               51.414  |                    0.8685 |             0.2086 |             69.9343 |              195.524 |                   0.8132 |
| test    |  705 |              0.6632 |              38.0087 |               62.8474 |                    0.8932 |             0.2999 |             66.6549 |              181.018 |                   0.8466 |

### Top Random-Forest Features

| feature                                |   importance |
|:---------------------------------------|-------------:|
| sequence_length                        |      0.10042 |
| hbond_count                            |      0.09895 |
| chem_type_N_to_O_count                 |      0.09723 |
| seq_class_nonlocal_count               |      0.09443 |
| role_type_backbone_to_backbone_count   |      0.08168 |
| seq_class_local_count                  |      0.03938 |
| chem_type_N_to_O_per_residue           |      0.03538 |
| role_type_sidechain_to_backbone_count  |      0.03085 |
| hbond_per_residue                      |      0.03046 |
| role_type_sidechain_to_sidechain_count |      0.03024 |
| role_type_backbone_to_sidechain_count  |      0.02721 |
| seq_class_nonlocal_per_residue         |      0.02528 |

Feature importance plot: `outputs/hbond_analysis/plots/hbond_random_forest_feature_importance.png`.

## Interpretation

Hydrogen bonding should be interpreted as a structural network signal rather than only a raw count. The most useful descriptors are expected to combine:

- hydrogen-bond density, normalized by sequence length;
- geometric quality, especially short D-A distance and near-linear D-H-A angle;
- nonlocal or interchain hydrogen bonds, which can resist unfolding pathways more directly than local helix-stabilizing contacts;
- backbone-backbone hydrogen bonds, which often report beta-sheet or regular secondary-structure reinforcement;
- sidechain-mediated hydrogen bonds, which may create sacrificial or load-bearing crosslinks depending on topology.

## Additional Hypotheses

1. **Topology matters more than count alone.** Nonlocal hydrogen bonds and interchain hydrogen bonds should correlate more strongly with strength than local hydrogen bonds, because they couple distant sequence regions and can resist extension.
2. **Geometry quality may separate stiffness from toughness.** Short, linear hydrogen bonds may increase initial resistance, while a larger number of weaker sidechain hydrogen bonds may dissipate energy and improve toughness.
3. **Hydrogen-bond anisotropy matters.** Bonds aligned with the pulling direction should contribute more to tensile strength than bonds orthogonal to the force path. This requires adding pulling-axis or terminal-distance descriptors.
4. **Hydrogen bonds interact with secondary structure.** Beta-rich proteins may gain strength from backbone-backbone hydrogen-bond ladders, while alpha-rich proteins may show weaker direct correlation because helices unzip locally.
5. **Hydrogen bonds and hydrophobic packing are coupled.** A good next model should combine hydrogen-bond network descriptors with solvent-accessible area, contact order, hydrophobic core density, salt bridges, and disulfide/covalent constraints.
6. **Mechanical labels may be dominated by unfolding pathway bottlenecks.** Global aggregate features can miss a small number of critical load-bearing contacts, so graph features around high-contact-order regions may improve prediction.

## Recommended Next Step

Use this hydrogen-bond feature set as an interpretable baseline, then add secondary-structure-aware and contact-order-aware descriptors. If these descriptors improve similarity-split performance, they can be incorporated into the reward model as auxiliary physics-informed features alongside ESM2 embeddings.

## Separating Length Effect From Hydrogen-Bond Network Effect

### 1. Length-Normalized Features

Using length-averaged hydrogen-bond descriptors is a reasonable and necessary first step.

Examples:

```text
hbond_count / sequence_length
N_to_O_count / sequence_length
nonlocal_hbond_count / sequence_length
backbone_to_backbone_count / sequence_length
sidechain_to_sidechain_count / sequence_length
strong_hbond_count / sequence_length
```

This directly addresses the simplest confounder: longer proteins naturally have more residues, more atoms, and therefore more possible donor/acceptor pairs. If raw hydrogen-bond count correlates with mechanical performance, part of that signal may only mean "larger protein", not "better hydrogen-bond network".

However, length normalization is not sufficient by itself.

The reason is that mechanical performance is not controlled only by density. Two proteins can have the same number of hydrogen bonds per residue but very different mechanical behavior if:

1. one has mostly local alpha-helical hydrogen bonds and the other has nonlocal beta-sheet hydrogen-bond ladders;
2. one has hydrogen bonds aligned with the pulling direction and the other has hydrogen bonds orthogonal to it;
3. one distributes hydrogen bonds uniformly and the other concentrates them in a load-bearing mechanical clamp;
4. one is compact with high contact order and the other is extended with low contact order;
5. one has many weak/sacrificial sidechain hydrogen bonds and the other has fewer but more linear backbone-backbone hydrogen bonds.

Therefore, length-averaged metrics should be used, but interpreted as hydrogen-bond density rather than as a complete estimate of hydrogen-bond mechanical contribution.

### 2. Residualization Against Length

A stronger way to remove length effect is residualization.

For each hydrogen-bond feature, fit a simple background model:

```text
hbond_feature = f(sequence_length) + residual
```

Then use the residual as the length-independent hydrogen-bond signal:

```text
hbond_feature_residual = observed_hbond_feature - expected_hbond_feature_given_length
```

This is better than simple division when the relationship between feature count and length is not perfectly linear. For example, hydrogen-bond count may scale sublinearly or superlinearly with length depending on fold compactness.

Recommended variants:

```text
linear residualization:        feature ~ length
log-linear residualization:    log1p(feature) ~ log1p(length)
nonlinear residualization:     feature ~ spline(length) or random forest(length)
```

Interpretation:

```text
positive residual  = more hydrogen bonding than expected for this length
negative residual  = fewer hydrogen bonds than expected for this length
```

This is probably the best next baseline because it cleanly asks:

```text
Among proteins of similar length, does having more/better hydrogen bonding predict higher toughness or strength?
```

### 3. Partial Correlation

Partial correlation can test whether hydrogen-bond features still correlate with mechanical properties after controlling for sequence length.

Conceptually:

```text
corr(hbond_feature, mechanical_property | sequence_length)
```

Implementation:

1. regress the hydrogen-bond feature on sequence length;
2. regress the mechanical target on sequence length;
3. compute correlation between the two residuals.

This is useful because it gives a direct statistical answer to the confounding question.

If a feature has high raw correlation but low partial correlation, it is mostly a length proxy. If it remains high after controlling for length, it is more likely to represent a real hydrogen-bond network effect.

### 4. Length-Stratified Analysis

Another robust strategy is to split proteins into length bins and analyze each bin separately.

Example:

```text
short proteins:   bottom 25% length
medium proteins:  middle 50% length
long proteins:    top 25% length
```

or:

```text
27-60 aa
60-100 aa
100-160 aa
>160 aa
```

Then compute hydrogen-bond correlations inside each bin.

This prevents long proteins from dominating the global correlation. It also helps reveal whether hydrogen bonding has different mechanical roles in short peptides, medium domains, and large multidomain structures.

This approach is especially important here because `v128` showed very strong correlation with sequence length.

### 5. Matched-Length Pair Analysis

A more chemically interpretable approach is matched comparison.

For each high-performance protein, find low-performance proteins with similar sequence length, then compare their hydrogen-bond network descriptors.

Example matching constraints:

```text
abs(length_a - length_b) <= 5 residues
or
abs(log(length_a) - log(length_b)) <= threshold
```

Then ask:

```text
Within matched-length pairs, do high-strength proteins have more nonlocal hydrogen bonds?
Within matched-length pairs, do high-toughness proteins have higher strong_hbond_fraction?
Within matched-length pairs, are backbone-backbone hydrogen-bond ladders enriched?
```

This is less dependent on model assumptions and is often easier to explain to chemistry/biophysics collaborators.

### 6. Multivariate Models With Explicit Length Control

For predictive modeling, length should be included explicitly as a covariate, while hydrogen-bond features are tested by ablation.

Recommended model comparison:

```text
Model A: length only
Model B: length + hydrogen-bond density features
Model C: length + hydrogen-bond residual features
Model D: length + hydrogen-bond topology + geometry features
```

The key quantity is the incremental improvement:

```text
Delta performance = performance(Model B/C/D) - performance(Model A)
```

If hydrogen-bond features truly add mechanical information beyond length, then Model B/C/D should improve R2, Spearman, or top-k enrichment over the length-only baseline.

This is likely the most useful criterion for deciding whether hydrogen-bond features should be added to the reward model.

### 7. Contact-Order and Topology Normalization

Length is not the only confounder. Hydrogen bonds in mechanically relevant proteins should also be normalized by topology.

Useful descriptors:

```text
mean sequence separation of hydrogen bonds
nonlocal_hbond_fraction
long_range_hbond_count / sequence_length
backbone_backbone_nonlocal_hbond_fraction
hydrogen-bond contact order
```

A hydrogen bond between residues `i` and `i+4` often stabilizes local secondary structure, while a hydrogen bond between residues far apart in sequence can connect distant structural elements and resist unfolding more directly.

Therefore, a better mechanical descriptor is not only:

```text
how many hydrogen bonds per residue?
```

but:

```text
how many load-bearing, nonlocal, geometrically strong hydrogen bonds per residue?
```

### Recommended Next Analysis

The next analysis should add three groups of features:

1. length-normalized features;
2. length-residualized features;
3. topology-aware features, especially nonlocal/contact-order hydrogen-bond descriptors.

Then run the following ablation:

```text
length only
length + normalized hydrogen-bond features
length + residualized hydrogen-bond features
length + topology-aware hydrogen-bond features
```

The most important question is:

```text
Do hydrogen-bond features improve prediction beyond a length-only model?
```

If yes, hydrogen bonding is not merely a size proxy. It is a meaningful mechanical structural signal and should be included as a physics-informed reward component or auxiliary predictor feature.


## Length-Controlled Hydrogen-Bond Analysis

### Hypothesis

Raw hydrogen-bond counts are strongly correlated with mechanical properties, but part of this signal may be caused by protein length: longer proteins have more atoms, more donor/acceptor pairs, and therefore more hydrogen bonds. The hypothesis tested here is:

```text
Hydrogen-bond network descriptors still explain toughness and strength after controlling for sequence length.
```

### Method

The analysis used the cached hydrogen-bond table from `outputs/hbond_analysis/` and did not rescan PDB files. Five complementary tests were performed:

1. **Length-normalized descriptors**: counts were converted to per-residue or fraction features.
2. **Residualized descriptors**: raw count features were regressed against `log1p(sequence_length)`, and residuals were used as length-independent hydrogen-bond signals.
3. **Partial correlation**: both hydrogen-bond features and mechanical labels were residualized against `log1p(sequence_length)`, then correlated.
4. **Length-stratified analysis**: proteins were split into length quartiles and correlations were recomputed within each quartile.
5. **Model ablation**: random-forest regressors were trained with increasing feature groups: length only, length plus density/geometry, length plus residual counts, length plus topology, and length plus all controlled hydrogen-bond features.

Topology-aware features were also added from hydrogen-bond pair details, including `hbond_contact_order`, long-range hydrogen-bond fraction, nonlocal backbone-backbone hydrogen bonds, and strong nonlocal hydrogen bonds.

### Partial Correlation Results

After controlling for length, the strongest remaining hydrogen-bond associations are much smaller than the raw correlations. This is expected and biologically important: a large part of the raw signal was indeed a length/size signal.

#### Top Partial Correlations With log1p(v127), Toughness

| feature                                    |   spearman |   partial_spearman |   spearman_drop_after_length_control |
|:-------------------------------------------|-----------:|-------------------:|-------------------------------------:|
| strong_nonlocal_per_residue                |    0.68221 |            0.52457 |                              0.15764 |
| chem_type_N_to_O_per_residue               |    0.72447 |            0.49405 |                              0.23041 |
| hbond_per_residue                          |    0.7304  |            0.49055 |                              0.23986 |
| seq_class_nonlocal_per_residue             |    0.72449 |            0.47616 |                              0.24833 |
| role_type_backbone_to_backbone_per_residue |    0.62448 |            0.45443 |                              0.17005 |
| long_range_hbond_per_residue               |    0.74082 |            0.44974 |                              0.29107 |
| nonlocal_backbone_backbone_per_residue     |    0.58184 |            0.41135 |                              0.17049 |
| strong_nonlocal_fraction                   |    0.3788  |            0.38933 |                             -0.01053 |

#### Top Partial Correlations With log1p(v128), Strength

| feature                                |   spearman |   partial_spearman |   spearman_drop_after_length_control |
|:---------------------------------------|-----------:|-------------------:|-------------------------------------:|
| strong_nonlocal_per_residue            |    0.36563 |           -0.07797 |                              0.44359 |
| seq_class_nonlocal_per_residue         |    0.47196 |           -0.07212 |                              0.54407 |
| hbond_per_residue                      |    0.46918 |           -0.06991 |                              0.53909 |
| chem_type_N_to_O_per_residue           |    0.45919 |           -0.06771 |                              0.52691 |
| nonlocal_backbone_backbone_per_residue |    0.34024 |           -0.06704 |                              0.40728 |
| strong_nonlocal_fraction               |    0.11125 |           -0.05325 |                              0.1645  |
| strong_hbond_fraction                  |    0.02993 |           -0.04816 |                              0.07809 |
| seq_class_nonlocal_count               |    0.75551 |           -0.04536 |                              0.80087 |

Full table: `outputs/hbond_analysis/hbond_length_controlled_partial_correlations.csv`.

### Length-Stratified Results

The table below shows representative within-bin Spearman correlations. These values ask whether hydrogen-bond density/topology still matters among proteins of similar length.

| length_bin   |    n |   length_min |   length_max | feature                        | target     |   spearman |
|:-------------|-----:|-------------:|-------------:|:-------------------------------|:-----------|-----------:|
| Q1_short     | 1809 |           27 |           58 | hbond_per_residue              | log1p_v127 |     0.559  |
| Q1_short     | 1809 |           27 |           58 | hbond_per_residue              | log1p_v128 |     0.1298 |
| Q1_short     | 1809 |           27 |           58 | seq_class_nonlocal_per_residue | log1p_v127 |     0.5253 |
| Q1_short     | 1809 |           27 |           58 | seq_class_nonlocal_per_residue | log1p_v128 |     0.0903 |
| Q1_short     | 1809 |           27 |           58 | hbond_contact_order            | log1p_v127 |     0.0465 |
| Q1_short     | 1809 |           27 |           58 | hbond_contact_order            | log1p_v128 |    -0.1196 |
| Q2           | 1742 |           59 |           89 | hbond_per_residue              | log1p_v127 |     0.5781 |
| Q2           | 1742 |           59 |           89 | hbond_per_residue              | log1p_v128 |    -0.0027 |
| Q2           | 1742 |           59 |           89 | seq_class_nonlocal_per_residue | log1p_v127 |     0.5676 |
| Q2           | 1742 |           59 |           89 | seq_class_nonlocal_per_residue | log1p_v128 |     0.0284 |
| Q2           | 1742 |           59 |           89 | hbond_contact_order            | log1p_v127 |     0.2834 |
| Q2           | 1742 |           59 |           89 | hbond_contact_order            | log1p_v128 |     0.0271 |
| Q3           | 1751 |           90 |          125 | hbond_per_residue              | log1p_v127 |     0.4756 |
| Q3           | 1751 |           90 |          125 | hbond_per_residue              | log1p_v128 |    -0.1109 |
| Q3           | 1751 |           90 |          125 | seq_class_nonlocal_per_residue | log1p_v127 |     0.4884 |
| Q3           | 1751 |           90 |          125 | seq_class_nonlocal_per_residue | log1p_v128 |    -0.0915 |
| Q3           | 1751 |           90 |          125 | hbond_contact_order            | log1p_v127 |     0.2179 |
| Q3           | 1751 |           90 |          125 | hbond_contact_order            | log1p_v128 |    -0.0224 |
| Q4_long      | 1739 |          126 |          586 | hbond_per_residue              | log1p_v127 |     0.409  |
| Q4_long      | 1739 |          126 |          586 | hbond_per_residue              | log1p_v128 |     0.0727 |
| Q4_long      | 1739 |          126 |          586 | seq_class_nonlocal_per_residue | log1p_v127 |     0.3563 |
| Q4_long      | 1739 |          126 |          586 | seq_class_nonlocal_per_residue | log1p_v128 |     0.0158 |
| Q4_long      | 1739 |          126 |          586 | hbond_contact_order            | log1p_v127 |     0.0291 |
| Q4_long      | 1739 |          126 |          586 | hbond_contact_order            | log1p_v128 |    -0.229  |

Full table: `outputs/hbond_analysis/hbond_length_stratified_correlations.csv`.

### Matched-Length Pair Results

High-performance proteins were matched to low-performance proteins with sequence lengths within 5 residues. The table reports high-minus-low feature differences. Positive values mean the high-performance member has more of that hydrogen-bond descriptor despite comparable length.

| target   | feature                                |   n_pairs |   mean_high_minus_low |   median_high_minus_low |   ttest_p |
|:---------|:---------------------------------------|----------:|----------------------:|------------------------:|----------:|
| v127     | strong_nonlocal_fraction               |        38 |               0.21508 |                 0.25676 |   0       |
| v127     | hbond_per_residue                      |        38 |               0.18422 |                 0.20259 |   0       |
| v127     | seq_class_nonlocal_per_residue         |        38 |               0.14917 |                 0.15978 |   2e-05   |
| v127     | nonlocal_backbone_backbone_per_residue |        38 |               0.08033 |                 0.09217 |   0.00062 |
| v127     | hbond_contact_order                    |        38 |               0.02217 |                 0.01471 |   0.13133 |
| v128     | hbond_contact_order                    |        36 |               0.0104  |                 0.01763 |   0.47646 |
| v128     | strong_nonlocal_fraction               |        36 |              -0.00462 |                 0.011   |   0.885   |
| v128     | nonlocal_backbone_backbone_per_residue |        36 |              -0.06324 |                -0.06364 |   0.00444 |
| v128     | seq_class_nonlocal_per_residue         |        36 |              -0.06536 |                -0.04111 |   0.02485 |
| v128     | hbond_per_residue                      |        36 |              -0.0884  |                -0.04753 |   0.01194 |

Full matched-pair table: `outputs/hbond_analysis/hbond_matched_length_pairs.csv`.

### Model Ablation Results

The key test is whether hydrogen-bond descriptors improve test performance beyond a length-only model.

| model                            |   n_features |   toughness_v127/r2 |   strength_v128/r2 |   toughness_v127/spearman |   strength_v128/spearman |   delta_toughness_v127/r2 |   delta_strength_v128/r2 |
|:---------------------------------|-------------:|--------------------:|-------------------:|--------------------------:|-------------------------:|--------------------------:|-------------------------:|
| length_only                      |            1 |              0.538  |             0.2162 |                    0.8192 |                   0.8364 |                    0      |                   0      |
| length_plus_density_geometry     |           42 |              0.6672 |             0.317  |                    0.8943 |                   0.8392 |                    0.1292 |                   0.1008 |
| length_plus_residual_counts      |           28 |              0.6784 |             0.323  |                    0.901  |                   0.8458 |                    0.1404 |                   0.1068 |
| length_plus_topology             |           18 |              0.6788 |             0.3272 |                    0.9006 |                   0.8443 |                    0.1408 |                   0.111  |
| length_plus_all_hbond_controlled |           63 |              0.6873 |             0.3466 |                    0.9022 |                   0.8454 |                    0.1493 |                   0.1304 |

Full model metrics: `outputs/hbond_analysis/hbond_length_controlled_model_ablation_metrics.csv`.

Main plots:

- `outputs/hbond_analysis/plots/hbond_length_controlled_model_ablation_r2.png`
- `outputs/hbond_analysis/plots/hbond_partial_correlation_after_length_control.png`

### Conclusion

Length normalization is useful, but it is not enough on its own. The raw correlation analysis overestimates hydrogen-bond importance because raw count features strongly encode protein size. After controlling for length, the remaining hydrogen-bond signal becomes more specific: density, nonlocality, contact order, and strong nonlocal hydrogen bonds are more informative than total hydrogen-bond count alone.

The model ablation is the most decision-relevant result. If `length + controlled hydrogen-bond features` improves over `length_only`, then hydrogen bonding is not merely a proxy for sequence length. In this dataset, the controlled hydrogen-bond descriptors should be treated as interpretable structural features and can be tested as auxiliary inputs to the mechanical-property predictor or as physics-informed reward components.

For the next modeling step, prefer topology-aware and residualized hydrogen-bond descriptors rather than raw hydrogen-bond counts. In particular, `nonlocal_hbond_per_residue`, `hbond_contact_order`, `nonlocal_backbone_backbone_fraction`, and `strong_nonlocal_fraction` are more chemically meaningful candidates than `hbond_count` alone.


## Final Hydrogen-Bond Statistical Model

### Hypothesis

The previous length-controlled analysis suggested that the most chemically meaningful hydrogen-bond descriptors are not raw counts, but length-normalized and topology-aware features. The final statistical model tests whether a compact feature set can fit mechanical properties while remaining interpretable.

### Method

The model used the cached length-disentanglement table and selected seven features:

```text
sequence_length
hbond_per_residue
seq_class_nonlocal_per_residue
strong_nonlocal_fraction
strong_nonlocal_per_residue
nonlocal_backbone_backbone_per_residue
hbond_contact_order
```

Targets were trained in `log1p(v127), log1p(v128)` space and inverse-transformed for metrics. Four compact statistical baselines were compared:

1. `length_only_ridge`: length-only linear baseline.
2. `selected_hbond_ridge`: linear model using the selected hydrogen-bond features.
3. `selected_hbond_random_forest`: nonlinear random forest using the selected features.
4. `selected_hbond_extra_trees`: nonlinear extremely randomized trees using the selected features.

### Results

| model                        |   n_features |   toughness_v127/r2 |   strength_v128/r2 |   toughness_v127/spearman |   strength_v128/spearman |   mean/r2 |   mean/spearman |
|:-----------------------------|-------------:|--------------------:|-------------------:|--------------------------:|-------------------------:|----------:|----------------:|
| selected_hbond_random_forest |            7 |              0.6566 |             0.3021 |                    0.8897 |                   0.8324 |    0.4794 |          0.8611 |
| selected_hbond_extra_trees   |            7 |              0.6411 |             0.2712 |                    0.8814 |                   0.8313 |    0.4561 |          0.8564 |
| selected_hbond_ridge         |            7 |              0.5343 |             0.2376 |                    0.8576 |                   0.8415 |    0.386  |          0.8495 |
| length_only_ridge            |            1 |              0.3253 |             0.2313 |                    0.829  |                   0.8514 |    0.2783 |          0.8402 |

Best final model: `selected_hbond_random_forest`.

Feature importance for the best final model:

| feature                                |   permutation_importance_mean |   permutation_importance_std |
|:---------------------------------------|------------------------------:|-----------------------------:|
| sequence_length                        |                       0.69395 |                      0.03145 |
| hbond_per_residue                      |                       0.03068 |                      0.00474 |
| strong_nonlocal_per_residue            |                       0.03031 |                      0.00426 |
| seq_class_nonlocal_per_residue         |                       0.02714 |                      0.00481 |
| hbond_contact_order                    |                       0.0219  |                      0.00606 |
| nonlocal_backbone_backbone_per_residue |                       0.01981 |                      0.00634 |
| strong_nonlocal_fraction               |                       0.00522 |                      0.00319 |

Output files:

- `outputs/hbond_analysis/hbond_final_model_metrics.csv`
- `outputs/hbond_analysis/hbond_final_model_predictions.csv`
- `outputs/hbond_analysis/hbond_final_model_feature_importance.csv`
- `outputs/hbond_analysis/plots/hbond_final_model_test_r2.png`
- `outputs/hbond_analysis/plots/hbond_final_model_feature_importance.png`
- `outputs/hbond_analysis/plots/hbond_final_model_test_predictions.png`

### Conclusion

The compact selected hydrogen-bond model improves over the length-only baseline while using only a small number of interpretable structural descriptors. This supports the conclusion that hydrogen bonding contributes information beyond sequence length, especially for toughness.

The final model should be treated as an interpretable physics-informed baseline rather than a replacement for the ESM2 predictor. Its best use is to provide auxiliary features or diagnostic reward components. In particular, `hbond_per_residue`, nonlocal hydrogen-bond density, strong nonlocal hydrogen-bond fraction, and contact-order-like descriptors are good candidates for the next mechanical-property predictor version.

For reinforcement learning reward design, this result suggests a two-part structural prior:

```text
reward_hbond = density term + nonlocal/topology term + geometry-quality term
```

but the strength component should still be handled carefully, because strength remains more length- and scale-sensitive than toughness.

## Explanation of Final Hydrogen-Bond Features

This section explains the seven features used in the final compact hydrogen-bond statistical model. These features were chosen because they are interpretable and capture different layers of structural information:

```text
protein size
hydrogen-bond density
nonlocal network topology
hydrogen-bond geometric quality
backbone load-bearing architecture
```

### 1. `sequence_length`

#### Definition

`sequence_length` is the number of amino-acid residues in the protein sequence.

```text
sequence_length = len(Sequence)
```

#### Why It Matters

Length is not a hydrogen-bond descriptor, but it is a necessary control variable. Longer proteins naturally have:

1. more atoms;
2. more possible hydrogen-bond donors;
3. more possible hydrogen-bond acceptors;
4. more total contacts;
5. more opportunities to form nonlocal stabilizing interactions.

Therefore, raw hydrogen-bond counts tend to correlate strongly with sequence length.

#### Interpretation

If a model uses only raw hydrogen-bond count without length control, it may learn:

```text
more hydrogen bonds = longer protein = higher mechanical label
```

instead of learning:

```text
better hydrogen-bond network = better mechanical behavior
```

For this reason, `sequence_length` is included in the final model so that other hydrogen-bond features are interpreted relative to protein size.

#### Important Caution

In this dataset, `v128` strength is especially sensitive to length. So a strong relation between a feature and `v128` must be checked carefully: it may be a size effect rather than a hydrogen-bond-network effect.

### 2. `hbond_per_residue`

#### Definition

`hbond_per_residue` is the total number of detected hydrogen bonds normalized by sequence length.

```text
hbond_per_residue = hbond_count / sequence_length
```

Hydrogen bonds were detected using explicit hydrogen atoms with the geometric criteria:

```text
D-A distance <= 3.50 Å
H-A distance <= 2.70 Å
D-H-A angle >= 120 degrees
```

where:

```text
D = donor heavy atom
H = hydrogen attached to donor
A = acceptor atom
```

#### What It Measures

This feature measures hydrogen-bond density.

It asks:

```text
How many hydrogen bonds does this protein form per amino acid?
```

#### Why It Matters Mechanically

A higher hydrogen-bond density can imply a more internally connected structure. In mechanical pulling or deformation, this may:

1. increase resistance to unfolding;
2. provide more sacrificial interactions that break progressively;
3. distribute force across more contacts;
4. stabilize secondary-structure elements.

This is especially relevant for toughness, because toughness is related to energy absorption. A dense hydrogen-bond network can dissipate energy through sequential hydrogen-bond rupture.

#### Limitation

This feature does not distinguish local from nonlocal hydrogen bonds.

For example:

```text
alpha-helix i -> i+4 hydrogen bond
beta-sheet long-range hydrogen bond
sidechain-mediated tertiary hydrogen bond
```

may all contribute to `hbond_per_residue`, but they can have very different mechanical roles.

### 3. `seq_class_nonlocal_per_residue`

#### Definition

`seq_class_nonlocal_per_residue` is the number of nonlocal hydrogen bonds normalized by sequence length.

```text
seq_class_nonlocal_per_residue = nonlocal_hbond_count / sequence_length
```

In the current analysis, a hydrogen bond is classified as nonlocal when donor and acceptor are on the same chain but separated by more than 4 residues in sequence:

```text
nonlocal if abs(donor_resseq - acceptor_resseq) > 4
```

Hydrogen bonds with sequence separation <= 4 are classified as local.

#### What It Measures

This feature measures the density of long-range hydrogen-bond connections.

It asks:

```text
How many hydrogen bonds connect sequence-distant regions per amino acid?
```

#### Why It Matters Mechanically

Nonlocal hydrogen bonds are often more mechanically meaningful than local hydrogen bonds because they connect distant parts of the chain. These interactions can:

1. stabilize tertiary structure;
2. connect beta-strands;
3. form mechanical clamps;
4. resist chain extension;
5. delay unfolding under pulling.

In contrast, many local hydrogen bonds mainly stabilize local secondary structure, such as alpha helices. Local helices can unzip under force, while nonlocal beta-sheet or tertiary hydrogen-bond networks can be more load-bearing.

#### Interpretation

High `seq_class_nonlocal_per_residue` suggests that the protein has a more crosslinked hydrogen-bond network after controlling for length.

This is one of the most important final features because it moves beyond:

```text
how many hydrogen bonds?
```

toward:

```text
how many topology-relevant hydrogen bonds?
```

### 4. `strong_nonlocal_fraction`

#### Definition

`strong_nonlocal_fraction` is the fraction of all hydrogen bonds that are both nonlocal and geometrically strong.

```text
strong_nonlocal_fraction = strong_nonlocal_hbond_count / hbond_count
```

In this analysis, a hydrogen bond is considered strong when:

```text
D-A distance <= 3.0 Å
D-H-A angle >= 150 degrees
```

and it is considered nonlocal when:

```text
abs(donor_resseq - acceptor_resseq) > 4
```

Therefore:

```text
strong_nonlocal_hbond
  = nonlocal hydrogen bond
    with short donor-acceptor distance
    and near-linear D-H-A geometry
```

#### What It Measures

This feature measures the quality composition of the hydrogen-bond network.

It asks:

```text
Among all hydrogen bonds, what fraction are strong and nonlocal?
```

#### Why It Matters Mechanically

Hydrogen-bond strength depends strongly on geometry:

1. shorter donor-acceptor distance usually indicates stronger interaction;
2. more linear D-H-A angle usually indicates better orbital alignment and stronger directionality;
3. nonlocal placement makes the bond more likely to connect distant structural elements.

A high `strong_nonlocal_fraction` means the protein's hydrogen-bond network is not just dense, but enriched in potentially load-bearing interactions.

#### Difference From `strong_nonlocal_per_residue`

`strong_nonlocal_fraction` is a composition feature:

```text
What percentage of hydrogen bonds are strong and nonlocal?
```

It does not directly measure how many such bonds exist per residue.

For example, a small protein with 4 hydrogen bonds, 2 of which are strong nonlocal bonds, has:

```text
strong_nonlocal_fraction = 2 / 4 = 0.5
```

but it may still have fewer absolute strong nonlocal bonds than a larger protein.

### 5. `strong_nonlocal_per_residue`

#### Definition

`strong_nonlocal_per_residue` is the number of strong nonlocal hydrogen bonds normalized by sequence length.

```text
strong_nonlocal_per_residue = strong_nonlocal_hbond_count / sequence_length
```

where:

```text
strong_nonlocal_hbond_count
  = count of hydrogen bonds with
    abs(donor_resseq - acceptor_resseq) > 4
    D-A distance <= 3.0 Å
    D-H-A angle >= 150 degrees
```

#### What It Measures

This feature combines:

```text
density + topology + geometry quality
```

It asks:

```text
How many geometrically strong, sequence-nonlocal hydrogen bonds exist per amino acid?
```

#### Why It Matters Mechanically

This is one of the most chemically meaningful descriptors in the final model.

A high value suggests that the protein has many high-quality nonlocal hydrogen bonds relative to its size. These interactions may act as:

1. load-bearing contacts;
2. mechanical clamps;
3. unfolding barriers;
4. energy-dissipating rupture points;
5. stabilizers of beta-sheet or tertiary contact networks.

In the length-controlled analysis, this feature retained a strong partial correlation with toughness, meaning it explains information beyond sequence length.

#### Interpretation

Compared with raw `hbond_count`, this feature is much closer to a mechanistic hypothesis:

```text
proteins with more strong nonlocal hydrogen bonds per residue can absorb more mechanical energy
```

### 6. `nonlocal_backbone_backbone_per_residue`

#### Definition

`nonlocal_backbone_backbone_per_residue` is the number of nonlocal hydrogen bonds where both donor and acceptor heavy atoms are backbone atoms, normalized by sequence length.

```text
nonlocal_backbone_backbone_per_residue
  = nonlocal_backbone_backbone_hbond_count / sequence_length
```

Backbone atoms considered here include:

```text
N
O
OXT
```

A hydrogen bond contributes to this feature when:

```text
donor atom is backbone
acceptor atom is backbone
abs(donor_resseq - acceptor_resseq) > 4
```

#### What It Measures

This feature measures the density of nonlocal backbone-backbone hydrogen bonds.

It is closely related to regular secondary-structure and tertiary backbone organization, especially beta-sheet-like hydrogen-bond ladders.

#### Why It Matters Mechanically

Backbone-backbone hydrogen bonds can be mechanically important because:

1. they are geometrically regular;
2. they often form repeated ladders or sheets;
3. they can distribute force across multiple adjacent bonds;
4. in beta structures, they may resist extension more effectively than isolated sidechain hydrogen bonds.

For mechanical proteins, a nonlocal backbone-backbone hydrogen-bond network can behave like a structural scaffold.

#### Limitation

This feature does not know the pulling direction. A beta-sheet hydrogen-bond ladder aligned against the pulling direction may be much more mechanically relevant than one oriented differently.

So this descriptor is useful, but a future version should combine it with pulling-axis or terminal-vector geometry.

### 7. `hbond_contact_order`

#### Definition

`hbond_contact_order` measures the average sequence separation of hydrogen bonds normalized by sequence length.

For each hydrogen bond:

```text
normalized_seq_sep = abs(donor_resseq - acceptor_resseq) / sequence_length
```

Then:

```text
hbond_contact_order = mean(normalized_seq_sep over all hydrogen bonds)
```

Interchain hydrogen bonds or undefined sequence separations are ignored in this mean.

#### What It Measures

This feature measures the topological range of the hydrogen-bond network.

It asks:

```text
On average, how far apart in sequence are hydrogen-bonded residues?
```

Low value:

```text
mostly local hydrogen bonds
```

High value:

```text
many long-range hydrogen bonds connecting distant sequence regions
```

#### Why It Matters Mechanically

Contact order is a classic protein-topology idea. High contact order often indicates that distant sequence regions are brought together in the folded structure.

For mechanics, high hydrogen-bond contact order may indicate:

1. long-range structural coupling;
2. tertiary mechanical clamps;
3. beta-sheet-like architecture;
4. resistance to simple local unfolding;
5. more complex unfolding pathways.

#### Interpretation

`hbond_contact_order` is not a count. It is a topology descriptor.

A protein can have:

```text
few hydrogen bonds but high contact order
```

or:

```text
many hydrogen bonds but low contact order
```

These two situations may produce very different mechanical behavior.

### Summary of Feature Roles

| Feature | Main Role | Main Mechanical Interpretation |
|:--|:--|:--|
| `sequence_length` | size control | separates scale effects from network effects |
| `hbond_per_residue` | density | total hydrogen-bond network density |
| `seq_class_nonlocal_per_residue` | topology-adjusted density | density of long-range hydrogen-bond links |
| `strong_nonlocal_fraction` | composition/quality | fraction of the network that is strong and nonlocal |
| `strong_nonlocal_per_residue` | density + topology + geometry | number of strong nonlocal load-bearing candidates per residue |
| `nonlocal_backbone_backbone_per_residue` | backbone architecture | beta-sheet/backbone scaffold-like hydrogen bonding |
| `hbond_contact_order` | topology | average sequence range of hydrogen-bond contacts |

### Practical Interpretation for This Project

For mechanical-protein reward design, these features suggest that hydrogen bonding should not be rewarded as a simple raw count.

A better reward prior is:

```text
encourage enough hydrogen-bond density
encourage nonlocal hydrogen-bond topology
encourage strong/linear hydrogen-bond geometry
encourage backbone-backbone long-range organization
avoid rewarding length alone
```

In other words:

```text
reward_hbond should prefer quality and topology over raw quantity.
```
