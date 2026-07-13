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
| train   | 5632 |              0.8821 |              20.3237 |               35.153  |                    0.9676 |             0.441  |             45.3241 |              206.752 |                   0.9467 |
| val     |  704 |              0.7087 |              37.2362 |               51.4405 |                    0.8686 |             0.2101 |             70.0571 |              195.343 |                   0.8123 |
| test    |  705 |              0.6629 |              38.0195 |               62.8774 |                    0.8933 |             0.3035 |             66.4775 |              180.56  |                   0.8469 |

### Top Random-Forest Features

| feature                                |   importance |
|:---------------------------------------|-------------:|
| sequence_length                        |      0.09998 |
| hbond_count                            |      0.09941 |
| chem_type_N_to_O_count                 |      0.09702 |
| seq_class_nonlocal_count               |      0.0945  |
| role_type_backbone_to_backbone_count   |      0.08239 |
| seq_class_local_count                  |      0.03951 |
| chem_type_N_to_O_per_residue           |      0.03511 |
| role_type_sidechain_to_backbone_count  |      0.03062 |
| hbond_per_residue                      |      0.03053 |
| role_type_sidechain_to_sidechain_count |      0.03052 |
| role_type_backbone_to_sidechain_count  |      0.02721 |
| seq_class_nonlocal_per_residue         |      0.02535 |

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


## Dataset Split Comparison for Hydrogen-Bond Statistical Models

### Current Split Used Before This Comparison

The previous `hbond_final.ipynb` model used a random split implemented with `sklearn.model_selection.train_test_split`:

```text
train: 80%
validation: 10%
test: 10%
random_state: 7
```

This is different from the earlier sequence-to-property deep-learning predictor, where we explicitly supported both random split and sequence-similarity split. Random split can place highly similar sequences into both training and test sets, so it usually estimates interpolation performance rather than out-of-family generalization.

### Hypothesis

If hydrogen-bond features mainly capture broad structural rules, performance should remain reasonably stable under sequence-similarity split. If performance drops strongly, the model is partly relying on similarity between train and test proteins.

### Method

Two notebooks were created:

```text
dataset-random.ipynb
dataset-similarity.ipynb
```

Both notebooks used the same final hydrogen-bond feature set:

```text
sequence_length
hbond_per_residue
seq_class_nonlocal_per_residue
strong_nonlocal_fraction
strong_nonlocal_per_residue
nonlocal_backbone_backbone_per_residue
hbond_contact_order
```

The random split used random 80/10/10 partitioning. The sequence-similarity split used the project's existing k-mer Jaccard grouping method:

```text
kmer_size = 5
similarity_threshold = 0.5
train/validation/test group split = approximately 80/10/10
```

Random split sizes:

```text
{'test': 705, 'train': 5632, 'val': 704}
```

Sequence-similarity split sizes:

```text
{'train': 5633, 'val': 704, 'test': 704}
```

### Results

| model                        |   random/toughness_v127/r2 |   similarity/toughness_v127/r2 |   similarity_minus_random/toughness_v127/r2 |   random/strength_v128/r2 |   similarity/strength_v128/r2 |   similarity_minus_random/strength_v128/r2 |   random/mean/r2 |   similarity/mean/r2 |   similarity_minus_random/mean/r2 |
|:-----------------------------|---------------------------:|-------------------------------:|--------------------------------------------:|--------------------------:|------------------------------:|-------------------------------------------:|-----------------:|---------------------:|----------------------------------:|
| selected_hbond_random_forest |                     0.6574 |                         0.631  |                                     -0.0263 |                    0.3062 |                        0.2211 |                                    -0.0851 |           0.4818 |               0.4261 |                           -0.0557 |
| selected_hbond_extra_trees   |                     0.6411 |                         0.6024 |                                     -0.0387 |                    0.2712 |                        0.1578 |                                    -0.1134 |           0.4561 |               0.3801 |                           -0.0761 |
| selected_hbond_ridge         |                     0.5343 |                         0.595  |                                      0.0607 |                    0.2376 |                        0.1734 |                                    -0.0642 |           0.386  |               0.3842 |                           -0.0018 |
| length_only_ridge            |                     0.3253 |                         0.421  |                                      0.0957 |                    0.2313 |                        0.1706 |                                    -0.0608 |           0.2783 |               0.2958 |                            0.0174 |

Output files:

- `outputs/hbond_analysis/dataset_random_hbond_model_metrics.csv`
- `outputs/hbond_analysis/dataset_similarity_hbond_model_metrics.csv`
- `outputs/hbond_analysis/dataset_split_hbond_model_comparison_metrics.csv`
- `outputs/hbond_analysis/dataset_split_hbond_model_comparison_test_wide.csv`
- `outputs/hbond_analysis/plots/dataset_split_random_vs_similarity_test_r2.png`

### Analysis

The random split result is the easier evaluation setting because homologous or very similar sequences may appear across train and test. The sequence-similarity split is closer to an out-of-family generalization test.

If the similarity-split metrics are close to random-split metrics, the selected hydrogen-bond features are likely capturing general physical structure-property rules. If the similarity-split metrics drop, the model has weaker extrapolation to dissimilar sequence families.

For this hydrogen-bond statistical model, the comparison should be interpreted as a structural-feature OOD check rather than a final predictor benchmark. These seven features are intentionally compact and cannot encode the full sequence/fold information that ESM2 captures.

### Conclusion

The current hydrogen-bond final model was originally evaluated with random split. The sequence-similarity split comparison is the more conservative result and should be preferred when deciding whether hydrogen-bond descriptors generalize across protein families.

If hydrogen-bond features remain useful under similarity split, they are good candidates for physics-informed auxiliary inputs to the ESM2 mechanical-property predictor. If they degrade under similarity split, they should still be useful for within-family diagnostics and reward shaping, but should not be trusted as a standalone OOD predictor.
