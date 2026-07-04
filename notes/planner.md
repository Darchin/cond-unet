# Algorithmic Selection of Target Spacing, Stem Stride, and Patch Size

## 1. Purpose

This document describes a rule-based procedure for selecting three architecture- and preprocessing-related parameters for medical image segmentation experiments within the nnU-Net framework:

1. the target image spacing,
2. the stride of the network stem, and
3. the input patch size.

The goal is to preserve compatibility with nnU-Net’s automated, planner-based ecosystem while introducing explicit control over early downsampling and patch geometry. Unlike the standard nnU-Net preset logic, where the M, L, and XL configurations primarily differ by their allotted VRAM budget and therefore by their patch size, the proposed presets are defined primarily by the desired stem downsampling factor.

## 2. Presets

We define three presets:

| Preset | Interpretation | Stem downsampling factor |
| ------ | -------------- | ------------------------ |
| `2x`   | heavy          | 2                        |
| `3x`   | medium         | 3                        |
| `4x`   | light          | 4                        |

Let the stem downsampling factor be denoted by (F), where:

[
F \in {2, 3, 4}.
]

The `2x` preset is considered the heaviest because it performs the least aggressive early downsampling, preserving a larger feature map after the stem. Conversely, the `4x` preset is considered the lightest because it applies the strongest early downsampling.

For all presets, all encoder stages after the stem use an isotropic stride of (2) along every axis. The stem stride is determined separately and may be isotropic or anisotropic depending on the dataset spacing.

The procedure is formulated for an arbitrary number of spatial dimensions (D), although the common use case is three-dimensional medical image segmentation.

## 3. Notation and Definitions

Let (s_{50}) denote the median per-axis spacing of the dataset:

[
s_{50} = (s_{50}[0], s_{50}[1], \ldots, s_{50}[D-1]).
]

The reference spacing (r) is defined as the smallest median spacing across axes:

[
r = \min_i s_{50}[i].
]

This corresponds to the sharpest axis of the dataset at median spacing.

For any spacing tuple (s), the per-axis anisotropy ratio is defined as:

[
a[i] = \frac{s[i]}{r}.
]

An axis with (a[i] = 1) has the reference spacing. Larger values indicate coarser spacing relative to the sharpest axis.

We also define the set of reference axes:

[
\mathcal{R} = {i : s_{50}[i] = r}.
]

If multiple axes share the sharpest median spacing, all of them are treated as reference axes.

## 4. Target Spacing and Stem Stride Selection

The target spacing and stem stride are selected in two stages. First, the target spacing is adjusted when necessary to reduce severe input-space anisotropy. Second, the stem stride is selected to reduce anisotropy after the stem downsampling operation.

### 4.1 Initial Target Spacing

The initial target spacing is the dataset median spacing:

[
t = s_{50}.
]

Here, (t) denotes the final target spacing to which images will be resampled.

### 4.2 Anisotropy Handling

The algorithm handles anisotropy in two steps:

1. adjust the target spacing for highly anisotropic axes;
2. adjust the stem stride to reduce anisotropy after stem downsampling.

The procedure operates independently on each spatial axis.

---

## 5. Step 1: Target Spacing Adjustment

The target spacing is adjusted only for axes whose anisotropy ratio exceeds the preset’s stem downsampling factor (F):

[
\frac{s_{50}[i]}{r} > F.
]

For such axes, the algorithm searches lower spacing percentiles to obtain a sharper target spacing.

Let (s_p[i]) denote the (p)-th percentile of the dataset spacing along axis (i). The search starts from the median percentile (p = 50) and proceeds downward in steps of (5%), for example:

[
50, 45, 40, 35, 30, 25.
]

The lower bound is a configurable parameter. A reasonable default is:

[
p_{\min} = 25.
]

For each candidate percentile (p), compute the candidate anisotropy ratio:

[
a_p[i] = \frac{s_p[i]}{r}.
]

As the percentile decreases, the candidate spacing is expected to stay the same or become sharper, so (a_p[i]) is expected to stay the same or decrease.

The selection rule is:

1. Search from (p=50) down to (p_{\min}).
2. If (a_p[i]) drops below (F), select the percentile immediately before the crossing.
3. If (a_p[i]) never drops below (F), use the spacing at (p_{\min}).
4. If the axis does not exceed the anisotropy threshold at the median, keep its median spacing.

This yields a target spacing (t[i]) for every axis.

The purpose of this step is not necessarily to make the input spacing fully isotropic. Instead, it reduces extreme anisotropy while preserving a controlled relationship between the spacing and the preset-specific downsampling factor.

---

## 6. Step 2: Stem Stride Selection

After the target spacing (t) has been selected, the stem stride (b) is determined.

The stride is initialized isotropically:

[
b = (F, F, \ldots, F).
]

The sharpest axis or axes, defined by (\mathcal{R}), are fixed to the maximum stride:

[
b[i] = F \quad \text{for all } i \in \mathcal{R}.
]

This gives the post-stem reference spacing:

[
r_{\text{stem}} = r \cdot F.
]

For every non-reference axis, candidate strides are evaluated from (1) to (F):

[
c \in {1, 2, \ldots, F}.
]

For a candidate stride (c) on axis (i), the post-stem spacing is:

[
t[i] \cdot c.
]

The corresponding post-stem anisotropy ratio is:

[
q_i(c) = \frac{t[i] \cdot c}{r_{\text{stem}}}.
]

The selected stride is the candidate that minimizes post-stem anisotropy relative to the post-stem reference spacing. Candidates are not rejected when they make an axis sharper than the reference spacing after the stem; they are still valid if they produce the lowest anisotropy.

[
\alpha_i(c) = \max(q_i(c), q_i(c)^{-1})
]

The final rule is:

[
b[i] = \arg\min_{c \in {1, \ldots, F}} \alpha_i(c).
]

Reference axes are still fixed to (F).

### Example

Suppose the target spacing is:

[
t = (2.0, 1.5, 1.0),
]

and we use the `3x` preset, so (F = 3).

The reference spacing is:

[
r = 1.0.
]

The sharpest axis receives stride (3), so the post-stem reference spacing is:

[
r_{\text{stem}} = 1.0 \cdot 3 = 3.0.
]

For the first axis, whose spacing is (2.0), the candidate strides are (1), (2), and (3):

| Candidate stride | Post-stem spacing | Post-stem anisotropy |
| ---------------- | ----------------- | -------------------- |
| 1                | (2.0)             | (0.67)               |
| 2                | (4.0)             | (1.33)               |
| 3                | (6.0)             | (2.00)               |

The stride (1) candidate is rejected because it would make the axis sharper than the post-stem reference spacing. Between the remaining candidates, stride (2) gives the lower anisotropy ratio, so the selected stride for this axis is (2).

---

## 7. Patch Size Selection

Once the target spacing and stem stride have been selected, the patch size is determined.

The patch-size rule is based on three quantities:

1. the target spacing (t),
2. the median image resolution after resampling to (t), and
3. the cumulative downsampling factor of the network.

The goal is to produce patch sizes that are valid with respect to the network’s downsampling structure while preserving a dataset-specific physical aspect ratio.

## 8. Physical Aspect Ratio

Let (m) denote the median image resolution after resampling all images to the target spacing (t):

[
m = (m[0], m[1], \ldots, m[D-1]).
]

The median physical image extent is computed by elementwise multiplication:

[
e = m \odot t.
]

That is:

[
e[i] = m[i] \cdot t[i].
]

The physical aspect ratio (A) is obtained by normalizing the physical extent so that the smallest axis has value (1):

[
A_{\text{float}}[i] = \frac{e[i]}{\min_j e[j]}.
]

The final aspect ratio tuple is obtained by rounding each value to the nearest integer:

[
A[i] = \operatorname{round}(A_{\text{float}}[i]).
]

Each element should be at least (1).

### Example

Suppose the median resampled resolution is:

[
m = (200, 130, 300),
]

and the target spacing is:

[
t = (1.0, 2.5, 5.0)\ \text{mm}.
]

The physical extent is:

[
e = (200, 325, 1500)\ \text{mm}.
]

Normalizing by the smallest value gives:

[
A_{\text{float}} = (1.0, 1.625, 7.5).
]

After rounding, the aspect ratio is:

[
A = (1, 2, 8).
]

---

## 9. Patch Unit

Let (b) be the selected stem stride.

Let (d_{\text{enc}}) denote the cumulative stride of the encoder stages after the stem. Since all post-stem encoder stages use isotropic stride (2), this value is typically:

[
d_{\text{enc}} = (2^L, 2^L, \ldots, 2^L),
]

where (L) is the number of post-stem downsampling stages.

The total input-to-bottleneck stride is:

[
d_{\text{total}} = b \odot d_{\text{enc}}.
]

The patch unit is then defined as:

[
u = d_{\text{total}} \odot A.
]

The patch unit links the architectural downsampling constraints with the dataset’s physical aspect ratio. Any patch size constructed as an integer multiple of this unit will preserve the desired aspect-ratio structure while remaining compatible with the network’s downsampling hierarchy.

---

## 10. Patch Size Construction

The patch size is constructed by scaling the patch unit with a single integer multiplier.

Let (m) denote the patch size multiplier:

[
m \in \mathbb{Z}^+, \quad m \geq 1.
]

The final patch size is defined as:

[
P = m \cdot u.
]

This formulation ensures that all patch sizes are geometrically similar, differing only by a uniform scaling factor. As (m) increases, the patch grows proportionally along all axes while preserving the aspect ratio defined by (u).

The multiplier (m) serves as the primary user-facing control for adjusting memory usage and field of view. Larger values of (m) result in larger patches and higher memory consumption.

---

## 11. Summary of the Full Procedure

Given a dataset and a preset (F \in {2, 3, 4}):

1. Compute the median per-axis spacing (s_{50}).

2. Define the reference spacing:

   [
   r = \min_i s_{50}[i].
   ]

3. Compute the per-axis anisotropy ratios:

   [
   a[i] = \frac{s_{50}[i]}{r}.
   ]

4. For each highly anisotropic axis, search lower spacing percentiles to reduce the target spacing.

5. Determine the stem stride per axis by minimizing post-stem anisotropy.

6. Resample the dataset to the selected target spacing.

7. Compute the median resampled image resolution.

8. Convert the median resampled resolution into a physical aspect ratio (A).

9. Compute the total input-to-bottleneck stride:

   [
   d_{\text{total}} = b \odot d_{\text{enc}}.
   ]

10. Compute the patch unit:

[
u = d_{\text{total}} \odot A.
]

11. Select a patch size multiplier (m).

12. Compute the final patch size:

[
P = m \cdot u.
]

---

## 12. Configurable Parameters

The following quantities should be exposed as configurable planner parameters:

| Parameter  | Meaning                                   | Suggested default |
| ---------- | ----------------------------------------- | ----------------- |
| (F)        | preset-specific stem downsampling factor  | `2`, `3`, or `4`  |
| (p_{\min}) | lower percentile bound for spacing search | `25`              |
| (\Delta p) | spacing percentile search step            | `5`               |
| (m)        | patch size multiplier                     | `1`               |

These defaults are intended to provide a reasonable initial rule set while allowing subclasses or experiments to override individual choices.

---

## 13. Implementation Notes

The procedure should be implemented in a dimension-agnostic way. No part of the algorithm fundamentally depends on the data being three-dimensional.

The target spacing search should be performed per axis. The reference spacing should remain fixed during the search to keep the procedure independent across axes.

When multiple axes share the smallest median spacing, all of them should be treated as reference axes and assigned the maximum stem stride (F).

The stem stride selection should explicitly evaluate all candidate strides from (1) to (F), with reference axes fixed to (F). This is necessary because the selected candidate may make a non-reference axis sharper than the post-stem reference spacing when doing so minimizes anisotropy.

The patch size should be constructed as a scaled version of the patch unit. This ensures that the patch remains compatible with the cumulative downsampling structure of the network while preserving the dataset-specific physical aspect ratio.

The multiplier (m) acts as the primary user-facing control for memory usage and field of view. Increasing (m) increases the patch size uniformly across all axes while maintaining the same aspect ratio.
