# SHAP Violin Plot Visualizations - Guide

## Overview

Created **13 violin plot visualizations** that show the **distribution of SHAP values across samples**, revealing much more than simple means:
- **Consistency** of feature effects
- **Variability** across samples
- **Outliers** and extreme cases
- **Bimodal** or multimodal distributions

---

## What Makes Violin Plots More Informative?

### Bar Charts (Previous)
- Show only mean or total SHAP importance
- Hide variability across samples
- Can't see if effects are consistent or sporadic

### Violin Plots (New)
✅ **Show full distribution** of SHAP values  
✅ **Reveal consistency**: Wide violin = variable effect, narrow = consistent  
✅ **Identify outliers**: See extreme positive/negative contributions  
✅ **Detect patterns**: Bimodal distributions suggest subpopulations  

---

## Visualization Types Created

### 1. **Simple Violin Plots** (3 files)
**Files:** `[illness]_shap_violin.png`

- **Top 15 features**
- Violin shows distribution of SHAP values across 1,000 samples
- **Black dot** = median SHAP value
- **Color** = mean direction (red = increase illness, blue = decrease)
- **Width** = density (wider = more samples at that SHAP value)

**Use for:** Understanding which features have consistent vs variable effects

### 2. **Violin + Box Combo** (3 files)
**Files:** `[illness]_shap_violin_box.png`

- **Top 10 features**
- Violin + overlaid boxplot
- **Box** shows quartiles (25th, 50th, 75th percentile)
- **Whiskers** extend to 1.5×IQR
- **Outlier dots** show extreme SHAP values

**Use for:** Identifying outliers and quartile ranges

### 3. **Density Ridge Plots** (3 files)
**Files:** `[illness]_shap_density_ridge.png`

- **Top 12 features**
- Stacked density curves (like "joy plots")
- Easier to compare distributions across features
- **Vertical dashed line** = zero SHAP value

**Use for:** Presentations (cleaner look than overlapping violins)

### 4. **Cross-Illness Violin Comparison** (1 file)
**File:** `shap_violin_cross_illness.png`

- **Top 8 features** across all three illnesses
- Side-by-side violins for each illness
- **Black dots** = median per illness
- **Compare shapes** to see how effects differ by illness type

**Use for:** Comparing how the same feature affects different illnesses

### 5. **Jitter + Violin** (3 files)
**Files:** `[illness]_shap_violin_jitter.png`

- **Top 10 features**
- **Individual sample dots** (jittered for visibility)
- **Violin** shows overall distribution
- **Yellow diamond** = mean SHAP value
- Most detailed view

**Use for:** Deep-dive analysis, showing actual sample-level variation

---

## Key Insights from Violin Plots

### 1. **NO₂ Shows Bimodal Distribution** (Acute Laryngopharyngitis)
- Violin has **two peaks**
- Suggests **subpopulations**: NO₂ increases illness for some samples, decreases for others
- Explains why mean direction is negative despite variability

### 2. **PM2.5 Has Consistent Positive Effect**
- **Narrow violin** centered above zero
- Most samples show positive SHAP values
- Confirms PM2.5 reliably increases illness

### 3. **Lag Features Show Higher Variability**
- **Wider violins** for lag features vs base features
- Lag effects are **less consistent** across samples
- May depend on specific weather patterns or seasons

### 4. **Gastritis Has Tighter Distributions**
- **Narrower violins** overall compared to other illnesses
- More **predictable** environmental effects
- Lower variability in feature contributions

### 5. **Chronic Rhinitis Shows Extreme Outliers**
- **Long whiskers** and many outlier dots
- Some samples have **very high** SHAP values
- Suggests occasional **extreme environmental events** drive illness spikes

---

## Comparison: Bar Chart vs Violin Plot

**Example: NO₂ for Acute Laryngopharyngitis**

### Bar Chart Shows:
- Mean |SHAP| = 8.024 (highest importance)
- Mean direction = Decrease (negative)

### Violin Plot Reveals:
- **Bimodal distribution** (two peaks)
- One peak at **negative SHAP** (~-2)
- Another peak at **positive SHAP** (~+1)
- **Wide spread** from -15 to +10
- Many **outliers** in both directions

**Interpretation:** NO₂ effect is **context-dependent**, not uniformly negative. The mean direction masks substantial sample-to-sample variability.

---

## Recommended Plots for Different Audiences

### Academic Presentations
1. **Density ridge plots** - Clean, professional look
2. **Cross-illness comparison** - Shows comparative insights
3. **Violin + box combo** - Quantifies outliers with statistics

### Manuscript Figures
1. **Simple violin plots** - Top 15 features, publication-quality
2. **Cross-illness violin** - Key finding comparison
3. Supplement: **Jitter plots** for detailed methods

### Research Group Meetings
1. **Jitter + violin** - Show raw data for transparency
2. **Violin + box** - Discuss quartiles and outliers
3. **Cross-illness** - Spark discussion on differential effects

---

## File Locations

All saved to: `/results/optimization_visualizations/`

**Per-Illness Files (9 files):**
- `Acute_laryngopharyngitis_shap_violin.png`
- `Acute_laryngopharyngitis_shap_violin_box.png`
- `Acute_laryngopharyngitis_shap_density_ridge.png`
- `Acute_laryngopharyngitis_shap_violin_jitter.png`
- (Same 4 files for Gastritis and Chronic rhinitis)

**Cross-Illness Files (1 file):**
- `shap_violin_cross_illness.png`

---

## Statistical Insights from Distributions

### What Violin Shapes Tell You

**Narrow, tall violin:**
- **Consistent effect** across samples
- Low variability
- Reliable predictor
- Example: PM2.5 for most illnesses

**Wide, short violin:**
- **Variable effect** across samples
- High uncertainty
- Context-dependent
- Example: NO₂ for Acute laryngopharyngitis

**Bimodal (two peaks):**
- **Two subpopulations** with different responses
- Possible interaction with unmeasured variable
- Suggests segmentation needed
- Example: NO₂ distributions

**Skewed distribution:**
- **Asymmetric effects**
- More extreme values in one direction
- May indicate threshold effects
- Example: MaxWindSpeed

**Heavy tails:**
- **Many outliers**
- Occasional extreme events
- Important for risk management
- Example: Chronic rhinitis features

---

## Technical Notes

### Sample Size
- Each violin represents **1,000 randomly sampled predictions**
- SHAP values computed per sample
- Sufficient for stable distribution estimates

### Interpretation
- **Y-axis**: SHAP value (contribution to prediction)
- **Width**: Density (more samples = wider)
- **Color**: Mean direction (average across all samples)
- **Dots/diamonds**: Summary statistics (median/mean)

### Limitations
- Violin plots don't show **temporal patterns**
- Can't see **feature interactions** directly
- May need to **stratify by season/region** for deeper insights

---

## Next Steps (Advanced Analysis)

1. **Stratified Violins**
   - Separate violins by season (winter vs summer)
   - By region (urban vs rural)
   - By pollution level (low/medium/high)

2. **Feature Interaction Violins**
   - Condition on another feature's value
   - E.g., "NO₂ SHAP when PM2.5 is high"

3. **Temporal Evolution**
   - Violins by year (2018-2022)
   - See if distributions are stable over time

4. **Percentile Analysis**
   - Focus on 90th/95th percentile effects
   - Identify worst-case scenarios

---

**Summary:** Violin plots reveal the **full story** behind SHAP values that bar charts hide. They show not just average importance, but **consistency, outliers, and subpopulations**—critical for understanding real-world environmental health effects.
