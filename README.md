# Lecture 7 — Optimize Distance Equation (Contrastive Learning) Remake

---

## 📖 Overview

Complete reimplementation with full class details, gradient analysis, and 4-vector test suite.

| Class | Role | Time | Space |
|-------|------|------|-------|
| `FeatureVector` | Immutable n-dim embedding wrapper | O(1) | O(n) |
| `L1Distance` | `\|\|x1-x2\|\|_1` Manhattan, per-dim breakdown | O(n) | O(1) |
| `L2Distance` | `\|\|x1-x2\|\|_2` Euclidean + squared variant | O(n) | O(1) |
| `CosineSimilarity` | Angle metric + cosine distance | O(n) | O(1) |
| `ContrastiveLoss` | `L=max(0,C-\|\|D(x1)-D(x2)\|\|_2)` + full Hadsell loss + sweep | O(n) | O(1) |
| `GradientAnalysis` | `dL/dD(x1)`, push direction, simulated update | O(n) | O(n) |

---

## 🚀 Compile & Run

```bash
cd src
javac -encoding UTF-8 ContrastiveLearning.java
java -Dfile.encoding=UTF-8 ContrastiveLearning
```

---

## 🔬 Key Formulas

### Assignment 1 — Contrastive Loss (Lecture Slide)
```
L = max(0, C - ||D(x1) - D(x2)||_2)
```

### L1 Distance
```
||x1-x2||_1 = Σ |x1_i - x2_i|      Time: O(n)
```

### L2 Distance
```
||x1-x2||_2 = sqrt( Σ (x1_i-x2_i)^2 )   Time: O(n)
```

### Cosine Similarity
```
cos(a,b) = (a·b) / (||a||_2 × ||b||_2)   ∈ [-1,1]   Time: O(n)
```

### Full Hadsell et al. (2006) Loss
```
y=1 (positive): L = ||D(x1)-D(x2)||_2^2          (pull together)
y=0 (negative): L = max(0, C - ||D(x1)-D(x2)||_2)^2  (push apart)
```

### Gradient of L w.r.t. D(x1) (when L > 0)
```
dL/dD(x1) = -(D(x1)-D(x2)) / ||D(x1)-D(x2)||_2
Push direction: (D(x1)-D(x2)) / ||D(x1)-D(x2)||_2   (unit vector)
Update: D(x1) += eta × push_direction
```

---

## 📊 Results (C = 2.0, 4-dim vectors)

| Metric | pos pair | neg (far) | neg (close) |
|--------|----------|-----------|-------------|
| L1 | 0.4000 | 11.5000 | 2.0000 |
| L2 | 0.2000 | 6.2650 | 1.0000 |
| Cosine | 0.9994 | 0.7783 | 0.9980 |
| **Loss** | **1.8000** | **0.0000 ✓** | **1.0000** |

### Loss Sweep (C=2.0)
| L2 dist | Loss L | Full(y=0) | Status |
|---------|--------|-----------|--------|
| 0.0 | 2.0000 | 4.0000 | penalized |
| 1.0 | 1.0000 | 1.0000 | penalized |
| 2.0 | 0.0000 | 0.0000 | ✓ no penalty |
| 3.0 | 0.0000 | 0.0000 | ✓ no penalty |

---

## 📚 References
- Lecture 7: Optimize Distance Equation — 黃祥睿 (Xiang-Rui Huang), NPUST
- R. Hadsell, S. Chopra, Y. LeCun, "Dimensionality Reduction by Learning an Invariant Mapping," CVPR 2006
- T. Chen et al., "A Simple Framework for Contrastive Learning," ICML 2020