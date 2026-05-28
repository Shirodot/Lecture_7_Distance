/**
 * ContrastiveLearning.java  (Lecture 7)
 *
 * Optimize Distance Equation — Contrastive Learning
 *
 * Implements ALL details of each class with code, printing time complexity.
 *
 * Classes:
 *   FeatureVector      — immutable n-dim embedding wrapper
 *   L1Distance         — ||x1-x2||_1  Manhattan,     O(n)
 *   L2Distance         — ||x1-x2||_2  Euclidean,     O(n)
 *   CosineSimilarity   — dot/(|a||b|), angle metric,  O(n)
 *   ContrastiveLoss    — L=max(0,C-||D(x1)-D(x2)||_2), O(n)
 *   GradientAnalysis   — dL/d||·|| and push direction, O(n)
 *   ContrastiveLearning— main demo runner
 *
 */
public class ContrastiveLearning {

    public static void main(String[] args) {
        printHeader();

        // ── Define feature vectors D(x) ───────────────────────────
        // Anchor, Positive (similar), Negative (dissimilar)
        FeatureVector Dx1 = new FeatureVector("D(x1)[anchor  ]",
                new double[]{1.0, 2.0, 3.0, 4.0});
        FeatureVector Dx2 = new FeatureVector("D(x2)[positive]",
                new double[]{1.1, 2.1, 2.9, 4.1});
        FeatureVector Dx3 = new FeatureVector("D(x3)[negative]",
                new double[]{5.0, 1.0, 0.5, 8.0});
        FeatureVector Dx4 = new FeatureVector("D(x4)[neg-close]",
                new double[]{1.5, 2.5, 3.5, 4.5});   // negative but nearby

        System.out.println("  Feature vectors (encoder embeddings):");
        Dx1.print(); Dx2.print(); Dx3.print(); Dx4.print();

        // ── 1. L1 Distance ────────────────────────────────────────
        section("1. L1 Distance  ||x1-x2||_1  (Manhattan)");
        L1Distance l1 = new L1Distance();
        l1.demo(Dx1, Dx2, "positive pair");
        l1.demo(Dx1, Dx3, "negative pair (far)");
        l1.demo(Dx1, Dx4, "negative pair (close)");
        l1.printComplexity();

        // ── 2. L2 Distance ────────────────────────────────────────
        section("2. L2 Distance  ||x1-x2||_2  (Euclidean)");
        L2Distance l2 = new L2Distance();
        l2.demo(Dx1, Dx2, "positive pair");
        l2.demo(Dx1, Dx3, "negative pair (far)");
        l2.demo(Dx1, Dx4, "negative pair (close)");
        l2.printComplexity();

        // ── 3. Cosine Similarity ──────────────────────────────────
        section("3. Cosine Similarity  cos(x1,x2)");
        CosineSimilarity cos = new CosineSimilarity();
        cos.demo(Dx1, Dx2, "positive pair");
        cos.demo(Dx1, Dx3, "negative pair (far)");
        cos.demo(Dx1, Dx4, "negative pair (close)");
        cos.printComplexity();

        // ── 4. Contrastive Loss ───────────────────────────────────
        section("4. Contrastive Loss  L = max(0, C - ||D(x1)-D(x2)||_2)");
        double C = 2.0;
        ContrastiveLoss cl = new ContrastiveLoss(C, l2);
        System.out.println("  Margin C = " + C);
        System.out.println();
        cl.demo(Dx1, Dx2, "positive pair (should be pulled together)");
        cl.demo(Dx1, Dx3, "negative pair (far)  — already separated");
        cl.demo(Dx1, Dx4, "negative pair (close)— too close, penalized");
        cl.sweepDemo(C);
        cl.printComplexity();

        // ── 5. Gradient Analysis ──────────────────────────────────
        section("5. Gradient Analysis — Push direction for negative pairs");
        GradientAnalysis ga = new GradientAnalysis(C, l2);
        ga.analyze(Dx1, Dx4, "negative pair (close) → push direction");
        ga.printComplexity();

        // ── 6. Summary Table ──────────────────────────────────────
        section("6. Summary Table");
        printSummary(Dx1, Dx2, Dx3, Dx4, l1, l2, cos, cl, C);
    }

    // ── Banner helpers ─────────────────────────────────────────────
    static void printHeader() {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║  Lecture 7 — Optimize Distance Equation (Remake)        ║");
        System.out.println("║  Contrastive Learning: L1 / L2 / Cosine / Loss / Grad   ║");
        System.out.println("║  Student: 翁祺展  |  Instructor: 黃祥睿               ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.println();
        System.out.println("  Taipei 101 Analogy:");
        System.out.println("  You only feel how tall 101 is when you stand NEXT to it.");
        System.out.println("  Contrastive learning: features become meaningful through");
        System.out.println("  comparison — not by looking at a single sample alone.");
        System.out.println();
    }

    static void section(String title) {
        System.out.println();
        System.out.println("  ━━━ " + title + " ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }

    static void printSummary(FeatureVector a, FeatureVector pos,
                              FeatureVector negFar, FeatureVector negClose,
                              L1Distance l1, L2Distance l2,
                              CosineSimilarity cos, ContrastiveLoss cl, double C) {
        System.out.printf("  %-20s %-12s %-12s %-12s %-10s%n",
                "Metric", "pos pair", "neg(far)", "neg(close)", "Time");
        System.out.println("  " + "─".repeat(70));
        System.out.printf("  %-20s %-12.4f %-12.4f %-12.4f %-10s%n",
                "L1 (Manhattan)",
                l1.compute(a,pos), l1.compute(a,negFar), l1.compute(a,negClose), "O(n)");
        System.out.printf("  %-20s %-12.4f %-12.4f %-12.4f %-10s%n",
                "L2 (Euclidean)",
                l2.compute(a,pos), l2.compute(a,negFar), l2.compute(a,negClose), "O(n)");
        System.out.printf("  %-20s %-12.4f %-12.4f %-12.4f %-10s%n",
                "Cosine Sim",
                cos.compute(a,pos), cos.compute(a,negFar), cos.compute(a,negClose), "O(n)");
        System.out.printf("  %-20s %-12.4f %-12.4f %-12.4f %-10s%n",
                "Contrastive Loss",
                cl.loss(a,pos), cl.loss(a,negFar), cl.loss(a,negClose), "O(n)");
        System.out.println();
        System.out.println("  Interpretation:");
        System.out.println("    pos pair  → L2 small (similar), Loss>0 (being penalized as neg)");
        System.out.println("    neg (far) → L2 large, Loss=0 (already beyond margin C=" + C + ") ✓");
        System.out.println("    neg(close)→ L2<C, Loss>0 (too close, embeddings must be pushed apart)");
    }
}


// ═══════════════════════════════════════════════════════════════════
// FeatureVector — immutable wrapper for an n-dimensional embedding D(x)
//
// Stores the encoder output as double[] and provides utility methods.
// Time: O(1) construction (reference copy), O(n) for print.
// ═══════════════════════════════════════════════════════════════════
class FeatureVector {
    private final String  label;
    private final double[] data;
    private final int      n;

    public FeatureVector(String label, double[] data) {
        this.label = label;
        this.data  = data.clone();   // defensive copy — immutable
        this.n     = data.length;
    }

    public double   get(int i) { return data[i]; }
    public int      dim()      { return n; }
    public String   label()    { return label; }
    public double[] data()     { return data.clone(); }

    public void print() {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < n; i++) {
            sb.append(String.format("%.1f", data[i]));
            if (i < n-1) sb.append(", ");
        }
        sb.append("]");
        System.out.printf("    %-22s %s  (dim=%d)%n", label, sb, n);
    }

    /** Check dimension compatibility */
    public static void checkDim(FeatureVector a, FeatureVector b) {
        if (a.dim() != b.dim())
            throw new IllegalArgumentException(
                "Dimension mismatch: " + a.dim() + " vs " + b.dim());
    }
}


// ═══════════════════════════════════════════════════════════════════
// L1Distance — Manhattan Distance
//
//   ||D(x1) - D(x2)||_1  =  Σ_{i=0}^{n-1} |D(x1)_i - D(x2)_i|
//
// Properties:
//   • Robust to outliers (penalizes large deviations linearly)
//   • Induces sparse gradients — good for sparse feature spaces
//   • Forms a diamond-shaped unit ball in 2D
//
// Time Complexity : O(n)  — one pass, n = embedding dimension
// Space Complexity: O(1)  — single accumulator
// ═══════════════════════════════════════════════════════════════════
class L1Distance {

    /**
     * Compute L1 (Manhattan) distance between two feature vectors.
     *
     * Algorithm:
     *   sum = 0
     *   for i in 0..n-1:          ← O(n) iterations
     *       sum += |a[i] - b[i]|  ← O(1) per step
     *   return sum                 ← total O(n)
     *
     * Time:  O(n)
     * Space: O(1)
     */
    public double compute(FeatureVector a, FeatureVector b) {
        FeatureVector.checkDim(a, b);
        double sum = 0.0;
        for (int i = 0; i < a.dim(); i++)
            sum += Math.abs(a.get(i) - b.get(i));   // |a_i - b_i|
        return sum;
    }

    /** Step-by-step breakdown per dimension */
    public void demo(FeatureVector a, FeatureVector b, String desc) {
        FeatureVector.checkDim(a, b);
        double sum = 0.0;
        StringBuilder steps = new StringBuilder();
        for (int i = 0; i < a.dim(); i++) {
            double diff = Math.abs(a.get(i) - b.get(i));
            sum += diff;
            steps.append(String.format("%.2f", diff));
            if (i < a.dim()-1) steps.append("+");
        }
        System.out.printf("  L1(%-22s) = %s = %.4f  [%s]%n",
                desc, steps, sum, "O(n="+a.dim()+")");
    }

    public void printComplexity() {
        System.out.println();
        System.out.println("  L1Distance Time Complexity:");
        System.out.println("    Formula : ||x1-x2||_1 = Σ|x1_i - x2_i|");
        System.out.println("    Time    : O(n)   n = embedding dimension");
        System.out.println("    Space   : O(1)   single accumulator");
        System.out.println("    Property: Robust to outliers, sparse gradient");
    }
}


// ═══════════════════════════════════════════════════════════════════
// L2Distance — Euclidean Distance
//
//   ||D(x1) - D(x2)||_2  =  sqrt( Σ_{i=0}^{n-1} (D(x1)_i - D(x2)_i)^2 )
//
// Properties:
//   • Standard metric in contrastive learning loss functions
//   • Straight-line distance in n-dimensional embedding space
//   • Gradient points in the direction of the difference vector
//   • Forms a circle-shaped unit ball in 2D
//
// Time Complexity : O(n) + O(1) for sqrt  →  O(n)
// Space Complexity: O(1)
// ═══════════════════════════════════════════════════════════════════
class L2Distance {

    /**
     * Compute L2 (Euclidean) distance.
     *
     * Algorithm:
     *   sumSq = 0
     *   for i in 0..n-1:              ← O(n)
     *       diff  = a[i] - b[i]       ← O(1)
     *       sumSq += diff * diff       ← O(1)
     *   return sqrt(sumSq)             ← O(1)
     *   total: O(n)
     *
     * Time:  O(n)
     * Space: O(1)
     */
    public double compute(FeatureVector a, FeatureVector b) {
        FeatureVector.checkDim(a, b);
        double sumSq = 0.0;
        for (int i = 0; i < a.dim(); i++) {
            double diff = a.get(i) - b.get(i);
            sumSq += diff * diff;           // (a_i - b_i)^2
        }
        return Math.sqrt(sumSq);           // sqrt( Σ diff^2 )
    }

    /** Compute squared L2 (avoids sqrt, useful in gradient) */
    public double computeSq(FeatureVector a, FeatureVector b) {
        FeatureVector.checkDim(a, b);
        double sumSq = 0.0;
        for (int i = 0; i < a.dim(); i++) {
            double diff = a.get(i) - b.get(i);
            sumSq += diff * diff;
        }
        return sumSq;
    }

    public void demo(FeatureVector a, FeatureVector b, String desc) {
        double d = compute(a, b);
        System.out.printf("  L2(%-22s) = sqrt(Σdiff²) = %.6f  [%s]%n",
                desc, d, "O(n="+a.dim()+")");
    }

    public void printComplexity() {
        System.out.println();
        System.out.println("  L2Distance Time Complexity:");
        System.out.println("    Formula : ||x1-x2||_2 = sqrt(Σ(x1_i-x2_i)^2)");
        System.out.println("    Time    : O(n)   n = embedding dimension");
        System.out.println("    Space   : O(1)   one accumulator + sqrt");
        System.out.println("    Property: Straight-line dist; used in contrastive loss");
        System.out.println("    Gradient: d/dx ||x-y||_2 = (x-y)/||x-y||_2  (unit vec)");
    }
}


// ═══════════════════════════════════════════════════════════════════
// CosineSimilarity — Angle-based metric
//
//   cos(x1,x2) = (x1·x2) / (||x1||_2 × ||x2||_2)   ∈ [-1, 1]
//
// Properties:
//   • 1  → same direction (maximally similar)
//   • 0  → orthogonal (unrelated)
//   • -1 → opposite direction (maximally dissimilar)
//   • Magnitude-invariant: only direction matters
//   • Common in NLP and vision embeddings
//
// Time Complexity : O(n)  — dot product + two norms in one pass
// Space Complexity: O(1)  — three scalar accumulators
// ═══════════════════════════════════════════════════════════════════
class CosineSimilarity {

    /**
     * Compute cosine similarity.
     *
     * One-pass algorithm (combines dot and norms):
     *   dot=0, normA=0, normB=0
     *   for i in 0..n-1:          ← O(n)
     *       dot   += a[i]*b[i]
     *       normA += a[i]*a[i]
     *       normB += b[i]*b[i]
     *   return dot / (sqrt(normA)*sqrt(normB))
     *
     * Time:  O(n)
     * Space: O(1)
     */
    public double compute(FeatureVector a, FeatureVector b) {
        FeatureVector.checkDim(a, b);
        double dot=0, normA=0, normB=0;
        for (int i = 0; i < a.dim(); i++) {
            dot   += a.get(i) * b.get(i);
            normA += a.get(i) * a.get(i);
            normB += b.get(i) * b.get(i);
        }
        if (normA == 0 || normB == 0) return 0.0;
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /** Cosine distance = 1 - cosine similarity ∈ [0, 2] */
    public double distance(FeatureVector a, FeatureVector b) {
        return 1.0 - compute(a, b);
    }

    public void demo(FeatureVector a, FeatureVector b, String desc) {
        double sim  = compute(a, b);
        double dist = distance(a, b);
        double angleDeg = Math.toDegrees(Math.acos(Math.min(1.0, Math.max(-1.0, sim))));
        System.out.printf("  cos(%-22s) = %.6f  angle=%.2f°  cosine-dist=%.6f%n",
                desc, sim, angleDeg, dist);
    }

    public void printComplexity() {
        System.out.println();
        System.out.println("  CosineSimilarity Time Complexity:");
        System.out.println("    Formula : cos(a,b) = (a·b)/(||a||_2 * ||b||_2)");
        System.out.println("    Time    : O(n)   one pass — dot + two norms");
        System.out.println("    Space   : O(1)   three scalar accumulators");
        System.out.println("    Range   : [-1,1]  1=same dir, 0=orthogonal, -1=opposite");
        System.out.println("    Cos-dist: 1-cos ∈ [0,2]  (metric for dissimilarity)");
    }
}


// ═══════════════════════════════════════════════════════════════════
// ContrastiveLoss — Assignment 1 core class
//
//   L = max(0,  C  −  ||D(x1) − D(x2)||_2)
//
// Semantics (negative-sample loss):
//   D(x)  encoder's feature embedding of input x
//   C     margin constant — desired MINIMUM separation for negatives
//   L > 0 → pair too close → model penalized → embeddings pushed apart
//   L = 0 → pair already ≥ C apart → no gradient → no update needed
//
// Full Hadsell et al. (2006) symmetric loss:
//   y=1 (positive): Lp = ||D(x1)-D(x2)||_2^2           (pull together)
//   y=0 (negative): Ln = max(0, C-||D(x1)-D(x2)||_2)^2 (push apart)
//
// Time Complexity : O(n)  — L2 is O(n) + max() is O(1)
// Space Complexity: O(1)
// ═══════════════════════════════════════════════════════════════════
class ContrastiveLoss {
    private final double     C;    // margin constant
    private final L2Distance l2;

    public ContrastiveLoss(double margin, L2Distance l2) {
        this.C  = margin;
        this.l2 = l2;
    }

    /**
     * Negative-sample contrastive loss (lecture formula):
     *   L = max(0, C - ||D(x1)-D(x2)||_2)
     *
     * Time:  O(n)   — L2 is O(n), max is O(1)
     * Space: O(1)
     */
    public double loss(FeatureVector Dx1, FeatureVector Dx2) {
        double dist = l2.compute(Dx1, Dx2);    // O(n)
        return Math.max(0.0, C - dist);         // O(1) hinge
    }

    /**
     * Full labeled-pair loss (Hadsell et al. 2006):
     *   y=1: L = dist^2
     *   y=0: L = max(0, C-dist)^2
     *
     * Time: O(n), Space: O(1)
     */
    public double fullLoss(FeatureVector Dx1, FeatureVector Dx2, int label) {
        double dist = l2.compute(Dx1, Dx2);
        if (label == 1) {
            return dist * dist;                          // positive: pull
        } else {
            double h = Math.max(0.0, C - dist);
            return h * h;                                // negative: push
        }
    }

    public void demo(FeatureVector Dx1, FeatureVector Dx2, String desc) {
        double dist  = l2.compute(Dx1, Dx2);
        double L     = loss(Dx1, Dx2);
        double Lp    = fullLoss(Dx1, Dx2, 1);
        double Ln    = fullLoss(Dx1, Dx2, 0);
        System.out.printf("  [%s]%n", desc);
        System.out.printf("    L2=%.4f  C=%.1f  L=max(0,%.1f-%.4f)=%.4f",
                dist, C, C, dist, L);
        System.out.println( L > 0
                ? "  ← penalized (push apart)"
                : "  ← no penalty ✓ (beyond margin)");
        System.out.printf("    Full loss: y=1 → %.4f  y=0 → %.4f%n%n",
                Lp, Ln);
    }

    /** Show how L changes as L2 distance varies from 0 to 3C */
    public void sweepDemo(double C) {
        System.out.println("  Loss sweep (C=" + C + "):");
        System.out.printf("  %-8s %-10s %-10s %-10s%n",
                "L2 dist","Loss L","FullLoss(y=1)","FullLoss(y=0)");
        System.out.println("  " + "─".repeat(42));
        double[] dists = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0};
        for (double d : dists) {
            double L  = Math.max(0, C - d);
            double Lp = d * d;
            double h  = Math.max(0, C - d);
            double Ln = h * h;
            System.out.printf("  %-8.2f %-10.4f %-10.4f %-10.4f%s%n",
                    d, L, Lp, Ln,
                    d >= C ? "  ← neg OK ✓" : "");
        }
        System.out.println();
    }

    public void printComplexity() {
        System.out.println("  ContrastiveLoss Time Complexity:");
        System.out.println("    Formula : L = max(0, C - ||D(x1)-D(x2)||_2)");
        System.out.println("    Time    : O(n)   L2 O(n) + max O(1)");
        System.out.println("    Space   : O(1)");
        System.out.println("    C (margin): desired minimum separation for negative pairs");
        System.out.println("    L=0  → no gradient, no update (pair well separated)");
        System.out.println("    L>0  → gradient pushes embeddings apart");
        System.out.println();
        System.out.println("    Full Hadsell et al. (2006):");
        System.out.println("      y=1 (pos): L = ||D(x1)-D(x2)||_2^2   (pull)");
        System.out.println("      y=0 (neg): L = max(0,C-dist)^2        (push)");
    }
}


// ═══════════════════════════════════════════════════════════════════
// GradientAnalysis — dL/dD(x1) for the negative-sample loss
//
// Loss: L = max(0, C - dist)  where dist = ||D(x1)-D(x2)||_2
//
// When L > 0 (pair too close):
//   dL/d(dist)    = -1
//   d(dist)/dD(x1)= (D(x1)-D(x2)) / dist   ← unit vector pointing away
//
//   Therefore:
//   dL/dD(x1) = -1 × (D(x1)-D(x2)) / dist
//             = (D(x2)-D(x1)) / dist        ← points TOWARD D(x2)
//
//   But since we MINIMIZE L, gradient DESCENT moves D(x1) AWAY from D(x2):
//   update: D(x1) += η × (D(x1)-D(x2)) / dist  ← pushes apart
//
// When L = 0: gradient is 0 (no update).
//
// Time Complexity : O(n) — one pass to compute gradient vector
// Space Complexity: O(n) — gradient vector of size n
// ═══════════════════════════════════════════════════════════════════
class GradientAnalysis {
    private final double     C;
    private final L2Distance l2;

    public GradientAnalysis(double C, L2Distance l2) {
        this.C = C;
        this.l2 = l2;
    }

    /**
     * Compute gradient of L = max(0, C-dist) w.r.t. D(x1).
     *
     * grad[i] = -(D(x1)_i - D(x2)_i) / dist   if L>0
     *         = 0                                if L=0
     *
     * Gradient descent update direction (to minimize L, push apart):
     *   D(x1) += eta * (D(x1)-D(x2)) / dist
     *
     * Time:  O(n)
     * Space: O(n) — gradient vector
     */
    public double[] gradient(FeatureVector Dx1, FeatureVector Dx2) {
        double dist = l2.compute(Dx1, Dx2);
        double L    = Math.max(0.0, C - dist);
        double[] grad = new double[Dx1.dim()];
        if (L <= 0 || dist == 0) return grad;   // no gradient
        // dL/dD(x1)_i = -(D(x1)_i - D(x2)_i) / dist
        for (int i = 0; i < Dx1.dim(); i++)
            grad[i] = -(Dx1.get(i) - Dx2.get(i)) / dist;
        return grad;
    }

    /** Push direction for gradient DESCENT (to push embeddings apart) */
    public double[] pushDirection(FeatureVector Dx1, FeatureVector Dx2) {
        double dist = l2.compute(Dx1, Dx2);
        if (dist == 0) return new double[Dx1.dim()];
        double[] dir = new double[Dx1.dim()];
        // Positive direction: (D(x1)-D(x2))/dist  (unit vector pointing away)
        for (int i = 0; i < Dx1.dim(); i++)
            dir[i] = (Dx1.get(i) - Dx2.get(i)) / dist;
        return dir;
    }

    public void analyze(FeatureVector Dx1, FeatureVector Dx2, String desc) {
        double dist = l2.compute(Dx1, Dx2);
        double L    = Math.max(0.0, C - dist);
        double[] grad = gradient(Dx1, Dx2);
        double[] push = pushDirection(Dx1, Dx2);

        System.out.printf("  [%s]%n", desc);
        System.out.printf("    L2=%.4f  C=%.1f  L=%.4f%n", dist, C, L);
        if (L > 0) {
            System.out.println("    Gradient dL/dD(x1) = -(D(x1)-D(x2))/dist:");
            printVec("      grad", grad);
            System.out.println("    Push direction (D(x1)-D(x2))/dist:");
            printVec("      push", push);
            double eta = 0.1;
            System.out.printf("    After update (eta=%.1f): D(x1) += eta*push%n", eta);
            double[] updated = new double[Dx1.dim()];
            for (int i = 0; i < Dx1.dim(); i++)
                updated[i] = Dx1.get(i) + eta * push[i];
            FeatureVector updatedVec = new FeatureVector("D(x1)_updated", updated);
            double newDist = l2.compute(updatedVec, Dx2);
            System.out.printf("    New L2 distance = %.4f  (was %.4f, increased by %.4f)%n%n",
                    newDist, dist, newDist-dist);
        } else {
            System.out.println("    L=0: no gradient, no update needed ✓");
        }
    }

    private void printVec(String label, double[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            sb.append(String.format("%.4f", v[i]));
            if (i < v.length-1) sb.append(", ");
        }
        sb.append("]");
        System.out.println(label + " = " + sb);
    }

    public void printComplexity() {
        System.out.println("  GradientAnalysis Time Complexity:");
        System.out.println("    dL/dD(x1) = -(D(x1)-D(x2))/dist  [if L>0]");
        System.out.println("    Time    : O(n)   one pass for gradient vector");
        System.out.println("    Space   : O(n)   gradient vector storage");
        System.out.println("    Push dir: (D(x1)-D(x2))/||D(x1)-D(x2)||_2 (unit vec)");
        System.out.println("    Update  : D(x1) += eta * push_dir  (gradient descent)");
    }
}
