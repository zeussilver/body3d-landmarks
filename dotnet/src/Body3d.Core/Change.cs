namespace Body3d.Core;

/// <summary>Static k-d tree for exact nearest-neighbour queries (cost does not grow with distance).</summary>
public sealed class KdTree
{
    readonly V3[] _pts;
    readonly int[] _idx;  // implicit tree: median of each range is the node, split axis = depth % 3

    public KdTree(V3[] pts)
    {
        _pts = pts;
        _idx = Enumerable.Range(0, pts.Length).ToArray();
        Build(0, pts.Length, 0);
    }

    static double Axis(V3 p, int a) => a == 0 ? p.X : a == 1 ? p.Y : p.Z;

    void Build(int lo, int hi, int depth)
    {
        if (hi - lo <= 1) return;
        int a = depth % 3, mid = (lo + hi) / 2;
        Array.Sort(_idx, lo, hi - lo, Comparer<int>.Create((i, j) => Axis(_pts[i], a).CompareTo(Axis(_pts[j], a))));
        Build(lo, mid, depth + 1);
        Build(mid + 1, hi, depth + 1);
    }

    public int Nearest(V3 q)
    {
        int best = -1;
        double bestD = double.PositiveInfinity;
        Search(0, _idx.Length, 0);
        return best;

        void Search(int lo, int hi, int depth)
        {
            if (lo >= hi) return;
            int a = depth % 3, mid = (lo + hi) / 2, node = _idx[mid];
            var d = _pts[node] - q;
            double dist = d.Dot(d);
            if (dist < bestD) { bestD = dist; best = node; }
            double diff = Axis(q, a) - Axis(_pts[node], a);
            var (near, far) = diff < 0 ? ((lo, mid), (mid + 1, hi)) : ((mid + 1, hi), (lo, mid));
            Search(near.Item1, near.Item2, depth + 1);
            if (diff * diff < bestD) Search(far.Item1, far.Item2, depth + 1);
        }
    }
}

public sealed record ChangeResult(Mesh Aligned, double[] DisplacementMm, double IcpRmsMm);

/// <summary>Rigid alignment and signed change maps, mirroring body3d/change.py (path B: STL only).</summary>
public static class Change
{
    public const double StableTop = 0.12, StableBottom = 0.30;

    /// <summary>1 for head and shin height bands (little change with weight), <paramref name="low"/> elsewhere.</summary>
    public static double[] StableWeights(Mesh m, double low = 0.05)
    {
        var (zmin, zmax) = m.ZRange();
        double h = zmax - zmin;
        return m.Vertices.Select(v => v.Z > zmax - StableTop * h || v.Z < zmin + StableBottom * h ? 1.0 : low).ToArray();
    }

    /// <summary>Point-to-plane ICP of <paramref name="src"/> onto (<paramref name="dst"/>, normals), multi-start over yaw.</summary>
    public static (M3 R, V3 T, double Rms) Icp(V3[] src, V3[] dst, V3[] dstN, double[]? w = null, int iters = 40,
                                               double trim = 0.9, double[]? yawStarts = null)
    {
        w ??= Enumerable.Repeat(1.0, src.Length).ToArray();
        var tree = new KdTree(dst);
        V3 Mean(V3[] p) => p.Aggregate(new V3(0, 0, 0), (a, b) => a + b) * (1.0 / p.Length);
        (M3, V3, double)? best = null;
        foreach (var yaw in yawStarts ?? [-40, -20, 0, 20, 40])
        {
            var r0 = M3.RotZ(yaw);
            var t0 = Mean(dst) - r0.Apply(Mean(src));
            var cand = Run(r0, t0, 8);
            if (best is null || cand.Item3 < best.Value.Item3) best = cand;
        }
        return Run(best!.Value.Item1, best.Value.Item2, iters);

        (M3, V3, double) Run(M3 r, V3 t, int n)
        {
            for (int it = 0; it < n; it++)
            {
                var (p, j, res, keep) = Match(r, t);
                var ata = new double[6, 6];
                var atb = new double[6];
                for (int i = 0; i < src.Length; i++)
                {
                    if (!keep[i]) continue;
                    var nq = dstN[j[i]];
                    var c = p[i].Cross(nq);
                    double[] row = [c.X, c.Y, c.Z, nq.X, nq.Y, nq.Z];
                    for (int a = 0; a < 6; a++)
                    {
                        atb[a] -= w[i] * row[a] * res[i];
                        for (int b = 0; b < 6; b++) ata[a, b] += w[i] * row[a] * row[b];
                    }
                }
                var x = Solve(ata, atb);
                var dr = M3.Rodrigues(new V3(x[0], x[1], x[2]));
                r = dr * r;
                t = dr.Apply(t) + new V3(x[3], x[4], x[5]);
            }
            var (_, _, resF, keepF) = Match(r, t);
            double num = 0, den = 0;
            for (int i = 0; i < src.Length; i++) if (keepF[i]) { num += w[i] * resF[i] * resF[i]; den += w[i]; }
            return (r, t, Math.Sqrt(num / den));
        }

        (V3[] P, int[] J, double[] Res, bool[] Keep) Match(M3 r, V3 t)
        {
            var p = src.Select(s => r.Apply(s) + t).ToArray();
            var j = p.Select(tree.Nearest).ToArray();
            var res = new double[p.Length];
            for (int i = 0; i < p.Length; i++) res[i] = (p[i] - dst[j[i]]).Dot(dstN[j[i]]);
            var abs = res.Select(Math.Abs).OrderBy(a => a).ToArray();
            double cut = Quantile(abs, trim);
            return (p, j, res, res.Select(v => Math.Abs(v) <= cut).ToArray());
        }
    }

    /// <summary>Linear-interpolated quantile of sorted values (numpy's default method).</summary>
    static double Quantile(double[] sorted, double q)
    {
        double pos = q * (sorted.Length - 1);
        int lo = (int)Math.Floor(pos);
        int hi = Math.Min(lo + 1, sorted.Length - 1);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
    }

    static double[] Solve(double[,] a, double[] b)
    {
        int n = b.Length;
        var m = (double[,])a.Clone();
        var x = (double[])b.Clone();
        for (int c = 0; c < n; c++)
        {
            int piv = c;
            for (int r = c + 1; r < n; r++) if (Math.Abs(m[r, c]) > Math.Abs(m[piv, c])) piv = r;
            for (int k = 0; k < n; k++) (m[c, k], m[piv, k]) = (m[piv, k], m[c, k]);
            (x[c], x[piv]) = (x[piv], x[c]);
            if (Math.Abs(m[c, c]) < 1e-18) continue;
            for (int r = 0; r < n; r++)
            {
                if (r == c) continue;
                double f = m[r, c] / m[c, c];
                for (int k = c; k < n; k++) m[r, k] -= f * m[c, k];
                x[r] -= f * x[c];
            }
        }
        for (int c = 0; c < n; c++) x[c] = Math.Abs(m[c, c]) < 1e-18 ? 0 : x[c] / m[c, c];
        return x;
    }

    /// <summary>Align <paramref name="followup"/> to <paramref name="baseline"/> and return the signed change
    /// (mm, + = more tissue) at every baseline vertex, measured to the nearest follow-up tangent plane.</summary>
    public static ChangeResult Surface(Mesh baseline, Mesh followup)
    {
        var n0 = baseline.VertexNormals();
        var (r, t, rms) = Icp(followup.Vertices, baseline.Vertices, n0, StableWeights(followup));
        var aligned = followup.Transformed(r, t);
        var nt = aligned.VertexNormals();
        var tree = new KdTree(aligned.Vertices);
        var disp = baseline.Vertices.Select(p =>
        {
            int j = tree.Nearest(p);
            return (aligned.Vertices[j] - p).Dot(nt[j]) * 1e3;
        }).ToArray();
        return new ChangeResult(aligned, disp, rms * 1e3);
    }
}
