namespace Body3d.Core;

public sealed record Girth(double ZMm, double ContourMm, double TapeMm);

/// <summary>Mesh-plane slicing and girths, mirroring body3d/change.py.</summary>
public static class Girths
{
    public const double ThighBelowCrotch = 0.05;  // m

    /// <summary>Closed cross-section polygons (xy) at height z. Intersection points are keyed by
    /// the mesh edge they lie on, so segments from neighbouring triangles chain exactly.</summary>
    public static List<(double X, double Y)[]> SliceLoops(Mesh m, double z)
    {
        var s = m.Vertices.Select(v => v.Z - z == 0 ? 1e-12 : v.Z - z).ToArray();
        long n = m.Vertices.Length;
        var links = new Dictionary<long, List<long>>();
        var points = new Dictionary<long, (double, double)>();
        var order = new List<long>();
        Span<long> keys = stackalloc long[2];
        for (int t = 0; t < m.FaceCount; t++)
        {
            Span<int> tri = [m.Faces[3 * t], m.Faces[3 * t + 1], m.Faces[3 * t + 2]];
            bool p0 = s[tri[0]] > 0, p1 = s[tri[1]] > 0, p2 = s[tri[2]] > 0;
            if (p0 == p1 && p1 == p2) continue;
            int nk = 0;
            for (int e = 0; e < 3; e++)
            {
                int a = tri[e], b = tri[(e + 1) % 3];
                if (s[a] > 0 == s[b] > 0) continue;
                long key = Math.Min(a, b) * n + Math.Max(a, b);
                if (!points.ContainsKey(key))
                {
                    double u = s[a] / (s[a] - s[b]);
                    var va = m.Vertices[a];
                    var vb = m.Vertices[b];
                    points[key] = (va.X + (vb.X - va.X) * u, va.Y + (vb.Y - va.Y) * u);
                }
                keys[nk++] = key;
            }
            Link(keys[0], keys[1]);
            Link(keys[1], keys[0]);
        }

        var loops = new List<(double, double)[]>();
        var seen = new HashSet<long>();
        foreach (var start in order)
        {
            if (!seen.Add(start)) continue;
            var loop = new List<(double, double)> { points[start] };
            long prev = -1, cur = start;
            while (true)
            {
                long next = -1;
                foreach (var k in links[cur]) if (k != prev) { next = k; break; }
                if (next < 0 || next == start || !seen.Add(next)) break;
                loop.Add(points[next]);
                prev = cur;
                cur = next;
            }
            loops.Add(loop.ToArray());
        }
        return loops;

        void Link(long a, long b)
        {
            if (!links.TryGetValue(a, out var l)) { links[a] = l = []; order.Add(a); }
            l.Add(b);
        }
    }

    public static double Area((double X, double Y)[] p)
    {
        double s = 0;
        for (int i = 0; i < p.Length; i++)
        {
            var (x0, y0) = p[i];
            var (x1, y1) = p[(i + 1) % p.Length];
            s += x0 * y1 - x1 * y0;
        }
        return Math.Abs(s) / 2;
    }

    /// <summary>(contour length, convex-hull length = what a tape measures), in input units.</summary>
    public static (double Contour, double Tape) Perimeters((double X, double Y)[] p)
    {
        double contour = 0;
        for (int i = 0; i < p.Length; i++) contour += Dist(p[i], p[(i + 1) % p.Length]);
        var hull = ConvexHull(p);
        double tape = 0;
        for (int i = 0; i < hull.Count; i++) tape += Dist(hull[i], hull[(i + 1) % hull.Count]);
        return (contour, tape);

        static double Dist((double X, double Y) a, (double X, double Y) b) => Math.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y));
    }

    static List<(double X, double Y)> ConvexHull((double X, double Y)[] pts)
    {
        var p = pts.OrderBy(q => q.X).ThenBy(q => q.Y).ToArray();
        var h = new List<(double X, double Y)>();
        for (int pass = 0; pass < 2; pass++)  // Andrew's monotone chain: lower then upper hull
        {
            int start = h.Count;
            foreach (var q in pass == 0 ? p : p.Reverse())
            {
                while (h.Count >= start + 2 && Cross(h[^2], h[^1], q) <= 0) h.RemoveAt(h.Count - 1);
                h.Add(q);
            }
            h.RemoveAt(h.Count - 1);
        }
        return h;

        static double Cross((double X, double Y) o, (double X, double Y) a, (double X, double Y) b) =>
            (a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X);
    }

    /// <summary>Waist (waist-front height), hip (buttock-point height) and left thigh (5 cm below the
    /// crotch). Waist/hip take the largest loop (the trunk), the thigh the loop furthest to the body's left.
    /// Mesh and landmarks in metres; results in mm.</summary>
    public static Dictionary<string, Girth> Compute(Mesh m, V3[] lm)
    {
        V3 Get(string name) => lm[Landmarks.Index(name)];
        var crotch = Get("crotch");
        var left = Get("acromion_l") - Get("acromion_r");
        double ln = Math.Sqrt(left.X * left.X + left.Y * left.Y);
        (double X, double Y) dir = (left.X / ln, left.Y / ln);
        var levels = new (string Name, double Z)[]
        {
            ("waist", Get("waist_front").Z), ("hip", Get("buttock_point").Z), ("thigh", crotch.Z - ThighBelowCrotch),
        };
        var result = new Dictionary<string, Girth>();
        foreach (var (name, z) in levels)
        {
            var loops = SliceLoops(m, z).Where(l => l.Length >= 3).ToList();
            var loop = name == "thigh"
                ? loops.MaxBy(l => (l.Average(q => q.X) - crotch.X) * dir.X + (l.Average(q => q.Y) - crotch.Y) * dir.Y)!
                : loops.MaxBy(Area)!;
            var (contour, tape) = Perimeters(loop);
            result[name] = new Girth(z * 1e3, contour * 1e3, tape * 1e3);
        }
        return result;
    }
}
