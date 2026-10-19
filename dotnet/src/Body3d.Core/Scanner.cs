namespace Body3d.Core;

/// <summary>
/// Orthographic turntable depth camera, a line-by-line port of body3d/scanner.py:
/// view k is the turntable frame rotated by yaw_k about z; image column follows x,
/// row follows -z, depth is y + Distance. Edge-function rasterisation at pixel centres
/// with barycentric depth and a z-buffer. Positions are in metres.
/// </summary>
public sealed record Scanner(int Views = 8, int Height = 256, int Width = 160, double Pixel = 0.0085, double Distance = 1.2)
{
    public const double DepthScale = 0.25;  // network input: (d - per-view median) / DepthScale

    public double Yaw(int k) => k * 360.0 / Views;

    /// <summary>Continuous pixel coordinates (integers are pixel centres) and depth of a turntable point in view k.</summary>
    public (double Col, double Row, double Depth) Project(V3 p, int k)
    {
        double a = Yaw(k) * Math.PI / 180, c = Math.Cos(a), s = Math.Sin(a);
        double x = c * p.X - s * p.Y, y = s * p.X + c * p.Y;
        return ((x + Width * Pixel / 2) / Pixel - 0.5, (Height * Pixel - p.Z) / Pixel - 0.5, y + Distance);
    }

    /// <summary>Depth per view, row-major H x W, +infinity where nothing is hit.</summary>
    public double[][] Render(Mesh m)
    {
        var depth = new double[Views][];
        var col = new double[m.Vertices.Length];
        var row = new double[m.Vertices.Length];
        var dep = new double[m.Vertices.Length];
        for (int k = 0; k < Views; k++)
        {
            for (int i = 0; i < m.Vertices.Length; i++)
                (col[i], row[i], dep[i]) = Project(m.Vertices[i], k);
            depth[k] = Rasterise(col, row, dep, m.Faces);
        }
        return depth;
    }

    double[] Rasterise(double[] col, double[] row, double[] d, int[] f)
    {
        var zbuf = new double[Height * Width];
        Array.Fill(zbuf, double.PositiveInfinity);
        for (int t = 0; t < f.Length / 3; t++)
        {
            int i0 = f[3 * t], i1 = f[3 * t + 1], i2 = f[3 * t + 2];
            double x0 = col[i0], x1 = col[i1], x2 = col[i2], y0 = row[i0], y1 = row[i1], y2 = row[i2];
            int c0 = (int)Math.Clamp(Math.Ceiling(Math.Min(x0, Math.Min(x1, x2))), 0, Width);
            int c1 = (int)Math.Clamp(Math.Floor(Math.Max(x0, Math.Max(x1, x2))), -1, Width - 1);
            int r0 = (int)Math.Clamp(Math.Ceiling(Math.Min(y0, Math.Min(y1, y2))), 0, Height);
            int r1 = (int)Math.Clamp(Math.Floor(Math.Max(y0, Math.Max(y1, y2))), -1, Height - 1);
            double den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
            if (Math.Abs(den) <= 1e-12) continue;
            for (int pr = r0; pr <= r1; pr++)
            for (int pc = c0; pc <= c1; pc++)
            {
                double w0 = ((y1 - y2) * (pc - x2) + (x2 - x1) * (pr - y2)) / den;
                double w1 = ((y2 - y0) * (pc - x2) + (x0 - x2) * (pr - y2)) / den;
                double w2 = 1.0 - w0 - w1;
                if (w0 < -1e-9 || w1 < -1e-9 || w2 < -1e-9) continue;
                double z = w0 * d[i0] + w1 * d[i1] + w2 * d[i2];
                int idx = pr * Width + pc;
                if (z < zbuf[idx]) zbuf[idx] = z;
            }
        }
        return zbuf;
    }

    /// <summary>Network input (views, 2, H, W): (depth - lower median) / DepthScale and a validity mask,
    /// matching body3d.model.depth_to_input (torch.nanmedian returns the lower median).</summary>
    public float[] ToInput(double[][] depth)
    {
        int hw = Height * Width;
        var x = new float[Views * 2 * hw];
        for (int k = 0; k < Views; k++)
        {
            var valid = depth[k].Where(double.IsFinite).Select(v => (float)v).ToArray();
            Array.Sort(valid);
            float med = valid.Length > 0 ? valid[(valid.Length - 1) / 2] : 0f;
            for (int i = 0; i < hw; i++)
            {
                double v = depth[k][i];
                bool hit = double.IsFinite(v);
                x[(2 * k) * hw + i] = hit ? ((float)v - med) / (float)DepthScale : 0f;
                x[(2 * k + 1) * hw + i] = hit ? 1f : 0f;
            }
        }
        return x;
    }
}
