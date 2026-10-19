namespace Body3d.Core;

/// <summary>Double-precision 3-vector (System.Numerics.Vector3 is single precision).</summary>
public readonly record struct V3(double X, double Y, double Z)
{
    public static V3 operator +(V3 a, V3 b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    public static V3 operator -(V3 a, V3 b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
    public static V3 operator *(double s, V3 a) => new(s * a.X, s * a.Y, s * a.Z);
    public static V3 operator *(V3 a, double s) => s * a;
    public double Dot(V3 b) => X * b.X + Y * b.Y + Z * b.Z;
    public V3 Cross(V3 b) => new(Y * b.Z - Z * b.Y, Z * b.X - X * b.Z, X * b.Y - Y * b.X);
    public double Norm => Math.Sqrt(Dot(this));
    public V3 Normalized(double eps = 1e-12) => this * (1.0 / Math.Max(Norm, eps));
}

/// <summary>Row-major 3x3 matrix.</summary>
public readonly record struct M3(double A, double B, double C, double D, double E, double F, double G, double H, double I)
{
    public static readonly M3 Identity = new(1, 0, 0, 0, 1, 0, 0, 0, 1);
    public V3 Apply(V3 v) => new(A * v.X + B * v.Y + C * v.Z, D * v.X + E * v.Y + F * v.Z, G * v.X + H * v.Y + I * v.Z);
    public static M3 operator *(M3 m, M3 n) => new(
        m.A * n.A + m.B * n.D + m.C * n.G, m.A * n.B + m.B * n.E + m.C * n.H, m.A * n.C + m.B * n.F + m.C * n.I,
        m.D * n.A + m.E * n.D + m.F * n.G, m.D * n.B + m.E * n.E + m.F * n.H, m.D * n.C + m.E * n.F + m.F * n.I,
        m.G * n.A + m.H * n.D + m.I * n.G, m.G * n.B + m.H * n.E + m.I * n.H, m.G * n.C + m.H * n.F + m.I * n.I);

    public static M3 RotZ(double deg)
    {
        double a = deg * Math.PI / 180, c = Math.Cos(a), s = Math.Sin(a);
        return new(c, -s, 0, s, c, 0, 0, 0, 1);
    }

    /// <summary>Rotation by the axis-angle vector <paramref name="w"/> (Rodrigues).</summary>
    public static M3 Rodrigues(V3 w)
    {
        double th = w.Norm;
        if (th < 1e-12) return Identity;
        var k = w * (1 / th);
        double c = Math.Cos(th), s = Math.Sin(th), t = 1 - c;
        return new(
            c + k.X * k.X * t, k.X * k.Y * t - k.Z * s, k.X * k.Z * t + k.Y * s,
            k.Y * k.X * t + k.Z * s, c + k.Y * k.Y * t, k.Y * k.Z * t - k.X * s,
            k.Z * k.X * t - k.Y * s, k.Z * k.Y * t + k.X * s, c + k.Z * k.Z * t);
    }
}

/// <summary>Indexed triangle mesh; <see cref="Faces"/> holds 3 vertex indices per triangle.</summary>
public sealed class Mesh(V3[] vertices, int[] faces)
{
    public V3[] Vertices { get; } = vertices;
    public int[] Faces { get; } = faces;
    public int FaceCount => Faces.Length / 3;

    public Mesh Scaled(double s) => new(Vertices.Select(v => v * s).ToArray(), Faces);

    public Mesh Transformed(M3 r, V3 t) => new(Vertices.Select(v => r.Apply(v) + t).ToArray(), Faces);

    /// <summary>Area-weighted unit vertex normals (outward for consistent winding).</summary>
    public V3[] VertexNormals()
    {
        var n = new V3[Vertices.Length];
        for (int f = 0; f < FaceCount; f++)
        {
            int a = Faces[3 * f], b = Faces[3 * f + 1], c = Faces[3 * f + 2];
            var fn = (Vertices[b] - Vertices[a]).Cross(Vertices[c] - Vertices[a]);
            n[a] += fn; n[b] += fn; n[c] += fn;
        }
        for (int i = 0; i < n.Length; i++) n[i] = n[i].Normalized();
        return n;
    }

    public (double Min, double Max) ZRange()
    {
        double lo = double.PositiveInfinity, hi = double.NegativeInfinity;
        foreach (var v in Vertices) { lo = Math.Min(lo, v.Z); hi = Math.Max(hi, v.Z); }
        return (lo, hi);
    }
}
