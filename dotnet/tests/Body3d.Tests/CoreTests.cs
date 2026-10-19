using System.Text;
using Body3d.Core;

namespace Body3d.Tests;

public class CoreTests
{
    static Mesh Cylinder(double rx, double ry, int n = 128, double h = 1.0)
    {
        var v = new List<V3>();
        for (int i = 0; i < n; i++) v.Add(new V3(rx * Math.Cos(2 * Math.PI * i / n), ry * Math.Sin(2 * Math.PI * i / n), 0));
        for (int i = 0; i < n; i++) v.Add(v[i] with { Z = h });
        v.Add(new V3(0, 0, 0));
        v.Add(new V3(0, 0, h));
        var f = new List<int>();
        for (int i = 0; i < n; i++)
        {
            int j = (i + 1) % n;
            f.AddRange([i, j, j + n, i, j + n, i + n, 2 * n, j, i, 2 * n + 1, i + n, j + n]);
        }
        return new Mesh(v.ToArray(), f.ToArray());
    }

    [Fact]
    public void StlBinaryRoundTripWeldsSharedCorners()
    {
        var m = Cylinder(0.15, 0.1, 32).Scaled(1000);
        var back = Stl.Read(Stl.WriteBinary(m, "SYNTHETIC"));
        Assert.Equal(m.Vertices.Length, back.Vertices.Length);
        Assert.Equal(m.FaceCount, back.FaceCount);
        for (int i = 0; i < m.Faces.Length; i++)
            Assert.True((m.Vertices[m.Faces[i]] - back.Vertices[back.Faces[i]]).Norm < 1e-3);
    }

    [Fact]
    public void StlAsciiParsesAndGarbageIsRejected()
    {
        const string ascii = "solid t\n facet normal 0 0 1\n  outer loop\n   vertex 0 0 0\n   vertex 1 0 0\n   vertex 0 1 0\n  endloop\n endfacet\nendsolid t\n";
        var m = Stl.Read(Encoding.ASCII.GetBytes(ascii));
        Assert.Equal(3, m.Vertices.Length);
        Assert.Throws<InvalidDataException>(() => Stl.Read(new byte[90]));
    }

    [Theory]
    [InlineData(0.15, 0.15)]
    [InlineData(0.18, 0.12)]
    public void SlicerMatchesAnalyticPerimeter(double rx, double ry)
    {
        var loop = Assert.Single(Girths.SliceLoops(Cylinder(rx, ry), 0.37));
        var (contour, tape) = Girths.Perimeters(loop);
        double h = Math.Pow((rx - ry) / (rx + ry), 2);
        double exact = Math.PI * (rx + ry) * (1 + 3 * h / (10 + Math.Sqrt(4 - 3 * h)));  // Ramanujan
        Assert.True(Math.Abs(contour - exact) / exact < 0.005);
        Assert.Equal(contour, tape, 9);
    }

    [Fact]
    public void IcpRecoversKnownRigidTransform()
    {
        var m = Fixtures.Mesh(Fixtures.Scans.First());
        var r = M3.RotZ(14) * M3.Rodrigues(new V3(2 * Math.PI / 180, 0, 0));
        var t = new V3(0.04, -0.03, 0.01);
        var moved = m.Transformed(r, t);
        var (r2, t2, _) = Change.Icp(moved.Vertices, m.Vertices, m.VertexNormals());
        double sq = 0;
        for (int i = 0; i < m.Vertices.Length; i++) sq += (r2.Apply(moved.Vertices[i]) + t2 - m.Vertices[i]).Dot(r2.Apply(moved.Vertices[i]) + t2 - m.Vertices[i]);
        Assert.True(Math.Sqrt(sq / m.Vertices.Length) * 1e3 < 2.0);
    }

    [Fact]
    public void ChangeOfARepositionedCopyIsNearZero()
    {
        var m = Fixtures.Mesh(Fixtures.Scans.First());
        var res = Change.Surface(m, m.Transformed(M3.RotZ(-12), new V3(0.02, 0.01, 0)));
        var d = res.DisplacementMm.Where(double.IsFinite).Select(Math.Abs).OrderBy(x => x).ToArray();
        Assert.True(d[d.Length / 2] < 0.5, $"median |change| {d[d.Length / 2]:F2} mm");
    }
}
