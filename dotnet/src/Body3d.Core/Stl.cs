using System.Buffers.Binary;
using System.Globalization;
using System.Text;

namespace Body3d.Core;

/// <summary>Binary/ASCII STL reader and binary writer, mirroring body3d/stl.py.</summary>
public static class Stl
{
    const int Header = 80, Record = 50;

    /// <summary>Triangle corners (3 per triangle) in file units.</summary>
    public static V3[] ReadTriangles(ReadOnlySpan<byte> raw)
    {
        if (raw.Length >= Header + 4)
        {
            long n = BinaryPrimitives.ReadUInt32LittleEndian(raw.Slice(Header, 4));
            if (raw.Length == Header + 4 + n * Record)
            {
                var corners = new V3[3 * n];
                for (int t = 0; t < n; t++)
                {
                    var rec = raw.Slice(Header + 4 + t * Record, Record);
                    for (int k = 0; k < 3; k++)
                    {
                        int o = 12 + 12 * k;
                        corners[3 * t + k] = new V3(
                            BinaryPrimitives.ReadSingleLittleEndian(rec.Slice(o, 4)),
                            BinaryPrimitives.ReadSingleLittleEndian(rec.Slice(o + 4, 4)),
                            BinaryPrimitives.ReadSingleLittleEndian(rec.Slice(o + 8, 4)));
                    }
                }
                return corners;
            }
        }
        var text = Encoding.ASCII.GetString(raw);
        if (!text.TrimStart().StartsWith("solid", StringComparison.Ordinal))
            throw new InvalidDataException("neither binary STL (size mismatch) nor ASCII STL");
        var verts = new List<V3>();
        foreach (var line in text.Split('\n'))
        {
            var parts = line.Trim().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length >= 4 && parts[0] == "vertex")
                verts.Add(new V3(Num(parts[1]), Num(parts[2]), Num(parts[3])));
        }
        if (verts.Count == 0 || verts.Count % 3 != 0)
            throw new InvalidDataException($"ASCII STL has {verts.Count} vertices, not a positive multiple of 3");
        return verts.ToArray();

        static double Num(string s) => double.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    /// <summary>Indexed mesh: corners equal after rounding to <paramref name="decimals"/> are welded.</summary>
    public static Mesh Read(ReadOnlySpan<byte> raw, int decimals = 6)
    {
        var corners = ReadTriangles(raw);
        var index = new Dictionary<(double, double, double), int>();
        var verts = new List<V3>();
        var faces = new int[corners.Length];
        for (int i = 0; i < corners.Length; i++)
        {
            var c = corners[i];
            var key = (Math.Round(c.X, decimals), Math.Round(c.Y, decimals), Math.Round(c.Z, decimals));
            if (!index.TryGetValue(key, out int id))
            {
                id = verts.Count;
                index[key] = id;
                verts.Add(c);
            }
            faces[i] = id;
        }
        return new Mesh(verts.ToArray(), faces);
    }

    public static Mesh Read(string path) => Read(File.ReadAllBytes(path));

    public static byte[] WriteBinary(Mesh m, string header = "")
    {
        var buf = new byte[Header + 4 + m.FaceCount * Record];
        Encoding.ASCII.GetBytes(header.PadRight(Header)[..Header]).CopyTo(buf, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(buf.AsSpan(Header, 4), (uint)m.FaceCount);
        for (int t = 0; t < m.FaceCount; t++)
        {
            var span = buf.AsSpan(Header + 4 + t * Record, Record);
            V3 a = m.Vertices[m.Faces[3 * t]], b = m.Vertices[m.Faces[3 * t + 1]], c = m.Vertices[m.Faces[3 * t + 2]];
            var n = (b - a).Cross(c - a).Normalized(1e-20);
            int o = 0;
            foreach (var v in new[] { n, a, b, c })
            {
                BinaryPrimitives.WriteSingleLittleEndian(span.Slice(o, 4), (float)v.X);
                BinaryPrimitives.WriteSingleLittleEndian(span.Slice(o + 4, 4), (float)v.Y);
                BinaryPrimitives.WriteSingleLittleEndian(span.Slice(o + 8, 4), (float)v.Z);
                o += 12;
            }
        }
        return buf;
    }
}
