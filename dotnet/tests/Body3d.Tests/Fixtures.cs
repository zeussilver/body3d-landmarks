using System.Text.Json;
using Body3d.Core;

namespace Body3d.Tests;

/// <summary>Parity fixtures written by `python -m body3d.export` (dotnet/fixtures).</summary>
public static class Fixtures
{
    public static readonly string Dir = Find();
    public static readonly JsonElement Expected = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "expected.json"))).RootElement;
    public static string OnnxPath => Path.Combine(Dir, "viewnet.onnx");
    public static string ResultsDir => Path.GetFullPath(Path.Combine(Dir, "..", "..", "results"));

    static string Find()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
        {
            var f = Path.Combine(d.FullName, "fixtures", "expected.json");
            if (File.Exists(f)) return Path.GetDirectoryName(f)!;
        }
        throw new FileNotFoundException("dotnet/fixtures/expected.json not found; run `python -m body3d.export`");
    }

    public static IEnumerable<JsonElement> Scans => Expected.GetProperty("scans").EnumerateArray();

    /// <summary>Scan mesh in metres.</summary>
    public static Mesh Mesh(JsonElement scan) => Stl.Read(Path.Combine(Dir, scan.GetProperty("stl").GetString()!)).Scaled(1e-3);

    public static float[] Depth(JsonElement scan)
    {
        var raw = File.ReadAllBytes(Path.Combine(Dir, scan.GetProperty("depth").GetString()!));
        var d = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, d, 0, raw.Length);
        return d;
    }

    public static V3[] LandmarksMm(JsonElement scan) =>
        scan.GetProperty("landmarks_mm").EnumerateArray()
            .Select(p => new V3(p[0].GetDouble(), p[1].GetDouble(), p[2].GetDouble())).ToArray();
}
