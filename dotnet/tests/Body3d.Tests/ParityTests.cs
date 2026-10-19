using System.Text.Json;
using Body3d.Core;
using Xunit.Abstractions;

namespace Body3d.Tests;

/// <summary>Python (body3d) vs C# (Body3d.Core) on the same STL files, within stated tolerances.
/// Writes results/parity_csharp.json.</summary>
public class ParityTests(ITestOutputHelper output)
{
    [Fact]
    public void DepthLandmarksAndGirthsMatchPython()
    {
        var tol = Fixtures.Expected.GetProperty("tolerances");
        double depthMm = tol.GetProperty("depth_mm").GetDouble(), pixelFrac = tol.GetProperty("depth_pixel_fraction").GetDouble();
        double lmMm = tol.GetProperty("landmark_mm").GetDouble(), girthMm = tol.GetProperty("girth_mm").GetDouble();
        using var model = new Landmarks(Fixtures.OnnxPath);
        var report = new List<object>();
        foreach (var scan in Fixtures.Scans)
        {
            var mesh = Fixtures.Mesh(scan);
            var py = Fixtures.Depth(scan);
            var cs = model.Scanner.Render(mesh);
            int hw = model.Scanner.Height * model.Scanner.Width, ok = 0, coverage = 0;
            double maxDiff = 0;
            for (int k = 0; k < model.Scanner.Views; k++)
            for (int i = 0; i < hw; i++)
            {
                double a = cs[k][i], b = py[k * hw + i];
                if (double.IsFinite(a) != double.IsFinite(b)) { coverage++; continue; }
                double diff = double.IsFinite(a) ? Math.Abs(a - b) * 1e3 : 0;
                maxDiff = Math.Max(maxDiff, diff);
                if (diff <= depthMm) ok++;
            }
            double frac = ok / (double)(model.Scanner.Views * hw);

            var lm = model.FromDepth(cs);
            var lmPy = Fixtures.LandmarksMm(scan);
            double lmDiff = lm.Select((p, j) => (p * 1e3 - lmPy[j]).Norm).Max();

            var girths = Girths.Compute(mesh, lmPy.Select(p => p * 1e-3).ToArray());
            double gDiff = 0;
            foreach (var g in scan.GetProperty("girths_mm").EnumerateObject())
            {
                gDiff = Math.Max(gDiff, Math.Abs(girths[g.Name].TapeMm - g.Value.GetProperty("tape").GetDouble()));
                gDiff = Math.Max(gDiff, Math.Abs(girths[g.Name].ContourMm - g.Value.GetProperty("contour").GetDouble()));
            }
            string name = scan.GetProperty("stl").GetString()!;
            output.WriteLine($"{name}: depth within {depthMm} mm {frac:P4} (coverage mismatches {coverage}, max diff {maxDiff:E2} mm), " +
                             $"landmarks max {lmDiff:F4} mm, girths max {gDiff:F5} mm");
            report.Add(new { stl = name, depth_pixels_within_tol = frac, coverage_mismatches = coverage, depth_max_abs_mm = maxDiff,
                             landmark_max_abs_mm = lmDiff, girth_max_abs_mm = gDiff });

            Assert.True(frac >= pixelFrac, $"{name}: only {frac:P3} of depth pixels within {depthMm} mm");
            Assert.True(lmDiff <= lmMm, $"{name}: landmark difference {lmDiff:F3} mm");
            Assert.True(gDiff <= girthMm, $"{name}: girth difference {gDiff:F4} mm");
        }
        Directory.CreateDirectory(Fixtures.ResultsDir);
        File.WriteAllText(Path.Combine(Fixtures.ResultsDir, "parity_csharp.json"), JsonSerializer.Serialize(new
        {
            data = "SYNTHETIC",
            tolerances = new { depth_mm = depthMm, depth_pixel_fraction = pixelFrac, landmark_mm = lmMm, girth_mm = girthMm },
            onnx_vs_torch_max_abs = Fixtures.Expected.GetProperty("onnx_vs_torch_max_abs").GetDouble(),
            scans = report,
        }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
    }
}
