using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Body3d.Core;

/// <summary>
/// Landmark inference: render the turntable views, run the exported per-view CNN in ONNX
/// Runtime, then soft-argmax and weighted least-squares triangulation (body3d/model.py).
/// Only the CNN is in ONNX; everything around it is plain C#.
/// </summary>
public sealed class Landmarks : IDisposable
{
    public static readonly string[] Names =
    [
        "vertex", "cervicale", "acromion_l", "acromion_r", "bust_point_l", "bust_point_r",
        "waist_front", "waist_back", "waist_side_l", "waist_side_r", "buttock_point",
        "crotch", "knee_l", "knee_r", "ankle_l", "ankle_r",
    ];

    public static int Index(string name) => Array.IndexOf(Names, name) is var i and >= 0 ? i : throw new ArgumentException(name);

    readonly InferenceSession _session;
    public Scanner Scanner { get; }

    public Landmarks(string onnxPath, Scanner? scanner = null)
    {
        _session = new InferenceSession(onnxPath);
        Scanner = scanner ?? new Scanner();
    }

    /// <summary>16 landmarks (metres, same frame as the mesh) for a mesh in the turntable frame (metres).</summary>
    public V3[] Predict(Mesh m) => FromDepth(Scanner.Render(m));

    public V3[] FromDepth(double[][] depth)
    {
        var s = Scanner;
        var input = new DenseTensor<float>(s.ToInput(depth), [s.Views, 2, s.Height, s.Width]);
        using var outputs = _session.Run([NamedOnnxValue.CreateFromTensor("depth", input)]);
        var heat = outputs.First(o => o.Name == "heatmap").AsTensor<float>();
        var conf = outputs.First(o => o.Name == "confidence").AsTensor<float>();
        int k = Names.Length, h = heat.Dimensions[2], w = heat.Dimensions[3];

        var px = new (double Col, double Row)[s.Views, k];
        var weight = new double[s.Views, k];
        for (int v = 0; v < s.Views; v++)
        for (int j = 0; j < k; j++)
        {
            double max = double.NegativeInfinity;
            for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) max = Math.Max(max, heat[v, j, r, c]);
            double sum = 0, sc = 0, sr = 0;
            for (int r = 0; r < h; r++)
            for (int c = 0; c < w; c++)
            {
                double e = Math.Exp(heat[v, j, r, c] - max);
                sum += e; sc += e * c; sr += e * r;
            }
            // half-resolution cell -> full-resolution pixel coordinates
            px[v, j] = ((sc / sum + 0.5) * 2 - 0.5, (sr / sum + 0.5) * 2 - 0.5);
            double x = conf[v, j];
            weight[v, j] = (x > 20 ? x : Math.Log(1 + Math.Exp(x))) + 1e-3;  // torch softplus (threshold 20)
        }
        return Triangulate(px, weight);
    }

    /// <summary>u = x cos(yaw) - y sin(yaw), w = z in every view -> per-landmark 2x2 normal equations.</summary>
    public V3[] Triangulate((double Col, double Row)[,] px, double[,] weight, double ridge = 1e-6)
    {
        var s = Scanner;
        int k = px.GetLength(1);
        var outp = new V3[k];
        for (int j = 0; j < k; j++)
        {
            double m11 = ridge, m12 = 0, m22 = ridge, b1 = 0, b2 = 0, zw = 0, aw = 0;
            for (int v = 0; v < s.Views; v++)
            {
                double th = s.Yaw(v) * Math.PI / 180, c = Math.Cos(th), sn = Math.Sin(th), a = weight[v, j];
                double u = (px[v, j].Col + 0.5) * s.Pixel - s.Width * s.Pixel / 2;
                double z = s.Height * s.Pixel - (px[v, j].Row + 0.5) * s.Pixel;
                m11 += a * c * c; m12 -= a * c * sn; m22 += a * sn * sn;
                b1 += a * c * u; b2 -= a * sn * u;
                zw += a * z; aw += a;
            }
            double det = m11 * m22 - m12 * m12;
            outp[j] = new V3((m22 * b1 - m12 * b2) / det, (m11 * b2 - m12 * b1) / det, zw / aw);
        }
        return outp;
    }

    public void Dispose() => _session.Dispose();
}
