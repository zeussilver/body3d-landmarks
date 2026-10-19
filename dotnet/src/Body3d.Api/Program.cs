// Body3d API: landmarks, girths and change maps from STL scans (model trained on SYNTHETIC data).
using System.Diagnostics;
using Amazon.S3;
using Body3d.Api;
using Body3d.Core;

var builder = WebApplication.CreateBuilder(args);
builder.WebHost.ConfigureKestrel(o => o.Limits.MaxRequestBodySize = 64 * 1024 * 1024);
var modelPath = builder.Configuration["Body3d:ModelPath"] ?? "viewnet.onnx";
if (!Path.IsPathRooted(modelPath)) modelPath = Path.Combine(builder.Environment.ContentRootPath, modelPath);
builder.Services.AddSingleton(_ => new Landmarks(modelPath));

// Scan storage: "local" (default, a directory) or "s3" (Body3d:Bucket, credentials from the AWS profile/env).
var cfg = builder.Configuration;
if (cfg["Body3d:Store"] == "s3")
{
    builder.Services.AddDefaultAWSOptions(cfg.GetAWSOptions());
    builder.Services.AddAWSService<IAmazonS3>();
    builder.Services.AddSingleton<IScanStore>(sp => new S3ScanStore(
        sp.GetRequiredService<IAmazonS3>(), cfg["Body3d:Bucket"] ?? throw new InvalidOperationException("Body3d:Bucket is required"),
        cfg["Body3d:ExpectedBucketOwner"]));
}
else
{
    var root = cfg["Body3d:LocalRoot"] ?? "scan-store";
    if (!Path.IsPathRooted(root)) root = Path.Combine(builder.Environment.ContentRootPath, root);
    builder.Services.AddSingleton<IScanStore>(new LocalScanStore(root));
}

var app = builder.Build();

app.MapGet("/health", () => Results.Ok(new { status = "ok", model = Path.GetFileName(modelPath), synthetic_training_data = true }));

// Body: binary or ASCII STL in the turntable frame (z up, floor at z = 0). ?units=mm (default) or m.
app.MapPost("/scans/landmarks", async (HttpRequest req, Landmarks model) =>
{
    var (mesh, error) = await ReadMesh(req.Body, req.Query["units"]);
    if (mesh is null) return error!;
    var (lm, timings) = Predict(model, mesh);
    return Results.Ok(new { units = "mm", landmarks = Named(lm), timings_ms = timings });
});

app.MapPost("/scans/girths", async (HttpRequest req, Landmarks model) =>
{
    var (mesh, error) = await ReadMesh(req.Body, req.Query["units"]);
    if (mesh is null) return error!;
    var (lm, timings) = Predict(model, mesh);
    return Results.Ok(new { units = "mm", landmarks = Named(lm), girths = Girths.Compute(mesh, lm), timings_ms = timings });
});

// Multipart form with files "baseline" and "followup".
app.MapPost("/scans/change", async (HttpRequest req, Landmarks model) =>
{
    if (!req.HasFormContentType) return Results.Problem("expected multipart form with 'baseline' and 'followup' STL files", statusCode: 400);
    var form = await req.ReadFormAsync();
    if (form.Files["baseline"] is not { } fb || form.Files["followup"] is not { } ff)
        return Results.Problem("missing 'baseline' or 'followup' file", statusCode: 400);
    var (b, eb) = await ReadMesh(fb.OpenReadStream(), req.Query["units"]);
    if (b is null) return eb!;
    var (f, ef) = await ReadMesh(ff.OpenReadStream(), req.Query["units"]);
    if (f is null) return ef!;
    return Results.Ok(Compare(model, b, f));
});

// Stored scans, one STL (mm) per subject and week, in the configured IScanStore.
app.MapPut("/subjects/{id}/scans/{week:int}", async (string id, int week, HttpRequest req, IScanStore store) =>
{
    if (!ScanKeys.Valid(id, week)) return Results.Problem("subject must match [a-z0-9-]{1,40}; week 0-520", statusCode: 400);
    using var ms = new MemoryStream();
    await req.Body.CopyToAsync(ms);
    var bytes = ms.ToArray();
    Mesh mesh;
    try { mesh = Stl.Read(bytes); }
    catch (InvalidDataException e) { return Results.Problem($"invalid STL: {e.Message}", statusCode: 400); }
    await store.PutAsync(id, week, bytes);
    return Results.Created($"/subjects/{id}/scans/{week}", new { subject = id, week, bytes = bytes.Length, triangles = mesh.FaceCount });
});

app.MapGet("/subjects/{id}/scans", async (string id, IScanStore store) =>
    ScanKeys.Valid(id, 0) ? Results.Ok(new { subject = id, weeks = await store.WeeksAsync(id) }) : Results.Problem("bad subject id", statusCode: 400));

app.MapGet("/subjects/{id}/change", async (string id, int from, int to, IScanStore store, Landmarks model) =>
{
    if (!ScanKeys.Valid(id, from) || !ScanKeys.Valid(id, to)) return Results.Problem("bad subject id or week", statusCode: 400);
    var a = await store.GetAsync(id, from);
    var b = await store.GetAsync(id, to);
    if (a is null || b is null) return Results.NotFound(new { subject = id, missing = a is null ? from : to });
    return Results.Ok(new { subject = id, from, to, result = Compare(model, Stl.Read(a).Scaled(1e-3), Stl.Read(b).Scaled(1e-3)) });
});

app.Run();

static async Task<(Mesh?, IResult?)> ReadMesh(Stream body, string? units)
{
    using var ms = new MemoryStream();
    await body.CopyToAsync(ms);
    double scale = units switch { null or "" or "mm" => 1e-3, "m" => 1.0, _ => double.NaN };
    if (double.IsNaN(scale)) return (null, Results.Problem("units must be 'mm' or 'm'", statusCode: 400));
    try
    {
        return (Stl.Read(ms.ToArray()).Scaled(scale), null);
    }
    catch (InvalidDataException e)
    {
        return (null, Results.Problem($"invalid STL: {e.Message}", statusCode: 400));
    }
}

static (V3[], object) Predict(Landmarks model, Mesh mesh)
{
    var sw = Stopwatch.StartNew();
    var depth = model.Scanner.Render(mesh);
    double render = sw.Elapsed.TotalMilliseconds;
    var lm = model.FromDepth(depth);
    return (lm, new { render, inference = sw.Elapsed.TotalMilliseconds - render });
}

static Dictionary<string, double[]> Named(V3[] lm) =>
    Landmarks.Names.Select((n, i) => (n, i)).ToDictionary(p => p.n, p => new[] { lm[p.i].X * 1e3, lm[p.i].Y * 1e3, lm[p.i].Z * 1e3 });

static object Compare(Landmarks model, Mesh baseline, Mesh followup)
{
    var sw = Stopwatch.StartNew();
    var change = Change.Surface(baseline, followup);
    double icpMs = sw.Elapsed.TotalMilliseconds;
    var gb = Girths.Compute(baseline, model.Predict(baseline));
    var gf = Girths.Compute(followup, model.Predict(followup));
    var d = change.DisplacementMm.Where(double.IsFinite).OrderBy(x => x).ToArray();
    return new
    {
        units = "mm",
        icp_rms_mm = change.IcpRmsMm,
        displacement_mm = new { p5 = Pct(d, 5), median = Pct(d, 50), p95 = Pct(d, 95), vertices = d.Length },
        girths = new { baseline = gb, followup = gf, delta_tape_mm = gb.ToDictionary(kv => kv.Key, kv => gf[kv.Key].TapeMm - kv.Value.TapeMm) },
        timings_ms = new { icp = icpMs, total = sw.Elapsed.TotalMilliseconds },
    };
}

static double Pct(double[] sorted, double p) => sorted.Length == 0 ? double.NaN : sorted[(int)Math.Round(p / 100 * (sorted.Length - 1))];

public partial class Program;
