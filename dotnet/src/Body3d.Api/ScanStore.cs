using System.Text.RegularExpressions;
using Amazon.S3;
using Amazon.S3.Model;

namespace Body3d.Api;

/// <summary>Where uploaded scans live: one STL per subject and week.</summary>
public interface IScanStore
{
    Task PutAsync(string subject, int week, byte[] stl);
    Task<byte[]?> GetAsync(string subject, int week);
    Task<IReadOnlyList<int>> WeeksAsync(string subject);
}

public static partial class ScanKeys
{
    [GeneratedRegex("^[a-z0-9-]{1,40}$")]
    private static partial Regex SubjectPattern();

    public static bool Valid(string subject, int week) => SubjectPattern().IsMatch(subject) && week is >= 0 and <= 520;

    public static string Name(int week) => $"week{week:00}.stl";

    public static int? Week(string name) =>
        name.StartsWith("week") && name.EndsWith(".stl") && int.TryParse(name[4..^4], out int w) ? w : null;
}

/// <summary>Local directory store (default; used by the tests).</summary>
public sealed class LocalScanStore(string root) : IScanStore
{
    string PathOf(string subject, int week) => Path.Combine(root, subject, ScanKeys.Name(week));

    public async Task PutAsync(string subject, int week, byte[] stl)
    {
        Directory.CreateDirectory(Path.Combine(root, subject));
        await File.WriteAllBytesAsync(PathOf(subject, week), stl);
    }

    public async Task<byte[]?> GetAsync(string subject, int week) =>
        File.Exists(PathOf(subject, week)) ? await File.ReadAllBytesAsync(PathOf(subject, week)) : null;

    public Task<IReadOnlyList<int>> WeeksAsync(string subject)
    {
        var dir = Path.Combine(root, subject);
        IReadOnlyList<int> weeks = Directory.Exists(dir)
            ? Directory.GetFiles(dir).Select(f => ScanKeys.Week(Path.GetFileName(f))).OfType<int>().Order().ToList()
            : [];
        return Task.FromResult(weeks);
    }
}

/// <summary>S3 store: s3://bucket/scans/{subject}/weekNN.stl, SHA-256 checksums on upload, and the
/// expected bucket owner on every call so a bucket re-created by someone else is never written to.</summary>
public sealed class S3ScanStore(IAmazonS3 s3, string bucket, string? expectedOwner, string prefix = "scans/") : IScanStore
{
    string Key(string subject, int week) => $"{prefix}{subject}/{ScanKeys.Name(week)}";

    public async Task PutAsync(string subject, int week, byte[] stl)
    {
        using var body = new MemoryStream(stl);
        await s3.PutObjectAsync(new PutObjectRequest
        {
            BucketName = bucket, Key = Key(subject, week), InputStream = body, ContentType = "model/stl",
            ChecksumAlgorithm = ChecksumAlgorithm.SHA256, ExpectedBucketOwner = expectedOwner,
        });
    }

    public async Task<byte[]?> GetAsync(string subject, int week)
    {
        try
        {
            using var res = await s3.GetObjectAsync(new GetObjectRequest
            {
                BucketName = bucket, Key = Key(subject, week), ExpectedBucketOwner = expectedOwner,
            });
            using var ms = new MemoryStream();
            await res.ResponseStream.CopyToAsync(ms);
            return ms.ToArray();
        }
        catch (AmazonS3Exception e) when (e.StatusCode == System.Net.HttpStatusCode.NotFound)
        {
            return null;
        }
    }

    public async Task<IReadOnlyList<int>> WeeksAsync(string subject)
    {
        var res = await s3.ListObjectsV2Async(new ListObjectsV2Request
        {
            BucketName = bucket, Prefix = $"{prefix}{subject}/", ExpectedBucketOwner = expectedOwner,
        });
        return (res.S3Objects ?? []).Select(o => ScanKeys.Week(o.Key[(o.Key.LastIndexOf('/') + 1)..])).OfType<int>().Order().ToList();
    }
}
