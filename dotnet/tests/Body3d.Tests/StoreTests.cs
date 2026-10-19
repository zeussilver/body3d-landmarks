using System.Net;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;

namespace Body3d.Tests;

/// <summary>Stored-scan endpoints against the local directory store (no AWS needed).</summary>
public class StoreTests(WebApplicationFactory<Program> factory) : IClassFixture<WebApplicationFactory<Program>>
{
    readonly string _root = Directory.CreateTempSubdirectory("body3d-store").FullName;

    HttpClient Client => factory.WithWebHostBuilder(b => b
        .UseSetting("Body3d:ModelPath", Fixtures.OnnxPath)
        .UseSetting("Body3d:Store", "local")
        .UseSetting("Body3d:LocalRoot", _root)).CreateClient();

    static ByteArrayContent Stl(string file) => new(File.ReadAllBytes(Path.Combine(Fixtures.Dir, file)));

    [Fact]
    public async Task PutListAndCompareStoredScans()
    {
        var c = Client;
        Assert.Equal(HttpStatusCode.Created, (await c.PutAsync("/subjects/demo-1/scans/0", Stl("s1_week00.stl"))).StatusCode);
        Assert.Equal(HttpStatusCode.Created, (await c.PutAsync("/subjects/demo-1/scans/24", Stl("s1_week00.stl"))).StatusCode);
        var list = JsonDocument.Parse(await c.GetStringAsync("/subjects/demo-1/scans")).RootElement;
        Assert.Equal([0, 24], list.GetProperty("weeks").EnumerateArray().Select(w => w.GetInt32()));
        var change = JsonDocument.Parse(await c.GetStringAsync("/subjects/demo-1/change?from=0&to=24")).RootElement;
        Assert.InRange(change.GetProperty("result").GetProperty("displacement_mm").GetProperty("median").GetDouble(), -0.5, 0.5);
    }

    [Fact]
    public async Task RejectsBadIdsBadStlAndMissingScans()
    {
        var c = Client;
        Assert.Equal(HttpStatusCode.BadRequest, (await c.PutAsync("/subjects/Bad_ID/scans/0", Stl("s1_week00.stl"))).StatusCode);
        Assert.Equal(HttpStatusCode.BadRequest, (await c.PutAsync("/subjects/ok/scans/0", new ByteArrayContent(new byte[90]))).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await c.GetAsync("/subjects/nobody/change?from=0&to=8")).StatusCode);
    }
}
