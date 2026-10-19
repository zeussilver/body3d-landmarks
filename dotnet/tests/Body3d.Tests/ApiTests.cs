using System.Net;
using System.Net.Http.Headers;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;

namespace Body3d.Tests;

public class ApiTests(WebApplicationFactory<Program> factory) : IClassFixture<WebApplicationFactory<Program>>
{
    HttpClient Client => factory.WithWebHostBuilder(b => b.UseSetting("Body3d:ModelPath", Fixtures.OnnxPath)).CreateClient();

    static ByteArrayContent Stl(string file) =>
        new(File.ReadAllBytes(Path.Combine(Fixtures.Dir, file))) { Headers = { ContentType = new MediaTypeHeaderValue("application/octet-stream") } };

    [Fact]
    public async Task GirthsEndpointReturnsLandmarksAndThreeGirths()
    {
        var res = await Client.PostAsync("/scans/girths", Stl("s1_week00.stl"));
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        var json = JsonDocument.Parse(await res.Content.ReadAsStringAsync()).RootElement;
        Assert.Equal(16, json.GetProperty("landmarks").EnumerateObject().Count());
        var waist = json.GetProperty("girths").GetProperty("waist").GetProperty("tapeMm").GetDouble();
        Assert.InRange(waist, 800, 1600);
    }

    [Fact]
    public async Task BadStlIsA400()
    {
        var res = await Client.PostAsync("/scans/landmarks", new ByteArrayContent(new byte[90]));
        Assert.Equal(HttpStatusCode.BadRequest, res.StatusCode);
    }

    [Fact]
    public async Task ChangeEndpointComparesTwoScans()
    {
        using var form = new MultipartFormDataContent { { Stl("s1_week00.stl"), "baseline", "a.stl" }, { Stl("s1_week00.stl"), "followup", "b.stl" } };
        var res = await Client.PostAsync("/scans/change", form);
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        var json = JsonDocument.Parse(await res.Content.ReadAsStringAsync()).RootElement;
        Assert.InRange(json.GetProperty("displacement_mm").GetProperty("median").GetDouble(), -0.5, 0.5);
    }
}
