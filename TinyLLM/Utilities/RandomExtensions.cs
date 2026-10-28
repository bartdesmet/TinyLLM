/// <summary>
/// Tiny helper for sampling a standard-normal value with <see cref="Random"/>.
/// .NET's BCL doesn't ship one, so we implement the Box-Muller transform.
/// </summary>
internal static class RandomExtensions
{
    public static double NextGaussian(this Random rng, double mean = 0.0, double stdDev = 1.0)
    {
        // Box-Muller: two uniforms in (0,1] -> one standard normal.
        double u1;
        do { u1 = rng.NextDouble(); } while (u1 <= double.Epsilon);
        var u2 = rng.NextDouble();
        var z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return mean + stdDev * z;
    }
}
