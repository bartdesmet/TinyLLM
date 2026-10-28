using System.Text;

namespace TinyLLM.Tokenization;

internal static class TokenizerDebugFormatter
{
    internal static string Quote(string text)
    {
        var sb = new StringBuilder("\"");
        foreach (var c in text)
        {
            sb.Append(c switch
            {
                '"' => "\\\"",
                '\\' => "\\\\",
                '\n' => "\\n",
                '\r' => "\\r",
                '\t' => "\\t",
                _ when char.IsWhiteSpace(c) || char.IsControl(c) || char.IsSurrogate(c)
                    => $"\\u{(int)c:X4}",
                _ => c.ToString(),
            });
        }
        return sb.Append('"').ToString();
    }
}
