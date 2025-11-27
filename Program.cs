using System;
using System.IO;
using System.Text.RegularExpressions;
using System.Linq;
using System.Collections.Generic;
using Azure;
using Azure.AI.OpenAI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using OpenAI.Embeddings;
using System.Threading.Tasks;
using System.Text.Json;
using System.Diagnostics;
using System.Net;

class Program
{
    static async Task Main()
    {
        var config = new ConfigurationBuilder()
            .AddUserSecrets<Program>()
            .Build();

        // --- Configuration ---
        string endpoint = config["AzureOpenAI:Endpoint"] ?? throw new InvalidOperationException("Endpoint not found.");
        string key = config["AzureOpenAI:Key"] ?? throw new InvalidOperationException("Key not found.");
        string deploymentName = config["AzureOpenAI:ChatDeployment"] ?? "gpt-4";
        string embeddingModel = config["AzureOpenAI:EmbeddingDeployment"] ?? "text-embedding-ada-002";

        // --- Clients ---
        var chatClient = new AzureOpenAIClient(new Uri(endpoint), new AzureKeyCredential(key))
            .GetChatClient(deploymentName)
            .AsIChatClient();
        var embeddingClient = new AzureOpenAIClient(new Uri(endpoint), new AzureKeyCredential(key))
            .GetEmbeddingClient(embeddingModel);

        // --- Chunking Helper --- 
        IEnumerable<string> ChunkTextByHeading(string input)
        {
            var sections = Regex.Split(input, @"(?=^#\s)", RegexOptions.Multiline);

            foreach (var section in sections)
            {
                var trimmed = section.Trim();
                if (!string.IsNullOrEmpty(trimmed))
                    yield return trimmed;
            }
        }

        // --- File Reading ---
        string filePath = "report.md";
        if (!File.Exists(filePath))
        {
            Console.WriteLine($"Input file '{filePath}' not found. Place your markdown at that path and rerun.");
            return;
        }

        string text = File.ReadAllText(filePath);

        // Clean and reorganize text by main headings
        var structuredSections = ChunkTextByHeading(text);
        text = string.Join("\n\n---\n\n", structuredSections);

        // --- Input Quality Check ---
        bool IsInputSuitable(out string? qualityWarning)
        {
            qualityWarning = null;

            if (string.IsNullOrWhiteSpace(text))
            {
                qualityWarning = "Input document is empty.";
                return false;
            }
            if (text.Length < 100)
            {
                qualityWarning = "Input document is very short (<100 characters). Summary may be unreliable.";
                return false;
            }
            return true;
        }

        if (!IsInputSuitable(out var inputWarning))
        {
            var htmlWarning = BuildHtmlPage(
                "AI Summary Skipped",
                $"<p>{WebUtility.HtmlEncode(inputWarning)}</p>",
                "AI output was not generated due to insufficient input."
            );
            File.WriteAllText("report.html", htmlWarning);
            OpenFileInBrowser("report.html");
            return;
        }

        // --- Retry Helper ---
        async Task<T> WithRetries<T>(Func<Task<T>> fn, int maxTries = 3)
        {
            int delayMs = 500;
            for (int attempt = 1; attempt <= maxTries; attempt++)
            {
                try { return await fn(); }
                catch (Exception) when (attempt < maxTries)
                {
                    await Task.Delay(delayMs);
                    delayMs *= 2;
                }
            }
            throw new Exception("Retries exhausted");
        }

        List<string> allSummaries = new List<string>();

        string[] prompts = {
            $"""
        %% 
        {text}
        %%

        Act as a professional summarizer. Create a concise and comprehensive summary of the text enclosed in %% above, while adhering to the guidelines enclosed in [ ] below.

        Guidelines:

        [
        Create a summary that is detailed, thorough, in-depth, and complex, while maintaining clarity and conciseness.
        The summary must cover all the key points and main ideas presented in the original text, while also condensing the information into a concise and easy-to-understand format.
        Ensure that the summary includes relevant details and examples that support the main ideas, while avoiding any unnecessary information or repetition.
        Rely strictly on the provided text, without including external information.
        The length of the summary must be appropriate for the length and complexity of the original text. The length must allow to capture the main points and key details, without being overly long.
        Ensure that the summary is well-organized and easy to read, with clear headings and subheadings to guide the reader through each section. Format each section in paragraph form.
        ]
        """,
            $"""
        %% 
        {text}
        %%

        Act as an expert analyst. Provide a comprehensive summary of the text enclosed in %% above. Follow the guidelines in [ ] below.

        Guidelines:

        [
        Create a detailed and thorough summary that captures all main points and key ideas from the original text.
        Condense the information into a clear, concise format that is easy to understand.
        Include relevant details and examples that support the main ideas, while avoiding repetition.
        Use only information from the provided text.
        Make the summary appropriate in length for the complexity of the original text.
        Organize the summary with clear sections to guide the reader.
        Format each section as a paragraph.
        ]
        """,
            $"""
        %% 
        {text}
        %%

        As a professional content analyst, create a clear and comprehensive summary of the text in %% above, following the instructions in [ ] below.

        Instructions:

        [
        The summary should be thorough yet concise, covering all main points and ideas.
        Condense the information into an easy-to-understand format.
        Include supporting details and examples where relevant.
        Do not add external information - use only the provided text.
        Keep the summary length appropriate for the original text.
        Structure the summary with clear sections for readability.
        Use paragraph format for each section.
        ]
        """
        };

        var summaryTasks = prompts.Select(prompt =>
            WithRetries(() => chatClient.GetResponseAsync(
                prompt,
                new ChatOptions { Temperature = 0.1f, TopP = 0.9f }))
        ).ToArray();

        var responses = await Task.WhenAll(summaryTasks);
        var summaries = responses.Select(r => r.Text.Trim()).ToList();
        allSummaries = summaries;

        string consistencyLevel = "Unknown";

        if (allSummaries.Count == 3)
        {
            try
            {
                var embeddingTasks = allSummaries.Select(summary =>
                    embeddingClient.GenerateEmbeddingAsync(summary)
                ).ToArray();

                var embeddingResponses = (await Task.WhenAll(embeddingTasks))
                    .Select(response => response.Value.ToFloats().ToArray())
                    .ToList();

                var similarities = new List<double>();
                for (int i = 0; i < embeddingResponses.Count; i++)
                {
                    for (int j = i + 1; j < embeddingResponses.Count; j++)
                    {
                        double similarity = CalculateCosineSimilarity(embeddingResponses[i], embeddingResponses[j]);
                        similarities.Add(similarity);
                        Console.WriteLine($"Cosine similarity between summary {i + 1} and {j + 1}: {similarity:F4}");
                    }
                }

                double avgSimilarity = similarities.Average();
                Console.WriteLine($"Average cosine similarity between summaries: {avgSimilarity:F4}");

                if (avgSimilarity >= 0.85) consistencyLevel = "High";
                else if (avgSimilarity >= 0.70) consistencyLevel = "Medium";
                else consistencyLevel = "Low";
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Embedding calculation failed: {ex.Message}");
                consistencyLevel = "Unknown";
            }
        }

        var summaryValidationResults = new List<(int supportedCount, int totalCount, List<string> unsupportedLines)>();

        async Task<(int supportedCount, int totalCount, List<string> unsupportedLines)> RunValidationAsync(string summary, string sourceText)
        {
            string validationPrompt = $@"
            You are an impartial fact-checker. Evaluate if each sentence in the summary is supported by the source.

            Original source:
            %%
            {sourceText}
            %%

            Summary:
            %%
            {summary}
            %%

            Instructions:
            - Split the summary into sentences (use punctuation like ., !, ? as delimiters).
            - For each sentence, output the result as JSON in this format:
            {{ ""sentence"": ""Sentence text..."", ""supported"": true or false }}
            - After all sentences, return the full result as a single JSON object:
            {{
            ""sentences"": [
                {{ ""sentence"": ""..."", ""supported"": true }},
                {{ ""sentence"": ""..."", ""supported"": false }}
            ],
            ""confidence"": ""High"" | ""Medium"" | ""Low""
            }}
            - Do NOT include any explanations, commentary, or extra text outside this JSON.
            - Use these rules for the confidence value:
            - High: ≥90% of sentences supported
            - Medium: 60–89% supported
            - Low: <60% supported
            ";

            var validationResponse = await chatClient.GetResponseAsync(
                validationPrompt,
                new ChatOptions { Temperature = 0.0f, TopP = 1.0f }
            );

            string rawJson = validationResponse.Text.Trim();

            if (rawJson.StartsWith("```"))
            {
                var lines = rawJson.Split('\n');
                var jsonLines = lines.Skip(1).Take(lines.Length - 2);
                rawJson = string.Join("\n", jsonLines);
            }

            try
            {
                using var doc = JsonDocument.Parse(rawJson);
                var root = doc.RootElement;

                if (!root.TryGetProperty("sentences", out var sentencesElement) || !sentencesElement.ValueKind.Equals(JsonValueKind.Array))
                {
                    throw new JsonException("Missing or invalid 'sentences' array.");
                }

                var unsupportedLines = new List<string>();
                int supportedCount = 0;
                int totalCount = 0;

                foreach (var sentenceElem in sentencesElement.EnumerateArray())
                {
                    if (!sentenceElem.TryGetProperty("sentence", out var sentenceProp) ||
                        !sentenceElem.TryGetProperty("supported", out var supportedProp))
                        continue;

                    string sentenceText = sentenceProp.GetString() ?? "";
                    bool isSupported = supportedProp.GetBoolean();

                    totalCount++;
                    if (isSupported)
                        supportedCount++;
                    else
                        unsupportedLines.Add(sentenceText);
                }

                return (supportedCount, totalCount, unsupportedLines);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Validation JSON parsing failed: {ex.Message}. Raw response: {rawJson}");
                return (0, 0, new List<string>());
            }
        }

        foreach (var summary in summaries)
        {
            var validationRuns = new List<(int supportedCount, int totalCount, List<string> unsupportedLines)>();

            for (int i = 0; i < 3; i++)
            {
                var result = await RunValidationAsync(summary, text);
                validationRuns.Add(result);
            }

            var bestRun = validationRuns.OrderByDescending(r => r.supportedCount).First();
            summaryValidationResults.Add(bestRun);
        }

        int bestSummaryIndex = summaryValidationResults
            .Select((result, index) => new { result, index })
            .OrderByDescending(x => x.result.totalCount > 0 ? (double)x.result.supportedCount / x.result.totalCount : 0.0)
            .First().index;

        string bestSummary = summaries[bestSummaryIndex];
        var bestValidation = summaryValidationResults[bestSummaryIndex];

        double sourceConfidence = bestValidation.totalCount > 0
            ? (double)bestValidation.supportedCount / bestValidation.totalCount
            : 0.0;

        string sourceConfidenceLevel = sourceConfidence switch
        {
            >= 0.90 => "High",
            >= 0.60 => "Medium",
            _ => "Low"
        };

        string trustLevel = (sourceConfidenceLevel, consistencyLevel) switch
        {
            ("High", "High") => "Very Trustworthy",
            ("High", _) or (_, "High") => "Trustworthy",
            ("Medium", "Medium") => "Check Before Using",
            _ => "Not Reliable"
        };

        string reviewNotes = "";
        if (bestValidation.unsupportedLines.Count > 0 && sourceConfidenceLevel != "High")
        {
            reviewNotes = "\n**Review Needed:** The following statements could not be verified in the source:\n" +
                          string.Join("\n", bestValidation.unsupportedLines.Select(line => $"- {line}"));
        }

        string finalOutput = $"""
            {bestSummary}
            """;

        // ✨ NEW: Single-line, readable, italic notice with ALL data
        string aiNotice = $"File: {Path.GetFileName(filePath)} | Model: {deploymentName} | " +
                          $"Temperature: 0.1 (summary), 0.0 (validation) | Consistency: {consistencyLevel} | " +
                          $"Source Confidence: {sourceConfidenceLevel} | Trust: {trustLevel}. Please review before use.";

        var htmlContent = BuildHtmlPage("AI Summary (Trust-Enhanced)", ConvertMarkdownToHtml(finalOutput), aiNotice);
        File.WriteAllText("report.html", htmlContent);
        OpenFileInBrowser("report.html");
    }

    static string ConvertMarkdownToHtml(string markdown)
    {
        string html = WebUtility.HtmlEncode(markdown);

        html = Regex.Replace(html, "\\*\\*(.+?)\\*\\*", "<strong>$1</strong>");
        html = Regex.Replace(html, "(^|\\n)---(\\n|$)", "<hr />", RegexOptions.Multiline);
        html = Regex.Replace(html, "(^|\\n)- (.+?)(?=(\\n|$))", "$1<ul><li>$2</li></ul>", RegexOptions.Multiline);

        var paragraphs = html.Split(new string[] { "\n\n" }, StringSplitOptions.None)
            .Select(p => $"<p>{p.Replace("\n", "<br />")}</p>");

        return string.Join("\n", paragraphs);
    }

    static string BuildHtmlPage(string title, string bodyHtml, string aiNotice)
    {
        var css = @"body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; margin: 24px; background: #f6f8fa; color: #0b1226; }
            .container { max-width: 900px; margin: 0 auto; background: white; padding: 24px; border-radius: 12px; box-shadow: 0 8px 24px rgba(11,18,38,0.08); }
            h1 { font-size: 22px; margin-bottom: 16px; }
            .ai-notice { 
                margin-top: 24px; 
                padding-top: 16px; 
                border-top: 1px solid #e0e0e0; 
                color: #526070; 
                font-style: italic; 
                font-size: 14px;
            }
            hr { border: none; border-top: 1px solid #e6eef8; margin: 16px 0; }
            ul { margin: 8px 0 8px 18px; }
            footer { margin-top: 20px; font-size: 13px; color: #526070; }
            ";

        return $@"<!doctype html>
<html>
<head>
    <meta charset=""utf-8"">
    <meta name=""viewport"" content=""width=device-width,initial-scale=1"">
    <title>{WebUtility.HtmlEncode(title)}</title>
    <style>{css}</style>
</head>
<body>
    <div class=""container"">
        <header>
            <h1>{WebUtility.HtmlEncode(title)}</h1>
        </header>
        <main>
            {bodyHtml}
            <div class=""ai-notice"">
                {WebUtility.HtmlEncode(aiNotice)}
            </div>
        </main>
        <footer><p>Generated by AI</p></footer>
    </div>
</body>
</html>";
    }

    static void OpenFileInBrowser(string path)
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = path,
                UseShellExecute = true
            };
            Process.Start(psi);
        }
        catch
        {
            try
            {
                Process.Start(new ProcessStartInfo("cmd", $"/c start \"\" \"{path}\"") { CreateNoWindow = true });
            }
            catch { /* ignore */ }
        }
    }

    static double CalculateCosineSimilarity(float[] vectorA, float[] vectorB)
    {
        double dotProduct = 0.0;
        double magnitudeA = 0.0;
        double magnitudeB = 0.0;

        for (int i = 0; i < vectorA.Length; i++)
        {
            dotProduct += vectorA[i] * vectorB[i];
            magnitudeA += Math.Pow(vectorA[i], 2);
            magnitudeB += Math.Pow(vectorB[i], 2);
        }

        magnitudeA = Math.Sqrt(magnitudeA);
        magnitudeB = Math.Sqrt(magnitudeB);

        if (magnitudeA == 0 || magnitudeB == 0)
            return 0.0;

        return dotProduct / (magnitudeA * magnitudeB);
    }
}