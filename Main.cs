using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Runtime.InteropServices;
using System.Threading.Channels;
using UglyToad.PdfPig;

namespace RagApp.Embeddings
{
    /// <summary>
    /// A single chunk of text plus its vector representation and provenance.
    /// </summary>
    public class EmbeddingRecord
    {
        public string Id { get; set; } = Guid.NewGuid().ToString();
        public string Text { get; set; } = "";
        public string Source { get; set; } = "";
        public float[] Vector { get; set; } = Array.Empty<float>();
    }

    /// <summary>
    /// Minimal contract for a chunk coming out of your existing chunker.
    /// Replace with your real chunk type, or map to this.
    /// </summary>
    public class TextChunkInput
    {
        public string Text { get; set; } = "";
        public string Source { get; set; } = "";
    }

    /// <summary>
    /// Common contract so KnowledgeBaseBuilder doesn't care which provider
    /// generated the vectors — OpenRouter, Gemini, Ollama, or anything else.
    /// </summary>
    public interface IEmbeddingClient
    {
        Task<List<float[]>> GetEmbeddingsAsync(IEnumerable<string> texts, CancellationToken ct = default);
        Task<float[]> GetEmbeddingAsync(string text, CancellationToken ct = default);
    }

    /// <summary>
    /// Thin wrapper around OpenRouter's OpenAI-compatible /embeddings endpoint.
    /// Supports batch input. Default model is the free Liquid embedding model
    /// (liquid/lfm-2.5-embedding-350m:free on OpenRouter).
    /// </summary>
    public class OpenRouterEmbeddingClient : IEmbeddingClient
    {
        private readonly HttpClient _http;
        private readonly string _model;

        public OpenRouterEmbeddingClient(
            HttpClient httpClient,
            string apiKey,
            string model = "liquid/lfm-2.5-embedding-350m:free")
        {
            _http = httpClient;
            _http.BaseAddress ??= new Uri("https://openrouter.ai/api/v1/");
            _http.Timeout = TimeSpan.FromMinutes(2);
            _http.DefaultRequestHeaders.Authorization =
                new AuthenticationHeaderValue("Bearer", apiKey);
            _model = model;
        }

        public async Task<List<float[]>> GetEmbeddingsAsync(
            IEnumerable<string> texts,
            CancellationToken ct = default)
        {
            var payload = new
            {
                model = _model,
                input = texts.ToArray()
            };

            var json = JsonSerializer.Serialize(payload);

            for (int attempt = 0; attempt < 6; attempt++)
            {
                using var content = new StringContent(json, Encoding.UTF8, "application/json");
                using var response = await _http.PostAsync("embeddings", content, ct);

                if (response.IsSuccessStatusCode)
                {
                    var body = await response.Content.ReadAsStringAsync(ct);
                    using var doc = JsonDocument.Parse(body);

                    var results = new List<float[]>();
                    foreach (var item in doc.RootElement.GetProperty("data").EnumerateArray())
                    {
                        var vector = item.GetProperty("embedding")
                            .EnumerateArray()
                            .Select(v => v.GetSingle())
                            .ToArray();
                        results.Add(vector);
                    }

                    return results;
                }

                var errorBody = await response.Content.ReadAsStringAsync(ct);

                if ((response.StatusCode == HttpStatusCode.TooManyRequests || (int)response.StatusCode == 529)
                    && attempt < 5)
                {
                    var delay = ParseRetryDelay(response, errorBody, attempt);

                    if (delay <= TimeSpan.FromMinutes(2))
                    {
                        Console.WriteLine($"OpenRouter rate limit ({(int)response.StatusCode}), retrying in {delay.TotalSeconds:F0}s...");
                        await Task.Delay(delay, ct);
                        continue;
                    }

                    // Long reset (e.g. the daily free-model window) — fail fast
                    // instead of burning the app waiting minutes/hours.
                    throw new HttpRequestException(
                        $"OpenRouter embeddings rate limited until {DateTimeOffset.UtcNow + delay:HH:mm} UTC ({delay.TotalMinutes:F0} min away): {errorBody}");
                }

                throw new HttpRequestException(
                    $"OpenRouter embeddings call failed ({(int)response.StatusCode}): {errorBody}");
            }

            throw new InvalidOperationException("OpenRouter embeddings retries exhausted.");
        }

        public async Task<float[]> GetEmbeddingAsync(string text, CancellationToken ct = default)
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("Text cannot be empty.", nameof(text));
            var results = await GetEmbeddingsAsync(new[] { text }, ct);
            if (results.Count == 0)
                throw new InvalidOperationException("Embedding provider returned no vectors for the query.");
            return results[0];
        }

        private static TimeSpan ParseRetryDelay(HttpResponseMessage response, string errorBody, int attempt)
        {
            long resetMs = 0;

            if (response.Headers.TryGetValues("X-RateLimit-Reset", out var headerReset))
            {
                long.TryParse(headerReset.FirstOrDefault(), out resetMs);
            }

            if (resetMs == 0)
            {
                try
                {
                    using var doc = JsonDocument.Parse(errorBody);
                    if (doc.RootElement.TryGetProperty("error", out var error) &&
                        error.TryGetProperty("metadata", out var metadata) &&
                        metadata.TryGetProperty("headers", out var headers) &&
                        headers.TryGetProperty("X-RateLimit-Reset", out var bodyReset))
                    {
                        long.TryParse(bodyReset.GetString(), out resetMs);
                    }
                }
                catch (JsonException)
                {
                }
            }

            if (resetMs > 0)
            {
                var delay = DateTimeOffset.FromUnixTimeMilliseconds(resetMs) - DateTimeOffset.UtcNow + TimeSpan.FromSeconds(1);
                if (delay > TimeSpan.Zero && delay < TimeSpan.FromHours(25))
                {
                    return delay;
                }
            }

            return TimeSpan.FromSeconds(5 * Math.Pow(2, attempt));
        }
    }

    /// <summary>
    /// Locally hosted embeddings via Ollama (https://ollama.com/). No API key,
    /// no rate limits — the model runs on your machine. Model is configurable;
    /// pull one with e.g. `ollama pull nomic-embed-text`.
    /// </summary>
    public class OllamaEmbeddingClient : IEmbeddingClient
    {
        private readonly HttpClient _http;
        private readonly string _model;

        public OllamaEmbeddingClient(
            HttpClient httpClient,
            string model = "nomic-embed-text",
            string baseUrl = "http://localhost:11434")
        {
            _http = httpClient;
            _model = model;
            _http.BaseAddress ??= new Uri(baseUrl.TrimEnd('/') + "/");
            _http.Timeout = TimeSpan.FromMinutes(10);
        }

        public async Task<List<float[]>> GetEmbeddingsAsync(
            IEnumerable<string> texts,
            CancellationToken ct = default)
        {
            var payload = new
            {
                model = _model,
                input = texts
            };

            var json = JsonSerializer.Serialize(payload);
            using var content = new StringContent(json, Encoding.UTF8, "application/json");

            using var response = await _http.PostAsync("api/embed", content, ct);
            if (!response.IsSuccessStatusCode)
            {
                var errorBody = await response.Content.ReadAsStringAsync(ct);
                throw new HttpRequestException(
                    $"Ollama embeddings call failed ({(int)response.StatusCode}): {errorBody}");
            }

            var body = await response.Content.ReadAsStringAsync(ct);
            using var doc = JsonDocument.Parse(body);

            var results = new List<float[]>();
            if (doc.RootElement.TryGetProperty("embeddings", out var embeddings))
            {
                foreach (var item in embeddings.EnumerateArray())
                {
                    var vector = item
                        .EnumerateArray()
                        .Select(v => v.GetSingle())
                        .ToArray();
                    results.Add(vector);
                }
            }

            if (results.Count == 0)
            {
                throw new InvalidOperationException(
                    $"Ollama returned no embeddings for the batch. Check that the model '{_model}' is pulled and responding.");
            }

            return results;
        }

        public async Task<float[]> GetEmbeddingAsync(string text, CancellationToken ct = default)
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("Text cannot be empty.", nameof(text));
            var results = await GetEmbeddingsAsync(new[] { text }, ct);

            // An unknown model comes back as a 200 with no vectors in it, so a bare
            // results[0] would throw an IndexOutOfRangeException with no hint about
            // which half of the setup is wrong.
            if (results.Count == 0)
            {
                throw new InvalidOperationException(
                    $"Ollama returned no embedding for the query. Check that the model " +
                    $"'{_model}' is pulled (ollama pull {_model}).");
            }

            return results[0];
        }
    }

    /// <summary>
    /// Free alternative to OpenRouter embeddings: Google's Gemini embedding
    /// API. Uses a separate Google AI Studio API key (not your OpenRouter
    /// key) and has a genuine no-cost free tier, unlike every embedding
    /// model currently on OpenRouter.
    /// Get a key at https://aistudio.google.com/apikey
    /// </summary>
    public class GeminiEmbeddingClient : IEmbeddingClient
    {
        private readonly HttpClient _http;
        private readonly string _apiKey;
        private readonly string _model;

        public GeminiEmbeddingClient(
            HttpClient httpClient,
            string apiKey,
            string model = "gemini-embedding-001")
        {
            _http = httpClient;
            _apiKey = apiKey;
            _model = model;
            _http.Timeout = TimeSpan.FromMinutes(2);
        }

        public async Task<List<float[]>> GetEmbeddingsAsync(
            IEnumerable<string> texts,
            CancellationToken ct = default)
        {
            var requests = texts.Select(t => new
            {
                model = $"models/{_model}",
                content = new { parts = new[] { new { text = t } } }
            });

            var json = JsonSerializer.Serialize(new { requests });
            var url = $"https://generativelanguage.googleapis.com/v1beta/models/{_model}:batchEmbedContents?key={_apiKey}";

            for (int attempt = 0; attempt < 6; attempt++)
            {
                using var content = new StringContent(json, Encoding.UTF8, "application/json");
                using var response = await _http.PostAsync(url, content, ct);

                if (response.IsSuccessStatusCode)
                {
                    var body = await response.Content.ReadAsStringAsync(ct);
                    using var doc = JsonDocument.Parse(body);

                    var results = new List<float[]>();
                    foreach (var item in doc.RootElement.GetProperty("embeddings").EnumerateArray())
                    {
                        var vector = item.GetProperty("values")
                            .EnumerateArray()
                            .Select(v => v.GetSingle())
                            .ToArray();
                        results.Add(vector);
                    }

                    return results;
                }

                var errorBody = await response.Content.ReadAsStringAsync(ct);

                if (response.StatusCode == HttpStatusCode.TooManyRequests && attempt < 5)
                {
                    var delay = ParseRetryDelay(errorBody, TimeSpan.FromSeconds(15 * (attempt + 1)));
                    Console.WriteLine($"Rate limited by Gemini, retrying in {delay.TotalSeconds:F0}s...");
                    await Task.Delay(delay, ct);
                    continue;
                }

                throw new HttpRequestException(
                    $"Gemini embeddings call failed ({(int)response.StatusCode}): {errorBody}");
            }

            throw new InvalidOperationException("Gemini embeddings retries exhausted.");
        }

        private static TimeSpan ParseRetryDelay(string errorBody, TimeSpan fallback)
        {
            try
            {
                using var doc = JsonDocument.Parse(errorBody);
                if (doc.RootElement.TryGetProperty("error", out var error) &&
                    error.TryGetProperty("details", out var details))
                {
                    foreach (var detail in details.EnumerateArray())
                    {
                        if (detail.TryGetProperty("retryDelay", out var retryDelay))
                        {
                            var value = retryDelay.GetString();
                            if (value is not null && TimeSpan.TryParse(value, out var parsed))
                            {
                                return parsed + TimeSpan.FromSeconds(1);
                            }
                        }
                    }
                }
            }
            catch (JsonException)
            {
            }
            return fallback;
        }

        public async Task<float[]> GetEmbeddingAsync(string text, CancellationToken ct = default)
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("Text cannot be empty.", nameof(text));
            var results = await GetEmbeddingsAsync(new[] { text }, ct);
            if (results.Count == 0)
                throw new InvalidOperationException("Gemini returned no embeddings for the query.");
            return results[0];
        }
    }

    /// <summary>
    /// Minimal in-memory vector store. Flat list, cosine similarity search.
    /// Guarded by a lock because drop-folder ingestion writes from a background
    /// worker while the console (and any future query path) reads.
    /// </summary>
    public class InMemoryVectorStore
    {
        private readonly List<EmbeddingRecord> _records = new();
        private readonly Lock _gate = new();

        public int Count
        {
            get { lock (_gate) return _records.Count; }
        }

        public IReadOnlyList<EmbeddingRecord> All
        {
            get { lock (_gate) return _records.ToArray(); }
        }

        public void Add(EmbeddingRecord record)
        {
            lock (_gate) _records.Add(record);
        }

        public void AddRange(IEnumerable<EmbeddingRecord> records)
        {
            lock (_gate) _records.AddRange(records);
        }

        public List<(EmbeddingRecord Record, float Score)> Search(float[] queryVector, int topK = 5)
        {
            List<EmbeddingRecord> snapshot;
            lock (_gate) snapshot = _records.ToList();

            return snapshot
                .Select(r => (Record: r, Score: CosineSimilarity(queryVector, r.Vector)))
                .OrderByDescending(x => x.Score)
                .Take(topK)
                .ToList();
        }

        public static float CosineSimilarity(float[] a, float[] b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Vectors must be the same length to compare.");

            float dot = 0, magA = 0, magB = 0;
            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
                magA += a[i] * a[i];
                magB += b[i] * b[i];
            }

            if (magA == 0 || magB == 0) return 0f;
            return dot / (MathF.Sqrt(magA) * MathF.Sqrt(magB));
        }
    }

    /// <summary>
    /// A single scored search result: which chunk matched, where it sits in the
    /// store, and how similar it was to the query vector.
    /// </summary>
    public class SearchHit
    {
        /// <summary>1-based global chunk number, matching what the console prints.</summary>
        public int Index { get; init; }

        public EmbeddingRecord Record { get; init; } = new();

        public float Score { get; init; }
    }

    /// <summary>
    /// Everything one semantic query produced: the ranked hits plus the counters
    /// needed to explain a thin or empty result (dimension mismatch, chunks below
    /// the score floor, how many were comparable at all).
    /// </summary>
    public class SemanticSearchResult
    {
        public string Query { get; init; } = "";
        public IReadOnlyList<SearchHit> Hits { get; init; } = Array.Empty<SearchHit>();
        public int QueryDimensions { get; init; }
        public int IndexedDimensions { get; init; }
        public int Scored { get; init; }
        public int SkippedDimensionMismatch { get; init; }
        public int BelowThreshold { get; init; }
        public float MinScore { get; init; }
    }

    /// <summary>
    /// The query half of the RAG loop: embeds a natural-language question with the
    /// same client that built the index, then ranks chunks by cosine similarity to
    /// the query vector.
    ///
    /// Scans the store directly rather than calling InMemoryVectorStore.Search
    /// because search has to survive a store holding vectors of mixed dimensions
    /// (embedding model swapped mid-run, or chunks ingested by two different
    /// providers). CosineSimilarity throws on a length mismatch, so records that
    /// cannot be compared are skipped and reported instead of failing the query.
    /// </summary>
    public class SemanticSearch
    {
        private readonly IEmbeddingClient _embeddingClient;
        private readonly InMemoryVectorStore _store;

        public SemanticSearch(IEmbeddingClient embeddingClient, InMemoryVectorStore store)
        {
            _embeddingClient = embeddingClient;
            _store = store;
        }

        /// <param name="query">Natural-language query; embedded before comparison.</param>
        /// <param name="topK">Maximum hits to return.</param>
        /// <param name="minScore">Cosine similarity floor; weaker hits are dropped.</param>
        /// <param name="ct">Cancels the embedding call.</param>
        public async Task<SemanticSearchResult> SearchAsync(
            string query,
            int topK = 5,
            float minScore = float.NegativeInfinity,
            CancellationToken ct = default)
        {
            if (string.IsNullOrWhiteSpace(query))
                throw new ArgumentException("Query cannot be empty.", nameof(query));
            if (topK < 1)
                throw new ArgumentOutOfRangeException(nameof(topK), topK, "topK must be at least 1.");

            var queryText = query.Trim();
            var snapshot = _store.All;

            if (snapshot.Count == 0)
                return new SemanticSearchResult { Query = queryText, MinScore = minScore };

            var queryVector = await _embeddingClient.GetEmbeddingAsync(queryText, ct);

            var hits = new List<SearchHit>();
            var skipped = 0;
            var below = 0;

            for (var i = 0; i < snapshot.Count; i++)
            {
                var record = snapshot[i];
                if (record.Vector.Length != queryVector.Length)
                {
                    skipped++;
                    continue;
                }

                var score = InMemoryVectorStore.CosineSimilarity(queryVector, record.Vector);
                if (score < minScore)
                {
                    below++;
                    continue;
                }

                hits.Add(new SearchHit { Index = i + 1, Record = record, Score = score });
            }

            return new SemanticSearchResult
            {
                Query = queryText,
                QueryDimensions = queryVector.Length,
                IndexedDimensions = DominantDimensions(snapshot),
                Scored = hits.Count,
                SkippedDimensionMismatch = skipped,
                BelowThreshold = below,
                MinScore = minScore,
                Hits = hits
                    .OrderByDescending(h => h.Score)
                    .Take(topK)
                    .ToList(),
            };
        }

        /// <summary>
        /// Dimensions of the largest same-dimension group in the store — what the
        /// index is mostly made of, and therefore what a query vector is being
        /// compared against. Ties break towards the wider vector.
        /// </summary>
        private static int DominantDimensions(IReadOnlyList<EmbeddingRecord> records)
        {
            return records
                .GroupBy(r => r.Vector.Length)
                .OrderByDescending(g => g.Count())
                .ThenByDescending(g => g.Key)
                .Select(g => g.Key)
                .FirstOrDefault();
        }
    }

    /// <summary>
    /// Whitespace flattening and truncation for console output. Shared so search
    /// results, the chunk listing and the file listing all trim text the same way.
    /// </summary>
    public static class ConsoleText
    {
        public static string Preview(string text, int length = 68)
        {
            var flat = Regex.Replace(text, "\\s+", " ").Trim();
            return Truncate(flat, length);
        }

        public static string Truncate(string value, int length)
        {
            if (value.Length <= length) return value;
            return string.Concat(value.AsSpan(0, length - 3), "...");
        }
    }

    /// <summary>
    /// Renders a SemanticSearchResult for the terminal. Shared by the interactive
    /// chunk console and the one-shot --query mode so both read identically.
    /// </summary>
    public static class SearchResultPrinter
    {
        public static void Print(SemanticSearchResult result, int topK)
        {
            Console.WriteLine();
            Console.WriteLine($"Query: \"{result.Query}\"");
            if (result.Hits.Count == 0)
            {
                Console.WriteLine();
                PrintEmpty(result);
                return;
            }

            var best = result.Hits[0];
            Console.WriteLine();
            var bestText = best.Record.Text.Trim();
            var answerLine = ExtractAnswer(bestText, result.Query);
            if (!string.IsNullOrWhiteSpace(answerLine))
            {
                Console.WriteLine($"Answer: {answerLine}");
                Console.WriteLine();
            }
            Console.WriteLine($"Source (chunk {best.Index}, score {best.Score:F3}, {best.Record.Source}):");
            Console.WriteLine(bestText);
            Console.WriteLine();
            Console.WriteLine("---");
            for (var i = 0; i < Math.Min(2, result.Hits.Count); i++)
            {
                var hit = result.Hits[i];
                Console.WriteLine($"[{i+1}] {hit.Score:F3} (chunk {hit.Index}): {ConsoleText.Preview(hit.Record.Text, 100)}");
            }
            if (result.Hits.Count > 2)
                Console.WriteLine($"(+{result.Hits.Count - 2} more matches)");
            Console.WriteLine();
        }

        private static void PrintEmpty(SemanticSearchResult result)
        {
            if (result.Scored == 0 && result.SkippedDimensionMismatch > 0)
            {
                Console.WriteLine(
                    $"  [fail] no stored vector matches the query's {result.QueryDimensions} dimensions " +
                    $"(the index is {result.IndexedDimensions}-dimensional) — the embedding model changed. " +
                    "Re-ingest the folder to rebuild it.");
                return;
            }

            if (result.BelowThreshold > 0)
            {
                Console.WriteLine(
                    $"  Nothing reached the {result.MinScore:F3} score floor; " +
                    $"{result.BelowThreshold} chunk(s) scored lower.");
                return;
            }

            Console.WriteLine("  No matches — nothing has been ingested yet.");
        }

        private static void PrintWarnings(SemanticSearchResult result)
        {
            if (result.SkippedDimensionMismatch > 0)
            {
                Console.WriteLine(
                    $"  [warn] skipped {result.SkippedDimensionMismatch} chunk(s) that are not " +
                    $"{result.QueryDimensions}-dimensional (index is mostly {result.IndexedDimensions}); " +
                    "re-ingest them with the current model.");
            }

            if (result.BelowThreshold > 0)
            {
                Console.WriteLine(
                    $"  [warn] {result.BelowThreshold} chunk(s) fell below the {result.MinScore:F3} score floor.");
            }
        }

        private static string ExtractAnswer(string text, string query)
        {
            var q = query.ToLowerInvariant();
            if (q.Contains("who"))
            {
                var m = Regex.Match(text, @"((?:[A-Z][a-z]+\s*){1,3}?)(?:wrote|created|developed|invented|founded|proposed|led|began)\b");
                if (m.Success)
                {
                    var name = Regex.Replace(m.Groups[1].Value.Trim(), @"(?<=[a-z])(?=[A-Z])", " ");
                    return name;
                }
                m = Regex.Match(text, @"([A-Z][a-z]+\s+[A-Z][a-z]+)\s+and\s+([A-Z][a-z]+\s+[A-Z][a-z]+)");
                if (m.Success) return $"{m.Groups[1].Value} and {m.Groups[2].Value}";
                m = Regex.Match(text, @"\b([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,})\b");
                if (m.Success) return m.Groups[1].Value;
                return "";
            }
            if (q.Contains("invent") || q.Contains("when") || q.Contains("create") || q.Contains("develop") || q.Contains("release"))
            {
                var m = Regex.Match(text, @"(?:invented|created|developed|began|started|released|launched).*?(1[89]\d{2})", RegexOptions.IgnoreCase);
                if (m.Success && m.Groups.Count > 1) return m.Groups[1].Value;
                m = Regex.Match(text, @"\b(1[89]\d{2})\b", RegexOptions.IgnoreCase);
                if (m.Success) return m.Groups[1].Value;
            }
            if (q.Contains("why") || q.Contains("reason") || q.Contains("advantage") || q.Contains("choose") || q.Contains("chosen") || q.Contains("over") || q.Contains("benefit"))
            {
                var m = Regex.Match(text, @"(?:because|to enable|to provide|so that|allows|independent of)[^\.\n]{10,120}(?:\.[^\n]|$)", RegexOptions.IgnoreCase);
                if (m.Success)
                {
                    var s = Regex.Replace(m.Value.Trim(), @"\s+", " ");
                    return s;
                }
                m = Regex.Match(text, @"platform independent[^\.]{0,60}", RegexOptions.IgnoreCase);
                if (m.Success) return $"Platform independent - {m.Value.Trim()}.";
                m = Regex.Match(text, @"preserve[^\.]{10,60}formatting[^\.]{0,20}", RegexOptions.IgnoreCase);
                if (m.Success) return $"To preserve formatting across platforms - {m.Value.Trim()}.";
            }
            return "";
        }
    }

    /// <summary>
    /// Stage 1: Knowledge Base Preparation.
    /// Takes chunks, embeds them in batches via the injected embedding client,
    /// populates the vector store.
    /// Runs alongside your existing BM25 index, not instead of it.
    /// </summary>
    public class KnowledgeBaseBuilder
    {
        private readonly IEmbeddingClient _embeddingClient;
        private readonly InMemoryVectorStore _store;
        private readonly int _batchSize;

        public KnowledgeBaseBuilder(
            IEmbeddingClient embeddingClient,
            InMemoryVectorStore store,
            int batchSize = 32)
        {
            _embeddingClient = embeddingClient;
            _store = store;
            _batchSize = batchSize;
        }

        public async Task BuildAsync(IEnumerable<TextChunkInput> chunks, CancellationToken ct = default)
        {
            var chunkList = chunks.ToList();
            if (chunkList.Count == 0) return;

            var totalBatches = (int)Math.Ceiling(chunkList.Count / (double)_batchSize);
            for (int i = 0; i < chunkList.Count; i += _batchSize)
            {
                var batchIndex = (i / _batchSize) + 1;
                var batch = chunkList.Skip(i).Take(_batchSize).ToList();
                var texts = batch.Select(c => c.Text).ToList();

                Console.WriteLine($"  embedding batch {batchIndex}/{totalBatches} ({batch.Count} chunk(s))...");
                var vectors = await _embeddingClient.GetEmbeddingsAsync(texts, ct);

                if (vectors.Count != batch.Count)
                {
                    throw new InvalidOperationException(
                        $"Embedding provider returned {vectors.Count} vectors for a batch of {batch.Count} texts; " +
                        "refusing to continue to avoid mis-attributing vectors to chunks.");
                }

                for (int j = 0; j < batch.Count; j++)
                {
                    _store.Add(new EmbeddingRecord
                    {
                        Text = batch[j].Text,
                        Source = batch[j].Source,
                        Vector = vectors[j]
                    });
                }
            }
        }

        /// <summary>
        /// One-call path: fetch each URL, strip HTML tags, chunk, embed, store.
        /// </summary>
        public async Task<int> BuildFromUrlsAsync(
            IEnumerable<string> urls,
            HttpClient loaderHttpClient,
            int chunkSize = 500,
            int overlap = 50,
            CancellationToken ct = default)
        {
            var loader = new WebDocumentLoader(loaderHttpClient);
            var documents = await loader.LoadManyAsync(urls, ct);

            var chunks = TextChunker.ChunkMany(documents, chunkSize, overlap);

            await BuildAsync(chunks, ct);

            return chunks.Count;
        }

        /// <summary>
        /// One-call path: read each PDF file (or every *.pdf under a directory),
        /// extract page text, chunk, embed, store.
        /// </summary>
        public async Task<int> BuildFromPdfFilesAsync(
            IEnumerable<string> pdfPaths,
            int chunkSize = 500,
            int overlap = 50,
            CancellationToken ct = default)
        {
            var documents = PdfDocumentLoader.LoadMany(pdfPaths);

            var chunks = TextChunker.ChunkMany(documents, chunkSize, overlap);

            await BuildAsync(chunks, ct);

            return chunks.Count;
        }
    }

    /// <summary>
    /// Strips HTML down to plain text: drops script/style blocks entirely,
    /// removes all remaining tags, decodes entities, collapses whitespace.
    /// Regex-based on purpose — no HTML parser dependency, good enough to
    /// get readable text out of a scraped page.
    /// </summary>
    public static class HtmlTextExtractor
    {
        public static string StripTags(string html)
        {
            if (string.IsNullOrWhiteSpace(html)) return "";

            html = Regex.Replace(html, "<script[^>]*>[\\s\\S]*?</script>", " ", RegexOptions.IgnoreCase);
            html = Regex.Replace(html, "<style[^>]*>[\\s\\S]*?</style>", " ", RegexOptions.IgnoreCase);
            html = Regex.Replace(html, "<[^>]+>", " ");
            html = System.Net.WebUtility.HtmlDecode(html);
            html = Regex.Replace(html, "\\s+", " ").Trim();

            return html;
        }
    }

    /// <summary>
    /// Fetches a URL and returns its plain-text content (tags stripped),
    /// paired with the URL as its source.
    /// </summary>
    public class WebDocumentLoader
    {
        private readonly HttpClient _http;

        public WebDocumentLoader(HttpClient httpClient)
        {
            _http = httpClient;
            if (!_http.DefaultRequestHeaders.Contains("User-Agent"))
            {
                _http.DefaultRequestHeaders.Add(
                    "User-Agent",
                    "Mozilla/5.0 (compatible; SimpleCrawler/1.0; +https://example.com/bot)");
            }
        }

        public async Task<(string Text, string Source)> LoadAsync(string url, CancellationToken ct = default)
        {
            var html = await _http.GetStringAsync(url, ct);
            var text = HtmlTextExtractor.StripTags(html);
            return (text, url);
        }

        public async Task<List<(string Text, string Source)>> LoadManyAsync(
            IEnumerable<string> urls,
            CancellationToken ct = default)
        {
            var results = new List<(string Text, string Source)>();
            foreach (var url in urls)
            {
                try
                {
                    results.Add(await LoadAsync(url, ct));
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to load {url}: {ex.Message}");
                }
            }
            return results;
        }
    }

    /// <summary>
    /// Reads PDF files off disk and returns per-PDF page text, paired with the
    /// file name as the source. Accepts individual .pdf paths or directories
    /// (every *.pdf inside is loaded). Uses the pure-managed PdfPig library —
    /// no native dependencies, no OCR. Scanned/image-only PDFs yield no text.
    /// </summary>
    public static class PdfDocumentLoader
    {
        public static List<(string Text, string Source)> LoadMany(IEnumerable<string> paths)
        {
            var results = new List<(string Text, string Source)>();

            foreach (var path in paths)
            {
                if (Directory.Exists(path))
                {
                    foreach (var file in Directory.EnumerateFiles(path, "*.pdf"))
                    {
                        results.Add(Load(file));
                    }
                }
                else
                {
                    results.Add(Load(path));
                }
            }

            return results;
        }

        public static (string Text, string Source) Load(string pdfPath)
        {
            try
            {
                using var document = PdfDocument.Open(pdfPath);

                var sb = new StringBuilder();
                foreach (var page in document.GetPages())
                {
                    sb.AppendLine(page.Text);
                }

                var text = sb.ToString();
                if (string.IsNullOrWhiteSpace(text))
                {
                    Console.WriteLine($"No text extracted from {pdfPath} (scanned/image PDF?).");
                }

                return (text, Path.GetFileName(pdfPath));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to read {pdfPath}: {ex.Message}");
                return ("", Path.GetFileName(pdfPath));
            }
        }
    }

    /// <summary>
    /// Splits raw document text into overlapping chunks. Tries to break on
    /// sentence boundaries near the target size instead of mid-word.
    /// Swap this out (or map its output) if your existing chunker should
    /// be used instead — this is a standalone default so Stage 1 is fully
    /// self-contained and runnable.
    /// </summary>
    public static class TextChunker
    {
        public static List<TextChunkInput> Chunk(string text, string source, int chunkSize = 500, int overlap = 50)
        {
            var chunks = new List<TextChunkInput>();
            if (string.IsNullOrWhiteSpace(text) || chunkSize <= 0) return chunks;

            if (overlap < 0) overlap = 0;
            if (overlap >= chunkSize) overlap = chunkSize - 1;

            text = text.Trim();
            int position = 0;

            while (position < text.Length)
            {
                int length = Math.Min(chunkSize, text.Length - position);
                int end = position + length;

                if (end < text.Length)
                {
                    int searchStart = Math.Max(position, end - length);
                    if (end > searchStart)
                    {
                        int lastBreak = text.LastIndexOfAny(
                            new[] { '.', '!', '?', '\n' },
                            end - 1,
                            end - searchStart);
                        if (lastBreak > position)
                        {
                            end = lastBreak + 1;
                        }
                    }
                }

                var chunkText = text.Substring(position, end - position).Trim();
                if (chunkText.Length > 0)
                {
                    chunks.Add(new TextChunkInput { Text = chunkText, Source = source });
                }

                if (end >= text.Length) break;

                position = Math.Max(end - overlap, position + 1);
            }

            return chunks;
        }

        public static List<TextChunkInput> ChunkMany(
            IEnumerable<(string Text, string Source)> documents,
            int chunkSize = 500,
            int overlap = 50)
        {
            var all = new List<TextChunkInput>();
            foreach (var (docText, docSource) in documents)
            {
                all.AddRange(Chunk(docText, docSource, chunkSize, overlap));
            }
            return all;
        }
    }

    /// <summary>
    /// Drop-folder ingestion. Point it at a directory, drop PDFs in, and each new
    /// file is extracted, chunked, embedded and added to the vector store on its
    /// own. A FileSystemWatcher reacts to drops, plus a periodic rescan as a
    /// safety net for anything the OS watcher misses.
    ///
    /// Ingestion is serialised through a single-consumer channel so embedding
    /// calls never overlap (keeps rate limits and the store simple). Files are
    /// held until their size stops changing, because a large PDF is still being
    /// copied for a moment after the drop.
    ///
    /// Vectors live in the store, so a restart re-embeds from scratch — the set
    /// of "already ingested" files resets with the process.
    /// </summary>
    public class PdfIngestWatcher
    {
        private const int SizeStablePolls = 3;
        private const int MaxWaitPolls = 60;
        private static readonly TimeSpan PollInterval = TimeSpan.FromMilliseconds(250);
        private static readonly TimeSpan RescanInterval = TimeSpan.FromSeconds(30);

        private readonly IEmbeddingClient _embeddingClient;
        private readonly InMemoryVectorStore _store;
        private readonly int _chunkSize;
        private readonly int _overlap;

        public PdfIngestWatcher(
            IEmbeddingClient embeddingClient,
            InMemoryVectorStore store,
            int chunkSize = 500,
            int overlap = 50)
        {
            _embeddingClient = embeddingClient;
            _store = store;
            _chunkSize = chunkSize;
            _overlap = overlap;
        }

        public async Task RunAsync(string directory, CancellationToken ct)
        {
            var full = Path.GetFullPath(directory);
            Directory.CreateDirectory(full);

            var queue = Channel.CreateUnbounded<string>();
            var ingested = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            using var watcher = new FileSystemWatcher(full)
            {
                NotifyFilter = NotifyFilters.FileName | NotifyFilters.LastWrite | NotifyFilters.Size,
                IncludeSubdirectories = false
            };

            void Enqueue(string path)
            {
                if (string.Equals(Path.GetExtension(path), ".pdf", StringComparison.OrdinalIgnoreCase))
                {
                    queue.Writer.TryWrite(Path.GetFullPath(path));
                }
            }

            watcher.Created += (_, e) => Enqueue(e.FullPath);
            watcher.Changed += (_, e) => Enqueue(e.FullPath);
            watcher.Renamed += (_, e) => Enqueue(e.FullPath);
            watcher.Error += (_, e) => Console.WriteLine($"Watcher glitch: {e.GetException().Message}. Rescans will pick up anything missed.");
            watcher.EnableRaisingEvents = true;

            foreach (var pdf in EnumeratePdfs(full))
            {
                queue.Writer.TryWrite(pdf);
            }

            var consumer = ConsumeAsync(queue.Reader, ingested, ct);

            using var timer = new PeriodicTimer(RescanInterval);
            try
            {
                while (await timer.WaitForNextTickAsync(ct))
                {
                    foreach (var pdf in EnumeratePdfs(full))
                    {
                        if (!ingested.Contains(pdf)) queue.Writer.TryWrite(pdf);
                    }
                }
            }
            catch (OperationCanceledException)
            {
            }

            queue.Writer.TryComplete();

            try
            {
                await consumer;
            }
            catch (OperationCanceledException)
            {
                // Normal shutdown: the consumer's in-flight embed was cancelled.
            }
        }

        private static List<string> EnumeratePdfs(string directory)
        {
            try
            {
                return Directory.EnumerateFiles(directory, "*.pdf")
                    .Select(Path.GetFullPath)
                    .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
                    .ToList();
            }
            catch (DirectoryNotFoundException)
            {
                return new List<string>();
            }
        }

        private async Task ConsumeAsync(
            ChannelReader<string> reader,
            HashSet<string> ingested,
            CancellationToken ct)
        {
            while (await reader.WaitToReadAsync(ct))
            {
                while (reader.TryRead(out var path))
                {
                    if (ingested.Contains(path)) continue;

                    bool finished;
                    try
                    {
                        (finished, _) = await IngestAsync(path, ct);
                    }
                    catch (OperationCanceledException) when (ct.IsCancellationRequested)
                    {
                        return;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[fail] {Path.GetFileName(path)}: {ex.Message} (skipping after failure)");
                        ingested.Add(path);
                        continue;
                    }

                    // A file we couldn't read to completion stays un-ingested so a
                    // later rescan retries it. One that yielded no text is marked
                    // done — rescanning a scanned PDF would just repeat the warning.
                    if (finished) ingested.Add(path);
                }
            }
        }

        private async Task<(bool Finished, int Chunks)> IngestAsync(string path, CancellationToken ct)
        {
            if (!await WaitUntilReadableAsync(path, ct))
            {
                if (!File.Exists(path))
                {
                    Console.WriteLine($"[skip] {Path.GetFileName(path)}: gone before it could be read.");
                    return (true, 0);
                }

                Console.WriteLine($"[wait] {Path.GetFileName(path)}: still being written, will retry.");
                return (false, 0);
            }

            var (text, source) = PdfDocumentLoader.Load(path);

            if (string.IsNullOrWhiteSpace(text))
            {
                Console.WriteLine($"[warn] {source}: no text extracted (scanned/image-only PDF?), nothing vectorized.");
                return (true, 0);
            }

            var chunks = TextChunker.Chunk(text, source, _chunkSize, _overlap);
            if (chunks.Count == 0)
            {
                Console.WriteLine($"[warn] {source}: text extracted but produced no chunks.");
                return (true, 0);
            }

            var builder = new KnowledgeBaseBuilder(_embeddingClient, _store);
            await builder.BuildAsync(chunks, ct);

            var all = _store.All;
            var dims = all.Count > 0 ? all[^1].Vector.Length : 0;
            Console.WriteLine($"[ok]   {source}: {chunks.Count} chunks vectorized ({dims} dims, {_store.Count} total).");

            return (true, chunks.Count);
        }

        /// <summary>
        /// Blocks until the file's size stops changing and it can be opened
        /// exclusively, so we never parse a half-copied PDF. Returns false if the
        /// file is still locked or empty.
        /// </summary>
        private static async Task<bool> WaitUntilReadableAsync(string path, CancellationToken ct)
        {
            var stable = 0;
            long previousSize = -1;

            for (var attempt = 0; attempt < MaxWaitPolls && stable < SizeStablePolls; attempt++)
            {
                if (!File.Exists(path)) return false;

                long size;
                try
                {
                    size = new FileInfo(path).Length;
                }
                catch (IOException)
                {
                    return false;
                }

                stable = size > 0 && size == previousSize ? stable + 1 : 0;
                previousSize = size;

                await Task.Delay(PollInterval, ct);
            }

            if (!File.Exists(path)) return false;

            try
            {
                await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None);
                return stream.Length > 0;
            }
            catch (IOException)
            {
                return false;
            }
            catch (UnauthorizedAccessException)
            {
                return false;
            }
        }
    }

    // ---------------------------------------------------------------------
    // Example wiring — goes straight from a list of URLs -> fetch ->
    // strip HTML -> chunk -> embed -> vector store. Swap urls for your
    // actual scraped links (e.g. what your DuckDuckGo search returns).
    // Embeddings run locally via Ollama — no API keys, no rate limits.
    // ---------------------------------------------------------------------
    public static class Stage1Example
    {
        public static async Task RunAsync(IEmbeddingClient embeddingClient)
        {
            using var loaderHttpClient = new HttpClient();

            var vectorStore = new InMemoryVectorStore();
            var builder = new KnowledgeBaseBuilder(embeddingClient, vectorStore);

            var urls = new List<string>
            {
                "https://en.wikipedia.org/wiki/Eiffel_Tower",
            };

            var chunkCount = await builder.BuildFromUrlsAsync(urls, loaderHttpClient);

            PrintSummary(vectorStore, $"Fetched {urls.Count} URL(s), chunked into {chunkCount} pieces, knowledge base ready: {vectorStore.Count} chunks embedded.");
        }

        public static async Task RunFromPdfsAsync(
            IEmbeddingClient embeddingClient,
            IEnumerable<string> pdfPaths,
            int chunkSize = 500,
            int overlap = 50)
        {
            var pdfList = pdfPaths
                .Where(p => !string.IsNullOrWhiteSpace(p))
                .Select(p => Path.GetFullPath(p))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            if (pdfList.Count == 0)
            {
                Console.WriteLine("No PDF paths provided.");
                return;
            }

            Console.WriteLine($"Loading PDF path(s): {string.Join(", ", pdfList)}");

            var anyExists = false;
            foreach (var p in pdfList)
            {
                if (File.Exists(p) || Directory.Exists(p))
                {
                    anyExists = true;
                    break;
                }
            }

            if (!anyExists)
            {
                Console.WriteLine("None of the provided paths exist.");
                Console.WriteLine("No chunks embedded. Possible causes:");
                Console.WriteLine("  - No PDFs found in the given paths");
                Console.WriteLine("  - Paths are invalid");
                return;
            }

            var vectorStore = new InMemoryVectorStore();
            var builder = new KnowledgeBaseBuilder(embeddingClient, vectorStore);

            var chunkCount = await builder.BuildFromPdfFilesAsync(pdfList, chunkSize, overlap);

            PrintSummary(vectorStore, $"Loaded {pdfList.Count} PDF path(s), chunked into {chunkCount} pieces, knowledge base ready: {vectorStore.Count} chunks embedded.");
        }

        /// <summary>
        /// One-shot semantic query: build the knowledge base from the given PDFs,
        /// then answer a single question and print the ranked chunks. Meant for
        /// piping — the store is thrown away afterwards, so a restart re-embeds
        /// from the source PDFs.
        /// </summary>
        public static async Task RunQueryAsync(
            IEmbeddingClient embeddingClient,
            string query,
            IEnumerable<string> pdfPaths,
            int chunkSize = 500,
            int overlap = 50,
            int topK = 5,
            float minScore = float.NegativeInfinity)
        {
            var pdfList = pdfPaths
                .Where(p => !string.IsNullOrWhiteSpace(p))
                .Select(p => Path.GetFullPath(p))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            if (pdfList.Count == 0)
            {
                Console.WriteLine("No PDF paths provided.");
                return;
            }

            Console.WriteLine($"Loading PDF path(s): {string.Join(", ", pdfList)}");

            var anyExists = false;
            foreach (var p in pdfList)
            {
                if (File.Exists(p) || Directory.Exists(p))
                {
                    anyExists = true;
                    break;
                }
            }

            if (!anyExists)
            {
                Console.WriteLine("None of the provided paths exist.");
                Console.WriteLine("No chunks embedded. Possible causes:");
                Console.WriteLine("  - No PDFs found in the given paths");
                Console.WriteLine("  - Paths are invalid");
                return;
            }

            var vectorStore = new InMemoryVectorStore();
            var builder = new KnowledgeBaseBuilder(embeddingClient, vectorStore);

            var chunkCount = await builder.BuildFromPdfFilesAsync(pdfList, chunkSize, overlap);

            Console.WriteLine(
                $"Loaded {pdfList.Count} PDF path(s), chunked into {chunkCount} pieces, " +
                $"knowledge base ready: {vectorStore.Count} chunks embedded.");

            if (vectorStore.Count == 0)
            {
                Console.WriteLine();
                Console.WriteLine("No matches — nothing has been ingested yet.");
                Console.WriteLine("No chunks embedded. Possible causes:");
                Console.WriteLine("  - No PDFs found in the given paths");
                Console.WriteLine("  - PDFs are scanned/image-only with no extractable text");
                Console.WriteLine("  - Files are unreadable or were skipped");
                return;
            }

            var search = new SemanticSearch(embeddingClient, vectorStore);
            var result = await search.SearchAsync(query, topK, minScore);

            SearchResultPrinter.Print(result, topK);
        }

        private static void PrintSummary(InMemoryVectorStore vectorStore, string summary)
        {
            Console.WriteLine(summary);

            if (vectorStore.Count == 0)
            {
                Console.WriteLine();
                Console.WriteLine("No chunks embedded. Possible causes:");
                Console.WriteLine("  - No PDFs found in the given paths");
                Console.WriteLine("  - PDFs are scanned/image-only with no extractable text");
                Console.WriteLine("  - Files are unreadable or were skipped");
                Console.WriteLine();
                return;
            }

            if (vectorStore.Count > 0)
            {
                var reference = vectorStore.All[0];
                var refPreview = reference.Text.Length > 60 ? reference.Text.Substring(0, 60) + "..." : reference.Text;

                Console.WriteLine();
                Console.WriteLine($"Reference chunk: \"{refPreview}\"");
                Console.WriteLine($"Vector dimensions: {reference.Vector.Length}");
                Console.WriteLine();
                Console.WriteLine("Top 10 closest chunks (by cosine similarity):");

                var closest = vectorStore.Search(reference.Vector, topK: 10);

                foreach (var (record, score) in closest)
                {
                    var preview = record.Text.Length > 60 ? record.Text.Substring(0, 60) + "..." : record.Text;
                    Console.WriteLine($"  [{score:F4}] {preview}");
                }
            }
        }
    }

    /// <summary>
    /// Interactive console for inspecting what has been vectorized. Chunk
    /// numbers are global across every ingested PDF, in ingestion order, so "42"
    /// means the same chunk for the lifetime of the process no matter how many
    /// files went in. `files` shows each PDF's global range, so you can tell
    /// which document a number belongs to.
    ///
    /// `search` runs the semantic half of the loop: the query is embedded with the
    /// same model that built the index and ranked against every chunk, so results
    /// are meaning-based rather than keyword-based. Because search reports global
    /// chunk numbers, a hit can be printed in full with the plain index commands.
    ///
    /// Meant to be pumped on a background task alongside PdfIngestWatcher: it
    /// reads stdin while ingestion continues, and stops listening (without
    /// shutting anything down) if stdin closes.
    /// </summary>
    public class ChunkConsole
    {
        private const int MaxPrintPerCommand = 25;
        private const int DefaultTopK = 5;

        private static readonly string[] SearchVerbs = { "s", "search", "find", "semantic" };

        private readonly InMemoryVectorStore _store;
        private readonly SemanticSearch _search;
        private readonly Action _onQuit;

        public ChunkConsole(
            InMemoryVectorStore store,
            IEmbeddingClient embeddingClient,
            Action onQuit)
        {
            _store = store;
            _search = new SemanticSearch(embeddingClient, store);
            _onQuit = onQuit;
        }

        /// <returns>Task that completes when the user quits or stdin closes.</returns>
        public async Task RunAsync(CancellationToken ct)
        {
            Console.WriteLine("Chunk commands: <n> | <n-m> | <n,m> | search <query> | list | files | stats | help | quit");
            Console.WriteLine();

            while (!ct.IsCancellationRequested)
            {
                Console.Write("chunk> ");
                Console.Out.Flush();

                string? line;
                try
                {
                    line = await Console.In.ReadLineAsync(ct);
                }
                catch (OperationCanceledException)
                {
                    return;
                }

                if (line is null) return; // stdin closed — keep watching

                var input = line.Trim();
                if (input.Length == 0) continue;

                if (await HandleCommand(input, ct)) return;
            }
        }

        /// <returns>True if the user asked to quit.</returns>
        private async Task<bool> HandleCommand(string input, CancellationToken ct)
        {
            switch (input.ToLowerInvariant())
            {
                case "q":
                case "quit":
                case "exit":
                    _onQuit();
                    return true;

                case "h":
                case "help":
                case "?":
                    PrintHelp();
                    return false;

                case "l":
                case "list":
                    PrintList();
                    return false;

                case "f":
                case "files":
                    PrintFiles();
                    return false;

                case "stats":
                    PrintStats();
                    return false;
            }

            // Checked before the numeric commands so a query like "search 42"
            // is treated as text, not as chunk 42.
            if (TryParseSearch(input, out var query, out var topK))
            {
                await RunSearchAsync(query, topK, ct);
                return false;
            }

            var total = _store.Count;

            if (TryParseIndices(input, total, out var indices, out var missing))
            {
                if (total == 0)
                {
                    Console.WriteLine("No chunks yet — drop a PDF into the watched folder.");
                    return false;
                }

                foreach (var index in indices.Take(MaxPrintPerCommand))
                {
                    PrintChunk(index, total);
                }

                if (indices.Count > MaxPrintPerCommand)
                {
                    Console.WriteLine($"  ... {indices.Count - MaxPrintPerCommand} more selected. Narrow the range to see them.");
                    Console.WriteLine();
                }

                if (Clamped)
                {
                    Console.WriteLine($"Range trimmed to the {total} chunks ingested so far.");
                }

                foreach (var index in missing)
                {
                    Console.WriteLine($"No chunk {index} (store has {total}).");
                }

                return false;
            }

            Console.WriteLine($"Unrecognised input: {input}. Type 'help' for commands.");
            return false;
        }

        /// <summary>
        /// Recognises "search &lt;query&gt;" and "search &lt;query&gt; &lt;n&gt;".
        /// The trailing number is only read as a result count when something is
        /// left over to be the query, so a bare "search 42" still searches for
        /// the text 42.
        /// </summary>
        private static bool TryParseSearch(string input, out string query, out int topK)
        {
            query = "";
            topK = DefaultTopK;

            var space = input.IndexOf(' ');
            var verb = (space < 0 ? input : input[..space]).ToLowerInvariant();

            if (!SearchVerbs.Contains(verb)) return false;

            if (space < 0) return true; // verb only — RunSearch explains what is missing

            var words = input[(space + 1)..].Split(' ', StringSplitOptions.RemoveEmptyEntries);

            if (words.Length >= 2 && int.TryParse(words[^1], out var count) && count > 0)
            {
                topK = count;
                words = words[..^1];
            }

            query = string.Join(' ', words);
            return true;
        }

        private async Task RunSearchAsync(string query, int topK, CancellationToken ct)
        {
            if (query.Length == 0)
            {
                Console.WriteLine("Search needs something to look for, e.g. 'search how are penalties calculated'.");
                return;
            }

            if (_store.Count == 0)
            {
                Console.WriteLine("No chunks yet — drop a PDF into the watched folder, then search.");
                return;
            }

            Console.WriteLine();
            Console.WriteLine($"  embedding \"{ConsoleText.Truncate(query, 60)}\" and ranking {_store.Count} chunk(s)...");

            SemanticSearchResult result;
            try
            {
                result = await _search.SearchAsync(query, topK, ct: ct);
            }
            catch (OperationCanceledException)
            {
                return;
            }
            catch (HttpRequestException ex)
            {
                Console.WriteLine($"  [fail] embedding the query failed: {ex.Message}");
                Console.WriteLine("         Is the embedding server running and the model pulled?");
                Console.WriteLine();
                return;
            }
            catch (InvalidOperationException ex)
            {
                Console.WriteLine($"  [fail] {ex.Message}");
                Console.WriteLine();
                return;
            }
            catch (Exception ex)
            {
                // Ingestion must outlive a bad query, so a failed search reports
                // and returns instead of tearing down the console loop.
                Console.WriteLine($"  [fail] search failed: {ex.Message}");
                Console.WriteLine();
                return;
            }

            SearchResultPrinter.Print(result, topK);
        }

        /// <summary>
        /// Accepts "7", "1-5", "3,9,11-13" and combinations. Ranges are clamped
        /// into 1..total. Numbers that name a chunk which doesn't exist come back
        /// in `missing` so we can say so instead of silently ignoring them.
        /// </summary>
        private bool TryParseIndices(string input, int total, out List<int> indices, out List<int> missing)
        {
            var picked = new List<int>();
            var absent = new List<int>();
            var trimmed = false;
            var sawToken = false;
            var seen = new HashSet<int>();

            void Add(int index)
            {
                if (index < 1 || index > total) return;
                if (seen.Add(index)) picked.Add(index);
            }

            void ReportMissing(int index)
            {
                if (!absent.Contains(index)) absent.Add(index);
            }

            foreach (var token in input.Split(new[] { ',', ' ', ';' }, StringSplitOptions.RemoveEmptyEntries))
            {
                if (int.TryParse(token, out var single))
                {
                    sawToken = true;
                    if (single < 1 || single > total) ReportMissing(single);
                    else Add(single);
                    continue;
                }

                var dash = token.IndexOf('-', 1);
                if (dash > 0 &&
                    int.TryParse(token.AsSpan(0, dash), out var from) &&
                    int.TryParse(token.AsSpan(dash + 1), out var to))
                {
                    sawToken = true;
                    if (from > to) (from, to) = (to, from);
                    if (from < 1) from = 1;

                    if (to > total)
                    {
                        trimmed = true;
                        to = total;
                    }

                    if (from > total) ReportMissing(from);
                    for (var i = from; i <= to; i++) Add(i);
                    continue;
                }

                indices = new List<int>();
                missing = new List<int>();
                return false;
            }

            indices = picked;
            missing = absent;
            Clamped = trimmed;
            return sawToken;
        }

        // Set by TryParseIndices; true when a range ran past the end of the store.
        private bool Clamped { get; set; }

        private void PrintChunk(int index, int total)
        {
            var all = _store.All;
            if (index < 1 || index > all.Count) return;

            var record = all[index - 1];

            Console.WriteLine();
            Console.WriteLine($"--- chunk {index}/{total}  |  {record.Source}  |  {record.Vector.Length} dims ---");
            Console.WriteLine(record.Text);
            Console.WriteLine();
        }

        private void PrintList()
        {
            var all = _store.All;
            if (all.Count == 0)
            {
                Console.WriteLine("No chunks yet — drop a PDF into the watched folder.");
                return;
            }

            Console.WriteLine();
            Console.WriteLine($"{"#",6}  {"source",-30}  text");
            for (var i = 0; i < all.Count; i++)
            {
                Console.WriteLine($"{i + 1,6}  {ConsoleText.Truncate(all[i].Source, 30),-30}  {ConsoleText.Preview(all[i].Text)}");
            }
            Console.WriteLine();
        }

        private void PrintFiles()
        {
            var all = _store.All;
            if (all.Count == 0)
            {
                Console.WriteLine("No files ingested yet.");
                return;
            }

            Console.WriteLine();
            var files = all
                .Select((record, i) => (Number: i + 1, record.Source))
                .GroupBy(x => x.Source, StringComparer.OrdinalIgnoreCase)
                .ToList();

            var width = Math.Max(12, files.Max(f => f.Key.Length));
            width = Math.Min(width, 40);

            for (var f = 0; f < files.Count; f++)
            {
                var group = files[f];
                var first = group.Min(x => x.Number);
                var last = group.Max(x => x.Number);
                var range = first == last ? $"{first}" : $"{first}-{last}";

                Console.WriteLine($"  [{f + 1}] {ConsoleText.Truncate(group.Key, width).PadRight(width)}  {group.Count(),5} chunks   global {range}");
            }
            Console.WriteLine();
        }

        private void PrintStats()
        {
            var all = _store.All;
            if (all.Count == 0)
            {
                Console.WriteLine("No chunks yet.");
                return;
            }

            var fileCount = all.Select(r => r.Source).Distinct(StringComparer.OrdinalIgnoreCase).Count();
            var lengths = all.Select(r => r.Vector.Length).Distinct().ToList();
            var dims = lengths.Count == 1
                ? lengths[0].ToString()
                : $"{lengths.Min()}-{lengths.Max()} (mixed!)";

            var text = all.Sum(r => (long)r.Text.Length);

            Console.WriteLine();
            Console.WriteLine($"  chunks : {all.Count}");
            Console.WriteLine($"  files  : {fileCount}");
            Console.WriteLine($"  dims   : {dims}");
            Console.WriteLine($"  text   : {text:N0} characters");
            Console.WriteLine();
        }

        private static void PrintHelp()
        {
            Console.WriteLine("""
                  <n>          print one chunk, e.g. 7
                  <n-m>        print a range, e.g. 1-5
                  <n,m,n-m>    mixed selection, e.g. 3,9,11-13
                  search <q>   semantic search: ranks chunks by meaning, not keywords
                  search <q> <n>  same, but return the top n hits (default 5)
                  list         every chunk with source and preview
                  files        ingested PDFs with their global chunk ranges
                  stats        chunk count, file count, dimensions
                  help         this message
                  quit         stop watching and exit

                Chunk numbers are global across all PDFs, in the order they were
                ingested. Print output is capped at 25 chunks per command.

                Search embeds your query with the same embedding model that built
                the index, so it finds related wording you did not type. Scores are
                cosine similarities in -1..1 — closer to 1 is a stronger match.
                Results are printed as chunk numbers, so type one to read it in
                full.
                """);
        }
    }

    /// <summary>
    /// Argument parsing for the drop-folder mode. Backwards compatible: bare
    /// arguments are still treated as PDF paths / directories for a one-shot
    /// ingest, and running with no arguments at all watches the default folder.
    /// </summary>
    public class CommandLineOptions
    {
        public string WatchDirectory { get; private set; } = "pdfs";
        public bool Watch { get; private set; }
        public bool RunDemo { get; private set; }
        public int ChunkSize { get; private set; } = 500;
        public int Overlap { get; private set; } = 50;

        /// <summary>Semantic query to answer in one shot; null runs the usual modes.</summary>
        public string? Query { get; private set; }

        /// <summary>How many hits --query returns.</summary>
        public int TopK { get; private set; } = 5;

        /// <summary>Cosine similarity floor for --query results.</summary>
        public float MinScore { get; private set; } = float.NegativeInfinity;

        public List<string> PdfPaths { get; } = new();

        /// <returns>Parsed options, or null if the user asked for help or passed bad input.</returns>
        public static CommandLineOptions? Parse(string[] args)
        {
            var options = new CommandLineOptions();

            for (var i = 0; i < args.Length; i++)
            {
                var arg = args[i];

                switch (arg)
                {
                    case "--watch" or "-w":
                        options.Watch = true;
                        if (i + 1 < args.Length && !args[i + 1].StartsWith("--"))
                        {
                            options.WatchDirectory = args[++i];
                        }
                        break;

                    case "--chunk" or "-c":
                        options.ChunkSize = ReadPositiveInt(args, ref i, arg, options.ChunkSize);
                        break;

                    case "--overlap" or "-o":
                        options.Overlap = ReadPositiveInt(args, ref i, arg, options.Overlap);
                        break;

                    case "--query" or "-q":
                        options.Query = ReadQueryText(args, ref i);
                        break;

                    case "--top" or "-k":
                        options.TopK = ReadPositiveInt(args, ref i, arg, options.TopK);
                        break;

                    case "--min-score" or "-m":
                        options.MinScore = ReadFloat(args, ref i, arg, options.MinScore);
                        break;

                    case "--demo":
                        options.RunDemo = true;
                        break;

                    case "--help" or "-h":
                        PrintUsage();
                        return null;

                    default:
                        if (arg.StartsWith("-"))
                        {
                            Console.WriteLine($"Unknown option: {arg}");
                            PrintUsage();
                            return null;
                        }

                        options.PdfPaths.Add(arg);
                        break;
                }
            }

            return options;
        }

        private static int ReadPositiveInt(string[] args, ref int index, string option, int fallback)
        {
            if (index + 1 >= args.Length ||
                !int.TryParse(args[index + 1], out var value) ||
                value <= 0)
            {
                Console.WriteLine($"{option} needs a positive number.");
                return fallback;
            }

            index++;
            return value;
        }

        private static float ReadFloat(string[] args, ref int index, string option, float fallback)
        {
            if (index + 1 >= args.Length ||
                !float.TryParse(args[index + 1], out var value) ||
                value < -1f || value > 1f)
            {
                Console.WriteLine($"{option} needs a number between -1 and 1.");
                return fallback;
            }

            index++;
            return value;
        }

        /// <summary>
        /// Reads the query text. Only consumes the next single argument so that
        /// positional PDF paths remain discoverable regardless of argument order.
        /// Use quotes for multi-word queries.
        /// </summary>
        private static string? ReadQueryText(string[] args, ref int index)
        {
            if (index + 1 >= args.Length)
            {
                Console.WriteLine("--query needs some text to search for.");
                return null;
            }

            var candidate = args[index + 1];
            if (string.IsNullOrWhiteSpace(candidate) || candidate.StartsWith("-"))
            {
                Console.WriteLine("--query needs some text to search for.");
                return null;
            }

            index++;
            return candidate;
        }

        private static void PrintUsage()
        {
            Console.WriteLine("""
                SimpleCrawler — vectorize PDFs into a searchable knowledge base.

                  dotnet run                              watch the ./pdfs folder
                  dotnet run -- --watch <dir>             watch a specific folder
                  dotnet run -- a.pdf folder/              one-shot ingest of paths
                  dotnet run -- --query <text> a.pdf       ingest, then search once
                  dotnet run -- --demo                    scrape the sample URL

                Options:
                  -w, --watch [dir]   folder to watch for dropped PDFs
                  -c, --chunk <n>     chunk size in characters (default 500)
                  -o, --overlap <n>   chunk overlap in characters (default 50)
                  -q, --query <text>  answer one semantic query and exit
                  -k, --top <n>       hits to return for --query (default 5)
                  -m, --min-score <f> cosine floor for --query, -1..1 (default none)
                  -h, --help          show this message

                Drop PDFs into the watched folder and they are chunked, embedded
                and added to the vector store as they appear. While it runs, type
                a chunk number to print it (numbers are global across all PDFs):
                "1" for one chunk, "1-5" for a range, "list" for every chunk,
                "files" for per-PDF ranges, "quit" to stop. Ctrl+C also stops;
                vectors are in-memory, so a restart rebuilds from the folder.

                Semantic search, either way:
                  interactive   search how are penalties calculated
                  one-shot      dotnet run -- --query penalties --top 3 pdfs/
                The query is embedded with the same model that built the index and
                ranked against every chunk by cosine similarity, so it matches on
                meaning rather than keywords. --query with no paths searches the
                --watch folder (default ./pdfs).
                """);
        }
    }

    public static class Program
    {
        public static async Task Main(string[] args)
        {
            try
            {
                var options = CommandLineOptions.Parse(args);
                if (options == null) return;

                using var embeddingHttpClient = new HttpClient();

                var model = Environment.GetEnvironmentVariable("OLLAMA_EMBEDDING_MODEL") ?? "nomic-embed-text";
                var baseUrl = Environment.GetEnvironmentVariable("OLLAMA_BASE_URL") ?? "http://localhost:11434";

                // Locally hosted embeddings via Ollama — no API key, no rate limits.
                // Ensure the Ollama server is running and the model is pulled:
                //   ollama serve    (usually starts automatically)
                //   ollama pull nomic-embed-text
                IEmbeddingClient embeddingClient = new OllamaEmbeddingClient(embeddingHttpClient, model, baseUrl);

                Console.WriteLine($"Embedding via local Ollama: {model} ({baseUrl})");

                if (options.RunDemo)
                {
                    await Stage1Example.RunAsync(embeddingClient);
                    return;
                }

                // A query is its own one-shot mode: it ingests first, answers, and
                // exits. Checked before the plain ingest path so --query always wins.
                if (options.Query != null)
                {
                    var queryPaths = options.PdfPaths.Count > 0
                        ? options.PdfPaths
                        : new List<string> { options.WatchDirectory };

                    try
                    {
                        await Stage1Example.RunQueryAsync(
                            embeddingClient, options.Query, queryPaths,
                            options.ChunkSize, options.Overlap, options.TopK, options.MinScore);
                    }
                    catch (HttpRequestException ex)
                    {
                        Console.WriteLine($"Embedding request failed: {ex.Message}");
                    }

                    return;
                }

                if (options.PdfPaths.Count > 0 && !options.Watch)
                {
                    await Stage1Example.RunFromPdfsAsync(
                        embeddingClient, options.PdfPaths, options.ChunkSize, options.Overlap);
                    return;
                }

                using var cts = new CancellationTokenSource();

                // Register for SIGINT/SIGTERM rather than relying on CancelKeyPress:
                // that only fires for a real terminal Ctrl+C, so a supervisor or
                // `kill -INT` would otherwise hard-kill the process mid-embed.
                using var sigint = PosixSignalRegistration.Create(PosixSignal.SIGINT, ctx =>
                {
                    ctx.Cancel = true;
                    cts.Cancel();
                });

                using var sigterm = TryRegisterTerminate(cts);

                var vectorStore = new InMemoryVectorStore();
                var watcher = new PdfIngestWatcher(
                    embeddingClient, vectorStore, options.ChunkSize, options.Overlap);

                var folder = Path.GetFullPath(options.WatchDirectory);
                Console.WriteLine();
                Console.WriteLine($"Watching for PDFs in: {folder}");
                Console.WriteLine("Drop .pdf files in at any time, type 'search <question>', or type a chunk number. Ctrl+C to stop.");
                Console.WriteLine();

                var quit = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

                var watcherTask = watcher.RunAsync(options.WatchDirectory, cts.Token);
                var consoleTask = new ChunkConsole(vectorStore, embeddingClient, () => quit.TrySetResult())
                    .RunAsync(cts.Token);

                // Either a signal shuts the watcher down, or the user types "quit".
                // The console ending on its own (stdin closed) must not stop
                // ingestion, so it is deliberately not awaited here.
                await Task.WhenAny(watcherTask, quit.Task);

                cts.Cancel();

                try
                {
                    await Task.WhenAll(watcherTask, consoleTask);
                }
                catch (OperationCanceledException)
                {
                }

                Console.WriteLine($"Stopped. {vectorStore.Count} chunks were vectorized this run.");
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine();
                Console.WriteLine("Stopped.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex}");
            }
        }

        /// <summary>
        /// SIGTERM only exists on Unix, so registering for it throws on Windows.
        /// </summary>
        private static PosixSignalRegistration? TryRegisterTerminate(CancellationTokenSource cts)
        {
            try
            {
                return PosixSignalRegistration.Create(PosixSignal.SIGTERM, ctx =>
                {
                    ctx.Cancel = true;
                    cts.Cancel();
                });
            }
            catch (PlatformNotSupportedException)
            {
                return null;
            }
        }
    }
}
