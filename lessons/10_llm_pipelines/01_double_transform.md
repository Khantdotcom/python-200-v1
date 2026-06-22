# The Double-Transform Pattern

In Weeks 5 through 7, you used language models the way most people first encounter them: as a conversational interface. You sent a message, got a response, maybe kept a conversation going. The model was a partner you were talking to.

This week, the framing shifts. An LLM is no longer the end product — it is a processing step inside a larger pipeline. But it is not the only processing step. This week's transform stage has two components: first, the weather classifier you built and saved in Week 4 makes a binary prediction; then an LLM adds a short natural-language recommendation. Both are transforms. They have different strengths and complement each other in ways that neither could achieve alone.

## Learning Objectives

By the end of this lesson, you will be able to:

- Place ML models and LLMs correctly in an ETL pipeline
- Explain what each transform type does well and when to use each
- Describe how a serialized sklearn Pipeline becomes a reusable pipeline component
- Design constrained prompts that return parseable output from an LLM in an automated context

## Where ML Models Fit in ETL

A data pipeline has three stages: Extract, Transform, Load. ML models and LLMs both belong in the Transform step — they receive data, process it, and return something new. They are not well-suited for Extract or Load, which involve I/O operations handled deterministically by the surrounding code.

But "Transform" covers a wide range of operations. The question is which tool to reach for:

| Task | Right tool | Why |
|------|-----------|-----|
| Binary classification on structured features | ML model | Fast, deterministic, cheap, reproducible |
| Natural-language explanation or summary | LLM | Requires language generation and judgment |
| Arithmetic, filtering, aggregation | Deterministic code | LLMs are slow, expensive, and unreliable for math |
| Classifying freeform text | LLM | Requires reading comprehension |
| Predicting from a fixed set of numeric features | ML model | Structured input → structured output |

The weather classifier sits firmly in the ML column: four numeric features, a binary output, trained and evaluated on labeled data. There is no reason to use an LLM for that classification — the ML model is faster, cheaper, and more reproducible.

But the ML model cannot explain its prediction in plain language. It cannot say "good running conditions — mild temperatures and no rain, though the wind will pick up in the afternoon." That requires language generation, which is exactly what LLMs do well. The two tools fill different roles in the same pipeline.

## The Saved Pipeline as a Pipeline Component

In Week 4, you saved the best GridSearchCV result as a `.pkl` file:

```python
joblib.dump(best_pipe, "models/weather_classifier.pkl")
```

That file contains the entire fitted Pipeline — scaler and logistic regression — frozen at the moment of training. Loading it in Week 10 is what makes the week 4 work reusable:

```python
import joblib

clf = joblib.load("models/weather_classifier.pkl")
```

`clf` is immediately ready to call `.predict()` and `.predict_proba()` on new data. No retraining, no re-fitting, no knowledge of how it was built. This is the entire point of model persistence: the training cost is paid once, and the resulting artifact can be used anywhere the `.pkl` file is available.

One constraint worth stating explicitly: the model expects input features in a specific order, with specific column names. The metadata file you saved alongside the model records what those names are:

```python
import json

with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]
# ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'wind_speed_10m_max']
```

When you build a DataFrame from Supabase rows, slicing by `FEATURES` ensures the columns arrive in the right order. The database column names were chosen to match the API field names for exactly this reason.

## What LLMs Do Well in Pipelines

The tasks where LLMs add the most value in a pipeline context tend to involve language that is too variable or ambiguous for rule-based code to handle reliably.

*Natural-language generation* is the clearest fit: taking structured data and producing a human-readable summary, explanation, or recommendation. This is what the LLM step does this week — it receives numeric features and a binary prediction and produces a sentence or two that a user could actually act on.

*Text classification* is also a strong use case: given a customer support ticket, classify it as billing, technical, or general. Given a job posting, classify it as entry-level, mid-level, or senior. These require reading comprehension and judgment that rule-based code handles poorly.

*Field extraction and normalization* covers irregular input: extracting a city name from a freeform address string, normalizing "NYC" / "New York City" / "New York, NY" to a canonical form.

The flip side is equally important. LLMs are the wrong tool when the task has a correct answer code can compute reliably. Arithmetic belongs in code. Date parsing belongs in code. Classifying records with a well-labeled training set and structured numeric features — like the weather classifier — belongs to an ML model.

A useful heuristic: if you can write a unit test with a single expected output, use code. If the "correct" answer requires judgment, language understanding, or generation, consider an LLM.

## Cost and Latency at Scale

When you use an LLM interactively, one API call goes unnoticed. In a pipeline processing thousands of records, those calls add up.

Using `gpt-4o-mini` as a reference (check the [OpenAI pricing page](https://openai.com/api/pricing/) for current rates):

- A short call with a ~200-token input and a 50-token output costs roughly $0.00005.
- For 365 records (one year of daily weather data): under $0.02 in API costs.
- For 100,000 records: roughly $5.00.

Cost is manageable with smaller, efficient models. Latency is the bigger concern. Each API call takes roughly 0.5 to 2 seconds. Calling an LLM sequentially for 10,000 records takes two to six hours of wall-clock time. For most course-scale pipelines, sequential calls are fine. At production scale, engineers introduce batching, concurrency, or OpenAI's Batch API (which processes requests asynchronously at reduced cost).

For 365 records — the size of the dataset in this week's project — sequential calls are entirely reasonable and will complete in a few minutes.

## Designing Prompts for Pipelines

Interactive LLM use rewards rich, open-ended prompts. Pipeline use rewards the opposite: precise, constrained prompts that return output your code can parse without ambiguity.

Consider the difference for a sentiment classification task:

```text
Bad:  "Describe the sentiment of this customer review."
Good: "Classify the sentiment. Reply with exactly one word: positive, negative, or neutral."
```

The first prompt might produce "This review expresses a mildly negative sentiment overall, though with some positive undertones." That is useful to a human reader but difficult to parse in a pipeline. The second constrains the output to a known set of values.

For the weather recommendation step, the goal is a short, human-readable sentence — not a one-word label. Even so, you can constrain it:

```text
"Write one sentence recommending whether to go for a run today,
 given the weather conditions and the model's prediction.
 Be direct and practical. Do not use bullet points or headers."
```

A one-sentence constraint is easy to verify: if the response has more than one sentence, something went wrong. Even with carefully constrained prompts, models occasionally produce output that does not match the expected format — pipeline code should always validate and handle this gracefully.

## Lesson Wrap-Up

This week's transform step uses both tools where each excels: the ML classifier for a fast, reproducible binary decision on structured numeric features; the LLM for a human-readable explanation that the classifier cannot produce. The serialized Pipeline from Week 4 is a first-class pipeline component — load it, call `.predict()`, move on. The LLM receives the result and adds the language layer on top.

In the next two lessons, you will implement both transform steps: ML inference on Supabase records, then LLM enrichment and writing to `weather_enriched`.