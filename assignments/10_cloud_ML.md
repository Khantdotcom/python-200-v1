# Week 10 Assignments

This week you built the Transform step of the pipeline: the ML classifier from Week 4 runs on each row in `weather_raw`, then an LLM adds a one-sentence recommendation, and the combined output goes into `weather_enriched`. The warmup checks your understanding of where ML and LLMs each belong and how they complement each other. The project has you build the complete double-transform end-to-end.

---

# Submission Instructions

In your `python200-homework` repository, create a folder called `assignments_10/`. Inside that folder, create:

1. `warmup_10.py` — warmup exercises (conceptual answers as comments, code as runnable Python)
2. `transform_10.py` — the complete double-transform pipeline
3. `models/` — copy your `weather_classifier.pkl` and `weather_classifier_metadata.json` from Week 4 here

When finished, commit and open a PR as described in the [assignments README](README.md).

**Prerequisites:** Your Week 9 project must have run successfully and populated `weather_raw` in your Supabase project before you run this week's transform script.

```bash
uv pip install supabase python-dotenv joblib scikit-learn pandas openai
```

---

# Part 1: Warmup

Put all warmup answers in `warmup_10.py`. Label each section and question with comments.

## ML vs. LLM in Pipelines

### ML/LLM Question 1

In a comment block, explain the difference between what the ML classifier produces and what the LLM produces in this week's pipeline. Why does each tool do what it does? What would go wrong if you tried to swap them — using the LLM to make the binary good/skip prediction and the ML model to write the recommendation?

### ML/LLM Question 2

For each task below, write one sentence in a comment block stating whether you would use a trained ML model, an LLM, or deterministic code, and why:

- Converting a date string like `"2023-07-04"` to day-of-week
- Classifying a job posting as "entry-level", "mid-level", or "senior" based on freeform text
- Predicting customer churn given 15 numeric features and a labeled training dataset
- Normalizing inconsistent city names ("NYC", "New York City", "New York, NY") to a canonical form
- Summing a column of revenue figures

### ML/LLM Question 3

In a comment block, answer: what is incremental processing, and why is it important for this pipeline? What would happen — in terms of cost and data correctness — if the transform script re-processed all 365 records every time it ran?

## Prompt Design

### Prompt Question 1

The lesson prompt asks the LLM for exactly one sentence. Write an alternative system prompt that asks for a two-sentence recommendation where the first sentence states the prediction and the second sentence explains the reasoning. In a comment, describe: what would you need to change in the validation logic to accommodate two sentences instead of one?

### Prompt Question 2

Write a function `call_with_retry(client, messages, max_retries=3)` that calls `client.chat.completions.create()` and retries up to `max_retries` times on any exception, with a 2-second wait between attempts. On final failure, return `None`. In a comment, describe when you would use this in a production pipeline.

---

# Part 2: Project — The Double-Transform Pipeline

Build `transform_10.py`, a script that reads from `weather_raw`, runs the ML classifier and LLM enrichment on each unprocessed record, and writes the results to `weather_enriched`.

## Step 1: Incremental Read

Load the model metadata from your `models/weather_classifier_metadata.json` file. Fetch all rows from `weather_raw`. Fetch all dates already present in `weather_enriched`. Determine which records still need processing.

Print a summary: how many raw records exist, how many are already enriched, and how many will be processed this run.

## Step 2: ML Transform

Load `models/weather_classifier.pkl`. Build a DataFrame from the unprocessed records, selecting feature columns in the order specified by the metadata. Run `predict` and `predict_proba`. Build a list of enrichment records with `date`, `good_for_running`, and `confidence`.

Print a summary of the predictions: how many days were classified as good, and what is the confidence range?

## Step 3: LLM Transform

Design a system prompt and user message function that passes each day's weather features and ML prediction to `gpt-4o-mini`. For each enrichment record, call the API and add an `llm_summary` field with the one-sentence recommendation.

Handle API errors gracefully: use a fallback string rather than crashing. Add a progress print every 50 records.

## Step 4: Load

Upsert all enrichment records into `weather_enriched`. Print the number of rows upserted.

## Step 5: Verify

Query `weather_enriched` and print:
- The total number of rows
- Five sample rows showing `date`, `good_for_running`, `confidence`, and `llm_summary`
- The number of days classified as good for running

Add a comment: look at a few of the LLM summaries. Do they accurately reflect the weather features and the model's prediction? Pick one you think is particularly good and one that seems off — what might have caused the weaker one?

## Step 6: Reflect

Add a comment block (at least 5–6 sentences) addressing:

1. The ML classifier was trained on Charlotte, NC data. If you loaded weather data for a different city in Week 9, do you expect the classifier's predictions to be accurate? Why or why not?
2. The LLM recommendations are generated from the model's prediction and the weather features. Does the LLM have any ability to "override" the classifier, or is it purely additive? What are the implications of that?
3. If you ran this pipeline on 50,000 records instead of 365, what would be your main concern: cost, latency, or something else? How would you address it?

## Video

Record a short video (target: 3–4 minutes, max: 5). Show:

1. The script running in your terminal with no errors and the progress output
2. The `weather_enriched` table in your Supabase dashboard with rows visible
3. A few printed sample rows showing the LLM recommendations

Paste the video link in a comment at the top of `transform_10.py`.