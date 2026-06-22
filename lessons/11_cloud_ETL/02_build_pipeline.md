# Building the Pipeline

You have already written all the pieces of this pipeline. In Week 9 you extracted weather data from Open-Meteo and loaded it into `weather_raw`. In Week 10 you read those rows back, classified them with the ML model from Week 4, enriched each prediction with an LLM recommendation, and wrote the results to `weather_enriched`. This lesson wires all of it into a single Prefect flow.

At this point, the main goal is not learning brand-new concepts — it is learning how the pieces connect into a real workflow. This is much closer to how production data pipelines are structured in practice.

For reference:

- [Prefect tasks documentation](https://docs.prefect.io/v3/concepts/tasks)
- [Week 9: The Extract + Load Pipeline](../09_cloud_data/03_loading_pipeline.md)
- [Week 10: ML Inference on Database Records](../10_llm_pipelines/02_ml_inference.md)
- [Week 10: LLM Enrichment and Writing to weather_enriched](../10_llm_pipelines/03_llm_enrichment.md)

## Learning Objectives

By the end of this lesson, you will be able to:

- Implement extract, load_raw, transform, and load_enriched as Prefect tasks using real cloud operations
- Wire the four tasks into a `@flow` and run the complete pipeline end-to-end
- Read a Prefect UI run to confirm success or identify a failure

## Setup

This pipeline requires packages from Weeks 9 and 10:

```bash
uv pip install prefect requests openai python-dotenv supabase joblib scikit-learn pandas
```

Your `models/weather_classifier.pkl` and `models/weather_classifier_metadata.json` files from Week 4 must be present in the directory where you run the script.

Define your shared constants and clients at the top of the script. The Supabase and OpenAI clients are initialized once and passed into tasks via closure — they do not need to be recreated on every task call:

```python
import os
import json
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)
FEATURES = metadata["features"]

SYSTEM_PROMPT = (
    "You are writing a one-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write exactly one sentence — direct, practical, and specific to the conditions. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)
```

## The Extract Task

The extract task calls the Open-Meteo historical archive API and returns a list of row dictionaries, one per day. Adding `retries=2` means Prefect will automatically retry up to twice if the request fails — useful since API calls can fail transiently:

```python
import requests
from prefect import task

LATITUDE  = 35.23
LONGITUDE = -80.84

@task(retries=2, retry_delay_seconds=10)
def extract() -> list:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   LATITUDE,
        "longitude":  LONGITUDE,
        "start_date": "2023-01-01",
        "end_date":   "2023-12-31",
        "daily": FEATURES,
        "timezone": "America/New_York",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()

    daily = response.json()["daily"]
    records = [
        {
            "date":               daily["time"][i],
            "temperature_2m_max": daily["temperature_2m_max"][i],
            "temperature_2m_min": daily["temperature_2m_min"][i],
            "precipitation_sum":  daily["precipitation_sum"][i],
            "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
        }
        for i in range(len(daily["time"]))
    ]

    print(f"Extracted {len(records)} daily records from Open-Meteo")
    return records
```

One task, one responsibility: fetch the data and return it. The task does not know where it came from or where it is going — that is the flow's job.

## The Load Raw Task

The load_raw task upserts the raw records into `weather_raw`. Using `upsert` with `on_conflict="date"` makes it idempotent — re-running the pipeline does not duplicate rows:

```python
@task(retries=2, retry_delay_seconds=5)
def load_raw(records: list) -> None:
    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )
    print(f"Upserted {len(response.data)} rows into weather_raw")
```

This task must complete before the transform task runs — the ML classifier's incremental check queries `weather_enriched`, not `weather_raw`, so there is no strict ordering dependency between the two load steps. But logically, raw data should be persisted before transformation begins. The sequential flow structure enforces this naturally.

## The Transform Task

The transform task encapsulates the double-transform from Week 10: ML inference followed by LLM enrichment. Structuring it as a single task keeps the flow clean while still making the two internal stages explicit through comments and print statements:

```python
import joblib
import pandas as pd
from prefect import task

@task
def transform(raw_records: list) -> list:
    # --- Incremental check ---
    already_done = {
        r["date"]
        for r in supabase.table("weather_enriched").select("date").execute().data
    }
    to_process = [r for r in raw_records if r["date"] not in already_done]
    print(f"Records to transform: {len(to_process)} (skipping {len(already_done)} already enriched)")

    if not to_process:
        print("All records already enriched — nothing to do.")
        return []

    # --- ML classify ---
    clf = joblib.load("models/weather_classifier.pkl")
    df  = pd.DataFrame(to_process)
    X   = df[FEATURES]

    predictions   = clf.predict(X)
    probabilities = clf.predict_proba(X)[:, 1]

    print(f"ML classification complete. Good days: {int(predictions.sum())} / {len(predictions)}")

    enrichment_records = [
        {
            "date":             to_process[i]["date"],
            "good_for_running": bool(predictions[i]),
            "confidence":       round(float(probabilities[i]), 4),
            "llm_summary":      None,
        }
        for i in range(len(to_process))
    ]

    # --- LLM enrich ---
    for i, record in enumerate(enrichment_records):
        raw_row = to_process[i]
        prediction_text = "good for running" if record["good_for_running"] else "not ideal for running"
        user_message = (
            f"Date: {raw_row['date']}\n"
            f"High: {raw_row['temperature_2m_max']}°C, Low: {raw_row['temperature_2m_min']}°C\n"
            f"Precipitation: {raw_row['precipitation_sum']} mm\n"
            f"Max wind speed: {raw_row['wind_speed_10m_max']} km/h\n"
            f"Model prediction: {prediction_text} (confidence: {record['confidence']:.0%})"
        )
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=100,
            )
            record["llm_summary"] = response.choices[0].message.content.strip() or "Recommendation unavailable."
        except Exception as e:
            print(f"  LLM error on {record['date']}: {e}")
            record["llm_summary"] = "Recommendation unavailable."

        if (i + 1) % 50 == 0:
            print(f"  LLM enriched {i + 1} / {len(enrichment_records)} records")

    print(f"Transform complete: {len(enrichment_records)} records enriched")
    return enrichment_records
```

Several production patterns are present here:

- The incremental check means re-running the flow on data already transformed does no extra work — and no extra LLM cost.
- `clf.predict` and `clf.predict_proba` are called once on the full DataFrame, not in a loop. Batch inference is significantly faster than record-by-record inference.
- LLM failures are caught and replaced with a fallback string rather than crashing the task.
- Progress logs every 50 records make long-running transforms easy to follow in the Prefect UI.

## The Load Enriched Task

The final task upserts the enrichment records into `weather_enriched`:

```python
@task(retries=2, retry_delay_seconds=5)
def load_enriched(enrichment_records: list) -> None:
    if not enrichment_records:
        print("No new enrichment records to load.")
        return

    response = (
        supabase.table("weather_enriched")
        .upsert(enrichment_records, on_conflict="date")
        .execute()
    )
    print(f"Upserted {len(response.data)} rows into weather_enriched")
```

The guard on `not enrichment_records` handles the case where the transform task found nothing to process — the task exits cleanly rather than sending an empty upsert to the database.

## Wiring the Flow

```python
from prefect import flow

@flow(log_prints=True)
def etl_pipeline():
    raw_records        = extract()
    load_raw(raw_records)
    enrichment_records = transform(raw_records)
    load_enriched(enrichment_records)
    print("Pipeline complete.")

if __name__ == "__main__":
    etl_pipeline()
```

The flow is five lines. Each task call is one line. The data dependencies — what each task returns and what the next task receives — are visible at a glance. This is the characteristic shape of a well-structured Prefect flow: the orchestration logic is minimal; the work happens inside the tasks.

## Running the Pipeline

Start the Prefect UI in one terminal:

```bash
prefect server start
```

Run the pipeline in a second terminal:

```bash
python etl_pipeline.py
```

On the first run, you should see:
1. The extract task fetch 365 records
2. The load_raw task upsert them into `weather_raw`
3. The transform task classify all 365 and enrich them with LLM recommendations (this takes several minutes — each LLM call is ~1 second)
4. The load_enriched task upsert them into `weather_enriched`

The transform step is the longest by far. Progress prints every 50 records give you visibility while it runs.

## Reading the Prefect UI

Open `http://localhost:4200` and click into the `etl-pipeline` run. A successful run shows all four tasks in *Completed* state. Click any task to see its logs — you should see the print statements from each task captured there.

If a task fails, it shows in *Failed* state (red). Click the failed task and open the *Logs* tab to find the exception traceback. Tasks that completed successfully remain green, so you can see exactly where in the pipeline the failure occurred without re-running the whole thing.

A useful exercise: cause a small failure intentionally and observe how the UI responds. For example, temporarily remove your `SUPABASE_KEY` from `.env`, run the pipeline, and read the failure trace. Then restore the key and re-run — the incremental check means you pick up from where you left off, not from the beginning.