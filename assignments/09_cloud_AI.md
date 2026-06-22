# Week 9 Assignments

This week you connected Python to a cloud database, learned to read and write rows with `supabase-py`, and built an Extract + Load pipeline that pulls weather data from the Open-Meteo API and stores it in Supabase. The warmup checks your understanding of those concepts and gives you practice with the SDK. The project has you build the pipeline end-to-end.

---

# Submission Instructions

In your `python200-homework` repository, create a folder called `assignments_09/`. Inside that folder, create:

1. `warmup_09.py` — warmup exercises (conceptual answers as comments, code as runnable Python)
2. `project_09.py` — the Extract + Load pipeline
3. `.env` — your Supabase credentials (**add `.env` to your `.gitignore` before committing**)

When finished, commit and open a PR as described in the [assignments README](README.md).

**Setup reminder:** Make sure your `.env` file is present and your Supabase tables exist before running any scripts.

```bash
uv pip install supabase python-dotenv requests
```

---

# Part 1: Warmup

Put all warmup answers in `warmup_09.py`. Use comments to label each section and question (e.g., `# --- Supabase Connection ---` and `# Q1`). For conceptual questions, write your answer as a comment block. For code questions, write working Python that connects to your actual Supabase project.

## Supabase Connection

### Connection Question 1

In a comment block, answer: what are the two pieces of information `supabase-py` needs to connect to your project? Where do you find them in the Supabase dashboard, and why should they never be hardcoded in a Python script?

### Connection Question 2

Write a function `get_client()` that:
1. Loads your credentials from environment variables using `python-dotenv`
2. Creates and returns a Supabase client

The function should raise a clear error if either environment variable is missing.

### Connection Question 3

In a comment block, answer: what is Row Level Security (RLS), and why did you disable it on your tables for this course? In what kind of real-world application would you want to keep it enabled?

## supabase-py CRUD

### CRUD Question 1

Write a function `insert_test_record(supabase)` that inserts a single row into `weather_raw` with today's date and plausible values for all four weather columns. Run it to confirm it works, then add a comment: what would happen if you ran the function twice? How would you change the call to make it safe to run multiple times?

### CRUD Question 2

Write a function `get_records_by_date_range(supabase, start, end)` that returns all rows from `weather_raw` where `date >= start` and `date <= end`. The function should return the list of row dictionaries. Test it with a date range that includes the row you inserted in Q1 and print the result.

### CRUD Question 3

In a comment block, explain the difference between `insert` and `upsert` in `supabase-py`. Give a concrete example of when you would choose each. Then write a function `safe_upsert(supabase, records)` that upserts a list of records into `weather_raw` using `date` as the conflict key and prints the number of rows affected.

## Idempotency

### Idempotency Question 1

"Idempotency" means that running an operation multiple times produces the same result as running it once. In a comment block, explain why idempotency matters for a data pipeline. Give one concrete example of what goes wrong in a non-idempotent pipeline when the script crashes halfway through and is restarted.

---

# Part 2: Project — Extract + Load Pipeline

Build `project_09.py`, a script that implements a complete Extract + Load pipeline: it fetches 2023 daily weather data from the Open-Meteo API for a city of your choice and loads it into your Supabase `weather_raw` table.

This is the same data your Week 4 classifier was trained on. In later weeks, you will use these rows as the input to the transform step — so make sure your column values match what the model expects.

## Step 1: Extract

Call the Open-Meteo historical archive API to retrieve daily weather data for your chosen city for the full year 2023 (start: `2023-01-01`, end: `2023-12-31`). Use these four daily variables:

- `temperature_2m_max`
- `temperature_2m_min`
- `precipitation_sum`
- `wind_speed_10m_max`

Use `response.raise_for_status()` to catch errors early. Print a summary of the response once it arrives.

## Step 2: Transform

Convert the API response from columnar format into a list of row dictionaries. Each dictionary should have keys that exactly match the column names in `weather_raw`.

Print the first and last record to confirm the transformation looks correct. Add a comment: how many records do you expect for a full year, and how many did you get? If the numbers differ, what might explain the discrepancy?

## Step 3: Load

Upsert all records into `weather_raw`. Print a confirmation message showing how many rows were upserted.

Run the script a second time and confirm the row count in `weather_raw` does not change. Add a comment: what does this tell you about idempotency?

## Step 4: Verify

After upserting, run a verification query that:
1. Prints the total number of rows in `weather_raw`
2. Prints the earliest and latest dates in the table
3. Prints the row for `2023-07-04` (or the nearest date if that date is missing)

You can also verify directly in the Supabase Table Editor — take a screenshot for the video below.

## Video

Record a short video (target: 3 minutes, max: 5). Show:

1. The script running in your terminal with no errors
2. The `weather_raw` table in your Supabase dashboard with rows visible
3. Your verification output printed to the terminal

Paste the video link in a comment at the top of `project_09.py`.