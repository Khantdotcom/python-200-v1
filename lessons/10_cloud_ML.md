# Week 10: LLMs in Pipelines

In weeks 5 through 7, you used language models interactively — building chatbots, augmenting them with retrieved knowledge, and wiring them into agents. In week 9, you stored structured weather data in Supabase. This week you connect those two skills, and add a third: the saved ML classifier from week 4.

The transform step you build this week is a **double transform**: first the weather classifier predicts whether each day's conditions are good for running, then an LLM generates a short natural-language recommendation explaining why. Both steps read from `weather_raw`; together they write to `weather_enriched`. The conceptual shift is treating both models — the sklearn Pipeline and the LLM — as data-processing components rather than interactive tools. You will be asked to articulate explicitly why the ML model handles the classification and the LLM handles the natural-language layer, not the other way around.

## Topics

1. [The Double-Transform Pattern](./01_double_transform.md)
Frames the week conceptually. Covers where ML models and LLMs each belong in an ETL pipeline, what kinds of tasks they do well versus poorly, and how passing ML outputs (including confidence scores) into an LLM prompt enables richer, hedged recommendations. Also covers the prompt-design principles that apply when an LLM is a batch processing step rather than a conversational partner.

2. [ML Inference on Database Records](./02_ml_inference.md)
The hands-on ML step. Reads rows from `weather_raw`, loads the saved sklearn Pipeline from week 4, runs `predict` and `predict_proba` on each record, and handles incremental processing — skipping dates already present in `weather_enriched`. Ends with a set of enrichment records ready to pass to the LLM step.

3. [LLM Enrichment and Writing to weather_enriched](./03_llm_enrichment.md)
The hands-on LLM step. Designs a constrained prompt that receives each day's weather features, ML prediction, and confidence score, calls the OpenAI API, validates the response, and writes complete enrichment records to `weather_enriched`. Closes with a spot-check query to confirm the output looks right.

## Week 10 Assignments

Once you finish the lessons, head on over to the [assignments](../../assignments/README.md) to get more hands-on practice with the material.
