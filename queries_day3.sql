-- ============================================================
-- queries_day3.sql
-- SQL Queries on the NLP Pipeline Log (pipeline_results table)
-- Week 5, Day 3 Assignment
-- ============================================================

-- TABLE SCHEMA REMINDER:
-- pipeline_results(
--   id            INTEGER PRIMARY KEY AUTOINCREMENT,
--   task          TEXT,
--   model_name    TEXT,
--   input_text    TEXT,
--   output_json   TEXT,
--   run_date      TEXT,
--   run_time_ms   REAL
-- )

-- ─────────────────────────────────────────────────────────────
-- QUERY 1 (Required)
-- Count total pipeline calls made today, grouped by task
-- Purpose: See how many times each NLP task was run in today's session
-- ─────────────────────────────────────────────────────────────
SELECT
    task,
    COUNT(*) AS total_calls
FROM pipeline_results
WHERE run_date = DATE('now')
GROUP BY task
ORDER BY total_calls DESC;


-- ─────────────────────────────────────────────────────────────
-- QUERY 2 (Required)
-- Find all sentiment analysis results where output contains 'NEGATIVE'
-- Purpose: Filter only the negative reviews from today's sentiment task
-- Note: output_json is stored as a JSON string, so we use LIKE
-- ─────────────────────────────────────────────────────────────
SELECT
    id,
    input_text,
    output_json
FROM pipeline_results
WHERE task = 'sentiment-analysis'
  AND output_json LIKE '%NEGATIVE%'
ORDER BY id;


-- ─────────────────────────────────────────────────────────────
-- QUERY 3 (Required)
-- List all unique models used, with the task they were used for
-- Purpose: Audit which model was used for which NLP task
-- ─────────────────────────────────────────────────────────────
SELECT DISTINCT
    task,
    model_name
FROM pipeline_results
ORDER BY task, model_name;


-- ─────────────────────────────────────────────────────────────
-- QUERY 4 (Required)
-- Find the longest input text submitted to any pipeline today
-- Purpose: Identify which pipeline call received the most text
-- LENGTH() counts characters in the input_text column
-- ─────────────────────────────────────────────────────────────
SELECT
    task,
    model_name,
    LENGTH(input_text) AS input_length_chars,
    SUBSTR(input_text, 1, 80) AS input_preview
FROM pipeline_results
ORDER BY input_length_chars DESC
LIMIT 5;


-- ─────────────────────────────────────────────────────────────
-- QUERY 5 (Own Design)
-- Confidence score analysis for sentiment analysis results
-- Purpose: Find which reviews the model was LEAST confident about
-- This is useful to flag borderline/ambiguous cases for human review
-- json_extract() pulls the 'score' value out of the JSON output
-- ─────────────────────────────────────────────────────────────
SELECT
    input_text,
    json_extract(output_json, '$.label') AS predicted_label,
    ROUND(CAST(json_extract(output_json, '$.score') AS REAL), 4) AS confidence_score
FROM pipeline_results
WHERE task = 'sentiment-analysis'
ORDER BY confidence_score ASC;  -- Lowest confidence first (most uncertain)