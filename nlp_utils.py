!pip install transformers datasets tokenizers sentence-transformers --quiet
"""
nlp_utils.py
============
Reusable NLP pipeline utility module built on HuggingFace Transformers.

This module provides a clean, cached interface to five core NLP tasks:
  1. Sentiment Analysis   – classify text as POSITIVE or NEGATIVE
  2. Named Entity Recognition (NER) – extract people, places, organisations
  3. Question Answering   – find answer spans inside a context passage
  4. Summarization        – compress long text into a short summary
  5. Translation          – translate English to Urdu (and other languages)
  6. Zero-Shot Classification – classify text into custom labels without retraining

All pipeline results can be logged to a local SQLite database using log_result().
The module caches loaded pipelines so each model is downloaded and loaded only once
per Python session, saving memory and startup time.

Author : [Umber Qasim]
Course : AI Course
Model  : HuggingFace Transformers pipeline() API
DB     : SQLite  →  nlp_logs.db
"""

from transformers import pipeline
import sqlite3
import json
import datetime

# ── Configuration ─────────────────────────────────────────────────────────────

DB_PATH = 'nlp_logs.db'        # SQLite database file path
PIPELINES = {}                  # Global cache: stores loaded pipelines by key


# ── Database Setup ────────────────────────────────────────────────────────────

def setup_database(db_path: str = DB_PATH) -> None:
    """
    Create the pipeline_results table in SQLite if it does not already exist.

    Args:
        db_path (str): Path to the SQLite database file. Defaults to DB_PATH.

    Returns:
        None
    """
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS pipeline_results (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            task          TEXT    NOT NULL,
            model_name    TEXT    NOT NULL,
            input_text    TEXT    NOT NULL,
            output_json   TEXT    NOT NULL,
            run_date      TEXT    NOT NULL,
            run_time_ms   REAL
        )
    ''')
    conn.commit()
    conn.close()
    print(f'[DB] Database ready: {db_path}')


# ── Pipeline Cache ────────────────────────────────────────────────────────────

def get_pipeline(task: str, model: str, device: int = -1):
    """
    Return a cached HuggingFace pipeline. Loads the model only on first call.

    Args:
        task   (str): HuggingFace task name e.g. 'text-classification', 'ner'.
        model  (str): Model checkpoint name from HuggingFace Hub.
        device (int): 0 = first GPU, -1 = CPU. Defaults to -1 (CPU).

    Returns:
        transformers.Pipeline: The loaded and cached pipeline object.
    """
    key = f'{task}|{model}'
    if key not in PIPELINES:
        print(f'[LOAD] Loading model: {model} for task: {task}')
        PIPELINES[key] = pipeline(task, model=model, device=device)
    return PIPELINES[key]


# ── Task 1: Sentiment Analysis ────────────────────────────────────────────────

def classify_sentiment(texts, device: int = -1):
    """
    Classify text as POSITIVE or NEGATIVE using DistilBERT fine-tuned on SST-2.

    Args:
        texts  (str or list[str]): A single review string or a list of strings.
        device (int): 0 = GPU, -1 = CPU. Defaults to -1.

    Returns:
        list[dict]: Each dict has 'label' (str) and 'score' (float).
                    Example: [{'label': 'POSITIVE', 'score': 0.9998}]
    """
    pipe = get_pipeline(
        'text-classification',
        'distilbert-base-uncased-finetuned-sst-2-english',
        device
    )
    if isinstance(texts, str):
        texts = [texts]
    return pipe(texts, batch_size=32, truncation=True, max_length=512)


# ── Task 2: Named Entity Recognition ─────────────────────────────────────────

def extract_entities(text: str, device: int = -1):
    """
    Extract named entities (persons, organisations, locations) from text.

    Uses bert-base-NER fine-tuned on CoNLL-2003. The aggregation_strategy='simple'
    setting merges multi-token entities like 'New York' into one result.

    Args:
        text   (str): The input text to extract entities from.
        device (int): 0 = GPU, -1 = CPU. Defaults to -1.

    Returns:
        list[dict]: Each dict has 'word', 'entity_group', 'score', 'start', 'end'.
                    entity_group values: PER (person), ORG, LOC, MISC.
    """
    pipe = get_pipeline('ner', 'dslim/bert-base-NER', device)
    return pipe(text, aggregation_strategy='simple')


# ── Task 3: Question Answering ────────────────────────────────────────────────

def answer_question(question: str, context: str, device: int = -1):
    """
    Extract the answer to a question from a given context passage (Extractive QA).

    Never fabricates information — the answer must exist in the context text.
    Uses DistilBERT fine-tuned on SQuAD v1.1.

    Args:
        question (str): The question to answer.
        context  (str): The passage of text that contains the answer.
        device   (int): 0 = GPU, -1 = CPU. Defaults to -1.

    Returns:
        dict: Contains 'answer' (str), 'score' (float), 'start' (int), 'end' (int).
    """
    pipe = get_pipeline(
        'question-answering',
        'distilbert-base-cased-distilled-squad',
        device
    )
    return pipe(question=question, context=context)


# ── Task 4: Summarization ─────────────────────────────────────────────────────

def summarise(text: str, max_length: int = 130, min_length: int = 50, device: int = -1):
    """
    Generate an abstractive summary of a long text using BART (CNN/DailyMail).

    Abstractive means the model writes new sentences, not just copies from the source.
    Good for news articles, research papers, long emails, and reports.

    Args:
        text       (str): The long text to summarise.
        max_length (int): Maximum number of tokens in the output. Default 130.
        min_length (int): Minimum number of tokens in the output. Default 50.
        device     (int): 0 = GPU, -1 = CPU. Defaults to -1.

    Returns:
        str: The generated summary text.
    """
    pipe = get_pipeline('summarization', 'facebook/bart-large-cnn', device)
    result = pipe(text, max_length=max_length, min_length=min_length, do_sample=False)
    return result[0]['summary_text']


# ── Task 5: Translation ───────────────────────────────────────────────────────

def translate_en_ur(text: str, device: int = -1):
    """
    Translate English text to Urdu using Helsinki-NLP opus-mt model.

    Helsinki-NLP provides 1,300+ translation models on HuggingFace.
    Urdu is the national language of Pakistan (230 million speakers).

    Args:
        text   (str): English text to translate.
        device (int): 0 = GPU, -1 = CPU. Defaults to -1.

    Returns:
        str: The translated Urdu text.
    """
    pipe = get_pipeline('translation', 'Helsinki-NLP/opus-mt-en-ur', device)
    return pipe(text)[0]['translation_text']


# ── Task 6: Zero-Shot Classification (New Function) ───────────────────────────

def zero_shot_classify(text: str, candidate_labels: list, device: int = -1):
    """
    Classify text into any custom categories WITHOUT any model retraining.

    This is extremely powerful for rapid prototyping: you define the labels yourself
    at runtime. Uses Facebook's BART model fine-tuned on MNLI (natural language inference).

    Real-world use case: categorise customer complaints into departments
    (billing, shipping, product quality) without labelled training data.

    Args:
        text             (str):       The text to classify.
        candidate_labels (list[str]): Your custom category labels.
                                      Example: ['positive', 'negative', 'neutral']
        device           (int):       0 = GPU, -1 = CPU. Defaults to -1.

    Returns:
        dict: Contains 'labels' (list sorted by score), 'scores' (list of floats),
              and 'sequence' (the original input text).
              The first label in 'labels' is the top prediction.

    Example:
        >>> result = zero_shot_classify(
        ...     "My order arrived two weeks late and was damaged.",
        ...     ["shipping complaint", "product defect", "billing issue", "praise"]
        ... )
        >>> print(result['labels'][0])   # → 'shipping complaint'
        >>> print(result['scores'][0])   # → 0.87 (confidence)
    """
    pipe = get_pipeline(
        'zero-shot-classification',
        'facebook/bart-large-mnli',
        device
    )
    return pipe(text, candidate_labels=candidate_labels)


# ── SQL Logging ───────────────────────────────────────────────────────────────

def log_result(task: str, model_name: str, input_text: str, output) -> None:
    """
    Log a single pipeline result to the SQLite database.

    Serialises the output to JSON automatically. Truncates input_text to 500
    characters to stay within SQLite practical limits.

    Args:
        task       (str):       The NLP task name e.g. 'sentiment-analysis'.
        model_name (str):       The HuggingFace model checkpoint name.
        input_text (str):       The original text sent to the pipeline.
        output     (any):       The pipeline's output (dict, list, or string).
                                Will be JSON-serialised automatically.

    Returns:
        None
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO pipeline_results VALUES (NULL,?,?,?,?,?,?)',
        (
            task,
            model_name,
            str(input_text)[:500],
            json.dumps(output, ensure_ascii=False, default=str),
            datetime.date.today().isoformat(),
            None
        )
    )
    conn.commit()
    conn.close()


# ── Main Demo ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    """
    Demonstration of every function in nlp_utils.py.
    Run this file directly:  python nlp_utils.py
    All results are printed to console AND logged to nlp_logs.db.
    """

    print('=' * 65)
    print('  nlp_utils.py  —  Full Demo')
    print('=' * 65)

    # ── Setup DB first ────────────────────────────────────────
    setup_database()

    # ── 1. Sentiment Analysis ─────────────────────────────────
    print('\n--- TASK 1: Sentiment Analysis ---')
    sample_reviews = [
        'The product quality is outstanding, very happy with my purchase!',
        'Terrible customer service, waited 3 weeks and got the wrong item.',
        'It is okay, nothing special but gets the job done.',
    ]
    sentiments = classify_sentiment(sample_reviews)
    for text, result in zip(sample_reviews, sentiments):
        print(f'  [{result["label"]} {result["score"]:.3f}] {text[:55]}')
        log_result('sentiment-analysis',
                   'distilbert-base-uncased-finetuned-sst-2-english',
                   text, result)

    # ── 2. Named Entity Recognition ───────────────────────────
    print('\n--- TASK 2: Named Entity Recognition ---')
    ner_text = ('Prime Minister Shehbaz Sharif met World Bank President '
                'Ajay Banga in Islamabad to discuss Pakistan economic reform.')
    entities = extract_entities(ner_text)
    for ent in entities:
        print(f'  {ent["word"]:<20} → {ent["entity_group"]} ({ent["score"]:.3f})')
    log_result('ner', 'dslim/bert-base-NER', ner_text, entities)

    # ── 3. Question Answering ─────────────────────────────────
    print('\n--- TASK 3: Question Answering ---')
    context = ('Pakistan IT exports crossed $2.6 billion in fiscal year 2024-25. '
               'The sector employs over 600,000 professionals. '
               'Pakistan ranks among the top five countries for freelance earnings.')
    question = 'How much did Pakistan IT exports earn in 2024-25?'
    qa_result = answer_question(question, context)
    print(f'  Q: {question}')
    print(f'  A: {qa_result["answer"]}  (confidence: {qa_result["score"]:.4f})')
    log_result('question-answering', 'distilbert-base-cased-distilled-squad',
               question, qa_result)

    # ── 4. Summarization ──────────────────────────────────────
    print('\n--- TASK 4: Summarization ---')
    long_article = (
        'Pakistan technology sector has seen unprecedented growth. '
        'IT exports crossed $2.6 billion in fiscal year 2024-25. '
        'The sector employs over 600,000 professionals growing at 25% annually. '
        'Pakistan ranks among the top five countries globally for freelance earnings. '
        'The government launched tax exemptions for IT companies and software parks '
        'in Islamabad, Lahore, and Karachi. Analysts project exports could reach '
        '$5 billion by 2027 if infrastructure issues are resolved.'
    )
    summary = summarise(long_article, max_length=60, min_length=25)
    print(f'  Original : {len(long_article.split())} words')
    print(f'  Summary  : {summary}')
    log_result('summarization', 'facebook/bart-large-cnn', long_article[:200],
               {'summary_text': summary})

    # ── 5. Translation ────────────────────────────────────────
    print('\n--- TASK 5: Translation (English → Urdu) ---')
    en_text = 'Artificial intelligence is transforming every industry in Pakistan.'
    urdu = translate_en_ur(en_text)
    print(f'  EN : {en_text}')
    print(f'  UR : {urdu}')
    log_result('translation', 'Helsinki-NLP/opus-mt-en-ur', en_text,
               {'language': 'Urdu', 'translation': urdu})

    # ── 6. Zero-Shot Classification ───────────────────────────
    print('\n--- TASK 6: Zero-Shot Classification ---')
    complaint = 'My parcel arrived 15 days late and the packaging was completely torn.'
    labels = ['shipping complaint', 'product defect', 'billing issue', 'positive feedback']
    zs_result = zero_shot_classify(complaint, labels)
    print(f'  Input : {complaint}')
    for label, score in zip(zs_result['labels'], zs_result['scores']):
        print(f'  {label:<25} → {score:.4f}')
    log_result('zero-shot-classification', 'facebook/bart-large-mnli',
               complaint, zs_result)

    print('\n' + '=' * 65)
    print('  All results logged to nlp_logs.db')
    print('  Run your queries_day3.sql to analyse the results.')
    print('=' * 65)
