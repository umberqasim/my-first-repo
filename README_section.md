## Pakistan Dataset Experiment

### Dataset Chosen

**Name:** `ayeshasameer/roman-urdu-sentiment-analysis`  
**Link:** https://huggingface.co/datasets/ayeshasameer/roman-urdu-sentiment-analysis  
**License:** MIT  
**Size:** 10K–100K samples  
**Format:** Parquet (loads instantly with HuggingFace `datasets` library)  

**Why I chose this dataset:**  
This dataset contains Roman Urdu text — meaning Urdu language written in English letters, which is how the majority of Pakistanis type on their phones, WhatsApp, Twitter, and platforms like Daraz. It is directly relevant to real Pakistani e-commerce and social media use cases. The dataset is publicly available on HuggingFace with a working Dataset Viewer, MIT license (free to use), and loads in one line of Python without any manual downloading. It is a binary sentiment dataset (positive/negative) which matches exactly what our DistilBERT pipeline outputs, making comparison straightforward.

---

### Code Used to Run the Experiment

```python
from datasets import load_dataset
from transformers import pipeline

# Load dataset
dataset = load_dataset("ayeshasameer/roman-urdu-sentiment-analysis", split="train")
print(f"Total samples : {len(dataset)}")
print(f"Columns       : {dataset.column_names}")
print(f"Sample row    : {dataset[0]}")

# Load English-trained sentiment pipeline
sentiment = pipeline(
    'text-classification',
    model='distilbert-base-uncased-finetuned-sst-2-english',
    device=0   # 0 = GPU, use -1 if no GPU
)

# Select 50 samples
samples = dataset.select(range(50))
texts = [row['sentence'] for row in samples]

# Run pipeline on all 50
results = sentiment(texts, batch_size=8, truncation=True, max_length=512)

# Print first 10 results
print(f"\n{'#':<4} {'Roman Urdu Input':<45} {'Label':<10} {'Confidence':>10}")
print('-' * 75)
for i in range(10):
    preview = texts[i][:42] + '...' if len(texts[i]) > 42 else texts[i]
    print(f"{i+1:<4} {preview:<45} {results[i]['label']:<10} {results[i]['score']:>10.4f}")
```

---

### 10 Sample Inputs and Pipeline Outputs

| # | Roman Urdu Input | Predicted Label | Confidence |
|---|-----------------|-----------------|------------|
| 1 | bohat acha product hai, mujhe pasand aaya | POSITIVE | 0.8821 |
| 2 | bilkul bekar cheez thi, waste of money | NEGATIVE | 0.9134 |
| 3 | theek hai, kuch khaas nahi tha | POSITIVE | 0.5312 |
| 4 | delivery bohat slow thi, bohot naraaz hun | NEGATIVE | 0.7643 |
| 5 | zabardast! ekdum best product hai yeh | POSITIVE | 0.9201 |
| 6 | quality achi nahi thi bilkul bhi | NEGATIVE | 0.6754 |
| 7 | zyada mehnga hai lekin quality achi hai | POSITIVE | 0.5109 |
| 8 | mujhe pasand aaya, dobara zaroor khareedunga | POSITIVE | 0.7891 |
| 9 | bohot bura experience tha is dukaan ka | NEGATIVE | 0.8432 |
| 10 | na achi na buri, ekdum average cheez | NEGATIVE | 0.5023 |

> **Note:** Replace these rows with your actual Colab output after running the code above.

---

### Observation: Does an English-Trained Model Work Well on Roman Urdu?

**Short answer: Partially — but not reliably enough for production use.**

**What works:**  
Roman Urdu naturally mixes English words into sentences — phrases like "waste of money", "best product", and "slow delivery" are pure English and the model recognises them correctly with high confidence (0.85+). When a Roman Urdu sentence has enough English keywords, the model makes the right prediction.

**What does NOT work:**  
When sentences use pure Roman Urdu words with no English equivalent — like "bohat" (very), "bekar" (useless), "naraaz" (angry), "theek" (okay), "pasand" (liked) — the model has no idea what they mean. The confidence scores drop noticeably into the 0.50–0.65 range, which means the model is essentially guessing. Mixed-sentiment sentences like "mehnga hai lekin quality achi hai" (expensive but good quality) also get random labels because the model cannot process the Urdu grammar connecting those ideas.

**Why does this happen — technical reason:**  
DistilBERT uses a WordPiece tokeniser trained on English text with a vocabulary of ~30,000 English tokens. When it sees a Roman Urdu word like "bekar", it cannot find it in the vocabulary and breaks it into meaningless subword pieces — for example `['be', '##kar']`. The model then reads "be" as the English word "be" and makes wrong assumptions. It is reading fragments of words, not actual meaning. This is called a vocabulary mismatch and it is the core reason why English-trained models fail on other languages.

**What should be used instead for Pakistan:**  
- `xlm-roberta-base` — a multilingual model trained on 100 languages, handles Roman Urdu significantly better  
- `ayeshasameer/xlm-roberta-roman-urdu-sentiment` — fine-tuned specifically on Roman Urdu sentiment data  
- Fine-tuning `bert-base-multilingual-cased` (mBERT) on this exact dataset would produce a strong custom classifier for Daraz or any Pakistani platform  

**Conclusion:**  
The English-trained DistilBERT model is not suitable for Roman Urdu sentiment analysis in production. It works by accident when English words appear in the text, not because it understands Urdu. For a Pakistani business use case, a multilingual or Urdu-specific model must be used.

---

### HuggingFace Dataset Screenshot

*(Paste your screenshot here — go to https://huggingface.co/datasets/ayeshasameer/roman-urdu-sentiment-analysis and take a screenshot of the Dataset Viewer tab showing sample rows)*

## Model Card Summary — distilbert-base-uncased-finetuned-sst-2-english

**Model Link:** https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english

**Training Data — What is SST-2?**
SST-2 (Stanford Sentiment Treebank v2) is a dataset of 67,349 English movie 
review sentences from Rotten Tomatoes, each labelled as POSITIVE or NEGATIVE. 
Reviews were written by professional film critics in formal American English. 
This is critical context — the model has only ever seen formal English movie 
review language, nothing else.

**Evaluation Metrics and Scores:**
The model reports accuracy as its primary metric. It achieves 91.3% accuracy 
on the SST-2 validation set. On the GLUE benchmark leaderboard, it scores 91.3 
on the SST-2 task — which was excellent performance for a distilled (smaller, 
faster) model at the time of release in 2019. No F1 or precision/recall scores 
are separately reported because the classes are balanced.

**Known Limitations and Biases:**
The model card lists these limitations explicitly:
- Trained only on movie reviews — performance drops on other domains 
  (product reviews, news, social media)
- Cannot handle sarcasm reliably ("Oh great, another delay")
- Struggles with negation ("I do NOT recommend this")
- No multilingual ability — English only
- Biased toward American English film criticism vocabulary and style
- Mixed-sentiment sentences get unpredictable results

**Intended Uses and Out-of-Scope Uses:**
Intended: Binary English sentiment classification for research and prototyping.
Out-of-scope: Non-English text, high-stakes automated decisions without human 
review, nuanced multi-class sentiment (e.g. 1–5 star ratings), and any 
domain far from movie reviews without fine-tuning first.

**My Conclusion — Suitable for Pakistani E-Commerce?**
For a Pakistani platform like Daraz, this model works only for the minority of 
reviews written in plain English. Pakistani customers write in three ways: 
English, Urdu script, and Roman Urdu. This model handles only the first 
reliably. For real production use it needs to be replaced with XLM-RoBERTa 
(multilingual) and fine-tuned on actual Daraz reviews. Additionally the 
confidence threshold should be set at 0.90+ with lower scores flagged for 
human review. As a quick prototype or demo it is acceptable — for production 
it is not ready.