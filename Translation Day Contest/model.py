# model.py
import pickle
import requests
import json
import uuid
import numpy as np
from langdetect import detect
from nltk.tokenize import word_tokenize
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import Levenshtein
from transformers import BertTokenizer, BertModel
import torch

# Initialize NLP models
nlp = spacy.load('en_core_web_md')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Azure Translator API credentials
key = "4OmnQCmYuVDlJAMrakRn3PEnY2WpgJGuuaudutgr2PgJ570AYXjaJQQJ99AKACGhslBXJ3w3AAAbACOGfyVA"
endpoint = "https://api.cognitive.microsofttranslator.com/"
location = "centralindia"

def detect_language(text):
    return detect(text)

def azure_translate(text, src_lang, target_lang):
    path = '/translate'
    constructed_url = endpoint + path
    params = {'api-version': '3.0', 'from': src_lang, 'to': [target_lang]}
    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    body = [{'text': text}]
    response = requests.post(constructed_url, params=params, headers=headers, json=body)
    return response.json()[0]['translations'][0]['text']

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def convert_floats(obj):
    if isinstance(obj, np.float32):  # Check if it's a numpy float32
        return float(obj)  # Convert to Python float
    if isinstance(obj, dict):  # If it's a dictionary, apply the function to its values
        return {key: convert_floats(value) for key, value in obj.items()}
    if isinstance(obj, list):  # If it's a list, apply the function to its elements
        return [convert_floats(item) for item in obj]
    return obj  # Return the object if it's neither a dict, list, nor numpy float32

def calculate_metrics(text1, text2):
    # Cosine Similarity
    vector1 = nlp(text1).vector
    vector2 = nlp(text2).vector
    cosine_sim = cosine_similarity([vector1], [vector2])[0][0]

    # BLEU Score
    bleu_score = sentence_bleu([text1.split()], text2.split())

    # ROUGE Score
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge_scorer_instance.score(text1, text2)

    # METEOR Score
    meteor = meteor_score([word_tokenize(text1)], word_tokenize(text2))

    # Levenshtein Distance
    lev_similarity = 1 - Levenshtein.distance(text1, text2) / max(len(text1), len(text2))

    # BERT-based Cosine Similarity
    bert_cos_sim = torch.nn.functional.cosine_similarity(get_embedding(text1), get_embedding(text2)).item()

    # Convert all metrics to float if they are numpy.float32
    aggregate_score = (
            0.3 * float(cosine_sim) + 0.2 * float(bleu_score) + 0.2 * float(0.5 * rouge_scores['rouge1'].fmeasure + 0.3 * rouge_scores['rouge2'].fmeasure + 0.2 * rouge_scores['rougeL'].fmeasure) +
            0.1 * float(meteor) + 0.1 * float(lev_similarity) + 0.1 * float(bert_cos_sim)
    )

    quality = 5 if aggregate_score >= 0.80 else 4 if aggregate_score >= 0.60 else 3 if aggregate_score >= 0.40 else 2 if aggregate_score >= 0.20 else 1

    # Return the metrics ensuring all values are in the correct format
    return {
        'Cosine Similarity': convert_floats(cosine_sim),
        'BLEU Score': convert_floats(bleu_score),
        'ROUGE-1 F1': convert_floats(rouge_scores['rouge1'].fmeasure),
        'ROUGE-2 F1': convert_floats(rouge_scores['rouge2'].fmeasure),
        'ROUGE-L F1': convert_floats(rouge_scores['rougeL'].fmeasure),
        'METEOR Score': convert_floats(meteor),
        'Levenshtein Similarity': convert_floats(lev_similarity),
        'BERT Cosine Similarity': convert_floats(bert_cos_sim),
        'Aggregate Quality Score': convert_floats(aggregate_score),
        'Quality Evaluation': quality
    }
def generate_translation_evaluation(source_text, human_translation):
    # Detect languages
    src_lang = detect_language(source_text)
    target_lang = detect_language(human_translation)

    # Translate texts using Azure
    A_prime = azure_translate(source_text, src_lang, target_lang)
    B_prime = azure_translate(human_translation, target_lang, src_lang)

    # Calculate metrics for both A -> B' and A' -> B
    metrics_AB_prime = calculate_metrics(source_text, B_prime)
    metrics_A_primeB = calculate_metrics(A_prime, human_translation)

    # Return the results as a dictionary with all floats properly converted
    return {
        "generated_translation": A_prime,
        "generated_source_text": B_prime,
        "metric_AB_prime": metrics_AB_prime,
        "metric_A_primeB": metrics_A_primeB
    }