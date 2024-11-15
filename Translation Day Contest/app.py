from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model import generate_translation_evaluation

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        source_text = data['source_text']
        human_translation = data['human_translation']

        evaluation_results = generate_translation_evaluation(source_text, human_translation)

        generated_source_text = evaluation_results['generated_source_text']
        generated_translation = evaluation_results['generated_translation']
        cosine_similarity_ab = evaluation_results['metric_AB_prime']['Cosine Similarity']
        cosine_similarity_ba = evaluation_results['metric_A_primeB']['Cosine Similarity']
        bleu_score_ab = evaluation_results['metric_AB_prime']['BLEU Score']
        bleu_score_ba = evaluation_results['metric_A_primeB']['BLEU Score']
        meteor_score_ab = evaluation_results['metric_AB_prime']['METEOR Score']
        meteor_score_ba = evaluation_results['metric_A_primeB']['METEOR Score']
        rouge_1_f1_ab = evaluation_results['metric_AB_prime']['ROUGE-1 F1']
        rouge_1_f1_ba = evaluation_results['metric_A_primeB']['ROUGE-1 F1']
        rouge_2_f1_ab = evaluation_results['metric_AB_prime']['ROUGE-2 F1']
        rouge_2_f1_ba = evaluation_results['metric_A_primeB']['ROUGE-2 F1']
        rouge_l_f1_ab = evaluation_results['metric_AB_prime']['ROUGE-L F1']
        rouge_l_f1_ba = evaluation_results['metric_A_primeB']['ROUGE-L F1']
        levenshtein_similarity_ab = evaluation_results['metric_AB_prime']['Levenshtein Similarity']
        levenshtein_similarity_ba = evaluation_results['metric_A_primeB']['Levenshtein Similarity']
        bert_cosine_similarity_ab = evaluation_results['metric_AB_prime']['BERT Cosine Similarity']
        bert_cosine_similarity_ba = evaluation_results['metric_A_primeB']['BERT Cosine Similarity']
        aggregate_score_ab = evaluation_results['metric_AB_prime']['Aggregate Quality Score']
        aggregate_score_apb = evaluation_results['metric_A_primeB']['Aggregate Quality Score']
        translation_quality_ab = evaluation_results['metric_AB_prime']['Quality Evaluation']
        translation_quality_apb = evaluation_results['metric_A_primeB']['Quality Evaluation']

        response_data = {
            "generated_source_text": generated_source_text,
            "generated_translation": generated_translation,
            "cosine_similarity_ab": cosine_similarity_ab,
            "cosine_similarity_ba": cosine_similarity_ba,
            "bleu_score_ab": bleu_score_ab,
            "bleu_score_ba": bleu_score_ba,
            "meteor_score_ab": meteor_score_ab,
            "meteor_score_ba": meteor_score_ba,
            "rouge_1_f1_ab": rouge_1_f1_ab,
            "rouge_1_f1_ba": rouge_1_f1_ba,
            "rouge_2_f1_ab": rouge_2_f1_ab,
            "rouge_2_f1_ba": rouge_2_f1_ba,
            "rouge_l_f1_ab": rouge_l_f1_ab,
            "rouge_l_f1_ba": rouge_l_f1_ba,
            "levenshtein_similarity_ab": levenshtein_similarity_ab,
            "levenshtein_similarity_ba": levenshtein_similarity_ba,
            "bert_cosine_similarity_ab": bert_cosine_similarity_ab,
            "bert_cosine_similarity_ba": bert_cosine_similarity_ba,
            "aggregate_score_ab": aggregate_score_ab,
            "aggregate_score_apb": aggregate_score_apb,
            "translation_quality_ab": translation_quality_ab,
            "translation_quality_apb": translation_quality_apb,
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)