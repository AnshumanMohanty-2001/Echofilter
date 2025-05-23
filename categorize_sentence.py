import torch
from sentence_transformers import SentenceTransformer, util

class CategorizeSentence:
    """
    A semantic and contextual classifier that assigns user-defined categories
    to sentences based on similarity using Sentence-BERT embeddings.

    Attributes:
        categories (List[str]): List of sensitive categories to detect.
        model (SentenceTransformer): Pretrained model for sentence embeddings.
        device (torch.device): Device to perform inference (CPU or GPU).
        category_embeddings (Tensor): Embeddings of the category list.

    Methods:
        classify_segment(segment: str): 
            Classifies a single sentence by comparing semantic similarity to categories.

        classify_with_context(sentences: List[str]):
            Classifies sentences with surrounding context (previous and next) for better accuracy and compares the results with semantic similarity and context window. It then returns the category with the highest similarity

        analyze_transcript(transcript: str):
            Splits a transcript into lines and applies contextual classification.
    """

    def __init__(self, category_list):
        self.categories = category_list
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer("paraphrase-mpnet-base-v2")
        self.model.to(self.device)
        self.category_embeddings = self.model.encode(
            self.categories, convert_to_tensor=True, device=self.device
        )

    def classify_segment(self, segment: str):
        segment_embedding = self.model.encode(
            segment, convert_to_tensor=True, device=self.device
        )
        similarities = util.cos_sim(segment_embedding, self.category_embeddings)[0]
        best_score = float(similarities.max())
        best_idx = int(similarities.argmax())
        return best_idx, best_score

    def classify_with_context(self, sentences):
        results = []
        n = len(sentences)

        for i, sentence in enumerate(sentences):
            current = sentence
            prev = sentences[i - 1] if i > 0 else ""
            next_ = sentences[i + 1] if i < n - 1 else ""

            idx1, score1 = self.classify_segment(current)
            combined = f"{prev} {current} {next_}".strip()
            idx2, score2 = self.classify_segment(combined) if prev or next_ else (None, -1)

            candidates = [(idx1, score1), (idx2, score2)]
            best_idx, best_score = max(candidates, key=lambda x: x[1])

            results.append({
                "segment": current,
                "category": self.categories[best_idx]
            })

        return results

    def analyze_transcript(self, transcript: str):
        sentences = [line.strip() for line in transcript.strip().split("\n") if line.strip()]
        return self.classify_with_context(sentences)
