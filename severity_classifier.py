from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import torch.nn.functional as F

class SeverityClassifier:
    """
    A zero-shot severity classifier that evaluates the sensitivity level of a sentence 
    (Safe, Warning, Critical) based on its content along with a prompt and a provided category using a 
    T5-style language model.

    Attributes:
        model_name (str): Name of the pretrained Hugging Face model to use (default: "google/flan-t5-base").
        tokenizer (AutoTokenizer): Tokenizer corresponding to the pretrained model.
        model (AutoModelForSeq2SeqLM): Sequence-to-sequence model for text generation and classification.
        device (torch.device): Device used for inference (CPU or GPU).

    Methods:
        classify_sentence(sentence: str, category: str) -> Tuple[str, Union[str, float]]:
            Uses prompt-based zero-shot classification to assign a severity level 
            ("Safe", "Warning", or "Critical") to the sentence based on its category.
            Returns the predicted label and the confidence score (if available).
    """

    def __init__(self, model_name="google/flan-t5-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def classify_sentence(self, sentence, category):
        prompt = f"""
        ### Definitions:
        - **Safe**: General discussion, non-sensitive small talk, or emotionally neutral content.
        - **Warning**: Gossip, rumors, mild emotional stress, or minor behavioral issues.
        - **Critical**: Any content that reflects harmful behavior (e.g., bullying, mental health crisis), unethical/criminal activity (e.g., cheating, substance use), or serious violations of conduct.

        ### Instruction:
        You are an AI assistant analyzing school-related conversations. Given a sentence and its category, determine whether it is Safe, Warning, or Critical.

        ### Input:
        Sentence: "{sentence}"
        Category: "{category}"

        ### Task:
        Classify the severity of the sentence as one of the following:
        - Safe
        - Warning
        - Critical

        Severity:
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True
            )

        label_token_id = output.sequences[0][-1]
        decoded = self.tokenizer.decode([label_token_id], skip_special_tokens=True).strip().lower()

        label_map = {"safe": "Safe", "warning": "Warning", "critical": "Critical"}
        label = label_map.get(decoded, "Unknown")

        if label == "Unknown":
            confidence = "N/A"
        else:
            scores = output.scores[0]
            probs = F.softmax(scores[0], dim=-1)
            confidence = round(probs[label_token_id].item(), 3)

        return label, confidence