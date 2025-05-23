from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ExplanationGenerator:
    def __init__(self, model_name="microsoft/phi-2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_explanation(self, sentence, flag, category):
        prompt = f"""
You are a helpful assistant that explains why a sentence belonging to Category: {category} is flagged as {flag.upper()}.

Sentence: "{sentence}"

Task: Explain clearly why the sentence is flagged under this category without repeating the sentence. Be concise, specific, and objective.

Rationale:
""".strip()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            do_sample=False,
            num_beams=4,
            eos_token_id=self.tokenizer.eos_token_id
        )

        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return explanation[len(prompt):].strip()

