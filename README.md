# EchoFilter-AI-Powered-Personal-Audio-Firewall

EchoFilter is a privacy-first AI tool that analyzes audio conversations by transcribing them and evaluating each sentence for semantic and contextual similarity to user-defined sensitive content categories. It flags and classifies detected segments into severity levels such as Safe, Warning, or Critical, and provides confidence scores along with rationale for each classification. The system also generates redacted versions of the transcript to mask the sensitive data.

[![Open App in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://echofilter-ai-powered-personal-audio-firewall-xrvzzd3ozwzc7om9.streamlit.app/)

## Table of contents
<ol>

  <li>
    <a href="#dataset-description">Dataset Description</a>
  </li>
  <li>
    <a href="#project-structure">Project Structure</a>
  </li>
<li>
    <a href="#project-structure">System Architecture Overview</a>
  </li>
  <li>
    <a href="#getting-started">Installation Steps</a>
  </li>
</ol>
<br>

## Dataset Description
The dataset consists of a single audio recording capturing a natural conversation between two friends, where one friend is updating the other about missing school. This audio serves as the input for the EchoFilter system, which transcribes and analyzes the conversation to detect sensitive topics such as bullying, mental health concerns, or substance use. The primary use case is parental supervision—helping parents monitor and identify potentially sensitive or critical content in their children’s conversations to promote safety and timely intervention.
<br>

## Project Structure
```sh
├── app.py                        # Main Streamlit app
├── categorize_sentence.py        # Assigns category label
├── severity_classifier.py        # Classifies severity of segments
├── transcribe.py                 # Transcription logic
├── summarizer.py                 # Explanation generator for segments
├── Sample_audio/                 # Input audio files
├── outputs/                      # Transcript outputs
├── requirements.txt              # Python dependencies
```
<br>

## System Architecture Overview
### transcribe.py
The Transcriber module handles audio-to-text conversion using the efficient Faster-Whisper model. It transcribes spoken audio into clean, sentence-level transcripts by applying beam search decoding and natural language sentence tokenization. Segments are joined and split into natural language sentences and it returns the cleaned, sentence-level transcript as a single string.

### Categorize_sentence.py
The CategorizeSentence module provides semantic and contextual classification of text segments using Sentence-BERT embeddings. It maps each sentence in a transcript to a user-defined sensitive category (e.g., bullying, mental health, substance use) by computing cosine similarity between the sentence and category embeddings. To improve classification accuracy, it evaluates both the sentence alone (semantic similarity) and the sentence combined with its surrounding context (contextual similarity). The category associated with the highest similarity score from either of these two methods is selected as the final label. This dual-level evaluation enables more robust detection of sensitive content, especially when meaning is implied across multiple sentences in conversational transcripts. It returns the category with the highest similarity.

### Severity_classifier.py
The SeverityClassifier is a zero-shot prompt-based severity classification module that uses a pretrained T5-style sequence-to-sequence language model (default: google/flan-t5-base) to evaluate the sensitivity level of sentences within a given category. By framing severity detection as a natural language generation task, the model classifies sentences into three levels — Safe, Warning, or Critical — based on predefined definitions provided in the prompt. This approach leverages contextual understanding and generalizes well to new categories without additional fine-tuning. The classifier outputs both the predicted severity label and a confidence score when available, enabling nuanced assessment of potentially sensitive or harmful content in conversational transcripts.

### summarizer.py
The ExplanationGenerator utilizes a pretrained causal language model (default: microsoft/phi-2) to generate concise, clear, and objective rationales explaining why a specific sentence was flagged under a given sensitive category and severity level. Given a sentence, its assigned category, and the severity flag (e.g., Safe, Warning, Critical), the model produces a natural language explanation without simply repeating the sentence. This helps provide transparent, human-readable insights into the reasons behind content flagging decisions, aiding in interpretability and trust for sensitive content detection systems.
<br>

## Installation Steps
1. Clone the repository to your local directory by entering the following command:
      ```sh
      git clone https://github.com/AnshumanMohanty-2001/EchoFilter-AI-Powered-Personal-Audio-Firewall.git
      ```

2. After navigating inside the project directory, we need to type in the command: 
      ```sh
      python -m venv venv
      ```
    Here, venv is the name of virtual environment.

3. Upon creation of the virtual environment, the following command needs to be entered to activate the virtual environment.
      ```sh
      cd venv/Scripts
      ```
      ```sh
      ./activate
      ```

4. The following command installs the required libraries: 
      ```sh
      pip install -r requirements.txt
      ```

5. Navigate to the path of app.py to run the application.
      ```sh
      streamlit run app.py
      ```

6. In the UI, upload your audio recording and enter the categories. Then click on process audio.

7. The result is generated in form of analyzed transcript and redacted transcript.
