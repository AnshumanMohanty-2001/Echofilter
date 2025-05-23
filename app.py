import streamlit as st
from categorize_sentence import CategorizeSentence
from severity_classifier import SeverityClassifier
from transcribe import Transcriber
from summarizer import ExplanationGenerator
import os
import html
from streamlit_tags import st_tags

st.set_page_config(page_title="EchoFilter", layout="wide")
st.title("🔊 EchoFilter - Smart Audio Firewall")

uploaded_file = st.file_uploader("Upload audio file (.wav or .mp3)", type=["wav", "mp3"])
audio_path = None

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    audio_path = os.path.join("temp_audio", uploaded_file.name)
    os.makedirs("temp_audio", exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

st.subheader("📌 Add custom categories for analysis")
category_ip = st_tags(
    label='### Enter categories (press Enter to add each)',
    text='Add a category...',
    value=[],
    suggestions=[],
    key='category_tags'
)

process = st.button("🚀 Process Audio")

if process:
    if not uploaded_file:
        st.error("❌ Please upload an audio file.")
    elif not category_ip:
        st.error("❌ Please enter at least one category.")
    else:
        with st.spinner("🔍 Transcribing and analyzing audio..."):
            transcriber = Transcriber()
            transcribed_segments = transcriber.speech_to_text(audio_path, output_path='outputs/translated_transcript.txt')

            content_filter = CategorizeSentence(category_list=category_ip)
            results = content_filter.analyze_transcript(transcribed_segments)

            severity_filter = SeverityClassifier()
            explanation = ExplanationGenerator()

            final_results = []
            for r in results:
                if r['category'] == 'General Discussions':
                    r['severity'] = 'Safe'
                    r['rationale'] = explanation.generate_explanation(r["segment"], r['severity'], r['category'])
                    severity_result = severity_filter.classify_sentence(r["segment"], r['category'])
                    r["severity"], r["confidence"] = severity_result

                    print('rationale:', r['rationale'])
                else:
                    severity_result = severity_filter.classify_sentence(r["segment"], r['category'])

                    if isinstance(severity_result, tuple):
                        r["severity"], r["confidence"] = severity_result
                    else:
                        r["severity"], r["confidence"] = severity_result
                    r['rationale'] = explanation.generate_explanation(r["segment"], r['severity'], r['category'])
                    print('rationale:', r['rationale'])
                final_results.append(r)

        print(final_results)

    st.subheader("🔎 Analyzed Transcript")

    st.markdown("""
    <style>
    .transcript-line {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        border-left: 4px solid;
        position: relative;
        cursor: help;
        transition: all 0.3s ease;
        color: black;
    }

    .transcript-line:hover {
        transform: translateX(5px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    .safe {
        background-color: #d4edda;
        border-left-color: #28a745;
    }

    .warning {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }

    .critical {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }

    .tooltip {
        visibility: hidden;
        width: 350px;
        max-height: 200px;
        overflow-y: auto;
        background-color: #333;
        color: white;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        margin-left: -175px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 11px;
        line-height: 1.3;
        word-wrap: break-word;
    }

    .tooltip::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #333 transparent transparent transparent;
    }

    .transcript-line:hover .tooltip {
        visibility: visible;
        opacity: 1;
    }

    .severity-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        margin-left: 10px;
    }

    .safe-badge { background-color: #28a745; color: white; }
    .warning-badge { background-color: #ffc107; color: black; }
    .critical-badge { background-color: #dc3545; color: white; }
    </style>
    """, unsafe_allow_html=True)

    if 'final_results' in locals():
        for i, result in enumerate(final_results):
            severity = result['severity'].lower()
            segment = result['segment']
            category = result['category']
            confidence = result.get('confidence', 'N/A')
            rationale = result['rationale']
            
            segment_escaped = html.escape(segment)
            category_escaped = html.escape(category)
            confidence_escaped = html.escape(str(confidence))
            rationale_escaped = html.escape(rationale)
            
            if len(rationale_escaped) > 200:
                rationale_tooltip = rationale_escaped[:200] + "..."
            else:
                rationale_tooltip = rationale_escaped
            
            badge_class = f"{severity}-badge"
            severity_display = result['severity'].upper()
            
            line_html = f"""
            <div class="transcript-line {severity}">
                <strong>Line {i+1}:</strong> {segment_escaped}
                <span class="severity-badge {badge_class}">{severity_display}</span>
                <div class="tooltip">
                    <strong>Category:</strong> {category_escaped}<br>
                    <strong>Severity:</strong> {severity_display}<br>
                    <strong>Confidence:</strong> {confidence_escaped}<br>
                    <strong>Rationale:</strong> {rationale_tooltip}
                </div>
            </div>
            """
            
            st.markdown(line_html, unsafe_allow_html=True)

        analyzed_transcript = ""
        for i, result in enumerate(final_results):
            analyzed_transcript += f"Line {i+1}: {result['segment']}\n"
            analyzed_transcript += f"  Category: {result['category']}\n"
            analyzed_transcript += f"  Severity: {result['severity'].upper()}\n"
            analyzed_transcript += f"  Confidence: {result.get('confidence', 'N/A')}\n"
            analyzed_transcript += f"  Rationale: {result['rationale']}\n"
            analyzed_transcript += "-" * 50 + "\n"

        st.download_button(
            label="📊 Download Analyzed Transcript",
            data=analyzed_transcript,
            file_name="analyzed_transcript.txt",
            mime="text/plain"
        )
    
    st.subheader("🔒 Redacted Transcript")
    
    if 'final_results' in locals():
        with st.container():
            for i, r in enumerate(final_results):
                line_num = f"[{i+1:02d}] "
                
                if r['severity'].lower() == 'critical':
                    st.markdown(
                        f'<div style="background-color: #f8d7da; padding: 8px; margin: 4px 0; border-radius: 4px; font-family: monospace; color: black;">'
                        f'<strong>{line_num}</strong>'
                        f'<span style="background-color: #343a40; color: white; padding: 2px 8px; border-radius: 3px;">🔒 [REDACTED: {r["category"]}]</span>'
                        f'</div>', 
                        unsafe_allow_html=True
                    )
                elif r['severity'].lower() == 'warning':
                    st.markdown(
                        f'<div style="background-color: #fff3cd; padding: 8px; margin: 4px 0; border-radius: 4px; font-family: monospace; color: black;">'
                        f'<strong>{line_num}</strong>{r["segment"]} '
                        f'<em style="color: #856404;">(⚠️ flagged as {r["category"]})</em>'
                        f'</div>', 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div style="background-color: #d4edda; padding: 8px; margin: 4px 0; border-radius: 4px; font-family: monospace; color: black;">'
                        f'<strong>{line_num}</strong>{r["segment"]}'
                        f'</div>', 
                        unsafe_allow_html=True
                    )
        
        transcript_with_redacts = ""
        for r in final_results:
            if r['severity'].lower() == 'critical':
                transcript_with_redacts += f'[REDACTED: {r["category"]}]\n'
            else:
                transcript_with_redacts += r['segment'] + '\n'
        
        st.download_button(
            label="📥 Download Redacted Transcript",
            data=transcript_with_redacts,
            file_name="redacted_transcript.txt",
            mime="text/plain"
        )