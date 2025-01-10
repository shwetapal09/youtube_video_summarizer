from flask import Flask, request, render_template
import os
from pytube import YouTube
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from youtube_transcript_api import YouTubeTranscriptApi
app = Flask(__name__)
# Load the Pegasus model for summarization
model_name = "google/pegasus-xsum"
model = PegasusForConditionalGeneration.from_pretrained(model_name)
tokenizer = PegasusTokenizer.from_pretrained(model_name)

def extract_transcript_details(youtube_video_url):
    try:
        # Extract video ID from different possible YouTube URL formats
        video_id = None
        if "youtu.be/" in youtube_video_url:
            video_id = youtube_video_url.split("youtu.be/")[1].split("?")[0]
        elif "youtube.com/watch" in youtube_video_url:
            video_id = youtube_video_url.split("v=")[1].split("&")[0]

        if not video_id:
            raise ValueError("Invalid YouTube URL")

        # Retrieve the transcript
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([i["text"] for i in transcript_text])

        return transcript

    except Exception as e:
        raise e

def chunk_text(text, max_length=500):
    """Split text into chunks of `max_length` words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length):
        chunk = ' '.join(words[i:i + max_length])
        chunks.append(chunk)
    return chunks

def summarize_text(text):
    inputs = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=100, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_long_text(transcript):
    chunks = chunk_text(transcript, max_length=500)  # Chunk transcript into 500-word parts
    summaries = [summarize_text(chunk) for chunk in chunks]
    long_summary = '.'.join(summaries)
    return long_summary

@app.route('/', methods=['GET', 'POST'])
def index():
    transcription = ""
    summary = ""

    if request.method == 'POST':
        video_url = request.form['youtube_link']
        transcription = extract_transcript_details(video_url)
        summary = summarize_long_text(transcription)

    return render_template('Index.html', transcription=transcription, summary=summary)

if __name__ == "__main__": 
    app.run(debug=True)
