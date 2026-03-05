# Audio Transcriber and Summarizer

An intelligent audio processing pipeline that transcribes, diarizes, and summarizes audio files or microphone input. Built with Python and Gradio, this tool uses state-of-the-art machine learning models to provide accurate speaker-annotated transcriptions and context-aware summaries.

## Features

- **Speaker Diarization**: Identifies different speakers in the audio using `pyannote/speaker-diarization`.
- **Audio Transcription**: Converts speech to text using OpenAI's `whisper` models.
- **Content Recognition & Summarization**: Uses Llama models (`meta-llama/Llama-3.2-3B-Instruct` or similar) to categorize the content (Meeting, Speech, or General Conversation) and generates an appropriate summary.
  - *Meeting*: Produces detailed minutes of the meeting including key points, takeaways, and action items.
  - *Speech*: Summarizes the key points, objectives, and goals.
  - *Conversation*: Summarizes the discussion while clearly attributing points to specific speakers.
- **Gradio Web Interface**: A user-friendly, interactive UI that allows users to upload audio files or record directly from their microphone.
- **Customizable Settings**: Adjust diarization models, transcription models, summarization models, minimum/maximum speaker counts, and quantization settings for optimized inference.

## Prerequisites

To run this project, you need the following dependencies installed. An NVIDIA GPU is recommended for faster inference.

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Gradio
- `pyannote.audio`
- `bitsandbytes`, `accelerate`
- `soundfile`

You will also need a **Hugging Face Token** with access to the specified Pyannote and Llama models.

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone <repository_url>
   cd AudioTranscriber-Summarizer
   ```

2. Install the required Python packages:
   ```bash
   pip install bitsandbytes accelerate transformers pyannote.audio soundfile gradio
   ```

3. Configure your Hugging Face Token:
   Make sure you are logged in to the Hugging Face hub to download the gated models:
   ```python
   from huggingface_hub import login
   login("YOUR_HF_TOKEN")
   ```

## Usage

You can run the full pipeline directly through the provided Jupyter Notebook (`MeetingMinutes_ConversationSummary.ipynb`).

1. Open the notebook in your preferred environment (e.g., Jupyter, Google Colab).
2. Execute the cells sequentially to initialize the diarization, transcription, and summarization components.
3. Run the Gradio cell at the end of the notebook to launch the web interface.
4. Once the Gradio interface is running, you can:
   - Upload an audio file or record audio using your microphone.
   - Adjust settings in the "⚙️ Settings" panel (e.g., choosing `pyannote.audio` and `whisper` model versions).
   - View the generated transcript with speaker labels.
   - Click "📝 Summarize" to generate minutes of the meeting or a conversation summary.

## Supported Models in UI

NOTE: Make sure that you have access to the models by logging in to the Hugging Face hub.
* **Audio Diarization**: 
  - `pyannote/speaker-diarization-3.1`
  - `pyannote/speaker-diarization-3.2`
* **Transcription**: 
  - `openai/whisper-small`
  - `openai/whisper-medium.en`
* **Summarization**: 
  - `meta-llama/Llama-3.2-1B-Instruct`
  - `meta-llama/Llama-3.2-3B-Instruct`
  - `meta-llama/Llama-3.1-8B-Instruct`

## Acknowledgements

- [Pyannote](https://github.com/pyannote/pyannote-audio) for Speaker Diarization.
- [OpenAI Whisper](https://github.com/openai/whisper) for Speech-to-Text.
- [Meta Llama](https://ai.meta.com/llama/) for Summarization.
- [Gradio](https://www.gradio.app/) for the Web UI.