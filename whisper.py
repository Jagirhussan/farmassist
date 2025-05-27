# this file calls whisper and returns the transcription
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from dataset import load_dataset

def transcribe_audio(audio_path):
    # Load the Whisper model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

    # Load the audio file
    audio_dataset = load_dataset("audio", data_files={"audio": audio_path})

    # Process the audio file
    inputs = processor(audio_dataset["audio"][0]["array"], return_tensors="pt", sampling_rate=16000)

    # Generate transcription
    generated_ids = model.generate(inputs.input_features)

    # Decode the generated ids to text
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return transcription