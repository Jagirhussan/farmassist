# this file calls whisper and returns the transcription
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

def transcribe_audio(audio_path):
    """
    Transcribe audio using the Whisper model.
    Args:
        audio_path (str): Path to the audio file. The audio file should be in a format supported by the Whisper model.
    Returns:
        str: Transcription of the audio file.

    Author: Alex Foster
    Date: 2023-10-01
    """

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
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return transcription

if __name__ == "__main__":
    import sys

    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python3 whisper.py <audio_file_path>")
        sys.exit(1)

    # Get the audio file path from command line arguments
    audio_file_path = sys.argv[1]
    transcription = transcribe_audio(audio_file_path)
    print("Transcription:", transcription)

    # save the transcription to a file
    with open("transcription.txt", "w") as f:
        f.write(transcription)