from pydub import AudioSegment
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load the audio file
audio_path = "audio.wav"
audio = AudioSegment.from_file(audio_path, format="wav")
segment_duration_ms = 30 * 1000  # 30 seconds in milliseconds
num_segments = len(audio) // segment_duration_ms

# Split the audio into 30-second segments
split_audio = [audio[i * segment_duration_ms:(i + 1) * segment_duration_ms] for i in range(num_segments)]

# Export the split audio files
for i, segment in enumerate(split_audio, start=1):
    segment.export(f"{i}_audio_file.wav", format="wav")

# Load Wav2Vec2 model and processor (replacing Wav2Vec2Tokenizer)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Function for transcribing audio
def transcribe_audio(file_path):
    # Load the audio using librosa, resampling to 16kHz
    speech, _ = librosa.load(file_path, sr=16000)
    
    # Use processor to prepare the audio for the model
    inputs = processor(speech, return_tensors="pt", padding=True)
    
    # Run inference with the model
    with torch.no_grad():
        logits = model(input_values=inputs.input_values).logits
    
    # Decode the predicted ids to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# Transcribe all split audio files and store results
transcriptions = []
for i in range(1, num_segments + 1):
    file_path = f"{i}_audio_file.wav"
    transcription = transcribe_audio(file_path)
    print(f"Transcription for segment {i}: {transcription}")
    transcriptions.append(transcription)

# Combine all transcriptions into one text block
final_complete_speech = " ".join(transcriptions)
print("Final Transcription:")
print(final_complete_speech)

# Save the final transcription to a text file
with open("op.txt", "w") as file:
    file.write(final_complete_speech)
