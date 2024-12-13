from pydub import AudioSegment
mp3_audio = AudioSegment.from_file(r"audio.wav", format="wav")
print(len(mp3_audio)/(1000*30))
counter_audio = 30
split_audio = [mp3_audio[:30*1000]]
for i in range(round(len(mp3_audio)/(1000*30))):
    split_audio.append(mp3_audio[counter_audio*1000:(counter_audio+30)*1000])
    counter_audio += 30
count = 0
for count, audio_object in enumerate(split_audio):
    count += 1
    with open(f"{count}_audi_file.wav", 'wb') as out_f:
        audio_object.export(out_f, format='wav')
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
# load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
# The base model pretrained and fine-tuned on 960 hours of Librispeech on 16kHz sampled speech audio.
# When using the model make sure that your speech input is also sampled at 16Khz.
# load any audio file of your choice
collection_of_text = []
for i in range(round(len(mp3_audio)/(1000*30))):
    speech, rate = librosa.load(f"{i+1}_audi_file.wav", sr=16000)
    input_values = tokenizer(speech, return_tensors='pt').input_values
    # Store logits (non-normalized predictions)
    with torch.no_grad():
        logits = model(input_values).logits
    # Store predicted id's
    predicted_ids = torch.argmax(logits, dim=-1)
    # decode the audio to generate text
    # Passing the prediction to the tokenzer decode to get the transcription
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    # transcriptions = tokenizer.decode(predicted_ids[0])
    print(transcription)
    collection_of_text.append(transcription)
print(collection_of_text)
final_complete_speech = ""
# convert batch of text into one complete sentence
for i in collection_of_text:
    final_complete_speech += i
print(final_complete_speech)
file = open("op.txt", "w")
file.write(final_complete_speech)