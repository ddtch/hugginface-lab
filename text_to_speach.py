from transformers import pipeline
import scipy
import gradio as gr

synthesiser = pipeline("text-to-speech", "suno/bark")

def text_to_speach(text):
  speech = synthesiser(text, forward_params={"do_sample": True})
  scipy.io.wavfile.write("bark_out.wav", rate=speech["sampling_rate"], data=speech["audio"])
  return speech

text_to_speach("Hello, my dog is cooler than you!")

gr.Interface(
    fn=text_to_speach,
    inputs=gr.Textbox(lines=2, placeholder="Enter text to convert to speech"),
    outputs=gr.Audio(type="file"),
    title="Text to Speech",
    description="Convert text to speech using the Suno Bark model"
).launch()
