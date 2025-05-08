import os
import gradio as gr



from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts

system_prompt = """You have to act as a professional doctor, I know you are not, but this is for learning purposes. 
What's in this image? Do you find anything wrong with it medically? If you make a differential, suggest some remedies. 
Do not add numbers or special characters. Your response should be one long paragraph, answering as if speaking to a real person. 
Do not say 'In the image I see' but rather 'With what I see, I think you have...'. 
Do not respond as an AI model in markdown. Keep the answer concise (max 2 sentences). No preamble, start immediately."""


def process_inputs(audio_filepath, image_filepath):
    speech_to_text_output = transcribe_with_groq(
        GROQ_API_KEY=os.environ.get("GROQ_API_KEY"), 
        audio_filepath=audio_filepath,
        stt_model="whisper-large-v3"
    )

    if image_filepath:
        doctor_response = analyze_image_with_query(
            query=system_prompt + speech_to_text_output, 
            encoded_image=encode_image(image_filepath), 
            model="meta-llama/llama-4-scout-17b-16e-instruct"
            #model= "llama-3.2-11b-vision-preview"
        )
    else:
        doctor_response = "No image provided for me to analyze"

    output_audio_path = "final.mp3"
    text_to_speech_with_gtts(input_text=doctor_response, output_filepath=output_audio_path) 

     # 🛑 Check if the file actually exists before returning
    if not os.path.exists(output_audio_path):
        raise FileNotFoundError(f"Error: {output_audio_path} was not created.")
    
    return speech_to_text_output, doctor_response, output_audio_path  # Return the correct file path


# Create the Gradio interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(type="filepath", label="Doctor's Voice Response")  # Fix the Audio Output
    ],
    title="CureVoice"
)

# Run Gradio with a public HTTPS link
iface.launch(share=True, debug=True)



#http://127.0.0.1:7860