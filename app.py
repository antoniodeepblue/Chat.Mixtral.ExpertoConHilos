# Importar las bibliotecas necesarias
from huggingface_hub import InferenceClient
import gradio as gr

# Crear un cliente de inferencia para el modelo preentrenado Mixtral-8x7B-Instruct-v0.1
client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

# Función para formatear el prompt con historial
def format_prompt(message, history, system_prompt):
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    prompt += f"[INST] {system_prompt}, {message} [/INST]"
    return prompt

# Función para generar respuestas dada una serie de parámetros
def generate(
    prompt, 
    history, 
    system_prompt= "Experto en servicios de abastecimiento, depuración, reutilización, alcantarillado y calidad del agua, para la empresa Canal de Isabel II.  Asistente que ofrece información: obras hidráulicas, cumplimiento de sanidad y seguridad de las instalaciones del Canal de Isabel II", 
    temperature=0.9, 
    max_new_tokens=4096, 
    top_p=0.95, 
    repetition_penalty=1.0, 
    input_max_length=256,):
    # Ajustar valores de temperatura y top_p para asegurar que estén en el rango adecuado
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    # Configurar los parámetros para la generación de texto
    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )

    # Formatear el prompt y obtener la respuesta del modelo de manera continua
    formatted_prompt = format_prompt(prompt, history, system_prompt)
    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    # Iterar a través de las respuestas en el stream
    for response in stream:
        output += response.token.text
        yield output
    return output

# Crear una interfaz de chat Gradio con el modelo generativo
gr.ChatInterface(
    fn=generate,
    chatbot=gr.Chatbot(
        avatar_images=["./botm.png", "./logoCanal.png"],
        bubble_full_width=False,
        show_label=False,
        show_share_button=False,
        show_copy_button=True,
        likeable=True,
        layout="panel",
        height=500,
    ),
    textbox=gr.Textbox(placeholder="¿Qué es una tubería de polietiteno?", container=False, scale=7),
    theme="soft",
    title="Mixtral 8B. TFG: Experto en sector del agua del Canal de Isabel ll",
    description='Autor: <a href=\"https://huggingface.co/Antonio49\">Antonio Fernández</a> de <a href=\"https://www.canaldeisabelsegunda.es/\">Canal de Isabel II</a>. Formación: <a href=\"https://www.uoc.edu/es/\">Grado Ingeniería Informática</a>.  Aplicación desarrollada para TFG',
        retry_btn="Repetir",
        undo_btn="Deshacer",
        clear_btn="Borrar",
        submit_btn="Enviar",
    concurrency_limit=20,
).launch(show_api=False)
