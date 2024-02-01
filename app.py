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
    prompt, history, system_prompt= "Soy un Experto en Inteligencia Artificial", temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0,
):
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

# Configurar inputs adicionales para la interfaz Gradio
additional_inputs = [
    # Entrada de texto para el System Prompt (puedes omitir esto si no lo necesitas)
    gr.Textbox(
        label="System Prompt",
        max_lines=1,
        interactive=True,
    ),
    # Control deslizante para la temperatura
    gr.Slider(
        label="Temperature",
        value=0.9,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        interactive=True,
        info="Valores más altos producen resultados más diversos",
    ),
    # Control deslizante para el número máximo de nuevos tokens
    gr.Slider(
        label="Max new tokens",
        value=256,
        minimum=0,
        maximum=1048,
        step=64,
        interactive=True,
        info="El máximo número de nuevos tokens",
    ),
    # Control deslizante para top-p (nucleus sampling)
    gr.Slider(
        label="Top-p (nucleus sampling)",
        value=0.90,
        minimum=0.0,
        maximum=1,
        step=0.05,
        interactive=True,
        info="Valores más altos muestrean más tokens de baja probabilidad",
    ),
    # Control deslizante para la penalización de repetición
    gr.Slider(
        label="Repetition penalty",
        value=1.2,
        minimum=1.0,
        maximum=2.0,
        step=0.05,
        interactive=True,
        info="Penaliza los tokens repetidos",
    )
]

# Ejemplos predefinidos para la interfaz Gradio
examples = [
    ["Cuéntame una historia de ciencia ficción que involucre viajes en el tiempo y descubrimientos científicos asombrosos.", None, 0.7, 200, 0.80, 1.2],
    ["Necesito ayuda para redactar un correo electrónico profesional. ¿Puedes sugerirme cómo empezar y qué puntos clave incluir?", None, 0.7, 200, 0.80, 1.2],
    ["Estoy organizando una fiesta sorpresa. ¿Puedes ayudarme a redactar una invitación creativa y emocionante?", None, 0.7, 200, 0.80, 1.2],
    ["Quiero aprender a tocar la guitarra. ¿Puedes proporcionar algunos consejos prácticos para principiantes y sugerencias de canciones fáciles?", None, 0.7, 200, 0.80, 1.2],
    ["¿Cuál es tu opinión sobre el impacto de la inteligencia artificial en la sociedad y cómo deberíamos abordar sus desafíos éticos?", None, 0.7, 200, 0.80, 1.2],
    ["Estoy planeando un viaje gastronómico. ¿Puedes sugerirme algunos destinos culinarios imperdibles y platos locales para probar?", None, 0.7, 200, 0.80, 1.2],
    ["Viaje a la isla de Ibiza", "Prompt del sistema sobre destinos turísticos", 0.8, 300, 0.85, 1.1],
    ["Cuentame una historia creativa", "Prompt del sistema crativo", 0.7, 200, 0.80, 1.2],
    ["Cuéntame una historia de ciencia ficción", "Prompt del sistema de ciencia ficción", 0.7, 150, 0.80, 1.1],
    ["Receta de cochinillo asado", "Prompt del sistema sobre cocina", 0.8, 250, 0.85, 1.2],
    ["Recomendación de película de acción", "Prompt del sistema de recomendación", 0.7, 180, 0.75, 1.2],
    ["¿Que es la vía Lactea?", "Prompt del sistema científico", 0.8, 180, 0.80, 1.1],
]

# Crear una interfaz de chat Gradio con el modelo generativo
gr.ChatInterface(
    fn=generate,
    chatbot=gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True, likeable=True, layout="panel"),
    additional_inputs=additional_inputs,
    title="Mixtral 8B Fines didácticos",
    description='Autor: <a href=\"https://huggingface.co/Antonio49\">Antonio Fernández</a> de <a href=\"https://saturdays.ai/\">SaturdaysAI</a>. Formación: <a href=\"https://cursos.saturdays.ai/courses/\">Cursos Online AI</a> Aplicación desarrollada con fines docentes',
    examples=examples,
    concurrency_limit=20,
).launch(show_api=False)
