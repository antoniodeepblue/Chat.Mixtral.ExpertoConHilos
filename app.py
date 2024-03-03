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
    prompt, history, system_prompt, temperature=0.9, max_new_tokens=4096, top_p=0.95, repetition_penalty=1.0,):
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
        value="Asistente para los usuarios y clientes de la empresa Canal de Isabel II, https://oficinavirtual.canaldeisabelsegunda.es/",
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
    # Tengo que comprobar el número máximo de nuevos tokens, por el momento lo fijo a 4096.
    gr.Slider(
        label="Max new tokens",
        value=4096,
        minimum=0,
        maximum=4096,
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
    ["Quiero que me verifiquen el contador de agua de mi vivienda", "Asistente para los usuarios y clientes de la empresa Canal de Isabel II, https://oficinavirtual.canaldeisabelsegunda.es/", 0.7, 1500, 0.80, 1.1],
    ["Muestrame un cuadro con las tarifas que se aplican en el abastecimiento, depuración y alcantarillado ", "Asistente para los usuarios y clientes de la empresa Canal de Isabel II, https://oficinavirtual.canaldeisabelsegunda.es/, https://www.canaldeisabelsegunda.es/clientes/", 0.8, 4096, 0.85, 1.2],
    ["¿Qué es una acometida?", "Asistente para los usuarios y clientes de la empresa Canal de Isabel II, https://oficinavirtual.canaldeisabelsegunda.es/", 0.7, 1800, 0.75, 1.2],
    ["¿Qué teléfono tiene para averías, información y página web?", "Asistente para los usuarios y clientes de la empresa Canal de Isabel II, https://oficinavirtual.canaldeisabelsegunda.es/", 0.8, 2048, 0.80, 1.1],
]

# Crear una interfaz de chat Gradio con el modelo generativo
gr.ChatInterface(
    fn=generate,
    chatbot=gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True, likeable=True, layout="panel", height=500),
    textbox=gr.Textbox(placeholder="¿Qué parámetros definen la calidad del agua?", container=False, scale=7),
    theme="syddharth/gray-minimal",
    additional_inputs=additional_inputs,
    title="Mixtral 8B Fines didácticos Asistente de usuarios/clientes de Canal de Isabel ll",
    description='Autor: <a href=\"https://huggingface.co/Antonio49\">Antonio Fernández</a> de <a href=\"https://saturdays.ai/\">SaturdaysAI</a>. Formación: <a href=\"https://cursos.saturdays.ai/courses/\">Cursos Online AI</a> Aplicación desarrollada con fines docentes',
    examples=examples,
    cache_examples=True,
      retry_btn="Repetir",
      undo_btn="Deshacer",
      clear_btn="Borrar",
      submit_btn="Enviar",
    concurrency_limit=20,
).launch(show_api=False)
