from huggingface_hub import InferenceClient
import gradio as gr
import threading

client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

# Variables para controlar el estado de la conversación
conversation_started = False
conversation_ongoing = True
system_prompt = "Asistente para los usuarios y clientes de la empresa Canal de Isabel II, https://oficinavirtual.canaldeisabelsegunda.es/"

def format_prompt(message, history, system_prompt):
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    prompt += f"[INST] {system_prompt}, {message} [/INST]"
    return prompt

def generate(
    prompt, history, system_prompt, temperature=0.9, max_new_tokens=4096, top_p=0.95, repetition_penalty=1.0,
):
    global conversation_started, conversation_ongoing

    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = format_prompt(prompt, history, system_prompt)
    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        if "¡Hasta luego!" in response.token.text:  # Puedes ajustar este condicional según las respuestas de tu modelo
            conversation_ongoing = False
        yield output

def start_conversation():
    global conversation_started
    conversation_started = True

def end_conversation():
    global conversation_ongoing
    conversation_ongoing = False

additional_inputs = [
    gr.Textbox(
        label="System Prompt",
        value=system_prompt,
        max_lines=1,
        interactive=True,
    ),
    gr.Slider(
        label="Temperature",
        value=0.9,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        interactive=True,
        info="Valores más altos producen resultados más diversos",
    ),
    gr.Slider(
        label="Max new tokens",
        value=4096,
        minimum=0,
        maximum=4096,
        step=64,
        interactive=True,
        info="El máximo número de nuevos tokens",
    ),
    gr.Slider(
        label="Top-p (nucleus sampling)",
        value=0.90,
        minimum=0.0,
        maximum=1,
        step=0.05,
        interactive=True,
        info="Valores más altos muestrean más tokens de baja probabilidad",
    ),
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

examples = [
    ["Quiero que me verifiquen el contador de agua de mi vivienda", system_prompt, 0.7, 1500, 0.80, 1.1],
    ["Muestrame un cuadro con las tarifas que se aplican en el abastecimiento, depuración y alcantarillado ", system_prompt, 0.8, 4096, 0.85, 1.2],
    ["¿Qué es una acometida?", system_prompt, 0.7, 1800, 0.75, 1.2],
    ["¿Qué teléfono tiene para averías, información y página web?", system_prompt, 0.8, 2048, 0.80, 1.1],
]

# Crear una interfaz de chat Gradio con el modelo generativo
iface = gr.ChatInterface(
    fn=generate,
    chatbot=gr.Chatbot(avatar_images=["./15f4b2d3-c4f4-4a29-93cd-e47214953bd9.png", "./botm.png"], bubble_full_width=False, show_label=False, show_share_button=False, show_copy_button=True, likeable=True, layout="panel", height=500),
    textbox=gr.Textbox(placeholder="¿Qué parámetros definen la calidad del agua?", container=False, scale=7),
    theme="soft",
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
)

# Iniciar un hilo de conversación inicial
threading.Thread(target=start_conversation).start()

# Actualizar la interfaz después de la conversación inicial
while not conversation_started:
    iface.update()

# Iniciar la interfaz principal
iface.launch(show_api=False)

