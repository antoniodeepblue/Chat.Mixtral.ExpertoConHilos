from huggingface_hub import InferenceClient
import gradio as gr

client = InferenceClient(
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
)


def format_prompt(message, history):
  prompt = "<s>"
  for user_prompt, bot_response in history:
    prompt += f"[INST] {user_prompt} [/INST]"
    prompt += f" {bot_response}</s> "
  prompt += f"[INST] {message} [/INST]"
  return prompt

def generate(
    prompt, history, system_prompt, temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0,
):
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

    formatted_prompt = format_prompt(f"{system_prompt}, {prompt}", history)
    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        yield output
    return output


additional_inputs=[
    gr.Textbox(
        label="System Prompt",
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
        info="Higher values produce more diverse outputs",
    ),
    gr.Slider(
        label="Max new tokens",
        value=256,
        minimum=0,
        maximum=1048,
        step=64,
        interactive=True,
        info="The maximum numbers of new tokens",
    ),
    gr.Slider(
        label="Top-p (nucleus sampling)",
        value=0.90,
        minimum=0.0,
        maximum=1,
        step=0.05,
        interactive=True,
        info="Higher values sample more low-probability tokens",
    ),
    gr.Slider(
        label="Repetition penalty",
        value=1.2,
        minimum=1.0,
        maximum=2.0,
        step=0.05,
        interactive=True,
        info="Penalize repeated tokens",
    )
]

examples=[["Estoy planeando unas vacaciones en Japón. ¿Puedes sugerir un itinerario de una semana que incluya lugares de visita obligada y cocinas locales para probar?", None, None, None, None, None, ],
          ["¿Puedes escribir una historia corta sobre un detective que viaja en el tiempo y resuelve misterios históricos?", None, None, None, None, None,],
          ["Estoy tratando de aprender francés. ¿Puedes proporcionar algunas frases comunes que serían útiles para un principiante, junto con sus pronunciaciones?", None, None, None, None, None,],
          ["Tengo pollo, arroz y pimientos morrones en mi cocina. ¿Puedes sugerir una receta fácil que pueda hacer con estos ingredientes?", None, None, None, None, None,],
          ["¿Puede explicar cómo funciona el algoritmo QuickSort y proporcionar una implementación de Python?", None, None, None, None, None,],
          ["¿Cuáles son algunas de las características únicas de Rust que lo hacen destacar en comparación con otros lenguajes de programación de sistemas como C++?", None, None, None, None, None,],
         ]

gr.ChatInterface(
    fn=generate,
    chatbot=gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True, likeable=True, layout="panel"),
    additional_inputs=additional_inputs,
    title="Mixtral 46.7B Fines didácticos ",
    examples=examples,
    concurrency_limit=20,
).launch(show_api=False)