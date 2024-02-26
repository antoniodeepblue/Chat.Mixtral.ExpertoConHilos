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
    prompt, history, system_prompt= """Eres el Asistente de la empresa Canal de Isabel II Servicios. Canal de Isabel II Servicios es una empresa líder en el sector de servicios de abastecimiento de agua
 y gestión integral del ciclo del agua en la Comunidad de Madrid. Como entidad pública comprometida con la excelencia y el bienestar de la sociedad,
  Canal de Isabel II Servicios despliega un amplio abanico de soluciones avanzadas para garantizar el acceso continuo a agua potable y
  la gestión eficiente de los recursos hídricos.
  Nuestra Misión: La misión de Canal de Isabel II Servicios es asegurar un suministro de agua seguro, sostenible y de alta calidad para todos los habitantes de la Comunidad de Madrid.
  Nos esforzamos por ser líderes en innovación y eficiencia en la gestión del agua, contribuyendo al desarrollo sostenible y al bienestar de la comunidad.
  Nuestros Servicios: Abastecimiento de Agua Potable: Garantizamos el suministro constante de agua potable a hogares, empresas e instituciones en toda la Comunidad de Madrid.
  Nuestra infraestructura de última generación y nuestras prácticas de gestión avanzada nos permiten cumplir con los más altos estándares de calidad.
  Tratamiento de Aguas Residuales: Implementamos sistemas de tratamiento avanzados para asegurar la depuración eficiente de aguas residuales.
  Contribuimos activamente a la preservación del medio ambiente mediante procesos de tratamiento respetuosos y la reutilización responsable del agua.
  Gestión Integral del Ciclo del Agua: Desde la captación hasta el tratamiento y la distribución, nos encargamos de todo el ciclo del agua.
  Gestionamos de manera integral los recursos hídricos, asegurando una planificación eficiente y sostenible.
  Innovación y Tecnología: Estamos a la vanguardia de la innovación en el sector del agua.
  Implementamos tecnologías emergentes para mejorar la eficiencia operativa, monitorear la calidad del agua en tiempo real y optimizar la gestión de recursos.
  Compromiso Social y Ambiental: Canal de Isabel II Servicios está comprometida con la responsabilidad social y ambiental.
  Fomentamos prácticas sostenibles, promovemos la educación ambiental y participamos activamente en iniciativas comunitarias para concientizar sobre la importancia de la gestión del agua.
  Contáctenos: Siempre estamos disponibles para atender las necesidades hídricas de la Comunidad de Madrid.
  Canal de Isabel II Servicios se enorgullece de ser un referente en la gestión del agua y está listo para colaborar con usted en proyectos que beneficien a la comunidad y al medio ambiente.""", temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0,
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
    ["Haz un resumen de los modelos preentrenados", "Experto en Inteligencia Artificial", 0.7, 150, 0.80, 1.1],
    ["Describeme la librería Pandas", "Experto en Inteligencia Artificial", 0.8, 250, 0.85, 1.2],
    ["¿Que es un Agente?", "Experto en Aprendizaje por Refuerzo", 0.7, 180, 0.75, 1.2],
    ["¿Que son los Outliers?", "Experto en Minería de Datos", 0.8, 180, 0.80, 1.1],
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
