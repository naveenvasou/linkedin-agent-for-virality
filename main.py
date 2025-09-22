import gradio as gr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyCnHlsIB-xBfouiUJHcA8dYYg4XMAdNOw0"


def summarizer_agent(model_id: str, blog_text: str):
    logs = ""
    summary = ""
    """
    Summarizes the input blog text by:
    - Extracting all key findings and exciting elements.
    - Preserving emotions like excitement, curiosity, and hook elements.
    - Creating a concise, punchy summary suitable for further LinkedIn post generation.
    """
    if not blog_text.strip():
        return "No text provided."
    
    system_template = """
        You are a professional content summarizer and writer.
        Your task is to summarize a blog while faithfully preserving the content.
        Ensure that:
        1. Every key finding, highlight, interesting fact, and most important/serious insight is captured.
        2. Elements that are attention-grabbing, engaging, or emotionally significant are preserved.
        3. The summary is concise but does NOT remove or dilute any crucial information.
        4. The original style and tone of the blog is maintained; do not alter the voice.
        5. Focus on accurately reflecting the content.
        """
    logs += "Started Creating Prompt Templates\n"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template("{blog_text}")
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    logs += "Created Prompt Templates\n"
    yield logs, summary
    try:
        llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.5,
        max_output_tokens=500,
        streaming=True,               
        callback_manager=callback_manager
        )
        logs += "Initialized LLM\n"
        yield logs, summary
        
        summarizer_chain = LLMChain(
            llm=llm,
            prompt=chat_prompt,
            verbose=True
        )
        
        logs+= "Initialized LLM Chain\n"
        yield logs, summary
        
        summary = summarizer_chain.run(blog_text=blog_text)
        logs += "Generated Summary\n"
        yield logs, summary
    
    except Exception as e:
        return f"Error: {str(e)}" , ""


with gr.Blocks() as demo:
    gr.Markdown("## 🚀 LinkedIn Post Generator\nPaste any text below and convert it into a professional LinkedIn post.")
    
    with gr.Row():
        input_text = gr.Textbox(
            label="Input Text (Paste blog/article/paragraph)",
            lines=10,
            placeholder="Paste your content here...",
        )
        output_text = gr.Textbox(
        label="Generated LinkedIn Post",
        lines=10,
        )
    model_id = gr.Textbox(label="Model ID", value="gemini-2.5-flash-lite")
    logs_box = gr.Textbox(label="Agent Logs", lines=5, interactive=False)
    
    
    generate_btn = gr.Button("Generate Post")
    
    generate_btn.click(
        fn=summarizer_agent,
        inputs=[model_id, input_text],
        outputs=[logs_box, output_text],
    )


demo.launch()