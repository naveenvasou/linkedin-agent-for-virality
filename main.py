import gradio as gr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import json

os.environ["GOOGLE_API_KEY"] = "AIzaSyCnHlsIB-xBfouiUJHcA8dYYg4XMAdNOw0"

def generate_hooks_endings(content, num_hooks=10, num_endings=5):
    
    hook_template  = """
        You are a professional content creator specializing in viral LinkedIn posts for educational AI/Tech blogs.
        Your task: Generate {num_hooks} short, punchy, surprising hook variations for this content:

        {content}

        Here are possible insights you can use:
        1. Hook matters most (compulsory) - Use short, punchy, surprising first lines (e.g., <give an example>). - Sometimes even a one-line hot take works as the hook. 
        2. Headline-style exaggeration for the hook - Overstate (without lying) to grab attention. - Examples: - “Google just killed coding as we know it.” - “OpenAI quietly dropped something HUGE yesterday…” 
        3. Present as “urgent news” - Use time markers (“just dropped”, “yesterday”, “in the last 24h”, "BREAKING"). - Creates FOMO → people want to be “in the know.” 
        4. Use contrasts - Old world vs. new world framing. - “Last year, this was impossible. Today, it’s one click.”
        
        Your task:
            - Dynamically choose which possible insights are most suitable for this content.
            - Apply only the insights that will improve engagement and relevance.
            - Explain briefly which insights you chose for each hook in the output

        Also, **score each hook from 1 to 10** based on its potential engagement.  
        Return output in JSON format:  
        [
        {
            "hook": "Hook text here",
            "score": 9,
            "applied_insights": ["Hook matters most", "Use contrasts"]
        },
        ...
        ]
    """
    ending_template = """
        You are a professional content creator specializing in viral LinkedIn posts for educational AI/Tech blogs.
        Your task: Generate {num_endings} ending variations for this content:

        {content}

        Here are possible insights you can use:
        1. Actionable takeaways (compulsory) - Always end with a simple insight, framework, or recommendation. - Position yourself as a guide, not just a reporter. 
        2. Personal credibility - Subtly include personal expertise (“As someone building AI tools…”). - Readers trust posts where the writer is positioned as a practitioner. 
        3. Call-to-engagement - End with a simple, open-ended question to spark comments. - Example: “Do you see this as a threat or opportunity?” 
        4. End with impact or open question - Tie it back to readers’ jobs, businesses, or daily life. - Example: “If this is real, what happens to [industry/job]?”
        
        Your task:
        - Dynamically choose which insights are most suitable for this content.
        - Apply only the insights that will improve engagement and relevance.
        - Explain briefly which insights you chose for each ending in the output.
        
        Also, **score each ending from 1 to 10** based on its potential engagement.  
        Return output in JSON format:  
        [
        {"ending": "Ending text here", 
        "score": 8,
        "applied_insights": ["Hook matters most", "Use contrasts"]},
        ...
        ]      
    """
    hook_prompt = PromptTemplate(input_variables=["content", "num_hooks"], template=hook_template)
    hook_chain = LLMChain(llm=llm, prompt=hook_prompt)
    
    ending_prompt = PromptTemplate(input_variables=["content", "num_endings"], template=ending_template)
    ending_chain = LLMChain(llm=llm, prompt=ending_prompt)
    
    hook_response = hook_chain.run(content=content, num_hooks=num_hooks)
    ending_response = ending_chain.run(content=content, num_endings=num_endings)
    
    # Parse JSON
    hooks = json.loads(hook_response)
    endings = json.loads(ending_response)
    
    return hooks, endings


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