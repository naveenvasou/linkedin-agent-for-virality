import gradio as gr
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any
from langchain.output_parsers import PydanticOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import json

os.environ["GOOGLE_API_KEY"] = "AIzaSyCnHlsIB-xBfouiUJHcA8dYYg4XMAdNOw0"
from fastapi import FastAPI
app = FastAPI()

MAXS = {"attention": 30, "relevance": 25, "curiosity": 25, "clarity": 10, "novelty": 10}
toggle_visibility = gr.update(visible=True)
toggle_invisibility = gr.update(visible=False)


with open("hook_templates.json", "r", encoding="utf-8") as f:
    hook_templates = json.load(f)

with open("ending_templates.json", "r", encoding="utf-8") as f:
    ending_templates = json.load(f)
    
with open("core_body_templates.json", "r", encoding="utf-8") as f:
    core_body_templates = json.load(f)

def handle_exception(e):
    return f"Error: {str(e)}"

class Hook(BaseModel):
    hook: str = Field(..., description="The hook text")
    score: int = Field(..., description="LLM's confidence score for this hook")
    applied_insights: List[str] = Field(default_factory=list, description="Insights applied to generate this hook") 

class Ending(BaseModel):
    ending: str = Field(..., description="The ending text")
    score: int = Field(..., description="LLM's confidence score for this ending")
    applied_insights: List[str] = Field(default_factory=list, description="Insights applied to generate this ending")

class Hooks(BaseModel):
    hooks: List[Hook]
    
class Endings(BaseModel):
    endings: List[Ending]
  
class AllScores(BaseModel):
    attention_grabbing_power: str = Field(..., description="Score for attention-grabbing power (0-30)")
    relevance_to_content: str = Field(..., description="Score for relevance to content (0-25)")
    curiosity_urgency: str = Field(..., description="Score for curiosity/urgency (0-25)")
    clarity_readability: str = Field(..., description="Score for clarity & readability (0-10)")
    novelty_uniqueness: str = Field(..., description="Score for novelty/uniqueness (0-10)")
    
class DraftItem(BaseModel):
    text: str = Field(..., description="The generated LinkedIn post draft")
    score: int = Field(..., ge=0, le=100, description="Score from 1-100 representing virality and credibility")
    subscores: AllScores = Field(
        ...,
        description="Detailed subscores: attention(0-30), relevance(0-25), curiosity(0-25), clarity(0-10), novelty(0-10)"
    )
    used_hook: str = Field(..., description="Exact hook string used")
    used_ending: str = Field(..., description="Exact ending string used")
    applied_templates: List[str] = Field(default_factory=list, description="Which templates from the chunk inspired this draft")
    rationale: str = Field(None, description="1-2 line justification for this draft's potential")
   
class DraftBatch(BaseModel):
    drafts: List[DraftItem]
       
def generate_hooks_endings(model_id, content, logs):
    num_hooks=10
    num_endings=5
    
    hook_categories = list(hook_templates.keys())
    chunk_size = 5
    hook_template_chunks = [hook_categories[i:i + chunk_size] for i in range(0, len(hook_categories), chunk_size)]
    
    llm = ChatGoogleGenerativeAI(
        model=model_id,
        temperature=0.5,
        max_output_tokens=1000             
        )

    hook_parser = PydanticOutputParser(pydantic_object=Hooks)
    ending_parser = PydanticOutputParser(pydantic_object=Endings)
    
    logs += "Generating Hooks in chunks\n"
    all_hooks = []
    
    for i, chunk in enumerate(hook_template_chunks):

        templates_text = ""
        for cat in chunk:
            for tpl in hook_templates[cat]:
                templates_text += f"- Template: {tpl['template']}\n  Example: {tpl['example']}\n"

        hook_prompt_text = """
            You are a professional content creator specializing in viral LinkedIn posts for educational AI/Tech blogs.
            Your task: Generate {num_hooks} short, punchy, surprising hook variations for the content below. 
            {content}
            
            Here are insights/ideas which worked for other viral posts: We can make use of these insights for ur idea generation.
            1. Curiosity-driven: Hooks must grab attention immediately and make the reader want to read more.
            - Use short, punchy, surprising first lines.
            - Questions, bold statements, or mini “teasers” are encouraged.
            2. Concrete facts with real-world impact: Many a times , hooks that go viral include a fact, stat, or insight from the content.
            3. Optional mini-narrative or scenario: Where possible, embed a micro-story or scenario in 1–2 hooks for higher engagement.
            4. Urgency & FOMO:
            - If using “BREAKING” or time markers (“Yesterday…”, “Just released”, “in the last 24h”), always connect it to a **real consequence, impact, or reader takeaway**.  
            
            Diverse angles: Maximize variety across hooks.
            - Use different strategies: urgent news, headline-style exaggeration, surprising insight, contrast old vs new, social relevance, global trends, micro-narrative, or scenario.
            - Ensure no hook is a copy of another in angle or style.

            Use the following viral hook templates as inspiration (adapt to the content). 

            {templates_text}
            
            Language & Style Guidelines:
                - Use simple, clear, and everyday vocabulary that Indians would understand.
                - Avoid uncommon or  complicated general English words
                - simplify complex words and phrasing while keeping the meaning.

            For each hook, provide:
            - "hook" (string)
            - "score" (integer -100) (Scoring: For each hook, provide a score from 1–100 (100 = highest potential for virality and engagement). 
                calculated based on: 
                1. Attention-grabbing power (max 30) 
                2. Relevance to the content (max 15) 
                3. Curiosity / urgency (max 30) 
                4. Clarity & readability (max 15) 
                5. Novelty / uniqueness (max 10)
                - Add up the points to give the total score)

            Maximize variety and virality. Ensure no hook is a copy of another.
            
            OUTPUT FORMAT:
            {format_instructions}
        """
        hook_prompt = PromptTemplate(
            input_variables=[num_hooks, content], 
            partial_variables={"templates_text": templates_text, "format_instructions": hook_parser.get_format_instructions()}, 
            template=hook_prompt_text)
        hook_chain = hook_prompt | llm | hook_parser
        chunk_hooks = hook_chain.invoke({"content":content, "num_hooks":num_hooks})
        all_hooks.extend(chunk_hooks.hooks)
        logs += "Loading\n"
        yield gr.update(choices=[], visible=False), gr.update(choices=[], visible=False), logs, gr.update(visible=False), gr.update(visible=False)

    logs += "Generated hooks for all chunks\n"
    yield gr.update(choices=[], visible=False), gr.update(choices=[], visible=False), logs, gr.update(visible=False), gr.update(visible=False)
    
    # Deduplicate & select top hooks by score
    unique_hooks = {}
    for h in all_hooks:
        if h.hook not in unique_hooks or h.score > unique_hooks[h.hook].score:
            unique_hooks[h.hook] = h
    sorted_hooks = sorted(unique_hooks.values(), key=lambda x: x.score, reverse=True)
    final_hooks = sorted_hooks[:num_hooks]
    hook_choices = [f"{h.hook} (Score: {h.score})" for h in final_hooks]

    ending_categories = list(ending_templates.keys())
    chunk_size = 5
    ending_template_chunks = [ending_categories[i:i + chunk_size] for i in range(0, len(ending_categories), chunk_size)]
    
    logs += "Generating Endings in chunks\n"
    all_endings = []
    
    for i, chunk in enumerate(ending_template_chunks):
        templates_text = ""
        for cat in chunk:
            for tpl in ending_templates[cat]:
                templates_text += f"- Template: {tpl['template']}\n  Example: {tpl['example']}\n"
    
        ending_template = """
            You are a professional content creator specializing in viral LinkedIn posts for educational AI/Tech blogs.
            Your task: Generate {num_endings} ending variations for this content:

            {content}

            Here are possible insights you can use:
            1. Actionable takeaways (compulsory) - Always end with a simple insight, framework, or recommendation. - Position yourself as a guide, not just a reporter. 
            2. Personal credibility - Subtly include personal expertise (“As someone building AI tools…”). - Readers trust posts where the writer is positioned as a practitioner. 
            3. Call-to-engagement - End with a simple, open-ended question to spark comments. - Example: “Do you see this as a threat or opportunity?” 
            4. End with impact or open question - Tie it back to readers’ jobs, businesses, or daily life. - Example: “If this is real, what happens to [industry/job]?”
            
            
            Diversity & Engagement Rules:
            - Apply only the insights that will improve engagement and relevance. Ensure at least 2–3 different insight types (Actionable, Role-targeted, Credibility, Impact, Call-to-engagement, Safety, Next-step) appear across outputs.
            - Maximize variety; do not repeat the same ending idea.
            - Make endings scroll-stopping and LinkedIn-friendly.
           
            Use the following viral ending templates as inspiration (adapt to the content). 
            {templates_text}
            
            Language & Style Guidelines:
            - Use simple, clear, everyday words that Indians will easily understand.
            - Prefer short sentences for LinkedIn readability.
            - Add a question, next step, or actionable instruction in every ending unless safety/ethics requires a cautionary tone.


            For each ending, provide:
                - "ending" (string)
                - "score" (integer -100) (Scoring: Assign each ending a score from 1–100 (100 = highest potential for engagement) 
                    calculated based on: 
                    1. Actionability / Practical value (max 30)
                    2. Relevance to content (max 15)
                    3. Engagement potential (max 35)
                    4. Clarity & readability (max 10)
                    5. Novelty / memorability (max 10)
                    - Add up the points to give the total score)
            Maximize variety and virality. Ensure no ending is a copy of another.
            
            Output Format:
            {format_instructions}   
        """    
        ending_prompt = PromptTemplate(
            input_variables=["content", "num_endings"], 
            partial_variables={"templates_text":templates_text, "format_instructions": ending_parser.get_format_instructions()},  
            template=ending_template)
        ending_chain = ending_prompt | llm | ending_parser
        chunk_endings = ending_chain.invoke({"content":content, "num_endings":num_endings})
        all_endings.extend(chunk_endings.endings)
        logs += "Loading\n"
        yield gr.update(choices=[], visible=False), gr.update(choices=[], visible=False), logs, gr.update(visible=False), gr.update(visible=False)
        
    logs += "Generated Endings for all chunks\n"
    yield gr.update(choices=[], visible=False), gr.update(choices=[], visible=False), logs, gr.update(visible=False), gr.update(visible=False)
    
    unique_endings = {}
    for h in all_endings:
        if h.ending not in unique_endings or h.score > unique_endings[h.ending].score:
            unique_endings[h.ending] = h
    sorted_endings = sorted(unique_endings.values(), key=lambda x: x.score, reverse=True)
    final_endings = sorted_endings[:num_endings]
    ending_choices = [f"{h.ending} (Score: {h.score})" for h in final_endings]
    yield gr.update(choices=hook_choices, visible=True, value=[]), gr.update(choices=ending_choices, visible=True, value=[]), logs, gr.update(visible=True), gr.update(visible=False)

def summarizer_agent(model_id: str, blog_text: str, logs: str):
    
    summary = ""
    
    """
    Summarizes the input blog text by:
    - Extracting all key findings and exciting elements.
    - Preserving emotions like excitement, curiosity, and hook elements.
    - Creating a concise, punchy summary suitable for further LinkedIn post generation.
    """
    if not blog_text.strip():
        return "No text provided.", gr.update(visible=True), gr.update(visible=False), logs + "No text provided.\n", gr.update(visible=True), summary
    
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
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template("{blog_text}")
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    logs += "Created Prompt Templates\n"
    yield gr.update(value=summary, visible=True), gr.update(visible=True), gr.update(visible=False), logs, gr.update(visible=True), summary
    
    try:
        llm = ChatGoogleGenerativeAI(
        model=model_id,
        temperature=0.5,
        max_output_tokens=500,
        )
        summarizer_chain = LLMChain(llm=llm, prompt=chat_prompt, verbose=True)
    
        logs+= "Generating Summary\n"
        yield summary, gr.update(visible=True), gr.update(visible=False), logs, gr.update(visible=True), summary
        summary = summarizer_chain.run(blog_text=blog_text)
        logs += "Generated Summary\n"

        yield summary, gr.update(visible=False), gr.update(visible=True), logs, gr.update(visible=False), summary
        #summary = summarizer_chain.run(blog_text=blog_text)
        #logs += "Generated Summary\n"
        #yield summary, gr.update(visible=False), gr.update(visible=True), logs
    
    except Exception as e:
        logs += f"Error during summarization: {str(e)}\n"
        return summary, gr.update(visible=True), gr.update(visible=False), logs, gr.update(visible=True), summary

def generate_top3_drafts(
        model_id: str,
        content_summary: str,
        hooks: List[str],
        endings: List[str],
        user_msg: str,
        logs: str,
        drafts_per_chunk: int = 3,
        max_candidates: int = 30
    ):
    """
    Generate draft candidates by iterating over templates_chunks.
    Returns (top_3_list, all_candidates_list).
    Each draft is a dict matching DraftItem fields.
    """
    core_body_templates_list = core_body_templates.get("templates", [])
    chunk_size = 5
    templates_chunks = [core_body_templates_list[i:i + chunk_size] for i in range(0, len(core_body_templates_list), chunk_size)]
    llm = ChatGoogleGenerativeAI(
        model=model_id, 
        temperature=0.8, 
        max_output_tokens=1000)
    BATCH_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", "You are an expert LinkedIn content creator and social media strategist for Tech/AI. Be concise and credible."),
        ("system", "Return valid JSON only, matching the format instructions provided."),
        ("human",
        """
        Content summary:
        {content}

        All hooks (choose from these):
        {hooks}

        All endings (choose from these):
        {endings}

        Templates chunk (inspiration for this batch):
        {templates_chunk}

        User instructions:(ignore if empty)
        {user_msg}

        Task:
        - Generate {batch_size} distinct LinkedIn post drafts inspired by the chunk.
        - Each draft must use one of the provided hooks and one provided ending.
        - For each draft return:
        - text (string)
        - score (int 0-100) [model's total score]
        - subscores: attention(0-30), relevance(0-25), curiosity(0-25), clarity(0-10), novelty(0-10)
        - used_hook (exact string)
        - used_ending (exact string)
        - applied_templates (list of strings)
        - rationale (1-2 sentences)
        - Keep language simple & Indian-audience friendly; keep technical terms intact.
        
        Return the JSON exactly in the format:
        {format_instructions}
        """)
    ])
    # Parser: Pydantic-based to enforce structure
    parser = PydanticOutputParser(pydantic_object=DraftBatch)

    all_candidates: List[DraftItem] = []
    
    hooks_text = "\n".join(hooks)
    endings_text = "\n".join(endings)
    logs += "Generating drafts in chunks\n"
    yield "", "", "", logs, toggle_visibility, toggle_invisibility, []
    for idx, chunk in enumerate(templates_chunks):
        templates_chunk_text = ""
        for t in chunk:
            tpl = t.get("template", "")
            ex = t.get("example", "")
            templates_chunk_text += f"- Idea: {tpl} Example: {ex}\n"
            
        format_instructions = parser.get_format_instructions()
        
        prompt_inputs = {
            "content": content_summary,
            "hooks": hooks_text,
            "endings": endings_text,
            "templates_chunk": templates_chunk_text,
            "user_msg": user_msg or "No user instruction provided.",
            "batch_size": drafts_per_chunk,
            "format_instructions": format_instructions
        }
        
        try:
            chain = BATCH_PROMPT_TEMPLATE | llm | parser
            batch_out: DraftBatch = chain.invoke(prompt_inputs)
            # append valid outputs
            for d in batch_out.drafts:
                # basic validation / clamp score
                all_candidates.append(d)
        except Exception as e:
            pass
        
        if len(all_candidates) >= max_candidates:
            break
        logs += "Loading\n"
        yield "", "", "", logs, toggle_visibility, toggle_invisibility, []
        
    logs += f"Generated {len(all_candidates)} draft candidates\n"
    #yield "", "", "", logs, toggle_visibility, toggle_invisibility, []
    all_drafts =  [c.dict() for c in all_candidates]
    top_k = rerank_candidates(all_drafts, 1, 1, 1, 1, 1, top_k=3)
    yield top_k[0]["text"], top_k[1]["text"], top_k[2]["text"], logs, toggle_visibility, toggle_visibility, all_drafts
    
def rerank_candidates(candidates: List[Dict[str, Any]],
                      w_attention: float, w_relevance: float, w_curiosity: float,
                      w_clarity: float, w_novelty: float,
                      top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Re-rank candidates based on user-provided weights. Return top_k dicts.
    If sum of weights == 0, fall back to original model score ordering.
    """
    weights = {
        "attention": float(w_attention),
        "relevance": float(w_relevance),
        "curiosity": float(w_curiosity),
        "clarity": float(w_clarity),
        "novelty": float(w_novelty)
    }
    total_weight = sum(weights.values())

    ranked = []
    for c in candidates:
        subs = c.get("subscores", {})
        # normalize subscore by its max
        a = int(subs.get("attention_grabbing_power", "0")) / MAXS["attention"]
        r = int(subs.get("relevance_to_content", "0")) / MAXS["relevance"]
        cu = int(subs.get("curiosity_urgency", "0")) / MAXS["curiosity"]
        cl = int(subs.get("clarity_readability", "0")) / MAXS["clarity"]
        n = int(subs.get("novelty_uniqueness", "0")) / MAXS["novelty"]

        if total_weight == 0:
            # fallback: use model's original score
            new_score = c.get("score", 0)
        else:
            weighted = (weights["attention"]*a +
                        weights["relevance"]*r +
                        weights["curiosity"]*cu +
                        weights["clarity"]*cl +
                        weights["novelty"]*n)
            new_score = (weighted / total_weight) * 100.0

        c_copy = c.copy()
        c_copy["weighted_score"] = round(float(new_score), 2)
        ranked.append(c_copy)

    # sort by weighted_score desc, tie-breaker original model score
    ranked_sorted = sorted(ranked, key=lambda x: (x["weighted_score"], x.get("score", 0)), reverse=True)
    return ranked_sorted[:top_k]

def update_drafts_display(drafts_list, w_att, w_rel, w_cur, w_cla, w_nov):
    if not drafts_list:
        return "", "", ""
    
    top_k = rerank_candidates(drafts_list, w_att, w_rel, w_cur, w_cla, w_nov, top_k=3)
    return (
        top_k[0]["text"] if len(top_k) > 0 else "",
        top_k[1]["text"] if len(top_k) > 1 else "", 
        top_k[2]["text"] if len(top_k) > 2 else ""
    )
    
with gr.Blocks() as demo:
    model_id = gr.Textbox(label="Model ID", value="gemini-2.5-flash-lite")
    user_msg = gr.Textbox(label="User Instruction", value=" ", lines=1,placeholder="Write if you have any specific instruction/idea....",)
    summary_state = gr.State("")
    drafts_state = gr.State("")
    with gr.Row():
        input_text = gr.Textbox(
            label="Input Text (Paste blog/article/paragraph)",
            lines=10,
            placeholder="Paste your content here...",
        )
        output_text = gr.Textbox(label="Generated Summary", lines=10, interactive=False, visible=False)
    
    with gr.Row():
        hooks_checkbox = gr.CheckboxGroup(label="Select Hooks", choices=[], visible=False)
        endings_checkbox = gr.CheckboxGroup(label="Select Endings", choices=[], visible=False)
    
    first_drafts_section = gr.Row(visible=False)
    with first_drafts_section:
        gr.Markdown("### Candidate drafts (raw from LLM)")
        draft_1 = gr.Textbox(label="Draft 1", lines=10, interactive=False)
        draft_2 = gr.Textbox(label="Draft 2", lines=10, interactive=False)
        draft_3 = gr.Textbox(label="Draft 3", lines=10, interactive=False)
    
    weights_sliders = gr.Row(visible=False)
    with weights_sliders:
        gr.Markdown("### Re-rank weights (adjust to taste)")
        w_att = gr.Slider(0, 5, value=1, step=0.1, label="Weight - Attention")
        w_rel = gr.Slider(0, 5, value=1, step=0.1, label="Weight - Relevance")
        w_cur = gr.Slider(0, 5, value=1, step=0.1, label="Weight - Curiosity / CTA")
        w_cla = gr.Slider(0, 5, value=1, step=0.1, label="Weight - Clarity")
        w_nov = gr.Slider(0, 5, value=1, step=0.1, label="Weight - Novelty")
        sliders = [w_att, w_rel, w_cur, w_cla, w_nov]
    
    generate_first_drafts_btn = gr.Button("Generate First Drafts", visible=False)
    generate_hooks_endings_btn = gr.Button("Generate Hooks and Endings", visible=False)
    generate_summary_btn = gr.Button("Generate Summary")
    logs_box = gr.Textbox(label="Agent Logs", lines=5, interactive=False)
    
    try:
        
        # Stage 1
        generate_summary_btn.click(
            fn=summarizer_agent,
            inputs=[model_id, input_text, logs_box],
            outputs=[output_text, input_text, generate_hooks_endings_btn, logs_box, generate_summary_btn, summary_state]
        )
        
        generate_hooks_endings_btn.click(
            fn=generate_hooks_endings,
            inputs=[model_id, summary_state, logs_box],
            outputs=[hooks_checkbox, endings_checkbox, logs_box, generate_first_drafts_btn, output_text]
        )
        
        # Stage 3
        generate_first_drafts_btn.click(
            fn=generate_top3_drafts,
            inputs=[model_id, summary_state, hooks_checkbox, endings_checkbox, user_msg, logs_box],
            outputs=[draft_1, draft_2, draft_3, logs_box, first_drafts_section, weights_sliders, drafts_state]
        )
        
        for slider in sliders:
            slider.change(
                fn=update_drafts_display,
                inputs=[drafts_state, w_att, w_rel, w_cur, w_cla, w_nov],
                outputs=[draft_1, draft_2, draft_3]
            )
        
    except Exception as e:
        pass
        
    
app = gr.mount_gradio_app(app, demo, path="/")
#demo.launch(reload=True)