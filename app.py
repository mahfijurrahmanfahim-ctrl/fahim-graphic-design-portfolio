
import streamlit as st
from transformers import pipeline
from datetime import datetime

# ==========================================
# FAHIM CUSTOM AI AGENT BUILDER PRO
# ==========================================

st.set_page_config(
    page_title="FAHIM CUSTOM AI AGENT BUILDER",
    page_icon="🤖",
    layout="wide"
)

MODEL_ID = "google/flan-t5-small"
MAX_INPUT_TOKENS = 512

# ==========================================
# DEFAULT PROFILE
# ==========================================

FAHIM_MASTER_PROFILE = """
Name: Mahfijur Rahman Fahim
Professional Brand: FAHIM GRAPHIC DESIGN
Location: Portugal

Profession:
Graphic Designer, AI-Assisted Creative Designer,
Image Editor and Video Content Creator.

Services:
Restaurant Menu Design, Social Media Design,
Logo Design, Flyers, Posters, Photo Editing,
AI Image Generation, Promotional Video Creation.

Skills:
Canva, ChatGPT, Google Gemini, CapCut,
Google Flow, AI Image Generation, Graphic Design,
Photo Editing and Video Editing.

Design Style:
Luxury Black and Gold, realistic imagery,
professional lighting, clean typography.

Business Tagline:
Creative Design. Professional Results.

Portfolio:
https://mahfijurrahmanfahim-ctrl.github.io/fahim-graphic-design-portfolio/

Fiverr Menu Design:
https://www.fiverr.com/s/X00EVme

Fiverr Social Media Design:
https://www.fiverr.com/s/qbbQr6y

TikTok:
@m.r.fahim

Business Offer:
First Menu Sample: 100% Free.
New Business Launch Pack: £59.

Target Clients:
Restaurants, Cafes, Gyms, Clothing Brands
and Small Businesses.
"""

# ==========================================
# LOAD MODEL ONLY WHEN NEEDED
# ==========================================

@st.cache_resource(show_spinner=False)
def load_ai():
    return pipeline(
        task="text2text-generation",
        model=MODEL_ID,
        device=-1
    )

# ==========================================
# TOKEN LIMIT HELPER
# ==========================================

def shorten_text(text, tokenizer, max_tokens):
    tokens = tokenizer.encode(
        text,
        truncation=True,
        max_length=max_tokens
    )
    return tokenizer.decode(
        tokens,
        skip_special_tokens=True
    )

# ==========================================
# AI RESPONSE GENERATOR
# ==========================================

def generate_answer(
    profile,
    role,
    assistant_name,
    language,
    tone,
    instructions,
    question,
    history
):
    ai = load_ai()
    tokenizer = ai.tokenizer

    safe_profile = shorten_text(
        profile, tokenizer, 220
    )

    safe_instructions = shorten_text(
        instructions, tokenizer, 55
    )

    safe_question = shorten_text(
        question, tokenizer, 100
    )

    previous_messages = ""

    for item in history[-4:]:
        previous_messages += (
            f"\n{item['role']}: "
            f"{shorten_text(item['content'], tokenizer, 35)}"
        )

    prompt = f"""
You are {assistant_name}.

Your role: {role}
Answer language: {language}
Communication style: {tone}

USER PROFILE:
{safe_profile}

ADDITIONAL INSTRUCTIONS:
{safe_instructions}

RECENT CONVERSATION:
{previous_messages}

CURRENT QUESTION:
{safe_question}

Give a useful, accurate and direct answer.
Do not invent facts that are not provided.
If information is missing, say so.

ANSWER:
"""

    result = ai(
        prompt,
        max_new_tokens=120,
        do_sample=False,
        truncation=True
    )

    return result[0]["generated_text"].strip()

# ==========================================
# HEADER
# ==========================================

st.markdown(
    "# 🤖 FAHIM CUSTOM AI AGENT BUILDER PRO"
)

st.markdown(
    "### Build your own AI assistant using your own profile."
)

st.caption(
    "Create assistants for personal use, business, "
    "marketing, education and professional work."
)

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================

st.sidebar.title("⚙️ Agent Configuration")

assistant_name = st.sidebar.text_input(
    "Assistant Name",
    value="My Personal AI"
)

role = st.sidebar.text_input(
    "Assistant Role",
    value="Personal and Professional Assistant"
)

language = st.sidebar.selectbox(
    "Response Language",
    [
        "English",
        "Bengali",
        "Portuguese",
        "Spanish",
        "Hindi",
        "French"
    ]
)

tone = st.sidebar.selectbox(
    "Communication Style",
    [
        "Professional",
        "Friendly",
        "Simple and Clear",
        "Creative",
        "Formal"
    ]
)

instructions = st.sidebar.text_area(
    "Custom Instructions",
    placeholder="Tell your AI how it should behave...",
    height=120
)

st.sidebar.divider()

st.sidebar.markdown("### 🌐 Example Portfolio")

st.sidebar.markdown(
    "[Fahim Graphic Design](https://mahfijurrahmanfahim-ctrl.github.io/fahim-graphic-design-portfolio/)"
)

st.sidebar.caption(
    "Model: FLAN-T5 Small | Free local inference"
)

# ==========================================
# PROFILE INPUT
# ==========================================

st.subheader("📁 Your Personal Knowledge Base")

st.info(
    "Paste your own profile below. "
    "Leave it empty to use the Fahim example profile."
)

profile_data = st.text_area(
    "Personal / Business Profile",
    value="",
    height=240,
    placeholder=(
        "Name:\n"
        "Profession:\n"
        "Business:\n"
        "Skills:\n"
        "Services:\n"
        "Website:\n"
        "Contact information:"
    )
)

active_profile = (
    profile_data.strip()
    if profile_data.strip()
    else FAHIM_MASTER_PROFILE
)

# ==========================================
# CHAT MEMORY
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 Chat With Your AI Agent")

with col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Display previous messages

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# CHAT INPUT
# ==========================================

question = st.chat_input(
    "Ask your personal AI assistant..."
)

if question:

    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "🤖 Your AI assistant is thinking..."
        ):
            try:
                response = generate_answer(
                    profile=active_profile,
                    role=role,
                    assistant_name=assistant_name,
                    language=language,
                    tone=tone,
                    instructions=instructions,
                    question=question,
                    history=st.session_state.chat_history[:-1]
                )

                st.markdown(response)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception as error:
                st.error(
                    f"AI Error: {str(error)}"
                )

# ==========================================
# DOWNLOAD CONVERSATION
# ==========================================

if st.session_state.chat_history:

    conversation = (
        f"{assistant_name}\n"
        f"Exported: {datetime.now().isoformat()}\n"
        f"{'=' * 40}\n\n"
    )

    for message in st.session_state.chat_history:
        conversation += (
            f"{message['role'].upper()}:\n"
            f"{message['content']}\n\n"
        )

    st.download_button(
        "📥 Download Conversation",
        data=conversation.encode("utf-8"),
        file_name="my_ai_conversation.txt",
        mime="text/plain"
    )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown("""
**FAHIM CUSTOM AI AGENT BUILDER PRO**

Creative Design. Professional Results.

Built by FAHIM GRAPHIC DESIGN.
""")

st.caption(
    "This application uses a free open-source model. "
    "It does not train or permanently remember user profiles."
)
