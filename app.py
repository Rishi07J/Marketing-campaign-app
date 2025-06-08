import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="AI Marketing Assistant",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .stSelectbox > div {
        font-size: 16px;
    }
    .big-font {
        font-size:22px !important;
        font-weight: 600;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Header section
st.markdown("<h1 style='text-align: center; color: #333;'>🧠 AI Marketing Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Craft personalized marketing content based on age & tone 🎯</p>", unsafe_allow_html=True)
st.markdown("---")

# Input form
form_input = st.text_area('✍️ Enter your prompt:', height=200, placeholder="E.g., Describe our new product launch...")

tasktype_option = st.selectbox(
    '⚙️ What would you like to generate?',
    ('Write a sales copy', 'Create a tweet', 'Write a product description'),
    key=1
)

age_option = st.selectbox(
    '🎯 Audience Tone:',
    ('Kid', 'Adult', 'Senior Citizen'),
    key=2
)

submit = st.button("🚀 Generate")

# Function to generate LLM response
def getLLMResponse(query, age_option, tasktype_option):
    llm = ChatGroq(
        temperature=0.9,
        model_name="llama3-70b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # Example data based on age group
    if age_option == "Kid":
        examples = [
            {"query": "What is a mobile?", "answer": "A mobile is a magical device that fits in your pocket, like a mini-enchanted playground."},
            {"query": "What are your dreams?", "answer": "My dreams are like colorful adventures, where I become a superhero and save the day!"},
            {"query": "What are your ambitions?", "answer": "I want to be a cookie baker and a blanket fort builder!"},
            {"query": "What happens when you get sick?", "answer": "It's like a sneaky monster visits. But with cuddles, I bounce back!"},
            {"query": "How much do you love your dad?", "answer": "To the moon and back with unicorns on top!"},
            {"query": "Tell me about your friend?", "answer": "My friend is like a sunshine rainbow. We laugh and play every day!"},
            {"query": "What math means to you?", "answer": "Math is like a puzzle that makes my brain sparkle!"},
            {"query": "What is your fear?", "answer": "Thunderstorms and monsters under the bed—but my teddy keeps me safe!"}
        ]
    elif age_option == "Adult":
        examples = [
            {"query": "What is a mobile?", "answer": "A mobile is a portable communication device that allows calls, messaging, apps, and internet access."},
            {"query": "What are your dreams?", "answer": "To explore new ideas, solve problems, and make the world better."},
            {"query": "What are your ambitions?", "answer": "To constantly learn, grow, and create meaningful impact."},
            {"query": "What happens when you get sick?", "answer": "I feel drained, but rest and care help me recover and reflect on health."},
            {"query": "Tell me about your friend?", "answer": "They’re supportive, kind, and a source of joy in my life."},
            {"query": "What math means to you?", "answer": "A universal tool for logic, analysis, and beauty in patterns."},
            {"query": "What is your fear?", "answer": "Missing potential—but fear motivates me to improve and persist."}
        ]
    elif age_option == "Senior Citizen":
        examples = [
            {"query": "What is a mobile?", "answer": "A device that connects people—it’s amazing how far we’ve come."},
            {"query": "What are your dreams?", "answer": "For my grandkids to live happily and do good in the world."},
            {"query": "What happens when you get sick?", "answer": "I slow down, take care, and appreciate life even more."},
            {"query": "How much do you love your dad?", "answer": "Even though he’s gone, his love guides me every day."},
            {"query": "Tell me about your friend?", "answer": "A friend is a lifelong treasure with shared memories and trust."},
            {"query": "What is your fear?", "answer": "Being alone—but love and connections ease that worry."}
        ]

    example_template = """
    Question: {query}
    Response: {answer}
    """

    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    prefix = """You are a {template_ageoption}, and {template_tasktype_option}:
Here are some examples:"""

    suffix = """
Question: {template_userInput}
Response:"""

    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt,
        max_length=200
    )

    new_prompt_template = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["template_userInput", "template_ageoption", "template_tasktype_option"],
        example_separator="\n"
    )

    final_prompt = new_prompt_template.format(
        template_userInput=query,
        template_ageoption=age_option,
        template_tasktype_option=tasktype_option
    )

    response = llm.invoke(final_prompt)
    return response

# Trigger generation
if submit:
    if not form_input.strip():
        st.error("⚠️ Please enter some prompt text to generate content.")
    else:
        with st.spinner("Generating your personalized response... 💡"):
            result = getLLMResponse(form_input, age_option, tasktype_option)
            st.markdown("### ✨ Generated Response:")
            st.success(result.content if hasattr(result, "content") else result)
