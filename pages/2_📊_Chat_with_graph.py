from dotenv import load_dotenv
import streamlit as st
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from PIL import Image
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from streamlit_extras.add_vertical_space import add_vertical_space

load_dotenv()

llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
prompt = PromptTemplate(
    input_variables=["question", "elements"],
    template="""You are a helpful assistant that can answer question related to a graph image. You have the ability to see the graph image and answer questions about it. 
    I will give you a question and a the associated data table of the grah and you will answer the question.
        \n\n
        #Question: {question}
        #Elements: {elements}
        \n\n
        Your structured response:""",
    )

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
    processor = Pix2StructProcessor.from_pretrained('google/deplot')
    return model, processor


def process_query(data_table, query):
    prompt = PromptTemplate(
    input_variables=["question", "elements"],
    template="""You are a helpful assistant capable of answering questions related to graph images.
     You possess the ability to view the graph image and respond to inquiries about it. 
     I will provide you with a question and the associated data table of the graph, and you will answer the question
        \n\n
        #Question: {question}
        #Elements: {elements}
        \n\n
        Your structured response:""",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, elements=data_table)
    return response

@st.cache_data(show_spinner="Processing image...")
def generate_table(uploaded_file):
    image = Image.open(uploaded_file)
    model, processor = load_model()
    inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
    predictions = model.generate(**inputs, max_new_tokens=512)
    return processor.decode(predictions[0], skip_special_tokens=True)

def app():
    st.title("Chat with your GRAPH 📊")

    # Sidebar contents
    with st.sidebar:
        st.title('About')
        st.markdown('''
        This app is built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models)
        - [Deplot](https://huggingface.co/google/deplot)
        ''')
        add_vertical_space(5)
        st.write('Made by [Nicolas tch](https://twitter.com/nicolas_tch)')
        st.write('Repository [Github](https://github.com/CodeAlchemyAI/ViLT-GPT)')

    uploaded_file = st.file_uploader('Upload a chart image.', type=['png', 'jpeg', 'jpg'], key="graphUploader")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        data_table = generate_table(uploaded_file)

        st.image(image, caption='Uploaded Image.')
        
        query = st.text_input('Ask a question to the IMAGE')

        if query:
            with st.spinner('Processing...'):
                answer = process_query(data_table, query)
                st.write(answer)

        if st.button('Cancel'):
            st.write('Cancelled by the user.')


app()