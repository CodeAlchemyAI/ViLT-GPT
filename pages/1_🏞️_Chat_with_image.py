from dotenv import load_dotenv
import os
import streamlit as st
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.chat_models import ChatOpenAI
from io import BytesIO

load_dotenv()

llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
prompt = PromptTemplate(
    input_variables=["question", "elements"],
    template="""You are a helpful assistant that can answer question related to an image. You have the ability to see the image and answer questions about it. 
    I will give you a question and element about the image and you will answer the question.
        \n\n
        #Question: {question}
        #Elements: {elements}
        \n\n
        Your structured response:""",
    )

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    return model, processor

def process_query(image, query):
    model, processor = load_model()
    encoding = processor(image, query, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, elements=model.config.id2label[idx])
    return response

def convert_png_to_jpg(image):
    rgb_image = image.convert('RGB')
    byte_arr = BytesIO()
    rgb_image.save(byte_arr, format='JPEG')
    byte_arr.seek(0)
    return Image.open(byte_arr)


def app():
    st.title("Chat with your IMAGE 🏞️")
    # Sidebar contents
    with st.sidebar:
        st.title('About')
        st.markdown('''
        This app is built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models)
        - [ViLT](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
        ''')
        add_vertical_space(5)
        st.write('Made by [Nicolas tch](https://twitter.com/nicolas_tch)')
        st.write('Repository [Github](https://github.com/CodeAlchemyAI/ViLT-GPT)')

    uploaded_file = st.file_uploader('Upload your IMAGE', type=['png', 'jpeg', 'jpg'], key="imageUploader")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # ViLT model only supports JPG images
        if image.format == 'PNG':
            image = convert_png_to_jpg(image)

        st.image(image, caption='Uploaded Image.', width=300)
        
        cancel_button = st.button('Cancel')
        query = st.text_input('Ask a question to the IMAGE')

        if query:
            with st.spinner('Processing...'):
                answer = process_query(image, query)
                st.write(answer)
          
        if cancel_button:
            st.stop()
            

app()