from dotenv import load_dotenv
import os
import streamlit as st
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.llms import OpenAI

load_dotenv()

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

llm = OpenAI(temperature=0.2)
prompt = PromptTemplate(
    input_variables=["question", "elements"],
    template="""Please generate a structured response using the following information:
        \n\n
        #Question: {question}
        #Response: {elements}
        \n\n
        Your structured response:""",
    )

def process_query(image, query):
    encoding = processor(image, query, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, elements=model.config.id2label[idx])
    return response

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    - [ViLT](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
    ''')
    add_vertical_space(5)
    st.write('Made by [Nicolas tch](https://twitter.com/nicolas_tch)')

load_dotenv()


def main():
    st.title("Chat with your IMAGE 💬")

    uploaded_file = st.file_uploader('Upload your IMAGE', type=['png', 'jpeg', 'jpg'])

    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', width=300)
        
        cancel_button = st.button('Cancel')
        query = st.text_input('Ask a question to the IMAGE')

        if query:
            with st.spinner('Processing...'):
                answer = process_query(image, query)
                st.write(answer)
          
        if cancel_button:
            st.stop()
            
if __name__ == "__main__":
    main()
