import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_result(input_text, no_words, blog_style):
    # Load model and tokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    
    # Generate input prompt
    input_prompt = f"Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words."
    
    # Tokenize input prompt
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
    
    # Generate blog
    output_ids = model.generate(input_ids, max_length=int(no_words), num_return_sequences=1)
    
    # Decode generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text

st.set_page_config(page_title="Generate Blogs", page_icon='🤖', layout='centered', initial_sidebar_state='collapsed')
st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")
no_words = st.text_input("Number of Words")
blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)
submit = st.button("Generate")

if submit:
    try:
        with st.spinner("Generating..."):
            generated_result = generate_result(input_text, no_words, blog_style)
            st.write("Generated Blog:")
            st.write(generated_result)
    except Exception as e:
        st.error(f"An error occurred: {e}")
