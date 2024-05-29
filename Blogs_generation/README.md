# Blog Generation Project

This project is a simple Streamlit web application that generates blog posts based on user input. It uses the Hugging Face `transformers` library to fine-tune and generate text using pre-trained language models.

## Features

- User can input a topic for the blog post.
- User can specify the desired number of words for the blog post.
- User can select the target audience for the blog post.
- The application generates a blog post based on the user input.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/blog-generation-project.git
cd blog-generation-project

2. Install the required dependencies:

pip install -r requirements.txt

## Usage
To run the application, execute the following command:

streamlit run app.py

## Dependencies
Streamlit
Hugging Face transformers

## Models Used
The application uses the following pre-trained language models for text generation:

openai-community/gpt2 from Hugging Face Transformers Hub.
You can use other model with less parameters.

## Note:- 
project requires high configration GPU to run smoothly.




