import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("C:\easy_5epochs")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Define function to generate summary
def generate_summary(article_text):
    # Tokenize input text
    inputs = tokenizer.encode("summarize: " + article_text, return_tensors="pt", max_length=512, truncation=True)
    # Generate summary
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    # Decode and return summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit app
def main():
    st.title("Text Summarization Using Curriculum RL and T5 PLM")

    # Text area for user input
    article_text = st.text_area("Enter the article text:")

    # Button to generate summary
    if st.button("Generate Summary"):
        if article_text:
            summary = generate_summary(article_text)
            # Display the summary
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text to summarize.")

if __name__ == "__main__":
    main()
