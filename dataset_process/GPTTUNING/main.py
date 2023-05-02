import tkinter as tk
import openai
import re

# Replace FINE_TUNED_MODEL with the name of your fine-tuned model
model_name = "ada:ft-personal-2023-05-02-07-34-38"


def on_submit():
    # Get the prompt from the input field
    original_prompt = input_field.get()
    
    # Modify the prompt to ask for the genre
    prompt = f"Analyze the following text and provide one relevant genre from the list: [thriller, fantasy, science fiction, history, horror, crime, romance, psychology, sports, travel].\n\nText: \"{original_prompt}\"\n\nGenre:"

    # Make the completion request
    completion = openai.Completion.create(model=model_name, prompt=prompt, max_tokens=100, temperature=0.8)

    print("Completion:", completion)

    # Clear the input field
    input_field.delete(0, "end")

    # Get the completion text from the first choice in the choices list
    text = completion.choices[0]["text"].strip()

    # Extract the genre from the text using a regular expression
    genre = re.findall(r'\b(?:thriller|fantasy|science fiction|history|horror|crime|romance|psychology|sports|travel)\b', text)[:2]

    # Join the genres found in the text (if any) and display them in the result text area
    result_text.config(state="normal")
    result_text.delete("1.0", "end")
    result_text.insert("end", ', '.join(genre))
    result_text.config(state="disabled")

# Create the main window
window = tk.Tk()
window.title("Fine-tuned GPT-3 for Genre Classification")

# Create the input field and submit button
input_field = tk.Entry(window)
submit_button = tk.Button(window, text="Submit", command=on_submit)

# Create the result text area
result_text = tk.Text(window, state="normal", width=80, height=20)

# Add the input field, submit button, and result text area to the window
input_field.pack()
submit_button.pack()
result_text.pack()

# Run the main loop
window.mainloop()
