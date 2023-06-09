## DAVID'S R&D SPACE

## Data Processing

<details>
<summary><h4>XLNet</h4></summary>

## XLNet
**XLNet** is a new unsupervised language representation learning method based on a novel generalized permutation language modeling objective. Additionally, XLNet employs [Transformer-XL](https://arxiv.org/abs/1901.02860) as the backbone model, exhibiting excellent performance for language tasks involving long context. Overall, XLNet achieves state-of-the-art (SOTA) results on various downstream language tasks including question answering, natural language inference, sentiment analysis, and document ranking. [Forked from here](https://github.com/zihangdai/xlnet)

Required packages
```sh
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import XLNetForSequenceClassification, XLNetTokenizer, Trainer, TrainingArguments
```
Genre Dataset encodings and labels using pytorch
```sh
class GenreDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
```
Training arguments for model. Issues: Computation cannot be handled on my GPU
```sh
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```
</details>

<details>
<summary><h4>Multinomial Model</h4></summary>

## Multinomial Logistic Regression Analysis
**Multinomial Logistic Regression** is a classification method that generalizes logistic regression to multiclass problems, i.e. with more than two possible discrete outcomes. That is, it is a model that is used to predict the probabilities of the different possible outcomes of a categorically distributed dependent variable, given a set of independent variables (which may be real-valued, binary-valued, categorical-valued, etc.).
The problem: The data for psychology, romance, sports, and travel genre is smaller respectively compared to the other genres, causing imbalance in outcomes.
The solution: We use Multinomial Logistic Regression to predict the probabilities of the different possible outcomes of a categorically distributed dependent variable, given a set of independent variables (which may be real-valued, binary-valued, categorical-valued, etc.). Then we fine tune the model with OpenAI to include the data from the other genres to improve the accuracy of the model (specifcially for the smaller genres).
[Kaggle dataset code](https://www.kaggle.com/code/athu1105/bookgenreprediciton)


Required packages
```sh
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk 
import string
from collections import Counter 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics
```
Genre count chart
![Genre chart](https://user-images.githubusercontent.com/48280799/235293899-d319fd9f-ec1c-4d09-a7cf-2f83980bac0c.png)

Confusion matrix
![Confusion Matrix](https://user-images.githubusercontent.com/48280799/235293936-c5a2f614-355d-4861-ae71-1b087432bb18.png)

<b>Key takeaways:</b> 
Here we can see that the model is getting confused with the horror, thriller and crime which is beacause they contain similar words.
There also needs to be more data for the genres sports, romance, travel, psychology - there is imbalance in dataset for accurate text classification.

</details>

<details>
<summary><h4>WordCloud Analysis</h4></summary>

## WordCloud Analysis and Modelling (no need for modelling since Multinomial Logistic Regression analysis is already done)
**WordCloud** is a data visualization technique used for representing text data in which the size of each word indicates its frequency or importance. Significant textual data points can be highlighted using a word cloud. Word clouds are widely used for analyzing data from social network websites. In this article, you will learn how to create a word cloud in Python.
top_words.json for top 15 most common words in each genre summary.
[Kaggle dataset code](https://www.kaggle.com/code/prathameshgadekar/book-genre-prediction-nlp)

Required packages
```sh
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import missingno as msno 

import plotly.offline as py
py.init_notebook_mode(connected=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import time
```
Generate wordcloud
```sh
def print_wordCloud(genre,summary):
    print(genre)
    wordcloud = WordCloud(width = 400, height = 400, 
                background_color ='white', 
                min_font_size = 10).generate(summary)
    plt.figure(figsize = (7, 7), facecolor = 'white', edgecolor='blue') 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show()
def make_string(genre):
    s = ""
    for row_index,row in data.iterrows():
        if(row['genre'] == genre):
            s+=(row['summary']+' ')
    return s
```

Thriller<br>
![thriller](https://user-images.githubusercontent.com/48280799/235294418-822d50ac-ce01-4784-b152-b40b82139d6e.png)

Fantasy<br>
![fantasy](https://user-images.githubusercontent.com/48280799/235294425-8b9a3e7e-4c9d-49bd-a7b6-c0234c8e08b4.png)

Science<br>
![science](https://user-images.githubusercontent.com/48280799/235294481-39eee485-efcc-4447-9691-53f473340326.png)

History<br>
![history](https://user-images.githubusercontent.com/48280799/235294533-fd323d15-9f27-443a-979e-b7d0ed282e23.png)

Horror<br>
![horror](https://user-images.githubusercontent.com/48280799/235294547-5bfe3c8b-be23-4955-8a80-43e3dac6fd46.png)

Crime<br>
![crime](https://user-images.githubusercontent.com/48280799/235294559-c5b72672-771d-4e97-84bc-c07f0e7d19b1.png)

Romance<br>
![romance](https://user-images.githubusercontent.com/48280799/235294566-88c5d4ef-7ce5-44d6-b8a8-dd10d148ca4d.png)

Psychology<br>
![psychology](https://user-images.githubusercontent.com/48280799/235294575-572a0489-c793-4471-9e43-d1877d29387e.png)

Sports<br>
![sports](https://user-images.githubusercontent.com/48280799/235294583-a5455e37-63cd-4b9b-b4ab-66c10404fd78.png)

Travel<br>
![travel](https://user-images.githubusercontent.com/48280799/235294589-7fc52e4d-fd15-420f-8787-8f5af8122d94.png)

Generate JSON of 20 most common words in each genre
```sh
def print_wordCloud(genre,summary):
    print(genre)
    wordcloud = WordCloud(width = 400, height = 400, 
                background_color ='white', 
                min_font_size = 10).generate(summary)
    plt.figure(figsize = (7, 7), facecolor = 'white', edgecolor='blue') 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show()
def make_string(genre):
    s = ""
    for row_index,row in data.iterrows():
        if(row['genre'] == genre):
            s+=(row['summary']+' ')
    return s
```
JSON of top 20 common words in each genre
```sh
{
  "thriller": [
    ["find", 1.0],
    ["less", 0.7212389380530974],
    ["life", 0.6519174041297935],
    ["alex", 0.5914454277286135],
    ["take", 0.5707964601769911],
    ["world", 0.5117994100294986],
    ["time", 0.5029498525073747],
    ["kill", 0.5],
    ["family", 0.5],
    ["murder", 0.4970501474926254],
    ["back", 0.4896755162241888],
    ["first", 0.47640117994100295],
    ["death", 0.4749262536873156],
    ["make", 0.4690265486725664],
    ["secret", 0.4557522123893805],
    ["know", 0.45132743362831856],
    ["father", 0.44542772861356933],
    ["meet", 0.4424778761061947],
    ["novel", 0.4306784660766962],
    ["help", 0.4247787610619469]
  ],
  .
  .
  .
}
```
[Full list here](https://github.com/MiSaengg/gunhee-RnD-space/blob/main/dataset_process/word_cloud/top_words.json)

</details>

<details>
<summary><h4>BERT (Bidirectional Encoder Representations from Transformers)</h4></summary>

## BERT (Bidirectional Encoder Representations from Transformers)
BERT, Bidirectional Encoder Representations from Transformers, is a family of masked-language models introduced in 2018 by researchers at Google. BERT is an open source machine learning framework for natural language processing (NLP). BERT is designed to help computers understand the meaning of ambiguous language in text by using surrounding text to establish context.
The following [kaggle code](https://www.kaggle.com/code/alexanderprokudaylo/book-genre-prediction) is a good example of how to use BERT to predict book genres.

BERT VS OPENAI GPT-3 for NLP text classification
![bert-vs-openai-](https://user-images.githubusercontent.com/48280799/235327808-ec963dcb-be71-4ed5-bd5a-52ec29193eda.jpg)
Full research paper [here](https://www.researchgate.net/publication/338931711_A_Short_Survey_of_Pre-trained_Language_Models_for_Conversational_AI-A_New_Age_in_NLP)

Confusion matrix of BERT:
![confusion matrix bert](https://user-images.githubusercontent.com/48280799/235328996-371c1fed-12a8-4809-bb6e-89c0e172082b.png)
We can see more true positives compared to the multinominal naive bayes model and sigthly higher accuracy for psychology, romance, sports, and travel genres. 
Due to no GPU, I was not able to run the model on the entire dataset. The result was sourced from the Kaggle dataset code.
To understand how confusion matrix works using scikit learn, please refer to [this article](https://www.jcchouinard.com/confusion-matrix-in-scikit-learn/).

</details>

<details>
<summary><h4>OpenAI Text NLP Text Classification</h4></summary>

## OpenAI Text NLP Text Classification
Open OPENAIRND folder in new workspace and run the following command

```sh
npm install express axios dotenv
npm install openai
```

In app.js, input your OpenAI API key
```sh
const config = new Configuration({
  apiKey: process.env.OPENAI_API_KEY, //replace process.env.OPENAI_API_KEY with your API key or input your API key to .env file
});
```

GPT prompt and response
```sh
app.post('/determine-genre', async (req, res) => {
  const { summary } = req.body;

  try {
    const prompt = `Based on the summary: "${summary}", determine two suitable genres from this list: thriller, fantasy, science fiction, history, horror, crime, romance, psychology, sports, travel.`;
    const response = await openai.createCompletion({
      model: 'text-davinci-003',
      prompt,
      max_tokens: 50,
      n: 1,
      temperature: 0.8,
    });

    const genresText = response.data.choices[0].text.trim();
    const genres = genresText.split(',').map(genre => genre.trim());
    res.json({ genres });
  } catch (error) {
    console.error('Error determining genre:', error);
    res.status(500).json({ error: 'Error determining genre', details: error.message });
  }
});
```
When given prompts from function fetchGenre()
![function](https://user-images.githubusercontent.com/48280799/235381723-1481d461-5b74-4ce9-9632-945315424066.png)

The output
![console output](https://user-images.githubusercontent.com/48280799/235381714-a8ddbfdf-61ab-4b70-baf1-5b4b58d73b9d.png)

<b>Key takeaways</b>: The GPT text completion model, text-davinci-03 provides a relatively accurate determination of the text classification of the summary. However, the model is not perfect and can be improved. The model is specifically set to determine two genres. Fine-tuning is a possibility to improve the model, however time considerations should be taken.

</details>

<details>
<summary><h4>GPT Tuning</h4></summary>

## GPT Tuning
[Official Documentation](https://platform.openai.com/docs/guides/fine-tuning)

Installation

```sh
pip install --upgrade openai
```

OpenAI API Key

```sh
export OPENAI_API_KEY="<OPENAI_API_KEY>"
```

Preparing training data - nmust be in JSONL format
Check prompt_pair.py for more details

```sh
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
...
```

CLI data preparation

```sh  
openai tools fine_tunes.prepare_data -f cli_ready.json
```

Uploading cli ready json to OPENAI
```sh
openai api files.create -f prompt_completion_pairs_prepared_train.jsonl -p fine-tune
openai api files.create -f prompt_completion_pairs_prepared_valid.jsonl -p fine-tune

openai api fine_tunes.create -t prompt_completion_pairs_prepared_train.json -v prompt_completion_pairs_prepared_valid.jsonl -m davinci
```

Create a fine-tuning run

```sh
gpt_tuning % openai api fine_tunes.create -t cli_ready_prepared_train.jsonl -v cli_ready_prepared_valid.jsonl -m davinci

openai api fine_tunes.follow -i <YOUR_FINE_TUNE_JOB_ID> // to follow the progress
```

Now we just wait for the fine-tuning to finish. This can take a while depending on the size of the dataset and the model you chose.
For financial and testing purposes, I chose the cheapest model, ada.

![image](https://user-images.githubusercontent.com/48280799/235599700-43be23c1-e5fb-4d89-9793-2d1e1022a677.png)

Once model is complete, we run the program and enter prompt text to generate the completion text.

```sh
import tkinter as tk
import openai
import re

# Replace FINE_TUNED_MODEL with the name of your fine-tuned model
model_name = "FINE_TUNED_MODEL"


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
```

To run the program:

```sh
python3 main.py
```

Results of the program:
![Response](https://user-images.githubusercontent.com/48280799/235793022-72732e5a-187a-4953-9a75-c189e649d52a.png)

</details>

## Next Steps:
- [ ] Create a web app to run the program
- [ ] Connect the program to node using flask or express using jsx version of the program
- [ ] More fine tuning to improve accuracy
- [ ] More testing by inputting more text to see if the program can properly detect the genre

## Font Style and Size

## To work on:
<ul>
    <li>Font style and size for readability</li>
</ul>
