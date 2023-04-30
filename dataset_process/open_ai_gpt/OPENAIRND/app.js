const express = require('express');
const axios = require('axios');
const dotenv = require('dotenv');

dotenv.config();
const app = express();
app.use(express.json());

const apiKey = ''; //Input your API key here
const apiUrl = 'https://api.openai.com/v1/engines/davinci/completions';

const path = require('path');

app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
  res.send('Welcome to the Genre Detector API!');
});

app.post('/determine-genre', async (req, res) => {
  const { summary } = req.body;

  try {
    const prompt = `Determine the genre from the following list (thriller, fantasy, science, history, horror, crime, romance, psychology, sports, travel) based on the summary: "${summary}". The genre is:`;
    const response = await axios.post(apiUrl, {
      prompt,
      max_tokens: 10,
      n: 1,
      temperature: 0.5,
    }, {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      }
    });

    const genre = response.data.choices[0].text.trim();
    res.json({ genre });
  } catch (error) {
    console.error('Error determining genre:', error);
    res.status(500).json({ error: 'Error determining genre', details: error.message });
  }
});


const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
