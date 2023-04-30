const express = require('express');
const dotenv = require('dotenv');
const { Configuration, OpenAIApi } = require('openai');

dotenv.config();
const app = express();
app.use(express.json());

const config = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});

const openai = new OpenAIApi(config);

const path = require('path');
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
  res.send('Genre Detector API');
});

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

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
