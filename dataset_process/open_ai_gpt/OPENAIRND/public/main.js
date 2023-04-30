async function getGenre(summary) {
  try {
    const response = await fetch('http://localhost:3000/determine-genre', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ summary })
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error('Error determining genre:', errorData);
      return;
    }

    const data = await response.json();
    return data.genre;
  } catch (error) {
    console.error('Error fetching data:', error);
  }
}


function fetchGenre() {
  const summary = " The novel concerns a man with a dream and an allegorical quest through Spain.";
  getGenre(summary).then(genre => {
    console.log('The genre of the summary is:', genre);
  });
}
window.fetchGenre = fetchGenre;