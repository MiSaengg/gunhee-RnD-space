const genres = [
    'thriller',
    'fantasy',
    'science',
    'history',
    'horror',
    'crime',
    'romance',
    'psychology',
    'sports',
    'travel',
  ];
  
  const genreKeywords = {
    thriller: ['suspense', 'mystery', 'tense', 'conspiracy'],
    fantasy: ['magic', 'wizard', 'dragon', 'mythical'],
    science: ['experiment', 'technology', 'research', 'discovery'],
    history: ['historical', 'past', 'ancient', 'war'],
    horror: ['frightening', 'scary', 'terror', 'ghost'],
    crime: ['detective', 'murder', 'investigation', 'heist'],
    romance: ['love', 'passion', 'relationship', 'heart'],
    psychology: ['mind', 'mental', 'behavior', 'psyche'],
    sports: ['athlete', 'competition', 'game', 'team'],
    travel: ['journey', 'adventure', 'explore', 'destination'],
  };
  //Change this with top_words.json
  
  function determineGenre(summary) {
    const summaryWords = summary.toLowerCase().split(' ');
    const genreScores = {};
  
    for (const genre of genres) {
      genreScores[genre] = 0;
      for (const keyword of genreKeywords[genre]) {
        if (summaryWords.includes(keyword)) {
          genreScores[genre]++;
        }
      }
    }
  
    const sortedGenres = Object.entries(genreScores).sort((a, b) => b[1] - a[1]);
    return sortedGenres[0][0];
  }
  
  const summary = 'A young wizard embarks on a magical journey to save the kingdom from an evil dragon.';
  const genre = determineGenre(summary);
  console.log('The genre of the summary is:', genre);
  