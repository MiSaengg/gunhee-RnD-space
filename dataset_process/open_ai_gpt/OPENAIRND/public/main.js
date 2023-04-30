async function getGenre(summary) {
  try {
    const response = await fetch('http://localhost:3000/determine-genre', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ summary })
    });

    const data = await response.json();
    console.log('Server response:', data); 
    return data.genres; 
  } catch (error) {
    console.error('Error fetching data:', error);
  }
}

function fetchGenre() {
  const summary = "The book begins where Roses are Red ended, with Dr. Alex Cross at the home of murdered FBI Agent Betsey Cavalierre. Alex is in his room when his cellphone rings. Alex answers and the Mastermind is on the line. "; //thriller
  const fantasy = "  Lief, Barda, and Jasmine leave the Lake of Tears, after they have retrieved the Ruby. They are now searching for the opal, which is located in Hira, or the City of the Rats. "; //fantasy
  const science = "  A British rocket, developed at minimal cost and secretly from officialdom, lifts off from the Woomera rocket range on a mission to Mars."; //science fiction
  const history = "  After being separated by their work in World War II, British Army nurse Claire Randall, and her husband Frank, an Oxford history professor who briefly worked for MI6, go on a second honeymoon to Inverness, Scotland."; //history
  const horror = " When dead guys start turning up as soon as the Moon family appears in Sunnydale Buffy knows that something is wrong. Mo, the mother, and her two daughters, Calli and Polly, all go to Sunnydale High."; //horror
  const crime = "  The story is told by Peter Aaron about the victim, Benjamin Sachs, his best friend whom he first meets as a fellow writer in a Greenwich Village bar in 1975. Peter decides to try to piece together the story of Ben's other life after agents from the F.B.I. approach him in the course of their investigation."; //crime
  const romance = "  Meet Emma Corrigan, a young woman with a huge heart, an irrepressible spirit, and a few little secrets: Secrets from her boyfriend: I've always thought Connor looks a bit like Ken."; //romance
  const psychology = " Freud's discovery that the dream is the means by which the unconscious can be explored is undoubtedly the most revolutionary step forward in the entire history of psychology. Dreams, according to his theory, represent the hidden fulfillment of our unconscious wishes."; //psychology
  const sports = " This is a book about young men who learned to play baseball during the 1930s and 1940s, and then went on to play for one of the most exciting major-league ball clubs ever fielded, the team that broke the color barrier with Jackie Robinson."; //sports
  const travel = " Paul Theroux invites you to join him on the journey of a lifetime, in the grand romantic tradition, by train across Europe, through the vast underbelly of Asia and in the heart of Russia, and then up to China."; //travel
  getGenre(summary).then(genre => {
    console.log('The genre of the summary is:', genre);
  });
  getGenre(fantasy).then(genre => {
    console.log('The genre of the summary is:', genre);
  });
  getGenre(science).then(genre => {
    console.log('The genre of the summary is:', genre);
  });
  getGenre(history).then(genre => {
    console.log('The genre of the summary is:', genre);
  });
  getGenre(horror).then(genre => {
    console.log('The genre of the summary is:', genre);
  });
  getGenre(crime).then(genre => {
    console.log('The genre of the summary is:', genre);
  });
  getGenre(romance).then(genre => {
    console.log('The genre of the summary is:', genre);
  });
  getGenre(psychology).then(genre => {
    console.log('The genre of the summary is:', genre);
  });
  getGenre(sports).then(genre => {
    console.log('The genre of the summary is:', genre);
  });
  getGenre(travel).then(genre => {
    console.log('The genre of the summary is:', genre);
  });
}
window.fetchGenre = fetchGenre;