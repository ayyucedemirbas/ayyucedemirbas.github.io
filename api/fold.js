export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).send('Only POST requests allowed');
  }

  try {
    const { sequence } = req.body;
    if (!sequence) {
      return res.status(400).json({ error: 'Sequence is required' });
    }

    const externalUrl = "https://api.esmatlas.com/foldSequence/v1/PDB/";
    
    const response = await fetch(externalUrl, {
      method: 'POST',
      body: sequence,
      headers: { 'Content-Type': 'text/plain' }
    });

    if (!response.ok) {
       const errText = await response.text();
       throw new Error(`ESM API Error: ${response.status}`);
    }

    const pdbData = await response.text();
    res.status(200).send(pdbData);

  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}