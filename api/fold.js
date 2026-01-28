export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { sequence } = req.body;
  
  if (!sequence || sequence.length > 400) {
    return res.status(400).json({ error: 'Sequence too long (max 400) or empty.' });
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 9000);

  try {
    // ESM Atlas API
    const response = await fetch("https://api.esmatlas.com/foldSequence/v1/PDB/", {
      method: 'POST',
      body: sequence,
      signal: controller.signal,
      headers: { 
        'Content-Type': 'text/plain',
        'User-Agent': 'BioToolbox/1.0' 
      }
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
       const errText = await response.text();
       console.error("ESM API Error:", errText);
       throw new Error(`External API Error: ${response.status}`);
    }

    const pdbData = await response.text();
    if (!pdbData.startsWith('HEADER') && !pdbData.startsWith('ATOM')) {
        throw new Error("Invalid PDB data received");
    }

    res.status(200).send(pdbData);

  } catch (error) {
    clearTimeout(timeoutId);
    console.error("Server Error:", error);
    
    if (error.name === 'AbortError') {
        return res.status(504).json({ error: 'Timeout: Protein folding took too long for the free server.' });
    }
    
    res.status(500).json({ error: error.message || 'Internal Server Error' });
  }
}
