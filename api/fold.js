export const config = {
  runtime: 'edge',
};

export default async function handler(request) {
  if (request.method === 'OPTIONS') {
    return new Response(null, {
      status: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
      },
    });
  }

  if (request.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), { 
        status: 405,
        headers: { 'Content-Type': 'application/json' }
    });
  }

  try {
    // Gelen veriyi al
    const body = await request.json();
    const sequence = body.sequence;

    if (!sequence) {
      return new Response(JSON.stringify({ error: 'Sequence required' }), { 
          status: 400,
          headers: { 'Access-Control-Allow-Origin': '*' }
      });
    }

    const response = await fetch("https://api.esmatlas.com/foldSequence/v1/PDB/", {
      method: 'POST',
      body: sequence,
      headers: { 'Content-Type': 'text/plain' }
    });

    if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    const pdbData = await response.text();

    return new Response(pdbData, {
      status: 200,
      headers: {
        'Access-Control-Allow-Origin': '*', // Her yerden erişime izin ver
        'Content-Type': 'text/plain',
      },
    });

  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: { 
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json'
        }
    });
  }
}
