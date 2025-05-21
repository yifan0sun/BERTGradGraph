import React, { useState, useEffect } from 'react';
import '../App.css';
import {cleanToken} from  '../utils.ts'

const SERVER_ADDRESS = 'http://localhost:8000';
export function BottomPanel({
  model,
  task,
  sentence,
  numLayers,
  onMatrixUpdate,
  onComputeMatrix,
  selectedLayer,
  setSelectedLayer,
  mode,
  setMode,
}: {
  model: string;
  task: string;
  sentence: string;
  numLayers: number;
  onMatrixUpdate: (matrix: number[][]) => void;
  onComputeMatrix: () => void;
  selectedLayer: number;
  setSelectedLayer: (layer: number) => void;
  mode: 'attention' | 'gradient';
  setMode: (mode: 'attention' | 'gradient') => void;
}) {
    const [allMatrices, setAllMatrices] = useState<{ attention: number[][]; gradient: number[][] }>({ attention: [], gradient: [] });

   

  const handleModeChange = (newMode: 'attention' | 'gradient') => {
    setMode(newMode);
    // Only update if matrix was already fetched
    const matrix = allMatrices[newMode];
    if (matrix.length > 0) {
      onMatrixUpdate(matrix);
    }
  };

  const layerOptions = Array.from({ length: numLayers }, (_, i) => i + 1);

 useEffect(() => {
  if (!model || !task || !sentence || !selectedLayer) return;

  const fetchMatrix = async () => {
    try {
      const res = await fetch(`${SERVER_ADDRESS}/get_grad_attn_matrix`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          task,
          sentence,
          selected_layer: selectedLayer,
        }),
      });

      const data = await res.json();
      if (data.error) throw new Error(data.error);

      const newMatrices = {
        attention: data.attn_matrix ?? [],
        gradient: data.grad_matrix ?? [],
      };

      setAllMatrices(newMatrices);
      onMatrixUpdate(newMatrices[mode]);
    } catch (err) {
      console.error('Error fetching matrix:', err);
    }
  };

  fetchMatrix();
}, [model, task, sentence, selectedLayer, mode]);



  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '2rem',
        padding: '1rem',
        fontFamily: 'monospace',
        fontSize: '14px',
      }}
    >
      {/* Layer Dropdown */}
      <div>
        <label>Layer:</label><br />
        <select
          value={selectedLayer}
          onChange={(e) => setSelectedLayer(parseInt(e.target.value))}
          style={{   padding: '0.25rem 0.4rem',  fontSize: '0.9rem',  fontFamily: 'monospace' }}
        >
          {layerOptions.map(layer => (
            <option key={layer} value={layer}>{layer}</option>
          ))}
        </select>
      </div>

      {/* Radio Buttons */}
      <div>
        <label>Mode:</label><br />
        <label style={{ marginRight: '1rem' }}>
          <input
            type="radio"
            value="attention"
            checked={mode === 'attention'}
            onChange={() => setMode('attention')}
          />
          Attention
        </label>
        <label>
          <input
            type="radio"
            value="gradient"
            checked={mode === 'gradient'}
            onChange={() => setMode('gradient')}
          />
          Gradient Norm
        </label>
      </div>

       
    </div>
  );
}
