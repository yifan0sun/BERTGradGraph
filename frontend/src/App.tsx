import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import { LeftPanel } from './components/LeftPanel';
import { BottomPanel } from './components/BottomPanel';
import './App.css';
import {cleanToken} from './utils.ts'


import { GraphGrad } from './components/GradGraph';
const SERVER_ADDRESS = 'http://localhost:8000';

const App = () => {
 

const [sentence, setSentence] = useState('');
const [maskedSentence, setMaskedSentence] = useState('');
const [model, setModel] = useState('');
const [task, setTask] = useState('');
const [numLayers, setNumLayers] = useState(0);

const [matrix, setMatrix] = useState<number[][]>([]);


  const [tokens, setTokens] = useState<string[]>([]);

const [selectedLayer, setSelectedLayer] = useState(1);
const [mode, setMode] = useState<'attention' | 'gradient'>('attention');


const handleComputeMatrix = async () => {
    try {
        const res = await fetch('http://localhost:8000/get_grad_attn_matrix', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model,
            task,
            sentence: maskedSentence,
            selected_layer: selectedLayer,
        }),
        });

        const data = await res.json();
        if (data.error) throw new Error(data.error);

        const attn = data.attn_matrix ?? [];
        const grad = data.grad_matrix ?? [];

        // Always send gradient matrix for now
        setMatrix(task === 'MLM' ? grad : attn);
    } catch (err) {
        console.error('Error computing matrix:', err);
    }
};
 
 

  return (
  <div style={{ height: '100vh', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
    {/* Description bar at top */}
    <div style={{ padding: '1.5rem', borderBottom: '1px solid #ccc', flexShrink: 0 }}>
      <h2>Attention and Gradient Matrix Viewer</h2>
      <p>
        This tool lets you visualize how transformer models distribute attention and how sensitive their outputs are to each input token.
        <br /><br />
        - <strong>Attention</strong> shows how much each token attends to others across a given layer.
        <br />
        - <strong>Gradient Norm</strong> captures how much a small change to each input token affects the model’s attention pattern at that specific layer — indicating influence.
        <br /><br />
        To use: select a model and task, optionally mask a word (for MLM), and click "Load Model Info" to visualize attention or gradient patterns by layer.
        <br />
        (We truncate inputs to 200 words to avoid glitches when the input is too long.)
      </p>
    </div>

    {/* Main content below the top bar */}
    <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
      {/* Left Panel */}
      <div style={{ width: '300px', overflowY: 'auto', borderRight: '1px solid #ccc' }}>
        <LeftPanel
          onStateUpdate={(state) => {
            
            setSentence(state.sentence);
            setMaskedSentence(state.maskedSentence);
            setTokens(state.tokens);
            setModel(state.model);
            setTask(state.task);
            setNumLayers(state.numLayers);
          }}
          onComputeMatrix={handleComputeMatrix}
        />
      </div>

      {/* Right Side (BottomPanel + GraphGrad) */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ flexShrink: 0, borderBottom: '1px solid #ccc'  }}>
          <BottomPanel
            model={model}
            task={task}
            sentence={maskedSentence}
            numLayers={numLayers}
            onMatrixUpdate={setMatrix}
            onComputeMatrix={handleComputeMatrix}
            selectedLayer={selectedLayer}
            setSelectedLayer={setSelectedLayer}
            mode={mode}
            setMode={setMode}
          />
        </div>

        {tokens.length > 0 && matrix.length > 0 && (
            <div style={{ flex: 1, overflowY: 'auto' }}>
            <GraphGrad
                tokens={tokens}
                model={model}
                task={task}
                layer={selectedLayer}
                sentence={sentence}
                matrix={matrix}
            />
            </div>
        )}
      </div>
    </div>
  </div>
);



};

export default App;
