import React, { useState } from 'react';
 import { LeftPanel } from './components/LeftPanel';
import { BottomPanel } from './components/BottomPanel';
import './App.css';
 

import { GraphGrad } from './components/GradGraph';
 
const App = () => {
 


const [numLayers, setNumLayers] = useState(0);
 
const [matrices, setMatrices] = useState<{
  attention: number[][][],
  gradient: number[][][]
}>({ attention: [], gradient: [] });





  const [tokens, setTokens] = useState<string[]>([]);

const [selectedLayer, setSelectedLayer] = useState(1);
const [mode, setMode] = useState<'attention' | 'gradient'>('attention');

 
 

  return (
  <div style={{ height: '100vh', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
    {/* Description bar at top */}
    <div style={{ padding: '1.5rem', borderBottom: '1px solid #ccc', flexShrink: 0 }}>
      <h2>Attention and Gradient Matrix Viewer for BERT family transformers</h2>
      <p>
        Interpretability is key to trust in AI. This tool turns hidden weights into intuitive graphs — helping you explore how models reason through language.
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
<br />
      
        <p style={{ textAlign: "left" , marginBottom: "0px" }}><b>More.</b> Visit  <a href="https://github.com/yifan0sun/BertGradGraph/blob/main/README.md">the project README file</a></p>
        <p style={{ textAlign: "left" , marginBottom: "0px" }}><b>Prototype.</b>  Feedback and suggestions are welcome! Please visit <a href="https://sites.google.com/view/visualizerprojects/home">optimalvisualizer.com</a> to give feedback or visit more visually pleasing explainability apps.</p>
          

    </div>



    <div style={{
  fontSize: '12px',
  fontFamily: 'monospace',
  backgroundColor: '#f4f4f4',
  padding: '0.5rem',
  borderBottom: '1px solid #ccc'
}}>
  <strong>[App] Debug:</strong><br />
  Tokens: {tokens.length}<br /> {tokens} <br />
  Num Layers: {numLayers}<br />
  Matrix shape ({mode}, layer {selectedLayer}): {matrices[mode]?.[selectedLayer - 1]?.length ?? 0} × {matrices[mode]?.[selectedLayer - 1]?.[0]?.length ?? 0}
</div>


    {/* Main content below the top bar */}
    <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
      {/* Left Panel */}
      <div style={{ width: '300px',overflowX: 'hidden', overflowY: 'auto', borderRight: '1px solid #ccc' }}>
        <LeftPanel
          onStateUpdate={(state) => {
            setTokens(state.tokens);
            setNumLayers(state.numLayers);
            setMatrices(state.matrices);   
          }}
        />
      </div>
      {/* Right Side (BottomPanel + GraphGrad) */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ flexShrink: 0, borderBottom: '1px solid #ccc'  }}>
          <BottomPanel
            numLayers={numLayers}
            selectedLayer={selectedLayer}
            setSelectedLayer={setSelectedLayer}
            mode={mode}
            setMode={setMode}
          />
        </div>

        
          {matrices.attention?.length === numLayers &&
        matrices.gradient?.length === numLayers &&
        matrices.gradient?.[selectedLayer - 1]?.length === tokens.length &&
        matrices.gradient?.[selectedLayer - 1]?.[0]?.length === tokens.length  &&
        matrices.attention?.[selectedLayer - 1]?.length === tokens.length &&
        matrices.attention?.[selectedLayer - 1]?.[0]?.length === tokens.length ? (
            <div style={{ flex: 1, overflowY: 'auto' }}>
              <GraphGrad
                tokens={tokens}
                matrices={matrices}
                layer={selectedLayer}
                mode={mode}
              />
            </div>
          ) : (
            <div style={{ fontSize: '16px', color: '#000', padding: '0.5rem', fontFamily: 'sans-serif' }}>
              ⚠️ <strong>GraphGrad not rendered</strong><br />
              Solution: Is the app still retrieving data? If so, please be patient. If not,  hit reset sentence and load/reset model, and make sure you inputed all the task entries. <br />

              If it repeatedly does not work, try refreshing the page. Glitches can also be reported if they are too annoying, on the main project site.

              {/*<br /><br />
              More details:
              Reason: matrix size does not match token count<br />
              Mode: {mode}<br />
              Layer: {selectedLayer}<br />
              Tokens: {tokens.length}<br />
              Matrix rows: {matrices[mode]?.[selectedLayer - 1]?.length ?? 'undefined'}<br />
              Matrix cols: {matrices[mode]?.[selectedLayer - 1]?.[0]?.length ?? 'undefined'}
              */}
            </div>
          )}


      </div>
    </div>
  </div>
);



};

export default App;
