import React, { useState, useEffect } from 'react';
//const SERVER_ADDRESS = 'https://yifan0sun-bertgradgraph.hf.space';
//
//const SERVER_ADDRESS = 'https://bertgradgraph.onrender.com';

const SERVER_ADDRESS = 'http://127.0.0.1:8000';

import '../App.css';
import { cleanToken } from  '../utils.ts'


export function LeftPanel({
      onStateUpdate,
}: {
onStateUpdate: (state: {
  tokens: string[];
  numLayers: number;
  matrices: {
    attention: number[][][];
    gradient: number[][][];
  };
}) => void;
}) {
    const DEFAULT_SENTENCE = 'The quick brown fox jumps over the lazy dog.';
    const [inputSentence, setInputSentence] = useState(DEFAULT_SENTENCE);
    const [confirmedSentence, setConfirmedSentence] = useState(DEFAULT_SENTENCE);

    const [selectedModel, setSelectedModel] = useState('BERT');
    const [selectedTask, setSelectedTask] = useState('MLM');
    const [selectedTokenIdxToMask, setSelectedTokenIdxToMask] = useState<number | null>(null);
    const [tokens, setTokens] = useState<string[]>([]);
    const [numLayers, setNumLayers] = useState<number | null>(null);
    const [modelInfo, setModelInfo] = useState<string | null>(null); 

    const [predictTokens, setPredictTokens] = useState<string[]>([]);
    const [predictProbs, setPredictProbs] = useState<number[]>([]);

    const [inputHypothesis, setInputHypothesis] = useState('');
    const [confirmedHypothesis, setConfirmedHypothesis] = useState('');


    const modelOptions = ['BERT', 'RoBERTa',  'DistilBERT'];
    const taskOptions = ['MLM', 'SST', 'MNLI'];

    
  const [isRetrievingMatrix, setIsRetrievingMatrix] = useState(false);

    
   const handleConfirmSentence = async () => {


     


  setConfirmedSentence(inputSentence);
  setConfirmedHypothesis(inputSentence); 
   
  setNumLayers(null);
  setSelectedTokenIdxToMask(null);
  setPredictTokens([]);
  setPredictProbs([]);

  onStateUpdate({
    tokens: tokens,
    numLayers: 0,
    matrices: {
      attention: [],  // one empty matrix per layer
      gradient: [],
    },
  });
};



const handleLoadModel = async (
  
  selectedModel: string,
  selectedTask: string,
  confirmedSentence: string,
  confirmedHypothesis: string

) => {
  try {
      const res = await fetch(`${SERVER_ADDRESS}/load_model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: selectedModel,
        task: selectedTask,
        sentence: confirmedSentence,
        hypothesis:  confirmedHypothesis
      }),
    });
   
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
 
    if (!data.tokens || !Array.isArray(data.tokens)) {
      throw new Error('Invalid or missing tokens array in server response.');
    }

    const newTokens = Array.isArray(data.tokens) ? data.tokens : [];
    const layerCount = typeof data.num_layers === 'number' ? data.num_layers : 0;
    
    setTokens(newTokens);
    setNumLayers(layerCount);
    
 
     


 onStateUpdate({
  tokens: newTokens,
  numLayers: layerCount,
  matrices: {
    attention: Array.from({ length: layerCount }, () => []),
    gradient: Array.from({ length: layerCount }, () => []),
  },
});
    

    
    setModelInfo(null);

   
  return { tokens: newTokens, numLayers: layerCount };

  } catch (err) {
    console.error('Load model failed:', err);
    setModelInfo(`Error loading model: ${String(err)}`);
  }

};




const handlePredict = async (
  
  selectedModel: string,
  selectedTask: string,
  confirmedSentence: string,
  confirmedHypothesis: string
  
) => {
  try {
    const payload: any = {
      model: selectedModel,
      task: selectedTask,
      hypothesis: '',
      maskID: 0,
      sentence: confirmedSentence
    };

    if (selectedTask === 'MLM') {
      payload.maskID = selectedTokenIdxToMask;
    } else if (selectedTask === 'MNLI') {
      payload.hypothesis = confirmedHypothesis;
    } 

    const res = await fetch(`${SERVER_ADDRESS}/predict_model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    if (data.error) throw new Error(data.error);

    setPredictTokens(data.decoded ?? []);
    setPredictProbs(data.top_probs ?? []);
  } catch (err) {
    setModelInfo(`Error during prediction: ${err}`);
  }
};




const handleComputeMatrix = async  (
  
  model: string,
  task: string,
  sentence:string,
  hypothesis: string,
  currentTokens: string[]   
  
) => {
  setIsRetrievingMatrix(true);  // ⬅️ Start loading

    const payload: any = {
      model,
      task,
      hypothesis: '', // default, overridden below
      sentence: sentence,
      maskID: 0,
    };

    if (task === 'MLM') {
      payload.maskID = selectedTokenIdxToMask;
    } else if (task === 'MNLI') {
      payload.hypothesis = hypothesis;
    }



     try {
        const res = await fetch(`${SERVER_ADDRESS}/get_grad_attn_matrix`, {

        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),  
        });

        const data = await res.json();
        if (data.error) throw new Error(data.error);

        const attn = data?.["attn_matrix"] ?? [];
        const grad = data?.["grad_matrix"] ?? [];


        onStateUpdate({
          tokens: currentTokens,
          numLayers: attn.length,
          matrices: {
            attention: attn,
            gradient: grad,
          },
        });
        

    } catch (err) {
        console.error('Error computing matrix:', err);
    } finally {
    setIsRetrievingMatrix(false);  // ⬅️ End loading
  }
};
 



 



 

 
 


  return (
    <div style={{ width: '300px', padding: '1rem', borderRight: '1px solid #ccc' }}>

{/*
<div style={{
  fontSize: '12px',
  fontFamily: 'monospace',
  marginTop: '1rem',
  backgroundColor: '#f4f4f4',
  padding: '0.5rem',
  borderTop: '1px solid #ccc'
}}>
  <strong>[LeftPanel] Debug:</strong><br />
  Tokens: {tokens.length}<br />
  Num Layers: {numLayers ?? 0}<br />
  Matrix shape (attention): {tokens.length > 0 && numLayers ? `${numLayers} × ${tokens.length}` : 'n/a'}<br />
  masked token: {selectedTokenIdxToMask}<br />
  sentence: {confirmedSentence}<br />
  server address: {SERVER_ADDRESS}
</div>
*/}


      {/* Sentence input */}
      <div style={{ marginBottom: '1rem' }}>
        <label>Sentence:</label><br />
        <textarea
        value={inputSentence}
        onChange={e => setInputSentence(e.target.value)}
        rows={4}
        placeholder="Type or paste your text here..."
        style={{
            width: '100%',
            fontSize: '.9rem',
            padding: '0.25rem',
            resize: 'vertical',
            boxSizing: 'border-box',
            fontFamily: 'monospace',
            lineHeight: '1.4',
        }}
        />
        <button onClick={handleConfirmSentence} style={{ marginTop: '0.5rem' }} className="uniform-button">
          Reset Sentence
        </button>
      </div>

      {/* Model dropdown */}
      <div style={{ marginBottom: '1rem' }}>
        <label>Model:</label><br />
        <select
            className="uniform-select"
            value={selectedModel}
            onChange={e => setSelectedModel(e.target.value)}
            style={{ width: '100%' }}
        >
          {modelOptions.map(model => (
            <option key={model} value={model}>{model}</option>
          ))}
        </select>

        
     

       </div>
 <div style={{ marginBottom: '1rem' }}>

         
      {/* Task dropdown */} 
        <label>Task:</label><br />
        <select
            className="uniform-select"
            value={selectedTask}
            onChange={(e) => {
              setSelectedTask(e.target.value);
              setPredictTokens([]);
              setPredictProbs([]);
            }}
            style={{ width: '100%' }}
            >
            {taskOptions.map(task => (
  <option key={task} value={task}>{task}</option>
))}
        </select>


      
      
      </div>


      {/* Mask dropdown (if MLM) */} 
        {selectedTask === 'MLM'   && tokens.length > 0 ? (

        <>

     


        


      <div style={{ marginBottom: '1rem' }}>
  <label>Word to Mask:</label><br />
  <select
    className="uniform-select"
    value={selectedTokenIdxToMask ?? ''}
    onChange={async (e) => {
      const val = e.target.value;
      const newMaskIdx = val === '' ? null : parseInt(val);
      setSelectedTokenIdxToMask(newMaskIdx);

      const { tokens: newTokens, numLayers: newLayers } = await handleLoadModel(
        selectedModel,
        selectedTask,
        confirmedSentence,
        confirmedHypothesis
      );

      await handlePredict(
        selectedModel,
        selectedTask,
        confirmedSentence,
        confirmedHypothesis
      );

      await handleComputeMatrix(
        selectedModel,
        selectedTask,
        confirmedSentence,
        confirmedHypothesis,
        newTokens
      );
    }}
    style={{ width: '100%' }}
  >
    <option value="">(pick a token to mask)</option>
    {tokens.map((tok, idx) => {
      const clean = cleanToken(tok);
      const unmaskable = ['[SEP]', '[MASK]', '[CLS]', '<s>', '</s>'];
      if (unmaskable.includes(clean)) return null;

      return (
        <option key={idx} value={idx}>
          {idx} {clean}
        </option>
      );
    })}
  </select>
</div>


        
        

        </>
        ) : null }

               {/* Load model button */}
      <button
        className="uniform-button"
        onClick={async () => {
          const { tokens: newTokens, numLayers: newLayers } = await handleLoadModel(
                selectedModel,
                selectedTask,
                confirmedSentence,
                confirmedHypothesis
              );
          await handlePredict(
            selectedModel,
            selectedTask,
            confirmedSentence,
            confirmedHypothesis
          );

          await handleComputeMatrix(
            selectedModel,
            selectedTask,
            confirmedSentence, 
            confirmedHypothesis,
                newTokens           
          );

        }}
        style={{ marginTop: '1rem' }}
      >
        Load / Reset Model
      </button>
        
        {selectedTask === 'MNLI' && (
        <div style={{ marginBottom: '1rem' }}>
            <label>Hypothesis:</label><br />
            <textarea
            value={inputHypothesis}
            onChange={e => setInputHypothesis(e.target.value)}
            rows={4}
            placeholder="Enter hypothesis for MNLI task..."
            style={{
                width: '100%',
                fontSize: '.9rem',
                padding: '0.25rem',
                resize: 'vertical',
                boxSizing: 'border-box',
                fontFamily: 'monospace',
                lineHeight: '1.4',
            }}
            />
        </div>
        )}
        {selectedTask === 'MNLI' && (
        <button
            onClick={async () => {
            setConfirmedHypothesis(inputHypothesis); 

            const { tokens: newTokens, numLayers: newLayers } = await handleLoadModel(
                selectedModel,
                selectedTask,
                confirmedSentence,
                confirmedHypothesis
              );

            await handlePredict(
              selectedModel,
              selectedTask,
              confirmedSentence,
              confirmedHypothesis
            );

            await handleComputeMatrix(
              selectedModel,
              selectedTask,
              confirmedSentence, 
              confirmedHypothesis,
                newTokens           
            );

            }}
            className="uniform-button"
            style={{ marginTop: '0.5rem' }}
        >
            Submit Hypothesis
        </button>
        )}

          {isRetrievingMatrix && (
            <div style={{
              fontStyle: 'italic',
              fontSize: '12px',
              color: '#555',
              marginBottom: '1rem'
            }}>
              Retrieving data...
            </div>
          )}


      {/* bar chart */}
      {predictTokens.length > 0 && predictProbs.length > 0 ? (
        <div style={{ marginBottom: '1rem' }}>
          <h4>Top Predictions</h4>
          <svg width="100%" height={predictTokens.length * 30}>
            {predictTokens.map((token, i) => (
              <g key={i} transform={`translate(0, ${i * 30})`}>
                <text x="0" y="20" style={{ fontSize: '12px' }}>{cleanToken(token)}</text>
                <rect
                  x="60"
                  y="10"
                  height="10"
                  width={`${predictProbs[i] * 200}`}
                  fill="#3b82f6"
                />
                <text x={`${predictProbs[i] * 200 + 65}`} y="20" fontSize="12">
                  {predictProbs[i].toFixed(2)}
                </text>
              </g>
            ))}
          </svg>
        </div>
      ) : (
        <div>
      

          {/* Model info feedback */}
          {predictTokens.length === 0 && predictProbs.length === 0 && (
            <div style={{ marginTop: '1rem', color: '#000' }}>
              {selectedTask === 'MLM' && tokens.length === 0 && (
                <>⚠️ To begin, reset your sentence, click <strong>Load Model</strong>, and choose a word to mask.</>
              )}
              {selectedTask === 'SST' && tokens.length === 0 && (
                <>⚠️ To begin, reset your sentence and click <strong>Load Model</strong>.</>
              )}
              {selectedTask === 'MNLI' && tokens.length === 0 && (
                <>⚠️ To begin, reset your sentence, click <strong>Load Model</strong>, and then type and submit a hypothesis.</>
              )}
              {!selectedTask && (
                <>⚠️ Select a task, reset the sentence, and click <strong>Load Model</strong> to get started.</>
              )}
            </div>
          )}
        </div>
      )}



      </div>

 
  );
}
