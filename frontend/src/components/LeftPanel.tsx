import React, { useState, useEffect } from 'react';
const SERVER_ADDRESS = 'http://localhost:8000';
import '../App.css';
import { cleanToken } from  '../utils.ts'


export function LeftPanel({
      onStateUpdate,
      onComputeMatrix,
}: {
  onStateUpdate: (state: {
    sentence: string;
    maskedSentence:string;
    tokens: string[];
    task: string;
    model: string;
    numLayers: number;
  }) => void;
  onComputeMatrix: () => void;
}) {
    const DEFAULT_SENTENCE = 'The quick brown fox jumps over the lazy dog.';
    const [inputSentence, setInputSentence] = useState(DEFAULT_SENTENCE);
    const [confirmedSentence, setConfirmedSentence] = useState(DEFAULT_SENTENCE);
    const [maskedSentence, setMaskedSentence] = useState(DEFAULT_SENTENCE);

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

    const [modelLoaded, setModelLoaded] = useState(false);
    const [predictionTriggered, setPredictionTriggered] = useState(false);

    const availableTasks = taskOptions.filter(task => {
        if (selectedModel === 'DistilBERT' && task === 'SQuAD') return false;
        return true;
    });

    
   const handleConfirmSentence = async () => {


    
  const rawWords = inputSentence.trim().split(/\s+/);
  const limited = rawWords.slice(0, 200).join(' ');


  setInputSentence(limited);              // ✅ update textarea
  setConfirmedSentence(limited);          // backend logic
  setConfirmedHypothesis(inputHypothesis);

  setMaskedSentence(limited);
  setTokens([]);
  setNumLayers(null);
  setSelectedTokenIdxToMask(null);
  setModelLoaded(false);
  setPredictTokens([]);
  setPredictProbs([]);

  onStateUpdate({
    sentence: limited,
    maskedSentence: limited,
    tokens: [],
    task: selectedTask,
    model: selectedModel,
    numLayers: 0,
  });
};




  const handleLoadModel = async () => {
  try {
    const res = await fetch(`${SERVER_ADDRESS}/load_model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: selectedModel,
        task:  selectedTask,
        sentence: selectedTask === 'MNLI'
            ? `${confirmedSentence} [SEP] ${confirmedHypothesis}`
            : confirmedSentence,
      }),
    });

    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const newTokens = data.tokens ?? [];
    setTokens(newTokens);
    setNumLayers(data.num_layers ?? null);

    const firstMaskableIdx = newTokens.findIndex(
      (tok) => !['[SEP]', '[MASK]', '[CLS]', '<s>', '</s>'].includes(tok)
    );
    setSelectedTokenIdxToMask(firstMaskableIdx !== -1 ? firstMaskableIdx : null);

    const newMaskedSentence =
      selectedTask === 'MLM' && firstMaskableIdx !== -1
        ? newTokens.map((tok, i) => (i === firstMaskableIdx ? '[MASK]' : tok)).join(' ')
        : confirmedSentence;

    setMaskedSentence(newMaskedSentence);

    onStateUpdate({
      sentence: confirmedSentence,
      maskedSentence: newMaskedSentence,
      tokens: newTokens,
      task: selectedTask,
      model: selectedModel,
      numLayers: data.num_layers ?? 0,
    });

     
    setModelLoaded(true);
  } catch (err) {
    setModelInfo(`Error loading model: ${err}`);
  }
};





useEffect(() => {
  if (selectedTask === 'MLM' && selectedTokenIdxToMask !== null) {
    handlePredict();        // forward
    onComputeMatrix();      // backward
  }
}, [selectedTokenIdxToMask]);


  const handlePredict = async () => {
  try {
    const res = await fetch(`${SERVER_ADDRESS}/predict_model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: selectedModel,
        task: selectedTask,
        sentence: selectedTask === 'MNLI'
            ? `${confirmedSentence} [SEP] ${confirmedHypothesis}`
            : maskedSentence,

      }),
    });

    const data = await res.json();
    if (data.error) throw new Error(data.error);

    setPredictTokens(data.decoded ?? []);
    setPredictProbs(data.top_probs ?? []);
    setPredictionTriggered(true);
  } catch (err) {
    setModelInfo(`Error during prediction: ${err}`);
  }
};



 

useEffect(() => {
  if (tokens.length === 0 || numLayers === null) return;

  setModelInfo(
    `Loaded model ${selectedModel}. ${numLayers} layers.\n` +
    `Original: "${confirmedSentence}"\n` +
    `Masked:   "${maskedSentence}"\n` +
    `Tokens: [${tokens.join(', ')}]`
  );
}, [confirmedSentence, maskedSentence, selectedTokenIdxToMask, tokens, selectedModel, numLayers]);


 
useEffect(() => {
  if (predictionTriggered) {
    handlePredict();
  }
}, [maskedSentence, selectedModel, selectedTask]);


useEffect(() => {
  if (selectedTask === 'MLM' && selectedTokenIdxToMask === null) {
    const firstMaskableIdx = tokens.findIndex(
      (tok) => !['[SEP]', '[MASK]', '[CLS]', '<s>', '</s>'].includes(tok)
    );
    if (firstMaskableIdx !== -1) {
      setSelectedTokenIdxToMask(firstMaskableIdx);
    }
  }
}, [tokens, selectedTask]);


useEffect(() => {
  if (selectedTask !== 'MLM') return;
  if (selectedTokenIdxToMask === null || tokens.length === 0) return;

  const newMaskedSentence = tokens
    .map((tok, i) => (i === selectedTokenIdxToMask ? '[MASK]' : tok))
    .join(' ');

  setMaskedSentence(
    selectedTask === 'MLM'
        ? limited.split(/\s+/).map((w, i) => (i === 0 ? '[MASK]' : w)).join(' ')
        : limited
    );

  // Send update to App so sentence -> BottomPanel updates too
  onStateUpdate({
    sentence: confirmedSentence,
    maskedSentence: newMaskedSentence,
    tokens,
    task: selectedTask,
    model: selectedModel,
    numLayers: numLayers ?? 0,
  });
}, [selectedTokenIdxToMask]);


useEffect(() => {
  if (modelLoaded && selectedTask !== 'MLM') {
    handlePredict();        // forward
    onComputeMatrix();      // backward
  }
}, [modelLoaded, selectedTask]);


  return (
    <div style={{ width: '300px', padding: '1rem', borderRight: '1px solid #ccc' }}>

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
            onChange={e => setSelectedTask(e.target.value)}
            style={{ width: '100%' }}
            >
            {availableTasks.map(task => (
                <option key={task} value={task}>{task}</option>
            ))}
        </select>


         {/* Load model button */}
      <button className="uniform-button" onClick={handleLoadModel} style={{ marginTop: '1rem' }}>
        Load Model
      </button>
      
      </div>


      {/* Mask dropdown (if MLM) */} 
        {selectedTask === 'MLM'   && tokens.length > 0 ? (

        <>

     


            <div style={{ marginBottom: '1rem' }}>
            <label>Word to Mask:</label><br />
            <select
                className="uniform-select"
                value={selectedTokenIdxToMask ?? ''}
                onChange={e => setSelectedTokenIdxToMask(parseInt(e.target.value))}
                style={{ width: '100%' }}
            >
                {tokens.map(cleanToken)
                    .map((word, idx) => (
                    !['[SEP]', '[MASK]', '[CLS]', '</s>', '<s>'].includes(word) ? (
                        <option key={idx} value={idx}>
                         {idx} {word}
                        </option>
                    ) : null
                    ))}

            </select>
            </div>
            

        
            
        

        </>
        ) : null }

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
            onClick={() => {
            handlePredict();
            onComputeMatrix();
            }}
            className="uniform-button"
            style={{ marginTop: '0.5rem' }}
        >
            Submit Hypothesis
        </button>
        )}


{/*modelLoaded && (
  <>
<button onClick={handlePredict} style={{ marginBottom: '1rem', marginRight: '1rem' }} className="uniform-button" >
            Forward prop
            </button>

    <button onClick={onComputeMatrix} style={{ marginBottom: '1rem' }} className="uniform-button">
            Backprop
        </button>
 </>
)*/}

      {/* bar chart */}
        { predictTokens.length > 0 && predictProbs.length > 0 && (
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
        )}


      {/* Model info feedback */}
      {/*
      {modelInfo && (
        <div style={{ marginTop: '1rem', fontStyle: 'italic', color: '#333' }}>
          {modelInfo}

        </div>
      )}*/}

    </div>
  );
}
