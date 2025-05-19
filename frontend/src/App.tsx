import React, { useState, useEffect } from 'react'
import dummyImage from './assets/dummy.png'
import './App.css'

import ScenarioCard from './components/ScenarioCard.tsx'

const SERVER_ADDR = "http://localhost:8000/load_model";



function maskSentence(original: string, tokenIndex: number, tokens: string[], maskToken: string): string {
  const words = original.split(" ");
  if (tokenIndex >= 0 && tokenIndex < words.length) {
    words[tokenIndex] = maskToken;
    return words.join(" ");
  }
  return original;
}




export default function App() {
  const defaultSentence = 'The quick brown fox jumps over the lazy dog.'
  const defaultModel = 'BERT'
  const defaultTask = 'MLM'


  const [sentence, setSentence] = useState(defaultSentence);
  const [tokensList, setTokensList] = useState<string[][]>([[]]);
  const [comparisons, setComparisons] = useState([{ model: defaultModel, task: defaultTask, selected: false }]);
  const [results, setResults] = useState([dummyImage]);
  const [layerCounts, setLayerCounts] = useState<number[]>([12]);
  const [maskedTokensList, setMaskedTokensList] = useState<string[][]>([[]]);
  const [predictData, setPredictData] = useState([{ tokens: [], probs: [], grads: [] }]);
  const [showProbs, setShowProbs] = useState([true]);
  const [showGrads, setShowGrads] = useState([false]);
  const [selectedLayers, setSelectedLayers] = useState<number[]>([-1]);
  const [selectedTokens, setSelectedTokens] = useState<number[]>([-1]);
  const [selectedMaskTokens, setSelectedMaskTokens] = useState<number[]>([-1]);
  const [maskTokens, setMaskTokens] = useState<string[]>(['[MASK]']);
  const [removeWarning, setRemoveWarning] = useState(false);
  const extendList = <T,>(list: T[], newValue: T): T[] => [...list, newValue];




  const handleLayerSelect = (idx: number, layer: number) => {
    const updated = [...selectedLayers];
    updated[idx] = layer;
    setSelectedLayers(updated);


  };
    
  const handleTokenSelect = (idx: number, token: number) => {
    const updated = [...selectedTokens];
    updated[idx] = token;
    setSelectedTokens(updated);

 
  };

  const handleInputChange = (index, key, value) => {
    const newComps = [...comparisons];
    const oldTask = newComps[index].task;
  
    newComps[index][key] = value;
    setComparisons(newComps);
  
    // Determine if we need to reset masked tokens
    const isNowMLM = (key === "task" && value === "MLM") || (key === "model" && newComps[index].task === "MLM");
    const wasNotMLM = oldTask !== "MLM";

    if (isNowMLM && wasNotMLM) {
      const newMasked = [...maskedTokensList];
      newMasked[index] = [...tokensList[index]];
      setMaskedTokensList(newMasked);
      console.log(`🔄 Resetting masked tokens for scenario ${index}`);
    }

  }

  const handleMaskToken = (idx, tokenIndex) => {
    const originalTokens = tokensList[idx];
    const newTokens = [...originalTokens];
    const maskedToken = originalTokens[tokenIndex];
  
    // Apply the mask
    newTokens[tokenIndex] = maskTokens[idx];
    const newSentence = newTokens.join(" ");
  
    console.log(`🔍 Masking token "${maskedToken}" at position ${tokenIndex}`);
    console.log(`📤 Prepared masked sentence: "${newSentence}"`);
  
    // Update maskedTokensList only (NOT tokensList)
    if (comparisons[idx].task === "MLM") {
      const newMasked = [...maskedTokensList];
      newMasked[idx] = newTokens;
      setMaskedTokensList(newMasked);
    }
  
    // Update selectedMaskTokens
    const newSelectedMasks = [...selectedMaskTokens];
    newSelectedMasks[idx] = tokenIndex;
    setSelectedMaskTokens(newSelectedMasks);
  };
  

  const handleGo = () => {
    setResults(Array(comparisons.length).fill(dummyImage));
    comparisons.forEach((comp, idx) => updateBackendInfo(comp, idx));
  };
  
  
 

const addComparison = () => {

 


  
  if (comparisons.length < 3) {
    const prev = comparisons[comparisons.length - 1] || {
      model: defaultModel,
      task: defaultTask,
    };

    const newComparison = { model: prev.model, task: prev.task, selected: false };

    setComparisons(extendList(comparisons, newComparison));
    setLayerCounts(extendList(layerCounts, 12));
    setResults(extendList(results, dummyImage));
    setTokensList(extendList(tokensList, []));
    setMaskedTokensList(extendList(maskedTokensList, []));
    setSelectedLayers(extendList(selectedLayers, -1));
    setSelectedTokens(extendList(selectedTokens, -1));
    setPredictData(extendList(predictData, { tokens: [], probs: [], grads: [] }));
    setShowProbs(extendList(showProbs, true));
    setShowGrads(extendList(showGrads, false));
    setSelectedMaskTokens(extendList(selectedMaskTokens, -1));
    setMaskTokens(extendList(maskTokens, '[MASK]'));
  }
};


const removeComparison = () => {
  const hasSelection = comparisons.some((c) => c.selected);

  if (!hasSelection) {
    setRemoveWarning(true);
    return;
  }

  setRemoveWarning(false); // clear warning if selection exists

  if (comparisons.length === 2) {
    setComparisons(comparisons.filter((_, idx) => !comparisons[idx].selected));
    setResults(results.slice(0, comparisons.length - 1));
  } else if (comparisons.length > 1) {
    setComparisons(comparisons.slice(0, -1));
    setResults(results.slice(0, -1));
  }
};


  const toggleSelection = (index) => {
    const newComps = [...comparisons]
    newComps[index].selected = !newComps[index].selected
    setComparisons(newComps)
  }


 const updateBackendInfo = async (comp, idx) => {
  const isMLM = comp.task === "MLM";
  const effectiveSentence = isMLM
    ? maskSentence(sentence, selectedMaskTokens[idx], tokensList[idx], maskTokens[idx])
    : sentence;

  console.log(`🧠 Sending text to /load_model: "${effectiveSentence}"`);

  const loadRes = await fetch("http://localhost:8000/load_model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: comp.model,
      task: comp.task,
      sentence: effectiveSentence,
      selected_layer: selectedLayers[idx],
      selected_token: selectedTokens[idx],
    }),
  });

  const loadData = await loadRes.json();
  console.log("✅ /load_model returned:", loadData);

  if (!loadData.error) {
    const newLayers = [...layerCounts];
    newLayers[idx] = loadData.num_layers;
    setLayerCounts(newLayers);

    if (tokensList[idx].length === 0) {
      const newTokens = [...tokensList];
      newTokens[idx] = loadData.tokens;
      setTokensList(newTokens);
    }

    if (comp.task === "MLM") {
      const newMasked = [...maskedTokensList];
      newMasked[idx] = loadData.tokens;
      setMaskedTokensList(newMasked);
    }

    const newMaskTokens = [...maskTokens];
    newMaskTokens[idx] = loadData.mask_token || newMaskTokens[idx];
    setMaskTokens(newMaskTokens);
  }

  
    const predictRes = await fetch("http://localhost:8000/predict_model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: comp.model, task: comp.task, sentence: effectiveSentence, selected_layer: selectedLayers[idx], selected_token: selectedTokens[idx] })
    });
    const predictJson = await predictRes.json();

    const newPredictData = [...predictData];
    newPredictData[idx] = {
      tokens: predictJson.decoded,
      probs: predictJson.top_probs,
      grads: predictJson.grads,
    };
    setPredictData(newPredictData);
  };
 

  useEffect(() => {
    setResults(Array(comparisons.length).fill(dummyImage))
  }, [comparisons.length])

 





  return (
    <div className="page">
      <div className="container">

        <h1 className="text-2xl font-bold mb-4">Transformers: gradients and graphs</h1>

            <div className="centered-stack">
              <div
                className="card input-dynamic-width"
                style={{
                  width: comparisons.length === 1
                    ? '800px'
                    : `min(${comparisons.length * 820}px, 100vw)`, // estimates card + gap
                }}
              >

                <p>
                  Type a sample passage here.    (Best performance if fewer than 500 words.)
                </p>
                <div className="input-row">
                <textarea
                  className="input-box"
                  placeholder="e.g. The quick brown fox jumps over the lazy dog."
                  value={sentence}
                  onChange={(e) => setSentence(e.target.value)}
                  rows={4}
                />
              <button className="button" onClick={handleGo}>Apply</button>
              </div>
                <p></p>

            </div> 
            <div className="comparison-scroll-container">

              <div style={{ fontSize: '0.8rem', color: '#4b5563', marginTop: '1rem' }}>
                    <div><strong>Tokens:</strong> {JSON.stringify(predictData[0].tokens)}</div>
                    <div><strong>Probs:</strong> {JSON.stringify(predictData[0].probs)}</div>
                    <div><strong>Grads:</strong> {JSON.stringify(predictData[0].grads)}</div>
                  </div>
              <div className="comparison-row">
                {comparisons.map((comp, idx) => (
                  
 
                  

                <ScenarioCard
                  key={idx}
                  idx={idx}
                  comparison={comp}
                  tokens={tokensList[idx]}
                  maskedTokens={maskedTokensList[idx]}
                  layerCount={layerCounts[idx]}
                  selectedLayer={selectedLayers[idx]}
                  selectedToken={selectedTokens[idx]}
                  selectedMaskToken={selectedMaskTokens[idx]}
                  maskToken={maskTokens[idx]}
                  predict={predictData[idx]}
                  showProb={showProbs[idx]}
                  showGrad={showGrads[idx]}
                  onInputChange={handleInputChange}
                  onApply={(i) => updateBackendInfo(comparisons[i], i)}
                  onToggleSelected={toggleSelection}
                  onLayerSelect={handleLayerSelect}
                  onTokenSelect={handleTokenSelect}
                  onMaskToken={handleMaskToken}
                  onToggleProb={(i) => {
                    const updated = [...showProbs];
                    updated[i] = !updated[i];
                    setShowProbs(updated);
                  }}
                  onToggleGrad={(i) => {
                    const updated = [...showGrads];
                    updated[i] = !updated[i];
                    setShowGrads(updated);
                  }}
                  selected={comp.selected}
                  onToggle={() => toggleSelection(idx)}
                  showCheckbox={true}
                />
              ))}
            </div> 
          </div>
        

          <div className="button-row">
            {comparisons.length < 3 && (
              <button className="button" onClick={addComparison}>Add a scenario</button>
            )}
            {comparisons.length > 1 && (
              <button className="button" onClick={removeComparison}>Remove a scenario</button>
            )}
              {removeWarning && <p className="warning-text">Select a scenario for removal.</p>}

          </div>
        </div> 
      </div>
    </div>

  )
}
