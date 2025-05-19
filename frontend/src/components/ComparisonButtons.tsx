import React from 'react'

export default function ComparisonButtons({
  tokens,
  maskedTokens,
  layerCount,
  model,
  task,
  onMaskToken,
  selectedLayer,
  selectedToken,
  onLayerSelect,
  onTokenSelect,
  selectedMaskToken
}: {
  tokens: string[]
  maskedTokens: string[]
  layerCount: number
  model: string
  task: string
  onMaskToken: (index: number) => void
  selectedLayer: number
  selectedToken: number
  onLayerSelect: (index: number) => void
  onTokenSelect: (index: number) => void
  selectedMaskToken: number
}) {
  
 
  return (
    <div className="layer-stack">

      {/*<p style={{ fontSize: '0.8rem', color: '#6b7280', wordWrap: 'break-word' }}>
        Debug tokens: {JSON.stringify(tokens)}
      </p> 
      <p style={{ fontSize: '0.8rem', color: '#4b5563' }}>
        Layers: <strong>{layerCount}</strong>
      </p>*/}
 
    {task === "MLM" && (
      <>
      
        <p style={{ fontWeight: 500, marginBottom: "0.5rem" }}>Select mask:</p>
        <div className="word-row" style={{ marginBottom: '1.5rem' }}>
          {tokens.map((word, i) => (
            <button
              key={i}
              className={`word-button ${selectedMaskToken === i ? "selected" : ""}`}
              onClick={() => onMaskToken(i)}
            >
              {word}
            </button>
            
          ))}
        </div>
 
      </>
    )}



<p style={{ fontWeight: 500, marginTop: '1rem' }}>Pick target layer and token:</p>
      <div className="layer-row">
      <button
        className={`layer-button input ${selectedLayer === -1 ? 'selected' : ''}`}
        onClick={() => onLayerSelect(-1)}
      >Input</button>

        {[...Array(layerCount)].map((_, i) => (
           <button
              key={i}
              className={`layer-button ${selectedLayer === i ? 'selected' : ''}`}
              onClick={() => onLayerSelect(i)}
            >
      Layer {i + 1}</button>
        ))}
        <button
            className={`layer-button output ${selectedLayer === -2 ? 'selected' : ''}`}
            onClick={() => onLayerSelect(-2)}
          >
            Output</button>
      </div>

      <div className="word-row">
        {(task === "MLM" ? maskedTokens : tokens).map((word, i) => (
          <button
          key={i}
          className={`word-button ${selectedToken === i ? 'selected' : ''}`}
          onClick={() => onTokenSelect(i)}
          >
          {word}</button>
        ))}
      </div>
    </div>

  )
}