// components/ComparisonDisplay.tsx
import React from 'react';
import Plot from 'react-plotly.js';

export  function ComparisonDisplayPredict({
  image,
  label,
  predict,
  showProb,
  showGrad,
  onToggleProb,
  onToggleGrad,
}: {
  image: string | null;
  label: string;
  predict: { tokens: string[]; probs: number[]; grads: number[] };
  showProb: boolean;
  showGrad: boolean;
  onToggleProb: () => void;
  onToggleGrad: () => void;
}) {
  return (
    <div className="flex flex-col items-center">
      {image && (
        <img src={image} alt={label} className="w-64 h-64 object-contain" />
      )}


      {predict.tokens.length > 0 && (
        <div style={{ marginTop: '1rem' }}>
          <Plot
            data={[
              showProb && {
                x: predict.tokens,
                y: predict.probs,
                type: 'bar',
                name: 'Probability',
                marker: { color: '#3b82f6' },
              },
              showGrad && {
                x: predict.tokens,
                y: predict.grads,
                type: 'bar',
                name: 'Grad Norm',
                marker: { color: '#facc15' },
              },
            ].filter(Boolean)}
            layout={{ title: 'Top Predicted Tokens', barmode: 'group' }}
          />
          <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem' }}>
            <label>
              <input type="checkbox" checked={showProb} onChange={onToggleProb} />
              Probability
            </label>
            <label>
              <input type="checkbox" checked={showGrad} onChange={onToggleGrad} />
              Grad Norm
            </label>
          </div>
        </div>
      )}

      
    </div>
  );
}
