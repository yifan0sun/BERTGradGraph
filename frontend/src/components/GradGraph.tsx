import React, { useEffect, useState } from 'react';
const SERVER_ADDRESS = 'http://localhost:8000';
import Plot from 'react-plotly.js';
import '../App.css';

import {cleanToken, displayToken} from '../utils.ts'

export function GraphGrad  ({ tokens, model, task, layer,sentence,matrix }: {
  tokens: string[];
  model: string;
  task: string;
  layer: number;
  sentence:string;
  matrix: number[][];
})  {
const [visibleLeftTokens, setVisibleLeftTokens] = useState<boolean[]>([]);
const [visibleRightTokens, setVisibleRightTokens] = useState<boolean[]>([]);


const [selectedTokenIdx, setSelectedTokenIdx] = useState<number | null>(null);


useEffect(() => {
  setVisibleLeftTokens(tokens.map(() => true));
  setVisibleRightTokens(tokens.map(() => true));
}, [tokens]);

 

  if (null) {
    return <p style={{ textAlign: 'center', fontStyle: 'italic', color: '#555' }}>Computing...
    
    matrix length {matrix.length}, token length {tokens.length}  {tokens.map((tok, idx) => (
      <li key={idx}>
        <strong>{idx}</strong>: {tok}
      </li>
    ))} < /p>;
  }
// DEBUG: show matrix in UI
const matrixDebug = (
  <div style={{ fontFamily: 'monospace', fontSize: '12px', maxHeight: '200px', overflowY: 'scroll', marginBottom: '1rem' }}>
    <strong>Matrix Debug (first 10 rows):</strong>
    <pre>
      {matrix.slice(0, 10).map((row, i) => `Row ${i}: ${row.slice(0, 10).map(v => v.toFixed(2)).join(', ')}`).join('\n')}
    </pre>
  </div>
);

  const colSums = Array(tokens.length).fill(0);
  for (let j = 0; j < tokens.length; j++) {
    for (let i = 0; i < tokens.length; i++) {
      if (visibleLeftTokens[i]) colSums[j] += matrix[i][j];
    }
  }
 const normalizedGradMatrix = matrix.map((row, i) =>
    row.map((val, j) =>
      visibleLeftTokens[i] && colSums[j] > 0 ? val / colSums[j] : 0
    )
  );

  const xLeft = .0;
  const xRight = 3;
  const nodeSpacing = 1.5;
  const graphNodeSpacing = 1.0;
  const n = tokens.length;

  const xSrc = tokens.map(() => xLeft);
  const ySrc = tokens.map((_, i) => i * graphNodeSpacing);
  const xTgt = tokens.map(() => xRight);
  const yTgt = tokens.map((_, i) => i * graphNodeSpacing);



for (let j = 0; j < tokens.length; j++) {
    // Collect contributions from *checked* left tokens
    let columnSum = 0;
    for (let i = 0; i < tokens.length; i++) {
        if (visibleLeftTokens[i]) {
        columnSum += matrix[i][j];
        }
    }

    for (let i = 0; i < tokens.length; i++) {
        if (visibleLeftTokens[i] && columnSum > 0) {
        normalizedGradMatrix[i][j] = matrix[i][j] / columnSum;
        } else {
        normalizedGradMatrix[i][j] = 0;
        }
    }
}



  const edges = [];
  for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const isVisible = visibleLeftTokens[i] && visibleRightTokens[j];

            const w = normalizedGradMatrix[i][j];
            const isConnected =
            selectedTokenIdx === null || selectedTokenIdx === i || selectedTokenIdx === j;
            if (w > 0.01 && isConnected && isVisible) {
                edges.push({
                x: [xLeft, xRight],
                y: [i * graphNodeSpacing, j * graphNodeSpacing],
                line: { width: w * 10, color: 'rgba(50,50,150,0.3)' },
                mode: 'lines',
                type: 'scatter',
                hoverinfo: 'skip',
                hovertemplate: null,
                text: [`Weight: ${w.toFixed(3)}`],
                showlegend: false,
                });
            }
        }
  }

  const srcLabels = {
    x: xSrc.map(x => x - 0.25),
    y: ySrc,
    text: tokens.map(cleanToken).map(displayToken),
    mode: 'text',
    type: 'scatter',
    textposition: 'middle left',
    textfont: { size: 14 },
    showlegend: false,
    hoverinfo: 'skip',
    hovertemplate: null,
  };

  const tgtLabels = {
    x: xTgt.map(x => x + 0.25),
    y: yTgt,
    text: tokens.map(cleanToken).map(displayToken),
    mode: 'text',
    type: 'scatter',
    textposition: 'middle right',
    textfont: { size: 14 },
    showlegend: false,
    hoverinfo: 'skip',
    hovertemplate: null,
  };

  const nodeDots = {
    x: [...xSrc, ...xTgt],
    y: [...ySrc, ...yTgt],
    mode: 'markers',
    type: 'scatter',
    marker: { color: 'blue', size: 15 },
    hoverinfo: 'skip',
    showlegend: false,
    hovertemplate: null,
  };

  return (
  <>
  {/*matrixDebug*/}
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }}>
      {/* Left checkboxes */}
      
      <div style={{ marginRight: '1rem', textAlign: 'right' }}>
        {tokens.map((token, i) => (
          <div key={`left-${i}`} style={{ height: `${nodeSpacing * 20}px`, display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
            <span style={{ marginRight: '6px' }}>{cleanToken(token)}</span>
            <input
              type="checkbox"
              style={{ width: '15px', height: '15px' }}
              checked={visibleLeftTokens[i]}
              onChange={(e) => {
                const newLeftVisibility = [...visibleLeftTokens];
                newLeftVisibility[i] = e.target.checked;
                setVisibleLeftTokens(newLeftVisibility);
              }}
            />
          </div>
        ))}
      </div>

      {/* Plotly graph */}
      
    
      <div style={{ maxWidth: '800px', position: 'relative' }}>
        <Plot
          data={[...edges, srcLabels, tgtLabels, nodeDots]}
          style={{ cursor: 'default' }}
          layout={{
            title: 'Gradient Flow Bipartite Graph',
            showlegend: false,
            margin: { l: 0, r: 0, t: 10, b: 10 },
            xaxis: { showticklabels: false, zeroline: false, range: [xLeft - 1, xRight + 1], fixedrange: true },
            yaxis: { showticklabels: false, zeroline: false, fixedrange: true },
            height: Math.max(400, tokens.length * 15),
            hovermode: false,
          }}
          config={{
            scrollZoom: false,
            doubleClick: false,
            displayModeBar: false,
            staticPlot: false,
          }}
        />
      </div>
      

      {/* Right checkboxes */}
      
      <div style={{ marginLeft: '1rem' }}>
        {tokens.map((token, i) => (
          <div key={`right-${i}`} style={{ height: `${nodeSpacing * 20}px`, display: 'flex', alignItems: 'center' }}>
            <input
              type="checkbox"
              style={{ width: '15px', height: '15px' }}
              checked={visibleRightTokens[i]}
              onChange={(e) => {
                const newRightVisibility = [...visibleRightTokens];
                newRightVisibility[i] = e.target.checked;
                setVisibleRightTokens(newRightVisibility);
              }}
            />
            <span style={{ marginLeft: '6px' }}>{cleanToken(token)}</span>
          </div>
        ))}
      </div>
      
      
    </div>
 
  </>
);

};
