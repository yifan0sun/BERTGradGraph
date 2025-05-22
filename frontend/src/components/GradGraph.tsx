import React, { useEffect, useState } from 'react';

import Plot from 'react-plotly.js';
import '../App.css';

import { cleanToken, displayToken } from '../utils.ts';

export function GraphGrad({
  tokens,
  matrices,
  layer,
  mode,
}: {
  tokens: string[];
  layer: number;
  mode: 'attention' | 'gradient';
  matrices: {
    attention: number[][][];
    gradient: number[][][];
  };
}) {

  const [visibleLeftTokens, setVisibleLeftTokens] = useState<boolean[]>([]);
  const [visibleRightTokens, setVisibleRightTokens] = useState<boolean[]>([]);
  const [selectedTokenIdx, setSelectedTokenIdx] = useState<number | null>(null);


  const [matrix, setMatrix] = useState<number[][]>([]);





  useEffect(() => {
    setVisibleLeftTokens(tokens.map(() => true));
    setVisibleRightTokens(tokens.map(() => true));
  }, [tokens]);

 

  if (matrices[mode]?.[layer - 1]?.length !== tokens.length || tokens.length === 0) {
    return (
      <div style={{ padding: '1rem', fontSize: '12px', fontStyle: 'italic', color: '#900' }}>
        ⚠️ Graph could not be rendered.<br />
        matrix.length = {matrix.length},{matrices[mode]?.[layer - 1]?.length}, tokens.length = {tokens.length}
        <ul>
          {tokens.map((tok, idx) => (
            <li key={idx}>
              <strong>{idx}</strong>: {tok}
            </li>
          ))}
        </ul>
      </div>
    );
  }

   useEffect(() => {
  const newMatrix = matrices[mode]?.[layer - 1];
  if (Array.isArray(newMatrix)) {
    setMatrix(newMatrix);

  } else {
    setMatrix([]);

  }
}, [matrices, mode, layer, tokens.length]);



if (
  !Array.isArray(matrix) ||
  matrix.length !== tokens.length ||
  matrix[0]?.length !== tokens.length
) {
  return (
    <div style={{ padding: '1rem', fontSize: '12px', fontStyle: 'italic', color: '#900' }}>
      ⚠️ Graph skipped: matrix shape is invalid or incomplete.<br />
      matrix.length = {matrix?.length ?? 'undefined'},{matrices[mode]?.[layer - 1]?.length}, expected = {tokens.length}, 
      mode = {mode}, layer =  {layer}
    </div>
  );
}



  const colSums = Array(tokens.length).fill(0);
  for (let j = 0; j < tokens.length; j++) {
    for (let i = 0; i < tokens.length; i++) {
      if (visibleLeftTokens[i]) colSums[j] += matrix[i][j];
    }
  }

  const isMatrixValid = (
    Array.isArray(matrix) &&
    matrix.length === tokens.length &&
    matrix.every(row => Array.isArray(row) && row.length === tokens.length)
  );

  if (!isMatrixValid) {
    return (
      <div style={{ color: '#900', fontSize: '12px', padding: '1rem' }}>
        ❌ Invalid matrix structure. Cannot render graph.
      </div>
    );
  }


  const normalizedGradMatrix = matrix.map((row, i) =>
    row.map((val, j) =>
      visibleLeftTokens[i] && colSums[j] > 0 ? val / colSums[j] : 0
    )
  );

  for (let j = 0; j < tokens.length; j++) {
    let columnSum = 0;
    for (let i = 0; i < tokens.length; i++) {
      if (visibleLeftTokens[i]) {
        columnSum += matrix[i][j];
      }
    }
    for (let i = 0; i < tokens.length; i++) {
      normalizedGradMatrix[i][j] =
        visibleLeftTokens[i] && columnSum > 0 ? matrix[i][j] / columnSum : 0;
    }
  }

  const xLeft = 0.0;
  const xRight = 3;
  const nodeSpacing = 1.25;
  const graphNodeSpacing = 1.25;
  const n = tokens.length;

  const xSrc = tokens.map(() => xLeft);
  const ySrc = tokens.map((_, i) => (tokens.length - 1 - i) * graphNodeSpacing);
  const xTgt = tokens.map(() => xRight);
 const yTgt = tokens.map((_, i) => (tokens.length - 1 - i) * graphNodeSpacing);

  const EDGE_THRESHOLD = 0.1;     // Only plot edges with normalized weight > 0.05
const EDGE_WIDTH_MULT = 100;     // Increase thickness exaggeration

 


  let edges: any[] = [];

if (
  Array.isArray(normalizedGradMatrix) &&
  normalizedGradMatrix.length === tokens.length &&
  normalizedGradMatrix.every(row => Array.isArray(row) && row.length === tokens.length)
) {
  edges = [];
  for (let i = 0; i < tokens.length; i++) {
    for (let j = 0; j < tokens.length; j++) {
      const isVisible = visibleLeftTokens[i] && visibleRightTokens[j];
      const w = normalizedGradMatrix[i]?.[j] ?? 0;
      const isConnected =
        selectedTokenIdx === null || selectedTokenIdx === i || selectedTokenIdx === j;
      if (w > EDGE_THRESHOLD && isConnected && isVisible) {
        edges.push({
          x: [xLeft, xRight],
          y: [(tokens.length - 1 - i) * graphNodeSpacing, (tokens.length - 1 - j) * graphNodeSpacing],

          line: { width: (w ** 2) * EDGE_WIDTH_MULT, color: 'rgba(50,50,150,0.3)' },
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
}



  const srcLabels = {
    x: xSrc.map((x) => x - 0.25),
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
    x: xTgt.map((x) => x + 0.25),
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
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }}>

{/*
        
        <div
          style={{
            fontSize: '12px',
            fontFamily: 'monospace',
            marginBottom: '1rem',
            backgroundColor: '#f4f4f4',
            padding: '0.5rem',
            border: '1px solid #ccc',
          }}
        >
          <strong>GraphGrad Debug:</strong><br />
          Layer: {layer}<br />
          Mode: {mode}<br />
          Tokens: {tokens.length}<br />
          Matrices Layers: attention={matrices.attention.length}, gradient={matrices.gradient.length}<br />
          Matrix shape: {matrices[mode]?.[layer - 1]?.length ?? 0} × {matrices[mode]?.[layer - 1]?.[0]?.length ?? 0}
          Matrix shape: {matrices[mode]?.[layer - 1]?.length ?? 0} × {matrices[mode]?.[layer - 1]?.[0]?.length ?? 0}<br /><br />

  <strong>Selected Matrix Sample (first 5×5):</strong>
{matrices[mode]?.[layer - 1]?.length >= 5 &&
 matrices[mode]?.[layer - 1]?.[0]?.length >= 5 && (
  <pre style={{ whiteSpace: 'pre-wrap' }}>
    {matrices[mode][layer - 1]
      .slice(0, 5)
      .map((row) =>
        row.slice(0, 5).map((val) => val.toFixed(2)).join(', ')
      )
      .join('\n')}
  </pre>
)}

        </div>

<div style={{ fontSize: '12px', fontFamily: 'monospace', marginBottom: '1rem' }}>
  ✅ Matrices received for layer {layer}:<br />
  - Attention: {matrices.attention?.[layer - 1]?.length ?? 0} × {matrices.attention?.[layer - 1]?.[0]?.length ?? 0}<br />
  - Gradient: {matrices.gradient?.[layer - 1]?.length ?? 0} × {matrices.gradient?.[layer - 1]?.[0]?.length ?? 0}
</div>

 */}



<div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }}>

        {/* Left checkboxes */}
        
        <div style={{ marginRight: '1rem', textAlign: 'right' }}>
          {tokens.map((token, i) => (
            <div
              key={`left-${i}`}
              style={{
                height: `${nodeSpacing * 20}px`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'flex-end',
              }}
            >
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
          



        
         








 

        <div style={{ maxWidth: '800px', position: 'relative' }}>
          
         
          <Plot
            key={`plot-${layer}-${mode}-${tokens.length}-${JSON.stringify(matrix[0] ?? [])}`}
            data={[...edges, srcLabels, tgtLabels, nodeDots]}
            style={{ cursor: 'default' }}
            layout={{
              title: 'Gradient Flow Bipartite Graph',
              showlegend: false,
              margin: { l: 0, r: 0, t: 10, b: 10 },
              xaxis: {
                showticklabels: false,
                zeroline: false,
                range: [xLeft - 1, xRight + 1],
                fixedrange: true,
              },
              yaxis: {
                showticklabels: false,
                zeroline: false,
                fixedrange: true,
              },
              height: Math.max(400, tokens.length * 15),
              hovermode: false,
            }}
            config={{
              scrollZoom: false,
              doubleClick: false,
              displayModeBar: false,
              staticPlot: false,
            }}

            onInitialized={() => {
              console.log('✅ Plotly initialized');
            }}
            onUpdate={() => {
              console.log('✅ Plotly updated');
            }}

          />
        </div>

        

         {/* Right checkboxes */}
         
        <div style={{ marginLeft: '1rem' }}>
          {tokens.map((token, i) => (
            <div
              key={`right-${i}`}
              style={{
                height: `${nodeSpacing * 20}px`,
                display: 'flex',
                alignItems: 'center',
              }}
            >
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
    </div>
      

 </>
  );
}
