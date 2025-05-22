import '../App.css';

export function BottomPanel({
  numLayers,
  selectedLayer,
  setSelectedLayer,
  mode,
  setMode,
}: {
  numLayers: number;
  selectedLayer: number;
  setSelectedLayer: (layer: number) => void;
  mode: 'attention' | 'gradient';  // ✅ CORRECT here
  setMode: (mode: 'attention' | 'gradient') => void;
}) {

   


  const layerOptions = Array.from({ length: numLayers }, (_, i) => i + 1);



 


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
          onChange={async (e) => {
            const newLayer = parseInt(e.target.value);
            setSelectedLayer(newLayer);
          }}
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
            onChange={async () => {
              setMode('attention');
            }}
          />
          Attention
        </label>
        <label>
          <input
            type="radio"
            value="gradient"
            checked={mode === 'gradient'}            
            onChange={async () => {
              setMode('gradient');
            }}
          />
          Gradient Norm
        </label>
      </div>

       
    </div>
  );
}
