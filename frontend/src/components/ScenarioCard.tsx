// components/ScenarioCard.tsx

import { MaskButtons, ComparisonButtons } from './ComparisonButtons';
import { ComparisonDisplayPredict } from './ComparisonDisplay';

export default function ScenarioCard({
  idx,
  comparison,
  tokens,
  maskedTokens,
  layerCount,
  selectedLayer,
  selectedToken,
  selectedMaskToken,
  predict,
  showProb,
  showGrad,
  onInputChange,
  onApply,
  onToggleSelected,
  onLayerSelect,
  onTokenSelect,
  onMaskToken,
  onToggleProb,
  onToggleGrad,
  selected,
  onToggle,
  showCheckbox,
}) {
  return (
    <div className="scenario-card">
      <div className="scenario-inner-card">
        {showCheckbox && (
        <div className="mt-2">
          <label className="inline-flex items-center">
            <input
              type="checkbox"
              checked={selected}
              onChange={onToggle}
              className="mr-2"
            />
            Select for Removal
          </label>
        </div>
      )}
        <div className="dropdown-row">
          <div className="dropdown-group">
            <label className="dropdown-label">Pick model</label>
            <select
              value={comparison.model}
              onChange={(e) => onInputChange(idx, 'model', e.target.value)}
            >
              <option value="">Select Model</option>
              <option>BERT</option>
              <option>BART</option>
              <option>RoBERTa</option>
              <option>DistilBERT</option>
            </select>
          </div>

          <div className="dropdown-group">
            <label className="dropdown-label">Pick task</label>
            <select
              value={comparison.task}
              onChange={(e) => onInputChange(idx, 'task', e.target.value)}
            >
              <option value="">Select Task</option>
              <option>MLM</option>
              <option>NSP</option>
              <option>SST2</option>
              <option>SQUAD</option>
            </select>
          </div>

          <div className="apply-button-container">
            <button className="button" onClick={() => onApply(idx)}>
              Apply
            </button>
          </div>
        </div>

        <MaskButtons
          tokens={tokens}
          task={comparison.task}
          selectedMaskToken={selectedMaskToken}
          onMaskToken={(tokenIdx) => onMaskToken(idx, tokenIdx)}
        />

        <ComparisonDisplayPredict
          image={null}
          label={`Result ${idx + 1}`}
          predict={predict}
          showProb={showProb}
          showGrad={showGrad}
          onToggleProb={() => onToggleProb(idx)}
          onToggleGrad={() => onToggleGrad(idx)}
        />
      </div>

      <div className="scenario-inner-card">
        <ComparisonButtons
          tokens={tokens}
          maskedTokens={maskedTokens}
          layerCount={layerCount}
          task={comparison.task}
          selectedLayer={selectedLayer}
          selectedToken={selectedToken}
          onLayerSelect={(layerIdx) => onLayerSelect(idx, layerIdx)}
          onTokenSelect={(tokenIdx) => onTokenSelect(idx, tokenIdx)}
        />

        
      </div>
    </div>
  );
}