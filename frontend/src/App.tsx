import React, { useState, useEffect } from 'react'
import dummyImage from './assets/dummy.png'
import ComparisonButtons from './components/ComparisonButtons.tsx'
import ComparisonDisplay from './components/ComparisonDisplay.tsx'

export default function App() {
  const defaultSentence = 'The quick brown fox jumps over the lazy dog.'
  const defaultModel = 'BERT'
  const defaultTask = 'MLM'

  const [sentence, setSentence] = useState(defaultSentence)
  const [comparisons, setComparisons] = useState([{ model: defaultModel, task: defaultTask, selected: false }])
  const [results, setResults] = useState([dummyImage])
  const [layerCounts, setLayerCounts] = useState<number[]>([12]) // dummy: one per comparison

  const modelLayers = {
    BERT: 12,
    BART: 6,
    RoBERTa: 12,
    DistilBERT: 6,
  }

  const handleInputChange = (index, key, value) => {
    const newComps = [...comparisons]
    newComps[index][key] = value
    setComparisons(newComps)

    if (key === 'model' && modelLayers[value]) {
      const newLayers = [...layerCounts]
      newLayers[index] = modelLayers[value]
      setLayerCounts(newLayers)
    }
  }


  const handleGo = () => {
    setResults(Array(comparisons.length).fill(dummyImage))
  }

  const addComparison = () => {
  if (comparisons.length < 3) {
    const prev = comparisons[comparisons.length - 1]
    setComparisons([...comparisons, {
      model: prev.model,
      task: prev.task,
      selected: false,
    }])
    setLayerCounts([...layerCounts, layerCounts[layerCounts.length - 1]])
    setResults([...results, dummyImage])
  }
}


  const removeComparison = () => {
    if (comparisons.length === 2) {
      setComparisons(comparisons.filter((_, idx) => !comparisons[idx].selected))
      setResults(results.slice(0, comparisons.length - 1))
    } else if (comparisons.length > 1) {
      setComparisons(comparisons.slice(0, -1))
      setResults(results.slice(0, -1))
    }
  }

  const toggleSelection = (index) => {
    const newComps = [...comparisons]
    newComps[index].selected = !newComps[index].selected
    setComparisons(newComps)
  }


  useEffect(() => {
    setResults(Array(comparisons.length).fill(dummyImage))
  }, [comparisons.length])

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold mb-4">Sentence Comparison Tool</h1>

      <div className="flex flex-col gap-4">
        <input
          type="text"
          placeholder="Enter a sentence..."
          value={sentence}
          onChange={(e) => setSentence(e.target.value)}
          className="p-2 border rounded"
        />

        <div className="flex gap-4 flex-wrap">
          {comparisons.map((comp, idx) => (
            <div key={idx} className="flex flex-col gap-2">
              <h2 className="font-semibold text-lg">Comparison {idx + 1}</h2>
              <select
                value={comp.model}
                onChange={(e) => handleInputChange(idx, 'model', e.target.value)}
                className="p-2 border rounded"
              >
                <option value="">Select Model</option>
                <option>BERT</option>
                <option>BART</option>
                <option>RoBERTa</option>
                <option>DistilBERT</option>

              </select>

              <select
                value={comp.task}
                onChange={(e) => handleInputChange(idx, 'task', e.target.value)}
                className="p-2 border rounded"
              >
                <option value="">Select Task</option>
                <option>MLM</option>
                <option>NSP</option>
                <option>SST2</option>
                <option>SQUAD</option>

              </select>
            </div>
          ))}
        </div>

        <button
          onClick={handleGo}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Go
        </button>
      </div>

      {results.length > 0 && (
        <div className="flex gap-4 mt-6">
          {results.map((res, idx) => (
            <div key={idx} className="border p-4 rounded shadow flex flex-col items-center">
              <h2 className="font-semibold mb-2">Comparison {idx + 1}</h2>
              <div className="flex flex-col items-center gap-4">
                <ComparisonButtons sentence={sentence} layerCount={layerCounts[idx]} />
                <ComparisonDisplay
                  image={res}
                  selected={comparisons[idx].selected}
                  onToggle={() => toggleSelection(idx)}
                  showCheckbox={comparisons.length > 1}
                  label={`Result ${idx + 1}`}
                />
              </div>

            </div>
          ))}
        </div>
      )}

      <div className="mt-6 space-x-4">
        {comparisons.length < 3 && (
          <button
            onClick={addComparison}
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
          >
            Add Side by Side Comparison
          </button>
        )}
        {comparisons.length > 1 && (
          <button
            onClick={removeComparison}
            className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
          >
            Remove a Comparison
          </button>
        )}
      </div>
    </div>
  )
}
