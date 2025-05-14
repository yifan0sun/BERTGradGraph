import React, { useState, useEffect } from 'react'
import dummyImage from './assets/dummy.png'
import './App.css'

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

  const [tokensList, setTokensList] = useState<string[][]>(comparisons.map(() => []))


  const handleInputChange = (index, key, value) => {
    const newComps = [...comparisons]
    newComps[index][key] = value
    setComparisons(newComps)

    updateBackendInfo(newComps[index], index)

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
    setTokensList([...tokensList, []])

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


  const updateBackendInfo = async (comp, idx) => {
    const res = await fetch("http://localhost:8000/load_model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: comp.model,
        task: comp.task,
        sentence,
      }),
    })
    const data = await res.json()
    if (!data.error) {
      const newLayers = [...layerCounts]
      newLayers[idx] = data.num_layers
      setLayerCounts(newLayers)

      const newTokens = [...tokensList]
      newTokens[idx] = data.tokens
      setTokensList(newTokens)
    }
  }


  useEffect(() => {
    setResults(Array(comparisons.length).fill(dummyImage))
  }, [comparisons.length])

  useEffect(() => {
  comparisons.forEach((comp, idx) => updateBackendInfo(comp, idx))
}, [sentence])

  return (
    <div className="min-h-screen bg-[#f8f9fa] font-serif">
  <   div className="max-w-5xl mx-auto py-12 px-4 sm:px-6 lg:px-8">

        <h1 className="text-2xl font-bold mb-4">Transformers: gradients and graphs</h1>

              <div className="max-w-3xl mx-auto bg-white border border-gray-200 rounded-xl shadow-md p-10 space-y-6 m-16">
                <p className="text-gray-500 text-md">
                  Type a sample passage here. <br />
                  (Best performance if fewer than 500 words.)
                </p>

                <input
                  type="text"
                  placeholder="e.g. The quick brown fox jumps over the lazy dog."
                  value={sentence}
                  onChange={(e) => setSentence(e.target.value)}
                  className="w-full p-5 text-lg border border-gray-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 transition"
                />

                <p></p>
              </div>


          <div className="mt-16">



              <div className="flex gap-6 overflow-x-auto pb-2">
                {comparisons.map((comp, idx) => (
                  <div key={idx} className="w-full max-w-sm flex flex-col gap-3 border p-4 rounded shadow bg-white items-center">

                    <h2 className="font-semibold text-lg">Scenario {idx + 1}</h2>

                    <select
                      value={comp.model}
                      onChange={(e) => handleInputChange(idx, 'model', e.target.value)}
                      className="p-2 border rounded w-48"
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
                      className="p-2 border rounded w-48"
                    >
                      <option value="">Select Task</option>
                      <option>MLM</option>
                      <option>NSP</option>
                      <option>SST2</option>
                      <option>SQUAD</option>
                    </select>

                    <ComparisonButtons
                      tokens={tokensList[idx]}
                      layerCount={layerCounts[idx]}
                    />

                    <ComparisonDisplay
                      image={results[idx]}
                      selected={comp.selected}
                      onToggle={() => toggleSelection(idx)}
                      showCheckbox={comparisons.length > 1}
                      label={`Result ${idx + 1}`}
                    />
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
      </div>
  )
}
