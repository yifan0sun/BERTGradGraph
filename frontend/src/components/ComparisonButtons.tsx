import React from 'react'

export default function ComparisonButtons({
  tokens,
  layerCount,
}: {
  tokens: string[]
  layerCount: number
}) {
const words = tokens

  return (
    <div className="flex flex-col gap-4">
      {/* Horizontal row of 8 buttons */}
      <div className="flex flex-wrap justify-center gap-2 mb-4">
        <button className="px-3 py-1 bg-gray-200 rounded">Input</button>
        {[...Array(layerCount)].map((_, i) => (
          <button key={i} className="px-3 py-1 bg-gray-300 rounded">Layer {i + 1}</button>
        ))}
        <button className="px-3 py-1 bg-gray-200 rounded">Output</button>
      </div>


      {/* Row of buttons for each word in the sentence */}
      <div className="flex gap-2 flex-wrap justify-center break-words max-w-full">
        {words.map((word, i) => (
          <button key={i} className="px-2 py-1 bg-blue-200 rounded">
            {word}
          </button>
        ))}
      </div>
    </div>
  )
}