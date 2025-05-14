import React from 'react'

export default function ComparisonDisplay({
  image,
  label,
  selected,
  onToggle,
  showCheckbox,
}: {
  image: string
  label: string
  selected: boolean
  onToggle: () => void
  showCheckbox: boolean
}) {
  return (
    <div className="flex flex-col items-center">
      <img src={image} alt={label} className="w-64 h-64 object-contain" />
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
    </div>
  )
}
