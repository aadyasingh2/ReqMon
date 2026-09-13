'use client'

import { useState } from 'react'
import { DriftMatrix } from '@/components/reqmon/drift-matrix'
import { RequirementParser } from '@/components/reqmon/requirement-parser'
import { SystemHeader } from '@/components/reqmon/system-header'

export default function Page() {
  const [requirementText, setRequirementText] = useState('Recall must remain above 93%')

  return (
    <main className="min-h-screen bg-canvas text-white">
      <SystemHeader />

      <div className="mx-auto flex max-w-[1400px] flex-col gap-5 px-5 py-6 sm:px-8">
        <RequirementParser
          value={requirementText}
          onChange={setRequirementText}
        />
        <DriftMatrix
          requirementText={requirementText}
        />
      </div>
    </main>
  )
}
