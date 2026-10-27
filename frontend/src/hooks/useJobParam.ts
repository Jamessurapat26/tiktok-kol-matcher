import { useCallback, useEffect, useState } from 'react'

function readJobParam(): string | null {
  return new URLSearchParams(window.location.search).get('job')
}

/** The selected job id, kept in ?job= so a refresh or a shared link restores it. */
export function useJobParam(): [string | null, (id: string | null) => void] {
  const [jobId, setJobIdState] = useState(readJobParam)

  useEffect(() => {
    const onPop = () => setJobIdState(readJobParam())
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  const setJobId = useCallback((id: string | null) => {
    const url = new URL(window.location.href)
    if (id) url.searchParams.set('job', id)
    else url.searchParams.delete('job')
    window.history.pushState(null, '', url)
    setJobIdState(id)
  }, [])

  return [jobId, setJobId]
}
