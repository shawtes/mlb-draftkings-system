export function SkeletonLoader() {
  return (
    <div className="animate-pulse space-y-6 p-6">
      {/* Header skeleton */}
      <div className="space-y-3">
        <div className="h-8 bg-slate-700 rounded w-1/3"></div>
        <div className="h-4 bg-slate-700 rounded w-1/2"></div>
      </div>
      
      {/* Content skeleton */}
      <div className="space-y-4">
        <div className="grid grid-cols-3 gap-4">
          <div className="h-32 bg-slate-700 rounded"></div>
          <div className="h-32 bg-slate-700 rounded"></div>
          <div className="h-32 bg-slate-700 rounded"></div>
        </div>
        
        <div className="space-y-2">
          <div className="h-4 bg-slate-700 rounded w-full"></div>
          <div className="h-4 bg-slate-700 rounded w-5/6"></div>
          <div className="h-4 bg-slate-700 rounded w-4/6"></div>
        </div>
        
        <div className="h-64 bg-slate-700 rounded"></div>
      </div>
    </div>
  );
}

export function SimpleLoader() {
  return (
    <div className="flex items-center justify-center h-full min-h-[400px]">
      <div className="text-center">
        <div className="relative w-16 h-16 mx-auto mb-4">
          <div className="absolute top-0 left-0 w-full h-full">
            <div className="w-16 h-16 border-4 border-slate-600 border-t-blue-500 rounded-full animate-spin"></div>
          </div>
        </div>
        <p className="text-slate-400 text-sm">Loading...</p>
      </div>
    </div>
  );
}

export function DashboardLoader() {
  return (
    <div className="animate-pulse space-y-6 p-6">
      {/* Stats cards skeleton */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="bg-slate-800 rounded-lg p-4 space-y-3">
            <div className="h-4 bg-slate-700 rounded w-2/3"></div>
            <div className="h-8 bg-slate-700 rounded w-1/2"></div>
            <div className="h-3 bg-slate-700 rounded w-1/3"></div>
          </div>
        ))}
      </div>

      {/* Table skeleton */}
      <div className="bg-slate-800 rounded-lg p-4 space-y-3">
        <div className="h-6 bg-slate-700 rounded w-1/4 mb-4"></div>
        {[1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="flex space-x-4">
            <div className="h-4 bg-slate-700 rounded flex-1"></div>
            <div className="h-4 bg-slate-700 rounded flex-1"></div>
            <div className="h-4 bg-slate-700 rounded flex-1"></div>
            <div className="h-4 bg-slate-700 rounded w-20"></div>
          </div>
        ))}
      </div>
    </div>
  );
}

