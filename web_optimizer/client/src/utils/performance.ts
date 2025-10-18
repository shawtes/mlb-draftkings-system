/**
 * Simple Performance Monitoring Utility
 * 
 * Use this to measure component render times and navigation speed
 */

export const measurePerformance = (componentName: string, operation: string) => {
  const startTime = performance.now();
  
  return {
    end: () => {
      const endTime = performance.now();
      const duration = endTime - startTime;
      
      if (process.env.NODE_ENV === 'development') {
        console.log(`⚡ ${componentName} - ${operation}: ${duration.toFixed(2)}ms`);
      }
      
      return duration;
    }
  };
};

export const logNavigationMetrics = () => {
  if (typeof window !== 'undefined' && window.performance) {
    const perfData = window.performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    
    if (perfData) {
      console.log('📊 Performance Metrics:');
      console.log(`  DNS Lookup: ${(perfData.domainLookupEnd - perfData.domainLookupStart).toFixed(2)}ms`);
      console.log(`  TCP Connection: ${(perfData.connectEnd - perfData.connectStart).toFixed(2)}ms`);
      console.log(`  Request Time: ${(perfData.responseStart - perfData.requestStart).toFixed(2)}ms`);
      console.log(`  Response Time: ${(perfData.responseEnd - perfData.responseStart).toFixed(2)}ms`);
      console.log(`  DOM Processing: ${(perfData.domComplete - perfData.domLoading).toFixed(2)}ms`);
      console.log(`  Total Load Time: ${(perfData.loadEventEnd - perfData.loadEventStart).toFixed(2)}ms`);
    }
  }
};

