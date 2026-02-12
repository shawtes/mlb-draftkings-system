import React, { useState, useCallback, useEffect } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Users, Link2, BarChart3, Target, Cpu, Star, Trophy, X, Plus } from 'lucide-react';
import { Sport } from './sport-config';
import LineupsTab from './LineupsTab';
import { dfsApi } from '../services/dfs-api';
import { toast } from 'react-hot-toast';

// Extracted components
import { Player, StackType } from './optimizer/types';
import BuildControlBar from './optimizer/BuildControlBar';
import PlayerTable from './optimizer/PlayerTable';
import TeamStacksTab from './optimizer/TeamStacksTab';
import StackExposureTab from './optimizer/StackExposureTab';
import TeamCombosTab from './optimizer/TeamCombosTab';
import AdvancedQuantTab from './optimizer/AdvancedQuantTab';
import MyEntriesTab from './optimizer/MyEntriesTab';
import Sidebar from './optimizer/Sidebar';
import GameSlate from './optimizer/GameSlate';

// Extracted hooks
import { useBuildManager } from './optimizer/hooks/useBuildManager';
import { useFileUpload } from './optimizer/hooks/useFileUpload';
import { useOptimizer } from './optimizer/hooks/useOptimizer';

const DFSOptimizer = React.memo(() => {
  const buildManager = useBuildManager();
  const {
    builds, activeBuildId, currentBuild, currentSport, sportConfig, sportLocked,
    addNewBuild, removeBuild, switchBuild, updateCurrentBuild,
    handleBuildSportChange, initializeStackSettings,
  } = buildManager;

  // Destructure current build state
  const activeTab = currentBuild.activeTab;
  const playerData = currentBuild.playerData;
  const selectedPlayers = currentBuild.selectedPlayers;
  const teamSelections = currentBuild.teamSelections;
  const teamExposures = currentBuild.teamExposures;
  const stackSettings = currentBuild.stackSettings;
  const advancedQuantSettings = currentBuild.advancedQuantSettings;
  const results = currentBuild.results;

  // Build state setters
  const setActiveTab = (tab: string) => updateCurrentBuild({ activeTab: tab });
  const setPlayerData = (data: Player[]) => updateCurrentBuild({ playerData: data });
  const setSelectedPlayers = (players: string[]) => updateCurrentBuild({ selectedPlayers: players });
  const setTeamSelections = (selections: Record<number | 'all', string[]>) => updateCurrentBuild({ teamSelections: selections });
  const setTeamExposures = (exposures: Record<string, { minExp: number; maxExp: number }>) => updateCurrentBuild({ teamExposures: exposures });
  const setStackSettings = (settings: StackType[]) => updateCurrentBuild({ stackSettings: settings });
  const setAdvancedQuantSettings = (settings: any) => updateCurrentBuild({ advancedQuantSettings: settings });
  const setResults = (newResults: any[]) => updateCurrentBuild({ results: newResults });

  // Optimization settings
  const [numLineups, setNumLineups] = useState(100);
  const [minUnique, setMinUnique] = useState(3);
  const minSalary = currentBuild.minSalary ?? (sportConfig?.defaultMinSalary ?? 0);
  const setMinSalary = (value: number) => {
    if (!sportConfig) return;
    const clamped = Math.max(0, Math.min(Number.isFinite(value) ? value : sportConfig.defaultMinSalary, sportConfig.maxSalary));
    updateCurrentBuild({ minSalary: clamped });
  };
  const [disableKelly, setDisableKelly] = useState(false);
  const [sortMethod, setSortMethod] = useState('points');
  const [lineups, setLineups] = useState<any[]>([]);
  const [isLoadingLineups] = useState(false);
  const [, setDkEntriesLoaded] = useState(false);
  const [portfolioMetrics, setPortfolioMetrics] = useState<any>(null);

  // Initialize stack settings
  useEffect(() => {
    if (!currentSport) return;
    if (stackSettings.length === 0) updateCurrentBuild({ stackSettings: initializeStackSettings(currentSport) });
  }, [currentSport, stackSettings.length]);

  // Sync sport with backend
  useEffect(() => {
    if (!currentSport) return;
    let isMounted = true;
    dfsApi.setSport(currentSport).catch((error) => { if (isMounted) console.error('Failed to set sport', error); });
    return () => { isMounted = false; };
  }, [currentSport]);

  // File upload hook
  const fileUpload = useFileUpload({
    currentSport,
    onPlayerDataChange: setPlayerData,
    onSelectedPlayersChange: setSelectedPlayers,
    onTeamSelectionsChange: setTeamSelections,
    onStackSettingsChange: setStackSettings,
    onActiveTabChange: setActiveTab,
    onSportChange: (sport: Sport) => handleBuildSportChange(sport),
    initializeStackSettings,
  });

  // Selected players handler
  const syncSelectionsWithBackend = useCallback(async (selectedIds: string[]) => {
    try {
      await dfsApi.bulkUpdatePlayers({ action: 'deselect', filters: {} });
      if (selectedIds.length > 0) await Promise.all(selectedIds.map(id => dfsApi.updatePlayer(id, { selected: true })));
    } catch (error) { console.error('Failed to sync', error); }
  }, []);

  const handleSelectedPlayersChange = useCallback(async (playerIds: string[]) => {
    setSelectedPlayers(playerIds);
    const updatedPlayers = (currentBuild.playerData || []).map(player => ({ ...player, selected: playerIds.includes(player.id) }));
    setPlayerData(updatedPlayers);
    try { await syncSelectionsWithBackend(playerIds); } catch {}
  }, [currentBuild.playerData, syncSelectionsWithBackend]);

  // Optimizer hook
  const optimizer = useOptimizer({
    currentSport, playerData, selectedPlayers, stackSettings, teamSelections, teamExposures,
    advancedQuantSettings, numLineups, minUnique, minSalary, sortMethod, disableKelly,
    onResultsChange: setResults,
    onLineupsChange: setLineups,
    onActiveTabChange: setActiveTab,
    onPlayerDataChange: setPlayerData,
    onPortfolioMetricsChange: setPortfolioMetrics,
  });

  // Fetch lineups
  useEffect(() => {
    if (activeTab !== 'lineups' || !currentSport) return;
    if (results.length > 0) {
      const transformed = results.map(r => ({
        id: r.id, totalSalary: r.salary || r.totalSalary, totalProjection: r.points || r.totalPoints,
        value: (r.points || r.totalPoints) / (r.salary || r.totalSalary) * 1000,
        strategy: 'Optimized', stacks: [], timestamp: new Date().toISOString(),
        players: (r.players || []).map((p: any) => ({
          ...p, projection: p.projection || p.projectedPoints || 0,
          projectedPoints: p.projection || p.projectedPoints || 0,
          value: (p.projection || p.projectedPoints || 0) ? ((p.projection || p.projectedPoints || 0) / p.salary * 1000) : 0
        }))
      }));
      setLineups(transformed);
    }
  }, [activeTab, currentSport, results]);

  // Sport change handler that also resets local state
  const onBuildSportChange = (sport: Sport) => {
    handleBuildSportChange(sport);
    setLineups([]);
    setDkEntriesLoaded(false);
  };

  const handleLoadEntries = useCallback(() => { setDkEntriesLoaded(true); alert('DraftKings entries file loaded.'); }, []);

  const handleExportDraftKings = useCallback(async () => {
    if (results.length === 0) { alert('No lineups to export.'); return; }
    try {
      const response = await fetch(`/api/export/draftkings?sport=${currentSport}`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `${currentSport?.toLowerCase()}_lineups_${new Date().toISOString().split('T')[0]}.csv`;
      document.body.appendChild(a); a.click();
      window.URL.revokeObjectURL(url); document.body.removeChild(a);
    } catch { alert('Export failed'); }
  }, [currentSport, results]);

  // Tab definitions — shortened labels to reduce chrome width
  const tabs = [
    { id: 'players', label: 'Players', icon: Users },
    { id: 'team-stacks', label: 'Stacks', icon: Link2 },
    { id: 'stack-exposure', label: 'Stack Types', icon: BarChart3 },
    { id: 'team-combos', label: 'Combos', icon: Target },
    { id: 'advanced-quant', label: 'Quant', icon: Cpu },
    { id: 'lineups', label: 'Lineups', icon: Trophy },
    { id: 'my-entries', label: 'Entries', icon: Star },
  ];

  return (
    <div className="flex h-full w-full flex-col bg-[var(--dfs-bg-primary)] text-white overflow-hidden" style={{ fontFamily: "'Inter', sans-serif" }}>
      <div className="flex-1 min-h-0 overflow-hidden">
        <div className="flex h-full flex-col overflow-hidden border border-[var(--dfs-border)] bg-[var(--dfs-bg-primary)] w-full max-w-[1920px] mx-auto">
          {/* Merged Build + Control Bar (single ~40px row) */}
          <BuildControlBar
            builds={builds}
            activeBuildId={activeBuildId}
            currentSport={currentSport}
            sportLocked={sportLocked}
            onSwitchBuild={switchBuild}
            onAddBuild={addNewBuild}
            onRemoveBuild={removeBuild}
            onSportChange={onBuildSportChange}
            onUploadCsv={() => fileUpload.workspaceCsvInputRef.current?.click()}
            onOpenBlending={() => fileUpload.setShowBlendingDialog(true)}
            onUploadOwnership={() => fileUpload.ownershipCsvInputRef.current?.click()}
            onLoadEntries={handleLoadEntries}
            numLineups={numLineups}
            onNumLineupsChange={setNumLineups}
            minUnique={minUnique}
            onMinUniqueChange={setMinUnique}
            minSalary={minSalary}
            onMinSalaryChange={setMinSalary}
            maxSalary={sportConfig?.maxSalary ?? 50000}
            sortMethod={sortMethod}
            onSortMethodChange={setSortMethod}
            disableKelly={disableKelly}
            onDisableKellyChange={setDisableKelly}
            onOptimize={optimizer.handleRunOptimization}
            onExport={handleExportDraftKings}
            isOptimizing={optimizer.isOptimizing}
            hasPlayerData={playerData.length > 0}
            hasResults={results.length > 0}
            playerCount={playerData.length}
            selectedCount={selectedPlayers.length}
            resultsCount={results.length}
          />

          {/* Hidden file inputs */}
          <input ref={fileUpload.workspaceCsvInputRef} type="file" accept=".csv" className="hidden" onChange={fileUpload.handleFileUpload} />
          <input ref={fileUpload.ownershipCsvInputRef} type="file" accept=".csv" className="hidden" onChange={fileUpload.handleOwnershipUpload} />
          <input ref={fileUpload.blendSourceInputRef} type="file" accept=".csv" className="hidden" onChange={fileUpload.handleAddProjectionSource} />

          {/* Game Slate */}
          {currentSport && playerData.length > 0 && (
            <GameSlate playerData={playerData} sport={currentSport} />
          )}

          {currentSport && sportConfig ? (
            <div className="flex-1 flex h-full min-h-0">
              <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col h-full min-h-0">
                <TabsList className="bg-[var(--dfs-bg-secondary)] border-b border-[var(--dfs-border)] w-full rounded-none h-auto flex flex-nowrap px-1">
                  {tabs.map((tab) => {
                    const Icon = tab.icon;
                    return (
                      <TabsTrigger key={tab.id} value={tab.id} className="flex items-center gap-1 px-3 py-1.5 text-xs whitespace-nowrap flex-shrink-0">
                        <Icon className="w-3 h-3" />
                        <span>{tab.label}</span>
                        {tab.id === 'players' && playerData.length > 0 && (
                          <span className="ml-0.5 text-[10px] bg-[var(--dfs-bg-tertiary)] px-1 py-0 rounded-full text-[var(--dfs-text-muted)]">{selectedPlayers.length}</span>
                        )}
                        {tab.id === 'lineups' && results.length > 0 && (
                          <span className="ml-0.5 text-[10px] bg-cyan-500/20 px-1 py-0 rounded-full text-[var(--dfs-accent)]">{results.length}</span>
                        )}
                      </TabsTrigger>
                    );
                  })}
                </TabsList>

                <div className="flex-1 overflow-hidden p-3 flex flex-col min-h-0">
                  <TabsContent value="players" className="mt-0 h-full flex flex-col overflow-hidden">
                    <PlayerTable playerData={playerData} selectedPlayers={selectedPlayers} sport={currentSport} onPlayersChange={handleSelectedPlayersChange} onPlayerDataChange={setPlayerData} />
                  </TabsContent>
                  <TabsContent value="team-stacks" className="mt-0 h-full overflow-auto">
                    <TeamStacksTab playerData={playerData} teamSelections={teamSelections} onTeamSelectionsChange={setTeamSelections} onTeamExposuresChange={setTeamExposures} sport={currentSport ?? undefined} />
                  </TabsContent>
                  <TabsContent value="stack-exposure" className="mt-0 h-full overflow-auto">
                    <StackExposureTab stackSettings={stackSettings} sport={currentSport} onStackSettingsChange={setStackSettings} />
                  </TabsContent>
                  <TabsContent value="team-combos" className="mt-0 h-full overflow-auto">
                    <TeamCombosTab playerData={playerData} />
                  </TabsContent>
                  <TabsContent value="advanced-quant" className="mt-0 h-full overflow-auto">
                    <AdvancedQuantTab settings={advancedQuantSettings} onSettingsChange={setAdvancedQuantSettings} />
                  </TabsContent>
                  <TabsContent value="lineups" className="mt-0 h-full overflow-auto">
                    {portfolioMetrics && (
                      <div className="grid grid-cols-4 gap-2 mb-3 p-2 bg-[var(--dfs-bg-tertiary)] border border-[var(--dfs-border)] rounded-lg">
                        {[
                          { label: 'Portfolio Sharpe', value: portfolioMetrics.sharpeRatio?.toFixed(2), good: portfolioMetrics.sharpeRatio > 2, warn: portfolioMetrics.sharpeRatio > 1 },
                          { label: 'Avg Uniqueness', value: `${(portfolioMetrics.avgUniqueness * 100)?.toFixed(0)}%`, good: portfolioMetrics.avgUniqueness > 0.7, warn: portfolioMetrics.avgUniqueness > 0.5 },
                          { label: 'Max Exposure', value: `${portfolioMetrics.maxExposure?.toFixed(0)}%`, good: portfolioMetrics.maxExposure < 30, warn: portfolioMetrics.maxExposure < 60 },
                          { label: 'Concentration', value: portfolioMetrics.exposureConcentration?.toFixed(3), good: portfolioMetrics.exposureConcentration < 0.1, warn: portfolioMetrics.exposureConcentration < 0.2 },
                        ].map((metric, i) => (
                          <div key={i} className="text-center">
                            <div className="text-[10px] text-[var(--dfs-text-muted)] uppercase tracking-wider">{metric.label}</div>
                            <div className={`text-sm font-bold font-mono ${metric.good ? 'text-green-400' : metric.warn ? 'text-yellow-400' : 'text-red-400'}`}>
                              {metric.value || '\u2014'}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                    <LineupsTab
                      sport={currentSport}
                      lineups={lineups}
                      isLoading={isLoadingLineups}
                      onExportLineups={async (format) => {
                        try {
                          const response = await fetch(`/api/export/${format}?sport=${currentSport}`);
                          if (response.ok) {
                            const blob = await response.blob();
                            const url = window.URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url; a.download = `${currentSport?.toLowerCase()}_lineups.csv`;
                            document.body.appendChild(a); a.click();
                            window.URL.revokeObjectURL(url); document.body.removeChild(a);
                          }
                        } catch (error) { console.error('Export failed:', error); }
                      }}
                      onSaveFavorite={async (lineup) => {
                        try {
                          await fetch('/api/favorites', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ lineup, name: `${currentSport} Lineup ${lineup.id?.slice(0, 8) || ''}` }) });
                        } catch (error) { console.error('Save favorite failed:', error); }
                      }}
                    />
                  </TabsContent>
                  <TabsContent value="my-entries" className="mt-0 h-full overflow-auto">
                    <MyEntriesTab results={results} sport={currentSport} />
                  </TabsContent>
                </div>
              </Tabs>

              {/* Right Sidebar */}
              <Sidebar
                results={results}
                numLineups={numLineups}
                onNumLineupsChange={setNumLineups}
                onViewLineups={() => { setActiveTab('lineups'); toast.success('Switched to Lineups tab'); }}
              />
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center px-6 py-10 text-center">
              <div className="max-w-md space-y-3">
                <h3 className="text-xl font-semibold text-white">No Sport Selected</h3>
                <p className="text-sm text-[var(--dfs-text-muted)]">Choose NFL, NBA, or MLB above to activate this optimizer build.</p>
                <p className="text-sm text-[var(--dfs-text-muted)]">Each build maintains its own sport and state.</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* CSV Preview Modal */}
      {fileUpload.showCsvPreview && (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center">
          <div className="absolute inset-0 bg-black/70" onClick={() => fileUpload.setShowCsvPreview(false)} />
          <div className="relative w-full max-w-2xl mx-4 rounded-lg border shadow-2xl flex flex-col" style={{ backgroundColor: 'var(--dfs-bg-tertiary)', borderColor: 'var(--dfs-border)', maxHeight: '85vh' }}>
            <div className="p-6 pb-2 flex-shrink-0">
              <button onClick={() => fileUpload.setShowCsvPreview(false)} className="absolute top-4 right-4 text-[var(--dfs-text-muted)] hover:text-white"><X className="w-5 h-5" /></button>
              <h2 className="text-lg font-semibold text-[var(--dfs-accent)] mb-1">CSV Preview</h2>
              <p className="text-sm text-[var(--dfs-text-muted)] mb-2">Review detected columns and remap if needed.</p>
            </div>
            <div className="flex-1 overflow-auto px-6 space-y-4 min-h-0">
              <div className="grid grid-cols-2 gap-2 text-sm">
                {fileUpload.csvPreviewData.headers.map((header) => (
                  <div key={header} className="flex items-center gap-2">
                    <span className="text-[var(--dfs-text-secondary)] text-xs w-32 truncate" title={header}>{header}</span>
                    <select value={fileUpload.columnMapping[header] || 'Skip'} onChange={(e) => fileUpload.setColumnMapping(prev => ({ ...prev, [header]: e.target.value }))} className="h-7 flex-1 rounded bg-[var(--dfs-bg-secondary)] border border-[var(--dfs-border)] text-white text-xs px-2">
                      {['Name', 'Team', 'Pos', 'Salary', 'Projection', 'Ownership', 'Ceiling', 'Floor', 'StdDev', 'Opponent', 'Skip'].map(opt => <option key={opt} value={opt}>{opt}</option>)}
                    </select>
                  </div>
                ))}
              </div>
              <div className="overflow-auto max-h-48 border border-[var(--dfs-border)] rounded">
                <table className="w-full text-xs">
                  <thead className="bg-[var(--dfs-bg-tertiary)] sticky top-0"><tr>{fileUpload.csvPreviewData.headers.map(h => <th key={h} className="px-2 py-1.5 text-left text-[var(--dfs-accent)] font-medium whitespace-nowrap">{h}</th>)}</tr></thead>
                  <tbody>{fileUpload.csvPreviewData.rows.map((row, i) => <tr key={i} className="border-t border-[var(--dfs-border)]">{row.map((cell, j) => <td key={j} className="px-2 py-1 text-[var(--dfs-text-secondary)] whitespace-nowrap">{cell}</td>)}</tr>)}</tbody>
                </table>
              </div>
            </div>
            <div className="flex justify-end gap-2 p-6 pt-4 flex-shrink-0 border-t border-[var(--dfs-border)]">
              <Button variant="secondary-action" onClick={() => fileUpload.setShowCsvPreview(false)}>Cancel</Button>
              <Button variant="primary-action" onClick={fileUpload.confirmCsvUpload}>Confirm & Upload</Button>
            </div>
          </div>
        </div>
      )}

      {/* Projection Blending Modal */}
      {fileUpload.showBlendingDialog && (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center">
          <div className="absolute inset-0 bg-black/70" onClick={() => fileUpload.setShowBlendingDialog(false)} />
          <div className="relative w-full max-w-lg mx-4 rounded-lg border p-6 shadow-2xl" style={{ backgroundColor: 'var(--dfs-bg-tertiary)', borderColor: 'var(--dfs-border)' }}>
            <button onClick={() => fileUpload.setShowBlendingDialog(false)} className="absolute top-4 right-4 text-[var(--dfs-text-muted)] hover:text-white"><X className="w-5 h-5" /></button>
            <h2 className="text-lg font-semibold text-[var(--dfs-accent)] mb-1">Projection Blending</h2>
            <p className="text-sm text-[var(--dfs-text-muted)] mb-4">Upload multiple projection sources and blend them.</p>
            <div className="space-y-4">
              {fileUpload.projectionSources.length === 0 ? (
                <div className="text-center py-6 text-[var(--dfs-text-muted)] text-sm">No projection sources added yet.</div>
              ) : (
                <div className="space-y-3">
                  {fileUpload.projectionSources.map(src => (
                    <div key={src.id} className="flex items-center gap-3 bg-[var(--dfs-bg-secondary)] rounded-lg p-3">
                      <div className="flex-1"><div className="text-sm font-medium text-white">{src.name}</div><div className="text-xs text-[var(--dfs-text-muted)]">{Object.keys(src.players).length} players</div></div>
                      <div className="flex items-center gap-2">
                        <Input type="number" min={0} max={100} value={src.weight} onChange={(e) => { const w = parseInt(e.target.value) || 0; fileUpload.setProjectionSources(prev => prev.map(s => s.id === src.id ? { ...s, weight: w } : s)); }} className="w-16 h-8 bg-[var(--dfs-bg-secondary)] border-[var(--dfs-border)] text-white text-xs text-right" />
                        <span className="text-xs text-[var(--dfs-text-muted)]">%</span>
                      </div>
                      <button onClick={() => fileUpload.setProjectionSources(prev => prev.filter(s => s.id !== src.id))} className="text-red-400 hover:text-red-300 p-1"><X className="w-4 h-4" /></button>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="flex justify-end gap-2 mt-4">
              <Button variant="secondary-action" onClick={() => fileUpload.blendSourceInputRef.current?.click()} className="border-cyan-500/40 text-cyan-300"><Plus className="w-4 h-4 mr-1" /> Add Source</Button>
              <Button variant="primary-action" disabled={fileUpload.projectionSources.length === 0} onClick={() => {
                if (fileUpload.projectionSources.length === 0) return;
                const totalWeight = fileUpload.projectionSources.reduce((sum, s) => sum + s.weight, 0);
                if (totalWeight === 0) return;
                const updated = playerData.map(p => {
                  const nameLower = p.name.toLowerCase();
                  let blendedSum = 0, weightSum = 0;
                  const sources: Array<{ source: string; value: number; weight: number }> = [];
                  fileUpload.projectionSources.forEach(src => {
                    const proj = src.players[nameLower];
                    if (proj !== undefined && proj > 0) { blendedSum += proj * src.weight; weightSum += src.weight; sources.push({ source: src.name, value: proj, weight: src.weight / totalWeight }); }
                  });
                  if (weightSum > 0) {
                    return { ...p, originalProjection: p.originalProjection ?? p.projectedPoints, projectedPoints: parseFloat((blendedSum / weightSum).toFixed(2)), projectionEdited: true, projectionSources: sources };
                  }
                  return p;
                });
                setPlayerData(updated);
                fileUpload.setShowBlendingDialog(false);
                toast.success('Projections blended and applied');
              }}>Blend & Apply</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

DFSOptimizer.displayName = 'DFSOptimizer';

export default DFSOptimizer;
