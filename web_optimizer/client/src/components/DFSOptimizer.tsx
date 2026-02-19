import React, { useState, useCallback, useEffect, useRef } from 'react';
import { PanelGroup, Panel, PanelResizeHandle } from 'react-resizable-panels';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { X, Plus, Search, CheckSquare, XSquare, Eye, EyeOff, Target } from 'lucide-react';
import { Sport, getPositionFilters, getPositionCount, filterPlayersByPosition } from './sport-config';
import LineupsTab from './LineupsTab';
import { dfsApi } from '../services/dfs-api';
import { toast } from 'react-hot-toast';
import WebSocketConnection, { WsStatus } from '../services/WebSocketConnection';

// Extracted components
import { Player, StackType, ContestPreset, CASH_FULL_PRESET, GPP_FULL_PRESET } from './optimizer/types';
import { BuildNavBar, FilterToolbar } from './optimizer/BuildControlBar';
import PlayerTable from './optimizer/PlayerTable';
import TeamStacksTab from './optimizer/TeamStacksTab';
import StackExposureTab from './optimizer/StackExposureTab';
import TeamCombosTab from './optimizer/TeamCombosTab';
import ExposureAnalysisTab from './optimizer/ExposureAnalysisTab';
import AdvancedQuantTab from './optimizer/AdvancedQuantTab';
import MyEntriesTab from './optimizer/MyEntriesTab';
import Sidebar from './optimizer/Sidebar';
import GameSlate from './optimizer/GameSlate';

// Extracted hooks
import { useBuildManager } from './optimizer/hooks/useBuildManager';
import { useFileUpload } from './optimizer/hooks/useFileUpload';
import { useOptimizer } from './optimizer/hooks/useOptimizer';

// ─── Inline header sub-components ───

const L1_STYLE: React.CSSProperties = {
  height: 32, display: 'flex', alignItems: 'center', padding: '0 12px', gap: 6,
  background: '#0a0e14', borderBottom: '1px solid var(--dfs-border)', fontSize: 10, flexShrink: 0,
};

const L5_STYLE: React.CSSProperties = {
  height: 28, display: 'flex', alignItems: 'center', padding: '0 12px',
  background: 'var(--dfs-bg-secondary)', borderBottom: '1px solid var(--dfs-border)', flexShrink: 0,
};

const L6_STYLE: React.CSSProperties = {
  height: 28, display: 'flex', alignItems: 'center', padding: '0 12px', gap: 3,
  borderBottom: '1px solid var(--dfs-border)', flexShrink: 0,
};

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

  // WebSocket connection
  const wsRef = useRef<WebSocketConnection | null>(null);
  const [wsStatus, setWsStatus] = useState<WsStatus>('disconnected');
  const [optimizationProgress, setOptimizationProgress] = useState<string | null>(null);

  useEffect(() => {
    const wsUrl = `ws://${window.location.hostname}:${window.location.port === '3000' ? '5001' : window.location.port}`;
    const ws = new WebSocketConnection(wsUrl, setWsStatus);
    wsRef.current = ws;

    ws.on('OPTIMIZATION_PROGRESS', (data: any) => {
      setOptimizationProgress(data.message || `${data.progress || 0}%`);
    });

    ws.on('OPTIMIZATION_COMPLETED', (data: any) => {
      setOptimizationProgress(null);
      if (data.lineups) {
        toast.success(`Optimization complete: ${data.lineups.length} lineups`);
      }
    });

    return () => ws.disconnect();
  }, []);

  // Optimization settings
  const [numLineups, setNumLineups] = useState(100);
  const [minUnique, setMinUnique] = useState(3);
  const minSalary = currentBuild.minSalary ?? (sportConfig?.defaultMinSalary ?? 0);
  const setMinSalary = (value: number) => {
    if (!sportConfig) return;
    const clamped = Math.max(0, Math.min(Number.isFinite(value) ? value : sportConfig.defaultMinSalary, sportConfig.maxSalary));
    updateCurrentBuild({ minSalary: clamped });
  };
  const [contestMode, setContestMode] = useState<'gpp' | 'cash'>('gpp');
  const [disableKelly, setDisableKelly] = useState(false);
  const [sortMethod, setSortMethod] = useState('points');
  const [lineups, setLineups] = useState<any[]>([]);
  const [isLoadingLineups] = useState(false);
  const [, setDkEntriesLoaded] = useState(false);
  const [portfolioMetrics, setPortfolioMetrics] = useState<any>(null);
  const [activePreset, setActivePreset] = useState<'cash' | 'gpp' | 'custom'>('gpp');

  const handleApplyPreset = useCallback((preset: ContestPreset) => {
    setContestMode(preset.contestMode);
    setMinUnique(preset.minUnique);
    setAdvancedQuantSettings({ ...advancedQuantSettings, ...preset.quant });
    // Update player max exposures — only override players still at the old default (100),
    // preserve any per-player customizations the user has already made
    const updated = playerData.map(p => {
      // If user hasn't customized this player's exposure, apply preset default
      if (p.maxExp === 100 || p.maxExp === 50 || p.maxExp === 80) {
        return { ...p, maxExp: preset.maxExposure };
      }
      return p;
    });
    setPlayerData(updated);
    setActivePreset(preset.contestMode);
  }, [advancedQuantSettings, playerData]);

  // L6 filter state (lifted from PlayerTable)
  const [positionFilter, setPositionFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [showAdvancedCols, setShowAdvancedCols] = useState(false);
  const [leveragePlaysFilter, setLeveragePlaysFilter] = useState(false);

  // Reset position filter when sport changes
  useEffect(() => {
    if (currentSport === 'MLB') setPositionFilter('all-batters');
    else if (currentSport === 'NFL') setPositionFilter('all-offense');
    else setPositionFilter('all');
  }, [currentSport]);

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

  // Auto-fetch players on mount if server has pre-loaded data
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const response = await dfsApi.getPlayers();
        if (cancelled || !response?.players?.length) return;
        const backendPlayers = response.players as any[];
        const detectedSport = (response as any).sport as Sport | undefined;
        if (detectedSport && !currentSport) {
          handleBuildSportChange(detectedSport);
        }
        const transformedPlayers: Player[] = backendPlayers.map((p: any) => ({
          id: p.id, name: p.name, team: p.team, position: p.position, salary: p.salary,
          projectedPoints: p.projection || p.projectedPoints || 0, minExp: p.minExposure ?? 0, maxExp: p.maxExposure ?? 100,
          selected: Boolean(p.selected), ownership: p.ownership || 0, locked: Boolean(p.locked), excluded: Boolean(p.excluded),
          ceiling: p.ceiling || undefined, floor: p.floor || undefined, stdDev: p.stdDev || undefined, opponent: p.opponent || undefined,
          probOver5: p.probOver5 || undefined, probOver10: p.probOver10 || undefined, probOver15: p.probOver15 || undefined,
          probOver20: p.probOver20 || undefined, probOver25: p.probOver25 || undefined, probOver30: p.probOver30 || undefined,
          garchVolatility: p.garchVolatility || undefined,
          garchConditionalVolatility: p.garchConditionalVolatility || undefined,
          volatilityRegime: p.volatilityRegime ?? undefined,
          bullRegime: p.bullRegime ?? undefined,
          regimeStrength: p.regimeStrength || undefined,
          momentumRegime: p.momentumRegime ?? undefined,
          consistencyRegime: p.consistencyRegime ?? undefined,
          entropy: p.entropy || undefined,
          hurstExponent: p.hurstExponent || undefined,
          rollingSharpe: p.rollingSharpe || undefined,
          avgPlayerCorrelation: p.avgPlayerCorrelation || undefined,
          correlationVolatility: p.correlationVolatility || undefined,
          evtReturnLevel: p.evtReturnLevel || undefined,
          exceedanceProb: p.exceedanceProb || undefined,
        }));
        if (!cancelled && transformedPlayers.length > 0) {
          setPlayerData(transformedPlayers);
          setSelectedPlayers(transformedPlayers.filter(p => p.selected).map(p => p.id));
          const uniqueTeams = [...new Set(transformedPlayers.map(p => p.team))].filter(Boolean);
          setTeamSelections({ all: uniqueTeams, 2: uniqueTeams, 3: uniqueTeams, 4: uniqueTeams, 5: uniqueTeams });
          setActiveTab('players');
          toast.success(`Auto-loaded ${transformedPlayers.length} players`);
        }
      } catch {
        // Server may not be ready yet, ignore
      }
    })();
    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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
    advancedQuantSettings, contestMode, numLineups, minUnique, minSalary, sortMethod, disableKelly,
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

  // Sport change handler
  const onBuildSportChange = (sport: Sport) => {
    handleBuildSportChange(sport);
    setLineups([]);
    setDkEntriesLoaded(false);
  };

  const handleLoadEntries = useCallback(() => { setDkEntriesLoaded(true); toast.success('DraftKings entries file loaded.'); }, []);

  const handleExportDraftKings = useCallback(async () => {
    if (results.length === 0) { toast.error('No lineups to export.'); return; }
    try {
      const response = await fetch(`/api/export/draftkings?sport=${currentSport}`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `${currentSport?.toLowerCase()}_lineups_${new Date().toISOString().split('T')[0]}.csv`;
      document.body.appendChild(a); a.click();
      window.URL.revokeObjectURL(url); document.body.removeChild(a);
    } catch { toast.error('Export failed'); }
  }, [currentSport, results]);

  // Sub-tab definitions for L5
  const subTabs = [
    { id: 'players', label: 'Players' },
    { id: 'team-stacks', label: 'Team Stacks' },
    { id: 'stack-exposure', label: 'Stack Types' },
    { id: 'team-combos', label: 'Stack Exposure' },
    { id: 'exposure-analysis', label: 'Exposure' },
    { id: 'advanced-quant', label: 'Quant' },
    { id: 'lineups', label: 'Lineups' },
    { id: 'my-entries', label: 'Entries' },
  ];

  // ─── Date string for L1 ───
  const dateStr = new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });

  return (
    <div className="app-shell" style={{ fontFamily: "'Inter', system-ui, sans-serif", background: 'var(--dfs-bg-primary)', color: 'var(--dfs-text-primary)', height: '100vh', display: 'flex', flexDirection: 'column', overflow: 'hidden', WebkitFontSmoothing: 'antialiased', fontSize: 11 }}>

      {/* ═══ LEVEL 1: Top Header — logo, date, slate ═══ */}
      <div style={L1_STYLE}>
        <span style={{ fontWeight: 700, color: 'var(--dfs-accent)', fontSize: 11, letterSpacing: -0.3 }}>UrSim</span>
        <span style={{ padding: '2px 8px', background: 'var(--dfs-bg-tertiary)', borderRadius: 3, color: 'var(--dfs-text-secondary)', fontSize: 9 }}>
          {dateStr}
        </span>
        {currentSport && (
          <span style={{ padding: '2px 8px', background: 'var(--dfs-bg-tertiary)', borderRadius: 3, color: 'var(--dfs-text-secondary)', fontSize: 9 }}>
            {currentSport} — Main
          </span>
        )}
        <span style={{ padding: '2px 8px', background: 'var(--dfs-bg-tertiary)', borderRadius: 3, color: 'var(--dfs-text-muted)', fontSize: 9, cursor: 'pointer' }}>
          + Add Slate
        </span>
        <span style={{ flex: 1 }} />
        {optimizationProgress && (
          <span style={{ fontSize: 8, color: 'var(--dfs-accent)', fontFamily: "'JetBrains Mono', monospace" }}>
            {optimizationProgress}
          </span>
        )}
        <span
          title={`Server: ${wsStatus}`}
          style={{
            width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
            background: wsStatus === 'connected' ? '#22c55e' : wsStatus === 'reconnecting' ? '#eab308' : '#ef4444',
            boxShadow: wsStatus === 'connected' ? '0 0 4px #22c55e' : undefined,
          }}
        />
        <span style={{ color: 'var(--dfs-text-muted)', fontSize: 9, cursor: 'pointer' }}>Switch to Classic</span>
      </div>

      {/* ═══ LEVEL 2: Build Nav — tabs, sport pills ═══ */}
      <BuildNavBar
        builds={builds}
        activeBuildId={activeBuildId}
        currentSport={currentSport}
        sportLocked={sportLocked}
        onSwitchBuild={switchBuild}
        onAddBuild={addNewBuild}
        onRemoveBuild={removeBuild}
        onSportChange={onBuildSportChange}
      />

      {/* ═══ LEVEL 3: Games Bar ═══ */}
      {currentSport && playerData.length > 0 && (
        <GameSlate playerData={playerData} sport={currentSport} />
      )}

      {/* ═══ LEVEL 4: Filter Toolbar ═══ */}
      {currentSport && (
        <FilterToolbar
          currentSport={currentSport}
          numLineups={numLineups}
          onNumLineupsChange={setNumLineups}
          minSalary={minSalary}
          onMinSalaryChange={setMinSalary}
          maxSalary={sportConfig?.maxSalary ?? 50000}
          minUnique={minUnique}
          onMinUniqueChange={setMinUnique}
          onUploadCsv={() => fileUpload.workspaceCsvInputRef.current?.click()}
          onOpenBlending={() => fileUpload.setShowBlendingDialog(true)}
          onUploadOwnership={() => fileUpload.ownershipCsvInputRef.current?.click()}
          onLoadEntries={handleLoadEntries}
          onOptimize={optimizer.handleRunOptimization}
          onExport={handleExportDraftKings}
          isOptimizing={optimizer.isOptimizing}
          hasPlayerData={playerData.length > 0}
          hasResults={results.length > 0}
          sortMethod={sortMethod}
          onSortMethodChange={setSortMethod}
          disableKelly={disableKelly}
          onDisableKellyChange={setDisableKelly}
          playerCount={playerData.length}
          selectedCount={selectedPlayers.length}
          resultsCount={results.length}
          activePreset={activePreset}
          onApplyPreset={handleApplyPreset}
        />
      )}

      {/* Hidden file inputs */}
      <input ref={fileUpload.workspaceCsvInputRef} type="file" accept=".csv" className="hidden" onChange={fileUpload.handleFileUpload} />
      <input ref={fileUpload.ownershipCsvInputRef} type="file" accept=".csv" className="hidden" onChange={fileUpload.handleOwnershipUpload} />
      <input ref={fileUpload.blendSourceInputRef} type="file" accept=".csv" className="hidden" onChange={fileUpload.handleAddProjectionSource} />

      {currentSport && sportConfig ? (
        <>
          {/* ═══ LEVEL 5: Sub-tabs ═══ */}
          <div style={L5_STYLE}>
            {subTabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  fontSize: 9, fontWeight: activeTab === tab.id ? 600 : 500,
                  color: activeTab === tab.id ? 'var(--dfs-text-primary)' : 'var(--dfs-text-muted)',
                  cursor: 'pointer', padding: '4px 8px', background: 'transparent', border: 'none',
                  fontFamily: 'inherit',
                }}
              >
                {tab.label}
                {tab.id === 'players' && playerData.length > 0 && (
                  <span style={{ marginLeft: 3, fontSize: 8, color: 'var(--dfs-text-muted)' }}>({selectedPlayers.length})</span>
                )}
                {tab.id === 'lineups' && results.length > 0 && (
                  <span style={{ marginLeft: 3, fontSize: 8, color: 'var(--dfs-accent)' }}>({results.length})</span>
                )}
              </button>
            ))}
            <span style={{ flex: 1 }} />
            <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
              <button style={{ padding: '2px 8px', fontSize: 8, fontWeight: 600, borderRadius: 3, border: '1px solid var(--dfs-border)', background: 'transparent', color: 'var(--dfs-text-secondary)', cursor: 'pointer', fontFamily: 'inherit' }}>Templates</button>
              <button onClick={() => setActiveTab('advanced-quant')} style={{ padding: '2px 8px', fontSize: 8, fontWeight: 600, borderRadius: 3, border: '1px solid var(--dfs-border)', background: 'transparent', color: 'var(--dfs-text-secondary)', cursor: 'pointer', fontFamily: 'inherit' }}>Settings</button>
              <button style={{ padding: '2px 8px', fontSize: 8, fontWeight: 700, borderRadius: 3, border: 'none', background: 'var(--dfs-accent-secondary, #22c55e)', color: '#000', cursor: 'pointer', fontFamily: 'inherit' }}>Apply</button>
              <button style={{ padding: '2px 8px', fontSize: 8, fontWeight: 600, borderRadius: 3, border: '1px solid var(--dfs-border)', background: 'transparent', color: 'var(--dfs-text-secondary)', cursor: 'pointer', fontFamily: 'inherit' }}>Save All</button>
            </div>
          </div>

          {/* ═══ LEVEL 6: Position Chips + Search (only when Players tab active) ═══ */}
          {activeTab === 'players' && currentSport && (
            <div style={L6_STYLE}>
              {getPositionFilters(currentSport).map((pos) => {
                const count = getPositionCount(playerData, pos.id, currentSport);
                const isActive = positionFilter === pos.id;
                // Compute checkbox state for this position group
                const posPlayers = filterPlayersByPosition(playerData, pos.id, currentSport);
                const posPlayerIds = posPlayers.map(p => p.id);
                const selectedInPos = posPlayerIds.filter(id => selectedPlayers.includes(id)).length;
                const allSelected = posPlayerIds.length > 0 && selectedInPos === posPlayerIds.length;
                const someSelected = selectedInPos > 0 && selectedInPos < posPlayerIds.length;
                return (
                  <div key={pos.id} style={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <span
                      onClick={(e) => {
                        e.stopPropagation();
                        if (allSelected) {
                          // Deselect all players in this position
                          const newSelected = selectedPlayers.filter(id => !posPlayerIds.includes(id));
                          handleSelectedPlayersChange(newSelected);
                        } else {
                          // Select all players in this position (add missing ones)
                          const newSelected = [...new Set([...selectedPlayers, ...posPlayerIds])];
                          handleSelectedPlayersChange(newSelected);
                        }
                      }}
                      style={{
                        width: 12, height: 12, borderRadius: 2, cursor: 'pointer', flexShrink: 0,
                        border: `1px solid ${allSelected || someSelected ? 'var(--dfs-accent)' : 'var(--dfs-border)'}`,
                        background: allSelected ? 'var(--dfs-accent)' : 'transparent',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: 8, color: '#fff', lineHeight: 1,
                      }}
                      title={allSelected ? `Deselect all ${pos.label}` : `Select all ${pos.label}`}
                    >
                      {allSelected ? '\u2713' : someSelected ? '\u2014' : ''}
                    </span>
                    <button
                      onClick={() => setPositionFilter(pos.id)}
                      style={{
                        padding: '2px 5px 2px 3px', fontSize: 8.5, fontWeight: 600, borderRadius: 3,
                        border: isActive ? '1px solid rgba(240,246,252,0.15)' : '1px solid transparent',
                        background: isActive ? 'var(--dfs-bg-surface3, #21262d)' : 'transparent',
                        color: isActive ? 'var(--dfs-text-primary)' : 'var(--dfs-text-muted)',
                        cursor: 'pointer', fontFamily: 'inherit',
                      }}
                    >
                      {pos.label}
                      <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 7.5, color: 'var(--dfs-text-muted)', marginLeft: 2 }}>({count})</span>
                    </button>
                  </div>
                );
              })}

              {/* GPP / Cash toggle */}
              <div style={{ width: 1, height: 16, background: 'var(--dfs-border)', margin: '0 4px' }} />
              <div style={{ display: 'flex', overflow: 'hidden', borderRadius: 3, border: '1px solid var(--dfs-border)' }}>
                <button
                  onClick={() => handleApplyPreset(GPP_FULL_PRESET)}
                  style={{ padding: '2px 8px', fontSize: 8, fontWeight: 700, border: 'none', cursor: 'pointer', fontFamily: 'inherit',
                    background: activePreset === 'gpp' ? '#7c3aed' : 'transparent',
                    color: activePreset === 'gpp' ? '#fff' : 'var(--dfs-text-muted)' }}
                >GPP</button>
                <button
                  onClick={() => handleApplyPreset(CASH_FULL_PRESET)}
                  style={{ padding: '2px 8px', fontSize: 8, fontWeight: 700, border: 'none', cursor: 'pointer', fontFamily: 'inherit',
                    background: activePreset === 'cash' ? '#059669' : 'transparent',
                    color: activePreset === 'cash' ? '#fff' : 'var(--dfs-text-muted)' }}
                >Cash</button>
              </div>

              {/* Quick filter buttons */}
              <div style={{ width: 1, height: 16, background: 'var(--dfs-border)', margin: '0 4px' }} />
              <button
                onClick={() => setShowAdvancedCols(prev => !prev)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 2, padding: '2px 6px', fontSize: 8, fontWeight: 500,
                  borderRadius: 3, border: '1px solid var(--dfs-border)', cursor: 'pointer', fontFamily: 'inherit',
                  background: showAdvancedCols ? 'rgba(167,139,250,0.1)' : 'transparent',
                  color: showAdvancedCols ? '#a78bfa' : 'var(--dfs-text-muted)',
                }}
              >
                {showAdvancedCols ? <EyeOff style={{ width: 10, height: 10 }} /> : <Eye style={{ width: 10, height: 10 }} />}
                Boom/Bust
              </button>
              <button
                onClick={() => setLeveragePlaysFilter(prev => !prev)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 2, padding: '2px 6px', fontSize: 8, fontWeight: 500,
                  borderRadius: 3, border: '1px solid var(--dfs-border)', cursor: 'pointer', fontFamily: 'inherit',
                  background: leveragePlaysFilter ? 'rgba(167,139,250,0.2)' : 'transparent',
                  color: leveragePlaysFilter ? '#a78bfa' : 'var(--dfs-text-muted)',
                }}
              >
                <Target style={{ width: 10, height: 10 }} />
                Leverage
              </button>

              <span style={{ flex: 1 }} />

              {/* Search */}
              <div style={{ position: 'relative' }}>
                <Search style={{ width: 10, height: 10, position: 'absolute', left: 6, top: '50%', transform: 'translateY(-50%)', color: 'var(--dfs-text-muted)' }} />
                <input
                  placeholder="Search player or team..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  style={{
                    padding: '3px 8px 3px 22px', fontSize: 9, background: 'var(--dfs-bg-primary)',
                    border: '1px solid var(--dfs-border)', borderRadius: 3, color: 'var(--dfs-text-primary)',
                    width: 160, fontFamily: 'inherit', outline: 'none',
                  }}
                  onFocus={(e) => { e.target.style.borderColor = 'var(--dfs-accent)'; }}
                  onBlur={(e) => { e.target.style.borderColor = 'var(--dfs-border)'; }}
                />
              </div>
            </div>
          )}

          {/* ═══ MAIN AREA: PanelGroup with Content + Right Panel ═══ */}
          <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
            <PanelGroup direction="horizontal">
              <Panel defaultSize={75} minSize={50}>
                <div style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                  {/* Active tab content — conditional render */}
                  {activeTab === 'players' && (
                    <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                      <PlayerTable
                        playerData={playerData}
                        selectedPlayers={selectedPlayers}
                        sport={currentSport}
                        onPlayersChange={handleSelectedPlayersChange}
                        onPlayerDataChange={setPlayerData}
                        contestMode={contestMode}
                        onContestModeChange={setContestMode}
                        positionFilter={positionFilter}
                        searchQuery={searchQuery}
                        leveragePlaysFilter={leveragePlaysFilter}
                        showAdvancedCols={showAdvancedCols}
                      />
                    </div>
                  )}

                  {activeTab === 'team-stacks' && (
                    <div style={{ flex: 1, overflow: 'auto', padding: 8 }}>
                      <TeamStacksTab playerData={playerData} teamSelections={teamSelections} onTeamSelectionsChange={setTeamSelections} onTeamExposuresChange={setTeamExposures} sport={currentSport ?? undefined} results={results} />
                    </div>
                  )}

                  {activeTab === 'stack-exposure' && (
                    <div style={{ flex: 1, overflow: 'auto', padding: 8 }}>
                      <StackExposureTab stackSettings={stackSettings} sport={currentSport} onStackSettingsChange={setStackSettings} />
                    </div>
                  )}

                  {activeTab === 'team-combos' && (
                    <div style={{ flex: 1, overflow: 'auto', padding: 8 }}>
                      <TeamCombosTab playerData={playerData} />
                    </div>
                  )}

                  {activeTab === 'exposure-analysis' && (
                    <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', padding: 8 }}>
                      <ExposureAnalysisTab playerData={playerData} results={results} />
                    </div>
                  )}

                  {activeTab === 'advanced-quant' && (
                    <div style={{ flex: 1, overflow: 'auto', padding: 8 }}>
                      <AdvancedQuantTab settings={advancedQuantSettings} onSettingsChange={setAdvancedQuantSettings} />
                    </div>
                  )}

                  {activeTab === 'lineups' && (
                    <div style={{ flex: 1, overflow: 'auto', padding: 8 }}>
                      {portfolioMetrics && (
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, marginBottom: 12, padding: 8, background: 'var(--dfs-bg-tertiary)', border: '1px solid var(--dfs-border)', borderRadius: 6 }}>
                          {[
                            { label: 'Portfolio Sharpe', value: portfolioMetrics.sharpeRatio?.toFixed(2), good: portfolioMetrics.sharpeRatio > 2, warn: portfolioMetrics.sharpeRatio > 1 },
                            { label: 'Avg Uniqueness', value: `${(portfolioMetrics.avgUniqueness * 100)?.toFixed(0)}%`, good: portfolioMetrics.avgUniqueness > 0.7, warn: portfolioMetrics.avgUniqueness > 0.5 },
                            { label: 'Max Exposure', value: `${portfolioMetrics.maxExposure?.toFixed(0)}%`, good: portfolioMetrics.maxExposure < 30, warn: portfolioMetrics.maxExposure < 60 },
                            { label: 'Concentration', value: portfolioMetrics.exposureConcentration?.toFixed(3), good: portfolioMetrics.exposureConcentration < 0.1, warn: portfolioMetrics.exposureConcentration < 0.2 },
                          ].map((metric, i) => (
                            <div key={i} style={{ textAlign: 'center' }}>
                              <div style={{ fontSize: 8, color: 'var(--dfs-text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{metric.label}</div>
                              <div style={{ fontSize: 13, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: metric.good ? '#22c55e' : metric.warn ? '#eab308' : '#ef4444' }}>
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
                    </div>
                  )}

                  {activeTab === 'my-entries' && (
                    <div style={{ flex: 1, overflow: 'auto', padding: 8 }}>
                      <MyEntriesTab results={results} sport={currentSport} />
                    </div>
                  )}
                </div>
              </Panel>

              <PanelResizeHandle style={{ width: 6, background: 'transparent', cursor: 'col-resize', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ width: 2, height: 40, borderRadius: 1, background: 'var(--dfs-ghost, #30363d)', transition: 'background 0.15s' }} />
              </PanelResizeHandle>

              <Panel defaultSize={25} minSize={15} maxSize={40}>
                <Sidebar
                  results={results}
                  numLineups={numLineups}
                  onNumLineupsChange={setNumLineups}
                  onViewLineups={() => { setActiveTab('lineups'); toast.success('Switched to Lineups tab'); }}
                  portfolioMetrics={portfolioMetrics}
                  onOptimize={optimizer.handleRunOptimization}
                  isOptimizing={optimizer.isOptimizing}
                  sport={currentSport}
                />
              </Panel>
            </PanelGroup>
          </div>
        </>
      ) : (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '24px 48px', textAlign: 'center' }}>
          <div style={{ maxWidth: 400 }}>
            <h3 style={{ fontSize: 18, fontWeight: 600, color: 'var(--dfs-text-primary)', marginBottom: 8 }}>No Sport Selected</h3>
            <p style={{ fontSize: 12, color: 'var(--dfs-text-muted)' }}>Choose NFL, NBA, or MLB above to activate this optimizer build.</p>
            <p style={{ fontSize: 12, color: 'var(--dfs-text-muted)', marginTop: 4 }}>Each build maintains its own sport and state.</p>
          </div>
        </div>
      )}

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
