import { useState, useRef, useCallback } from 'react';
import { Sport, SPORT_CONFIGS } from '../../sport-config';
import { Player, StackType } from '../types';
import { dfsApi } from '../../../services/dfs-api';
import { toast } from 'react-hot-toast';

interface UseFileUploadProps {
  currentSport: Sport | null;
  onPlayerDataChange: (data: Player[]) => void;
  onSelectedPlayersChange: (ids: string[]) => void;
  onTeamSelectionsChange: (selections: Record<number | 'all', string[]>) => void;
  onStackSettingsChange: (settings: StackType[]) => void;
  onActiveTabChange: (tab: string) => void;
  onSportChange: (sport: Sport) => void;
  initializeStackSettings: (sport: Sport) => StackType[];
}

export function useFileUpload(props: UseFileUploadProps) {
  const {
    currentSport,
    onPlayerDataChange,
    onSelectedPlayersChange,
    onTeamSelectionsChange,
    onStackSettingsChange,
    onActiveTabChange,
    onSportChange,
    initializeStackSettings,
  } = props;

  const [showCsvPreview, setShowCsvPreview] = useState(false);
  const [csvPreviewData, setCsvPreviewData] = useState<{ headers: string[]; rows: string[][] }>({ headers: [], rows: [] });
  const [csvPendingFile, setCsvPendingFile] = useState<File | null>(null);
  const [columnMapping, setColumnMapping] = useState<Record<string, string>>({});
  const [showBlendingDialog, setShowBlendingDialog] = useState(false);
  const [projectionSources, setProjectionSources] = useState<Array<{ id: string; name: string; weight: number; players: Record<string, number> }>>([]);

  const workspaceCsvInputRef = useRef<HTMLInputElement | null>(null);
  const ownershipCsvInputRef = useRef<HTMLInputElement | null>(null);
  const blendSourceInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    let effectiveSport = currentSport;
    if (!effectiveSport) {
      const filenameLower = file.name.toLowerCase();
      if (filenameLower.includes('nba') || filenameLower.includes('basketball')) effectiveSport = 'NBA';
      else if (filenameLower.includes('nfl') || filenameLower.includes('football')) effectiveSport = 'NFL';
      else if (filenameLower.includes('mlb') || filenameLower.includes('baseball')) effectiveSport = 'MLB';
      if (effectiveSport) onSportChange(effectiveSport);
    }

    if (!effectiveSport) { toast.error('Please select a sport before uploading.'); return; }

    const capturedFile = new File([file], file.name, { type: file.type });
    event.target.value = '';

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      if (!text) return;
      const lines = text.split('\n').filter(l => l.trim());
      if (lines.length === 0) return;
      const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
      const rows = lines.slice(1, 6).map(line => line.split(',').map(cell => cell.trim().replace(/^"|"$/g, '')));

      const mapping: Record<string, string> = {};
      headers.forEach(h => {
        const lower = h.toLowerCase();
        if (['name', 'player', 'name_dk', 'player_name', 'player_name_proj', 'playername'].includes(lower)) mapping[h] = 'Name';
        else if (['team', 'teamabbrev', 'tm', 'team_proj', 'teamid'].includes(lower)) mapping[h] = 'Team';
        else if (['pos', 'position', 'roster_position', 'position_proj'].includes(lower)) mapping[h] = 'Pos';
        else if (['salary', 'cost', 'dk_salary'].includes(lower)) mapping[h] = 'Salary';
        else if (['projection', 'predicted_dk_points', 'my_proj', 'avgpointspergame', 'ppg_projection', 'adjusted_projection', 'projected_points', 'projected_dk_points', 'fantasypointsdraftkings', 'fantasypoints', 'fpts'].includes(lower)) mapping[h] = 'Projection';
        else if (['ownership', 'own', 'own%'].includes(lower)) mapping[h] = 'Ownership';
        else if (['ceiling', 'ceil'].includes(lower)) mapping[h] = 'Ceiling';
        else if (['floor'].includes(lower)) mapping[h] = 'Floor';
        else if (['stddev', 'std_dev', 'sd'].includes(lower)) mapping[h] = 'StdDev';
        else if (['opponent', 'opp', 'opponentid'].includes(lower)) mapping[h] = 'Opponent';
        else mapping[h] = 'Skip';
      });

      setCsvPreviewData({ headers, rows });
      setColumnMapping(mapping);
      setCsvPendingFile(capturedFile);
      setShowCsvPreview(true);
    };
    reader.readAsText(file);
  }, [currentSport, onSportChange]);

  const confirmCsvUpload = useCallback(async () => {
    if (!csvPendingFile || !currentSport) return;
    const sportConfig = SPORT_CONFIGS[currentSport];
    if (!sportConfig) return;
    setShowCsvPreview(false);

    try {
      const uploadResult = await dfsApi.uploadPlayers(csvPendingFile);
      if (uploadResult?.success) {
        const playersResponse = await dfsApi.getPlayers();
        const backendPlayers = (playersResponse?.players ?? []) as any[];
        const transformedPlayers: Player[] = backendPlayers.map((p: any) => ({
          id: p.id, name: p.name, team: p.team, position: p.position, salary: p.salary,
          projectedPoints: p.projection || p.projectedPoints || 0, minExp: p.minExposure ?? 0, maxExp: p.maxExposure ?? 100,
          selected: Boolean(p.selected), ownership: p.ownership || 0, locked: Boolean(p.locked), excluded: Boolean(p.excluded),
          ceiling: p.ceiling || undefined, floor: p.floor || undefined, stdDev: p.stdDev || undefined, opponent: p.opponent || undefined,
        }));
        onPlayerDataChange(transformedPlayers);
        onSelectedPlayersChange(transformedPlayers.map(p => p.id));
        const uniqueTeams = [...new Set(transformedPlayers.map(p => p.team))].filter(Boolean);
        onTeamSelectionsChange({ all: uniqueTeams, 2: uniqueTeams, 3: uniqueTeams, 4: uniqueTeams, 5: uniqueTeams });
        const initialStackSettings = initializeStackSettings(currentSport);
        onStackSettingsChange(initialStackSettings.map(s => ({ ...s, enabled: true })));
        onActiveTabChange('players');
        toast.success(`Loaded ${transformedPlayers.length} players`);
      } else { toast.error(`Upload failed: ${uploadResult?.error || 'Unknown error'}`); }
    } catch (error) {
      const apiError = dfsApi.handleApiError(error);
      toast.error(`Upload failed: ${apiError.message}`);
    }
    setCsvPendingFile(null);
  }, [csvPendingFile, currentSport, onPlayerDataChange, onSelectedPlayersChange, onTeamSelectionsChange, onStackSettingsChange, onActiveTabChange, initializeStackSettings]);

  const handleOwnershipUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    event.target.value = '';
    try {
      const result = await dfsApi.uploadOwnership(file);
      if (result?.success) toast.success(`Ownership merged: ${result.matched}/${result.total} players matched`);
    } catch (error) {
      const apiError = dfsApi.handleApiError(error);
      toast.error(`Ownership upload failed: ${apiError.message}`);
    }
  }, []);

  const handleAddProjectionSource = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    event.target.value = '';
    try {
      const sourceName = file.name.replace('.csv', '');
      const result = await dfsApi.uploadProjectionSource(file, sourceName);
      if (result?.success) {
        const playersMap: Record<string, number> = {};
        result.players.forEach((p: { name: string; projection: number }) => { playersMap[p.name.toLowerCase()] = p.projection; });
        setProjectionSources(prev => [...prev, { id: `src-${Date.now()}`, name: result.source, weight: 100, players: playersMap }]);
        toast.success(`Added source "${result.source}" (${result.players.length} players)`);
      }
    } catch { toast.error('Failed to add projection source'); }
  }, []);

  return {
    showCsvPreview, setShowCsvPreview,
    csvPreviewData, columnMapping, setColumnMapping,
    showBlendingDialog, setShowBlendingDialog,
    projectionSources, setProjectionSources,
    workspaceCsvInputRef, ownershipCsvInputRef, blendSourceInputRef,
    handleFileUpload, confirmCsvUpload,
    handleOwnershipUpload, handleAddProjectionSource,
  };
}
