import { Card } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Badge } from './ui/badge';
import { Upload, Download, Edit, Save } from 'lucide-react';
import { useState } from 'react';

interface ProjectionManagerProps {
  sport: string;
}

const mockProjections = [
  { id: 1, position: 'QB', name: 'Patrick Mahomes', team: 'KC', ourProj: 24.8, customProj: null, difference: 0 },
  { id: 2, position: 'QB', name: 'Josh Allen', team: 'BUF', ourProj: 23.4, customProj: 25.1, difference: 1.7 },
  { id: 3, position: 'RB', name: 'Christian McCaffrey', team: 'SF', ourProj: 22.1, customProj: null, difference: 0 },
  { id: 4, position: 'RB', name: 'Austin Ekeler', team: 'LAC', ourProj: 18.3, customProj: 17.5, difference: -0.8 },
  { id: 5, position: 'WR', name: 'Tyreek Hill', team: 'MIA', ourProj: 19.6, customProj: null, difference: 0 },
  { id: 6, position: 'WR', name: 'Justin Jefferson', team: 'MIN', ourProj: 18.8, customProj: 20.2, difference: 1.4 },
  { id: 7, position: 'TE', name: 'Travis Kelce', team: 'KC', ourProj: 15.2, customProj: null, difference: 0 },
];

export default function ProjectionManager({ sport }: ProjectionManagerProps) {
  const [editingId, setEditingId] = useState<number | null>(null);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-white mb-2">Manage Projections</h1>
          <p className="text-slate-400">View, edit, and import custom player projections</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" className="border-slate-600 text-slate-300">
            <Upload className="w-4 h-4 mr-2" />
            Import CSV
          </Button>
          <Button variant="outline" className="border-slate-600 text-slate-300">
            <Download className="w-4 h-4 mr-2" />
            Export CSV
          </Button>
        </div>
      </div>

      {/* Info Cards */}
      <div className="grid md:grid-cols-3 gap-6">
        <Card className="bg-slate-800 border-slate-700 p-6">
          <div className="text-slate-400 text-sm mb-1">Total Players</div>
          <div className="text-white text-3xl">247</div>
        </Card>
        <Card className="bg-slate-800 border-slate-700 p-6">
          <div className="text-slate-400 text-sm mb-1">Custom Projections</div>
          <div className="text-blue-400 text-3xl">42</div>
        </Card>
        <Card className="bg-slate-800 border-slate-700 p-6">
          <div className="text-slate-400 text-sm mb-1">Avg Difference</div>
          <div className="text-white text-3xl">+1.2</div>
        </Card>
      </div>

      {/* Projections Table */}
      <Card className="bg-slate-800 border-slate-700 p-6">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-white">Player Projections</h3>
          <Input
            placeholder="Search players..."
            className="bg-slate-700 border-slate-600 text-white max-w-xs"
          />
        </div>

        <div className="overflow-auto">
          <Table>
            <TableHeader>
              <TableRow className="border-slate-700">
                <TableHead className="text-slate-300">Pos</TableHead>
                <TableHead className="text-slate-300">Player</TableHead>
                <TableHead className="text-slate-300">Team</TableHead>
                <TableHead className="text-slate-300">Our Projection</TableHead>
                <TableHead className="text-slate-300">Your Projection</TableHead>
                <TableHead className="text-slate-300">Difference</TableHead>
                <TableHead className="text-slate-300">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {mockProjections.map((player) => (
                <TableRow key={player.id} className="border-slate-700">
                  <TableCell>
                    <Badge variant="outline" className="border-slate-600 text-slate-300">
                      {player.position}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-white">{player.name}</TableCell>
                  <TableCell className="text-slate-400">{player.team}</TableCell>
                  <TableCell className="text-slate-300">{player.ourProj.toFixed(1)}</TableCell>
                  <TableCell>
                    {editingId === player.id ? (
                      <Input
                        type="number"
                        step="0.1"
                        defaultValue={player.customProj || player.ourProj}
                        className="bg-slate-700 border-slate-600 text-white w-24"
                      />
                    ) : (
                      <span className={player.customProj ? 'text-blue-400' : 'text-slate-500'}>
                        {player.customProj?.toFixed(1) || '-'}
                      </span>
                    )}
                  </TableCell>
                  <TableCell>
                    {player.customProj && (
                      <span className={player.difference > 0 ? 'text-blue-400' : 'text-red-400'}>
                        {player.difference > 0 ? '+' : ''}{player.difference.toFixed(1)}
                      </span>
                    )}
                  </TableCell>
                  <TableCell>
                    {editingId === player.id ? (
                      <Button 
                        size="sm" 
                        onClick={() => setEditingId(null)}
                        className="bg-blue-500 hover:bg-blue-600 text-white"
                      >
                        <Save className="w-4 h-4 mr-1" />
                        Save
                      </Button>
                    ) : (
                      <Button 
                        size="sm" 
                        variant="ghost"
                        onClick={() => setEditingId(player.id)}
                        className="text-slate-400 hover:text-white"
                      >
                        <Edit className="w-4 h-4 mr-1" />
                        Edit
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </Card>

      {/* Import Guide */}
      <Card className="bg-slate-800 border-slate-700 p-6">
        <h3 className="text-white mb-4">Import Guide</h3>
        <div className="space-y-3 text-slate-300">
          <p>To import custom projections, prepare a CSV file with the following columns:</p>
          <ul className="list-disc list-inside space-y-1 text-slate-400">
            <li>Player Name (required)</li>
            <li>Team (required)</li>
            <li>Position (required)</li>
            <li>Projection (required - fantasy points)</li>
          </ul>
          <p className="text-slate-400">
            Your custom projections will override our default projections in the lineup optimizer.
          </p>
        </div>
      </Card>
    </div>
  );
}
