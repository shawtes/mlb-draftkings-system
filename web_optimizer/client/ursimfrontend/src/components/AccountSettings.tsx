import { useState } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { Switch } from './ui/switch';
import { Separator } from './ui/separator';
import {
  User,
  Mail,
  Lock,
  Bell,
  CreditCard,
  Shield,
  Eye,
  EyeOff,
  CheckCircle2,
  AlertCircle,
  Smartphone,
  Download,
  FileText,
  DollarSign,
  TrendingUp,
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { formatCurrency } from '../utils/formatters';

export default function AccountSettings() {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState('profile');
  const [showPassword, setShowPassword] = useState(false);
  const [twoFactorEnabled, setTwoFactorEnabled] = useState(false);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-white mb-2 bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
            Account Settings
          </h1>
          <p className="text-slate-400">Manage your account, security, and preferences</p>
        </div>
        <Badge className="bg-green-500/10 text-green-400 border-green-500/30">
          <CheckCircle2 className="w-3 h-3 mr-1" />
          Verified Account
        </Badge>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="bg-slate-800 border-slate-700 grid w-full grid-cols-2 lg:grid-cols-5">
          <TabsTrigger value="profile">Profile</TabsTrigger>
          <TabsTrigger value="security">Security</TabsTrigger>
          <TabsTrigger value="notifications">Notifications</TabsTrigger>
          <TabsTrigger value="billing">Billing</TabsTrigger>
          <TabsTrigger value="preferences">Preferences</TabsTrigger>
        </TabsList>

        {/* Profile Tab */}
        <TabsContent value="profile" className="space-y-6 mt-6">
          <Card className="bg-gradient-to-br from-slate-900 to-slate-800 border-cyan-500/20 p-6">
            <h3 className="text-white mb-6 flex items-center gap-2">
              <User className="w-5 h-5 text-cyan-400" />
              Personal Information
            </h3>

            <div className="space-y-4">
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <Label className="text-slate-300">First Name</Label>
                  <Input
                    defaultValue="John"
                    className="bg-slate-800 border-slate-700 text-white mt-1"
                  />
                </div>
                <div>
                  <Label className="text-slate-300">Last Name</Label>
                  <Input
                    defaultValue="Doe"
                    className="bg-slate-800 border-slate-700 text-white mt-1"
                  />
                </div>
              </div>

              <div>
                <Label className="text-slate-300">Email Address</Label>
                <div className="relative mt-1">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                  <Input
                    type="email"
                    defaultValue={user?.email || ''}
                    className="bg-slate-800 border-slate-700 text-white pl-10"
                    disabled
                  />
                  <Badge className="absolute right-3 top-1/2 -translate-y-1/2 bg-green-500/10 text-green-400 border-green-500/30 text-xs">
                    Verified
                  </Badge>
                </div>
              </div>

              <div>
                <Label className="text-slate-300">Username</Label>
                <Input
                  defaultValue="johndoe123"
                  className="bg-slate-800 border-slate-700 text-white mt-1"
                />
              </div>

              <div>
                <Label className="text-slate-300">Bio</Label>
                <textarea
                  defaultValue="Professional DFS player and sports betting enthusiast"
                  className="w-full bg-slate-800 border border-slate-700 text-white rounded-lg p-3 mt-1 min-h-[100px] resize-none"
                  placeholder="Tell us about yourself..."
                />
              </div>

              <div className="flex justify-end gap-3 pt-4">
                <Button variant="outline" className="border-slate-700 text-slate-300">
                  Cancel
                </Button>
                <Button className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500">
                  Save Changes
                </Button>
              </div>
            </div>
          </Card>

          {/* Account Stats */}
          <div className="grid md:grid-cols-3 gap-6">
            <Card className="bg-gradient-to-br from-slate-900 to-slate-800 border-cyan-500/20 p-6">
              <div className="flex items-center gap-3 mb-2">
                <div className="p-2 bg-cyan-500/10 rounded-lg">
                  <TrendingUp className="w-5 h-5 text-cyan-400" />
                </div>
                <div className="text-sm text-slate-400">Member Since</div>
              </div>
              <div className="text-2xl text-white font-bold">Jan 2024</div>
              <div className="text-xs text-cyan-400 mt-1">14 months active</div>
            </Card>

            <Card className="bg-gradient-to-br from-slate-900 to-slate-800 border-blue-500/20 p-6">
              <div className="flex items-center gap-3 mb-2">
                <div className="p-2 bg-blue-500/10 rounded-lg">
                  <FileText className="w-5 h-5 text-blue-400" />
                </div>
                <div className="text-sm text-slate-400">Total Lineups</div>
              </div>
              <div className="text-2xl text-white font-bold">1,247</div>
              <div className="text-xs text-blue-400 mt-1">+32 this week</div>
            </Card>

            <Card className="bg-gradient-to-br from-slate-900 to-slate-800 border-purple-500/20 p-6">
              <div className="flex items-center gap-3 mb-2">
                <div className="p-2 bg-purple-500/10 rounded-lg">
                  <DollarSign className="w-5 h-5 text-purple-400" />
                </div>
                <div className="text-sm text-slate-400">Total ROI</div>
              </div>
              <div className="text-2xl text-white font-bold">+18.5%</div>
              <div className="text-xs text-purple-400 mt-1">Last 30 days</div>
            </Card>
          </div>
        </TabsContent>

        {/* Security Tab */}
        <TabsContent value="security" className="space-y-6 mt-6">
          <Card className="bg-gradient-to-br from-slate-900 to-slate-800 border-cyan-500/20 p-6">
            <h3 className="text-white mb-6 flex items-center gap-2">
              <Shield className="w-5 h-5 text-cyan-400" />
              Security Settings
            </h3>

            <div className="space-y-6">
              {/* Password */}
              <div>
                <Label className="text-slate-300 mb-3 block">Change Password</Label>
                <div className="space-y-3">
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                    <Input
                      type={showPassword ? 'text' : 'password'}
                      placeholder="Current password"
                      className="bg-slate-800 border-slate-700 text-white pl-10 pr-10"
                    />
                    <button
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                    >
                      {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                  <Input
                    type="password"
                    placeholder="New password"
                    className="bg-slate-800 border-slate-700 text-white"
                  />
                  <Input
                    type="password"
                    placeholder="Confirm new password"
                    className="bg-slate-800 border-slate-700 text-white"
                  />
                </div>
              </div>

              <Separator className="bg-slate-700" />

              {/* Two-Factor Authentication */}
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3 flex-1">
                  <div className="p-2 bg-green-500/10 rounded-lg mt-1">
                    <Smartphone className="w-5 h-5 text-green-400" />
                  </div>
                  <div>
                    <h4 className="text-white font-medium mb-1">Two-Factor Authentication</h4>
                    <p className="text-slate-400 text-sm mb-3">
                      Add an extra layer of security to your account
                    </p>
                    {twoFactorEnabled && (
                      <Badge className="bg-green-500/10 text-green-400 border-green-500/30">
                        <CheckCircle2 className="w-3 h-3 mr-1" />
                        Enabled
                      </Badge>
                    )}
                  </div>
                </div>
                <Switch
                  checked={twoFactorEnabled}
                  onCheckedChange={setTwoFactorEnabled}
                />
              </div>

              <Separator className="bg-slate-700" />

              {/* SSL Certificate */}
              <div className="flex items-start gap-3 p-4 bg-green-500/5 border border-green-500/20 rounded-lg">
                <CheckCircle2 className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
                <div>
                  <h4 className="text-green-400 font-medium mb-1">SSL Encrypted Connection</h4>
                  <p className="text-slate-400 text-sm">
                    All your data is transmitted securely using 256-bit SSL encryption
                  </p>
                </div>
              </div>

              {/* Active Sessions */}
              <div>
                <Label className="text-slate-300 mb-3 block">Active Sessions</Label>
                <div className="space-y-2">
                  <div className="flex items-center justify-between p-3 bg-slate-800/50 border border-slate-700 rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-cyan-500/10 rounded-lg">
                        <Smartphone className="w-4 h-4 text-cyan-400" />
                      </div>
                      <div>
                        <div className="text-white text-sm">Windows • Chrome</div>
                        <div className="text-slate-400 text-xs">New York, US • Current session</div>
                      </div>
                    </div>
                    <Badge className="bg-green-500/10 text-green-400 border-green-500/30 text-xs">
                      Active
                    </Badge>
                  </div>
                </div>
              </div>

              <div className="flex justify-end gap-3 pt-4">
                <Button variant="outline" className="border-slate-700 text-slate-300">
                  Cancel
                </Button>
                <Button className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500">
                  Update Security
                </Button>
              </div>
            </div>
          </Card>

          {/* Security Audit */}
          <Card className="bg-gradient-to-br from-slate-900 to-slate-800 border-blue-500/20 p-6">
            <h3 className="text-white mb-4 flex items-center gap-2">
              <Shield className="w-5 h-5 text-blue-400" />
              Security Audit
            </h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-green-500/5 border border-green-500/20 rounded-lg">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-green-400" />
                  <span className="text-sm text-slate-300">Strong password</span>
                </div>
                <Badge className="bg-green-500/10 text-green-400 border-green-500/30 text-xs">
                  Passed
                </Badge>
              </div>
              <div className="flex items-center justify-between p-3 bg-green-500/5 border border-green-500/20 rounded-lg">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-green-400" />
                  <span className="text-sm text-slate-300">Email verified</span>
                </div>
                <Badge className="bg-green-500/10 text-green-400 border-green-500/30 text-xs">
                  Passed
                </Badge>
              </div>
              <div className="flex items-center justify-between p-3 bg-yellow-500/5 border border-yellow-500/20 rounded-lg">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-yellow-400" />
                  <span className="text-sm text-slate-300">Two-factor authentication</span>
                </div>
                <Badge className="bg-yellow-500/10 text-yellow-400 border-yellow-500/30 text-xs">
                  {twoFactorEnabled ? 'Passed' : 'Recommended'}
                </Badge>
              </div>
            </div>
          </Card>
        </TabsContent>

        {/* Notifications Tab */}
        <TabsContent value="notifications" className="space-y-6 mt-6">
          <Card className="bg-gradient-to-br from-slate-900 to-slate-800 border-cyan-500/20 p-6">
            <h3 className="text-white mb-6 flex items-center gap-2">
              <Bell className="w-5 h-5 text-cyan-400" />
              Notification Preferences
            </h3>

            <div className="space-y-6">
              {[
                { label: 'Injury News Alerts', description: 'Get notified of player injuries', enabled: true },
                { label: 'Line Movement', description: 'Alert when betting lines move significantly', enabled: true },
                { label: 'Prop Value Alerts', description: 'Notify when high-value props are found', enabled: true },
                { label: 'Lineup Export Ready', description: 'Alert when lineup generation completes', enabled: false },
                { label: 'Weekly Summary', description: 'Receive weekly performance summaries', enabled: true },
                { label: 'Marketing Emails', description: 'Promotional offers and new features', enabled: false },
              ].map((item, index) => (
                <div key={index}>
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="text-white text-sm font-medium">{item.label}</h4>
                      <p className="text-slate-400 text-xs mt-0.5">{item.description}</p>
                    </div>
                    <Switch defaultChecked={item.enabled} />
                  </div>
                  {index < 5 && <Separator className="bg-slate-700 mt-4" />}
                </div>
              ))}
            </div>
          </Card>
        </TabsContent>

        {/* Billing Tab */}
        <TabsContent value="billing" className="space-y-6 mt-6">
          <Card className="bg-gradient-to-br from-slate-900 to-slate-800 border-cyan-500/20 p-6">
            <h3 className="text-white mb-6 flex items-center gap-2">
              <CreditCard className="w-5 h-5 text-cyan-400" />
              Subscription & Billing
            </h3>

            {/* Current Plan */}
            <div className="bg-gradient-to-r from-cyan-500/10 to-blue-500/10 border border-cyan-500/30 rounded-lg p-6 mb-6">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h4 className="text-white text-lg font-semibold mb-1">Pro Plan</h4>
                  <p className="text-slate-400 text-sm">Unlimited lineups & advanced features</p>
                </div>
                <Badge className="bg-cyan-500/20 text-cyan-400 border-cyan-500/40">
                  Active
                </Badge>
              </div>
              <div className="flex items-baseline gap-2 mb-4">
                <span className="text-4xl text-white font-bold">{formatCurrency(49.99)}</span>
                <span className="text-slate-400">/month</span>
              </div>
              <div className="text-slate-400 text-sm mb-4">
                Next billing date: <span className="text-white">March 15, 2025</span>
              </div>
              <div className="flex gap-3">
                <Button variant="outline" className="border-slate-700 text-slate-300">
                  Change Plan
                </Button>
                <Button variant="outline" className="border-red-500/30 text-red-400 hover:bg-red-500/10">
                  Cancel Subscription
                </Button>
              </div>
            </div>

            {/* Payment Method */}
            <div className="mb-6">
              <h4 className="text-white mb-4">Payment Method</h4>
              <div className="flex items-center justify-between p-4 bg-slate-800/50 border border-slate-700 rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-blue-500/10 rounded">
                    <CreditCard className="w-5 h-5 text-blue-400" />
                  </div>
                  <div>
                    <div className="text-white text-sm">•••• •••• •••• 4242</div>
                    <div className="text-slate-400 text-xs">Expires 12/25</div>
                  </div>
                </div>
                <Button variant="outline" size="sm" className="border-slate-700 text-slate-300">
                  Update
                </Button>
              </div>
            </div>

            {/* Billing History */}
            <div>
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-white">Billing History</h4>
                <Button variant="outline" size="sm" className="border-slate-700 text-slate-300">
                  <Download className="w-4 h-4 mr-2" />
                  Download All
                </Button>
              </div>
              <div className="space-y-2">
                {[
                  { date: 'Feb 15, 2025', amount: 49.99, status: 'Paid' },
                  { date: 'Jan 15, 2025', amount: 49.99, status: 'Paid' },
                  { date: 'Dec 15, 2024', amount: 49.99, status: 'Paid' },
                ].map((invoice, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-slate-800/30 border border-slate-700/50 rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className="text-white text-sm">{invoice.date}</div>
                      <Badge className="bg-green-500/10 text-green-400 border-green-500/30 text-xs">
                        {invoice.status}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-white font-semibold">{formatCurrency(invoice.amount)}</span>
                      <Button variant="ghost" size="sm" className="text-cyan-400 hover:text-cyan-300">
                        <Download className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </TabsContent>

        {/* Preferences Tab */}
        <TabsContent value="preferences" className="space-y-6 mt-6">
          <Card className="bg-gradient-to-br from-slate-900 to-slate-800 border-cyan-500/20 p-6">
            <h3 className="text-white mb-6">Display Preferences</h3>
            
            <div className="space-y-6">
              <div>
                <Label className="text-slate-300 mb-3 block">Default Odds Format</Label>
                <select className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white">
                  <option value="american">American (-110, +150)</option>
                  <option value="decimal">Decimal (1.91, 2.50)</option>
                  <option value="fractional">Fractional (10/11, 3/2)</option>
                </select>
              </div>

              <div>
                <Label className="text-slate-300 mb-3 block">Default Sport</Label>
                <select className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white">
                  <option value="nfl">NFL</option>
                  <option value="nba">NBA</option>
                  <option value="mlb">MLB</option>
                  <option value="nhl">NHL</option>
                </select>
              </div>

              <div>
                <Label className="text-slate-300 mb-3 block">Time Zone</Label>
                <select className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white">
                  <option value="est">Eastern Time (ET)</option>
                  <option value="cst">Central Time (CT)</option>
                  <option value="mst">Mountain Time (MT)</option>
                  <option value="pst">Pacific Time (PT)</option>
                </select>
              </div>

              <Separator className="bg-slate-700" />

              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-white text-sm font-medium">Compact Mode</h4>
                  <p className="text-slate-400 text-xs mt-0.5">Show more data in less space</p>
                </div>
                <Switch />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-white text-sm font-medium">Auto-refresh Data</h4>
                  <p className="text-slate-400 text-xs mt-0.5">Automatically update odds and lineups</p>
                </div>
                <Switch defaultChecked />
              </div>
            </div>

            <div className="flex justify-end gap-3 pt-6">
              <Button variant="outline" className="border-slate-700 text-slate-300">
                Reset to Default
              </Button>
              <Button className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500">
                Save Preferences
              </Button>
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

