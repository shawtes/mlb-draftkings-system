import React, { useState, useCallback } from 'react';
import { 
  User, 
  CreditCard, 
  FileText, 
  Settings as SettingsIcon,
  ChevronRight,
  Mail,
  Lock,
  Download,
  Bell,
  Palette,
  Database,
  Menu,
  UserCircle,
  CreditCard as PaymentIcon,
  FileSpreadsheet,
  Trophy,
  Shield
} from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { useIsMobile } from './ui/use-mobile';
import { Sheet, SheetContent, SheetTrigger } from './ui/sheet';
import { Switch } from './ui/switch';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';

type SettingsSection = 'profile' | 'lineups' | 'billing' | 'general';

interface SectionItem {
  id: SettingsSection;
  label: string;
  icon: React.ReactNode;
  description: string;
}

const sections: SectionItem[] = [
  {
    id: 'profile',
    label: 'User & Account',
    icon: <User className="w-5 h-5" />,
    description: 'Manage your profile and account details'
  },
  {
    id: 'lineups',
    label: 'Lineup Management',
    icon: <Trophy className="w-5 h-5" />,
    description: 'View and manage your lineups'
  },
  {
    id: 'billing',
    label: 'Billing',
    icon: <CreditCard className="w-5 h-5" />,
    description: 'Subscription and payment information'
  },
  {
    id: 'general',
    label: 'General Settings',
    icon: <SettingsIcon className="w-5 h-5" />,
    description: 'Preferences and app settings'
  }
];

const AccountSettings = React.memo(() => {
  const [activeSection, setActiveSection] = useState<SettingsSection>('profile');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const isMobile = useIsMobile();

  const handleSectionChange = useCallback((section: SettingsSection) => {
    setActiveSection(section);
    if (isMobile) {
      setMobileMenuOpen(false);
    }
  }, [isMobile]);

  const SidebarContent = () => (
    <div className="flex flex-col h-full">
      <div className="p-6 border-b border-slate-700/50">
        <h2 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 via-cyan-300 to-blue-400 bg-clip-text text-transparent">
          Settings
        </h2>
        <p className="text-slate-400 text-sm mt-1">Manage your account and preferences</p>
      </div>
      
      <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
        {sections.map((section) => {
          const isActive = activeSection === section.id;
          return (
            <button
              key={section.id}
              onClick={() => handleSectionChange(section.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 text-left group ${
                isActive
                  ? 'bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 shadow-lg shadow-cyan-500/5'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent'
              }`}
            >
              <div className={`${isActive ? 'text-cyan-400' : 'text-slate-500 group-hover:text-slate-300'}`}>
                {section.icon}
              </div>
              <div className="flex-1 min-w-0">
                <div className={`font-medium ${isActive ? 'text-cyan-300' : 'text-slate-300'}`}>
                  {section.label}
                </div>
                <div className={`text-xs mt-0.5 ${isActive ? 'text-cyan-400/70' : 'text-slate-500'}`}>
                  {section.description}
                </div>
              </div>
              {isActive && (
                <ChevronRight className="w-4 h-4 text-cyan-400" />
              )}
            </button>
          );
        })}
      </nav>
    </div>
  );


  return (
    <div className="h-screen flex bg-slate-900 text-white overflow-hidden">
      {!isMobile && (
        <aside className="w-64 bg-slate-800/95 backdrop-blur-sm border-r border-slate-700/50 flex-shrink-0 flex flex-col h-full">
          <SidebarContent />
        </aside>
      )}
      
      {isMobile && (
        <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
          <SheetTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="fixed top-4 left-4 z-50 bg-slate-800/90 backdrop-blur-sm border border-slate-700/50 text-slate-300 hover:text-cyan-400 hover:bg-slate-700/50 shadow-lg"
            >
              <Menu className="w-5 h-5" />
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="w-[280px] p-0 bg-slate-900 border-slate-700/50">
            <SidebarContent />
          </SheetContent>
        </Sheet>
      )}
      
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-6xl mx-auto p-6 lg:p-8 pb-12 lg:pb-16">
          {activeSection === 'profile' && <ProfileSection />}
          {activeSection === 'lineups' && <LineupsSection />}
          {activeSection === 'billing' && <BillingSection />}
          {activeSection === 'general' && <GeneralSection />}
        </div>
      </main>
    </div>
  );
});

// Profile Section Component
const ProfileSection = React.memo(() => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">User & Account</h1>
        <p className="text-slate-400">Manage your profile information and account settings</p>
      </div>

      <div className="grid gap-6">
        {/* Profile Information */}
        <Card className="bg-slate-800/50 border-slate-700/50">
          <CardHeader>
            <CardTitle className="text-xl text-cyan-300 flex items-center gap-2">
              <UserCircle className="w-5 h-5" />
              Profile Information
            </CardTitle>
            <CardDescription className="text-slate-400">
              Update your personal information and profile details
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-6 pb-4 border-b border-slate-700/50">
              <div className="w-20 h-20 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center shadow-lg shadow-cyan-500/30">
                <User className="w-10 h-10 text-white" />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-white mb-1">Profile Picture</h3>
                <p className="text-sm text-slate-400 mb-3">JPG, PNG or GIF. Max size 2MB</p>
                <Button variant="outline" size="sm" className="border-slate-600 text-slate-300 hover:bg-slate-700">
                  Change Avatar
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="firstName" className="text-slate-300">First Name</Label>
                <Input
                  id="firstName"
                  placeholder="John"
                  className="bg-slate-900/50 border-slate-600 text-white placeholder:text-slate-500 focus:border-cyan-500"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="lastName" className="text-slate-300">Last Name</Label>
                <Input
                  id="lastName"
                  placeholder="Doe"
                  className="bg-slate-900/50 border-slate-600 text-white placeholder:text-slate-500 focus:border-cyan-500"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="email" className="text-slate-300 flex items-center gap-2">
                  <Mail className="w-4 h-4" />
                  Email Address
                </Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="john.doe@example.com"
                  className="bg-slate-900/50 border-slate-600 text-white placeholder:text-slate-500 focus:border-cyan-500"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="username" className="text-slate-300">Username</Label>
                <Input
                  id="username"
                  placeholder="johndoe"
                  className="bg-slate-900/50 border-slate-600 text-white placeholder:text-slate-500 focus:border-cyan-500"
                />
              </div>
            </div>

            <div className="flex justify-end pt-4">
              <Button className="bg-cyan-500 hover:bg-cyan-600 text-white">
                Save Changes
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Account Security */}
        <Card className="bg-slate-800/50 border-slate-700/50">
          <CardHeader>
            <CardTitle className="text-xl text-cyan-300 flex items-center gap-2">
              <Shield className="w-5 h-5" />
              Account Security
            </CardTitle>
            <CardDescription className="text-slate-400">
              Manage your password and security settings
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="currentPassword" className="text-slate-300 flex items-center gap-2">
                <Lock className="w-4 h-4" />
                Current Password
              </Label>
              <Input
                id="currentPassword"
                type="password"
                placeholder="Enter current password"
                className="bg-slate-900/50 border-slate-600 text-white placeholder:text-slate-500 focus:border-cyan-500"
              />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="newPassword" className="text-slate-300">New Password</Label>
                <Input
                  id="newPassword"
                  type="password"
                  placeholder="Enter new password"
                  className="bg-slate-900/50 border-slate-600 text-white placeholder:text-slate-500 focus:border-cyan-500"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="confirmPassword" className="text-slate-300">Confirm Password</Label>
                <Input
                  id="confirmPassword"
                  type="password"
                  placeholder="Confirm new password"
                  className="bg-slate-900/50 border-slate-600 text-white placeholder:text-slate-500 focus:border-cyan-500"
                />
              </div>
            </div>
            <div className="flex justify-end pt-4">
              <Button className="bg-cyan-500 hover:bg-cyan-600 text-white">
                Update Password
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
});

// Lineups Section Component
const LineupsSection = React.memo(() => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Lineup Management</h1>
        <p className="text-slate-400">View and manage your previous lineups and saved configurations</p>
      </div>

      <div className="grid gap-6">
        <Card className="bg-slate-800/50 border-slate-700/50">
          <CardHeader>
            <CardTitle className="text-xl text-cyan-300 flex items-center gap-2">
              <Trophy className="w-5 h-5" />
              Previous Lineups
            </CardTitle>
            <CardDescription className="text-slate-400">
              Access your lineup history and saved configurations
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-center py-12">
              <Trophy className="w-16 h-16 text-slate-600 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-slate-300 mb-2">No Lineups Yet</h3>
              <p className="text-slate-500 mb-6">Your previous lineups will appear here</p>
              <Button className="bg-cyan-500 hover:bg-cyan-600 text-white">
                Create New Lineup
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700/50">
          <CardHeader>
            <CardTitle className="text-xl text-cyan-300 flex items-center gap-2">
              <FileSpreadsheet className="w-5 h-5" />
              Export History
            </CardTitle>
            <CardDescription className="text-slate-400">
              Download your lineup data and export history
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-4 bg-slate-900/50 rounded-lg border border-slate-700/50">
                <div className="flex items-center gap-3">
                  <Download className="w-5 h-5 text-slate-400" />
                  <div>
                    <div className="text-sm font-medium text-white">Export All Lineups</div>
                    <div className="text-xs text-slate-500">CSV format</div>
                  </div>
                </div>
                <Button variant="outline" size="sm" className="border-slate-600 text-slate-300 hover:bg-slate-700">
                  Export
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
});

// Billing Section Component
const BillingSection = React.memo(() => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Billing & Subscription</h1>
        <p className="text-slate-400">Manage your subscription, payment methods, and billing history</p>
      </div>

      <div className="grid gap-6">
        <Card className="bg-slate-800/50 border-slate-700/50">
          <CardHeader>
            <CardTitle className="text-xl text-cyan-300 flex items-center gap-2">
              <CreditCard className="w-5 h-5" />
              Current Plan
            </CardTitle>
            <CardDescription className="text-slate-400">
              Your current subscription details
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between p-6 bg-gradient-to-r from-cyan-500/10 to-blue-500/10 rounded-lg border border-cyan-500/20">
              <div>
                <h3 className="text-2xl font-bold text-white mb-1">Free Plan</h3>
                <p className="text-slate-400">Basic features and limited access</p>
              </div>
              <Button className="bg-cyan-500 hover:bg-cyan-600 text-white">
                Upgrade Plan
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700/50">
          <CardHeader>
            <CardTitle className="text-xl text-cyan-300 flex items-center gap-2">
              <PaymentIcon className="w-5 h-5" />
              Payment Methods
            </CardTitle>
            <CardDescription className="text-slate-400">
              Manage your payment methods
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-center py-12">
              <CreditCard className="w-16 h-16 text-slate-600 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-slate-300 mb-2">No Payment Methods</h3>
              <p className="text-slate-500 mb-6">Add a payment method to upgrade your plan</p>
              <Button className="bg-cyan-500 hover:bg-cyan-600 text-white">
                Add Payment Method
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700/50">
          <CardHeader>
            <CardTitle className="text-xl text-cyan-300 flex items-center gap-2">
              <FileText className="w-5 h-5" />
              Billing History
            </CardTitle>
            <CardDescription className="text-slate-400">
              View your past invoices and transactions
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-center py-12">
              <FileText className="w-16 h-16 text-slate-600 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-slate-300 mb-2">No Billing History</h3>
              <p className="text-slate-500">Your invoices will appear here</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
});

// General Settings Section Component
const GeneralSection = React.memo(() => {
  const [notifications, setNotifications] = useState(true);
  const [emailUpdates, setEmailUpdates] = useState(true);
  const [darkMode, setDarkMode] = useState(true);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">General Settings</h1>
        <p className="text-slate-400">Customize your app experience and preferences</p>
      </div>

      <div className="grid gap-6">
        <Card className="bg-slate-800/50 border-slate-700/50">
          <CardHeader>
            <CardTitle className="text-xl text-cyan-300 flex items-center gap-2">
              <Bell className="w-5 h-5" />
              Notifications
            </CardTitle>
            <CardDescription className="text-slate-400">
              Manage your notification preferences
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between py-3 border-b border-slate-700/50">
              <div className="flex-1">
                <div className="font-medium text-white">Push Notifications</div>
                <div className="text-sm text-slate-400">Receive browser notifications</div>
              </div>
              <Switch
                checked={notifications}
                onCheckedChange={setNotifications}
                className="data-[state=checked]:bg-cyan-500"
              />
            </div>
            <div className="flex items-center justify-between py-3">
              <div className="flex-1">
                <div className="font-medium text-white">Email Updates</div>
                <div className="text-sm text-slate-400">Receive email notifications about updates</div>
              </div>
              <Switch
                checked={emailUpdates}
                onCheckedChange={setEmailUpdates}
                className="data-[state=checked]:bg-cyan-500"
              />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700/50">
          <CardHeader>
            <CardTitle className="text-xl text-cyan-300 flex items-center gap-2">
              <Palette className="w-5 h-5" />
              Appearance
            </CardTitle>
            <CardDescription className="text-slate-400">
              Customize the look and feel of the app
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between py-3">
              <div className="flex-1">
                <div className="font-medium text-white">Dark Mode</div>
                <div className="text-sm text-slate-400">Use dark theme</div>
              </div>
              <Switch
                checked={darkMode}
                onCheckedChange={setDarkMode}
                className="data-[state=checked]:bg-cyan-500"
              />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700/50">
          <CardHeader>
            <CardTitle className="text-xl text-cyan-300 flex items-center gap-2">
              <Database className="w-5 h-5" />
              Data Management
            </CardTitle>
            <CardDescription className="text-slate-400">
              Export or delete your account data
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between py-3 border-b border-slate-700/50">
              <div className="flex-1">
                <div className="font-medium text-white">Export Data</div>
                <div className="text-sm text-slate-400">Download all your account data</div>
              </div>
              <Button variant="outline" size="sm" className="border-slate-600 text-slate-300 hover:bg-slate-700">
                <Download className="w-4 h-4 mr-2" />
                Export
              </Button>
            </div>
            <div className="flex items-center justify-between py-3">
              <div className="flex-1">
                <div className="font-medium text-red-400">Delete Account</div>
                <div className="text-sm text-slate-400">Permanently delete your account and all data</div>
              </div>
              <Button variant="outline" size="sm" className="border-red-600/50 text-red-400 hover:bg-red-500/10">
                Delete
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
});

ProfileSection.displayName = 'ProfileSection';
LineupsSection.displayName = 'LineupsSection';
BillingSection.displayName = 'BillingSection';
GeneralSection.displayName = 'GeneralSection';

AccountSettings.displayName = 'AccountSettings';

export default AccountSettings;