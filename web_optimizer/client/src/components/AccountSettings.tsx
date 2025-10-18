import React from 'react';

const AccountSettings = React.memo(() => {
  return (
    <div className="h-full overflow-auto p-6">
      {/* Main Card Container */}
      <div className="bg-slate-800 backdrop-blur-sm rounded-2xl border border-cyan-500/20 shadow-2xl relative overflow-hidden min-h-full">
        
        {/* Content */}
        <div className="relative z-10 flex items-center justify-center min-h-full p-12">
          <div className="text-center max-w-3xl">
          <div className="mb-8">
            <div className="w-24 h-24 mx-auto bg-gradient-to-br from-cyan-500 to-blue-600 rounded-2xl flex items-center justify-center shadow-lg shadow-cyan-500/30 mb-6">
              <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </div>
            <h1 className="text-5xl font-bold bg-gradient-to-r from-cyan-400 via-cyan-300 to-blue-400 bg-clip-text text-transparent mb-4 tracking-tight">
            Account Settings
          </h1>
            <p className="text-slate-300 text-xl mb-2 font-medium">
              Manage your preferences and account
            </p>
              </div>

          <div className="bg-slate-700/40 border border-slate-600/50 rounded-2xl p-10 backdrop-blur-sm shadow-xl">
            <h2 className="text-cyan-300 text-2xl font-semibold mb-4 tracking-wide">
              Ready to Build
            </h2>
            <p className="text-slate-200 leading-relaxed text-lg">
              This component is ready for your custom settings implementation.
              Build your profile, security, notifications, and preferences here.
            </p>
              </div>

          <div className="mt-8 flex items-center justify-center gap-3">
            <div className="w-2.5 h-2.5 bg-cyan-400 rounded-full animate-pulse shadow-lg shadow-cyan-400/50" />
            <span className="text-slate-300 text-base font-medium">Clean slate ready</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
  );
});

AccountSettings.displayName = 'AccountSettings';

export default AccountSettings;
