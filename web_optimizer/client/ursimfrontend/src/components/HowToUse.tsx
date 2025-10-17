import { useState } from 'react';
import { Card } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from './ui/accordion';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { 
  BookOpen, 
  PlayCircle, 
  Lightbulb, 
  HelpCircle, 
  Search,
  Trophy,
  Target,
  BarChart3,
  Zap,
  Shield,
  TrendingUp,
  Users,
  Calculator
} from 'lucide-react';

export default function HowToUse() {
  const [searchQuery, setSearchQuery] = useState('');

  const tutorials = [
    {
      id: 'getting-started',
      title: 'Getting Started',
      icon: Zap,
      description: 'Learn the basics of UrSim and set up your account',
      content: [
        {
          question: 'What is UrSim?',
          answer: 'UrSim is a professional DFS (Daily Fantasy Sports) lineup optimizer and prop betting analysis platform. It combines advanced algorithms, real-time data, and powerful tools to help you build winning lineups and identify valuable betting opportunities.'
        },
        {
          question: 'How do I get started?',
          answer: 'Start by navigating to the Games Hub to research upcoming games and props. For DFS, head to the DFS Lineup Optimizer to upload your player pool. For prop betting, visit the Prop Betting Center to find edges and build parlays.'
        },
        {
          question: 'What sports are supported?',
          answer: 'UrSim supports NFL, NBA, MLB, NHL, and other major sports. The platform automatically adjusts position requirements and optimization strategies based on the sport selected.'
        }
      ]
    },
    {
      id: 'dfs-optimizer-beginner',
      title: 'DFS Optimizer - Beginner',
      icon: Users,
      description: 'Step-by-step guide to using the DFS Optimizer for the first time',
      content: [
        {
          question: '🚀 Getting Started - Your First Lineup',
          answer: 'Welcome to the DFS Optimizer! This tool helps you build winning DraftKings lineups using advanced algorithms. Follow these steps to create your first optimized lineup in minutes.'
        },
        {
          question: 'Step 1: Load Your Player Pool',
          answer: 'In the Control Panel (right side), click "Load CSV" button. Select your DraftKings player export file. The system will parse the CSV and populate the Players tab with all available players. You should see columns for Name, Team, Position, Salary, and Projected Points. The status bar will show "X players loaded".'
        },
        {
          question: 'Step 2: Select Players (Players Tab)',
          answer: 'Click the "Players" tab on the left. You will see all players organized by position. Use the position filters at the top (All Batters, C, 1B, 2B, 3B, SS, OF, P) to view specific positions. Click checkboxes to select players you want in your lineups. TIP: Select at least 30-50 players for good lineup diversity. Use "Select All" to quickly select everyone, then uncheck players you want to avoid.'
        },
        {
          question: 'Step 3: Understanding the Player Table',
          answer: 'The player table shows key information: Salary (what they cost), Proj (projected points), and Value (points per $1000 of salary). Higher value = more efficient. The Min Exp and Max Exp columns let you control how often a player appears (0-100%). Leave these at defaults (0 min, 100 max) for your first optimization.'
        },
        {
          question: 'Step 4: Configure Team Stacks',
          answer: 'Click the "Team Stacks" tab. This is where you choose which teams to stack (use multiple players from). Click a stack size tab at the top (start with "4 Stack"). Check the teams you want to stack - typically teams with high projected run totals (shown in green). For beginners, select 2-3 top offensive teams. You can skip this step for your first run if you want.'
        },
        {
          question: 'Step 5: Enable Stack Types (Stack Exposure Tab)',
          answer: 'Click "Stack Exposure" tab. You will see a list of stack types (5, 4, 3, 2, No Stacks). Check the box next to "4" to enable 4-player stacks. This tells the optimizer to build lineups with 4 players from the same team. For your first run, just enable one stack type. Leave Min Exp and Max Exp at defaults (0 and 100).'
        },
        {
          question: 'Step 6: Set Number of Lineups',
          answer: 'In the Control Panel (right side), find "Number of Lineups" under Optimization Settings. For your first run, try 20 lineups. This is a good starting point to see how the optimizer works. You can increase to 100-150 once you are comfortable.'
        },
        {
          question: 'Step 7: Run the Optimization',
          answer: 'Click the blue "Run Contest Sim" button in the Control Panel under Actions. The button will change to "Optimizing..." and the optimizer will generate your lineups. This may take 10-60 seconds depending on settings. Watch the Status Bar at the bottom of the Control Panel for progress updates.'
        },
        {
          question: 'Step 8: Review Your Results',
          answer: 'After optimization completes, the Results Summary in the Control Panel will show: Lineups Generated, Avg Points, and Avg Salary. This gives you a quick overview of your lineup pool. All generated lineups meet DraftKings constraints (salary cap, position requirements).'
        },
        {
          question: 'Step 9: Export to DraftKings',
          answer: 'Click "Save CSV for DK" button in the Control Panel. This downloads a CSV file formatted for DraftKings upload. Go to DraftKings.com, navigate to your contest, and use their "Upload Lineups" feature to import your CSV file. Your lineups will be instantly entered!'
        },
        {
          question: '⚠️ Common Beginner Mistakes',
          answer: 'Mistake 1: Selecting too few players (under 30) - this limits diversity. Mistake 2: Not enabling any stack types - stacking is crucial for correlation. Mistake 3: Using too high min exposure - this forces players into too many lineups. Mistake 4: Setting number of lineups too high on first try - start with 20-50. Mistake 5: Not checking team stacks - always verify your stack selections before running.'
        },
        {
          question: '💡 Quick Tips for Beginners',
          answer: 'Tip 1: Start simple - use just the Players tab and Stack Exposure tab for your first few runs. Tip 2: Focus on 4-stacks - they offer good correlation without being too risky. Tip 3: Select players with high "Value" scores - these are the most efficient. Tip 4: Generate 20-50 lineups to start - enough for diversity without being overwhelming. Tip 5: Review the "By Run" breakdown in My Entries tab to track your optimization sessions.'
        },
        {
          question: '❓ Troubleshooting',
          answer: 'Problem: "No lineups generated" - Make sure you selected enough players (30+) and enabled at least one stack type. Problem: "All lineups look the same" - Increase Min Unique setting in Control Panel. Problem: "Optimization taking too long" - Reduce number of lineups or simplify stack settings. Problem: "Players not loading" - Check CSV format (must have Name, Team, Position, Salary, Predicted_DK_Points columns).'
        }
      ]
    },
    {
      id: 'dfs-optimizer-advanced',
      title: 'DFS Optimizer - Advanced',
      icon: Calculator,
      description: 'Advanced features, strategies, and professional techniques',
      content: [
        {
          question: '🎯 Advanced Workflow Overview',
          answer: 'Professional DFS players use multi-session optimization to build diverse lineup portfolios. This involves running the optimizer multiple times with different settings, saving the best lineups from each run to Favorites, and exporting a final curated pool. This advanced guide covers all tabs and features.'
        },
        {
          question: '📊 Players Tab - Advanced Features',
          answer: 'Min/Max Exposure Controls: Set Min Exp to force a player into at least X% of lineups (e.g., 30% min for core plays). Set Max Exp to cap a player (e.g., 40% max to limit chalk exposure). Multi-Position Players: Players like SS/2B appear in both position tabs - selecting them in one selects in all. Position Strategy: For GPP, load heavy on OF (3 spots, most variance). For Cash, focus on high-floor C/1B/2B. Sorting: Use "Value" sort to find efficient plays, "Points" for ceiling plays.'
        },
        {
          question: '🏟️ Team Stacks Tab - Stack Size Strategy',
          answer: 'All Stacks Tab: Select teams here to make them available for ALL stack sizes. Use this for versatile teams. Specific Stack Tabs: Override "All Stacks" by selecting different teams per stack size. Example: 5 Stack → NYY/LAD only, 3 Stack → ATL/SF/CHC. Priority System: Specific tab selections override All Stacks. Test Detection: Click this button to log your exact team selections to console - critical for debugging. Batter Validation: Teams without enough batters are auto-disabled (grayed out) for that stack size.'
        },
        {
          question: '📈 Stack Exposure Tab - Distribution Strategy',
          answer: 'Simple Stacks: Enable one type (e.g., just "4") for focused strategy. Multi-Stack Mix: Enable multiple (e.g., 5, 4, 3) to diversify lineup construction. Min/Max Usage: Set Min Exp to guarantee X% of lineups use that stack (e.g., 4 Stack min 60%). Set Max Exp to cap (e.g., 5 Stack max 20%). Conflict Detection: Watch for red warnings if total Min Exp exceeds 100% - adjust until green "Ready" shows. Complex Stacks: Use 4|2|2 or 3|3|2 for multi-game correlation (advanced GPP strategy).'
        },
        {
          question: '🔢 Team Combinations Tab - Exhaustive Testing',
          answer: 'Purpose: Automatically generate EVERY possible team pairing for a stack pattern. Example: Select 4 teams + "4|2" pattern = 12 combinations (NYY(4)+LAD(2), NYY(4)+ATL(2), etc.). Workflow: Select 3-5 teams → Choose stack pattern → Set lineups per combo (5-10) → Generate. Use Cases: Testing all team pairings, comprehensive GPP coverage, finding optimal team synergies. Warning: Combinations explode fast! 6 teams + 3|3|2 = 120 combos. Start small (3 teams, 4|2 pattern = 6 combos).'
        },
        {
          question: '🧮 Advanced Quant Tab - Financial Modeling',
          answer: 'Master Toggle: Enable "Advanced Quantitative Optimization" to unlock Wall Street-level features. Optimization Strategies: Combined (recommended) = uses multiple techniques. Kelly Criterion = optimal bankroll growth. Risk Parity = equal volatility contribution. Mean-Variance = Markowitz portfolio optimization. Risk Tolerance Slider: <1.0 = conservative (cash games), 1.0 = neutral, >1.0 = aggressive (GPP). Monte Carlo Simulations: 10,000 iterations recommended. Higher = more accurate but slower. GARCH Modeling: Leave at GARCH(1,1) default for time-varying volatility modeling.'
        },
        {
          question: '💾 My Entries Tab - Multi-Session Portfolio',
          answer: 'Building a 150-Lineup Portfolio: Run 1: 4-Stack with top teams → Add top 30 to favorites. Run 2: 5-Stack contrarian → Add top 25. Run 3: 3|3|2 multi-stack → Add top 30. Run 4: Different team selections → Add top 30. Run 5: Final adjustments → Add top 35. Total: 150 diverse lineups from 5 sessions. Run Color Coding: Blue=Run 1, Green=Run 2, Yellow=Run 3, Orange=Run 4, Purple=Run 5+. Sort & Filter: Use filters to isolate specific runs, sort by points to find best lineups. Export: Select lineups to export (or all), click "Export Favorites" for DraftKings CSV.'
        },
        {
          question: '⚙️ Control Panel - Quick Reference',
          answer: 'File Operations: "Load CSV" = player pool, "Load DK Predictions" = custom projections, "Load DK Entries" = pre-filled entry templates. Optimization Settings: Num Lineups (20-500), Min Unique (3-9, higher = more different), Min Salary (45000-49999). Disable Kelly to skip bankroll-based position sizing. Risk Management: Enable for Kelly criterion bet sizing based on bankroll. Bankroll = your total budget. Risk Profile (conservative/medium/aggressive) adjusts recommendation. Actions: "Run Contest Sim" = start optimization. "Save CSV for DK" = download results. "Fill Entries w/ Lineups" = populate DK template file.'
        },
        {
          question: '🎯 Advanced Strategy: Cash Games',
          answer: 'Player Selection: 30-50 players, focus on high floor. Stack Configuration: Use 3-Stack only, select 3-4 top teams. Stack Exposure: Enable only "3" stack type, 0-100% exposure. Settings: 10-20 lineups, Min Unique 5-7. Risk: Conservative risk profile, lower risk tolerance (0.6-0.8). Goal: Consistent, safe lineups that finish in top 50%. Avoid: High variance players, 5-stacks, too many punt plays.'
        },
        {
          question: '🚀 Advanced Strategy: GPP Tournaments',
          answer: 'Player Selection: 50-100+ players, include chalk AND contrarian. Stack Configuration: Use multiple stack sizes (5, 4, 3), select 6-10 teams mixing favorites and underdogs. Stack Exposure: Enable 5 Stack (10-20%), 4 Stack (60-70%), 3 Stack (10-20%). Settings: 100-150 lineups, Min Unique 7-9, enable Advanced Quant. Risk: Aggressive profile (1.2-1.8), Monte Carlo 10K+ iterations. Goal: Unique, leveraged lineups with ceiling to win entire tournament.'
        },
        {
          question: '🔬 Using Advanced Quant Features',
          answer: 'When to Use: Large bankrolls ($1000+), professional play, risk-conscious optimization. GARCH Volatility: Models time-varying player variance. Use GARCH(1,1) default. Lookback 100 days balances stability and responsiveness. Copula Modeling: Gaussian = general use. Gumbel = offensive stacks (succeed together). Clayton = defensive correlations (fail together). Kelly Criterion: Max Kelly 25% = quarter Kelly (recommended). Expected Win Rate: 50% for cash games, 20% for GPP top 20%, 10% for top 10%. VaR Confidence: 95% shows worst-case loss in 95% of scenarios. Performance Impact: Adds 30-60 seconds to optimization. Worth it for serious players.'
        },
        {
          question: '⚡ Team Combinations - Power User Feature',
          answer: 'Best Use Case: Testing every team pairing systematically. Example Setup: Select 4 teams (NYY, LAD, ATL, SF). Choose pattern "4|2". Set 5 lineups per combo. Result: 12 unique combinations × 5 lineups each = 60 total lineups covering every NYY/LAD/ATL/SF pairing. When to Use: You want comprehensive coverage of specific teams. You are testing which team pairings work best. You are entering large-field GPPs (need 100+ diverse lineups). Combination Math: Pattern "4|2" with N teams = C(N,2) × 2! combinations. 3 teams = 6 combos, 4 teams = 12 combos, 5 teams = 20 combos, 6 teams = 30 combos.'
        },
        {
          question: '📊 Exposure Management - Professional Technique',
          answer: 'Player Exposure: Core Play (30-50% min): Your highest confidence play. Balanced (0-100%): Most players, let optimizer decide. Capped Chalk (0-40% max): Limit overexposure to popular players. Sprinkle (0-15% max): Low-owned contrarian plays. Team Exposure: Configure in Team Stacks tab. High-scoring teams should have higher exposure. Bad weather/low total teams should have max caps. Stack Type Exposure: Cash: 100% in 3-stacks for safety. Balanced GPP: 15% five-stacks, 60% four-stacks, 25% three-stacks. Aggressive GPP: 40% five-stacks, 40% 5|3 multi-stacks, 20% four-stacks.'
        },
        {
          question: '🎲 Multi-Session Workflow (150 Lineup Build)',
          answer: 'Session 1 (Conservative): 4-Stack, top 3 teams, 100 lineups → Add top 30 to Favorites. Session 2 (Aggressive): 5-Stack, contrarian teams, 100 lineups → Add top 25 to Favorites. Session 3 (Multi-Stack): 4|2|2 pattern, 6 teams, 80 lineups → Add top 30 to Favorites. Session 4 (Balanced): 3-Stack, mid-tier teams, 100 lineups → Add top 30 to Favorites. Session 5 (Final): Mixed settings, 100 lineups → Add top 35 to Favorites. Result: 150 diverse lineups from 5 different strategies in My Entries tab. Export all 150 for maximum tournament coverage.'
        },
        {
          question: '⚙️ Optimization Settings Explained',
          answer: 'Number of Lineups: Start with 20-50, professionals use 100-150. More = better coverage but slower. Min Unique: How many players must be different between lineups. 3 = very similar, 9 = very different. Use 7-9 for GPP, 3-5 for cash. Min Salary: Minimum total salary ($45K-$49.9K). Higher forces optimizer to use expensive players. Default $45K works well. Disable Kelly: Unchecking uses bankroll-based position sizing. Keep checked for equal weighting (simpler). Sorting: Points = highest projected first. Value = most efficient first. Salary = most expensive first. Use Points for main results view.'
        },
        {
          question: '🔍 Validation & Warnings',
          answer: 'Yellow Warnings (Can Proceed): "Only 25 players selected" - you can continue but might want more. "Total min exposure 110%" - conflicting constraints, adjust to continue. Red Errors (Must Fix): "No stack types selected" - must enable at least one in Stack Exposure. "Pattern requires 3 teams, only 2 selected" - select more teams in Team Combinations. Green Ready: "Ready to optimize" - all validations passed, good to go! Status Indicators: Pulsing cyan dot = feature enabled. Gray text = disabled/inactive. Cyan text = active/selected.'
        },
        {
          question: '📤 Export Formats & DraftKings Integration',
          answer: 'Save CSV for DK: Standard DraftKings upload format with player IDs. Columns: P, P, C, 1B, 2B, 3B, SS, OF, OF, OF (10 players). Fill Entries w/ Lineups: If you loaded a DK entries template, this fills it with optimized lineups. Export Favorites: Exports lineups from My Entries tab (curated from multiple sessions). File Format: CSV compatible with DraftKings bulk upload. Go to DK contest → "Upload" → Select your exported file → Lineups auto-populate!'
        }
      ]
    },
    {
      id: 'prop-betting',
      title: 'Prop Betting & Parlays',
      icon: Target,
      description: 'Find edges and build winning parlays',
      content: [
        {
          question: 'Understanding Edge',
          answer: 'Edge is the percentage difference between the sportsbook line and our projection. A +8% edge means we project the player will exceed the line by 8%. Higher edge = more confident bet. Look for edges above 5% for best results.'
        },
        {
          question: 'Building a Parlay',
          answer: 'In the Prop Betting Center, click "Add to Parlay" on props you like. The betting slip will show combined odds and potential payout. Aim for 2-4 leg parlays for the best balance of odds and hit probability. Avoid massive 10+ leg parlays.'
        },
        {
          question: 'What is Kelly Criterion?',
          answer: 'Kelly Criterion is a mathematical formula that calculates optimal bet sizing based on your edge and bankroll. It prevents over-betting and maximizes long-term growth. UrSim shows recommended stake sizes using Kelly. Conservative bettors should use fractional Kelly (25-50%).'
        },
        {
          question: 'How to read confidence scores',
          answer: 'Confidence (0-100%) indicates how certain our model is about the projection. 90%+ = very confident, 70-89% = moderately confident, <70% = less reliable. Higher confidence props are better for parlays. Lower confidence can offer value in single bets.'
        },
        {
          question: 'Hit Rate vs Edge',
          answer: 'Hit Rate shows historical success percentage for similar props. A 68% hit rate means the player exceeded this line in 68% of past games. Combine high hit rate with positive edge for the best opportunities. Beware of high hit rate but negative edge (line has moved).'
        },
        {
          question: 'Correlated props',
          answer: 'Some props correlate (QB passing yards + WR receiving yards from same team). While parlaying correlated props can be valuable, sportsbooks often adjust odds to account for this. Use correlation strategically but be aware of reduced payouts.'
        }
      ]
    },
    {
      id: 'games-hub',
      title: 'Using Games Hub',
      icon: BarChart3,
      description: 'Research games, matchups, and market trends',
      content: [
        {
          question: 'What is Games Hub?',
          answer: 'Games Hub is your research center. View all games, matchups, team stats, defensive rankings, weather conditions, and prop markets in one place. Use filters to find specific games or teams.'
        },
        {
          question: 'Analyzing matchups',
          answer: 'Click any game to see detailed matchup analysis. Review offensive vs defensive rankings, recent form, injuries, and situational factors. Games with high totals and mismatches offer the best DFS stacking opportunities.'
        },
        {
          question: 'Weather and injuries',
          answer: 'Weather icons indicate conditions affecting gameplay (wind, rain, snow). Injury indicators show key players out or questionable. Always check these before finalizing lineups. High wind reduces passing games, benefiting running backs.'
        },
        {
          question: 'Finding prop betting opportunities',
          answer: 'Games Hub shows prop market overview for each game. Look for games with high totals, favorable matchups, and positive edges across multiple props. These are prime targets for correlated parlays.'
        }
      ]
    },
    {
      id: 'advanced-strategies',
      title: 'Advanced Strategies',
      icon: Lightbulb,
      description: 'Pro tips and advanced techniques',
      content: [
        {
          question: 'Ownership projections',
          answer: 'In tournaments, ownership matters. Fading (avoiding) chalk (high-owned) players and targeting low-owned value creates differentiation. Use the Results tab to review projected ownership before submitting lineups.'
        },
        {
          question: 'Late swap strategy',
          answer: 'DraftKings allows lineup changes until each game starts. Monitor injury news and adjust players in late games. This is especially valuable in NFL Sunday-Monday swaps.'
        },
        {
          question: 'Bankroll management',
          answer: 'Never risk more than 5-10% of your bankroll on a single slate. Diversify across multiple contests. Use Kelly Criterion for prop bets. Track ROI monthly and adjust strategy based on results.'
        },
        {
          question: 'Multi-entry strategy',
          answer: 'In large-field tournaments, enter multiple lineups (20-150) to maximize exposure to your player pool while maintaining uniqueness. Use high uniqueness settings (7-9 players different between lineups).'
        },
        {
          question: 'Leverage in tournaments',
          answer: 'Leverage = ownership adjusted value. A 25% owned player at 3x value has less leverage than a 5% owned player at 2.5x value. Target low-owned, high-upside players in tournaments for maximum leverage.'
        },
        {
          question: 'Game theory optimal (GTO)',
          answer: 'In large fields, optimal strategy balances chalk, value, and contrarian plays. Dont be 100% contrarian or 100% chalk. Mix popular and unpopular players for optimal tournament equity.'
        }
      ]
    },
    {
      id: 'faq',
      title: 'FAQ',
      icon: HelpCircle,
      description: 'Frequently asked questions',
      content: [
        {
          question: 'How accurate are the projections?',
          answer: 'Projections are based on advanced algorithms, historical data, and real-time updates. While no projection is perfect, our models have been backtested extensively. Always use projections as a guide alongside your own research.'
        },
        {
          question: 'Can I import custom projections?',
          answer: 'Yes! In the Players & Projections tab, you can manually edit projections or upload your own CSV with custom values. This allows you to incorporate your own research and edge.'
        },
        {
          question: 'What file format for CSV upload?',
          answer: 'Use the standard DraftKings or FanDuel CSV export format. Required columns: Name, Position, Team, Salary, and FPPG (projected points). Optional: Ownership, Ceiling, Floor.'
        },
        {
          question: 'How many lineups should I generate?',
          answer: 'For single-entry contests: 1 lineup. For 3-entry: 3 lineups with high uniqueness. For 20-entry tournaments: 20 lineups with 7-8 unique players. For max-entry: 150 lineups with maximum diversity settings.'
        },
        {
          question: 'Is there mobile support?',
          answer: 'Yes! UrSim is fully responsive. Access all features on mobile devices. Use mobile for quick lineup checks and late swaps. For detailed optimization, desktop is recommended.'
        },
        {
          question: 'How do I contact support?',
          answer: 'Click the help icon in the top navigation bar or visit our support center. We offer live chat, email support, and comprehensive documentation for all features.'
        }
      ]
    },
    {
      id: 'glossary',
      title: 'Glossary',
      icon: BookOpen,
      description: 'Key terms and definitions',
      content: [
        {
          question: 'DFS (Daily Fantasy Sports)',
          answer: 'Contest format where you draft a new team for each slate. Unlike season-long fantasy, DFS resets every day/week.'
        },
        {
          question: 'GPP (Guaranteed Prize Pool)',
          answer: 'Large tournaments with guaranteed payouts. Top-heavy payout structures reward unique, high-ceiling lineups.'
        },
        {
          question: 'Cash Game',
          answer: '50/50, double-ups, or head-to-heads where ~50% of entries win. Requires safe, high-floor lineups.'
        },
        {
          question: 'Chalk',
          answer: 'Highly-owned players. In tournaments, chalk players need to bust for contrarian lineups to win.'
        },
        {
          question: 'Ceiling',
          answer: 'Maximum realistic points a player could score. Important for tournaments where you need upside.'
        },
        {
          question: 'Floor',
          answer: 'Minimum likely points a player will score. Important for cash games where consistency matters.'
        },
        {
          question: 'Value/Points per Dollar',
          answer: 'Projected points divided by salary (multiplied by 1000). Higher value = more efficient spend.'
        },
        {
          question: 'Exposure',
          answer: 'Percentage of lineups containing a specific player. Manage exposure to limit risk from a single player.'
        },
        {
          question: 'Stacking',
          answer: 'Using multiple players from the same team (QB+WR, Pitcher+Hitters). Increases correlation and upside.'
        },
        {
          question: 'Correlation',
          answer: 'When one players success increases anothers likelihood of success. QBs are positively correlated with their receivers.'
        },
        {
          question: 'ROI (Return on Investment)',
          answer: 'Profit divided by entry fees, expressed as percentage. 20% ROI = you make $1.20 for every $1 invested.'
        },
        {
          question: 'Ownership Leverage',
          answer: 'Gaining advantage through low-owned players. If a 5% owned player scores 40 points, you gain significant edge over the field.'
        }
      ]
    }
  ];

  const filteredTutorials = tutorials.map(tutorial => ({
    ...tutorial,
    content: tutorial.content.filter(item =>
      item.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.answer.toLowerCase().includes(searchQuery.toLowerCase())
    )
  })).filter(tutorial => tutorial.content.length > 0);

  return (
    <div className="h-full overflow-auto p-6">
      {/* Main Card Container */}
      <div className="bg-slate-800 backdrop-blur-sm rounded-2xl border border-cyan-500/20 shadow-2xl relative overflow-hidden min-h-full flex flex-col">

      {/* Content */}
      <div className="relative z-10 flex flex-col h-full p-6">
        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-cyan-500/30">
              <BookOpen className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">
                How to Use UrSim
              </h1>
              <p className="text-white">Complete guide to mastering DFS and prop betting</p>
            </div>
          </div>

          {/* Search */}
          <div className="relative mt-4">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-300" />
            <Input
              type="text"
              placeholder="Search tutorials, tips, and FAQs..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 bg-black/60 border-cyan-500/20 text-white placeholder:text-slate-300"
            />
          </div>
        </div>

        {/* Tutorials */}
        <Card className="flex-1 overflow-hidden bg-slate-800 backdrop-blur-sm border-cyan-500/20">
          <Tabs defaultValue="getting-started" className="h-full flex flex-col">
            <TabsList className="bg-slate-700 border-b border-cyan-500/20 justify-start overflow-x-auto flex-wrap h-auto">
              {tutorials.map((tutorial) => {
                const Icon = tutorial.icon;
                return (
                  <TabsTrigger 
                    key={tutorial.id} 
                    value={tutorial.id}
                    className="data-[state=active]:bg-cyan-500/10 data-[state=active]:text-cyan-400"
                  >
                    <Icon className="w-4 h-4 mr-2" />
                    {tutorial.title}
                  </TabsTrigger>
                );
              })}
            </TabsList>

            <div className="flex-1 overflow-auto p-6">
              {(searchQuery ? filteredTutorials : tutorials).map((tutorial) => {
                const Icon = tutorial.icon;
                return (
                  <TabsContent key={tutorial.id} value={tutorial.id} className="mt-0">
                    <div className="mb-6">
                      <div className="flex items-center gap-3 mb-2">
                        <Icon className="w-6 h-6 text-cyan-400" />
                        <h2 className="text-2xl font-bold text-white">{tutorial.title}</h2>
                      </div>
                      <p className="text-white">{tutorial.description}</p>
                    </div>

                    <Accordion type="multiple" className="space-y-3">
                      {tutorial.content.map((item, index) => (
                        <AccordionItem 
                          key={index} 
                          value={`item-${index}`}
                          className="bg-slate-700/60 border border-cyan-500/10 rounded-lg px-4"
                        >
                          <AccordionTrigger className="text-left hover:text-cyan-400 transition-colors">
                            <span className="font-medium text-white">{item.question}</span>
                          </AccordionTrigger>
                          <AccordionContent className="text-white leading-relaxed">
                            {item.answer}
                          </AccordionContent>
                        </AccordionItem>
                      ))}
                    </Accordion>

                    {tutorial.content.length === 0 && searchQuery && (
                      <p className="text-white text-center py-8">
                        No results found for "{searchQuery}"
                      </p>
                    )}
                  </TabsContent>
                );
              })}
            </div>
          </Tabs>
        </Card>

        {/* Quick Tips Footer */}
        <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3">
          <Card className="bg-cyan-500/5 border-cyan-500/20 p-3">
            <div className="flex items-start gap-2">
              <Zap className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-cyan-400">Pro Tip</p>
                <p className="text-xs text-white">Start with small stakes while learning. Increase as you gain confidence.</p>
              </div>
            </div>
          </Card>
          <Card className="bg-blue-500/5 border-blue-500/20 p-3">
            <div className="flex items-start gap-2">
              <Shield className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-blue-400">Risk Management</p>
                <p className="text-xs text-white">Never risk more than 5-10% of bankroll on a single slate.</p>
              </div>
            </div>
          </Card>
          <Card className="bg-purple-500/5 border-purple-500/20 p-3">
            <div className="flex items-start gap-2">
              <TrendingUp className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-purple-400">Track Results</p>
                <p className="text-xs text-white">Keep detailed records of all bets and lineups to improve over time.</p>
              </div>
            </div>
          </Card>
        </div>
        </div>
      </div>
    </div>
  );
}

