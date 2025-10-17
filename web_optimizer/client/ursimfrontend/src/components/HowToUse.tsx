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
      id: 'dfs-lineups',
      title: 'Building DFS Lineups',
      icon: Trophy,
      description: 'Master the art of DFS lineup optimization',
      content: [
        {
          question: 'Step 1: Upload Your Player Pool',
          answer: 'Go to DFS Lineup Optimizer > Players & Projections tab. Click "Upload CSV" and select your player pool file from DraftKings or FanDuel. The system will automatically parse and display all players with their salaries and projections.'
        },
        {
          question: 'Step 2: Set Your Strategy',
          answer: 'Navigate to the Strategy & Settings tab. Configure team stacking preferences (2-stack, 3-stack, 4-stack), set exposure limits for players and teams, and define correlation rules. For tournaments, aggressive stacking is recommended. For cash games, use balanced lineups.'
        },
        {
          question: 'Step 3: Configure Optimization',
          answer: 'In the Optimization Control tab, set the number of lineups you want to generate (1-150), adjust salary constraints, and set uniqueness (how different lineups should be from each other). Advanced users can enable Monte Carlo simulations for variance modeling.'
        },
        {
          question: 'Step 4: Run & Export',
          answer: 'Click "Start Optimization" and watch real-time progress. Once complete, view your lineups in the Results & Export tab. You can sort by projection, analyze stack diversity, and export directly to DraftKings or FanDuel format.'
        },
        {
          question: 'What are Lock and Exclude controls?',
          answer: 'Lock forces a player into ALL lineups - use this for high-conviction plays. Exclude removes a player from consideration - use this for injured players or unfavorable matchups. Be strategic with locks as they reduce lineup diversity.'
        },
        {
          question: 'How do I use stacking effectively?',
          answer: 'Stacking correlates players from the same team (QB + WR in NFL, Pitcher + Batter in MLB). In tournaments, 3-4 player stacks from high-scoring games increase upside. In cash games, use conservative 2-stacks. Always stack the team expected to score the most points.'
        },
        {
          question: 'What is Monte Carlo simulation?',
          answer: 'Monte Carlo runs thousands of iterations with randomized projections based on variance. This accounts for uncertainty and helps generate more robust lineups. Set iterations between 100-1000. Higher iterations take longer but produce more reliable results.'
        },
        {
          question: 'Cash vs GPP optimization',
          answer: 'Cash Games (50/50, Double-ups): Use lower ownership, balanced lineups, and conservative stacking. Focus on safe, high-floor players. GPP (Tournaments): Use contrarian plays, aggressive stacking, and high-ceiling players. Accept more risk for higher upside.'
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
              <p className="text-slate-200">Complete guide to mastering DFS and prop betting</p>
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
                      <p className="text-slate-200">{tutorial.description}</p>
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
                          <AccordionContent className="text-slate-200 leading-relaxed">
                            {item.answer}
                          </AccordionContent>
                        </AccordionItem>
                      ))}
                    </Accordion>

                    {tutorial.content.length === 0 && searchQuery && (
                      <p className="text-slate-200 text-center py-8">
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
                <p className="text-xs text-slate-200">Start with small stakes while learning. Increase as you gain confidence.</p>
              </div>
            </div>
          </Card>
          <Card className="bg-blue-500/5 border-blue-500/20 p-3">
            <div className="flex items-start gap-2">
              <Shield className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-blue-400">Risk Management</p>
                <p className="text-xs text-slate-200">Never risk more than 5-10% of bankroll on a single slate.</p>
              </div>
            </div>
          </Card>
          <Card className="bg-purple-500/5 border-purple-500/20 p-3">
            <div className="flex items-start gap-2">
              <TrendingUp className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-purple-400">Track Results</p>
                <p className="text-xs text-slate-200">Keep detailed records of all bets and lineups to improve over time.</p>
              </div>
            </div>
          </Card>
        </div>
      </div>
      </div>
    </div>
  );
}

