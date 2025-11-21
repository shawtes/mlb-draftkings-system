import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from './ui/accordion';
import { Input } from './ui/input';
import { 
  BookOpen, 
  Search,
  Zap,
  Target,
  TrendingUp,
  AlertCircle,
  ArrowRight,
  CheckCircle2
} from 'lucide-react';

export default function HowToUse() {
  const [searchQuery, setSearchQuery] = useState('');

  const tutorials = [
    {
      id: 'first-steps',
      title: 'First Steps',
      icon: Zap,
      description: 'What is this tool and why should you use it?',
      content: [
        {
          question: 'What is the DFS Optimizer?',
          answer: 'The DFS Optimizer is a powerful tool that automatically builds DraftKings lineups for you. Instead of manually picking players and trying to fit them under the salary cap, the optimizer uses advanced algorithms to find the best combinations of players that maximize projected points while staying within budget. Think of it as having a professional lineup builder working for you 24/7.'
        },
        {
          question: 'Why use an optimizer instead of picking manually?',
          answer: 'Manual lineup building is slow, error-prone, and limits your options. The optimizer can test millions of player combinations in seconds, finding lineups you would never think of. It ensures you never accidentally go over the salary cap, always meet position requirements, and helps you build multiple diverse lineups quickly. Most importantly, it helps you find value - players who are projected to score more points than their salary suggests they should.'
        },
        {
          question: 'Do I need to be an expert to use this?',
          answer: 'Absolutely not! This guide is designed for complete beginners. You just need to know how to upload a player file from DraftKings and understand basic concepts like salary cap and positions. The optimizer does all the heavy lifting. As you get more comfortable, you can explore advanced features, but you can start winning with just the basics.'
        },
        {
          question: 'What makes a "smart pick"?',
          answer: 'A smart pick is a player who offers good value - meaning their projected points are high relative to their salary cost. For example, a player costing $3,000 who projects for 15 points is better value than a $9,000 player projecting for 20 points. The optimizer automatically finds these value plays and builds lineups around them. Smart picks also consider matchups, recent form, and game situations that favor the player.'
        }
      ]
    },
    {
      id: 'getting-first-lineup',
      title: 'Getting Your First Lineup',
      icon: ArrowRight,
      description: 'Step-by-step walkthrough to create your first optimized lineup',
      content: [
        {
          question: 'Step 1: Get Your Player File from DraftKings',
          answer: 'Go to DraftKings.com and navigate to the contest you want to enter. Look for an "Export" or "Download" button that gives you a CSV file with all available players, their salaries, and projected points. This file contains everything the optimizer needs. Save it somewhere easy to find on your computer.'
        },
        {
          question: 'Step 2: Load the File into the Optimizer',
          answer: 'In the DFS Optimizer, look for the "Load CSV" button (usually in the Control Panel on the right side). Click it and select the file you just downloaded from DraftKings. The system will automatically read all the players and their information. You should see a message confirming how many players were loaded.'
        },
        {
          question: 'Step 3: Select Your Player Pool',
          answer: 'Go to the "Players" tab. You\'ll see all available players organized by position. For your first lineup, start simple: select 40-60 players you think have good potential. Don\'t worry about being perfect - you can always adjust. Use the position filters to view specific positions. Tip: Look for players with high "Value" scores - these are the most efficient picks.'
        },
        {
          question: 'Step 4: Set Up Basic Stacking (Optional but Recommended)',
          answer: 'Stacking means using multiple players from the same team. This is important because when a team scores well, multiple players benefit together. Go to the "Stack Exposure" tab and check the box next to "4" to enable 4-player stacks. This tells the optimizer to build lineups with 4 players from the same team. For beginners, this is the sweet spot - good correlation without being too risky.'
        },
        {
          question: 'Step 5: Choose Which Teams to Stack',
          answer: 'Go to the "Team Stacks" tab and click on the "4 Stack" section. Select 2-3 teams that you think will score the most runs/points. Look for teams playing in games with high projected totals (shown in green). These teams are more likely to have multiple players score well together. Don\'t overthink this - pick the teams you feel most confident about.'
        },
        {
          question: 'Step 6: Set Number of Lineups',
          answer: 'In the Control Panel, find "Number of Lineups" under Optimization Settings. For your very first run, start with 10-20 lineups. This is enough to see how the optimizer works without being overwhelming. Once you\'re comfortable, you can increase this to 50-100 for tournaments.'
        },
        {
          question: 'Step 7: Run the Optimizer',
          answer: 'Click the "Run Contest Sim" button in the Control Panel. The optimizer will start working - this usually takes 10-30 seconds. You\'ll see progress updates. When it\'s done, you\'ll see your generated lineups with projected points and salary usage. All lineups automatically meet DraftKings requirements (salary cap, position limits).'
        },
        {
          question: 'Step 8: Review and Export',
          answer: 'Look through your generated lineups. Check that they make sense - do you recognize the players? Are the teams you wanted to stack represented? If something looks off, you can adjust your player selections and run again. When you\'re happy, click "Save CSV for DK" to download a file you can upload directly to DraftKings.'
        },
        {
          question: 'Step 9: Upload to DraftKings',
          answer: 'Go back to DraftKings.com, navigate to your contest, and look for an "Upload Lineups" or "Import" button. Select the CSV file you just downloaded. Your lineups will be automatically entered into the contest. That\'s it - you\'re done!'
        }
      ]
    },
    {
      id: 'finding-smart-picks',
      title: 'Finding Smart Picks',
      icon: Target,
      description: 'How to identify value players and make intelligent selections',
      content: [
        {
          question: 'Understanding Value: The Key to Winning',
          answer: 'Value = Projected Points ÷ (Salary ÷ 1000). A player with a value of 5.0 means they project for 5 points per $1,000 of salary. Higher value = better pick. The optimizer automatically calculates this, but you should understand it too. Look for players with value scores above 4.5 - these are your smart picks. Players under 4.0 are usually overpriced.'
        },
        {
          question: 'What Makes a Player a Smart Pick?',
          answer: 'Smart picks have three characteristics: 1) Good value (high points per dollar), 2) Favorable matchup (playing against weak defense/pitching), 3) Recent form (playing well lately). The optimizer considers all of this, but you should too. A $4,000 player in a great matchup who\'s been hot is often smarter than a $8,000 star in a tough spot.'
        },
        {
          question: 'How to Use the Player Table',
          answer: 'In the Players tab, you\'ll see columns for Salary, Proj (projected points), and Value. Sort by "Value" to see the most efficient players first. These are your smart picks. Don\'t just pick the highest projected players - they might be too expensive. Instead, look for players with good projections AND good value. The optimizer will automatically favor these players.'
        },
        {
          question: 'The 80/20 Rule for Player Selection',
          answer: 'You don\'t need to be perfect. Select about 80% of players you feel confident about, and let the optimizer fill in the other 20% with value plays you might have missed. This gives you control while still benefiting from the optimizer\'s ability to find hidden gems. Don\'t try to hand-pick every single player - that defeats the purpose of using an optimizer.'
        },
        {
          question: 'When to Trust Projections vs Your Gut',
          answer: 'Use projections as your foundation, but don\'t ignore obvious factors. If a star player is in a terrible matchup or just got injured, don\'t include them even if the projection looks good. Conversely, if you know a cheap player has a great opportunity (injury to starter, weather advantage, etc.), include them even if their projection seems low. The optimizer will find ways to use them if they\'re truly valuable.'
        },
        {
          question: 'Avoiding Chalk Traps',
          answer: 'Chalk = highly-owned players everyone picks. While chalk players can be good, they\'re often overpriced and don\'t differentiate your lineups. The optimizer helps by finding lower-owned value plays. Don\'t feel you need to include every popular player. Sometimes fading (avoiding) chalk and finding your own value is the smarter play, especially in tournaments.'
        }
      ]
    },
    {
      id: 'optimizer-settings',
      title: 'Optimizer Settings',
      icon: TrendingUp,
      description: 'What settings matter for new users and how to use them',
      content: [
        {
          question: 'Number of Lineups: Start Small, Scale Up',
          answer: 'For your first few runs, use 10-20 lineups. This lets you see how the optimizer works and review each lineup. Once comfortable, increase to 50-100 for tournaments. More lineups = more diversity and better coverage of your player pool, but takes longer to generate. Start small, then scale up as you gain confidence.'
        },
        {
          question: 'Min Unique: How Different Should Lineups Be?',
          answer: 'Min Unique controls how many players must be different between lineups. Higher = more diverse lineups. For beginners, leave this at the default (usually 5-7). This ensures your lineups aren\'t too similar. If all your lineups look the same, increase this number. If they\'re too different and you want more consistency, decrease it.'
        },
        {
          question: 'Stack Exposure: Simple is Better',
          answer: 'For new users, keep stacking simple. Enable just one stack type (like "4 Stack") and leave Min/Max Exposure at defaults (0-100%). This tells the optimizer "use 4-player stacks when it makes sense" without forcing it. As you get more advanced, you can control exactly what percentage of lineups use each stack type.'
        },
        {
          question: 'Min Salary: Don\'t Overthink This',
          answer: 'Min Salary is the minimum total salary your lineups must use. The default (usually $45,000-$47,000) works well for most situations. Only change this if you have a specific strategy. Higher min salary forces the optimizer to use more expensive players. Lower allows more budget flexibility. For beginners, leave it at default.'
        },
        {
          question: 'Player Exposure: Let the Optimizer Decide (At First)',
          answer: 'Player exposure controls how often each player appears across your lineups. For beginners, leave Min Exp and Max Exp at defaults (0% and 100%) for all players. This lets the optimizer decide based on value. Once you understand the tool better, you can force your favorite players into more lineups (higher Min Exp) or limit players you\'re unsure about (lower Max Exp).'
        },
        {
          question: 'Advanced Settings: Skip for Now',
          answer: 'The optimizer has advanced features like Kelly Criterion, Monte Carlo simulations, and complex risk models. These are powerful but not necessary for beginners. Focus on mastering the basics first: player selection, stacking, and number of lineups. Come back to advanced features once you\'re comfortable with the fundamentals.'
        }
      ]
    },
    {
      id: 'maximum-value',
      title: 'Getting Maximum Value',
      icon: TrendingUp,
      description: 'Strategies for realistic success and maximizing your edge',
      content: [
        {
          question: 'The Realistic Path to Success',
          answer: 'Success in DFS doesn\'t mean winning every contest. Realistic goals: finish in the top 20-30% consistently, cash in 40-50% of contests, and occasionally hit big in tournaments. The optimizer helps you achieve this by building mathematically sound lineups. Don\'t expect to win every time - even the best players lose more than they win. Focus on long-term profitability.'
        },
        {
          question: 'Value Over Projections: The Winning Formula',
          answer: 'The optimizer maximizes value, not just raw points. A lineup with 120 projected points using $50,000 salary is better than 125 points using $49,000. Why? Because you\'re getting more points per dollar. The optimizer automatically finds this balance. Trust it - it\'s doing complex math you can\'t do manually. Your job is to give it a good player pool to work with.'
        },
        {
          question: 'Diversity is Your Friend',
          answer: 'Don\'t put all your eggs in one basket. Generate multiple lineups (20-50 for tournaments) with different player combinations. This spreads your risk. If one player busts, it doesn\'t kill all your lineups. The optimizer\'s "Min Unique" setting helps ensure diversity. More diverse lineups = more chances to hit different game scripts and outcomes.'
        },
        {
          question: 'Stacking Strategy: Correlation Wins',
          answer: 'Stacking (using multiple players from the same team) is crucial because it creates correlation - when one player scores well, others from that team often do too. A 4-player stack that goes off can carry your entire lineup. The optimizer automatically builds stacks when you enable them. For maximum value, stack teams in high-scoring games (look for high totals).'
        },
        {
          question: 'When to Use Multiple Stack Sizes',
          answer: 'Once comfortable, try mixing stack sizes. Use mostly 4-player stacks (safe, good correlation) but also some 5-player stacks (higher risk, higher reward) and 3-player stacks (safer, less correlation). The optimizer can handle this - just enable multiple stack types in Stack Exposure. This gives you both safe and aggressive lineups in the same pool.'
        },
        {
          question: 'Review and Learn: Track What Works',
          answer: 'After each contest, review which lineups performed best. What did they have in common? Which players worked? Which stacks hit? Use this information to improve your player selection next time. The optimizer gives you the tools, but learning from results makes you better. Keep notes on what strategies work for you.'
        },
        {
          question: 'Bankroll Management: The Unsexy Secret',
          answer: 'Even with the best optimizer, you need proper bankroll management. Never risk more than 5-10% of your bankroll on a single slate. If you have $100, don\'t enter $50 worth of contests. Spread it across multiple contests and slates. The optimizer helps you build good lineups, but managing your money ensures you can keep playing and learning.'
        }
      ]
    },
    {
      id: 'common-mistakes',
      title: 'Common Mistakes',
      icon: AlertCircle,
      description: 'What to avoid as a beginner',
      content: [
        {
          question: 'Mistake 1: Selecting Too Few Players',
          answer: 'If you only select 20-25 players, the optimizer has very few options and your lineups will look very similar. Select at least 40-60 players to give the optimizer flexibility. More players = more diverse lineups. Don\'t worry about including "bad" players - the optimizer won\'t use them unless they provide value.'
        },
        {
          question: 'Mistake 2: Not Enabling Any Stack Types',
          answer: 'Stacking is essential for correlation and upside. If you don\'t enable any stack types in Stack Exposure, your lineups will be random collections of players with no correlation. Always enable at least one stack type (start with "4 Stack"). This is one of the most common beginner mistakes.'
        },
        {
          question: 'Mistake 3: Overthinking Everything',
          answer: 'Beginners often try to control every aspect - setting exposure limits on every player, using complex stack patterns, tweaking every setting. Start simple. Select players, enable one stack type, set number of lineups, and run. You can get fancy later. Simple often works better than complicated, especially when you\'re learning.'
        },
        {
          question: 'Mistake 4: Ignoring Value Scores',
          answer: 'Don\'t just pick players with high projected points. A $9,000 player projecting 25 points might be worse value than a $4,000 player projecting 18 points. Always check the Value column. The optimizer automatically favors value, but if you manually exclude too many value plays, you\'re hurting yourself.'
        },
        {
          question: 'Mistake 5: Generating Too Many Lineups Too Soon',
          answer: 'Starting with 150 lineups on your first run is overwhelming and you won\'t learn anything. Start with 10-20, review them, understand what the optimizer did, then gradually increase. Quality review of fewer lineups beats blindly generating hundreds.'
        },
        {
          question: 'Mistake 6: Not Checking Team Stacks',
          answer: 'If you enable stacking but don\'t select any teams in the Team Stacks tab, the optimizer can\'t build proper stacks. Always verify you\'ve selected teams in the Team Stacks tab that match your stack type. This is a simple step that beginners often skip.'
        },
        {
          question: 'Mistake 7: Expecting Immediate Success',
          answer: 'DFS is a skill that takes time to develop. Even with the best optimizer, you won\'t win every contest. Some slates will be losses. That\'s normal. Focus on building good habits, learning from results, and improving over time. The optimizer gives you an edge, but it\'s not a guarantee of instant profits.'
        },
        {
          question: 'Mistake 8: Not Exporting Correctly',
          answer: 'After generating lineups, make sure you click "Save CSV for DK" (not just any export button). This creates a file in the exact format DraftKings needs. If you try to upload the wrong format, it won\'t work. Double-check the file downloads before trying to upload to DraftKings.'
        }
      ]
    },
    {
      id: 'next-steps',
      title: 'Next Steps',
      icon: CheckCircle2,
      description: 'How to progress after your first success',
      content: [
        {
          question: 'After Your First Successful Run',
          answer: 'Congratulations! You\'ve created your first optimized lineups. Now what? Review what worked. Which players performed well? Which stacks hit? Use this knowledge to refine your player selection process. Try different team combinations. Experiment with different numbers of lineups. The more you use the optimizer, the better you\'ll get at giving it the right inputs.'
        },
        {
          question: 'Expanding Your Strategy',
          answer: 'Once comfortable with basics, try: 1) Multiple stack sizes (enable 5, 4, and 3 stacks together), 2) Different team combinations (try contrarian teams, not just favorites), 3) Player exposure limits (force your favorite players into more lineups), 4) More lineups (scale up to 50-100 for tournaments). Each new technique adds another tool to your arsenal.'
        },
        {
          question: 'Learning from Results',
          answer: 'Track your results. Which types of lineups cash? Which stack patterns work? What player profiles succeed? The optimizer gives you data - use it. After each slate, spend 10 minutes reviewing what worked and what didn\'t. This feedback loop is how you improve. The optimizer does the math, but you need to learn the game theory.'
        },
        {
          question: 'When to Explore Advanced Features',
          answer: 'Once you\'re consistently building good lineups and understand the basics, explore advanced features: Team Combinations tab (systematic team pairing), Advanced Quant settings (risk modeling), Multi-session workflows (building large portfolios). But don\'t rush - master fundamentals first. Advanced features are powerful but won\'t help if you don\'t understand the basics.'
        },
        {
          question: 'Building a Routine',
          answer: 'Develop a consistent process: 1) Research games and matchups, 2) Load player file, 3) Select player pool based on value and matchups, 4) Set up stacking strategy, 5) Generate lineups, 6) Review and adjust if needed, 7) Export and upload. Having a routine makes you faster and more consistent. The optimizer handles the hard part - you just need a good process.'
        },
        {
          question: 'Staying Realistic',
          answer: 'Remember: even professional DFS players have losing slates. The optimizer gives you an edge, but variance is real. Focus on making good decisions consistently, not on winning every single contest. Long-term profitability comes from good process, not short-term results. Trust the optimizer, learn from experience, and stay disciplined with bankroll management.'
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
    <div className="flex h-full w-full flex-col overflow-hidden">
      <div className="flex h-full w-full items-center justify-center px-6 pt-8 pb-8">
        <div className="w-full max-w-7xl mx-auto flex h-full flex-col gap-8 pb-8">
          {/* Header */}
          <div className="w-full">
            <div className="flex items-center gap-4 mb-4">
              <div className="w-14 h-14 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-cyan-500/30">
                <BookOpen className="w-7 h-7 text-white" />
              </div>
              <div>
                <h1 className="text-4xl font-bold text-white">
                  How to Use the Optimizer
                </h1>
                <p className="text-white text-lg mt-1">A beginner's guide to getting smart picks and maximum value</p>
              </div>
            </div>

            {/* Search */}
            <div className="relative mt-6 max-w-2xl">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
              <Input
                type="text"
                placeholder="Search for help topics..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-12 h-12 bg-slate-800/50 border-slate-700 text-white placeholder:text-slate-400 text-base"
              />
            </div>
          </div>

          {/* Tutorials */}
          <div className="flex-1 overflow-hidden w-full">
            <Tabs defaultValue="first-steps" className="flex h-full flex-col">
              <TabsList className="h-auto flex-wrap justify-start overflow-x-auto border-b border-slate-700 bg-transparent p-0 mb-6">
                {tutorials.map((tutorial) => {
                  const Icon = tutorial.icon;
                  return (
                    <TabsTrigger
                      key={tutorial.id}
                      value={tutorial.id}
                      className="data-[state=active]:bg-transparent data-[state=active]:text-cyan-400 data-[state=active]:border-b-2 data-[state=active]:border-cyan-400 rounded-none border-b-2 border-transparent text-slate-400 hover:text-slate-300 px-6 py-3"
                    >
                      <Icon className="mr-2 h-4 w-4" />
                      {tutorial.title}
                    </TabsTrigger>
                  );
                })}
              </TabsList>

              <div className="flex-1 overflow-auto pr-2 pb-8">
                {(searchQuery ? filteredTutorials : tutorials).map((tutorial) => {
                  const Icon = tutorial.icon;
                  return (
                    <TabsContent key={tutorial.id} value={tutorial.id} className="mt-0">
                      <div className="mb-8">
                        <div className="mb-3 flex items-center gap-3">
                          <Icon className="h-7 w-7 text-cyan-400" />
                          <h2 className="text-3xl font-bold text-white">{tutorial.title}</h2>
                        </div>
                        <p className="text-slate-300 text-lg">{tutorial.description}</p>
                      </div>

                      <Accordion type="multiple" className="space-y-4">
                        {tutorial.content.map((item, index) => (
                          <AccordionItem
                            key={index}
                            value={`item-${index}`}
                            className="rounded-lg border border-slate-700 bg-slate-800/30 px-6 py-2"
                          >
                            <AccordionTrigger className="text-left transition-colors hover:text-cyan-400 py-4">
                              <span className="font-semibold text-white text-lg">{item.question}</span>
                            </AccordionTrigger>
                            <AccordionContent className="text-slate-200 leading-relaxed text-base pb-4">
                              {item.answer}
                            </AccordionContent>
                          </AccordionItem>
                        ))}
                      </Accordion>

                      {tutorial.content.length === 0 && searchQuery && (
                        <p className="py-12 text-center text-slate-400 text-lg">
                          No results found for "{searchQuery}"
                        </p>
                      )}
                    </TabsContent>
                  );
                })}
              </div>
            </Tabs>
          </div>
        </div>
      </div>
    </div>
  );
}
