import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:intl/intl.dart';

void main() {
  runApp(const WorkoutApp());
}

// ── Data Models ──────────────────────────────────────────────────────────────

class Exercise {
  final int id;
  final String name;
  final IconData icon;
  final String cat;
  final String sets;
  final String muscle;
  final String detail;
  const Exercise({
    required this.id,
    required this.name,
    required this.icon,
    required this.cat,
    required this.sets,
    required this.muscle,
    required this.detail,
  });
}

class WorkoutExercise {
  final String name;
  final String detail;
  final int totalSets;
  final double defaultWeight;
  final int defaultReps;
  const WorkoutExercise({
    required this.name,
    required this.detail,
    required this.totalSets,
    required this.defaultWeight,
    required this.defaultReps,
  });
}

class Issue {
  final int id;
  final String title;
  final String description;
  final String status;
  final DateTime dateReported;

  const Issue({
    required this.id,
    required this.title,
    required this.description,
    required this.status,
    required this.dateReported,
  });
}

class _WeekDay {
  final String day;
  final String date;
  final bool hasWorkout;
  final bool isScheduled;
  const _WeekDay(this.day, this.date, this.hasWorkout, this.isScheduled);
}

class _WorkoutPlan {
  final String dateLabel;
  final String name;
  final String status;
  final List<String> tags;
  const _WorkoutPlan(this.dateLabel, this.name, this.status, this.tags);
}

class _SettingItem {
  final IconData icon;
  final String label;
  const _SettingItem(this.icon, this.label);
}

class _ExercisePreview {
  final String name;
  final String sets;
  const _ExercisePreview(this.name, this.sets);
}

// ── Exercise & Workout Data ──────────────────────────────────────────────────

const List<Exercise> allExercises = [
  Exercise(id: 1, name: 'Bench Press', icon: Icons.fitness_center_rounded, cat: 'chest', sets: '4×8', muscle: 'Chest', detail: 'Compound movement'),
  Exercise(id: 2, name: 'Pull-Ups', icon: Icons.stairs_rounded, cat: 'back', sets: '4×8', muscle: 'Back', detail: 'Bodyweight pull'),
  Exercise(id: 3, name: 'Squats', icon: Icons.accessibility_new_rounded, cat: 'legs', sets: '4×10', muscle: 'Quads', detail: 'Compound lower body'),
  Exercise(id: 4, name: 'Bicep Curls', icon: Icons.fitness_center_rounded, cat: 'arms', sets: '3×12', muscle: 'Biceps', detail: 'Isolation curl'),
  Exercise(id: 5, name: 'Plank', icon: Icons.spa_rounded, cat: 'core', sets: '3×60s', muscle: 'Core', detail: 'Isometric hold'),
  Exercise(id: 6, name: 'Shoulder Press', icon: Icons.accessibility_new_rounded, cat: 'chest', sets: '3×10', muscle: 'Shoulders', detail: 'Overhead press'),
  Exercise(id: 7, name: 'Deadlift', icon: Icons.thunderstorm_rounded, cat: 'back', sets: '4×6', muscle: 'Posterior', detail: 'Compound hinge'),
  Exercise(id: 8, name: 'Lunges', icon: Icons.run_circle_rounded, cat: 'legs', sets: '3×12', muscle: 'Glutes', detail: 'Unilateral legs'),
  Exercise(id: 9, name: 'Tricep Dips', icon: Icons.directions_run_rounded, cat: 'arms', sets: '3×12', muscle: 'Triceps', detail: 'Bodyweight push'),
  Exercise(id: 10, name: 'Crunches', icon: Icons.refresh_rounded, cat: 'core', sets: '3×20', muscle: 'Abs', detail: 'Flexion movement'),
  Exercise(id: 11, name: 'Chest Fly', icon: Icons.flight_rounded, cat: 'chest', sets: '3×12', muscle: 'Chest', detail: 'Isolation fly'),
  Exercise(id: 12, name: 'Lat Pulldown', icon: Icons.straighten_rounded, cat: 'back', sets: '3×12', muscle: 'Lats', detail: 'Machine pull'),
];

const List<WorkoutExercise> workoutExercises = [
  WorkoutExercise(name: 'Bench Press', detail: 'Chest · Compound movement', totalSets: 4, defaultWeight: 60, defaultReps: 8),
  WorkoutExercise(name: 'Shoulder Press', detail: 'Shoulders · Overhead press', totalSets: 3, defaultWeight: 40, defaultReps: 10),
  WorkoutExercise(name: 'Chest Fly', detail: 'Chest · Isolation movement', totalSets: 3, defaultWeight: 20, defaultReps: 12),
  WorkoutExercise(name: 'Tricep Dips', detail: 'Triceps · Bodyweight', totalSets: 3, defaultWeight: 0, defaultReps: 12),
  WorkoutExercise(name: 'Cable Pushdown', detail: 'Triceps · Isolation', totalSets: 3, defaultWeight: 25, defaultReps: 15),
  WorkoutExercise(name: 'Push-Ups', detail: 'Chest · Bodyweight', totalSets: 3, defaultWeight: 0, defaultReps: 15),
];

// ── Theme Colors ─────────────────────────────────────────────────────────────

class AppColors {
  static const accent = Color(0xFFFF5733);
  static const accent2 = Color(0xFFFF8C42);
  static const bg = Color(0xFFF8F7F4);
  static const surface = Color(0xFFFFFFFF);
  static const surface2 = Color(0xFFF2F1EE);
  static const surface3 = Color(0xFFE8E6E1);
  static const text = Color(0xFF1A1A1A);
  static const muted = Color(0xFF888880);
  static const border = Color(0xFFE0DED8);
  static const green = Color(0xFF22C55E);
  static const greenBg = Color(0xFFEAFBF2);
  static const accentBg = Color(0xFFFFF0ED);
  static const yellow = Color(0xFFFACC15);
  static const yellowBg = Color(0xFFFFFBEB);
}

// ── State Management ─────────────────────────────────────────────────────────

class IssueData extends ChangeNotifier {
  final List<Issue> _issues;

  IssueData()
      : _issues = [
          Issue(id: 1, title: 'App crash on workout completion', description: 'The app crashes sometimes after completing a workout, preventing stat saving.', status: 'Open', dateReported: DateTime(2023, 10, 26, 14, 30)),
          Issue(id: 2, title: 'Inaccurate calorie calculation', description: 'The calorie burned estimation seems too high for low-intensity workouts.', status: 'In Progress', dateReported: DateTime(2023, 10, 25, 10, 0)),
          Issue(id: 3, title: 'Typo in "Browse" screen', description: 'Exercise "Bicep Curls" is sometimes displayed as "Bicep Curles".', status: 'Resolved', dateReported: DateTime(2023, 10, 24, 16, 15)),
          Issue(id: 4, title: 'Add custom exercises', description: 'Would be great to add my own exercises to the list.', status: 'Open', dateReported: DateTime(2023, 10, 23, 9, 0)),
        ];

  List<Issue> get issues => List<Issue>.unmodifiable(_issues);

  void addIssue(String title, String description) {
    _issues.add(Issue(
      id: _issues.length + 1,
      title: title,
      description: description,
      status: 'Open',
      dateReported: DateTime.now(),
    ));
    notifyListeners();
  }

  void updateIssueStatus(int id, String newStatus) {
    final int index = _issues.indexWhere((issue) => issue.id == id);
    if (index != -1) {
      _issues[index] = Issue(
        id: _issues[index].id,
        title: _issues[index].title,
        description: _issues[index].description,
        status: newStatus,
        dateReported: _issues[index].dateReported,
      );
      notifyListeners();
    }
  }
}

// ── Root App ──────────────────────────────────────────────────────────────────

class WorkoutApp extends StatelessWidget {
  const WorkoutApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider<IssueData>(
      create: (_) => IssueData(),
      child: MaterialApp(
        title: 'FitFlow',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: AppColors.accent, brightness: Brightness.light),
          scaffoldBackgroundColor: AppColors.bg,
          fontFamily: 'Roboto',
          useMaterial3: true,
        ),
        home: const MainShell(),
      ),
    );
  }
}

// ── Main Shell ────────────────────────────────────────────────────────────────

class MainShell extends StatefulWidget {
  const MainShell({super.key});

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  int _currentIndex = 0;
  final Set<int> _selectedExercises = {};

  void _toggleExercise(int id) {
    setState(() {
      if (_selectedExercises.contains(id)) {
        _selectedExercises.remove(id);
      } else {
        _selectedExercises.add(id);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final screens = [
      HomeScreen(
        selectedExercises: _selectedExercises,
        onToggle: _toggleExercise,
        onStartWorkout: () {
          Navigator.push(context, MaterialPageRoute(builder: (_) => const WorkoutScreen()));
        },
      ),
      BrowseScreen(selectedExercises: _selectedExercises, onToggle: _toggleExercise),
      const PlanScreen(),
      const ProfileScreen(),
      const IssuesScreen(),
    ];

    return Scaffold(
      backgroundColor: AppColors.bg,
      body: screens[_currentIndex],
      bottomNavigationBar: _BottomNavBar(
        currentIndex: _currentIndex,
        onTap: (index) => setState(() => _currentIndex = index),
      ),
    );
  }
}

class _BottomNavBar extends StatelessWidget {
  final int currentIndex;
  final ValueChanged<int> onTap;
  const _BottomNavBar({required this.currentIndex, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final items = [
      {'icon': Icons.home_rounded, 'label': 'Home'},
      {'icon': Icons.search_rounded, 'label': 'Browse'},
      {'icon': Icons.calendar_month_rounded, 'label': 'Plan'},
      {'icon': Icons.person_rounded, 'label': 'Profile'},
      {'icon': Icons.bug_report_rounded, 'label': 'Issues'},
    ];

    return Container(
      decoration: const BoxDecoration(
        color: AppColors.surface,
        border: Border(top: BorderSide(color: AppColors.border)),
      ),
      child: SafeArea(
        top: false,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: List.generate(items.length, (i) {
              final active = currentIndex == i;
              return GestureDetector(
                onTap: () => onTap(i),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                  decoration: BoxDecoration(
                    color: active ? AppColors.accentBg : Colors.transparent,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(items[i]['icon'] as IconData,
                          color: active ? AppColors.accent : AppColors.muted, size: 22),
                      const SizedBox(height: 2),
                      Text(items[i]['label'] as String,
                          style: TextStyle(
                              fontSize: 10,
                              fontWeight: FontWeight.w600,
                              color: active ? AppColors.accent : AppColors.muted)),
                    ],
                  ),
                ),
              );
            }),
          ),
        ),
      ),
    );
  }
}

// ── HOME SCREEN ───────────────────────────────────────────────────────────────

class HomeScreen extends StatefulWidget {
  final Set<int> selectedExercises;
  final void Function(int) onToggle;
  final VoidCallback onStartWorkout;

  const HomeScreen({
    super.key,
    required this.selectedExercises,
    required this.onToggle,
    required this.onStartWorkout,
  });

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String _filter = 'all';
  final List<String> _cats = ['All', 'Chest', 'Back', 'Legs', 'Arms', 'Core'];

  List<Exercise> get _filtered {
    if (_filter == 'all') return allExercises;
    return allExercises.where((e) => e.cat == _filter).toList();
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: CustomScrollView(
        slivers: [
          const SliverToBoxAdapter(child: HomeHeader()),
          const SliverToBoxAdapter(child: StreakBar()),
          SliverToBoxAdapter(
            child: SectionHeader(title: "Today's Workout", action: 'View Plan'),
          ),
          SliverToBoxAdapter(
            child: TodayWorkoutCard(onStartWorkout: widget.onStartWorkout),
          ),
          SliverToBoxAdapter(
            child: SectionHeader(title: 'Browse Exercises', action: 'See All'),
          ),
          SliverToBoxAdapter(
            child: CategoryPills(
              categories: _cats,
              selectedCategory: _filter,
              onCategorySelected: (cat) => setState(() => _filter = cat),
            ),
          ),
          SliverPadding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            sliver: SliverGrid(
              delegate: SliverChildBuilderDelegate(
                (ctx, i) => ExerciseCard(
                  exercise: _filtered[i],
                  selected: widget.selectedExercises.contains(_filtered[i].id),
                  onTap: () {
                    final wasSelected = widget.selectedExercises.contains(_filtered[i].id);
                    widget.onToggle(_filtered[i].id);
                    _showToast(context, wasSelected ? 'Removed from plan' : 'Added to plan ✓');
                  },
                ),
                childCount: _filtered.length,
              ),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2, mainAxisSpacing: 12, crossAxisSpacing: 12, childAspectRatio: 1.0,
              ),
            ),
          ),
          const SliverToBoxAdapter(child: SizedBox(height: 20)),
        ],
      ),
    );
  }
}

class HomeHeader extends StatelessWidget {
  const HomeHeader({super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 24, 20, 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Good morning,', style: TextStyle(fontSize: 13, color: AppColors.muted, fontWeight: FontWeight.w500)),
            const SizedBox(height: 2),
            const Text('Alex 👋', style: TextStyle(fontSize: 26, fontWeight: FontWeight.w800, color: AppColors.text)),
          ]),
          Row(children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(
                color: AppColors.accentBg,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: AppColors.accent.withOpacity(0.3)),
              ),
              child: Row(children: [
                Icon(Icons.local_fire_department_rounded, size: 14, color: AppColors.accent),
                const SizedBox(width: 4),
                Text('5 day streak', style: TextStyle(fontSize: 11, color: AppColors.accent, fontWeight: FontWeight.w700)),
              ]),
            ),
            const SizedBox(width: 10),
            Container(
              width: 44, height: 44,
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                    colors: [AppColors.accent, AppColors.accent2],
                    begin: Alignment.topLeft, end: Alignment.bottomRight),
                shape: BoxShape.circle,
                boxShadow: [BoxShadow(color: AppColors.accent.withOpacity(0.3), blurRadius: 10, offset: const Offset(0, 3))],
              ),
              child: const Center(child: Text('AX', style: TextStyle(color: Colors.white, fontWeight: FontWeight.w800, fontSize: 14))),
            ),
          ]),
        ],
      ),
    );
  }
}

class StreakBar extends StatelessWidget {
  const StreakBar({super.key});

  @override
  Widget build(BuildContext context) {
    final days = ['M', 'T', 'W', 'T', 'F', 'S', 'S'];
    return Container(
      margin: const EdgeInsets.fromLTRB(20, 0, 20, 20),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 8, offset: const Offset(0, 2))],
      ),
      child: Row(
        children: [
          Container(
            width: 42, height: 42,
            decoration: BoxDecoration(
              gradient: const LinearGradient(colors: [Color(0xFFFF7043), Color(0xFFFF5733)]),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.local_fire_department_rounded, color: Colors.white, size: 22),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('WEEKLY PROGRESS', style: TextStyle(fontSize: 10, color: AppColors.muted, fontWeight: FontWeight.w700, letterSpacing: 0.8)),
              const SizedBox(height: 4),
              Row(
                children: List.generate(7, (i) {
                  final done = i < 5;
                  return Expanded(
                    child: Container(
                      margin: const EdgeInsets.symmetric(horizontal: 2),
                      child: Column(children: [
                        Container(
                          height: 6,
                          decoration: BoxDecoration(
                            gradient: done
                                ? const LinearGradient(colors: [AppColors.accent2, AppColors.accent])
                                : null,
                            color: done ? null : AppColors.surface3,
                            borderRadius: BorderRadius.circular(3),
                          ),
                        ),
                        const SizedBox(height: 3),
                        Text(days[i], style: TextStyle(fontSize: 8, color: done ? AppColors.accent : AppColors.muted, fontWeight: FontWeight.w600)),
                      ]),
                    ),
                  );
                }),
              ),
            ]),
          ),
          const SizedBox(width: 12),
          Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
            Text('5/7', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800, color: AppColors.accent)),
            Text('days done', style: TextStyle(fontSize: 10, color: AppColors.muted)),
          ]),
        ],
      ),
    );
  }
}

class SectionHeader extends StatelessWidget {
  final String title;
  final String action;

  const SectionHeader({super.key, required this.title, required this.action});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 0, 20, 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(title, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w800, color: AppColors.text)),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: AppColors.accentBg,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(action, style: const TextStyle(fontSize: 12, color: AppColors.accent, fontWeight: FontWeight.w600)),
          ),
        ],
      ),
    );
  }
}

class TodayWorkoutCard extends StatelessWidget {
  final VoidCallback onStartWorkout;

  const TodayWorkoutCard({required this.onStartWorkout, super.key});

  @override
  Widget build(BuildContext context) {
    final previews = [
      _ExercisePreview('Bench Press', '4×8'),
      _ExercisePreview('Shoulder Press', '3×10'),
      _ExercisePreview('Tricep Dips', '3×12'),
    ];

    return GestureDetector(
      onTap: onStartWorkout,
      child: Container(
        margin: const EdgeInsets.fromLTRB(20, 0, 20, 24),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(22),
          boxShadow: [
            BoxShadow(color: AppColors.accent.withOpacity(0.18), blurRadius: 20, offset: const Offset(0, 6)),
          ],
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(22),
          child: Stack(
            children: [
              // Background gradient
              Container(
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Color(0xFFFFF3F0), Color(0xFFFFE0D0), Color(0xFFFFD0BC)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                ),
              ),
              // Decorative circle
              Positioned(
                right: -30, top: -30,
                child: Container(
                  width: 140, height: 140,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: AppColors.accent.withOpacity(0.08),
                  ),
                ),
              ),
              Positioned(
                right: 20, bottom: -40,
                child: Container(
                  width: 100, height: 100,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: AppColors.accent2.withOpacity(0.1),
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(children: [
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: AppColors.accent.withOpacity(0.15),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(color: AppColors.accent.withOpacity(0.3)),
                        ),
                        child: Text('DAY 6 · UPPER BODY',
                            style: TextStyle(fontSize: 10, color: AppColors.accent, fontWeight: FontWeight.w700, letterSpacing: 0.6)),
                      ),
                      const Spacer(),
                      Icon(Icons.play_circle_filled_rounded, color: AppColors.accent, size: 32),
                    ]),
                    const SizedBox(height: 10),
                    const Text('Push Power', style: TextStyle(fontSize: 24, fontWeight: FontWeight.w800, color: AppColors.text)),
                    const SizedBox(height: 6),
                    Row(children: [
                      MetaChip(icon: Icons.timer_rounded, label: '45 min'),
                      const SizedBox(width: 16),
                      MetaChip(icon: Icons.local_fire_department_rounded, label: '380 kcal'),
                      const SizedBox(width: 16),
                      MetaChip(icon: Icons.fitness_center_rounded, label: '6 sets'),
                    ]),
                    const SizedBox(height: 14),
                    ...previews.map((e) => Container(
                      margin: const EdgeInsets.only(bottom: 6),
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.65),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: Colors.white.withOpacity(0.8)),
                      ),
                      child: Row(children: [
                        Container(width: 7, height: 7,
                            decoration: const BoxDecoration(color: AppColors.accent, shape: BoxShape.circle)),
                        const SizedBox(width: 10),
                        Expanded(child: Text(e.name,
                            style: const TextStyle(fontSize: 13, color: AppColors.text, fontWeight: FontWeight.w600))),
                        Text(e.sets, style: TextStyle(fontSize: 12, color: AppColors.muted, fontWeight: FontWeight.w500)),
                      ]),
                    )),
                    const SizedBox(height: 14),
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(vertical: 15),
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                            colors: [AppColors.accent2, AppColors.accent],
                            begin: Alignment.centerLeft, end: Alignment.centerRight),
                        borderRadius: BorderRadius.circular(14),
                        boxShadow: [BoxShadow(color: AppColors.accent.withOpacity(0.35), blurRadius: 12, offset: const Offset(0, 4))],
                      ),
                      child: const Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Text('Start Workout', style: TextStyle(color: Colors.white, fontWeight: FontWeight.w800, fontSize: 15, letterSpacing: 0.3)),
                          SizedBox(width: 8),
                          Icon(Icons.arrow_forward_rounded, color: Colors.white, size: 18),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class MetaChip extends StatelessWidget {
  final IconData icon;
  final String label;
  const MetaChip({required this.icon, required this.label, super.key});

  @override
  Widget build(BuildContext context) {
    return Row(children: [
      Icon(icon, size: 13, color: AppColors.accent),
      const SizedBox(width: 4),
      Text(label, style: TextStyle(fontSize: 12, color: AppColors.muted, fontWeight: FontWeight.w500)),
    ]);
  }
}

class CategoryPills extends StatelessWidget {
  final List<String> categories;
  final String selectedCategory;
  final ValueChanged<String> onCategorySelected;

  const CategoryPills({
    required this.categories,
    required this.selectedCategory,
    required this.onCategorySelected,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 44,
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 20),
        itemCount: categories.length,
        itemBuilder: (ctx, i) {
          final cat = categories[i].toLowerCase();
          final active = (i == 0 && selectedCategory == 'all') || (i > 0 && selectedCategory == cat);
          return GestureDetector(
            onTap: () => onCategorySelected(i == 0 ? 'all' : cat),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 180),
              margin: const EdgeInsets.only(right: 8, bottom: 12),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
              decoration: BoxDecoration(
                gradient: active
                    ? const LinearGradient(colors: [AppColors.accent2, AppColors.accent])
                    : null,
                color: active ? null : AppColors.surface,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: active ? AppColors.accent : AppColors.border),
                boxShadow: active ? [BoxShadow(color: AppColors.accent.withOpacity(0.25), blurRadius: 8, offset: const Offset(0, 2))] : null,
              ),
              child: Text(categories[i],
                  style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600,
                      color: active ? Colors.white : AppColors.muted)),
            ),
          );
        },
      ),
    );
  }
}

// ── Exercise Card ─────────────────────────────────────────────────────────────

class ExerciseCard extends StatelessWidget {
  final Exercise exercise;
  final bool selected;
  final VoidCallback onTap;

  const ExerciseCard({required this.exercise, required this.selected, required this.onTap, super.key});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: selected ? AppColors.accentBg : AppColors.surface,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: selected ? AppColors.accent : AppColors.border, width: selected ? 1.5 : 1),
          boxShadow: [
            BoxShadow(
              color: selected ? AppColors.accent.withOpacity(0.12) : Colors.black.withOpacity(0.04),
              blurRadius: 8, offset: const Offset(0, 2),
            )
          ],
        ),
        child: Stack(
          children: [
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  width: 38, height: 38,
                  decoration: BoxDecoration(
                    color: selected ? AppColors.accent.withOpacity(0.15) : AppColors.surface2,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Icon(exercise.icon, size: 20, color: selected ? AppColors.accent : AppColors.muted),
                ),
                const SizedBox(height: 8),
                Text(exercise.name, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w800, color: AppColors.text)),
                const SizedBox(height: 2),
                Text(exercise.sets, style: TextStyle(fontSize: 11, color: AppColors.muted)),
                const Spacer(),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                      color: selected ? AppColors.accent.withOpacity(0.12) : AppColors.surface2,
                      borderRadius: BorderRadius.circular(10)),
                  child: Text(exercise.muscle,
                      style: TextStyle(fontSize: 10, color: selected ? AppColors.accent : AppColors.muted, fontWeight: FontWeight.w600)),
                ),
              ],
            ),
            if (selected)
              Positioned(
                top: 0, right: 0,
                child: Container(
                  width: 22, height: 22,
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(colors: [AppColors.accent2, AppColors.accent]),
                    shape: BoxShape.circle,
                    boxShadow: [BoxShadow(color: AppColors.accent.withOpacity(0.4), blurRadius: 6, offset: const Offset(0, 2))],
                  ),
                  child: const Icon(Icons.check, color: Colors.white, size: 12),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

// ── BROWSE SCREEN ─────────────────────────────────────────────────────────────

class BrowseScreen extends StatefulWidget {
  final Set<int> selectedExercises;
  final void Function(int) onToggle;

  const BrowseScreen({super.key, required this.selectedExercises, required this.onToggle});

  @override
  State<BrowseScreen> createState() => _BrowseScreenState();
}

class _BrowseScreenState extends State<BrowseScreen> {
  String _filter = 'all';
  final List<String> _cats = ['All', 'Chest', 'Back', 'Legs', 'Arms', 'Core'];

  List<Exercise> get _filtered {
    if (_filter == 'all') return allExercises;
    return allExercises.where((e) => e.cat == _filter).toList();
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 24, 20, 12),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Exercises', style: TextStyle(fontSize: 26, fontWeight: FontWeight.w800, color: AppColors.text)),
                AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    gradient: widget.selectedExercises.isEmpty
                        ? null
                        : const LinearGradient(colors: [AppColors.accent2, AppColors.accent]),
                    color: widget.selectedExercises.isEmpty ? AppColors.surface2 : null,
                    borderRadius: BorderRadius.circular(20),
                    boxShadow: widget.selectedExercises.isEmpty
                        ? null
                        : [BoxShadow(color: AppColors.accent.withOpacity(0.3), blurRadius: 8, offset: const Offset(0, 2))],
                  ),
                  child: Text('${widget.selectedExercises.length} selected',
                      style: TextStyle(
                          fontSize: 12,
                          color: widget.selectedExercises.isEmpty ? AppColors.muted : Colors.white,
                          fontWeight: FontWeight.w700)),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 0, 20, 12),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 13),
              decoration: BoxDecoration(
                color: AppColors.surface,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: AppColors.border),
                boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.03), blurRadius: 6, offset: const Offset(0, 2))],
              ),
              child: Row(children: [
                Icon(Icons.search_rounded, color: AppColors.muted, size: 18),
                const SizedBox(width: 8),
                Text('Search exercises...', style: TextStyle(fontSize: 14, color: AppColors.muted)),
              ]),
            ),
          ),
          CategoryPills(
            categories: _cats,
            selectedCategory: _filter,
            onCategorySelected: (cat) => setState(() => _filter = cat),
          ),
          const SizedBox(height: 8),
          Expanded(
            child: GridView.builder(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2, mainAxisSpacing: 12, crossAxisSpacing: 12, childAspectRatio: 1.0,
              ),
              itemCount: _filtered.length,
              itemBuilder: (ctx, i) => ExerciseCard(
                exercise: _filtered[i],
                selected: widget.selectedExercises.contains(_filtered[i].id),
                onTap: () {
                  final wasSelected = widget.selectedExercises.contains(_filtered[i].id);
                  widget.onToggle(_filtered[i].id);
                  setState(() {});
                  _showToast(context, wasSelected ? 'Removed from plan' : 'Added to plan ✓');
                },
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ── PLAN SCREEN ───────────────────────────────────────────────────────────────

class PlanScreen extends StatelessWidget {
  const PlanScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final weekDays = [
      _WeekDay('Mon', '30', true, true),
      _WeekDay('Tue', '31', true, true),
      _WeekDay('Wed', '1', true, true),
      _WeekDay('Thu', '2', true, true),
      _WeekDay('Fri', '3', true, true),
      _WeekDay('Sat', '4', false, true),
      _WeekDay('Sun', '5', false, false),
    ];

    final workouts = [
      _WorkoutPlan('Monday · Mar 30', 'Pull Day', 'done', ['Back', 'Biceps', '5 exercises', '42 min']),
      _WorkoutPlan('Tuesday · Apr 1', 'Leg Day', 'done', ['Quads', 'Hamstrings', '6 exercises', '50 min']),
      _WorkoutPlan('Wednesday · Apr 2', 'Push Day', 'done', ['Chest', 'Shoulders', '6 exercises', '45 min']),
      _WorkoutPlan('Thursday · Apr 3', 'Core & Cardio', 'done', ['Core', 'Cardio', '5 exercises', '35 min']),
      _WorkoutPlan('Friday · Apr 4', 'Full Body', 'done', ['Full Body', '7 exercises', '55 min']),
      _WorkoutPlan('Saturday · Apr 5 · TODAY', 'Push Power', 'today', ['Chest', 'Triceps', '6 exercises', '45 min']),
      _WorkoutPlan('Sunday · Apr 6', 'Rest Day', 'rest', ['Recovery', 'Stretch']),
    ];

    return SafeArea(
      child: CustomScrollView(
        slivers: [
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 24, 20, 16),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text('Weekly Plan', style: TextStyle(fontSize: 26, fontWeight: FontWeight.w800, color: AppColors.text)),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: AppColors.greenBg,
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(color: AppColors.green.withOpacity(0.3)),
                    ),
                    child: Text('5/6 done ✓',
                        style: const TextStyle(fontSize: 12, color: AppColors.green, fontWeight: FontWeight.w700)),
                  ),
                ],
              ),
            ),
          ),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
              child: Row(
                children: weekDays.map((d) {
                  final isToday = d.date == '4' && d.day == 'Sat';
                  return Expanded(
                    child: Container(
                      margin: const EdgeInsets.symmetric(horizontal: 3),
                      padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 2),
                      decoration: BoxDecoration(
                        gradient: isToday
                            ? const LinearGradient(colors: [AppColors.accent2, AppColors.accent], begin: Alignment.topCenter, end: Alignment.bottomCenter)
                            : null,
                        color: isToday ? null : d.hasWorkout ? AppColors.greenBg : AppColors.surface,
                        borderRadius: BorderRadius.circular(14),
                        border: Border.all(
                          color: isToday ? AppColors.accent : d.hasWorkout ? AppColors.green.withOpacity(0.4) : AppColors.border,
                        ),
                        boxShadow: isToday
                            ? [BoxShadow(color: AppColors.accent.withOpacity(0.3), blurRadius: 8, offset: const Offset(0, 3))]
                            : null,
                      ),
                      child: Column(children: [
                        Text(d.day, style: TextStyle(fontSize: 9, fontWeight: FontWeight.w700,
                            color: isToday ? Colors.white70 : AppColors.muted)),
                        const SizedBox(height: 4),
                        Text(d.date, style: TextStyle(fontSize: 15, fontWeight: FontWeight.w800,
                            color: isToday ? Colors.white : AppColors.text)),
                        const SizedBox(height: 4),
                        if (d.isScheduled)
                          Container(width: 5, height: 5, decoration: BoxDecoration(
                            color: isToday ? Colors.white : d.hasWorkout ? AppColors.green : AppColors.border,
                            shape: BoxShape.circle,
                          )),
                      ]),
                    ),
                  );
                }).toList(),
              ),
            ),
          ),
          SliverList(
            delegate: SliverChildBuilderDelegate(
              (ctx, i) {
                final w = workouts[i];
                return GestureDetector(
                  onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const WorkoutScreen())),
                  child: Container(
                    margin: const EdgeInsets.fromLTRB(20, 0, 20, 10),
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: AppColors.surface,
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(
                        color: w.status == 'today' ? AppColors.accent.withOpacity(0.4) : AppColors.border,
                        width: w.status == 'today' ? 1.5 : 1,
                      ),
                      boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.03), blurRadius: 6, offset: const Offset(0, 2))],
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Expanded(
                              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                                Text(w.dateLabel, style: TextStyle(fontSize: 10, color: AppColors.muted, fontWeight: FontWeight.w600)),
                                const SizedBox(height: 2),
                                Text(w.name, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w800, color: AppColors.text)),
                              ]),
                            ),
                            StatusBadge(status: w.status),
                          ],
                        ),
                        const SizedBox(height: 10),
                        Wrap(
                          spacing: 6, runSpacing: 4,
                          children: w.tags.map((tag) => Container(
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                            decoration: BoxDecoration(color: AppColors.surface2, borderRadius: BorderRadius.circular(8)),
                            child: Text(tag, style: TextStyle(fontSize: 11, color: AppColors.muted, fontWeight: FontWeight.w500)),
                          )).toList(),
                        ),
                      ],
                    ),
                  ),
                );
              },
              childCount: workouts.length,
            ),
          ),
          const SliverToBoxAdapter(child: SizedBox(height: 20)),
        ],
      ),
    );
  }
}

class StatusBadge extends StatelessWidget {
  final String status;
  const StatusBadge({required this.status, super.key});

  @override
  Widget build(BuildContext context) {
    Color bg;
    Color fg;
    String label;
    switch (status) {
      case 'done':
        bg = AppColors.greenBg; fg = AppColors.green; label = 'Done ✓';
        break;
      case 'today':
        bg = AppColors.accentBg; fg = AppColors.accent; label = 'Today';
        break;
      default:
        bg = AppColors.surface2; fg = AppColors.muted; label = 'Rest';
    }
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(20)),
      child: Text(label, style: TextStyle(fontSize: 11, color: fg, fontWeight: FontWeight.w700)),
    );
  }
}

// ── WORKOUT SCREEN ────────────────────────────────────────────────────────────

class WorkoutScreen extends StatefulWidget {
  const WorkoutScreen({super.key});

  @override
  State<WorkoutScreen> createState() => _WorkoutScreenState();
}

class _WorkoutScreenState extends State<WorkoutScreen> {
  int _exIdx = 0;
  int _setIdx = 0;
  double _weight = 60;
  int _reps = 8;

  WorkoutExercise get _current => workoutExercises[_exIdx];

  double get _progress =>
      (_exIdx / workoutExercises.length) + (1 / workoutExercises.length) * (_setIdx / _current.totalSets);

  void _completeSet() {
    setState(() {
      _setIdx++;
      if (_setIdx >= _current.totalSets) {
        _setIdx = 0;
        _exIdx++;
        if (_exIdx >= workoutExercises.length) {
          _showCompletion();
          return;
        }
        _weight = workoutExercises[_exIdx].defaultWeight;
        _reps = workoutExercises[_exIdx].defaultReps;
        _showToast(context, 'Next exercise →');
      } else {
        _showToast(context, 'Set $_setIdx done! Rest 60s');
      }
    });
  }

  void _showCompletion() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => Dialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
        backgroundColor: AppColors.surface,
        child: Padding(
          padding: const EdgeInsets.all(28),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 80, height: 80,
                decoration: BoxDecoration(
                  gradient: const LinearGradient(colors: [Color(0xFFFFB347), Color(0xFFFFCC02)]),
                  shape: BoxShape.circle,
                  boxShadow: [BoxShadow(color: Colors.amber.withOpacity(0.4), blurRadius: 16, offset: const Offset(0, 4))],
                ),
                child: const Icon(Icons.emoji_events_rounded, size: 44, color: Colors.white),
              ),
              const SizedBox(height: 16),
              const Text('Workout Done!', style: TextStyle(fontSize: 26, fontWeight: FontWeight.w800, color: AppColors.text)),
              const SizedBox(height: 6),
              Text('You crushed Push Power today.', style: TextStyle(fontSize: 14, color: AppColors.muted)),
              const SizedBox(height: 24),
              GridView.count(
                crossAxisCount: 2, shrinkWrap: true, mainAxisSpacing: 10, crossAxisSpacing: 10,
                childAspectRatio: 1.6,
                children: const [
                  CompletionStat(value: '6', label: 'Exercises'),
                  CompletionStat(value: '45', label: 'Minutes'),
                  CompletionStat(value: '380', label: 'Calories'),
                  CompletionStat(value: '24', label: 'Total Sets'),
                ],
              ),
              const SizedBox(height: 20),
              GestureDetector(
                onTap: () {
                  Navigator.pop(context);
                  Navigator.pop(context);
                },
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(vertical: 15),
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(colors: [AppColors.accent2, AppColors.accent]),
                    borderRadius: BorderRadius.circular(14),
                    boxShadow: [BoxShadow(color: AppColors.accent.withOpacity(0.35), blurRadius: 12, offset: const Offset(0, 4))],
                  ),
                  child: const Center(
                    child: Text('Back to Home', style: TextStyle(color: Colors.white, fontWeight: FontWeight.w800, fontSize: 16)),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_exIdx >= workoutExercises.length) return const SizedBox.shrink();
    final ex = workoutExercises[_exIdx];
    final hasNext = _exIdx + 1 < workoutExercises.length;

    return Scaffold(
      backgroundColor: AppColors.bg,
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 16),
              decoration: const BoxDecoration(
                color: AppColors.surface,
                border: Border(bottom: BorderSide(color: AppColors.border)),
              ),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                GestureDetector(
                  onTap: () => Navigator.pop(context),
                  child: Row(children: [
                    Icon(Icons.arrow_back_ios_new_rounded, size: 16, color: AppColors.accent),
                    const SizedBox(width: 4),
                    Text('Back', style: TextStyle(color: AppColors.accent, fontSize: 14, fontWeight: FontWeight.w600)),
                  ]),
                ),
                const SizedBox(height: 12),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text('Push Power', style: TextStyle(fontSize: 26, fontWeight: FontWeight.w800, color: AppColors.text)),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                      decoration: BoxDecoration(
                        color: AppColors.accentBg,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text('${_exIdx + 1} / ${workoutExercises.length}',
                          style: TextStyle(fontSize: 12, color: AppColors.accent, fontWeight: FontWeight.w700)),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Stack(children: [
                  Container(height: 7, decoration: BoxDecoration(color: AppColors.surface2, borderRadius: BorderRadius.circular(4))),
                  FractionallySizedBox(
                    widthFactor: _progress.clamp(0.0, 1.0),
                    child: Container(
                      height: 7,
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(colors: [AppColors.accent2, AppColors.accent]),
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                  ),
                ]),
              ]),
            ),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      padding: const EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        color: AppColors.surface,
                        borderRadius: BorderRadius.circular(18),
                        border: Border.all(color: AppColors.border),
                        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 10, offset: const Offset(0, 3))],
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(children: [
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
                              decoration: BoxDecoration(color: AppColors.accentBg, borderRadius: BorderRadius.circular(20)),
                              child: Text('Exercise ${_exIdx + 1} of ${workoutExercises.length}',
                                  style: TextStyle(fontSize: 10, color: AppColors.accent, fontWeight: FontWeight.w700)),
                            ),
                          ]),
                          const SizedBox(height: 8),
                          Text(ex.name, style: const TextStyle(fontSize: 26, fontWeight: FontWeight.w800, color: AppColors.text)),
                          const SizedBox(height: 2),
                          Text(ex.detail, style: TextStyle(fontSize: 13, color: AppColors.muted)),
                          const SizedBox(height: 20),
                          Row(children: [
                            Expanded(child: ControlBox(
                                label: 'Weight', value: _weight.toInt().toString(), unit: 'kg',
                                onMinus: () => setState(() => _weight = (_weight - 2.5).clamp(0, 300)),
                                onPlus: () => setState(() => _weight = (_weight + 2.5).clamp(0, 300)))),
                            const SizedBox(width: 12),
                            Expanded(child: ControlBox(
                                label: 'Reps', value: _reps.toString(), unit: 'reps',
                                onMinus: () => setState(() => _reps = (_reps - 1).clamp(1, 100)),
                                onPlus: () => setState(() => _reps = (_reps + 1).clamp(1, 100)))),
                          ]),
                        ],
                      ),
                    ),
                    const SizedBox(height: 16),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text('Sets', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: AppColors.text)),
                        Text('$_setIdx / ${ex.totalSets} completed', style: TextStyle(fontSize: 12, color: AppColors.muted)),
                      ],
                    ),
                    const SizedBox(height: 10),
                    Row(
                      children: List.generate(ex.totalSets, (i) {
                        final done = i < _setIdx;
                        final current = i == _setIdx;
                        return Container(
                          width: 42, height: 42, margin: const EdgeInsets.only(right: 8),
                          decoration: BoxDecoration(
                            gradient: done ? const LinearGradient(colors: [AppColors.accent2, AppColors.accent]) : null,
                            color: done ? null : AppColors.surface,
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: done ? AppColors.accent : current ? AppColors.accent : AppColors.border,
                              width: 1.5,
                            ),
                            boxShadow: done ? [BoxShadow(color: AppColors.accent.withOpacity(0.3), blurRadius: 6, offset: const Offset(0, 2))] : null,
                          ),
                          child: Center(
                            child: done
                                ? const Icon(Icons.check, color: Colors.white, size: 17)
                                : Text('${i + 1}', style: TextStyle(
                                    fontSize: 13, fontWeight: FontWeight.w700,
                                    color: current ? AppColors.accent : AppColors.muted)),
                          ),
                        );
                      }),
                    ),
                    const SizedBox(height: 16),
                    GestureDetector(
                      onTap: _completeSet,
                      child: Container(
                        width: double.infinity,
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        decoration: BoxDecoration(
                          gradient: const LinearGradient(colors: [AppColors.accent2, AppColors.accent]),
                          borderRadius: BorderRadius.circular(14),
                          boxShadow: [BoxShadow(color: AppColors.accent.withOpacity(0.35), blurRadius: 12, offset: const Offset(0, 4))],
                        ),
                        child: const Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.check_circle_outline_rounded, color: Colors.white, size: 20),
                            SizedBox(width: 8),
                            Text('Complete Set', style: TextStyle(color: Colors.white, fontWeight: FontWeight.w800, fontSize: 16)),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: 12),
                    if (hasNext)
                      Container(
                        padding: const EdgeInsets.all(14),
                        decoration: BoxDecoration(
                          color: AppColors.surface,
                          borderRadius: BorderRadius.circular(14),
                          border: Border.all(color: AppColors.border),
                        ),
                        child: Row(children: [
                          Container(
                            width: 38, height: 38,
                            decoration: BoxDecoration(color: AppColors.surface2, borderRadius: BorderRadius.circular(10)),
                            child: const Icon(Icons.fast_forward_rounded, size: 18, color: AppColors.muted),
                          ),
                          const SizedBox(width: 12),
                          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                            Text('Up next', style: TextStyle(fontSize: 11, color: AppColors.muted, fontWeight: FontWeight.w500)),
                            Text(workoutExercises[_exIdx + 1].name,
                                style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w800, color: AppColors.text)),
                          ]),
                          const Spacer(),
                          Icon(Icons.chevron_right_rounded, color: AppColors.muted),
                        ]),
                      ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class CompletionStat extends StatelessWidget {
  final String value;
  final String label;
  const CompletionStat({required this.value, required this.label, super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.accentBg,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.accent.withOpacity(0.15)),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(value, style: const TextStyle(fontSize: 24, fontWeight: FontWeight.w800, color: AppColors.accent)),
          Text(label, style: TextStyle(fontSize: 11, color: AppColors.muted)),
        ],
      ),
    );
  }
}

class ControlBox extends StatelessWidget {
  final String label;
  final String value;
  final String unit;
  final VoidCallback onMinus;
  final VoidCallback onPlus;

  const ControlBox({
    required this.label,
    required this.value,
    required this.unit,
    required this.onMinus,
    required this.onPlus,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.surface2,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        children: [
          Text(label, style: TextStyle(fontSize: 11, color: AppColors.muted, fontWeight: FontWeight.w600)),
          const SizedBox(height: 6),
          Text(value, style: const TextStyle(fontSize: 32, fontWeight: FontWeight.w800, color: AppColors.text)),
          Text(unit, style: TextStyle(fontSize: 11, color: AppColors.muted)),
          const SizedBox(height: 10),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              GestureDetector(
                onTap: onMinus,
                child: Container(
                  width: 32, height: 32,
                  decoration: BoxDecoration(
                    color: AppColors.surface, shape: BoxShape.circle,
                    border: Border.all(color: AppColors.border),
                    boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 4, offset: const Offset(0, 1))],
                  ),
                  child: const Icon(Icons.remove_rounded, size: 16, color: AppColors.text),
                ),
              ),
              const SizedBox(width: 22),
              GestureDetector(
                onTap: onPlus,
                child: Container(
                  width: 32, height: 32,
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(colors: [AppColors.accent2, AppColors.accent]),
                    shape: BoxShape.circle,
                    boxShadow: [BoxShadow(color: AppColors.accent.withOpacity(0.35), blurRadius: 6, offset: const Offset(0, 2))],
                  ),
                  child: const Icon(Icons.add_rounded, size: 16, color: Colors.white),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ── PROFILE SCREEN ────────────────────────────────────────────────────────────

class ProfileScreen extends StatelessWidget {
  const ProfileScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final settings = [
      _SettingItem(Icons.track_changes_rounded, 'Goals'),
      _SettingItem(Icons.notifications_rounded, 'Notifications'),
      _SettingItem(Icons.bar_chart_rounded, 'Progress & Stats'),
      _SettingItem(Icons.settings_rounded, 'Settings'),
    ];

    return SafeArea(
      child: SingleChildScrollView(
        child: Column(
          children: [
            // Hero
            Container(
              width: double.infinity,
              decoration: const BoxDecoration(
                color: AppColors.surface,
                border: Border(bottom: BorderSide(color: AppColors.border)),
              ),
              child: Stack(
                children: [
                  Positioned(
                    right: -20, top: -20,
                    child: Container(
                      width: 120, height: 120,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: AppColors.accent.withOpacity(0.06),
                      ),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(20, 28, 20, 24),
                    child: Column(children: [
                      Container(
                        width: 88, height: 88,
                        decoration: BoxDecoration(
                          gradient: const LinearGradient(colors: [AppColors.accent, AppColors.accent2],
                              begin: Alignment.topLeft, end: Alignment.bottomRight),
                          shape: BoxShape.circle,
                          boxShadow: [BoxShadow(color: AppColors.accent.withOpacity(0.35), blurRadius: 20, offset: const Offset(0, 6))],
                        ),
                        child: const Center(child: Text('AX', style: TextStyle(color: Colors.white, fontWeight: FontWeight.w800, fontSize: 30))),
                      ),
                      const SizedBox(height: 14),
                      const Text('Alex', style: TextStyle(fontSize: 26, fontWeight: FontWeight.w800, color: AppColors.text)),
                      const SizedBox(height: 4),
                      Text('Intermediate · 6 months active', style: TextStyle(fontSize: 13, color: AppColors.muted)),
                    ]),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(20),
              child: Row(
                children: [
                  StatBox(value: '47', label: 'Workouts'),
                  const SizedBox(width: 10),
                  StatBox(value: '5', label: 'Streak', icon: Icons.local_fire_department_rounded),
                  const SizedBox(width: 10),
                  StatBox(value: '18k', label: 'Calories'),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
              child: Container(
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
                  color: AppColors.surface,
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(color: AppColors.border),
                  boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.03), blurRadius: 8, offset: const Offset(0, 2))],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text('Monthly Volume', style: TextStyle(fontSize: 15, fontWeight: FontWeight.w800, color: AppColors.text)),
                        Text('Apr 2026', style: TextStyle(fontSize: 12, color: AppColors.muted)),
                      ],
                    ),
                    const SizedBox(height: 16),
                    SizedBox(
                      height: 90,
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.end,
                        children: [
                          VolumeBar(heightFactor: 0.4, label: 'Dec'),
                          VolumeBar(heightFactor: 0.55, label: 'Jan'),
                          VolumeBar(heightFactor: 0.7, label: 'Feb'),
                          VolumeBar(heightFactor: 0.85, label: 'Mar'),
                          VolumeBar(heightFactor: 1.0, label: 'Apr'),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
            ...settings.map((s) => Container(
              decoration: const BoxDecoration(border: Border(bottom: BorderSide(color: AppColors.border))),
              child: ListTile(
                contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 4),
                leading: Container(
                  width: 38, height: 38,
                  decoration: BoxDecoration(
                    color: AppColors.accentBg,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(s.icon, color: AppColors.accent, size: 18),
                ),
                title: Text(s.label, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: AppColors.text)),
                trailing: const Icon(Icons.chevron_right_rounded, color: AppColors.muted),
                onTap: () {},
              ),
            )),
            const SizedBox(height: 30),
          ],
        ),
      ),
    );
  }
}

class StatBox extends StatelessWidget {
  final String value;
  final String label;
  final IconData? icon;

  const StatBox({required this.value, required this.label, this.icon, super.key});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: AppColors.border),
          boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.03), blurRadius: 6, offset: const Offset(0, 2))],
        ),
        child: Column(children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(value, style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w800, color: AppColors.accent)),
              if (icon != null) ...[
                const SizedBox(width: 4),
                Icon(icon, size: 18, color: AppColors.accent),
              ],
            ],
          ),
          const SizedBox(height: 2),
          Text(label, style: TextStyle(fontSize: 11, color: AppColors.muted)),
        ]),
      ),
    );
  }
}

class VolumeBar extends StatelessWidget {
  final double heightFactor;
  final String label;

  const VolumeBar({required this.heightFactor, required this.label, super.key});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 4),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.end,
          children: [
            Flexible(
              child: FractionallySizedBox(
                alignment: Alignment.bottomCenter,
                heightFactor: heightFactor,
                child: Container(
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                        colors: [AppColors.accent2, AppColors.accent],
                        begin: Alignment.topCenter, end: Alignment.bottomCenter),
                    borderRadius: BorderRadius.circular(7),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 6),
            Text(label, style: TextStyle(fontSize: 10, color: AppColors.muted, fontWeight: FontWeight.w500)),
          ],
        ),
      ),
    );
  }
}

// ── ISSUES SCREEN ─────────────────────────────────────────────────────────────

class IssuesScreen extends StatelessWidget {
  const IssuesScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 24, 20, 12),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Issues & Feedback',
                    style: TextStyle(fontSize: 22, fontWeight: FontWeight.w800, color: AppColors.text)),
                GestureDetector(
                  onTap: () => _showReportIssueDialog(context),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(colors: [AppColors.accent2, AppColors.accent]),
                      borderRadius: BorderRadius.circular(20),
                      boxShadow: [BoxShadow(color: AppColors.accent.withOpacity(0.3), blurRadius: 8, offset: const Offset(0, 2))],
                    ),
                    child: Row(children: const [
                      Icon(Icons.add_rounded, size: 14, color: Colors.white),
                      SizedBox(width: 4),
                      Text('Report', style: TextStyle(fontSize: 12, color: Colors.white, fontWeight: FontWeight.w700)),
                    ]),
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: Consumer<IssueData>(
              builder: (context, issueData, _) {
                if (issueData.issues.isEmpty) {
                  return Center(
                    child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                      Container(
                        width: 70, height: 70,
                        decoration: BoxDecoration(color: AppColors.surface2, borderRadius: BorderRadius.circular(20)),
                        child: Icon(Icons.notes_rounded, size: 36, color: AppColors.muted.withOpacity(0.5)),
                      ),
                      const SizedBox(height: 14),
                      Text('No issues yet!', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: AppColors.text)),
                      const SizedBox(height: 4),
                      Text('Help us improve by reporting one.', style: TextStyle(fontSize: 12, color: AppColors.muted)),
                    ]),
                  );
                }
                return ListView.builder(
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
                  itemCount: issueData.issues.length,
                  itemBuilder: (context, index) => _IssueCard(issue: issueData.issues[index]),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  void _showReportIssueDialog(BuildContext context) {
    final titleController = TextEditingController();
    final descriptionController = TextEditingController();

    showDialog(
      context: context,
      builder: (dialogContext) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        backgroundColor: AppColors.surface,
        title: const Text('Report an Issue', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: titleController,
              decoration: InputDecoration(
                labelText: 'Title',
                hintText: 'e.g., App slow on startup',
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                fillColor: AppColors.surface2,
                filled: true,
              ),
            ),
            const SizedBox(height: 14),
            TextField(
              controller: descriptionController,
              maxLines: 3,
              decoration: InputDecoration(
                labelText: 'Description',
                hintText: 'Describe the issue in detail...',
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                fillColor: AppColors.surface2,
                filled: true,
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext),
            child: Text('Cancel', style: TextStyle(color: AppColors.muted)),
          ),
          ElevatedButton(
            onPressed: () {
              if (titleController.text.isNotEmpty && descriptionController.text.isNotEmpty) {
                Provider.of<IssueData>(dialogContext, listen: false)
                    .addIssue(titleController.text, descriptionController.text);
                Navigator.pop(dialogContext);
                _showToast(context, 'Issue reported successfully!');
              } else {
                _showToast(context, 'Please fill all fields.');
              }
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.accent,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
            ),
            child: const Text('Submit', style: TextStyle(fontWeight: FontWeight.w700)),
          ),
        ],
      ),
    );
  }
}

class _IssueCard extends StatelessWidget {
  final Issue issue;
  const _IssueCard({required this.issue});

  @override
  Widget build(BuildContext context) {
    Color statusColor;
    Color statusBgColor;
    IconData statusIcon;

    switch (issue.status) {
      case 'Open':
        statusColor = AppColors.accent;
        statusBgColor = AppColors.accentBg;
        statusIcon = Icons.error_outline_rounded;
        break;
      case 'In Progress':
        statusColor = AppColors.yellow;
        statusBgColor = AppColors.yellowBg;
        statusIcon = Icons.hourglass_empty_rounded;
        break;
      case 'Resolved':
        statusColor = AppColors.green;
        statusBgColor = AppColors.greenBg;
        statusIcon = Icons.check_circle_outline_rounded;
        break;
      default:
        statusColor = AppColors.muted;
        statusBgColor = AppColors.surface2;
        statusIcon = Icons.info_outline_rounded;
    }

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.03), blurRadius: 6, offset: const Offset(0, 2))],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Text(issue.title,
                    style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: AppColors.text),
                    maxLines: 1, overflow: TextOverflow.ellipsis),
              ),
              const SizedBox(width: 8),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(color: statusBgColor, borderRadius: BorderRadius.circular(20)),
                child: Row(mainAxisSize: MainAxisSize.min, children: [
                  Icon(statusIcon, size: 13, color: statusColor),
                  const SizedBox(width: 4),
                  Text(issue.status, style: TextStyle(fontSize: 11, color: statusColor, fontWeight: FontWeight.w700)),
                ]),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(issue.description,
              style: TextStyle(fontSize: 13, color: AppColors.muted, height: 1.4),
              maxLines: 2, overflow: TextOverflow.ellipsis),
          const SizedBox(height: 10),
          Row(children: [
            Icon(Icons.calendar_today_outlined, size: 11, color: AppColors.muted.withOpacity(0.7)),
            const SizedBox(width: 4),
            Text('Reported: ${DateFormat('MMM dd, yyyy').format(issue.dateReported)}',
                style: TextStyle(fontSize: 10, color: AppColors.muted.withOpacity(0.7), fontWeight: FontWeight.w500)),
          ]),
        ],
      ),
    );
  }
}

// ── Toast Helper ──────────────────────────────────────────────────────────────

void _showToast(BuildContext context, String message) {
  ScaffoldMessenger.of(context).clearSnackBars();
  ScaffoldMessenger.of(context).showSnackBar(
    SnackBar(
      content: Row(children: [
        const Icon(Icons.check_circle_rounded, color: Colors.white, size: 16),
        const SizedBox(width: 8),
        Text(message, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13)),
      ]),
      backgroundColor: AppColors.green,
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
      duration: const Duration(seconds: 2),
      margin: const EdgeInsets.fromLTRB(20, 0, 20, 90),
    ),
  );
}
