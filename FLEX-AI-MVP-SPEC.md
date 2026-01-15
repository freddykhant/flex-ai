# FLEX-AI MVP Specification

> **Version:** 1.0  
> **Last Updated:** January 2025  
> **Stack:** Next.js (T3), Tailwind CSS, tRPC, Drizzle ORM, PostgreSQL, Vercel AI SDK

---

## ⚠️ IMPORTANT: CLAUDE CODE OPERATING MODE

**READ THIS FIRST — THIS IS CRITICAL**

Claude Code is acting as a **mentor and pair programmer**, NOT an autonomous code generator.

### Rules of Engagement:

1. **DO NOT write or generate code unless explicitly asked** (e.g., "write this for me", "generate the schema", "code this component")

2. **Default behavior should be:**
   - Explain concepts and approaches
   - Guide through implementation step-by-step
   - Ask clarifying questions
   - Review code I write and suggest improvements
   - Point out potential issues before they become problems
   - Teach best practices as we go

3. **When I ask for help, prefer:**
   - Pseudocode and explanations first
   - Small, focused code snippets to illustrate points
   - Walking me through the "why" not just the "what"

4. **You are an expert in:**
   - TypeScript (strict mode, advanced patterns, type safety)
   - Vercel AI SDK (streaming, tool calls, multi-model support)
   - T3 Stack (Next.js App Router, tRPC, Drizzle, NextAuth)
   - React patterns (hooks, state management, composition)
   - Tailwind CSS (utility-first, responsive design)

5. **Teaching style:**
   - Patient and thorough
   - Explain trade-offs between different approaches
   - Connect new concepts to things I already know
   - Celebrate wins, learn from mistakes together

**The goal is for ME to become a better developer, not just to ship code fast.**

---

## Project Overview

### What is Flex-AI?

An AI-powered fitness platform focused on hypertrophy, nutrition, and bodybuilding. The core experience is a knowledgeable chatbot that provides personalized advice, supported by meal planning and workout programming features.

### Target User

People serious about building muscle who want personalized, science-backed advice without the noise of generic fitness apps.

### Core Value Proposition

- Hypertrophy-focused (not generic "get fit" advice)
- Personalized to user's stats, goals, and experience
- Generates actionable meal plans and workout programs
- Single AI assistant that knows your context

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Framework | Next.js 14+ (App Router) |
| Styling | Tailwind CSS (no component library) |
| API | tRPC for CRUD, Next.js API routes for AI streaming |
| Database | PostgreSQL |
| ORM | Drizzle |
| Auth | NextAuth (Credentials + Google OAuth) |
| AI | Vercel AI SDK with Claude/GPT models |

### AI Model Strategy

```typescript
// Environment-based model selection
// Development: cheap model (gpt-4o-mini or claude-3-haiku)
// Production: quality model (claude-sonnet-4-20250514)

const model = process.env.NODE_ENV === 'production'
  ? anthropic('claude-sonnet-4-20250514')
  : openai('gpt-4o-mini');
```

---

## Database Schema

### user_profiles

Stores user fitness data and calculated TDEE. One-to-one with auth users table.

```typescript
// src/server/db/schema/userProfiles.ts

export const userProfiles = pgTable('user_profiles', {
  id: uuid('id').defaultRandom().primaryKey(),
  userId: uuid('user_id').notNull().unique().references(() => users.id, { onDelete: 'cascade' }),
  
  // Physical stats (stored in metric)
  heightCm: decimal('height_cm', { precision: 5, scale: 2 }).notNull(),
  weightKg: decimal('weight_kg', { precision: 5, scale: 2 }).notNull(),
  age: integer('age').notNull(),
  sex: pgEnum('sex', ['male', 'female']).notNull(),
  
  // Fitness context
  activityLevel: pgEnum('activity_level', [
    'sedentary',
    'lightly_active', 
    'moderately_active',
    'very_active',
    'extremely_active'
  ]).notNull(),
  experience: pgEnum('experience', ['beginner', 'intermediate', 'advanced']).notNull(),
  goal: pgEnum('goal', ['bulk', 'cut', 'recomp', 'weight_loss']).notNull(),
  
  // Calculated & preferences
  tdee: integer('tdee').notNull(),
  unitPreference: pgEnum('unit_preference', ['metric', 'imperial']).default('metric'),
  onboardingCompleted: boolean('onboarding_completed').default(false),
  
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow(),
});
```

### meal_plans

One meal plan per user (MVP). Stores generated markdown content plus metadata.

```typescript
// src/server/db/schema/mealPlans.ts

export const mealPlans = pgTable('meal_plans', {
  id: uuid('id').defaultRandom().primaryKey(),
  userId: uuid('user_id').notNull().unique().references(() => users.id, { onDelete: 'cascade' }),
  
  // Snapshot of settings when generated
  goalSnapshot: pgEnum('goal', ['bulk', 'cut', 'recomp', 'weight_loss']).notNull(),
  targetCalories: integer('target_calories').notNull(),
  proteinG: integer('protein_g').notNull(),
  carbsG: integer('carbs_g').notNull(),
  fatsG: integer('fats_g').notNull(),
  mealsPerDay: integer('meals_per_day').notNull(), // 3, 4, or 5
  weeklyChangeKg: decimal('weekly_change_kg', { precision: 4, scale: 2 }).notNull(),
  
  // The actual plan
  content: text('content').notNull(), // Markdown
  
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow(),
});
```

### workout_programs

One workout program per user (MVP). Stores generated markdown content plus metadata.

```typescript
// src/server/db/schema/workoutPrograms.ts

export const workoutPrograms = pgTable('workout_programs', {
  id: uuid('id').defaultRandom().primaryKey(),
  userId: uuid('user_id').notNull().unique().references(() => users.id, { onDelete: 'cascade' }),
  
  // Program settings
  splitType: pgEnum('split_type', [
    'upper_lower',
    'ppl',
    'bro_split', 
    'full_body',
    'anterior_posterior'
  ]).notNull(),
  volumeFrequency: pgEnum('volume_frequency', [
    'high_vol_low_freq',
    'low_vol_high_freq',
    'balanced'
  ]).notNull(),
  daysPerWeek: integer('days_per_week').notNull(),
  
  // The actual program
  content: text('content').notNull(), // Markdown
  
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow(),
});
```

### chat_messages

Single conversation per user (MVP). Messages stored after each exchange.

```typescript
// src/server/db/schema/chatMessages.ts

export const chatMessages = pgTable('chat_messages', {
  id: uuid('id').defaultRandom().primaryKey(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  
  role: pgEnum('role', ['user', 'assistant']).notNull(),
  content: text('content').notNull(),
  
  createdAt: timestamp('created_at').defaultNow(),
});

// Index for fetching user's chat history
export const chatMessagesUserIdIdx = index('chat_messages_user_id_idx').on(chatMessages.userId);
```

---

## Application Structure

### File Organization

```
src/
├── app/
│   ├── (auth)/
│   │   ├── login/page.tsx
│   │   └── register/page.tsx
│   ├── (main)/
│   │   ├── layout.tsx              # Sidebar + main + artifact panel
│   │   ├── chat/page.tsx           # Default landing
│   │   ├── meal-plan/page.tsx
│   │   ├── workout/page.tsx
│   │   └── profile/page.tsx
│   ├── onboarding/
│   │   ├── layout.tsx              # Minimal layout for onboarding
│   │   └── page.tsx                # Multi-step form
│   ├── api/
│   │   ├── chat/route.ts           # Vercel AI SDK streaming
│   │   ├── generate/
│   │   │   ├── meal-plan/route.ts
│   │   │   └── workout/route.ts
│   │   └── trpc/[trpc]/route.ts
│   ├── layout.tsx
│   └── page.tsx                    # Landing/marketing or redirect
│
├── components/
│   ├── layout/
│   │   ├── Sidebar.tsx
│   │   ├── MainContent.tsx
│   │   └── ArtifactPanel.tsx
│   ├── onboarding/
│   │   ├── StepBasicStats.tsx
│   │   ├── StepLifestyle.tsx
│   │   └── StepGoals.tsx
│   ├── chat/
│   │   ├── ChatContainer.tsx
│   │   ├── MessageList.tsx
│   │   ├── MessageBubble.tsx
│   │   └── ChatInput.tsx
│   ├── meal-plan/
│   │   ├── MealPlanForm.tsx
│   │   └── MealPlanArtifact.tsx
│   ├── workout/
│   │   ├── WorkoutForm.tsx
│   │   └── WorkoutArtifact.tsx
│   └── ui/
│       ├── Button.tsx
│       ├── Input.tsx
│       ├── Select.tsx
│       ├── RadioGroup.tsx
│       └── ... (shared UI primitives)
│
├── server/
│   ├── db/
│   │   ├── index.ts                # Drizzle client
│   │   └── schema/
│   │       ├── index.ts            # Export all schemas
│   │       ├── userProfiles.ts
│   │       ├── mealPlans.ts
│   │       ├── workoutPrograms.ts
│   │       └── chatMessages.ts
│   ├── api/
│   │   ├── root.ts                 # tRPC root router
│   │   ├── trpc.ts                 # tRPC setup
│   │   └── routers/
│   │       ├── profile.ts
│   │       ├── mealPlan.ts
│   │       ├── workoutProgram.ts
│   │       └── chat.ts
│   └── auth.ts                     # NextAuth config
│
├── lib/
│   ├── ai/
│   │   ├── index.ts                # Model configuration
│   │   ├── prompts/
│   │   │   ├── chat-system.ts      # Chat system prompt
│   │   │   ├── meal-plan.ts        # Meal plan generation prompt
│   │   │   └── workout.ts          # Workout generation prompt
│   │   └── context.ts              # Build user context for AI
│   ├── utils/
│   │   ├── tdee.ts                 # TDEE calculation
│   │   ├── units.ts                # Unit conversion helpers
│   │   └── macros.ts               # Macro calculation helpers
│   └── constants.ts                # App-wide constants
│
└── types/
    └── index.ts                    # Shared TypeScript types
```

---

## UI Layout

### Main Application Layout

```
┌────────────────┬────────────────────────────────┬─────────────────────────────┐
│                │                                │                             │
│  [FLEX LOGO]   │                                │                             │
│                │                                │                             │
│  ────────────  │                                │                             │
│                │                                │                             │
│  💬 Chat       │       Main Content             │      Artifact Panel         │
│                │       ────────────             │      ──────────────         │
│  🍽️ Meal Plan  │                                │                             │
│                │       - Chat messages          │      - Generated content    │
│  🏋️ Workout    │       - Generation forms       │      - Markdown rendered    │
│                │       - Profile editing        │      - [Save] button        │
│  👤 Profile    │                                │                             │
│                │                                │      (hidden when empty)    │
│                │                                │                             │
│                │                                │                             │
│  ────────────  │                                │                             │
│                │                                │                             │
│  [Avatar]      │                                │                             │
│  User Name     │                                │                             │
│  [Logout]      │                                │                             │
│                │                                │                             │
└────────────────┴────────────────────────────────┴─────────────────────────────┘
     ~200px              flexible                        ~400px (or hidden)
```

### Sidebar States

- **Expanded:** Icon + label for each nav item
- **Logo:** Flex branding at top
- **User section:** Avatar, display name, logout button at bottom
- **Active state:** Highlight current route

### Artifact Panel Behavior

- **Hidden:** When on Chat page with no generated content
- **Visible:** When on Meal Plan or Workout pages
- **Content:** Rendered markdown with save button
- **Collapsible:** User can collapse to focus on form/chat

---

## Feature Specifications

### Onboarding Flow

**Route:** `/onboarding`

**Redirect Logic:**
- After auth, check `userProfiles.onboardingCompleted`
- If `false` or no profile → redirect to `/onboarding`
- If `true` → redirect to `/chat`

**Step 1: Basic Stats**
```
┌─────────────────────────────────────┐
│  Step 1 of 3: Basic Stats           │
│  ─────────────────────────────────  │
│                                     │
│  Height                             │
│  [___________] [cm ▼] / [ft/in ▼]  │
│                                     │
│  Weight                             │
│  [___________] [kg ▼] / [lbs ▼]    │
│                                     │
│  Age                                │
│  [___________]                      │
│                                     │
│  Biological Sex                     │
│  ○ Male  ○ Female                   │
│                                     │
│              [Next →]               │
└─────────────────────────────────────┘
```

**Step 2: Lifestyle & Experience**
```
┌─────────────────────────────────────┐
│  Step 2 of 3: Lifestyle             │
│  ─────────────────────────────────  │
│                                     │
│  Activity Level                     │
│  ○ Sedentary                        │
│    Desk job, little to no exercise  │
│  ○ Lightly Active                   │
│    Light exercise 1-3 days/week     │
│  ○ Moderately Active                │
│    Moderate exercise 3-5 days/week  │
│  ○ Very Active                      │
│    Hard exercise 6-7 days/week      │
│  ○ Extremely Active                 │
│    Physical job + hard exercise     │
│                                     │
│  Training Experience                │
│  ○ Beginner (<1 year consistent)    │
│  ○ Intermediate (1-3 years)         │
│  ○ Advanced (3+ years)              │
│                                     │
│        [← Back]    [Next →]         │
└─────────────────────────────────────┘
```

**Step 3: Goal Selection**
```
┌─────────────────────────────────────┐
│  Step 3 of 3: Your Goal             │
│  ─────────────────────────────────  │
│                                     │
│  What's your primary goal?          │
│                                     │
│  ┌─────────────────────────────┐    │
│  │ 💪 Bulk                     │    │
│  │ Build muscle, gain weight   │    │
│  │ Caloric surplus             │    │
│  └─────────────────────────────┘    │
│                                     │
│  ┌─────────────────────────────┐    │
│  │ 🔥 Cut                      │    │
│  │ Lose fat, preserve muscle   │    │
│  │ Caloric deficit             │    │
│  └─────────────────────────────┘    │
│                                     │
│  ┌─────────────────────────────┐    │
│  │ ⚖️ Recomposition            │    │
│  │ Build muscle while losing   │    │
│  │ fat (maintenance calories)  │    │
│  └─────────────────────────────┘    │
│                                     │
│  ┌─────────────────────────────┐    │
│  │ 📉 Weight Loss              │    │
│  │ General weight loss         │    │
│  │ (less muscle-focused)       │    │
│  └─────────────────────────────┘    │
│                                     │
│        [← Back]    [Complete →]     │
└─────────────────────────────────────┘
```

**On Complete:**
1. Calculate TDEE using Mifflin-St Jeor formula
2. Save profile to database
3. Set `onboardingCompleted = true`
4. Redirect to `/chat`

---

### TDEE Calculation

**Formula:** Mifflin-St Jeor

```typescript
// lib/utils/tdee.ts

type Sex = 'male' | 'female';
type ActivityLevel = 'sedentary' | 'lightly_active' | 'moderately_active' | 'very_active' | 'extremely_active';

const ACTIVITY_MULTIPLIERS: Record<ActivityLevel, number> = {
  sedentary: 1.2,
  lightly_active: 1.375,
  moderately_active: 1.55,
  very_active: 1.725,
  extremely_active: 1.9,
};

export function calculateTDEE(
  weightKg: number,
  heightCm: number,
  age: number,
  sex: Sex,
  activityLevel: ActivityLevel
): number {
  // Mifflin-St Jeor BMR
  let bmr: number;
  
  if (sex === 'male') {
    bmr = (10 * weightKg) + (6.25 * heightCm) - (5 * age) + 5;
  } else {
    bmr = (10 * weightKg) + (6.25 * heightCm) - (5 * age) - 161;
  }
  
  // Apply activity multiplier
  const tdee = Math.round(bmr * ACTIVITY_MULTIPLIERS[activityLevel]);
  
  return tdee;
}
```

---

### Chat Feature

**Route:** `/chat`

**Behavior:**
- Single persistent conversation per user
- Messages stream in real-time
- Context includes user profile, TDEE, goal, experience
- Chat history loaded on mount, saved after each exchange
- Refuses PED-related advice

**System Prompt Context:**
```typescript
// lib/ai/context.ts

export function buildUserContext(profile: UserProfile) {
  return `
## User Profile
- Height: ${profile.heightCm}cm
- Weight: ${profile.weightKg}kg  
- Age: ${profile.age}
- Sex: ${profile.sex}
- Activity Level: ${profile.activityLevel.replace('_', ' ')}
- Training Experience: ${profile.experience}
- Current Goal: ${profile.goal}
- Maintenance Calories (TDEE): ${profile.tdee} kcal/day
`;
}
```

**API Route Structure:**
```typescript
// app/api/chat/route.ts

import { streamText } from 'ai';
import { getModel } from '@/lib/ai';
import { CHAT_SYSTEM_PROMPT } from '@/lib/ai/prompts/chat-system';
import { buildUserContext } from '@/lib/ai/context';

export async function POST(req: Request) {
  const { messages } = await req.json();
  
  // Get user profile from session/db
  const profile = await getUserProfile();
  const userContext = buildUserContext(profile);
  
  const result = await streamText({
    model: getModel(),
    system: CHAT_SYSTEM_PROMPT + userContext,
    messages,
  });
  
  return result.toDataStreamResponse();
}
```

---

### Meal Plan Generator

**Route:** `/meal-plan`

**Form Inputs:**
- Goal (display only, from profile)
- Rate of change:
  - Bulk: +0.25kg/week, +0.5kg/week
  - Cut: -0.25kg/week, -0.5kg/week, -0.75kg/week
  - Recomp: Maintenance
  - Weight Loss: -0.25kg/week, -0.5kg/week, -0.75kg/week
- Meals per day: 3 / 4 / 5

**Calorie Calculation:**
```typescript
// lib/utils/macros.ts

export function calculateTargetCalories(
  tdee: number,
  goal: Goal,
  weeklyChangeKg: number
): number {
  // 1kg of body weight ≈ 7700 kcal
  const dailyChange = (weeklyChangeKg * 7700) / 7;
  
  switch (goal) {
    case 'bulk':
      return Math.round(tdee + dailyChange);
    case 'cut':
    case 'weight_loss':
      return Math.round(tdee - Math.abs(dailyChange));
    case 'recomp':
      return tdee;
  }
}

export function calculateMacros(
  targetCalories: number,
  weightKg: number,
  goal: Goal
): { protein: number; carbs: number; fats: number } {
  // Protein: 1.6-2.2g per kg bodyweight (higher end for cutting)
  const proteinMultiplier = goal === 'cut' ? 2.2 : 1.8;
  const protein = Math.round(weightKg * proteinMultiplier);
  
  // Fat: 25-30% of calories
  const fatCalories = targetCalories * 0.25;
  const fats = Math.round(fatCalories / 9);
  
  // Carbs: remaining calories
  const proteinCalories = protein * 4;
  const remainingCalories = targetCalories - proteinCalories - fatCalories;
  const carbs = Math.round(remainingCalories / 4);
  
  return { protein, carbs, fats };
}
```

**Generated Output (Markdown):**
```markdown
# Your 7-Day Meal Plan

## Overview
- **Daily Calories:** 2,450 kcal
- **Protein:** 165g (27%)
- **Carbohydrates:** 280g (46%)  
- **Fats:** 75g (27%)
- **Meals Per Day:** 4

---

## Day 1

### Meal 1: Breakfast (620 kcal)
- Oats, 80g (rolled, cooked with water)
- Whole eggs, 3 (scrambled)
- Banana, 1 medium
- Peanut butter, 1 tbsp

### Meal 2: Lunch (650 kcal)
- Chicken breast, 200g (grilled)
- White rice, 150g (cooked)
- Broccoli, 100g (steamed)
- Olive oil, 1 tbsp (drizzled)

... [continues for all meals and days]
```

---

### Workout Program Generator

**Route:** `/workout`

**Form Inputs:**
- Split preference:
  - Upper/Lower → 4 days/week
  - PPL (Push/Pull/Legs) → 6 days/week
  - Bro Split → 5 days/week
  - Full Body → 3 days/week
  - Anterior/Posterior → 4 days/week
- Volume/Frequency:
  - High volume / Low frequency
  - Low volume / High frequency
  - Balanced (default)

**Generated Output (Markdown):**
```markdown
# Your 4-Week Upper/Lower Program

## Program Overview
- **Split:** Upper/Lower
- **Days Per Week:** 4 (Upper, Lower, Rest, Upper, Lower, Rest, Rest)
- **Training Style:** Balanced volume and frequency
- **Experience Level:** Intermediate

## Progressive Overload Protocol
- Weeks 1-3: Add 1 rep per set when possible
- When you hit the top of rep range for all sets, increase weight by 2.5-5kg
- Week 4: Deload - reduce weight by 10%, focus on technique

---

## Upper Day A

| Exercise | Sets | Reps | RPE | Notes |
|----------|------|------|-----|-------|
| Horizontal Press - Barbell Bench Press | 4 | 6-8 | 8 | Control the eccentric |
| Horizontal Row - Barbell Row | 4 | 8-10 | 8 | Full stretch at bottom |
| Vertical Press - Overhead Press | 3 | 8-10 | 7-8 | Strict form, no leg drive |
| Vertical Pull - Lat Pulldown | 3 | 10-12 | 8 | Full stretch at top |
| Bicep Curl - Dumbbell Curl | 3 | 10-12 | 8 | |
| Tricep Extension - Cable Pushdown | 3 | 12-15 | 8 | |

... [continues for all days]
```

---

### Profile Page

**Route:** `/profile`

**Sections:**
1. **Personal Stats** (editable)
   - Height, Weight, Age, Sex
   - Activity Level, Experience
   - Goal
   - Unit preference toggle

2. **Calculated Stats** (read-only display)
   - Current TDEE
   - Suggested calories based on goal

3. **Saved Plans** (links)
   - View saved meal plan (or "No plan saved")
   - View saved workout program (or "No program saved")

**On Update:**
- Recalculate TDEE
- Show confirmation toast
- Note: Does NOT automatically update saved plans (they're snapshots)

---

## AI System Prompts

### Chat System Prompt

```typescript
// lib/ai/prompts/chat-system.ts

export const CHAT_SYSTEM_PROMPT = `
You are Flex, an expert AI fitness coach specializing in hypertrophy training, nutrition, and bodybuilding. You have deep knowledge of:

- Resistance training programming (volume, intensity, frequency, periodization)
- Muscle hypertrophy science (mechanical tension, metabolic stress, muscle damage)
- Nutrition for body composition (calories, macros, meal timing, supplements)
- Recovery and lifestyle factors (sleep, stress, deloads)
- Exercise technique and selection

## Your Personality
- Knowledgeable but approachable
- Direct and practical - give actionable advice
- Evidence-based but not overly academic
- Encouraging without being cheesy
- Honest about limitations and individual variation

## Guidelines
1. Always consider the user's profile (stats, goal, experience) when giving advice
2. Be specific with recommendations (numbers, sets, reps, grams)
3. Explain the "why" briefly when it helps understanding
4. If asked about something outside your expertise, say so
5. Encourage consistency over perfection

## Hard Rules
- NEVER provide advice on performance-enhancing drugs (PEDs), anabolic steroids, SARMs, or similar substances
- If asked about PEDs, politely decline and redirect to natural training methods
- Example response: "I focus on natural training methods and nutrition. There's a lot we can optimize with your programming, diet, and recovery before considering anything else. Let's talk about [relevant natural approach]."

## Context
The user's profile information will be provided below. Use this to personalize your responses.
`;
```

### Meal Plan Generation Prompt

```typescript
// lib/ai/prompts/meal-plan.ts

export const MEAL_PLAN_PROMPT = `
Generate a detailed 7-day meal plan based on the user's specifications.

## Requirements
- Daily calorie target: {{targetCalories}} kcal
- Macros: {{proteinG}}g protein, {{carbsG}}g carbs, {{fatsG}}g fat
- Meals per day: {{mealsPerDay}}
- Goal: {{goal}}

## Format
Use the following markdown structure:

# Your 7-Day Meal Plan

## Overview
- Daily Calories, Protein, Carbs, Fats with percentages

## Day 1
### Meal 1: [Meal Name] (X kcal)
- Food item, portion (preparation method)
- Food item, portion

[Continue for all meals and all 7 days]

## Guidelines
1. Use common, accessible foods
2. Include portion sizes in grams or standard measures
3. Add brief preparation notes (e.g., "grilled", "steamed", "air fried")
4. Prioritize protein sources in each meal
5. Vary foods across days for adherence
6. Consider practical meal prep (some repetition is fine)
7. Round calorie/macro totals to reasonable numbers

## Protein Sources to Include
Chicken breast, lean beef, fish, eggs, Greek yogurt, cottage cheese, whey protein

## Carb Sources to Include  
Rice, oats, potatoes, sweet potatoes, bread, pasta, fruits

## Fat Sources to Include
Olive oil, nuts, avocado, egg yolks, fatty fish
`;
```

### Workout Generation Prompt

```typescript
// lib/ai/prompts/workout.ts

export const WORKOUT_PROMPT = `
Generate a 4-week hypertrophy training program based on the user's specifications.

## Requirements
- Split: {{splitType}}
- Days per week: {{daysPerWeek}}
- Volume/Frequency preference: {{volumeFrequency}}
- Experience level: {{experience}}

## Format
Use the following markdown structure:

# Your 4-Week {{splitType}} Program

## Program Overview
- Split, Days Per Week, Training Style, Experience Level

## Progressive Overload Protocol
- Week-by-week progression guidance
- When to increase weight
- Deload recommendations

## [Day Name]

| Exercise | Sets | Reps | RPE | Notes |
|----------|------|------|-----|-------|
| Movement Pattern - Specific Exercise | X | X-X | X | Brief cue |

[Continue for all training days]

## Guidelines
1. Format exercises as "Movement Pattern - Specific Exercise Example"
   - Examples: "Horizontal Press - Barbell Bench Press", "Hip Hinge - Romanian Deadlift"
2. Include sets, reps (as ranges like 8-10), and RPE (7-9 range typically)
3. Add brief technique notes where helpful
4. Balance compound and isolation movements
5. Ensure adequate volume per muscle group per week
6. Consider recovery between sessions for same muscle groups

## Movement Patterns to Include
- Horizontal Press (bench press, dumbbell press, push-up variations)
- Horizontal Pull (rows - barbell, dumbbell, cable, machine)
- Vertical Press (overhead press, dumbbell shoulder press)
- Vertical Pull (pull-ups, lat pulldown)
- Hip Hinge (deadlift, RDL, hip thrust)
- Squat (back squat, front squat, leg press)
- Single-leg (lunges, split squats, step-ups)
- Isolation (curls, extensions, raises, etc.)

## Volume Guidelines by Experience
- Beginner: 10-14 sets per muscle group per week
- Intermediate: 14-18 sets per muscle group per week  
- Advanced: 18-22+ sets per muscle group per week
`;
```

---

## API Routes

### tRPC Routers

```typescript
// server/api/routers/profile.ts
- getProfile(): Get current user's profile
- createProfile(data): Create profile (onboarding)
- updateProfile(data): Update profile, recalculate TDEE

// server/api/routers/mealPlan.ts  
- getMealPlan(): Get user's saved meal plan (or null)
- saveMealPlan(data): Save/overwrite meal plan

// server/api/routers/workoutProgram.ts
- getWorkoutProgram(): Get user's saved program (or null)
- saveWorkoutProgram(data): Save/overwrite program

// server/api/routers/chat.ts
- getMessages(): Get user's chat history
- saveMessage(data): Save a single message
```

### Next.js API Routes (AI)

```typescript
// app/api/chat/route.ts
POST - Stream chat response

// app/api/generate/meal-plan/route.ts  
POST - Stream meal plan generation

// app/api/generate/workout/route.ts
POST - Stream workout program generation
```

---

## Implementation Phases

### Phase 1: Foundation
1. Drizzle schema setup
2. tRPC router stubs
3. Layout components (Sidebar, MainContent, ArtifactPanel)

### Phase 2: Onboarding
4. Multi-step onboarding form
5. TDEE calculation utility
6. Auth middleware (redirect logic)

### Phase 3: Core Features  
7. Chat with Vercel AI SDK
8. Meal plan generator
9. Workout program generator

### Phase 4: Polish
10. Profile page
11. System prompt refinement
12. Error handling, loading states

---

## Environment Variables

```env
# Database
DATABASE_URL=

# Auth
NEXTAUTH_SECRET=
NEXTAUTH_URL=
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=

# AI (add based on chosen provider)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
```

---

## Notes for Development

### Key Decisions Made
- Single conversation per user (MVP simplicity)
- One saved plan per type (meal plan, workout)
- Markdown for generated content (renders nicely, easy to store)
- TDEE calculated and stored server-side
- Unit preference stored, conversion on display

### Out of Scope for MVP
- Multiple saved plans
- Multiple chat conversations
- Dietary restrictions/allergies
- Custom exercise selection
- Social features
- Mobile app

### Future Considerations
- Plan history/versioning
- AI-powered plan modifications via chat
- Progress tracking
- Integration with fitness trackers
- Premium tier with advanced features
