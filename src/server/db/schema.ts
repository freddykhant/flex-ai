import { relations } from "drizzle-orm";
import {
  boolean,
  index,
  integer,
  numeric,
  pgEnum,
  pgTableCreator,
  primaryKey,
  text,
  timestamp,
} from "drizzle-orm/pg-core";
import { type AdapterAccount } from "next-auth/adapters";

/**
 * This is an example of how to use the multi-project schema feature of Drizzle ORM. Use the same
 * database instance for multiple projects.
 *
 * @see https://orm.drizzle.team/docs/goodies#multi-project-schema
 */
export const createTable = pgTableCreator((name) => `flex-ai_${name}`);

/**
 * Enums for Flex-AI fitness app
 */
export const sexEnum = pgEnum("sex", ["male", "female"]);

export const activityLevelEnum = pgEnum("activity_level", [
  "sedentary",
  "lightly_active",
  "moderately_active",
  "very_active",
  "extremely_active",
]);

export const experienceEnum = pgEnum("experience", [
  "beginner",
  "intermediate",
  "advanced",
]);

export const goalEnum = pgEnum("goal", ["bulk", "cut", "recomp", "weight_loss"]);

export const unitPreferenceEnum = pgEnum("unit_preference", [
  "metric",
  "imperial",
]);

export const splitTypeEnum = pgEnum("split_type", [
  "upper_lower",
  "ppl",
  "bro_split",
  "full_body",
  "anterior_posterior",
]);

export const volumeFrequencyEnum = pgEnum("volume_frequency", [
  "high_vol_low_freq",
  "low_vol_high_freq",
  "balanced",
]);

export const roleEnum = pgEnum("role", ["user", "assistant"]);

export const posts = createTable(
  "post",
  (d) => ({
    id: d.integer().primaryKey().generatedByDefaultAsIdentity(),
    name: d.varchar({ length: 256 }),
    createdById: d
      .varchar({ length: 255 })
      .notNull()
      .references(() => users.id),
    createdAt: d
      .timestamp({ withTimezone: true })
      .$defaultFn(() => /* @__PURE__ */ new Date())
      .notNull(),
    updatedAt: d.timestamp({ withTimezone: true }).$onUpdate(() => new Date()),
  }),
  (t) => [
    index("created_by_idx").on(t.createdById),
    index("name_idx").on(t.name),
  ],
);

export const users = createTable("user", (d) => ({
  id: d
    .varchar({ length: 255 })
    .notNull()
    .primaryKey()
    .$defaultFn(() => crypto.randomUUID()),
  name: d.varchar({ length: 255 }),
  email: d.varchar({ length: 255 }).notNull(),
  emailVerified: d
    .timestamp({
      mode: "date",
      withTimezone: true,
    })
    .$defaultFn(() => /* @__PURE__ */ new Date()),
  image: d.varchar({ length: 255 }),
}));

export const usersRelations = relations(users, ({ one, many }) => ({
  accounts: many(accounts),
  profile: one(userProfiles),
  mealPlan: one(mealPlans),
  workoutProgram: one(workoutPrograms),
  chatMessages: many(chatMessages),
}));

export const accounts = createTable(
  "account",
  (d) => ({
    userId: d
      .varchar({ length: 255 })
      .notNull()
      .references(() => users.id),
    type: d.varchar({ length: 255 }).$type<AdapterAccount["type"]>().notNull(),
    provider: d.varchar({ length: 255 }).notNull(),
    providerAccountId: d.varchar({ length: 255 }).notNull(),
    refresh_token: d.text(),
    access_token: d.text(),
    expires_at: d.integer(),
    token_type: d.varchar({ length: 255 }),
    scope: d.varchar({ length: 255 }),
    id_token: d.text(),
    session_state: d.varchar({ length: 255 }),
  }),
  (t) => [
    primaryKey({ columns: [t.provider, t.providerAccountId] }),
    index("account_user_id_idx").on(t.userId),
  ],
);

export const accountsRelations = relations(accounts, ({ one }) => ({
  user: one(users, { fields: [accounts.userId], references: [users.id] }),
}));

export const sessions = createTable(
  "session",
  (d) => ({
    sessionToken: d.varchar({ length: 255 }).notNull().primaryKey(),
    userId: d
      .varchar({ length: 255 })
      .notNull()
      .references(() => users.id),
    expires: d.timestamp({ mode: "date", withTimezone: true }).notNull(),
  }),
  (t) => [index("t_user_id_idx").on(t.userId)],
);

export const sessionsRelations = relations(sessions, ({ one }) => ({
  user: one(users, { fields: [sessions.userId], references: [users.id] }),
}));

export const verificationTokens = createTable(
  "verification_token",
  (d) => ({
    identifier: d.varchar({ length: 255 }).notNull(),
    token: d.varchar({ length: 255 }).notNull(),
    expires: d.timestamp({ mode: "date", withTimezone: true }).notNull(),
  }),
  (t) => [primaryKey({ columns: [t.identifier, t.token] })],
);

// ─── Flex-AI feature tables ───────────────────────────────────────────────────

export const userProfiles = createTable("user_profile", (d) => ({
  id: d
    .varchar({ length: 255 })
    .notNull()
    .primaryKey()
    .$defaultFn(() => crypto.randomUUID()),
  userId: d
    .varchar({ length: 255 })
    .notNull()
    .unique()
    .references(() => users.id, { onDelete: "cascade" }),

  // Physical stats (stored in metric)
  heightCm: numeric("height_cm", { precision: 5, scale: 2 }).notNull(),
  weightKg: numeric("weight_kg", { precision: 5, scale: 2 }).notNull(),
  age: integer("age").notNull(),
  sex: sexEnum("sex").notNull(),

  // Fitness context
  activityLevel: activityLevelEnum("activity_level").notNull(),
  experience: experienceEnum("experience").notNull(),
  goal: goalEnum("goal").notNull(),

  // Calculated & preferences
  tdee: integer("tdee").notNull(),
  unitPreference: unitPreferenceEnum("unit_preference").default("metric"),
  onboardingCompleted: boolean("onboarding_completed").default(false),

  createdAt: timestamp("created_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .$onUpdate(() => new Date()),
}));

export const userProfilesRelations = relations(userProfiles, ({ one }) => ({
  user: one(users, { fields: [userProfiles.userId], references: [users.id] }),
}));

export const mealPlans = createTable("meal_plan", (d) => ({
  id: d
    .varchar({ length: 255 })
    .notNull()
    .primaryKey()
    .$defaultFn(() => crypto.randomUUID()),
  userId: d
    .varchar({ length: 255 })
    .notNull()
    .unique()
    .references(() => users.id, { onDelete: "cascade" }),

  // Snapshot of settings when generated
  goalSnapshot: goalEnum("goal_snapshot").notNull(),
  targetCalories: integer("target_calories").notNull(),
  proteinG: integer("protein_g").notNull(),
  carbsG: integer("carbs_g").notNull(),
  fatsG: integer("fats_g").notNull(),
  mealsPerDay: integer("meals_per_day").notNull(),
  weeklyChangeKg: numeric("weekly_change_kg", { precision: 4, scale: 2 }).notNull(),

  // The actual plan
  content: text("content").notNull(),

  createdAt: timestamp("created_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .$onUpdate(() => new Date()),
}));

export const mealPlansRelations = relations(mealPlans, ({ one }) => ({
  user: one(users, { fields: [mealPlans.userId], references: [users.id] }),
}));

export const workoutPrograms = createTable("workout_program", (d) => ({
  id: d
    .varchar({ length: 255 })
    .notNull()
    .primaryKey()
    .$defaultFn(() => crypto.randomUUID()),
  userId: d
    .varchar({ length: 255 })
    .notNull()
    .unique()
    .references(() => users.id, { onDelete: "cascade" }),

  // Program settings
  splitType: splitTypeEnum("split_type").notNull(),
  volumeFrequency: volumeFrequencyEnum("volume_frequency").notNull(),
  daysPerWeek: integer("days_per_week").notNull(),

  // The actual program
  content: text("content").notNull(),

  createdAt: timestamp("created_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true })
    .$defaultFn(() => new Date())
    .$onUpdate(() => new Date()),
}));

export const workoutProgramsRelations = relations(workoutPrograms, ({ one }) => ({
  user: one(users, { fields: [workoutPrograms.userId], references: [users.id] }),
}));

export const chatMessages = createTable(
  "chat_message",
  (d) => ({
    id: d
      .varchar({ length: 255 })
      .notNull()
      .primaryKey()
      .$defaultFn(() => crypto.randomUUID()),
    userId: d
      .varchar({ length: 255 })
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),

    role: roleEnum("role").notNull(),
    content: text("content").notNull(),

    createdAt: timestamp("created_at", { withTimezone: true })
      .$defaultFn(() => new Date())
      .notNull(),
  }),
  (t) => [index("chat_messages_user_id_idx").on(t.userId)],
);

export const chatMessagesRelations = relations(chatMessages, ({ one }) => ({
  user: one(users, { fields: [chatMessages.userId], references: [users.id] }),
}));
