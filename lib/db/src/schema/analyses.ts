import { pgTable, serial, text, integer, real, boolean, timestamp, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";
import { patientsTable } from "./patients";

export const analysesTable = pgTable("analyses", {
  id: serial("id").primaryKey(),
  patientId: integer("patient_id").notNull().references(() => patientsTable.id),
  eyeSide: text("eye_side").notNull(),
  imageName: text("image_name").notNull(),
  imageUrl: text("image_url"),
  imageQualityScore: real("image_quality_score").notNull(),
  imageQualityOk: boolean("image_quality_ok").notNull(),
  overallRisk: text("overall_risk").notNull(),
  diseases: jsonb("diseases").notNull().$type<Array<{
    name: string;
    detected: boolean;
    confidence: number;
    severity: string;
    findings: string;
  }>>(),
  summary: text("summary").notNull(),
  recommendations: text("recommendations").notNull(),
  analysisTime: real("analysis_time").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertAnalysisSchema = createInsertSchema(analysesTable).omit({ id: true, createdAt: true });
export type InsertAnalysis = z.infer<typeof insertAnalysisSchema>;
export type Analysis = typeof analysesTable.$inferSelect;
