import { Router, type IRouter } from "express";
import { db, patientsTable, analysesTable } from "@workspace/db";
import { eq } from "drizzle-orm";

const router: IRouter = Router();

router.get("/patients", async (_req, res) => {
  try {
    const patients = await db.select().from(patientsTable).orderBy(patientsTable.createdAt);
    res.json(patients);
  } catch (err) {
    res.status(500).json({ error: "Failed to list patients" });
  }
});

router.post("/patients", async (req, res) => {
  try {
    const { name, age, gender } = req.body;
    if (!name || !age || !gender) {
      return res.status(400).json({ error: "name, age, gender are required" });
    }
    const [patient] = await db.insert(patientsTable).values({ name, age: Number(age), gender }).returning();
    res.status(201).json(patient);
  } catch (err) {
    res.status(500).json({ error: "Failed to create patient" });
  }
});

router.get("/patients/:patientId", async (req, res) => {
  try {
    const id = Number(req.params.patientId);
    const [patient] = await db.select().from(patientsTable).where(eq(patientsTable.id, id));
    if (!patient) return res.status(404).json({ error: "Patient not found" });
    res.json(patient);
  } catch (err) {
    res.status(500).json({ error: "Failed to get patient" });
  }
});

router.get("/patients/:patientId/analyses", async (req, res) => {
  try {
    const patientId = Number(req.params.patientId);
    const analyses = await db.select().from(analysesTable).where(eq(analysesTable.patientId, patientId)).orderBy(analysesTable.createdAt);
    res.json(analyses);
  } catch (err) {
    res.status(500).json({ error: "Failed to list analyses" });
  }
});

export default router;
