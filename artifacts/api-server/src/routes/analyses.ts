import { Router, type IRouter, type Request } from "express";
import multer from "multer";
import { db, analysesTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import { openai } from "@workspace/integrations-openai-ai-server";

const router: IRouter = Router();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } });

const DISEASES = [
  "Diabetic Retinopathy",
  "Age-related Macular Degeneration (Dry)",
  "Age-related Macular Degeneration (Wet)",
  "Retinal Vein Occlusion",
  "Glaucoma (Suspicious)",
  "Epiretinal Membrane",
  "Retinal Detachment",
  "Macular Hole",
  "Retinitis Pigmentosa",
  "Central Serous Chorioretinopathy",
  "Optic Atrophy",
  "Retinal Artery Occlusion",
  "Pathologic Myopia",
];

async function runAiAnalysis(imageData: string, patientId: number, eyeSide: string, imageName: string, startTime: number) {
  const prompt = `You are an expert AI ophthalmologist specializing in retinal fundus image analysis. Analyze this retinal fundus image and provide a comprehensive diagnosis report.

Analyze the image for the following 13 retinal disorders:
${DISEASES.map((d, i) => `${i + 1}. ${d}`).join("\n")}

Return ONLY a valid JSON object with this exact structure:
{
  "imageQualityScore": <number 0-100>,
  "imageQualityOk": <boolean, true if score >= 60>,
  "diseases": [
    {
      "name": "<disease name exactly as listed above>",
      "detected": <boolean>,
      "confidence": <number 0-100>,
      "severity": "<none|mild|moderate|severe>",
      "findings": "<specific clinical findings observed or 'No significant findings'>"
    }
  ],
  "overallRisk": "<normal|low|moderate|high|critical>",
  "summary": "<2-3 sentence professional clinical summary of findings>",
  "recommendations": "<professional clinical recommendations for follow-up care>"
}

Be clinically accurate based on what you can observe in the image. Provide all 13 diseases in the response.`;

  const response = await openai.chat.completions.create({
    model: "gpt-5.2",
    max_completion_tokens: 4096,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image_url",
            image_url: { url: imageData, detail: "high" },
          },
          { type: "text", text: prompt },
        ],
      },
    ],
  });

  const content = response.choices[0]?.message?.content ?? "{}";

  let parsed: any = {};
  try {
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (jsonMatch) parsed = JSON.parse(jsonMatch[0]);
  } catch {
    parsed = {};
  }

  const diseases = Array.isArray(parsed.diseases)
    ? parsed.diseases.map((d: any) => ({
        name: d.name ?? "Unknown",
        detected: Boolean(d.detected),
        confidence: Number(d.confidence ?? 0),
        severity: d.severity ?? "none",
        findings: d.findings ?? "No findings",
      }))
    : DISEASES.map((name) => ({
        name,
        detected: false,
        confidence: 0,
        severity: "none",
        findings: "Analysis unavailable",
      }));

  const imageQualityScore = Number(parsed.imageQualityScore ?? 75);
  const imageQualityOk = Boolean(parsed.imageQualityOk ?? imageQualityScore >= 60);
  const overallRisk = parsed.overallRisk ?? "normal";
  const summary = parsed.summary ?? "AI analysis completed.";
  const recommendations = parsed.recommendations ?? "Please consult with an ophthalmologist.";
  const analysisTime = (Date.now() - startTime) / 1000;

  const [analysis] = await db.insert(analysesTable).values({
    patientId: Number(patientId),
    eyeSide,
    imageName,
    imageUrl: imageData.substring(0, 500),
    imageQualityScore,
    imageQualityOk,
    overallRisk,
    diseases,
    summary,
    recommendations,
    analysisTime,
  }).returning();

  return { ...analysis, imageUrl: imageData };
}

router.post("/analyses", upload.none(), async (req: Request, res: any) => {
  const startTime = Date.now();
  try {
    // Handle both multipart/form-data (from generated client) and JSON
    const body = req.body;
    const patientId = body.patientId;
    const imageBase64 = body.imageBase64;
    const imageName = body.imageName;
    const eyeSide = body.eyeSide;

    if (!patientId || !imageBase64 || !imageName || !eyeSide) {
      return res.status(400).json({ error: "patientId, imageBase64, imageName, eyeSide are required" });
    }

    const imageData = imageBase64.startsWith("data:") ? imageBase64 : `data:image/jpeg;base64,${imageBase64}`;

    const result = await runAiAnalysis(imageData, Number(patientId), eyeSide, imageName, startTime);
    res.status(201).json(result);
  } catch (err: any) {
    console.error("Analysis error:", err);
    res.status(500).json({ error: `Analysis failed: ${err.message}` });
  }
});

router.get("/analyses/:analysisId", async (req: Request, res: any) => {
  try {
    const id = Number(req.params.analysisId);
    const [analysis] = await db.select().from(analysesTable).where(eq(analysesTable.id, id));
    if (!analysis) return res.status(404).json({ error: "Analysis not found" });
    res.json(analysis);
  } catch (err) {
    res.status(500).json({ error: "Failed to get analysis" });
  }
});

export default router;
