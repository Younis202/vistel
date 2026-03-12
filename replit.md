# AI Eye Diagnosis (EyeWisdom Clone)

## Overview

Full-stack AI-powered retinal fundus image diagnosis system, inspired by Vistel's EyeWisdom platform. Analyzes fundus retinal images using GPT-5.2 vision and generates professional medical diagnostic reports for 13 retinal diseases.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)
- **Frontend**: React + Vite + Tailwind CSS + shadcn/ui
- **AI**: OpenAI GPT-5.2 vision via Replit AI Integrations

## Features

- Patient management (create, list, view)
- Fundus image upload with drag-and-drop
- AI analysis pipeline (quality check → lesion detection → disease classification)
- 13 retinal disease detection: DR, AMD (Dry/Wet), RVO, Glaucoma, ERM, Retinal Detachment, Macular Hole, RP, CSC, Optic Atrophy, RAO, Pathologic Myopia
- Professional medical report generation
- Analysis history per patient
- Risk stratification (normal/low/moderate/high/critical)

## Structure

```text
artifacts-monorepo/
├── artifacts/
│   ├── api-server/         # Express API server (backend)
│   └── eye-diagnosis/      # React + Vite frontend (at /)
├── lib/
│   ├── api-spec/           # OpenAPI spec + Orval codegen config
│   ├── api-client-react/   # Generated React Query hooks
│   ├── api-zod/            # Generated Zod schemas from OpenAPI
│   ├── db/                 # Drizzle ORM schema + DB connection
│   └── integrations-openai-ai-server/  # OpenAI AI integration
```

## Database Schema

- `patients` - Patient profiles (name, age, gender)
- `analyses` - AI analysis results (image info, quality, disease results, report)

## API Endpoints

- `GET /api/patients` - List all patients
- `POST /api/patients` - Create patient (name, age, gender)
- `GET /api/patients/:id` - Get patient
- `GET /api/patients/:id/analyses` - List patient analyses
- `POST /api/analyses` - Create analysis (multipart: patientId, imageBase64, imageName, eyeSide)
- `GET /api/analyses/:id` - Get analysis

## TypeScript & Composite Projects

Every package extends `tsconfig.base.json`. Root `tsconfig.json` lists all libs as project references.

- `pnpm run typecheck` — full check
- `pnpm run build` — builds all

## Key Commands

- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks/schemas
- `pnpm --filter @workspace/db run push` — push DB schema changes
- `pnpm --filter @workspace/api-server run dev` — run API server
- `pnpm --filter @workspace/eye-diagnosis run dev` — run frontend
