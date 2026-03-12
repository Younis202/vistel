import { Router, type IRouter } from "express";
import healthRouter from "./health";
import patientsRouter from "./patients";
import analysesRouter from "./analyses";

const router: IRouter = Router();

router.use(healthRouter);
router.use(patientsRouter);
router.use(analysesRouter);

export default router;
