import express, { type Express } from "express";
import cors from "cors";
import { createProxyMiddleware } from "http-proxy-middleware";
import router from "./routes";

const app: Express = express();

app.use(cors());

// Proxy all /api/retina/* requests to the Python backend (port 8000)
app.use(
  "/api/retina",
  createProxyMiddleware({
    target: "http://localhost:8000",
    changeOrigin: true,
    pathRewrite: { "^/api/retina": "" },
    on: {
      error: (err, _req, res: any) => {
        console.error("[Proxy] Python backend error:", err.message);
        res.status(502).json({
          error: "Python backend unavailable",
          detail: "The RetinaGPT AI backend is not responding. Please try again.",
        });
      },
    },
  }),
);

app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

app.use("/api", router);

export default app;
