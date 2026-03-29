# MINA — Fish Gate Classifier: Handover Prompt

## Copy everything below this line and paste it into a new conversation

---

## Project: MINA — Offline Fish Disease Detection PWA

**Repo:** `MGuruNikhil/fishcareyolo` (branch: `fish-or-notfish`)

MINA is a **React + Vite progressive web app** that detects fish diseases on-device using a YOLOv8n model running via `onnxruntime-web` (WASM backend). The app is fully offline-capable — the model is downloaded once and cached by the service worker (Workbox). Scan history is stored in IndexedDB. There is **no backend server** — all inference happens in the browser.

---

## The Problem

The YOLOv8 disease model was trained to classify 5 classes:
`bacterial_infection`, `fungal_infection`, `healthy`, `parasite`, `white_tail`

When a photo that **does not contain a fish** is submitted (e.g. a keyboard, a hand, random background), the YOLO model still predicts bounding boxes — usually classifying the object as "healthy". This creates a confusing and incorrect user experience.

---

## Chosen Solution: Two-Stage Pipeline

We are adding a **lightweight binary gate classifier** that runs *before* the YOLO model:

```
User submits image
        │
        ▼
┌───────────────────────┐
│  Gate Model           │  ← NEW  (fish_gate.onnx, ~2.5 MB, MobileNetV3-Small)
│  Input: 224×224       │
│  Output: sigmoid [0,1]│  sigmoid > 0.6 → is fish
└───────────┬───────────┘
            │
     Is fish? ── No ──→  Show "No fish detected" UI state.
            │               Do NOT run YOLO. Do NOT save to history.
           Yes
            │
            ▼
┌───────────────────────┐
│  Disease Model        │  ← EXISTING  (best.onnx, ~12 MB, YOLOv8n)
│  Input: 640×640       │
│  Output: detections   │
└───────────┬───────────┘
            │
            ▼
       Show results
```

**Why MobileNetV3-Small via ONNX (not TensorFlow.js):**
The app already uses `onnxruntime-web` for the YOLO model. Adding TF.js would be a second runtime (~2 MB extra bundle). Instead, we train MobileNetV3-Small in PyTorch, export to ONNX, and run it through the existing ORT-Web infrastructure. The gate model is also automatically cached by the existing Workbox service worker config (which already handles `*.onnx` files under `/model/`).

---

## Phase 1: DONE ✅ — Model Training Pipeline (Python)

All code lives in `model/` on the `fish-or-notfish` branch and follows the exact conventions of the existing YOLOv8 pipeline.

### New files created

| File | Purpose |
|---|---|
| `model/mina/core/constants.py` | Appended gate constants: `GATE_IMAGE_SIZE=224`, `GATE_THRESHOLD=0.6`, `GATE_RUNS_DIR`, `GATE_DATA_DIR`, `IMAGENET_MEAN/STD` |
| `model/mina/gate_dataset.py` | Downloads two Kaggle datasets; samples 1000 images per class (fish/no_fish); builds `gate_data/` as an ImageFolder structure |
| `model/mina/gate_train.py` | Two-phase MobileNetV3-Small transfer learning: Phase 1 (5 epochs, head only, lr=1e-3) → Phase 2 (15 epochs, fine-tune last 2 backbone blocks, lr=5e-5). Label flipping so `fish→1.0` for sigmoid. |
| `model/mina/gate_export.py` | Exports `fish_gate_best.pt` → `fish_gate.onnx` (opset 17, input name `images`, output name `output`, shape `[1,1]`) + onnxruntime CPU verification |
| `model/mina/gate_evaluate.py` | Evaluates ONNX model on val split: accuracy, precision, recall, F1 + acceptance criteria check |
| `model/cli/gate_download.py` | `uv run gate-download` CLI |
| `model/cli/gate_train.py` | `uv run gate-train` CLI |
| `model/cli/gate_export.py` | `uv run gate-export` CLI |
| `model/cli/gate_evaluate.py` | `uv run gate-evaluate` CLI |
| `model/gate_train_colab.ipynb` | Full Google Colab notebook (9 cells + resume section) to run Phase 1 on a T4 GPU |
| `model/pyproject.toml` | Added `kaggle>=1.6.0`, `onnxruntime>=1.17.0`, 4 new CLI scripts |

### Output of Phase 1
Running the Colab notebook produces `runs/gate/fish_gate.onnx` which must be copied to:
```
web/public/model/fish_gate.onnx
```

### Acceptance criteria (Phase 1 complete when)
- Val accuracy ≥ 90%
- Precision ≥ 88%, Recall ≥ 88%
- ONNX size ≤ 4 MB
- `verify_onnx` confirms output shape `(1, 1)`

---

## Current Web App Architecture (context for Phase 2)

**Tech stack:** React 19, Vite 7, TypeScript, Tailwind v4, `onnxruntime-web@1.20`, `vite-plugin-pwa` (Workbox)

**Inference engine:**
- `web/app/lib/inference/worker.ts` — Web Worker that loads and runs the ONNX model via ORT-Web. Handles NMS, postprocessing, letterbox.
- `web/app/lib/inference/service.ts` — Main thread service class (`InferenceService`) that owns the worker, manages model load state, and exposes `serve()` / `run(imageElement)` / `unserved()`.
- `web/app/lib/inference/transform.ts` — Converts raw worker results to typed `InferenceResult` / `Detection` objects.
- `web/app/lib/model/types.ts` — `DiseaseClass`, `Detection`, `BoundingBox`, `InferenceResult` types.
- `web/app/lib/model/disease/` — `info.ts` (disease descriptions, symptoms, treatments), `severity.ts`.

**Model file locations:**
- Disease model: `web/public/model/best.onnx` (~12 MB, YOLOv8n)
- Gate model (to be placed): `web/public/model/fish_gate.onnx` (~2.5 MB, MobileNetV3-Small)

**Service worker:** `vite.config.ts` already has Workbox `runtimeCaching` for `/model/*.onnx` → `CacheFirst`, 1-year TTL. `globPatterns` already includes `*.onnx`. No Vite config changes needed for the gate model.

**Key ImageNet normalization values** (must match in JS gate worker):
```
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
```
**Gate model output:** raw logit (NOT sigmoid-activated). Apply `sigmoid(logit) > 0.6` in JS to decide "is fish".

---

## Remaining Phases

### Phase 2: Web App Integration — NOT STARTED
Add the gate classifier to the existing inference pipeline in the browser.

High-level work:
1. **`gate-worker.ts`** (new) — Separate Web Worker that loads `fish_gate.onnx`, preprocesses image to 224×224 with ImageNet normalization, runs ORT-Web, returns `{ isFish: boolean, confidence: number }`
2. **`gate-service.ts`** (new) — Service class mirroring `InferenceService` for the gate model. Exposes `serve(modelUrl)` and `run(imageElement)`.
3. **Load both models in parallel** — update app initialisation to `Promise.all([inferenceService.serve(), gateService.serve()])` so there is no sequential loading delay.
4. **Gate-then-disease call chain** — wherever `inferenceService.run(image)` is called today, first call `gateService.run(image)`. If `isFish === false`, return early with a `noFish` result. Only proceed to disease model if gate passes.

### Phase 3: UI — NOT STARTED
- Add a distinct `"no_fish"` UI state on the results/analysis page (not an error card — a friendly message like *"No fish found — please photograph your fish directly"*)
- This state must NOT be saved to history
- Settings page: show gate model + disease model download status separately

### Phase 4: Testing — NOT STARTED
Manual test cases:
1. Clear fish photo → gate passes → disease model runs → results shown
2. Keyboard/random photo → gate blocks → "No fish" UI shown → NOT saved to history
3. Empty aquarium water → gate blocks
4. Blurry fish photo → gate passes (threshold tunable)
5. Diseased fish → gate passes → disease correctly detected
6. Offline (after first load) → same behavior for all above
7. Low-end mobile (throttled CPU) → gate + disease completes in < 5s

---

## Please plan Phase 2

Please understand the existing inference architecture described above (especially `worker.ts`, `service.ts`, and `transform.ts`) and create a detailed implementation plan for Phase 2. The plan should cover:
- Exact new files to create and existing files to modify
- The gate worker TypeScript code (preprocessing with ImageNet normalization, sigmoid postprocessing)  
- The gate service TypeScript code
- How to wire the gate into the existing call sites
- How to handle parallel model loading at startup
- Any type changes needed in `types.ts`

Then ask for approval before writing any code.
