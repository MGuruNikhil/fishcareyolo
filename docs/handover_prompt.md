# MINA — Fish Gate Classifier: Handover Prompt

## Copy everything below this line and paste it into a new conversation

---

## Project: MINA — Offline Fish Disease Detection PWA

**Repo:** `MGuruNikhil/fishcareyolo` (branch: `fish-or-notfish`)

MINA is a **React + Vite progressive web app** that detects fish diseases on-device using a YOLOv8n model running via `onnxruntime-web` (WASM backend). The app is fully offline-capable — the model is downloaded once and cached by the service worker (Workbox). Scan history is stored in IndexedDB. There is **no backend server** — all inference happens in the browser.

---

## The Problem (solved)

The YOLOv8 disease model was trained on 5 classes:
`bacterial_infection`, `fungal_infection`, `healthy`, `parasite`, `white_tail`

When a photo that **does not contain a fish** is submitted, the YOLO model still predicts bounding boxes — usually classifying the object as "healthy". A two-stage pipeline was built to fix this.

---

## Two-Stage Pipeline (implemented)

```
User submits image
        │
        ▼
┌───────────────────────┐
│  Gate Model           │  ← fish_gate.onnx (~2.5 MB, MobileNetV3-Small)
│  Input: 224×224       │  ImageNet normalised, CHW layout
│  Output: sigmoid [0,1]│  sigmoid > 0.6 → is fish
└───────────┬───────────┘
            │
     Is fish? ── No ──→  Navigate to /no-fish page.
            │               Do NOT run YOLO. Do NOT save to history.
           Yes
            │
            ▼
┌───────────────────────┐
│  Disease Model        │  best.onnx (~12 MB, YOLOv8n)
│  Input: 640×640       │
│  Output: detections   │
└───────────┬───────────┘
            │
            ▼
       Navigate to /results
```

Both models are released on **`fishcareyolo/fishcareyolo`** GitHub Releases and are cached by the Workbox service worker (`CacheFirst`, 1-year TTL for `/model/*.onnx`).

---

## Phase 1: DONE ✅ — Model Training Pipeline (Python)

All code lives in `model/` on the `fish-or-notfish` branch.

| File | Purpose |
|---|---|
| `model/mina/core/constants.py` | Gate constants: `GATE_IMAGE_SIZE=224`, `GATE_THRESHOLD=0.6`, `GATE_RUNS_DIR`, `GATE_DATA_DIR`, `IMAGENET_MEAN/STD` |
| `model/mina/gate_dataset.py` | Downloads Kaggle datasets; samples 1000 images/class; builds `gate_data/` ImageFolder |
| `model/mina/gate_train.py` | Two-phase MobileNetV3-Small: Phase 1 (5 epochs, head only) → Phase 2 (15 epochs, fine-tune last 2 blocks) |
| `model/mina/gate_export.py` | Exports `.pt` → `fish_gate.onnx` (opset 17, input `images`, output `output`, shape `[1,1]`) |
| `model/mina/gate_evaluate.py` | Accuracy, precision, recall, F1 + acceptance criteria check |
| `model/cli/gate_*.py` | CLI entry points for download / train / export / evaluate |
| `model/gate_train_colab.ipynb` | Full Colab notebook for GPU training |

**Output:** `fish_gate.onnx` is published on GitHub Releases alongside `best.onnx`.

---

## Phase 2: DONE ✅ — Web App Integration

**All files implemented, `bun run build` exits 0, TypeScript clean.**

### Tech stack
React 19, Vite 7, TypeScript, Tailwind v4, `onnxruntime-web@1.20`, `vite-plugin-pwa` (Workbox).
Routing: `file-system-router` + `import.meta.glob("./pages/**/*.tsx")` — new pages auto-register by filename.

### New files created

| File | What it does |
|---|---|
| `web/app/lib/inference/gate-worker.ts` | ORT-Web Web Worker for the gate model. Handles 224×224 scale-to-fill resize, ImageNet CHW normalisation (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`), runs `session.run({images})`, applies `sigmoid(logit) > 0.6`, returns `{ isFish, confidence }`. Same message protocol as `worker.ts` (`load/run/release`). |
| `web/app/lib/inference/gate-service.ts` | `GateService` class — exact mirror of `InferenceService`. Wraps `GateWorker`, exposes `serve() / run(imageElement) / unserved()`, same `idle→loading→ready→error` state machine and `onStatusChange` callback pattern. Default URL: `${BASE_URL}model/fish_gate.onnx` (override via `VITE_GATE_MODEL_URL`). Exports singleton `gateService`. |
| `web/app/pages/no-fish.tsx` | `/no-fish` route page shown when gate rejects image. Friendly UI: fish emoji, headline "No fish detected", tips, confidence readout. Single CTA: "Try Again" → `/`. Never saved to history. |

### Files modified

| File | Change |
|---|---|
| `web/app/lib/inference/index.ts` | Re-exports `gateService`, `GateResult`, `GateState`, `GateStatus`, `GateStatusCallback` |
| `web/app/lib/model/types.ts` | Added `AnalysisOutcome` discriminated union: `{ kind: "no_fish"; gateConfidence: number } \| { kind: "detections"; result: InferenceResult }` |
| `web/app/lib/detection/context.tsx` | Renamed `currentResult → currentOutcome`, typed as `AnalysisOutcome \| null` |
| `web/app/pages/analysis.tsx` | Two-stage pipeline: parallel `gateService.serve()` + `inferenceService.serve()` via `Promise.all`, new `"running-gate"` step, early exit to `/no-fish` if `!isFish`, saves history + navigates `/results` only on gate pass. Uses `setCurrentOutcome`. |
| `web/app/pages/results.tsx` | Reads `currentOutcome`, narrows to `kind === "detections"` before passing `result` to `ResultsView`. |
| `web/app/pages/settings.tsx` | Added **AI Models** section: two rows (Fish Gate ~2.5 MB, Disease Detector ~12 MB) with live status badges (`Cached ✓ / Loading… / Error / Not loaded`) subscribed to `onStatusChange` of both services. |

### Key types (current state)

```ts
// web/app/lib/model/types.ts

export type AnalysisOutcome =
  | { kind: "no_fish"; gateConfidence: number }
  | { kind: "detections"; result: InferenceResult }

// web/app/lib/inference/gate-service.ts

export interface GateResult {
  isFish: boolean
  confidence: number
}

export type GateStatus = "idle" | "loading" | "ready" | "error"
```

### Detection context (current state)

```ts
// web/app/lib/detection/context.tsx
interface DetectionState {
  currentOutcome: AnalysisOutcome | null
  setCurrentOutcome: (outcome: AnalysisOutcome | null) => void
}
```

---

## Phase 3: UI Polish — NOT STARTED

Phase 2 implemented the minimum functional UI. Phase 3 is about polish and completeness.

### What still needs doing

#### 3a. History page — gate scan entries
Currently the history page shows past disease scans. No-fish scans are correctly **not saved** (enforced in `analysis.tsx`). But you should verify the history list UI still renders correctly after the `currentOutcome` rename and that there are no stale references to `currentResult` anywhere in the history components.

- Check `web/app/pages/history/` and any sub-components for references to `currentResult` / `InferenceResult` that may need updating.

#### 3b. `/no-fish` page — deep-link guard
If a user deep-links directly to `/no-fish` (or hits back/refresh after a gate block), `currentOutcome` will be `null`. The page currently handles this gracefully (shows no confidence readout), but you may want to auto-redirect to `/` after a short delay instead of leaving the user on a context-less page.

#### 3c. Analysis page — gate model loading indicator
Currently the `loading-model` step label says "Loading AI models" and shows a "Downloading…" sub-label based only on the **disease** model status. Since the gate loads in parallel, consider:
- Show a combined progress indicator (e.g. both download bars, or just the slowest one)
- Or show separate sub-labels for gate vs disease download

#### 3d. Results page — gate confidence in results view
Currently `ResultsView` shows disease detections only. Optionally: show a small "Fish confirmed ✓" badge with the gate confidence (e.g. `fish confidence: 94%`) somewhere in the results header. This reassures users the gate validated their image.

#### 3e. Offline first-load behavior
On the very first cold start (no cached models), the user sees the analysis spinner while both models download. For a ~14 MB total download on a slow connection this could take a long time. Consider:
- A progress bar derived from `inferenceService`'s `progress` field (already tracked, just not displayed as a percentage)
- Or a "models are downloading for the first time" informational banner on the home screen

---

## Phase 4: Testing — NOT STARTED

Manual test cases (run on device or Chrome DevTools mobile emulation):

| # | Input | Expected outcome |
|---|---|---|
| 1 | Clear fish photo | Gate passes → disease model runs → `/results` |
| 2 | Keyboard / hand / random | Gate blocks → `/no-fish` shown → NOT in history |
| 3 | Empty aquarium water | Gate blocks → `/no-fish` |
| 4 | Blurry fish | Gate passes (threshold 0.6 is lenient) |
| 5 | Diseased fish | Gate passes → disease correctly detected |
| 6 | Offline after first load | All cases work identically (Workbox CacheFirst) |
| 7 | Low-end mobile (throttled CPU) | Gate + disease completes < 5s |
| 8 | Direct navigation to `/no-fish` | Handled gracefully (no crash, no stale context) |
| 9 | Settings page on cold start | Both model badges show "Not loaded" → "Loading…" → "Cached ✓" |
| 10 | Settings page after analysis | Both model badges show "Cached ✓" |

---

## Please plan Phase 3

Read the files listed in Phase 2 above (especially `web/app/pages/history/`, `web/app/pages/analysis.tsx`, and `web/app/pages/results.tsx`) and create a detailed plan for the Phase 3 UI polish items. Focus on:

- Any broken references to `currentResult` in history components
- The `/no-fish` deep-link guard (auto-redirect or not)
- Whether to add a gate confidence badge to the results view
- The loading progress UX for first-time model download

Then ask for approval before writing any code.
