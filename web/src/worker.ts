// Compute worker: owns the WASM session (prepared table + shared kNN list)
// and runs one embedding at a time, posting progress snapshots.

import init, { Session } from "./wasm/scalerank_wasm.js";
import type { FromWorker, Quality, ToWorker } from "./protocol";

let session: Session | null = null;
let labels: Uint32Array | null = null;
let knnReady = false;

const post = (msg: FromWorker, transfer: Transferable[] = []) =>
  (self as unknown as Worker).postMessage(msg, transfer);

// Snapshots at most every ~120 ms, plus the final step.
function reporter(id: number) {
  let last = 0;
  return (step: number, total: number, y: Float32Array) => {
    const now = performance.now();
    if (now - last > 120 || step === total) {
      last = now;
      post({ type: "progress", id, step, total, embedding: y }, [y.buffer]);
    }
    return true;
  };
}

async function handle(msg: ToWorker) {
  if (msg.type === "load") {
    session?.free();
    const t0 = performance.now();
    session = new Session(msg.raw, msg.n, msg.d, msg.standardize, 50);
    labels = msg.labels;
    knnReady = false;
    post({
      type: "loaded",
      n: session.n,
      dim: session.dim,
      dropped: session.dropped,
      explained: session.explained,
      ms: performance.now() - t0,
    });
    return;
  }
  if (!session) throw new Error("no data loaded");
  const s = session;
  let knnMs = 0;
  if (!knnReady) {
    post({ type: "stage", id: msg.id, text: "Finding nearest neighbours…" });
    const t = performance.now();
    s.compute_knn();
    knnMs = performance.now() - t;
    knnReady = true;
  }
  post({ type: "stage", id: msg.id, text: "Optimising…" });
  const t0 = performance.now();
  const every = 1;
  const cb = reporter(msg.id);
  let y: Float32Array;
  if (msg.method === "scalerank") {
    const p = msg.params;
    y = s.scalerank(p.gamma, p.nEpochs, p.negScale, msg.seed, every, cb);
  } else if (msg.method === "umap") {
    y = s.umap(msg.params.nNeighbors, msg.params.minDist, msg.seed, every, cb);
  } else {
    y = s.tsne(msg.params.perplexity, msg.seed, every, cb);
  }
  const ms = performance.now() - t0;
  post({ type: "stage", id: msg.id, text: "Measuring quality…" });
  const nAnchors = Math.min(s.n, 500);
  const quality = s.quality(y, labels ?? undefined, nAnchors) as Quality;
  post({ type: "done", id: msg.id, embedding: y, quality, ms, knnMs }, [y.buffer]);
}

let queue: Promise<unknown> = init().then(() => post({ type: "ready" }));
self.onmessage = (e: MessageEvent<ToWorker>) => {
  queue = queue.then(() =>
    handle(e.data).catch((err: unknown) => post({ type: "error", message: String((err as Error)?.message ?? err) })),
  );
};
