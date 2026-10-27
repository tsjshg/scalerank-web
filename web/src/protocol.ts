// Messages between the page and the compute worker.

export type Method = "scalerank" | "umap" | "tsne";

export interface ScaleRankParams {
  gamma: number;
  nEpochs: number;
  negScale: number;
}

export interface UmapParams {
  nNeighbors: number;
  minDist: number;
}

export interface TsneParams {
  perplexity: number;
}

export type RunRequest =
  | { method: "scalerank"; params: ScaleRankParams }
  | { method: "umap"; params: UmapParams }
  | { method: "tsne"; params: TsneParams };

export type ToWorker =
  | { type: "load"; raw: Float32Array; n: number; d: number; standardize: boolean; labels: Uint32Array | null }
  | ({ type: "run"; id: number; seed: number } & RunRequest);

export interface Quality {
  rnx10: number;
  auc: number;
  trust10: number;
  cont10: number;
  distSpearman: number;
  densityCorr: number;
  knnAcc?: number;
  nAnchors: number;
  rnx: Float32Array;
  overlap10: Float32Array;
}

export type FromWorker =
  | { type: "ready" }
  | { type: "loaded"; n: number; dim: number; dropped: number; explained: number; ms: number }
  | { type: "stage"; id: number; text: string }
  | { type: "progress"; id: number; step: number; total: number; embedding: Float32Array }
  | { type: "done"; id: number; embedding: Float32Array; quality: Quality; ms: number; knnMs: number }
  | { type: "error"; message: string };
