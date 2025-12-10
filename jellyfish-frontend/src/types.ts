// Message schemas for WS protocol

export type ResetMsg = {
  type: "reset",
  seed: number,
  J: number,
  wall_contrast: number
};

export type StepObs = {
  t: number,
  j: number,
  contrast_front: number,
  collided_prev: 0 | 1,
  heading_index: number
};

export type StepMsg = {
  type: "step",
  obs: StepObs
};

export type DoneMsg = { type: "done" };

export type Outgoing = ResetMsg | StepMsg | DoneMsg;

export type ActionPayload = {
  L: 0 | 1,
  R: 0 | 1,
  P: 1 | 2 | 3,
  debug?: Record<string, unknown>
};

export type Incoming =
  | { type: "ok" }
  | { type: "action", act: ActionPayload };


  export type NetworkLayerDesc = {
  weights: number[][]; // [rows][cols]
  bias: number[];
};

export type NetworkDescription = {
  mode: string;
  has_brain: boolean;
  has_policy?: boolean;
  input_labels?: string[];
  hidden_size?: number;
  output_labels?: string[];
  layers?: {
    input_to_hidden: NetworkLayerDesc;
    hidden_to_output: NetworkLayerDesc;
  };
};
