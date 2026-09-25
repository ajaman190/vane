/** Shared request/response types for the Vane HTTP API. */

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export type JsonObject = { readonly [key: string]: JsonValue };

/** Image payload: data URL / base64 string, or raw bytes. */
export type ImageInput = string | Uint8Array;

/**
 * State: string, JSON value, or object that may carry ``images``.
 * Prefer ``StateWithImages`` or pass ``images`` via ``SystemOneRequest.images``.
 */
export type StateObject = {
  readonly [key: string]: JsonValue | readonly ImageInput[] | undefined;
  readonly images?: readonly ImageInput[];
};

export type State = string | JsonValue | StateObject;

export type NoulQuestion = {
  readonly type: "noul";
  readonly instructions?: JsonValue;
  readonly criteria?: {
    readonly true?: string | null;
    readonly false?: string | null;
  };
};

export type ChoiceQuestion = {
  readonly type: "choice";
  readonly instructions?: JsonValue;
  readonly criteria: { readonly [option: string]: string | null };
};

export type ScoreQuestion = {
  readonly type: "score";
  readonly instructions?: JsonValue;
  /** Ordered rubric levels, lowest first (2..10). */
  readonly criteria: readonly (string | null)[];
};

export type Question = NoulQuestion | ChoiceQuestion | ScoreQuestion;
export type Questions = { readonly [id: string]: Question };

export type NoulAnswer = {
  readonly type: "noul";
  readonly noul: number;
};

export type ChoiceAnswer = {
  readonly type: "choice";
  readonly choice: string;
  readonly probabilities: { readonly [option: string]: number };
  readonly confidence: number;
};

export type ScoreAnswer = {
  readonly type: "score";
  readonly score: number;
  /** Ordered rubric levels, lowest first (matches architecture.md). */
  readonly legend: readonly (string | null)[];
  readonly probabilities: { readonly [level: string]: number };
  readonly confidence: number;
};

export type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer;

export type RoutingReason = "explicit" | "images" | "text";

export type Routing = {
  readonly checkpoint: string;
  readonly reason: RoutingReason;
  readonly design_id?: string;
  readonly alias?: string;
  readonly alias_unresolved?: boolean;
  readonly requested?: string;
  readonly note?: string;
};

export type Usage = {
  readonly input_tokens: number;
  readonly output_tokens: number;
};

export type SystemOneRequest = {
  readonly state: State;
  readonly questions: Questions;
  readonly model?: string;
  /**
   * First-class images: merged into ``state.images`` on the wire when set.
   * If ``state`` is already an object with ``images``, those win unless this
   * field is provided (this field overrides).
   */
  readonly images?: readonly ImageInput[];
};

export type SystemOneResult = {
  readonly model: string;
  readonly answers: { readonly [id: string]: Answer };
  readonly routing?: Routing;
  readonly usage: Usage;
};

export type ModelCard = {
  readonly name: string;
  readonly description: string;
  readonly release_date: string;
  readonly aliases?: readonly string[];
  readonly status?: string;
};

/** GET /v1/models response body. */
export type ModelsResponse = {
  readonly models: readonly ModelCard[];
};

export type ClientOptions = {
  readonly baseUrl: string;
  /** Optional Bearer token. Omitted → no Authorization header. */
  readonly apiKey?: string;
  readonly fetch?: typeof fetch;
  readonly defaultHeaders?: Record<string, string>;
};

export class VaneHttpError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(message: string, status: number, body: unknown) {
    super(message);
    this.name = "VaneHttpError";
    this.status = status;
    this.body = body;
  }
}
