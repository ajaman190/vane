import type {
  ClientOptions,
  ImageInput,
  ModelCard,
  ModelsResponse,
  State,
  SystemOneRequest,
  SystemOneResult,
} from "./types.js";
import { VaneHttpError } from "./types.js";

/**
 * Typed client for a Vane server (POST /v1/systemone, GET /v1/models).
 * TypeSafe-compatible HTTP shape; ids are vane-*, not third-party brands.
 * No API key required — Authorization is sent only when ``apiKey`` is set.
 */
export class Client {
  private readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly fetchImpl: typeof fetch;
  private readonly defaultHeaders: Record<string, string>;

  constructor(options: ClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.fetchImpl = options.fetch ?? fetch;
    this.defaultHeaders = { ...(options.defaultHeaders ?? {}) };
  }

  async systemone(request: SystemOneRequest): Promise<SystemOneResult> {
    const body = {
      ...request,
      state: mergeStateImages(request.state, request.images),
    };
    // Do not send the top-level images convenience field — wire uses state.images.
    const { images: _images, ...wire } = body;
    void _images;
    return this.requestJson<SystemOneResult>("POST", "/v1/systemone", {
      state: wire.state,
      questions: wire.questions,
      ...(wire.model !== undefined ? { model: wire.model } : {}),
    });
  }

  async models(): Promise<ModelCard[]> {
    const raw = await this.requestJson<ModelsResponse | ModelCard[]>(
      "GET",
      "/v1/models",
    );
    if (Array.isArray(raw)) {
      return raw;
    }
    if (
      raw !== null &&
      typeof raw === "object" &&
      Array.isArray((raw as ModelsResponse).models)
    ) {
      return [...(raw as ModelsResponse).models];
    }
    throw new TypeError("GET /v1/models expected { models: ModelCard[] }");
  }

  private async requestJson<T>(
    method: "GET" | "POST",
    path: string,
    body?: unknown,
  ): Promise<T> {
    const headers: Record<string, string> = {
      Accept: "application/json",
      ...this.defaultHeaders,
    };
    if (this.apiKey !== undefined && this.apiKey.length > 0) {
      headers.Authorization = `Bearer ${this.apiKey}`;
    }
    const init: RequestInit = { method, headers };
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body, jsonReplacer);
    }
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, init);
    const text = await response.text();
    let parsed: unknown = null;
    if (text.length > 0) {
      try {
        parsed = JSON.parse(text) as unknown;
      } catch {
        parsed = text;
      }
    }
    if (!response.ok) {
      throw new VaneHttpError(
        `HTTP ${response.status} ${method} ${path}`,
        response.status,
        parsed,
      );
    }
    return parsed as T;
  }
}

/** Merge optional top-level images into state.images for the wire payload. */
export function mergeStateImages(
  state: State,
  images?: readonly ImageInput[],
): State {
  if (images === undefined) {
    return state;
  }
  if (state !== null && typeof state === "object" && !Array.isArray(state)) {
    return { ...(state as Record<string, unknown>), images };
  }
  return { text: state as string | number | boolean | null, images };
}

function jsonReplacer(_key: string, value: unknown): unknown {
  if (value instanceof Uint8Array) {
    // Base64 for JSON transport; server field-checks images presence.
    let binary = "";
    const chunk = 0x8000;
    for (let i = 0; i < value.length; i += chunk) {
      binary += String.fromCharCode(...value.subarray(i, i + chunk));
    }
    return btoa(binary);
  }
  return value;
}
