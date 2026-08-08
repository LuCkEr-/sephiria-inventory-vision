/**
 * Deduplicates concurrent asynchronous initialization while allowing a later
 * call to retry after failure. Successful values remain cached for process life.
 */
export class RetryableLoader<T> {
  readonly #load: () => Promise<T>;
  readonly #failureMessage: string | undefined;
  #pending: Promise<T> | undefined;

  constructor(load: () => Promise<T>, failureMessage?: string) {
    this.#load = load;
    this.#failureMessage = failureMessage;
  }

  get(): Promise<T> {
    if (this.#pending) return this.#pending;

    const attempt = Promise.resolve().then(this.#load);
    const guarded = attempt.catch((error: unknown) => {
      if (this.#pending === guarded) this.#pending = undefined;
      if (this.#failureMessage) {
        throw new Error(this.#failureMessage, { cause: error });
      }
      throw error;
    });
    this.#pending = guarded;
    return guarded;
  }
}
