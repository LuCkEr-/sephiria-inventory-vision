/**
 * Deduplicates concurrent asynchronous initialization while allowing a later
 * call to retry after failure. Successful values remain cached for process life.
 */
export class RetryableLoader {
    #load;
    #failureMessage;
    #pending;
    constructor(load, failureMessage) {
        this.#load = load;
        this.#failureMessage = failureMessage;
    }
    get() {
        if (this.#pending)
            return this.#pending;
        const attempt = Promise.resolve().then(this.#load);
        const guarded = attempt.catch((error) => {
            if (this.#pending === guarded)
                this.#pending = undefined;
            if (this.#failureMessage) {
                throw new Error(this.#failureMessage, { cause: error });
            }
            throw error;
        });
        this.#pending = guarded;
        return guarded;
    }
}
//# sourceMappingURL=retryable-loader.js.map