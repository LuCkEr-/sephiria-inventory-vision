/**
 * Deduplicates concurrent asynchronous initialization while allowing a later
 * call to retry after failure. Successful values remain cached for process life.
 */
export declare class RetryableLoader<T> {
    #private;
    constructor(load: () => Promise<T>, failureMessage?: string);
    get(): Promise<T>;
}
//# sourceMappingURL=retryable-loader.d.ts.map