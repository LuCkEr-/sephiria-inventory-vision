/**
 * Coordinates synchronous disposal with asynchronous operations. Calling
 * dispose prevents new work immediately, while resources remain alive until
 * every operation that already entered the scope has left it.
 */
export declare class ResourceLifecycle {
    #private;
    constructor(resourceName: string, releaseResources: () => void);
    enter(): () => void;
    dispose(): void;
}
//# sourceMappingURL=lifecycle.d.ts.map