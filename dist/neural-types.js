/** @internal */
export function validateTrainOptions(options) {
    for (const [name, value] of [
        ["epochs", options.epochs],
        ["batchSize", options.batchSize],
    ]) {
        if (value !== undefined && (!Number.isSafeInteger(value) || value <= 0)) {
            throw new Error(`${name} must be a positive integer, received ${value}`);
        }
    }
    if (options.learningRate !== undefined &&
        (!Number.isFinite(options.learningRate) || options.learningRate <= 0)) {
        throw new Error(`learningRate must be positive and finite, received ${options.learningRate}`);
    }
    if (options.seed !== undefined && !Number.isSafeInteger(options.seed)) {
        throw new Error(`seed must be a safe integer, received ${options.seed}`);
    }
}
//# sourceMappingURL=neural-types.js.map