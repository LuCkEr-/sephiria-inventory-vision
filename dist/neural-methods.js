import { validateTrainOptions } from "./neural-types.js";
import { RetryableLoader } from "./retryable-loader.js";
export { nonMaximumSuppression } from "./non-maximum-suppression.js";
const tensorflowLoader = new RetryableLoader(() => import("@tensorflow/tfjs"), "Neural lab methods require the optional @tensorflow/tfjs peer dependency");
function getTensorFlow() {
    return tensorflowLoader.get();
}
function validatePixels(pixels, context) {
    const expectedLength = 32 * 32 * 3;
    if (pixels.length !== expectedLength) {
        throw new Error(`${context} must contain 32x32 RGB pixels exactly, received ${pixels.length} values`);
    }
    for (let index = 0; index < expectedLength; index += 1) {
        const value = pixels[index];
        if (!Number.isInteger(value) || (value ?? -1) < 0 || (value ?? 256) > 255) {
            throw new Error(`${context} must contain byte-valued RGB pixels; invalid value at ${index}`);
        }
    }
}
function validateLabel(label, context) {
    if (typeof label !== "string" || label.length === 0) {
        throw new Error(`${context} label must be a non-empty string`);
    }
}
function asPixels(samples) {
    const values = new Float32Array(samples.length * 32 * 32 * 3);
    samples.forEach((sample, sampleIndex) => {
        validatePixels(sample.pixels, "CNN samples");
        const offset = sampleIndex * 32 * 32 * 3;
        for (let index = 0; index < 32 * 32 * 3; index += 1) {
            values[offset + index] = (sample.pixels[index] ?? 0) / 255;
        }
    });
    return values;
}
function outputTensor(value) {
    if (Array.isArray(value)) {
        const first = value[0];
        if (!first)
            throw new Error("Model returned no tensors");
        return first;
    }
    return value;
}
async function prepareTensorFlow() {
    const tf = await getTensorFlow();
    await tf.setBackend("cpu");
    await tf.ready();
    return tf;
}
function createTinyCnnModels(tf, labelCount, seed, learningRate) {
    const input = tf.input({ shape: [32, 32, 3], name: "cell" });
    let value = tf.layers
        .conv2d({
        filters: 8,
        kernelSize: 3,
        padding: "same",
        activation: "relu",
        kernelInitializer: tf.initializers.glorotUniform({ seed }),
    })
        .apply(input);
    value = tf.layers.maxPooling2d({ poolSize: 2 }).apply(value);
    value = tf.layers
        .conv2d({
        filters: 16,
        kernelSize: 3,
        padding: "same",
        activation: "relu",
        kernelInitializer: tf.initializers.glorotUniform({ seed: seed + 1 }),
    })
        .apply(value);
    value = tf.layers.maxPooling2d({ poolSize: 2 }).apply(value);
    const embedding = tf.layers
        .globalAveragePooling2d({ name: "embedding" })
        .apply(value);
    const output = tf.layers
        .dense({
        units: labelCount,
        activation: "softmax",
        kernelInitializer: tf.initializers.glorotUniform({ seed: seed + 2 }),
        name: "identity",
    })
        .apply(embedding);
    const model = tf.model({ inputs: input, outputs: output });
    model.compile({
        optimizer: tf.train.adam(learningRate),
        loss: "sparseCategoricalCrossentropy",
        metrics: ["accuracy"],
    });
    return { model, embedder: tf.model({ inputs: input, outputs: embedding }) };
}
async function fitTinyCnn(tf, model, samples, labelIndex, options) {
    const xs = tf.tensor4d(asPixels(samples), [samples.length, 32, 32, 3]);
    const ys = tf.tensor1d(samples.map((sample) => labelIndex.get(sample.label) ?? 0), "float32");
    try {
        await model.fit(xs, ys, {
            epochs: options.epochs ?? 18,
            batchSize: Math.min(options.batchSize ?? 32, samples.length),
            shuffle: true,
            verbose: 0,
        });
    }
    finally {
        xs.dispose();
        ys.dispose();
    }
}
function cnnInputTensor(tf, pixels) {
    validatePixels(pixels, "CNN input");
    const values = new Float32Array(32 * 32 * 3);
    for (let index = 0; index < values.length; index += 1) {
        values[index] = (pixels[index] ?? 0) / 255;
    }
    return tf.tensor4d(values, [1, 32, 32, 3]);
}
function bestProbabilityIndex(probabilities, count, offset = 0) {
    let bestIndex = 0;
    for (let index = 1; index < count; index += 1) {
        if ((probabilities[offset + index] ?? 0) > (probabilities[offset + bestIndex] ?? 0)) {
            bestIndex = index;
        }
    }
    return bestIndex;
}
function createTinyCnnRuntime(tf, model, embedder, labels) {
    let disposed = false;
    const assertActive = () => {
        if (disposed)
            throw new Error("Tiny CNN has been disposed");
    };
    const embed = (pixels) => tf.tidy(() => {
        const tensor = cnnInputTensor(tf, pixels);
        return [...outputTensor(embedder.predict(tensor)).dataSync()];
    });
    return {
        labels,
        embed(pixels) {
            assertActive();
            return embed(pixels);
        },
        predict(pixels) {
            assertActive();
            return tf.tidy(() => {
                const tensor = cnnInputTensor(tf, pixels);
                const probabilities = [...outputTensor(model.predict(tensor)).dataSync()];
                const bestIndex = bestProbabilityIndex(probabilities, probabilities.length);
                return {
                    label: labels[bestIndex] ?? "__empty__",
                    confidence: probabilities[bestIndex] ?? 0,
                    embedding: [...outputTensor(embedder.predict(tensor)).dataSync()],
                };
            });
        },
        predictBatch(cells) {
            assertActive();
            if (cells.length === 0)
                return [];
            return predictCnnBatch(tf, model, embedder, labels, cells);
        },
        dispose() {
            if (disposed)
                return;
            // Both models share the convolutional weights. The classifier owns and
            // releases them, so disposing the embedder too would double-dispose.
            model.dispose();
            disposed = true;
        },
    };
}
function predictCnnBatch(tf, model, embedder, labels, cells) {
    return tf.tidy(() => {
        const packed = asPixels(cells.map((pixels) => ({ label: "unused", pixels })));
        const tensor = tf.tensor4d(packed, [cells.length, 32, 32, 3]);
        const probabilities = outputTensor(model.predict(tensor)).dataSync();
        const embeddings = outputTensor(embedder.predict(tensor)).dataSync();
        const embeddingSize = embeddings.length / cells.length;
        return cells.map((_, cellIndex) => {
            const offset = cellIndex * labels.length;
            const bestIndex = bestProbabilityIndex(probabilities, labels.length, offset);
            return {
                label: labels[bestIndex] ?? "__empty__",
                confidence: probabilities[offset + bestIndex] ?? 0,
                embedding: Array.from(embeddings.slice(cellIndex * embeddingSize, (cellIndex + 1) * embeddingSize)),
            };
        });
    });
}
/** Trains the small real convolutional classifier used by the method lab. */
export async function trainTinyCnn(samples, options = {}) {
    if (samples.length === 0)
        throw new Error("Cannot train a CNN without samples");
    validateTrainOptions(options);
    samples.forEach((sample) => {
        validateLabel(sample.label, "CNN sample");
        validatePixels(sample.pixels, "CNN samples");
    });
    const tf = await prepareTensorFlow();
    const labels = [...new Set(samples.map((sample) => sample.label))].sort();
    const labelIndex = new Map(labels.map((label, index) => [label, index]));
    const seed = options.seed ?? 731;
    const { model, embedder } = createTinyCnnModels(tf, labels.length, seed, options.learningRate ?? 0.004);
    try {
        await fitTinyCnn(tf, model, samples, labelIndex, options);
        return createTinyCnnRuntime(tf, model, embedder, labels);
    }
    catch (error) {
        model.dispose();
        throw error;
    }
}
function normalizedTensor(tf, tensor) {
    return tf.div(tensor, tf.maximum(tf.norm(tensor, 2, 1, true), tf.scalar(1e-8)));
}
function validateSiameseSamples(samples) {
    samples.forEach((sample) => {
        validateLabel(sample.label, "Siamese sample");
        if (sample.features.some((feature) => !Number.isFinite(feature))) {
            throw new Error("Siamese feature vectors must contain only finite values");
        }
    });
    const featureSize = samples[0]?.features.length ?? 0;
    if (featureSize === 0 || samples.some((sample) => sample.features.length !== featureSize)) {
        throw new Error("Siamese feature vectors must have one non-zero shared size");
    }
    return featureSize;
}
function createSiameseTower(tf, featureSize, seed) {
    return tf.sequential({
        layers: [
            tf.layers.dense({
                inputShape: [featureSize],
                units: 64,
                activation: "relu",
                kernelInitializer: tf.initializers.glorotUniform({ seed }),
            }),
            tf.layers.dense({
                units: 24,
                activation: "linear",
                kernelInitializer: tf.initializers.glorotUniform({ seed: seed + 1 }),
            }),
        ],
    });
}
function groupSamplesByLabel(samples) {
    const byLabel = new Map();
    samples.forEach((sample, index) => {
        const bucket = byLabel.get(sample.label) ?? [];
        bucket.push(index);
        byLabel.set(sample.label, bucket);
    });
    return byLabel;
}
function seededRandom(seed) {
    let state = seed >>> 0;
    return () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 0x100000000;
    };
}
function differentLabel(labels, excluded, random) {
    let label = labels[Math.floor(random() * labels.length)];
    while (label === excluded && labels.length > 1) {
        label = labels[Math.floor(random() * labels.length)];
    }
    return label;
}
function randomBucketIndex(byLabel, label, random) {
    const bucket = byLabel.get(label);
    return bucket[Math.floor(random() * bucket.length)];
}
function createSiamesePairs(samples, byLabel, labels, pairCount, random) {
    const batch = { left: [], right: [], targets: [] };
    for (let pair = 0; pair < pairCount; pair += 1) {
        const same = pair % 2 === 0;
        const leftIndex = Math.floor(random() * samples.length);
        const leftSample = samples[leftIndex];
        const rightLabel = same ? leftSample.label : differentLabel(labels, leftSample.label, random);
        const rightIndex = randomBucketIndex(byLabel, rightLabel, random);
        const rightSample = samples[rightIndex];
        batch.left.push(...leftSample.features);
        batch.right.push(...rightSample.features);
        batch.targets.push(same ? 1 : 0);
    }
    return batch;
}
function contrastiveLoss(tf, tower, leftTensor, rightTensor, targetTensor) {
    const leftEmbedding = normalizedTensor(tf, outputTensor(tower.apply(leftTensor, { training: true })));
    const rightEmbedding = normalizedTensor(tf, outputTensor(tower.apply(rightTensor, { training: true })));
    const distances = tf.sqrt(tf.maximum(tf.sum(tf.square(tf.sub(leftEmbedding, rightEmbedding)), 1), tf.scalar(1e-8)));
    const positiveLoss = tf.mul(targetTensor, tf.square(distances));
    const negativeLoss = tf.mul(tf.sub(1, targetTensor), tf.square(tf.maximum(tf.sub(1, distances), 0)));
    return tf.mean(tf.add(positiveLoss, negativeLoss));
}
function trainSiameseBatch(tf, tower, optimizer, batch, pairCount, featureSize) {
    const leftTensor = tf.tensor2d(batch.left, [pairCount, featureSize]);
    const rightTensor = tf.tensor2d(batch.right, [pairCount, featureSize]);
    const targetTensor = tf.tensor1d(batch.targets);
    try {
        optimizer.minimize(() => tf.tidy(() => contrastiveLoss(tf, tower, leftTensor, rightTensor, targetTensor)), false);
    }
    finally {
        leftTensor.dispose();
        rightTensor.dispose();
        targetTensor.dispose();
    }
}
function createSiameseRuntime(tf, tower, optimizer, featureSize) {
    let disposed = false;
    return {
        embed(features) {
            if (disposed)
                throw new Error("Siamese embedding has been disposed");
            if (features.length !== featureSize) {
                throw new Error(`Expected ${featureSize} Siamese input values`);
            }
            return tf.tidy(() => {
                const tensor = tf.tensor2d([...features], [1, featureSize]);
                const embedded = normalizedTensor(tf, outputTensor(tower.predict(tensor)));
                return [...embedded.dataSync()];
            });
        },
        dispose() {
            if (disposed)
                return;
            optimizer.dispose();
            tower.dispose();
            disposed = true;
        },
    };
}
/** Trains a genuine shared-weight twin network with contrastive loss. */
export async function trainSiameseEmbedding(samples, options = {}) {
    if (samples.length < 2)
        throw new Error("Cannot train a Siamese model without sample pairs");
    validateTrainOptions(options);
    const featureSize = validateSiameseSamples(samples);
    const tf = await prepareTensorFlow();
    const seed = options.seed ?? 919;
    const tower = createSiameseTower(tf, featureSize, seed);
    const optimizer = tf.train.adam(options.learningRate ?? 0.003);
    const byLabel = groupSamplesByLabel(samples);
    const labels = [...byLabel.keys()];
    const pairCount = Math.min(1024, Math.max(128, samples.length * 4));
    const random = seededRandom(seed);
    try {
        for (let epoch = 0; epoch < (options.epochs ?? 32); epoch += 1) {
            const batch = createSiamesePairs(samples, byLabel, labels, pairCount, random);
            trainSiameseBatch(tf, tower, optimizer, batch, pairCount, featureSize);
        }
        return createSiameseRuntime(tf, tower, optimizer, featureSize);
    }
    catch (error) {
        optimizer.dispose();
        tower.dispose();
        throw error;
    }
}
//# sourceMappingURL=neural-methods.js.map