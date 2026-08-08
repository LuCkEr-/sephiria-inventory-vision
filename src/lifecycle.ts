/**
 * Coordinates synchronous disposal with asynchronous operations. Calling
 * dispose prevents new work immediately, while resources remain alive until
 * every operation that already entered the scope has left it.
 */
export class ResourceLifecycle {
  readonly #resourceName: string;
  readonly #releaseResources: () => void;
  #activeOperations = 0;
  #disposeRequested = false;
  #resourcesReleased = false;

  constructor(resourceName: string, releaseResources: () => void) {
    this.#resourceName = resourceName;
    this.#releaseResources = releaseResources;
  }

  enter(): () => void {
    if (this.#disposeRequested) throw new Error(`${this.#resourceName} has been disposed`);
    this.#activeOperations += 1;
    let left = false;
    return () => {
      if (left) return;
      left = true;
      this.#activeOperations -= 1;
      this.#releaseIfIdle();
    };
  }

  dispose(): void {
    if (this.#disposeRequested) return;
    this.#disposeRequested = true;
    this.#releaseIfIdle();
  }

  #releaseIfIdle(): void {
    if (!this.#disposeRequested || this.#activeOperations !== 0 || this.#resourcesReleased) return;
    this.#resourcesReleased = true;
    this.#releaseResources();
  }
}
